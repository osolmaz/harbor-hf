from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
from collections import deque
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import BinaryIO, Literal, Protocol, cast

import zstandard
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harbor_hf.models import TrialEvidencePolicy

TRIAL_EVIDENCE_SCHEMA = "harbor-hf/trial-evidence/v1"
WORKSPACE_INDEX_SCHEMA = "harbor-hf/workspace-file/v1"
JUDGE_SELECTION_SCHEMA = "harbor-hf/judge-selection/v1"
WORKSPACE_SOURCE = PurePosixPath("/app")
WORKSPACE_DESTINATION = PurePosixPath("workspace/app")
_CHUNK = 1024 * 1024
MediaType = Literal[
    "application/json",
    "application/octet-stream",
    "application/vnd.harbor-hf.workspace+tar+zstd",
    "application/x-ndjson",
    "text/plain",
]


class _BinaryWriter(Protocol):
    def write(self, data: bytes, /) -> int: ...


class _BoundedWriter:
    def __init__(self, target: BinaryIO, maximum: int) -> None:
        self._target = target
        self._maximum = maximum
        self._written = 0

    def write(self, data: bytes, /) -> int:
        if self._written + len(data) > self._maximum:
            raise TrialEvidenceError("workspace archive exceeds configured byte limit")
        written = self._target.write(data)
        self._written += written
        return written

    def flush(self) -> None:
        self._target.flush()


class _DeadlineReader:
    def __init__(self, source: BinaryIO, deadline: float | None) -> None:
        self._source = source
        self._deadline = deadline

    def read(self, size: int = -1, /) -> bytes:
        _check_workspace_deadline(self._deadline)
        data = self._source.read(size)
        _check_workspace_deadline(self._deadline)
        return data


class TrialEvidenceError(RuntimeError):
    """Raised when exact trial evidence cannot be captured or validated."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FileReference(FrozenModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    media_type: MediaType

    @model_validator(mode="after")
    def path_is_safe(self) -> FileReference:
        _safe_relative_path(self.path)
        return self


class WorkspaceFile(FrozenModel):
    schema_version: Literal["harbor-hf/workspace-file/v1"] = WORKSPACE_INDEX_SCHEMA
    path: str = Field(min_length=1)
    type: Literal["directory", "file", "symlink"]
    mode: int = Field(ge=0, le=0o7777)
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    target: str | None = None

    @model_validator(mode="after")
    def fields_match_type(self) -> WorkspaceFile:
        if self.path == ".":
            if self.type != "directory":
                raise ValueError("workspace root index entry must be a directory")
        else:
            _safe_relative_path(self.path)
        if self.type == "file" and (self.sha256 is None or self.size_bytes is None):
            raise ValueError("workspace file requires size and sha256")
        if self.type == "symlink" and not self.target:
            raise ValueError("workspace link requires target")
        if self.type != "symlink" and self.target is not None:
            raise ValueError("workspace non-link cannot have target")
        if self.type != "file" and (
            self.sha256 is not None or self.size_bytes is not None
        ):
            raise ValueError("workspace non-file cannot have size or sha256")
        return self


class WorkspaceEvidence(FrozenModel):
    status: Literal["captured"] = "captured"
    root: Literal["/app"] = "/app"
    archive: FileReference
    file_index: FileReference
    entry_count: int = Field(ge=1)
    regular_file_count: int = Field(ge=0)
    regular_file_bytes: int = Field(ge=0)
    archive_format: Literal["pax"] = "pax"
    compression: Literal["zstd"] = "zstd"
    compression_level: Literal[10] = 10


class AgentEvidence(FrozenModel):
    status: Literal["captured"] = "captured"
    sessions: list[FileReference]
    trajectories: list[FileReference]
    logs: list[FileReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def lists_are_canonical(self) -> AgentEvidence:
        _require_sorted_unique(self.sessions)
        _require_sorted_unique(self.trajectories)
        _require_sorted_unique(self.logs)
        return self


class JudgeExchangeReference(FrozenModel):
    exchange_id: str = Field(pattern=r"^judge-[0-9]{4}$")
    attempt: int = Field(ge=1)
    record: FileReference

    @model_validator(mode="after")
    def identity_is_consistent(self) -> JudgeExchangeReference:
        if int(self.exchange_id.removeprefix("judge-")) != self.attempt:
            raise ValueError("judge exchange reference identity mismatch")
        if PurePosixPath(self.record.path).parent.name != self.exchange_id:
            raise ValueError("judge exchange record path identity mismatch")
        return self


class JudgeEvidence(FrozenModel):
    status: Literal["captured", "not_called", "not_expected"]
    expected: bool
    model: str | None = None
    recorder_summary: FileReference | None = None
    exchanges: list[JudgeExchangeReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def expectation_is_consistent(self) -> JudgeEvidence:
        exchange_ids = [item.exchange_id for item in self.exchanges]
        if exchange_ids != sorted(exchange_ids) or len(exchange_ids) != len(
            set(exchange_ids)
        ):
            raise ValueError("judge exchange references must be sorted and unique")
        if self.expected and (not self.model or self.recorder_summary is None):
            raise ValueError("expected judge evidence requires a recorder summary")
        if not self.expected and (
            self.model is not None
            or self.recorder_summary is not None
            or self.exchanges
        ):
            raise ValueError("unexpected judge evidence must be empty")
        expected_status = (
            "captured"
            if self.exchanges
            else ("not_called" if self.expected else "not_expected")
        )
        if self.status != expected_status:
            raise ValueError("judge status disagrees with recorded exchanges")
        return self


class VerifierEvidence(FrozenModel):
    status: Literal["captured"] = "captured"
    scorecard: FileReference
    reward: FileReference
    stdout: FileReference
    stderr: FileReference
    judge_selection: FileReference | None = None
    selected_judge_exchange_id: str | None = Field(
        default=None, pattern=r"^judge-[0-9]{4}$"
    )
    logs: list[FileReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def logs_are_canonical(self) -> VerifierEvidence:
        _require_sorted_unique(self.logs)
        if (self.judge_selection is None) != (self.selected_judge_exchange_id is None):
            raise ValueError("judge selection file and selected exchange must agree")
        return self


EvidenceRequirementName = Literal[
    "agent_session",
    "agent_trajectory",
    "judge_exchange",
    "judge_recorder",
    "judge_selection",
    "verifier_reward",
    "verifier_scorecard",
    "verifier_stderr",
    "verifier_stdout",
    "workspace",
]


class CompletionEvidence(FrozenModel):
    status: Literal["complete"] = "complete"
    requirements: list[EvidenceRequirementName]

    @model_validator(mode="after")
    def requirements_are_canonical(self) -> CompletionEvidence:
        if self.requirements != sorted(self.requirements) or len(
            self.requirements
        ) != len(set(self.requirements)):
            raise ValueError("evidence requirements must be sorted and unique")
        return self


class TrialEvidenceManifest(FrozenModel):
    schema_version: Literal["harbor-hf/trial-evidence/v1"] = TRIAL_EVIDENCE_SCHEMA
    campaign_id: str | None = None
    run_id: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    task_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    logical_attempt: int = Field(ge=1)
    physical_attempt: int = Field(ge=1)
    captured_at: datetime
    workspace: WorkspaceEvidence
    agent: AgentEvidence
    judge: JudgeEvidence
    verifier: VerifierEvidence
    completion: CompletionEvidence

    @model_validator(mode="after")
    def component_paths_and_requirements_match(self) -> TrialEvidenceManifest:
        _validate_manifest_capture_time(self)
        _validate_manifest_component_paths(self)
        _validate_manifest_requirements(self)
        return self


def _validate_manifest_capture_time(manifest: TrialEvidenceManifest) -> None:
    if manifest.captured_at.tzinfo is None:
        raise ValueError("trial evidence capture time must include a timezone")


def _validate_manifest_component_paths(manifest: TrialEvidenceManifest) -> None:
    workspace_paths = [
        manifest.workspace.archive.path,
        manifest.workspace.file_index.path,
    ]
    if any(not path.startswith("evidence/") for path in workspace_paths):
        raise ValueError("workspace evidence must remain under evidence/")
    agent_refs = [
        *manifest.agent.sessions,
        *manifest.agent.trajectories,
        *manifest.agent.logs,
    ]
    if any(not item.path.startswith("agent/") for item in agent_refs):
        raise ValueError("agent evidence must remain under agent/")
    judge_refs = [item.record for item in manifest.judge.exchanges]
    if manifest.judge.recorder_summary is not None:
        judge_refs.append(manifest.judge.recorder_summary)
    if any(not item.path.startswith("evidence/judge/") for item in judge_refs):
        raise ValueError("judge evidence must remain under evidence/judge/")
    verifier_refs = [
        manifest.verifier.scorecard,
        manifest.verifier.reward,
        manifest.verifier.stdout,
        manifest.verifier.stderr,
        *manifest.verifier.logs,
    ]
    if manifest.verifier.judge_selection is not None:
        verifier_refs.append(manifest.verifier.judge_selection)
    if any(not item.path.startswith("verifier/") for item in verifier_refs):
        raise ValueError("verifier evidence must remain under verifier/")


def _validate_manifest_requirements(manifest: TrialEvidenceManifest) -> None:
    expected: list[EvidenceRequirementName] = [
        "agent_session",
        "agent_trajectory",
        "verifier_reward",
        "verifier_scorecard",
        "verifier_stderr",
        "verifier_stdout",
        "workspace",
    ]
    if manifest.judge.expected:
        expected.append("judge_recorder")
    if manifest.judge.exchanges:
        expected.extend(["judge_exchange", "judge_selection"])
    if manifest.completion.requirements != sorted(expected):
        raise ValueError("trial evidence has an incomplete requirement set")


class JudgeSelection(FrozenModel):
    schema_version: Literal["harbor-hf/judge-selection/v1"] = JUDGE_SELECTION_SCHEMA
    exchange_id: str = Field(pattern=r"^judge-[0-9]{4}$")


class WorkspacePackage(FrozenModel):
    evidence: WorkspaceEvidence
    entries: list[WorkspaceFile]


def package_workspace(
    snapshot: Path,
    evidence_dir: Path,
    *,
    policy: TrialEvidencePolicy,
) -> WorkspacePackage:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise TrialEvidenceError("frozen /app workspace is not a directory")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    deadline = monotonic() + policy.workspace_capture_timeout_seconds
    entries = list(_workspace_entries(snapshot, policy, deadline=deadline))
    _validate_workspace_symlink_graph(snapshot, entries)
    index_path = evidence_dir / "workspace-files.jsonl"
    _atomic_write(
        index_path,
        b"".join(
            json.dumps(
                entry.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
            for entry in entries
        ),
    )
    archive_path = evidence_dir / "workspace.tar.zst"
    _check_workspace_deadline(deadline)
    _write_workspace_archive(
        snapshot,
        entries,
        archive_path,
        deadline=deadline,
        maximum_bytes=policy.workspace_max_archive_bytes,
    )
    files = [entry for entry in entries if entry.type == "file"]
    unpacked = sum(entry.size_bytes or 0 for entry in files)
    evidence = WorkspaceEvidence(
        archive=_reference(archive_path, evidence_dir.parent, deadline=deadline),
        file_index=_reference(index_path, evidence_dir.parent, deadline=deadline),
        entry_count=len(entries),
        regular_file_count=len(files),
        regular_file_bytes=unpacked,
    )
    package = WorkspacePackage(evidence=evidence, entries=entries)
    verify_workspace_package(
        evidence_dir.parent, package.evidence, deep=True, deadline=deadline
    )
    return package


def verify_workspace_package(
    trial_root: Path,
    evidence: WorkspaceEvidence,
    *,
    deep: bool,
    deadline: float | None = None,
) -> list[WorkspaceFile]:
    _verify_reference(trial_root, evidence.archive, deadline=deadline)
    _verify_reference(trial_root, evidence.file_index, deadline=deadline)
    entries = _load_workspace_index(
        trial_root / evidence.file_index.path, deadline=deadline
    )
    if len(entries) != evidence.entry_count:
        raise TrialEvidenceError("workspace index entry count mismatch")
    if sum(item.type == "file" for item in entries) != evidence.regular_file_count:
        raise TrialEvidenceError("workspace index file count mismatch")
    if (
        sum(item.size_bytes or 0 for item in entries if item.type == "file")
        != evidence.regular_file_bytes
    ):
        raise TrialEvidenceError("workspace index byte count mismatch")
    if deep:
        with tempfile.TemporaryDirectory(prefix="harbor-hf-workspace-verify-") as raw:
            restore_workspace(trial_root, evidence, Path(raw), deadline=deadline)
            observed = list(
                _workspace_entries(
                    Path(raw) / "app",
                    None,
                    deadline=deadline,
                    max_nodes=evidence.entry_count,
                )
            )
            if observed != entries:
                raise TrialEvidenceError("restored workspace does not match file index")
    return entries


def restore_workspace(
    trial_root: Path,
    evidence: WorkspaceEvidence,
    destination: Path,
    *,
    deadline: float | None = None,
) -> None:
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise TrialEvidenceError("workspace restore destination must be a directory")
    if destination.exists() and any(destination.iterdir()):
        raise TrialEvidenceError("workspace restore destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = trial_root / evidence.archive.path
    with tempfile.NamedTemporaryFile(
        prefix="harbor-hf-workspace-", suffix=".tar"
    ) as raw:
        with archive_path.open("rb") as source:
            _decompress_workspace_archive(source, raw, evidence, deadline=deadline)
        raw.flush()
        with tarfile.open(raw.name, mode="r:") as archive:
            members = _validate_tar_members(archive, evidence, deadline=deadline)
            with tempfile.TemporaryDirectory(
                prefix=".harbor-hf-restore-", dir=destination
            ) as staging_raw:
                staging = Path(staging_raw)
                _extract_tar_members(archive, members, staging, deadline=deadline)
                restored_app = staging / "app"
                observed = list(
                    _workspace_entries(
                        restored_app,
                        None,
                        deadline=deadline,
                        max_nodes=evidence.entry_count,
                    )
                )
                _validate_workspace_symlink_graph(restored_app, observed)
                expected = _load_workspace_index(
                    trial_root / evidence.file_index.path, deadline=deadline
                )
                if observed != expected:
                    raise TrialEvidenceError(
                        "restored workspace does not match file index"
                    )
                os.replace(restored_app, destination / "app")


def _decompress_workspace_archive(
    source: BinaryIO,
    target: _BinaryWriter,
    evidence: WorkspaceEvidence,
    *,
    deadline: float | None = None,
) -> None:
    maximum = min(
        64 * 1024**3,
        evidence.regular_file_bytes + evidence.entry_count * 8192 + 1024**2,
    )
    written = 0
    with zstandard.ZstdDecompressor().stream_reader(source) as reader:
        while chunk := reader.read(_CHUNK):
            _check_workspace_deadline(deadline)
            written += len(chunk)
            if written > maximum:
                raise TrialEvidenceError("workspace archive expands beyond safe limit")
            target.write(chunk)


def _move_judge_records(source: Path | None, evidence_dir: Path) -> None:
    if source is None or not source.exists():
        return
    destination = evidence_dir / "judge"
    if destination.exists():
        raise TrialEvidenceError("judge evidence destination already exists")
    os.replace(source, destination)


def _collect_agent_evidence(trial_root: Path) -> AgentEvidence:
    agent_dir = trial_root / "agent"
    sessions = _references_matching(
        trial_root,
        agent_dir,
        lambda path: (
            path.suffix == ".jsonl"
            and "session" in path.as_posix().lower()
            and "trajectory" not in path.name
        ),
    )
    trajectories = _references_matching(
        trial_root, agent_dir, lambda path: "trajectory" in path.name.lower()
    )
    if not sessions or any(reference.size_bytes == 0 for reference in sessions):
        raise TrialEvidenceError("agent execution has no session JSONL with content")
    if not trajectories or any(reference.size_bytes == 0 for reference in trajectories):
        raise TrialEvidenceError("agent trajectory is missing or empty")
    retained = {reference.path for reference in [*sessions, *trajectories]}
    logs = [
        _reference(path, trial_root)
        for path in sorted(agent_dir.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(trial_root).as_posix() not in retained
    ]
    return AgentEvidence(sessions=sessions, trajectories=trajectories, logs=logs)


def _judge_exchange_manifests(evidence_dir: Path, judge_expected: bool) -> list[Path]:
    exchanges = sorted((evidence_dir / "judge").glob("judge-*/exchange.json"))
    if not judge_expected and exchanges:
        raise TrialEvidenceError("judge evidence exists for deterministic task")
    return exchanges


def _collect_judge_evidence(
    trial_root: Path,
    evidence_dir: Path,
    *,
    judge_expected: bool,
    judge_model: str | None,
    execution_id: str,
    trial_id: str,
) -> tuple[JudgeEvidence, list[Path]]:
    from harbor_hf.judge_recorder import (
        JudgeRecorderError,
        verify_judge_recorder_summary,
    )

    exchanges = _judge_exchange_manifests(evidence_dir, judge_expected)
    if not judge_expected:
        if (evidence_dir / "judge").exists():
            raise TrialEvidenceError("judge recorder exists for deterministic task")
        return JudgeEvidence(status="not_expected", expected=False), exchanges
    summary_path = evidence_dir / "judge" / "recorder.json"
    try:
        summary = verify_judge_recorder_summary(summary_path)
    except JudgeRecorderError as error:
        raise TrialEvidenceError("judge recorder summary is invalid") from error
    if (
        summary.execution_id != execution_id
        or summary.trial_id != trial_id
        or summary.model != judge_model
    ):
        raise TrialEvidenceError("judge recorder summary identity mismatch")
    if summary.rejected_call_count:
        raise TrialEvidenceError("judge recorder rejected one or more calls")
    if summary.exchange_count != len(exchanges):
        raise TrialEvidenceError("judge recorder exchange count mismatch")
    _require_successful_judge_exchanges(exchanges)
    return (
        JudgeEvidence(
            status="captured" if exchanges else "not_called",
            expected=True,
            model=judge_model,
            recorder_summary=_reference(summary_path, trial_root),
            exchanges=[
                JudgeExchangeReference(
                    exchange_id=path.parent.name,
                    attempt=int(path.parent.name.removeprefix("judge-")),
                    record=_reference(path, trial_root),
                )
                for path in exchanges
            ],
        ),
        exchanges,
    )


def _require_successful_judge_exchanges(exchanges: list[Path]) -> None:
    from harbor_hf.judge_recorder import JudgeRecorderError, verify_judge_exchange

    try:
        recorded = [verify_judge_exchange(path.parent) for path in exchanges]
    except JudgeRecorderError as error:
        raise TrialEvidenceError("judge exchange is invalid") from error
    if any(exchange.outcome != "success" for exchange in recorded):
        raise TrialEvidenceError("judge exchange did not complete successfully")


def _collect_verifier_evidence(
    trial_root: Path,
    exchanges: list[Path],
    *,
    judge_expected: bool,
) -> VerifierEvidence:
    verifier_dir = trial_root / "verifier"
    stdout = _normalize_verifier_stream(verifier_dir, "test-stdout.txt")
    stderr = _normalize_verifier_stream(verifier_dir, "test-stderr.txt")
    scorecard = verifier_dir / "scorecard.json"
    reward = verifier_dir / "reward.txt"
    if not scorecard.is_file() or not reward.is_file():
        raise TrialEvidenceError("verifier scorecard or reward is missing")
    selection, selected_exchange_id = _judge_selection_reference(
        trial_root, verifier_dir, exchanges, judge_expected=judge_expected
    )
    known = {
        scorecard,
        reward,
        stdout,
        stderr,
        verifier_dir / "judge-selection.json",
    }
    logs = [
        _reference(path, trial_root)
        for path in sorted(verifier_dir.rglob("*"))
        if path.is_file() and path not in known
    ]
    return VerifierEvidence(
        scorecard=_reference(scorecard, trial_root),
        reward=_reference(reward, trial_root),
        stdout=_reference(stdout, trial_root),
        stderr=_reference(stderr, trial_root),
        judge_selection=selection,
        selected_judge_exchange_id=selected_exchange_id,
        logs=logs,
    )


def _judge_selection_reference(
    trial_root: Path,
    verifier_dir: Path,
    exchanges: list[Path],
    *,
    judge_expected: bool,
) -> tuple[FileReference | None, str | None]:
    path = verifier_dir / "judge-selection.json"
    if not judge_expected:
        return None, None
    if not exchanges:
        if path.exists():
            raise TrialEvidenceError("judge selection exists without an exchange")
        return None, None
    if not path.is_file():
        raise TrialEvidenceError("judge selection is missing")
    selection = JudgeSelection.model_validate_json(path.read_text())
    if not any(item.parent.name == selection.exchange_id for item in exchanges):
        raise TrialEvidenceError("judge selection references no complete exchange")
    return _reference(path, trial_root), selection.exchange_id


def _completion_requirements(
    *, judge_expected: bool, has_exchanges: bool
) -> list[EvidenceRequirementName]:
    requirements: list[EvidenceRequirementName] = [
        "agent_session",
        "agent_trajectory",
        "verifier_reward",
        "verifier_scorecard",
        "verifier_stderr",
        "verifier_stdout",
        "workspace",
    ]
    if judge_expected:
        requirements.append("judge_recorder")
    if has_exchanges:
        requirements.extend(["judge_exchange", "judge_selection"])
    return sorted(requirements)


def assemble_trial_evidence(
    trial_root: Path,
    *,
    campaign_id: str | None,
    run_id: str,
    execution_id: str,
    trial_id: str,
    task_name: str,
    task_digest: str,
    logical_attempt: int,
    physical_attempt: int,
    judge_expected: bool,
    judge_model: str | None,
    policy: TrialEvidencePolicy,
    captured_at: datetime | None = None,
    judge_records_dir: Path | None = None,
    known_secrets: tuple[str, ...] = (),
    remove_raw_workspace: bool = True,
) -> TrialEvidenceManifest:
    evidence_dir = trial_root / "evidence"
    if evidence_dir.exists():
        raise TrialEvidenceError("trial evidence already exists")
    evidence_dir.mkdir()
    snapshot = trial_root / "artifacts" / WORKSPACE_DESTINATION
    assert_known_secrets_absent(trial_root, known_secrets)
    package = package_workspace(snapshot, evidence_dir, policy=policy)
    _move_judge_records(judge_records_dir, evidence_dir)
    agent = _collect_agent_evidence(trial_root)
    judge, exchanges = _collect_judge_evidence(
        trial_root,
        evidence_dir,
        judge_expected=judge_expected,
        judge_model=judge_model,
        execution_id=execution_id,
        trial_id=trial_id,
    )
    verifier = _collect_verifier_evidence(
        trial_root, exchanges, judge_expected=judge_expected
    )
    manifest = TrialEvidenceManifest(
        campaign_id=campaign_id,
        run_id=run_id,
        execution_id=execution_id,
        trial_id=trial_id,
        task_name=task_name,
        task_digest=task_digest,
        logical_attempt=logical_attempt,
        physical_attempt=physical_attempt,
        captured_at=(captured_at or datetime.now(UTC)).astimezone(UTC),
        workspace=package.evidence,
        agent=agent,
        judge=judge,
        verifier=verifier,
        completion=CompletionEvidence(
            requirements=_completion_requirements(
                judge_expected=judge_expected, has_exchanges=bool(exchanges)
            )
        ),
    )
    manifest_path = evidence_dir / "manifest.json"
    _atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    try:
        verify_trial_evidence(trial_root, deep=True)
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise
    if remove_raw_workspace:
        shutil.rmtree(snapshot)
    return manifest


def verify_trial_evidence(
    trial_root: Path, *, deep: bool = False
) -> TrialEvidenceManifest:
    manifest_path = trial_root / "evidence" / "manifest.json"
    try:
        manifest = TrialEvidenceManifest.model_validate_json(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise TrialEvidenceError("trial evidence manifest is invalid") from error
    references = [
        manifest.workspace.archive,
        manifest.workspace.file_index,
        *manifest.agent.sessions,
        *manifest.agent.trajectories,
        *manifest.agent.logs,
        *(item.record for item in manifest.judge.exchanges),
        manifest.verifier.scorecard,
        manifest.verifier.reward,
        manifest.verifier.stdout,
        manifest.verifier.stderr,
        *manifest.verifier.logs,
    ]
    if manifest.judge.recorder_summary is not None:
        references.append(manifest.judge.recorder_summary)
    if manifest.verifier.judge_selection is not None:
        references.append(manifest.verifier.judge_selection)
    paths = [reference.path for reference in references]
    if len(paths) != len(set(paths)):
        raise TrialEvidenceError("trial evidence references duplicate files")
    for reference in references:
        _verify_reference(trial_root, reference)
    _verify_component_file_set(
        trial_root / "agent",
        {
            reference.path
            for reference in [
                *manifest.agent.sessions,
                *manifest.agent.trajectories,
                *manifest.agent.logs,
            ]
        },
        trial_root,
    )
    _verify_component_file_set(
        trial_root / "verifier",
        {
            reference.path
            for reference in [
                manifest.verifier.scorecard,
                manifest.verifier.reward,
                manifest.verifier.stdout,
                manifest.verifier.stderr,
                *manifest.verifier.logs,
                *(
                    [manifest.verifier.judge_selection]
                    if manifest.verifier.judge_selection is not None
                    else []
                ),
            ]
        },
        trial_root,
    )
    _verify_evidence_root_files(trial_root, manifest)
    _verify_manifest_judge_evidence(trial_root, manifest)
    verify_workspace_package(trial_root, manifest.workspace, deep=deep)
    return manifest


def _verify_component_file_set(
    component_root: Path, expected: set[str], trial_root: Path
) -> None:
    observed: set[str] = set()
    for path in component_root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise TrialEvidenceError("evidence component contains unsafe file type")
        if path.is_file():
            observed.add(path.relative_to(trial_root).as_posix())
    if observed != expected:
        raise TrialEvidenceError("evidence component file set is incomplete")


def _verify_evidence_root_files(
    trial_root: Path, manifest: TrialEvidenceManifest
) -> None:
    evidence_root = trial_root / "evidence"
    expected = {
        "manifest.json",
        PurePosixPath(manifest.workspace.archive.path).name,
        PurePosixPath(manifest.workspace.file_index.path).name,
    }
    children = list(evidence_root.iterdir())
    if any(path.is_symlink() for path in children):
        raise TrialEvidenceError("evidence root contains a symlink")
    observed = {path.name for path in children if path.is_file()}
    if observed != expected:
        raise TrialEvidenceError("evidence root file set is incomplete")
    expected_directories = {"judge"} if manifest.judge.expected else set()
    observed_directories = {path.name for path in children if path.is_dir()}
    if observed_directories != expected_directories:
        raise TrialEvidenceError("judge evidence directory expectation mismatch")


def _verify_manifest_judge_evidence(
    trial_root: Path, manifest: TrialEvidenceManifest
) -> None:
    from harbor_hf.judge_recorder import (
        JudgeRecorderError,
        verify_judge_exchange,
        verify_judge_recorder_summary,
    )

    try:
        summary_reference = manifest.judge.recorder_summary
        if summary_reference is not None:
            summary = verify_judge_recorder_summary(trial_root / summary_reference.path)
            if (
                summary.execution_id != manifest.execution_id
                or summary.trial_id != manifest.trial_id
                or summary.model != manifest.judge.model
                or summary.exchange_count != len(manifest.judge.exchanges)
                or summary.rejected_call_count != 0
            ):
                raise TrialEvidenceError(
                    "judge recorder summary disagrees with manifest"
                )
        exchanges = [
            verify_judge_exchange((trial_root / reference.record.path).parent)
            for reference in manifest.judge.exchanges
        ]
    except JudgeRecorderError as error:
        raise TrialEvidenceError("judge evidence is invalid") from error
    if manifest.judge.expected:
        _verify_judge_directory_set(trial_root, manifest.judge.exchanges)
    if any(
        exchange.execution_id != manifest.execution_id
        or exchange.trial_id != manifest.trial_id
        or exchange.forwarded_model != manifest.judge.model
        for exchange in exchanges
    ):
        raise TrialEvidenceError("judge exchange identity disagrees with manifest")
    if any(exchange.outcome != "success" for exchange in exchanges):
        raise TrialEvidenceError("judge exchange did not complete successfully")
    _verify_judge_selection(trial_root, manifest)


def _verify_judge_directory_set(
    trial_root: Path, exchanges: list[JudgeExchangeReference]
) -> None:
    judge_dir = trial_root / "evidence" / "judge"
    children = list(judge_dir.iterdir())
    if any(path.is_symlink() for path in children):
        raise TrialEvidenceError("judge evidence contains a symlink")
    observed = {path.name for path in children}
    expected = {"recorder.json", *(reference.exchange_id for reference in exchanges)}
    if observed != expected:
        raise TrialEvidenceError("judge evidence file set is incomplete")


def _verify_judge_selection(trial_root: Path, manifest: TrialEvidenceManifest) -> None:
    selection_reference = manifest.verifier.judge_selection
    if not manifest.judge.exchanges:
        if selection_reference is not None:
            raise TrialEvidenceError("judge selection exists without an exchange")
        return
    if selection_reference is None:
        raise TrialEvidenceError("judge selection is missing")
    try:
        selection = JudgeSelection.model_validate_json(
            (trial_root / selection_reference.path).read_text()
        )
    except (OSError, ValueError) as error:
        raise TrialEvidenceError("judge selection is invalid") from error
    exchange_ids = {reference.exchange_id for reference in manifest.judge.exchanges}
    if selection.exchange_id not in exchange_ids:
        raise TrialEvidenceError("judge selection references no complete exchange")
    if selection.exchange_id != manifest.verifier.selected_judge_exchange_id:
        raise TrialEvidenceError(
            "selected judge exchange disagrees with selection file"
        )


def assert_known_secrets_absent(root: Path, secrets: tuple[str, ...]) -> None:
    needles = tuple(secret.encode() for secret in secrets if secret)
    if not needles:
        return
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(secret in relative for secret in secrets if secret):
            raise TrialEvidenceError("known secret detected in workspace path")
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as stream:
            carry = b""
            overlap = max(len(needle) for needle in needles) - 1
            while chunk := stream.read(_CHUNK):
                data = carry + chunk
                if any(needle in data for needle in needles):
                    raise TrialEvidenceError("known secret detected in workspace file")
                carry = data[-overlap:] if overlap else b""


def write_trial_evidence_schemas(destination: Path) -> None:
    from harbor_hf.judge_recorder import JudgeRecorderSummary

    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        destination / "trial-evidence-v1.schema.json",
        TrialEvidenceManifest.model_json_schema(),
    )
    _atomic_write_json(
        destination / "workspace-file-v1.schema.json", WorkspaceFile.model_json_schema()
    )
    _atomic_write_json(
        destination / "judge-selection-v1.schema.json",
        JudgeSelection.model_json_schema(),
    )
    _atomic_write_json(
        destination / "judge-recorder-summary-v1.schema.json",
        JudgeRecorderSummary.model_json_schema(),
    )


def _workspace_entries(
    snapshot: Path,
    policy: TrialEvidencePolicy | None,
    *,
    deadline: float | None = None,
    max_nodes: int | None = None,
) -> Iterator[WorkspaceFile]:
    total = 0
    root_metadata = snapshot.lstat()
    root_entry = WorkspaceFile(
        path=".",
        type="directory",
        mode=stat.S_IMODE(root_metadata.st_mode),
    )
    _validate_workspace_limits(root_entry, 1, total, policy)
    yield root_entry
    effective_max = policy.workspace_max_nodes if policy is not None else max_nodes
    paths = _bounded_workspace_paths(snapshot, effective_max)
    for count, path in enumerate(paths, 2):
        _check_workspace_deadline(deadline)
        entry = _workspace_entry(snapshot, path, deadline=deadline)
        if entry.type == "file":
            total += entry.size_bytes or 0
        _validate_workspace_limits(entry, count, total, policy)
        yield entry


def _bounded_workspace_paths(snapshot: Path, max_nodes: int | None) -> list[Path]:
    paths: list[Path] = []
    pending = [snapshot]
    while pending:
        directory = pending.pop()
        try:
            entries = os.scandir(directory)
        except OSError as error:
            raise TrialEvidenceError("workspace directory cannot be read") from error
        with entries:
            for item in entries:
                path = Path(item.path)
                paths.append(path)
                if max_nodes is not None and len(paths) + 1 > max_nodes:
                    raise TrialEvidenceError("workspace exceeds configured file limit")
                if item.is_dir(follow_symlinks=False):
                    pending.append(path)
    return sorted(
        paths, key=lambda item: item.relative_to(snapshot).as_posix().encode()
    )


def _validate_workspace_symlink_graph(
    snapshot: Path, entries: list[WorkspaceFile]
) -> None:
    graph = _workspace_directory_graph(snapshot, entries)
    _require_acyclic_directory_graph(graph)


def _workspace_directory_graph(
    snapshot: Path, entries: list[WorkspaceFile]
) -> dict[str, set[str]]:
    root = snapshot.resolve(strict=True)
    directories = {entry.path for entry in entries if entry.type == "directory"}
    graph = {path: set[str]() for path in directories}
    for path in directories - {"."}:
        parent = PurePosixPath(path).parent.as_posix()
        graph.setdefault(parent, set()).add(path)
    for entry in entries:
        if entry.type != "symlink":
            continue
        resolved = _safe_workspace_symlink_target(snapshot, root, entry)
        if resolved.is_dir():
            parent = PurePosixPath(entry.path).parent.as_posix()
            graph.setdefault(parent, set()).add(resolved.relative_to(root).as_posix())
    return graph


def _safe_workspace_symlink_target(
    snapshot: Path, root: Path, entry: WorkspaceFile
) -> Path:
    target = entry.target or ""
    if PurePosixPath(target).is_absolute():
        raise TrialEvidenceError("workspace symlink target must be relative")
    try:
        resolved = (snapshot / entry.path).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise TrialEvidenceError(
            "workspace symlink target is dangling or cyclic"
        ) from error
    if not resolved.is_relative_to(root):
        raise TrialEvidenceError("workspace symlink target escapes /app")
    return resolved


def _require_acyclic_directory_graph(graph: dict[str, set[str]]) -> None:
    indegree = {path: 0 for path in graph}
    for targets in graph.values():
        for target in targets:
            indegree[target] = indegree.get(target, 0) + 1
    pending = deque(path for path, degree in indegree.items() if degree == 0)
    visited = 0
    while pending:
        current = pending.popleft()
        visited += 1
        for target in graph.get(current, set()):
            indegree[target] -= 1
            if indegree[target] == 0:
                pending.append(target)
    if visited != len(indegree):
        raise TrialEvidenceError("workspace symlink graph contains a cycle")


def _workspace_entry(
    snapshot: Path,
    path: Path,
    *,
    deadline: float | None,
) -> WorkspaceFile:
    relative = path.relative_to(snapshot).as_posix()
    _safe_relative_path(relative)
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISDIR(metadata.st_mode):
        return WorkspaceFile(path=relative, type="directory", mode=mode)
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        if "\x00" in target:
            raise TrialEvidenceError("workspace symlink target contains NUL")
        return WorkspaceFile(path=relative, type="symlink", mode=mode, target=target)
    if not stat.S_ISREG(metadata.st_mode):
        raise TrialEvidenceError(
            f"workspace contains unsupported special file: {relative}"
        )
    return WorkspaceFile(
        path=relative,
        type="file",
        mode=mode,
        size_bytes=metadata.st_size,
        sha256=_digest(path, deadline=deadline),
    )


def _validate_workspace_limits(
    entry: WorkspaceFile,
    count: int,
    total: int,
    policy: TrialEvidencePolicy | None,
) -> None:
    if policy is None:
        return
    if count > policy.workspace_max_nodes:
        raise TrialEvidenceError("workspace exceeds configured file limit")
    if (
        entry.type == "file"
        and entry.size_bytes is not None
        and entry.size_bytes > policy.workspace_max_file_bytes
    ):
        raise TrialEvidenceError("workspace file exceeds configured byte limit")
    if total > policy.workspace_max_total_bytes:
        raise TrialEvidenceError("workspace exceeds configured byte limit")


def _write_workspace_archive(
    snapshot: Path,
    entries: list[WorkspaceFile],
    destination: Path,
    *,
    deadline: float | None,
    maximum_bytes: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    compressed = destination.with_name(destination.name + ".tmp")
    try:
        with compressed.open("wb") as raw_target:
            target = _BoundedWriter(raw_target, maximum_bytes)
            compressor = zstandard.ZstdCompressor(
                level=10, threads=0, write_checksum=True
            )
            with (
                compressor.stream_writer(
                    cast(BinaryIO, target), closefd=False
                ) as writer,
                tarfile.open(
                    fileobj=cast(BinaryIO, writer),
                    mode="w|",
                    format=tarfile.PAX_FORMAT,
                ) as archive,
            ):
                for entry in entries:
                    _check_workspace_deadline(deadline)
                    path = snapshot / entry.path
                    archive_name = "app" if entry.path == "." else f"app/{entry.path}"
                    info = tarfile.TarInfo(archive_name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = entry.mode
                    if entry.type == "directory":
                        info.type = tarfile.DIRTYPE
                        archive.addfile(info)
                    elif entry.type == "symlink":
                        info.type = tarfile.SYMTYPE
                        info.linkname = entry.target or ""
                        archive.addfile(info)
                    else:
                        info.type = tarfile.REGTYPE
                        info.size = entry.size_bytes or 0
                        with path.open("rb") as stream:
                            archive.addfile(
                                info,
                                cast(BinaryIO, _DeadlineReader(stream, deadline)),
                            )
        _check_workspace_deadline(deadline)
        os.replace(compressed, destination)
    finally:
        compressed.unlink(missing_ok=True)


def _validate_tar_members(
    archive: tarfile.TarFile,
    evidence: WorkspaceEvidence,
    *,
    deadline: float | None = None,
) -> list[tarfile.TarInfo]:
    seen: dict[str, Literal["directory", "file", "symlink"]] = {}
    symlinks: set[PurePosixPath] = set()
    members = archive.getmembers()
    if not members or members[0].name != "app" or not members[0].isdir():
        raise TrialEvidenceError("workspace archive has no top-level app directory")
    for member in members:
        _check_workspace_deadline(deadline)
        relative = _archive_relative_path(member.name)
        member_path = PurePosixPath(relative)
        if relative in seen:
            raise TrialEvidenceError("workspace archive contains duplicate path")
        if any(parent in symlinks for parent in member_path.parents):
            raise TrialEvidenceError("workspace archive writes through a symlink")
        seen[relative] = _tar_member_kind(member, seen, symlinks, member_path)
    if len(members) != evidence.entry_count:
        raise TrialEvidenceError("workspace archive entry count mismatch")
    unpacked = sum(member.size for member in members if member.isfile())
    if unpacked != evidence.regular_file_bytes:
        raise TrialEvidenceError("workspace archive byte count mismatch")
    return members


def _extract_tar_members(
    archive: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    destination: Path,
    *,
    deadline: float | None,
) -> None:
    for member in members:
        _check_workspace_deadline(deadline)
        target = destination / member.name
        if member.isdir():
            target.mkdir()
            target.chmod(member.mode)
        elif member.issym():
            os.symlink(member.linkname, target)
        else:
            source = archive.extractfile(member)
            if source is None:
                raise TrialEvidenceError("workspace archive file cannot be read")
            with source, target.open("xb") as output:
                while chunk := source.read(_CHUNK):
                    _check_workspace_deadline(deadline)
                    output.write(chunk)
            target.chmod(member.mode)


def _archive_relative_path(member_name: str) -> str:
    if member_name == "app":
        return "."
    if member_name.startswith("app/"):
        relative = member_name.removeprefix("app/")
        _safe_relative_path(relative)
        return relative
    raise TrialEvidenceError("workspace archive has no top-level app directory")


def _tar_member_kind(
    member: tarfile.TarInfo,
    seen: dict[str, Literal["directory", "file", "symlink"]],
    symlinks: set[PurePosixPath],
    member_path: PurePosixPath,
) -> Literal["directory", "file", "symlink"]:
    if member.isdir():
        return "directory"
    if member.isfile():
        return "file"
    if member.issym():
        symlinks.add(member_path)
        return "symlink"
    raise TrialEvidenceError("workspace archive contains unsupported member")


def _load_workspace_index(
    path: Path, *, deadline: float | None = None
) -> list[WorkspaceFile]:
    entries: list[WorkspaceFile] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                _check_workspace_deadline(deadline)
                entries.append(WorkspaceFile.model_validate_json(line))
    except (OSError, ValueError) as error:
        raise TrialEvidenceError("workspace file index is invalid") from error
    paths = [entry.path for entry in entries]
    if paths != sorted(paths, key=lambda value: value.encode()) or len(paths) != len(
        set(paths)
    ):
        raise TrialEvidenceError("workspace index paths must be byte-sorted and unique")
    return entries


def _normalize_verifier_stream(root: Path, name: str) -> Path:
    path = root / name
    if not path.exists():
        _atomic_write(path, b"")
    if not path.is_file() or path.is_symlink():
        raise TrialEvidenceError(f"verifier stream is invalid: {name}")
    return path


def _references_matching(
    trial_root: Path, root: Path, predicate: Callable[[Path], bool]
) -> list[FileReference]:
    if not root.is_dir():
        return []
    return [
        _reference(path, trial_root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and predicate(path)
    ]


def _reference(
    path: Path, root: Path, *, deadline: float | None = None
) -> FileReference:
    if path.is_symlink() or not path.is_file():
        raise TrialEvidenceError(f"evidence reference is not a regular file: {path}")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise TrialEvidenceError("evidence reference escapes trial root") from error
    return FileReference(
        path=relative,
        size_bytes=path.stat().st_size,
        sha256=_digest(path, deadline=deadline),
        media_type=_media_type(path),
    )


def _media_type(path: Path) -> MediaType:
    if path.name == "workspace.tar.zst":
        return "application/vnd.harbor-hf.workspace+tar+zstd"
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".jsonl":
        return "application/x-ndjson"
    if path.suffix in {".log", ".txt"}:
        return "text/plain"
    return "application/octet-stream"


def _verify_reference(
    root: Path, reference: FileReference, *, deadline: float | None = None
) -> None:
    path = root / reference.path
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise TrialEvidenceError(
            f"evidence reference is missing or unsafe: {reference.path}"
        )
    if (
        path.stat().st_size != reference.size_bytes
        or _digest(path, deadline=deadline) != reference.sha256
    ):
        raise TrialEvidenceError(
            f"evidence reference digest mismatch: {reference.path}"
        )


def _require_sorted_unique(references: list[FileReference]) -> None:
    paths = [item.path for item in references]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("file references must be sorted and unique")


def _safe_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError("evidence path must be normalized and relative")


def _digest(path: Path, *, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            _check_workspace_deadline(deadline)
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _check_workspace_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() > deadline:
        raise TrialEvidenceError("workspace capture exceeded configured timeout")


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write(
        path, json.dumps(value, indent=2, sort_keys=True, default=str).encode() + b"\n"
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
