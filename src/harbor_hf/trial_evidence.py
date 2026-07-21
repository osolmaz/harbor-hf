from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import BinaryIO, Literal, Protocol

import zstandard
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harbor_hf.models import TrialEvidencePolicy

TRIAL_EVIDENCE_SCHEMA = "harbor-hf/trial-evidence/v1"
WORKSPACE_INDEX_SCHEMA = "harbor-hf/workspace-file/v1"
JUDGE_SELECTION_SCHEMA = "harbor-hf/judge-selection/v1"
WORKSPACE_SOURCE = PurePosixPath("/app")
WORKSPACE_DESTINATION = PurePosixPath("workspace/app")
_CHUNK = 1024 * 1024


class _BinaryWriter(Protocol):
    def write(self, data: bytes, /) -> int: ...


class TrialEvidenceError(RuntimeError):
    """Raised when exact trial evidence cannot be captured or validated."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FileReference(FrozenModel):
    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def path_is_safe(self) -> FileReference:
        _safe_relative_path(self.path)
        return self


class WorkspaceFile(FrozenModel):
    schema_version: Literal["harbor-hf/workspace-file/v1"] = WORKSPACE_INDEX_SCHEMA
    path: str = Field(min_length=1)
    type: Literal["directory", "file", "symlink", "hardlink"]
    mode: int = Field(ge=0, le=0o7777)
    size: int = Field(ge=0)
    sha256: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    target: str | None = None

    @model_validator(mode="after")
    def fields_match_type(self) -> WorkspaceFile:
        _safe_relative_path(self.path)
        if self.type == "file" and self.sha256 is None:
            raise ValueError("workspace file requires sha256")
        if self.type in {"symlink", "hardlink"} and not self.target:
            raise ValueError("workspace link requires target")
        if self.type not in {"symlink", "hardlink"} and self.target is not None:
            raise ValueError("workspace non-link cannot have target")
        if self.type != "file" and self.sha256 is not None:
            raise ValueError("workspace non-file cannot have sha256")
        return self


class WorkspaceEvidence(FrozenModel):
    status: Literal["captured"] = "captured"
    root: Literal["/app"] = "/app"
    archive: FileReference
    index: FileReference
    entry_count: int = Field(ge=0)
    file_count: int = Field(ge=0)
    unpacked_bytes: int = Field(ge=0)
    archive_bytes: int = Field(ge=0)
    compression: Literal["zstd"] = "zstd"


class AgentEvidence(FrozenModel):
    sessions: list[FileReference]
    trajectories: list[FileReference]
    logs: list[FileReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def lists_are_canonical(self) -> AgentEvidence:
        _require_sorted_unique(self.sessions)
        _require_sorted_unique(self.trajectories)
        _require_sorted_unique(self.logs)
        return self


class JudgeEvidence(FrozenModel):
    expected: bool
    model: str | None = None
    recorder_summary: FileReference | None = None
    exchanges: list[FileReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def expectation_is_consistent(self) -> JudgeEvidence:
        _require_sorted_unique(self.exchanges)
        if self.expected and (not self.model or self.recorder_summary is None):
            raise ValueError("expected judge evidence requires a recorder summary")
        if not self.expected and (
            self.model is not None
            or self.recorder_summary is not None
            or self.exchanges
        ):
            raise ValueError("unexpected judge evidence must be empty")
        return self


class VerifierEvidence(FrozenModel):
    scorecard: FileReference
    reward: FileReference
    stdout: FileReference
    stderr: FileReference
    judge_selection: FileReference | None = None
    logs: list[FileReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def logs_are_canonical(self) -> VerifierEvidence:
        _require_sorted_unique(self.logs)
        return self


class EvidenceRequirement(FrozenModel):
    name: Literal["workspace", "agent", "judge", "verifier"]
    required: bool
    satisfied: bool

    @model_validator(mode="after")
    def required_is_satisfied(self) -> EvidenceRequirement:
        if self.required and not self.satisfied:
            raise ValueError(f"required evidence is missing: {self.name}")
        return self


class CompletionEvidence(FrozenModel):
    status: Literal["complete"] = "complete"
    requirements: list[EvidenceRequirement]

    @model_validator(mode="after")
    def requirements_are_canonical(self) -> CompletionEvidence:
        names = [item.name for item in self.requirements]
        if names != sorted(names) or len(names) != len(set(names)):
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
    workspace_paths = [manifest.workspace.archive.path, manifest.workspace.index.path]
    if any(not path.startswith("evidence/") for path in workspace_paths):
        raise ValueError("workspace evidence must remain under evidence/")
    agent_refs = [
        *manifest.agent.sessions,
        *manifest.agent.trajectories,
        *manifest.agent.logs,
    ]
    if any(not item.path.startswith("agent/") for item in agent_refs):
        raise ValueError("agent evidence must remain under agent/")
    judge_refs = [*manifest.judge.exchanges]
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
    requirements = {item.name: item for item in manifest.completion.requirements}
    if set(requirements) != {"agent", "judge", "verifier", "workspace"}:
        raise ValueError("trial evidence has an incomplete requirement set")
    expected_required: dict[
        Literal["workspace", "agent", "judge", "verifier"], bool
    ] = {
        "agent": True,
        "judge": manifest.judge.expected,
        "verifier": True,
        "workspace": True,
    }
    if any(
        requirements[name].required != required
        for name, required in expected_required.items()
    ):
        raise ValueError("trial evidence requirement policy is inconsistent")


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
    _write_workspace_archive(snapshot, entries, archive_path, deadline=deadline)
    archive_size = archive_path.stat().st_size
    if archive_size > policy.workspace_max_archive_bytes:
        archive_path.unlink(missing_ok=True)
        raise TrialEvidenceError("workspace archive exceeds configured byte limit")
    files = [entry for entry in entries if entry.type == "file"]
    unpacked = sum(entry.size for entry in files)
    evidence = WorkspaceEvidence(
        archive=_reference(archive_path, evidence_dir.parent),
        index=_reference(index_path, evidence_dir.parent),
        entry_count=len(entries),
        file_count=len(files),
        unpacked_bytes=unpacked,
        archive_bytes=archive_size,
    )
    package = WorkspacePackage(evidence=evidence, entries=entries)
    verify_workspace_package(evidence_dir.parent, package.evidence, deep=True)
    return package


def verify_workspace_package(
    trial_root: Path, evidence: WorkspaceEvidence, *, deep: bool
) -> list[WorkspaceFile]:
    _verify_reference(trial_root, evidence.archive)
    _verify_reference(trial_root, evidence.index)
    entries = _load_workspace_index(trial_root / evidence.index.path)
    if len(entries) != evidence.entry_count:
        raise TrialEvidenceError("workspace index entry count mismatch")
    if sum(item.type == "file" for item in entries) != evidence.file_count:
        raise TrialEvidenceError("workspace index file count mismatch")
    if (
        sum(item.size for item in entries if item.type == "file")
        != evidence.unpacked_bytes
    ):
        raise TrialEvidenceError("workspace index byte count mismatch")
    if (trial_root / evidence.archive.path).stat().st_size != evidence.archive_bytes:
        raise TrialEvidenceError("workspace archive byte count mismatch")
    if deep:
        with tempfile.TemporaryDirectory(prefix="harbor-hf-workspace-verify-") as raw:
            restore_workspace(trial_root, evidence, Path(raw))
            observed = list(_workspace_entries(Path(raw), None))
            if observed != entries:
                raise TrialEvidenceError("restored workspace does not match file index")
    return entries


def restore_workspace(
    trial_root: Path, evidence: WorkspaceEvidence, destination: Path
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
            _decompress_workspace_archive(source, raw, evidence)
        raw.flush()
        with tarfile.open(raw.name, mode="r:") as archive:
            _validate_tar_members(archive, evidence)
            archive.extractall(destination, filter="fully_trusted")


def _decompress_workspace_archive(
    source: BinaryIO, target: _BinaryWriter, evidence: WorkspaceEvidence
) -> None:
    maximum = min(
        64 * 1024**3,
        evidence.unpacked_bytes + evidence.entry_count * 8192 + 1024**2,
    )
    written = 0
    with zstandard.ZstdDecompressor().stream_reader(source) as reader:
        while chunk := reader.read(_CHUNK):
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
    if not sessions:
        raise TrialEvidenceError("agent execution has no session JSONL")
    if not trajectories:
        raise TrialEvidenceError("agent trajectory is missing")
    logs = _references_matching(
        trial_root, agent_dir, lambda path: path.suffix in {".log", ".txt"}
    )
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
) -> tuple[JudgeEvidence, list[Path]]:
    from harbor_hf.judge_recorder import (
        JudgeRecorderError,
        verify_judge_recorder_summary,
    )

    exchanges = _judge_exchange_manifests(evidence_dir, judge_expected)
    if not judge_expected:
        if (evidence_dir / "judge").exists():
            raise TrialEvidenceError("judge recorder exists for deterministic task")
        return JudgeEvidence(expected=False), exchanges
    summary_path = evidence_dir / "judge" / "recorder.json"
    try:
        summary = verify_judge_recorder_summary(summary_path)
    except JudgeRecorderError as error:
        raise TrialEvidenceError("judge recorder summary is invalid") from error
    if summary.execution_id != execution_id or summary.model != judge_model:
        raise TrialEvidenceError("judge recorder summary identity mismatch")
    if summary.exchange_count != len(exchanges):
        raise TrialEvidenceError("judge recorder exchange count mismatch")
    return (
        JudgeEvidence(
            expected=True,
            model=judge_model,
            recorder_summary=_reference(summary_path, trial_root),
            exchanges=[_reference(path, trial_root) for path in exchanges],
        ),
        exchanges,
    )


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
    selection = _judge_selection_reference(
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
        logs=logs,
    )


def _judge_selection_reference(
    trial_root: Path,
    verifier_dir: Path,
    exchanges: list[Path],
    *,
    judge_expected: bool,
) -> FileReference | None:
    path = verifier_dir / "judge-selection.json"
    if not judge_expected:
        return None
    if not exchanges:
        if path.exists():
            raise TrialEvidenceError("judge selection exists without an exchange")
        return None
    if not path.exists() and len(exchanges) == 1:
        _atomic_write_json(
            path,
            JudgeSelection(exchange_id=exchanges[0].parent.name).model_dump(
                mode="json"
            ),
        )
    if not path.is_file():
        raise TrialEvidenceError("judge selection is missing")
    selection = JudgeSelection.model_validate_json(path.read_text())
    if not any(item.parent.name == selection.exchange_id for item in exchanges):
        raise TrialEvidenceError("judge selection references no complete exchange")
    return _reference(path, trial_root)


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
    assert_known_secrets_absent(snapshot, known_secrets)
    package = package_workspace(snapshot, evidence_dir, policy=policy)
    _move_judge_records(judge_records_dir, evidence_dir)
    agent = _collect_agent_evidence(trial_root)
    judge, exchanges = _collect_judge_evidence(
        trial_root,
        evidence_dir,
        judge_expected=judge_expected,
        judge_model=judge_model,
        execution_id=execution_id,
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
            requirements=[
                EvidenceRequirement(name="agent", required=True, satisfied=True),
                EvidenceRequirement(
                    name="judge",
                    required=judge_expected,
                    satisfied=judge_expected,
                ),
                EvidenceRequirement(name="verifier", required=True, satisfied=True),
                EvidenceRequirement(name="workspace", required=True, satisfied=True),
            ]
        ),
    )
    _atomic_write_json(evidence_dir / "manifest.json", manifest.model_dump(mode="json"))
    verify_trial_evidence(trial_root, deep=True)
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
        manifest.workspace.index,
        *manifest.agent.sessions,
        *manifest.agent.trajectories,
        *manifest.agent.logs,
        *manifest.judge.exchanges,
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
    _verify_manifest_judge_evidence(trial_root, manifest)
    verify_workspace_package(trial_root, manifest.workspace, deep=deep)
    return manifest


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
                or summary.model != manifest.judge.model
                or summary.exchange_count != len(manifest.judge.exchanges)
            ):
                raise TrialEvidenceError(
                    "judge recorder summary disagrees with manifest"
                )
        exchanges = [
            verify_judge_exchange((trial_root / reference.path).parent)
            for reference in manifest.judge.exchanges
        ]
    except JudgeRecorderError as error:
        raise TrialEvidenceError("judge evidence is invalid") from error
    if any(
        exchange.execution_id != manifest.execution_id
        or exchange.forwarded_model != manifest.judge.model
        for exchange in exchanges
    ):
        raise TrialEvidenceError("judge exchange identity disagrees with manifest")
    _verify_judge_selection(trial_root, manifest)


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
    exchange_ids = {
        Path(reference.path).parent.name for reference in manifest.judge.exchanges
    }
    if selection.exchange_id not in exchange_ids:
        raise TrialEvidenceError("judge selection references no complete exchange")


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
) -> Iterator[WorkspaceFile]:
    total = 0
    inodes: dict[tuple[int, int], str] = {}
    paths = sorted(
        snapshot.rglob("*"),
        key=lambda item: item.relative_to(snapshot).as_posix().encode(),
    )
    for count, path in enumerate(paths, 1):
        _check_workspace_deadline(deadline)
        entry = _workspace_entry(snapshot, path, inodes, deadline=deadline)
        if entry.type == "file":
            total += entry.size
        _validate_workspace_limits(entry, count, total, policy)
        yield entry


def _workspace_entry(
    snapshot: Path,
    path: Path,
    inodes: dict[tuple[int, int], str],
    *,
    deadline: float | None,
) -> WorkspaceFile:
    relative = path.relative_to(snapshot).as_posix()
    _safe_relative_path(relative)
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISDIR(metadata.st_mode):
        return WorkspaceFile(path=relative, type="directory", mode=mode, size=0)
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        if "\x00" in target:
            raise TrialEvidenceError("workspace symlink target contains NUL")
        return WorkspaceFile(
            path=relative, type="symlink", mode=mode, size=0, target=target
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise TrialEvidenceError(
            f"workspace contains unsupported special file: {relative}"
        )
    inode = (metadata.st_dev, metadata.st_ino)
    if metadata.st_nlink > 1 and inode in inodes:
        return WorkspaceFile(
            path=relative,
            type="hardlink",
            mode=mode,
            size=0,
            target=inodes[inode],
        )
    inodes[inode] = relative
    return WorkspaceFile(
        path=relative,
        type="file",
        mode=mode,
        size=metadata.st_size,
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
    if entry.type == "file" and entry.size > policy.workspace_max_file_bytes:
        raise TrialEvidenceError("workspace file exceeds configured byte limit")
    if total > policy.workspace_max_total_bytes:
        raise TrialEvidenceError("workspace exceeds configured byte limit")


def _write_workspace_archive(
    snapshot: Path,
    entries: list[WorkspaceFile],
    destination: Path,
    *,
    deadline: float | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tar_name = tempfile.mkstemp(
        prefix=".workspace-", suffix=".tar", dir=destination.parent
    )
    os.close(descriptor)
    tar_path = Path(tar_name)
    compressed = destination.with_name(destination.name + ".tmp")
    try:
        with tarfile.open(tar_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for entry in entries:
                _check_workspace_deadline(deadline)
                path = snapshot / entry.path
                info = tarfile.TarInfo(entry.path)
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
                elif entry.type == "hardlink":
                    info.type = tarfile.LNKTYPE
                    info.linkname = entry.target or ""
                    archive.addfile(info)
                else:
                    info.type = tarfile.REGTYPE
                    info.size = entry.size
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
        with tar_path.open("rb") as source, compressed.open("wb") as target:
            compressor = zstandard.ZstdCompressor(
                level=10, threads=0, write_checksum=True
            )
            with compressor.stream_writer(target, closefd=False) as writer:
                while chunk := source.read(_CHUNK):
                    _check_workspace_deadline(deadline)
                    writer.write(chunk)
        _check_workspace_deadline(deadline)
        os.replace(compressed, destination)
    finally:
        tar_path.unlink(missing_ok=True)
        compressed.unlink(missing_ok=True)


def _validate_tar_members(
    archive: tarfile.TarFile, evidence: WorkspaceEvidence
) -> None:
    seen: dict[str, Literal["directory", "file", "symlink", "hardlink"]] = {}
    symlinks: set[PurePosixPath] = set()
    members = archive.getmembers()
    for member in members:
        _safe_relative_path(member.name)
        member_path = PurePosixPath(member.name)
        if member.name in seen:
            raise TrialEvidenceError("workspace archive contains duplicate path")
        if any(parent in symlinks for parent in member_path.parents):
            raise TrialEvidenceError("workspace archive writes through a symlink")
        seen[member.name] = _tar_member_kind(member, seen, symlinks, member_path)
    if len(members) != evidence.entry_count:
        raise TrialEvidenceError("workspace archive entry count mismatch")
    unpacked = sum(member.size for member in members if member.isfile())
    if unpacked != evidence.unpacked_bytes:
        raise TrialEvidenceError("workspace archive byte count mismatch")


def _tar_member_kind(
    member: tarfile.TarInfo,
    seen: dict[str, Literal["directory", "file", "symlink", "hardlink"]],
    symlinks: set[PurePosixPath],
    member_path: PurePosixPath,
) -> Literal["directory", "file", "symlink", "hardlink"]:
    if member.isdir():
        return "directory"
    if member.isfile():
        return "file"
    if member.issym():
        symlinks.add(member_path)
        return "symlink"
    if member.islnk():
        _safe_relative_path(member.linkname)
        if seen.get(member.linkname) != "file":
            raise TrialEvidenceError("workspace hardlink target is not a prior file")
        return "hardlink"
    raise TrialEvidenceError("workspace archive contains unsupported member")


def _load_workspace_index(path: Path) -> list[WorkspaceFile]:
    entries: list[WorkspaceFile] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
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


def _reference(path: Path, root: Path) -> FileReference:
    if path.is_symlink() or not path.is_file():
        raise TrialEvidenceError(f"evidence reference is not a regular file: {path}")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise TrialEvidenceError("evidence reference escapes trial root") from error
    return FileReference(path=relative, size=path.stat().st_size, sha256=_digest(path))


def _verify_reference(root: Path, reference: FileReference) -> None:
    path = root / reference.path
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise TrialEvidenceError(
            f"evidence reference is missing or unsafe: {reference.path}"
        )
    if path.stat().st_size != reference.size or _digest(path) != reference.sha256:
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
