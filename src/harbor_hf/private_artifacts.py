from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harbor_hf.evidence import write_json
from harbor_hf.harbor_adapter.exporter import (
    ArtifactKind as PrivateArtifactKind,
)
from harbor_hf.harbor_adapter.exporter import classify_private_artifact
from harbor_hf.harbor_adapter.models import Sha256Digest

DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_PRIVATE_BUNDLE_BYTES = 512 * 1024 * 1024
_DERIVED_FILES = frozenset(
    {
        "_CANCELLED",
        "_FAILED",
        "_SUCCESS",
        "artifacts.tar.gz",
        "checksums.json",
        "private-artifact-rejections.json",
        "private-artifacts.json",
    }
)
_REJECTION_FILE = "private-artifact-rejections.json"
_RETENTION_PRIORITY: dict[PrivateArtifactKind, int] = {
    "other": 0,
    "execution_log": 1,
    "agent_log": 2,
    "verifier": 3,
    "trajectory": 4,
    "session": 5,
    "runtime": 6,
    "configuration": 7,
    "result": 8,
    "lock": 9,
}


class PrivateArtifactRequirementError(RuntimeError):
    """Raised when required terminal private evidence is missing."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PrivateArtifactEntry(FrozenModel):
    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    digest: Sha256Digest
    kind: PrivateArtifactKind
    classification: Literal["private"] = "private"

    @model_validator(mode="after")
    def path_is_safe(self) -> PrivateArtifactEntry:
        _safe_relative_path(self.path)
        return self


class PrivateArtifactRequirement(FrozenModel):
    name: Literal["openclaw_session_jsonl"] = "openclaw_session_jsonl"
    required: bool
    satisfied: bool
    paths: list[str]

    @model_validator(mode="after")
    def state_is_consistent(self) -> PrivateArtifactRequirement:
        for path in self.paths:
            _safe_relative_path(path)
        if self.satisfied != bool(self.paths):
            raise ValueError("artifact requirement paths disagree with satisfaction")
        return self


class PrivateArtifactRejection(FrozenModel):
    path: str = Field(min_length=1)
    reason: Literal["symlink", "file_size", "bundle_size"]
    size: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def path_is_safe(self) -> PrivateArtifactRejection:
        _safe_relative_path(self.path)
        return self


class PrivateArtifactManifest(FrozenModel):
    schema_version: Literal["harbor-hf/private-artifacts/v1"] = (
        "harbor-hf/private-artifacts/v1"
    )
    execution_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    total_bytes: int = Field(ge=0)
    entries: list[PrivateArtifactEntry]
    requirements: list[PrivateArtifactRequirement]
    rejections: list[PrivateArtifactRejection] = Field(default_factory=list)

    @model_validator(mode="after")
    def entries_are_canonical(self) -> PrivateArtifactManifest:
        paths = [entry.path for entry in self.entries]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("private artifact entries must be sorted and unique")
        if self.total_bytes != sum(entry.size for entry in self.entries):
            raise ValueError("private artifact total does not match its entries")
        rejection_paths = [rejection.path for rejection in self.rejections]
        if rejection_paths != sorted(rejection_paths) or len(rejection_paths) != len(
            set(rejection_paths)
        ):
            raise ValueError("private artifact rejections must be sorted and unique")
        return self


class _ExecutionIdentity(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    execution_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)


class _AgentInfo(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str


class _TimingProbe(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    started_at: str | None = None


class _StepProbe(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    agent_execution: _TimingProbe | None = None


class _TrialProbe(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    agent_info: _AgentInfo
    agent_execution: _TimingProbe | None = None
    step_results: list[_StepProbe] | None = None


class _VerificationProbe(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    expected_agent_name: str | None = None


class _RequestProbe(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    verification: _VerificationProbe


def build_private_artifact_manifest(
    root: Path,
    *,
    strict_session: bool,
    execution_id: str | None = None,
    trial_id: str | None = None,
    session_required: bool | None = None,
    max_file_bytes: int = DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_PRIVATE_BUNDLE_BYTES,
) -> PrivateArtifactManifest:
    identity = _artifact_identity(root, execution_id, trial_id)
    entries: list[PrivateArtifactEntry] = []
    total_bytes = 0
    candidates = sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    )
    for candidate in candidates:
        if candidate.is_symlink():
            raise RuntimeError("private artifact evidence cannot contain symlinks")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative in _DERIVED_FILES:
            continue
        size = candidate.stat().st_size
        if size > max_file_bytes:
            raise RuntimeError(f"private artifact exceeds file size limit: {relative}")
        total_bytes += size
        if total_bytes > max_bundle_bytes:
            raise RuntimeError("private artifact bundle exceeds size limit")
        entries.append(
            PrivateArtifactEntry(
                path=relative,
                size=size,
                digest=_digest(candidate),
                kind=classify_private_artifact(relative),
            )
        )

    session_paths = [
        entry.path
        for entry in entries
        if entry.kind == "session" and entry.path.endswith(".jsonl")
    ]
    required = (
        openclaw_execution_started(root)
        if session_required is None
        else session_required
    )
    requirement = PrivateArtifactRequirement(
        required=required,
        satisfied=bool(session_paths),
        paths=session_paths,
    )
    if strict_session and required and not requirement.satisfied:
        raise PrivateArtifactRequirementError(
            "successful OpenClaw execution has no session JSONL"
        )
    return PrivateArtifactManifest(
        execution_id=identity.execution_id,
        trial_id=identity.trial_id,
        total_bytes=total_bytes,
        entries=entries,
        requirements=[requirement],
        rejections=_load_rejections(root),
    )


def write_private_artifact_manifest(
    root: Path,
    *,
    strict_session: bool,
    execution_id: str | None = None,
    trial_id: str | None = None,
    session_required: bool | None = None,
    max_file_bytes: int = DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_PRIVATE_BUNDLE_BYTES,
) -> PrivateArtifactManifest:
    manifest = build_private_artifact_manifest(
        root,
        strict_session=strict_session,
        execution_id=execution_id,
        trial_id=trial_id,
        session_required=session_required,
        max_file_bytes=max_file_bytes,
        max_bundle_bytes=max_bundle_bytes,
    )
    write_json(root / "private-artifacts.json", manifest.model_dump(mode="json"))
    return manifest


def sanitize_private_artifact_tree(
    root: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_PRIVATE_BUNDLE_BYTES,
) -> list[PrivateArtifactRejection]:
    """Remove unsafe or over-limit evidence while retaining a typed rejection log."""
    rejected = _load_rejections(root)
    rejected.extend(_remove_symlinks(root))
    files, file_rejections = _remove_oversized_files(root, max_file_bytes)
    rejected.extend(file_rejections)
    rejected.extend(_trim_bundle(files, max_bundle_bytes))
    _write_rejections(root, rejected)
    return rejected


def sanitize_private_artifact_symlinks(
    root: Path,
) -> list[PrivateArtifactRejection]:
    """Remove symlinks without applying an aggregate size limit to child trials."""
    rejected = _load_rejections(root)
    rejected.extend(_remove_symlinks(root))
    _write_rejections(root, rejected)
    return rejected


def _write_rejections(root: Path, rejected: list[PrivateArtifactRejection]) -> None:
    rejected[:] = sorted(
        {rejection.path: rejection for rejection in rejected}.values(),
        key=lambda rejection: rejection.path,
    )
    if rejected:
        write_json(
            root / _REJECTION_FILE,
            {
                "schema_version": "harbor-hf/private-artifact-rejections/v1",
                "rejections": [item.model_dump(mode="json") for item in rejected],
            },
        )


def _remove_symlinks(root: Path) -> list[PrivateArtifactRejection]:
    rejected: list[PrivateArtifactRejection] = []
    candidates = sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    )
    for candidate in candidates:
        if not candidate.is_symlink():
            continue
        relative = candidate.relative_to(root).as_posix()
        candidate.unlink()
        rejected.append(PrivateArtifactRejection(path=relative, reason="symlink"))
    return rejected


def _remove_oversized_files(
    root: Path, max_file_bytes: int
) -> tuple[
    list[tuple[Path, str, int, PrivateArtifactKind]],
    list[PrivateArtifactRejection],
]:
    rejected: list[PrivateArtifactRejection] = []
    files: list[tuple[Path, str, int, PrivateArtifactKind]] = []
    for candidate in sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    ):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative in _DERIVED_FILES or relative == _REJECTION_FILE:
            continue
        size = candidate.stat().st_size
        if size > max_file_bytes:
            candidate.unlink()
            rejected.append(
                PrivateArtifactRejection(
                    path=relative,
                    reason="file_size",
                    size=size,
                )
            )
            continue
        files.append((candidate, relative, size, classify_private_artifact(relative)))
    return files, rejected


def _trim_bundle(
    files: list[tuple[Path, str, int, PrivateArtifactKind]], max_bundle_bytes: int
) -> list[PrivateArtifactRejection]:
    rejected: list[PrivateArtifactRejection] = []
    total = sum(item[2] for item in files)
    removal_order = sorted(
        files,
        key=lambda item: (
            _RETENTION_PRIORITY[item[3]],
            -item[2],
            item[1],
        ),
    )
    for candidate, relative, size, _kind in removal_order:
        if total <= max_bundle_bytes:
            break
        candidate.unlink()
        total -= size
        rejected.append(
            PrivateArtifactRejection(
                path=relative,
                reason="bundle_size",
                size=size,
            )
        )
    return rejected


def openclaw_execution_started(root: Path, *, fallback_attempted: bool = False) -> bool:
    readable_result_found = False
    for result_path in _trial_result_paths(root):
        try:
            probe = _TrialProbe.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        readable_result_found = True
        timings = [probe.agent_execution]
        timings.extend(step.agent_execution for step in probe.step_results or [])
        if probe.agent_info.name == "openclaw" and any(
            timing is not None and timing.started_at is not None for timing in timings
        ):
            return True
    if readable_result_found:
        return False
    return fallback_attempted or openclaw_execution_was_attempted(root)


def _trial_result_paths(root: Path) -> list[Path]:
    direct = root / "result.json"
    nested = sorted((root / "harbor-jobs").glob("*/*/result.json"))
    return ([direct] if direct.is_file() else []) + nested


def _artifact_identity(
    root: Path, execution_id: str | None, trial_id: str | None
) -> _ExecutionIdentity:
    if (execution_id is None) != (trial_id is None):
        raise ValueError("private artifact identity must be provided together")
    if execution_id is not None and trial_id is not None:
        return _ExecutionIdentity(execution_id=execution_id, trial_id=trial_id)
    return _ExecutionIdentity.model_validate_json(
        (root / "execution.lock.json").read_text(encoding="utf-8")
    )


def openclaw_execution_was_attempted(root: Path) -> bool:
    request_path = root / "harbor-request.json"
    try:
        request = _RequestProbe.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    if request.verification.expected_agent_name != "openclaw":
        return False
    try:
        event_lines = (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in event_lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "harbor_started":
            return True
    return False


def _load_rejections(root: Path) -> list[PrivateArtifactRejection]:
    path = root / _REJECTION_FILE
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rejections"), list):
        raise ValueError("private artifact rejection record is malformed")
    return [
        PrivateArtifactRejection.model_validate(item) for item in value["rejections"]
    ]


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError("private artifact path is not safely relative")
    return path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
