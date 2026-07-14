from __future__ import annotations

import hashlib
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
        "private-artifacts.json",
    }
)


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


class PrivateArtifactManifest(FrozenModel):
    schema_version: Literal["harbor-hf/private-artifacts/v1"] = (
        "harbor-hf/private-artifacts/v1"
    )
    execution_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    total_bytes: int = Field(ge=0)
    entries: list[PrivateArtifactEntry]
    requirements: list[PrivateArtifactRequirement]

    @model_validator(mode="after")
    def entries_are_canonical(self) -> PrivateArtifactManifest:
        paths = [entry.path for entry in self.entries]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("private artifact entries must be sorted and unique")
        if self.total_bytes != sum(entry.size for entry in self.entries):
            raise ValueError("private artifact total does not match its entries")
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
    step_results: list[_StepProbe] = Field(default_factory=list)


def build_private_artifact_manifest(
    root: Path,
    *,
    strict_session: bool,
    max_file_bytes: int = DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_PRIVATE_BUNDLE_BYTES,
) -> PrivateArtifactManifest:
    identity = _ExecutionIdentity.model_validate_json(
        (root / "execution.lock.json").read_text(encoding="utf-8")
    )
    entries: list[PrivateArtifactEntry] = []
    total_bytes = 0
    for candidate in sorted(root.rglob("*")):
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
    session_required = _openclaw_execution_started(root)
    requirement = PrivateArtifactRequirement(
        required=session_required,
        satisfied=bool(session_paths),
        paths=session_paths,
    )
    if strict_session and session_required and not requirement.satisfied:
        raise PrivateArtifactRequirementError(
            "successful OpenClaw execution has no session JSONL"
        )
    return PrivateArtifactManifest(
        execution_id=identity.execution_id,
        trial_id=identity.trial_id,
        total_bytes=total_bytes,
        entries=entries,
        requirements=[requirement],
    )


def write_private_artifact_manifest(
    root: Path,
    *,
    strict_session: bool,
    max_file_bytes: int = DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_PRIVATE_BUNDLE_BYTES,
) -> PrivateArtifactManifest:
    manifest = build_private_artifact_manifest(
        root,
        strict_session=strict_session,
        max_file_bytes=max_file_bytes,
        max_bundle_bytes=max_bundle_bytes,
    )
    write_json(root / "private-artifacts.json", manifest.model_dump(mode="json"))
    return manifest


def _openclaw_execution_started(root: Path) -> bool:
    for result_path in sorted((root / "harbor-jobs").glob("*/*/result.json")):
        try:
            probe = _TrialProbe.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        timings = [probe.agent_execution]
        timings.extend(step.agent_execution for step in probe.step_results)
        if probe.agent_info.name == "openclaw" and any(
            timing is not None and timing.started_at is not None for timing in timings
        ):
            return True
    return False


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
