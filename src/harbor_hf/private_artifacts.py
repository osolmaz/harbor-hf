from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harbor_hf.harbor_adapter.exporter import (
    ArtifactKind as PrivateArtifactKind,
)
from harbor_hf.harbor_adapter.exporter import classify_private_artifact
from harbor_hf.harbor_adapter.models import Sha256Digest

DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_PRIVATE_BUNDLE_BYTES = 40 * 1024 * 1024 * 1024
MAX_WORKSPACE_ARCHIVE_BYTES = 32 * 1024 * 1024 * 1024
DEFAULT_MAX_PRIVATE_ARTIFACT_FILES = 100_000
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
    name: Literal["openclaw_session_jsonl", "trial_evidence_complete"]
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
    reason: Literal[
        "symlink",
        "special_file",
        "file_size",
        "bundle_size",
        "reserved_path",
    ]
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
    trust_rejections: bool = False,
    max_file_bytes: int = DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_PRIVATE_BUNDLE_BYTES,
    max_file_count: int = DEFAULT_MAX_PRIVATE_ARTIFACT_FILES,
) -> PrivateArtifactManifest:
    identity = _artifact_identity(root, execution_id, trial_id)
    entries, total_bytes = _private_artifact_entries(
        root,
        trust_rejections=trust_rejections,
        max_file_bytes=max_file_bytes,
        max_bundle_bytes=max_bundle_bytes,
        max_file_count=max_file_count,
    )

    session_paths = [
        entry.path
        for entry in entries
        if entry.kind == "session"
        and entry.path.endswith(".jsonl")
        and not entry.path.endswith(".trajectory.jsonl")
        and _valid_jsonl_objects(root / entry.path)
    ]
    required = (
        openclaw_execution_started(root)
        if session_required is None
        else session_required
    )
    session_requirement = PrivateArtifactRequirement(
        name="openclaw_session_jsonl",
        required=required,
        satisfied=bool(session_paths),
        paths=session_paths,
    )
    evidence_paths = _valid_trial_evidence_manifests(root)
    evidence_required = strict_session and _trial_evidence_was_requested(root)
    evidence_requirement = PrivateArtifactRequirement(
        name="trial_evidence_complete",
        required=evidence_required,
        satisfied=bool(evidence_paths),
        paths=evidence_paths,
    )
    if strict_session and required and not session_requirement.satisfied:
        raise PrivateArtifactRequirementError(
            "successful OpenClaw execution has no session JSONL"
        )
    if evidence_required and not evidence_requirement.satisfied:
        raise PrivateArtifactRequirementError(
            "successful execution has no complete trial evidence bundle"
        )
    return PrivateArtifactManifest(
        execution_id=identity.execution_id,
        trial_id=identity.trial_id,
        total_bytes=total_bytes,
        entries=entries,
        requirements=[session_requirement]
        + ([evidence_requirement] if evidence_required else []),
        rejections=_manifest_rejections(root, trust_rejections=trust_rejections),
    )


def _trial_evidence_was_requested(root: Path) -> bool:
    candidates = [root / "harbor-job.json"]
    candidates.extend(parent / "harbor-job.json" for parent in list(root.parents)[:4])
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            continue
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        if isinstance(artifacts, list) and any(
            isinstance(item, dict)
            and item.get("source") == "/app"
            and item.get("destination") == "workspace/app"
            for item in artifacts
        ):
            return True
    return False


def _valid_trial_evidence_manifests(root: Path) -> list[str]:
    from harbor_hf.trial_evidence import TrialEvidenceError, verify_trial_evidence

    manifests: list[str] = []
    candidates = list(root.glob("harbor-jobs/*/*/evidence/manifest.json"))
    local_manifest = root / "evidence" / "manifest.json"
    if local_manifest.is_file():
        candidates.append(local_manifest)
    for path in sorted(candidates):
        trial_root = path.parent.parent
        try:
            verify_trial_evidence(trial_root, deep=False)
        except TrialEvidenceError:
            continue
        manifests.append(path.relative_to(root).as_posix())
    return manifests


def _private_artifact_entries(
    root: Path,
    *,
    trust_rejections: bool,
    max_file_bytes: int,
    max_bundle_bytes: int,
    max_file_count: int,
) -> tuple[list[PrivateArtifactEntry], int]:
    entries: list[PrivateArtifactEntry] = []
    total_bytes = 0
    candidates = sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    )
    for candidate in candidates:
        entry = _private_artifact_entry(
            candidate,
            root,
            trust_rejections=trust_rejections,
            max_file_bytes=max_file_bytes,
        )
        if entry is None:
            continue
        total_bytes += entry.size
        if total_bytes > max_bundle_bytes:
            raise RuntimeError("private artifact bundle exceeds size limit")
        entries.append(entry)
        if len(entries) > max_file_count:
            raise RuntimeError("private artifact bundle exceeds file count limit")
    return entries, total_bytes


def _private_artifact_entry(
    candidate: Path,
    root: Path,
    *,
    trust_rejections: bool,
    max_file_bytes: int,
) -> PrivateArtifactEntry | None:
    if candidate.is_symlink():
        raise RuntimeError("private artifact evidence cannot contain symlinks")
    relative = candidate.relative_to(root).as_posix()
    if relative in _DERIVED_FILES:
        _validate_reserved_path(candidate, relative, trust_rejections)
        return None
    if candidate.is_dir():
        return None
    if not candidate.is_file():
        raise RuntimeError(f"private artifact has unsupported file type: {relative}")
    size = candidate.stat().st_size
    effective_limit = _private_artifact_file_limit(relative, max_file_bytes)
    if size > effective_limit:
        raise RuntimeError(f"private artifact exceeds file size limit: {relative}")
    return PrivateArtifactEntry(
        path=relative,
        size=size,
        digest=_digest(candidate),
        kind=classify_private_artifact(relative),
    )


def write_private_artifact_manifest(
    root: Path,
    *,
    strict_session: bool,
    execution_id: str | None = None,
    trial_id: str | None = None,
    session_required: bool | None = None,
    trust_rejections: bool = False,
    max_file_bytes: int = DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_PRIVATE_BUNDLE_BYTES,
    max_file_count: int = DEFAULT_MAX_PRIVATE_ARTIFACT_FILES,
) -> PrivateArtifactManifest:
    manifest = build_private_artifact_manifest(
        root,
        strict_session=strict_session,
        execution_id=execution_id,
        trial_id=trial_id,
        session_required=session_required,
        trust_rejections=trust_rejections,
        max_file_bytes=max_file_bytes,
        max_bundle_bytes=max_bundle_bytes,
        max_file_count=max_file_count,
    )
    payload = (
        json.dumps(
            manifest.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    ).encode()
    if len(payload) > max_file_bytes:
        raise RuntimeError("private artifact manifest exceeds file size limit")
    (root / "private-artifacts.json").write_bytes(payload)
    return manifest


def sanitize_private_artifact_tree(
    root: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_PRIVATE_BUNDLE_BYTES,
    trust_existing_rejections: bool = False,
    required_directories: tuple[str, ...] = (),
    max_file_count: int = DEFAULT_MAX_PRIVATE_ARTIFACT_FILES,
) -> list[PrivateArtifactRejection]:
    """Remove unsafe or over-limit evidence while retaining a typed rejection log."""
    rejected = _remove_symlinks(root)
    rejected.extend(_remove_required_directory_collisions(root, required_directories))
    rejected.extend(_remove_special_files(root))
    rejected.extend(_remove_reserved_paths(root))
    existing_rejections, omitted_count = _consume_existing_rejections(
        root, trust_existing=trust_existing_rejections
    )
    rejected.extend(existing_rejections)
    files, file_rejections = _remove_oversized_files(root, max_file_bytes)
    rejected.extend(file_rejections)
    return _finalize_bounded_rejections(
        root,
        files,
        rejected,
        omitted_count=omitted_count,
        max_file_bytes=max_file_bytes,
        max_bundle_bytes=max_bundle_bytes,
        max_file_count=max_file_count,
    )


def sanitize_private_artifact_symlinks(
    root: Path, *, max_depth: int | None = None
) -> list[PrivateArtifactRejection]:
    """Remove symlinks without applying an aggregate size limit to child trials."""
    rejected = _remove_symlinks(root, max_depth=max_depth)
    existing_rejections, omitted_count = _consume_existing_rejections(
        root, trust_existing=False
    )
    rejected.extend(existing_rejections)
    return _write_rejections(
        root,
        rejected,
        base_omitted_count=omitted_count,
    )


def sanitize_private_artifact_special_files(
    root: Path,
) -> list[PrivateArtifactRejection]:
    """Remove non-file artifact nodes and retain a typed rejection record."""
    rejected = _remove_special_files(root)
    existing_rejections, omitted_count = _consume_existing_rejections(
        root, trust_existing=False
    )
    rejected.extend(existing_rejections)
    return _write_rejections(
        root,
        rejected,
        base_omitted_count=omitted_count,
    )


def validate_private_artifact_directory_files(
    root: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_PRIVATE_BUNDLE_BYTES,
    max_file_count: int = DEFAULT_MAX_PRIVATE_ARTIFACT_FILES,
    allowed_directories: tuple[str, ...] | None = None,
) -> None:
    """Validate only files directly inside a directory, excluding child bundles."""
    allowed = (
        None
        if allowed_directories is None
        else {_direct_directory_name(name) for name in allowed_directories}
    )
    total_bytes = 0
    file_count = 0
    for candidate in sorted(root.iterdir()):
        if candidate.is_symlink():
            raise RuntimeError("private artifact evidence cannot contain symlinks")
        if candidate.is_dir():
            _validate_direct_directory(candidate, allowed)
            continue
        if not candidate.is_file():
            raise RuntimeError(
                f"private artifact has unsupported file type: {candidate.name}"
            )
        size = candidate.stat().st_size
        file_count += 1
        if size > max_file_bytes:
            raise RuntimeError(
                f"private artifact exceeds file size limit: {candidate.name}"
            )
        total_bytes += size
        if total_bytes > max_bundle_bytes:
            raise RuntimeError("private artifact bundle exceeds size limit")
        if file_count > max_file_count:
            raise RuntimeError("private artifact bundle exceeds file count limit")


def _validate_direct_directory(candidate: Path, allowed: set[str] | None) -> None:
    if allowed is not None and candidate.name not in allowed:
        raise RuntimeError(
            f"private artifact has unexpected directory: {candidate.name}"
        )


def sanitize_private_artifact_directory_files(
    root: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES,
    max_bundle_bytes: int = DEFAULT_MAX_PRIVATE_BUNDLE_BYTES,
    trust_existing_rejections: bool = False,
    required_directories: tuple[str, ...] = (),
    preserved_files: tuple[str, ...] = (),
    max_file_count: int = DEFAULT_MAX_PRIVATE_ARTIFACT_FILES,
    allowed_directories: tuple[str, ...] | None = None,
) -> list[PrivateArtifactRejection]:
    """Bound direct files without charging independently bounded child bundles."""
    preserved = {_direct_file_name(relative) for relative in preserved_files}
    rejected = _remove_required_directory_collisions(root, required_directories)
    rejected.extend(_remove_reserved_paths(root, preserved=preserved))
    rejected.extend(_remove_unexpected_directories(root, allowed_directories))
    existing_rejections, omitted_count = _consume_existing_rejections(
        root, trust_existing=trust_existing_rejections
    )
    rejected.extend(existing_rejections)
    files: list[tuple[Path, str, int, PrivateArtifactKind]] = []
    for candidate in sorted(root.iterdir()):
        if candidate.is_symlink() or candidate.is_dir():
            continue
        relative = candidate.name
        if relative == _REJECTION_FILE or relative in preserved:
            continue
        if not candidate.is_file():
            candidate.unlink()
            rejected.append(
                PrivateArtifactRejection(
                    path=relative,
                    reason="special_file",
                )
            )
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
    return _finalize_bounded_rejections(
        root,
        files,
        rejected,
        omitted_count=omitted_count,
        max_file_bytes=max_file_bytes,
        max_bundle_bytes=max_bundle_bytes,
        max_file_count=max_file_count,
    )


def _finalize_bounded_rejections(
    root: Path,
    files: list[tuple[Path, str, int, PrivateArtifactKind]],
    rejected: list[PrivateArtifactRejection],
    *,
    omitted_count: int,
    max_file_bytes: int,
    max_bundle_bytes: int,
    max_file_count: int,
) -> list[PrivateArtifactRejection]:
    record_limit = min(max_file_bytes, max_bundle_bytes)
    while True:
        retained = _write_rejections(
            root,
            rejected,
            max_record_bytes=record_limit,
            base_omitted_count=omitted_count,
        )
        record = root / _REJECTION_FILE
        record_bytes = record.stat().st_size if record.is_file() else 0
        bundle_rejections = _trim_bundle(
            files,
            max(0, max_bundle_bytes - record_bytes),
            max_file_count,
        )
        if not bundle_rejections:
            return retained
        rejected.extend(bundle_rejections)
        files = [item for item in files if item[0].is_file()]


def _write_rejections(
    root: Path,
    rejected: list[PrivateArtifactRejection],
    *,
    max_record_bytes: int = DEFAULT_MAX_PRIVATE_ARTIFACT_BYTES,
    base_omitted_count: int = 0,
) -> list[PrivateArtifactRejection]:
    canonical = sorted(
        {rejection.path: rejection for rejection in rejected}.values(),
        key=lambda rejection: rejection.path,
    )
    path = root / _REJECTION_FILE
    if not canonical and base_omitted_count == 0:
        path.unlink(missing_ok=True)
        return []
    low = 0
    high = len(canonical)
    selected = b""
    selected_count = 0
    while low <= high:
        count = (low + high) // 2
        payload = _rejection_payload(
            canonical,
            count,
            base_omitted_count=base_omitted_count,
        )
        if len(payload) <= max_record_bytes:
            selected = payload
            selected_count = count
            low = count + 1
        else:
            high = count - 1
    if not selected:
        path.unlink(missing_ok=True)
        return []
    path.write_bytes(selected)
    return canonical[:selected_count]


def _rejection_payload(
    rejected: list[PrivateArtifactRejection],
    count: int,
    *,
    base_omitted_count: int,
) -> bytes:
    record = {
        "schema_version": "harbor-hf/private-artifact-rejections/v1",
        "rejections": [item.model_dump(mode="json") for item in rejected[:count]],
        "omitted_count": base_omitted_count + len(rejected) - count,
    }
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _remove_symlinks(
    root: Path, *, max_depth: int | None = None
) -> list[PrivateArtifactRejection]:
    rejected: list[PrivateArtifactRejection] = []
    candidates = sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    )
    for candidate in candidates:
        if max_depth is not None and len(candidate.relative_to(root).parts) > max_depth:
            continue
        if not candidate.is_symlink():
            continue
        relative = candidate.relative_to(root).as_posix()
        candidate.unlink()
        rejected.append(PrivateArtifactRejection(path=relative, reason="symlink"))
    return rejected


def _remove_required_directory_collisions(
    root: Path, required_directories: tuple[str, ...]
) -> list[PrivateArtifactRejection]:
    rejected: list[PrivateArtifactRejection] = []
    for relative in sorted(set(required_directories)):
        path = _direct_child(root, relative)
        if path.is_dir() and not path.is_symlink():
            continue
        if not path.exists() and not path.is_symlink():
            continue
        is_symlink = path.is_symlink()
        size = path.stat().st_size if path.is_file() and not is_symlink else None
        path.unlink()
        rejected.append(
            PrivateArtifactRejection(
                path=relative,
                reason="symlink" if is_symlink else "reserved_path",
                size=size,
            )
        )
    return rejected


def _remove_special_files(root: Path) -> list[PrivateArtifactRejection]:
    rejected: list[PrivateArtifactRejection] = []
    for candidate in sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    ):
        if candidate.is_symlink() or candidate.is_dir() or candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        candidate.unlink()
        rejected.append(PrivateArtifactRejection(path=relative, reason="special_file"))
    return rejected


def _direct_child(root: Path, relative: str) -> Path:
    path = _safe_relative_path(relative)
    if len(path.parts) != 1 or relative in _DERIVED_FILES:
        raise ValueError("required artifact directory must be a direct child")
    return root / relative


def _direct_file_name(relative: str) -> str:
    path = _safe_relative_path(relative)
    if len(path.parts) != 1 or relative == _REJECTION_FILE:
        raise ValueError("preserved artifact file must be a direct child")
    return relative


def _direct_directory_name(relative: str) -> str:
    path = _safe_relative_path(relative)
    if len(path.parts) != 1 or relative in _DERIVED_FILES:
        raise ValueError("allowed artifact directory must be a direct child")
    return relative


def _consume_existing_rejections(
    root: Path, *, trust_existing: bool
) -> tuple[list[PrivateArtifactRejection], int]:
    path = root / _REJECTION_FILE
    if not path.exists():
        return [], 0
    if trust_existing:
        return _load_rejection_record(root)
    size = path.stat().st_size
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return (
        [
            PrivateArtifactRejection(
                path=_REJECTION_FILE,
                reason="reserved_path",
                size=size,
            )
        ],
        0,
    )


def _validate_reserved_path(path: Path, relative: str, trust_rejections: bool) -> None:
    if relative == _REJECTION_FILE and trust_rejections and path.is_file():
        return
    raise RuntimeError(
        f"private artifact contains controller-reserved path: {relative}"
    )


def _remove_reserved_paths(
    root: Path, *, preserved: set[str] | frozenset[str] = frozenset()
) -> list[PrivateArtifactRejection]:
    rejected: list[PrivateArtifactRejection] = []
    for name in sorted(_DERIVED_FILES - {_REJECTION_FILE} - preserved):
        path = root / name
        if not path.exists() and not path.is_symlink():
            continue
        is_symlink = path.is_symlink()
        size = path.stat().st_size if path.is_file() and not is_symlink else None
        if path.is_dir() and not is_symlink:
            shutil.rmtree(path)
        else:
            path.unlink()
        rejected.append(
            PrivateArtifactRejection(
                path=name,
                reason="symlink" if is_symlink else "reserved_path",
                size=size,
            )
        )
    return rejected


def _remove_unexpected_directories(
    root: Path, allowed_directories: tuple[str, ...] | None
) -> list[PrivateArtifactRejection]:
    if allowed_directories is None:
        return []
    allowed = {_direct_directory_name(name) for name in allowed_directories}
    rejected: list[PrivateArtifactRejection] = []
    for candidate in sorted(root.iterdir()):
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or candidate.name in allowed
        ):
            continue
        shutil.rmtree(candidate)
        rejected.append(
            PrivateArtifactRejection(
                path=candidate.name,
                reason="reserved_path",
            )
        )
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
        if size > _private_artifact_file_limit(relative, max_file_bytes):
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


def _private_artifact_file_limit(relative: str, default: int) -> int:
    workspace_archive = "evidence/workspace.tar.zst"
    if relative == workspace_archive or relative.endswith(f"/{workspace_archive}"):
        return MAX_WORKSPACE_ARCHIVE_BYTES
    return default


def _trim_bundle(
    files: list[tuple[Path, str, int, PrivateArtifactKind]],
    max_bundle_bytes: int,
    max_file_count: int,
) -> list[PrivateArtifactRejection]:
    rejected: list[PrivateArtifactRejection] = []
    total = sum(item[2] for item in files)
    retained_count = len(files)
    removal_order = sorted(
        files,
        key=lambda item: (
            _RETENTION_PRIORITY[item[3]],
            -item[2],
            item[1],
        ),
    )
    for candidate, relative, size, _kind in removal_order:
        if total <= max_bundle_bytes and retained_count <= max_file_count:
            break
        candidate.unlink()
        total -= size
        retained_count -= 1
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
    rejections, _omitted_count = _load_rejection_record(root)
    return rejections


def _load_rejection_record(
    root: Path,
) -> tuple[list[PrivateArtifactRejection], int]:
    path = root / _REJECTION_FILE
    if not path.is_file():
        return [], 0
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rejections"), list):
        raise ValueError("private artifact rejection record is malformed")
    omitted_count = value.get("omitted_count", 0)
    if type(omitted_count) is not int or omitted_count < 0:
        raise ValueError("private artifact rejection record is malformed")
    return (
        [PrivateArtifactRejection.model_validate(item) for item in value["rejections"]],
        omitted_count,
    )


def _manifest_rejections(
    root: Path, *, trust_rejections: bool
) -> list[PrivateArtifactRejection]:
    if not (root / _REJECTION_FILE).exists():
        return []
    if not trust_rejections:
        raise RuntimeError("private artifact rejection path is controller-reserved")
    return _load_rejections(root)


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


def _valid_jsonl_objects(path: Path) -> bool:
    found = False
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                if not isinstance(json.loads(line), dict):
                    return False
                found = True
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        return False
    return found


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
