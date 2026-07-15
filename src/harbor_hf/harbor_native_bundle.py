from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harbor_hf.harbor_adapter.models import (
    HarborCompatibilityBundle,
    Sha256Digest,
    canonical_json_bytes,
    sha256_digest,
)

HARBOR_NATIVE_BUNDLE_V1ALPHA1 = "harbor-hf/harbor-native-bundle/v1alpha1"
HARBOR_NATIVE_BUNDLE_PATH = "harbor-native-bundle.json"


class NativeBundleError(RuntimeError):
    """Raised when retained Harbor output cannot form a verified native bundle."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BundleObject(FrozenModel):
    path: str = Field(min_length=1)
    digest: Sha256Digest
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1)

    @model_validator(mode="after")
    def path_is_relative(self) -> BundleObject:
        _relative_path(self.path)
        return self


class NativeDocument(FrozenModel):
    kind: Literal["job_lock", "job_result", "trial_lock", "trial_result"]
    path: str = Field(min_length=1)
    digest: Sha256Digest

    @model_validator(mode="after")
    def path_is_relative(self) -> NativeDocument:
        _relative_path(self.path)
        return self


class HarborNativeBundle(FrozenModel):
    """Packaging metadata around Harbor-owned serialized result documents."""

    schema_version: Literal["harbor-hf/harbor-native-bundle/v1alpha1"] = (
        HARBOR_NATIVE_BUNDLE_V1ALPHA1
    )
    contract_status: Literal["compatibility"] = "compatibility"
    harbor_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    harbor_version: str = Field(min_length=1)
    request_digest: Sha256Digest
    compatibility_schema: str = Field(min_length=1)
    archive: BundleObject
    compatibility: BundleObject
    documents: list[NativeDocument]

    @model_validator(mode="after")
    def documents_are_unique(self) -> HarborNativeBundle:
        keys = [(document.kind, document.path) for document in self.documents]
        if len(keys) != len(set(keys)):
            raise ValueError("native bundle contains duplicate documents")
        if not self.documents:
            raise ValueError("native bundle contains no Harbor documents")
        return self


def build_harbor_native_bundle(root: Path) -> HarborNativeBundle:
    compatibility_path = _regular_file(root, "harbor-compatibility.json")
    archive_path = _regular_file(root, "artifacts.tar.gz")
    jobs_root = root / "harbor-jobs"
    if jobs_root.is_symlink() or not jobs_root.is_dir():
        raise NativeBundleError("retained Harbor jobs root is not a directory")
    try:
        compatibility = HarborCompatibilityBundle.model_validate_json(
            compatibility_path.read_bytes()
        )
    except Exception as error:
        raise NativeBundleError("Harbor compatibility export is invalid") from error

    documents: list[NativeDocument] = []
    for job in compatibility.jobs:
        documents.extend(
            _document_pair(
                root,
                job.path,
                lock_kind="job_lock",
                result_kind="job_result",
                lock_digest=job.lock_digest,
                result_digest=job.result_digest,
            )
        )
    for trial in compatibility.trials:
        documents.extend(
            _document_pair(
                root,
                trial.path,
                lock_kind="trial_lock",
                result_kind="trial_result",
                lock_digest=trial.lock_digest,
                result_digest=trial.result_digest,
            )
        )
    try:
        return HarborNativeBundle(
            harbor_revision=compatibility.harbor_revision,
            harbor_version=compatibility.harbor_version,
            request_digest=compatibility.request_digest,
            compatibility_schema=compatibility.schema_version,
            archive=_object(
                root,
                archive_path,
                media_type="application/gzip",
            ),
            compatibility=_object(
                root,
                compatibility_path,
                media_type="application/json",
            ),
            documents=sorted(documents, key=lambda item: (item.path, item.kind)),
        )
    except ValueError as error:
        raise NativeBundleError("Harbor native bundle is invalid") from error


def write_harbor_native_bundle(
    root: Path, *, required: bool
) -> HarborNativeBundle | None:
    try:
        bundle = build_harbor_native_bundle(root)
    except NativeBundleError:
        if required:
            raise
        return None
    (root / HARBOR_NATIVE_BUNDLE_PATH).write_bytes(
        canonical_json_bytes(bundle.model_dump(mode="json")) + b"\n"
    )
    return bundle


def load_harbor_native_bundle(content: bytes) -> HarborNativeBundle:
    try:
        return HarborNativeBundle.model_validate_json(content)
    except Exception as error:
        raise NativeBundleError("Harbor native bundle manifest is invalid") from error


def harbor_native_bundle_schema() -> dict[str, object]:
    return HarborNativeBundle.model_json_schema()


def _document_pair(
    root: Path,
    relative_directory: str,
    *,
    lock_kind: Literal["job_lock", "trial_lock"],
    result_kind: Literal["job_result", "trial_result"],
    lock_digest: str,
    result_digest: str,
) -> list[NativeDocument]:
    directory = _relative_path(relative_directory)
    return [
        _document(
            root,
            PurePosixPath("harbor-jobs", directory, "lock.json"),
            lock_kind,
            lock_digest,
        ),
        _document(
            root,
            PurePosixPath("harbor-jobs", directory, "result.json"),
            result_kind,
            result_digest,
        ),
    ]


def _document(
    root: Path,
    relative: PurePosixPath,
    kind: Literal["job_lock", "job_result", "trial_lock", "trial_result"],
    expected_digest: str,
) -> NativeDocument:
    path = _regular_file(root, relative.as_posix())
    observed = sha256_digest(path.read_bytes())
    if observed != expected_digest:
        raise NativeBundleError(f"native Harbor document digest differs: {relative}")
    return NativeDocument(kind=kind, path=relative.as_posix(), digest=observed)


def _object(root: Path, path: Path, *, media_type: str) -> BundleObject:
    return BundleObject(
        path=path.relative_to(root).as_posix(),
        digest=sha256_digest(path.read_bytes()),
        size_bytes=path.stat().st_size,
        media_type=media_type,
    )


def _regular_file(root: Path, relative: str) -> Path:
    parts = _relative_path(relative).parts
    candidate = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise NativeBundleError(f"native bundle path is a symlink: {relative}")
    if not candidate.is_file():
        raise NativeBundleError(f"native bundle path is not a file: {relative}")
    return candidate


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value != path.as_posix()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("native bundle path must be canonical and relative")
    return path
