from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from harbor_hf.harbor_adapter.models import Sha256Digest

PUBLICATION_ENVELOPE_V2 = "harbor-hf/publication-envelope/v2"
PUBLICATION_ENVELOPE_PATH = "publication-envelope.v2.json"
PROJECTION_VERSION = "harbor-hf/results-projection/v2"
SANITIZER_VERSION = "harbor-hf/public-results/v1"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ObjectReference(FrozenModel):
    path: str = Field(min_length=1)
    digest: Sha256Digest
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def path_is_relative(self) -> ObjectReference:
        _relative_path(self.path)
        return self


class ProfileDigests(FrozenModel):
    experiment: Sha256Digest
    model: Sha256Digest
    deployment: Sha256Digest
    agent: Sha256Digest


class RuntimeIdentity(FrozenModel):
    kind: Literal["endpoint", "provider"]
    provider: str = Field(min_length=1)
    region: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    accelerator_count: int = Field(ge=0)


class HarborBundleReference(FrozenModel):
    manifest: ObjectReference
    archive: ObjectReference
    harbor_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    harbor_version: str = Field(min_length=1)
    compatibility_schema: str = Field(min_length=1)
    request_digest: Sha256Digest
    document_count: int = Field(ge=1)


class PhysicalExecutionReference(FrozenModel):
    execution_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    physical_attempt: int = Field(ge=1)
    status: Literal["succeeded", "failed_infrastructure", "cancelled"]
    started_at: AwareDatetime
    completed_at: AwareDatetime
    retry_reason: str | None = None
    remote_job_id: str | None = None
    harbor_bundle: HarborBundleReference | None = None

    @model_validator(mode="after")
    def values_are_consistent(self) -> PhysicalExecutionReference:
        if self.completed_at < self.started_at:
            raise ValueError("physical execution completion precedes start")
        if self.status == "succeeded" and self.harbor_bundle is None:
            raise ValueError("successful execution has no Harbor bundle")
        return self


class PublicationEnvelopeV2(FrozenModel):
    """HF execution metadata around referenced Harbor-native bundles."""

    schema_version: Literal["harbor-hf/publication-envelope/v2"] = (
        PUBLICATION_ENVELOPE_V2
    )
    run_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    created_at: AwareDatetime
    completed_at: AwareDatetime
    evidence_bucket: str = Field(min_length=1)
    evidence_prefix: str = Field(min_length=1)
    run_lock: ObjectReference
    profiles: ProfileDigests
    runtime: RuntimeIdentity
    sanitizer_version: Literal["harbor-hf/public-results/v1"] = SANITIZER_VERSION
    projection_version: Literal["harbor-hf/results-projection/v2"] = PROJECTION_VERSION
    cleanup_outcome: Literal["verified", "not_applicable"]
    executions: list[PhysicalExecutionReference]

    @model_validator(mode="after")
    def values_are_consistent(self) -> PublicationEnvelopeV2:
        if self.completed_at < self.created_at:
            raise ValueError("publication completion precedes creation")
        ids = [execution.execution_id for execution in self.executions]
        if len(ids) != len(set(ids)):
            raise ValueError("publication envelope has duplicate executions")
        if not any(execution.status == "succeeded" for execution in self.executions):
            raise ValueError("publication envelope has no successful execution")
        _relative_path(self.evidence_prefix)
        return self


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def object_reference(path: str, content: bytes) -> ObjectReference:
    return ObjectReference(
        path=path,
        digest="sha256:" + hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def profile_digest(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return canonical_digest(value)


def publication_envelope_schema() -> dict[str, object]:
    return PublicationEnvelopeV2.model_json_schema()


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value != path.as_posix()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("publication path must be canonical and relative")
    return path
