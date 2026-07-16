from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from harbor_hf.control import RetryCategory
from harbor_hf.harbor_adapter.models import Sha256Digest

PUBLICATION_ENVELOPE_V1 = "harbor-hf/publication-envelope/v1"
PUBLICATION_ENVELOPE_PATH = "publication-envelope.v1.json"
PROJECTION_VERSION = "harbor-hf/results-projection/v1"
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


class SourceTrialSelection(FrozenModel):
    task_name: str = Field(min_length=1)
    logical_attempt: int = Field(ge=1)


class SourcePublicationReference(FrozenModel):
    role: Literal["base", "correction"]
    publication_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    result_dataset: str = Field(min_length=1)
    result_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    source_checksum: Sha256Digest
    selected_trials: list[SourceTrialSelection] = Field(min_length=1)

    @model_validator(mode="after")
    def trials_are_unique(self) -> SourcePublicationReference:
        identities = [
            (trial.task_name, trial.logical_attempt) for trial in self.selected_trials
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("source publication has duplicate selected trials")
        return self


class PhysicalExecutionReference(FrozenModel):
    execution_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    physical_attempt: int = Field(ge=1)
    status: Literal["succeeded", "failed", "cancelled"]
    failure_category: RetryCategory | None
    started_at: AwareDatetime
    completed_at: AwareDatetime
    retry_reason: str | None = None
    remote_job_id: str | None = None
    bundle_status: Literal["verified", "not_available", "source_publication"]
    harbor_bundle: HarborBundleReference | None = None

    @model_validator(mode="after")
    def values_are_consistent(self) -> PhysicalExecutionReference:
        if self.completed_at < self.started_at:
            raise ValueError("physical execution completion precedes start")
        if self.bundle_status == "verified" and self.harbor_bundle is None:
            raise ValueError("verified Harbor bundle is missing")
        if self.bundle_status != "verified" and self.harbor_bundle is not None:
            raise ValueError("unavailable Harbor bundle is present")
        if self.status == "succeeded" and self.bundle_status not in {
            "verified",
            "source_publication",
        }:
            raise ValueError("successful execution requires a verified Harbor bundle")
        if (self.status == "failed") != (self.failure_category is not None):
            raise ValueError(
                "physical execution failure category conflicts with status"
            )
        return self


class PublicationEnvelope(FrozenModel):
    """HF execution metadata around referenced Harbor-native bundles."""

    schema_version: Literal["harbor-hf/publication-envelope/v1"] = (
        PUBLICATION_ENVELOPE_V1
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
    projection_version: Literal["harbor-hf/results-projection/v1"] = PROJECTION_VERSION
    cleanup_outcome: Literal["verified", "not_applicable"]
    executions: list[PhysicalExecutionReference]
    sources: list[SourcePublicationReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def values_are_consistent(self) -> PublicationEnvelope:
        if self.completed_at < self.created_at:
            raise ValueError("publication completion precedes creation")
        ids = [execution.execution_id for execution in self.executions]
        if len(ids) != len(set(ids)):
            raise ValueError("publication envelope has duplicate executions")
        if not any(execution.status == "succeeded" for execution in self.executions):
            raise ValueError("publication envelope has no successful execution")
        source_ids = [source.publication_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("publication envelope has duplicate source publications")
        sourced = any(
            execution.bundle_status == "source_publication"
            for execution in self.executions
        )
        if sourced != bool(self.sources):
            raise ValueError(
                "source-backed executions conflict with publication sources"
            )
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


def execution_profile_digest(
    *, model: object, deployment: object, agent: object
) -> str:
    return canonical_digest(
        {
            "model": _profile_value(model, exclude={"id"}),
            "deployment": _deployment_profile_value(deployment),
            "agent": _profile_value(agent, exclude={"id"}),
        }
    )


def _deployment_profile_value(value: object) -> object:
    profile = _profile_value(value, exclude={"id"})
    if not isinstance(profile, dict):
        raise TypeError("deployment profile must serialize to a mapping")
    return {
        key: (
            {
                endpoint_key: endpoint_item
                for endpoint_key, endpoint_item in item.items()
                if endpoint_key == "served_model_name"
            }
            if key == "endpoint" and isinstance(item, dict)
            else item
        )
        for key, item in profile.items()
    }


def _profile_value(value: object, *, exclude: set[str]) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude=exclude, exclude_none=True)
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if key not in exclude}
    raise TypeError("execution profile must be a model or mapping")


def publication_envelope_schema() -> dict[str, object]:
    return PublicationEnvelope.model_json_schema()


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
