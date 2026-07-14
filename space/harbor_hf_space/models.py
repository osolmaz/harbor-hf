from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Digest = str
ResultKind = Literal["ordinary", "composite", "manual"]
ResultOutcome = Literal["complete", "partial"]
OwnerType = Literal["run", "trial", "execution"]

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_COMMIT_PATTERN = r"^[0-9a-f]{40,64}$"
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_DATASET_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"


class PublishedRow(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    schema_version: str
    publication_id: str = Field(pattern=_ID_PATTERN)
    run_id: str = Field(pattern=_ID_PATTERN)


class GlobalIndexRow(PublishedRow):
    schema_version: Literal["harbor-hf/results/index/v1"]
    campaign_id: str = Field(pattern=_ID_PATTERN)
    benchmark: str = Field(min_length=1)
    result_kind: ResultKind
    outcome: ResultOutcome
    completed_at: AwareDatetime
    model_repo: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    agent_revision: str = Field(min_length=1)
    result_dataset: str = Field(pattern=_DATASET_PATTERN)
    result_revision: str = Field(pattern=_COMMIT_PATTERN)
    source_checksum: Digest = Field(pattern=_DIGEST_PATTERN)
    control_commit: str = Field(pattern=_COMMIT_PATTERN)

    @property
    def result_label(self) -> str:
        return f"{self.outcome.upper()} · {self.result_kind.upper()}"


class TraceRow(PublishedRow):
    source_bucket: str = Field(min_length=1)
    source_prefix: str = Field(min_length=1)
    source_checksum: Digest = Field(pattern=_DIGEST_PATTERN)
    run_lock_path: str = Field(min_length=1)
    run_lock_sha256: Digest = Field(pattern=_DIGEST_PATTERN)
    control_commit: str = Field(pattern=_COMMIT_PATTERN)


class RunRow(TraceRow):
    schema_version: Literal["harbor-hf/results/runs/v1"]
    campaign_id: str = Field(pattern=_ID_PATTERN)
    experiment: str = Field(min_length=1)
    benchmark: str = Field(min_length=1)
    benchmark_revision: str = Field(min_length=1)
    result_kind: ResultKind
    outcome: ResultOutcome
    created_at: AwareDatetime
    completed_at: AwareDatetime
    model_id: str = Field(min_length=1)
    model_repo: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    deployment_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    region: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    accelerator_count: int = Field(ge=0)
    agent_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    agent_revision: str = Field(min_length=1)
    trial_count: int = Field(ge=0)
    execution_count: int = Field(ge=0)

    @model_validator(mode="after")
    def completion_follows_creation(self) -> RunRow:
        if self.completed_at < self.created_at:
            raise ValueError("run completion precedes creation")
        return self


class TrialRow(TraceRow):
    schema_version: Literal["harbor-hf/results/trials/v1"]
    trial_id: str = Field(pattern=_ID_PATTERN)
    task_name: str = Field(min_length=1)
    task_digest: Digest = Field(pattern=_DIGEST_PATTERN)
    logical_attempt: int = Field(ge=1)
    selected_execution_id: str = Field(pattern=_ID_PATTERN)
    outcome: str = Field(min_length=1)


class ExecutionRow(TraceRow):
    schema_version: Literal["harbor-hf/results/executions/v1"]
    execution_id: str = Field(pattern=_ID_PATTERN)
    trial_id: str = Field(pattern=_ID_PATTERN)
    physical_attempt: int = Field(ge=1)
    runtime_kind: Literal["endpoint", "provider"]
    status: str = Field(min_length=1)
    started_at: AwareDatetime
    completed_at: AwareDatetime
    retry_reason: str | None = None
    remote_job_id: str | None = None

    @model_validator(mode="after")
    def completion_follows_start(self) -> ExecutionRow:
        if self.completed_at < self.started_at:
            raise ValueError("execution completion precedes start")
        return self


class MetricRow(TraceRow):
    schema_version: Literal["harbor-hf/results/metrics/v1"]
    metric_id: str = Field(pattern=_ID_PATTERN)
    owner_type: OwnerType
    owner_id: str = Field(pattern=_ID_PATTERN)
    name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1)
    aggregation: str | None = None


class ArtifactRow(TraceRow):
    schema_version: Literal["harbor-hf/results/artifacts/v1"]
    artifact_id: str = Field(pattern=_ID_PATTERN)
    owner_type: OwnerType
    owner_id: str = Field(pattern=_ID_PATTERN)
    kind: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: Digest = Field(pattern=_DIGEST_PATTERN)
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)


def isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
