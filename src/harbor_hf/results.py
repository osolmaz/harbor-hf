from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Annotated, Literal, Protocol, TypedDict

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)

from harbor_hf.control import RetryCategory
from harbor_hf.models import (
    AgentProfile,
    ComponentKind,
    DeploymentTarget,
    EvaluationId,
    ModelProfile,
    PublicationRole,
)
from harbor_hf.publication_envelope import (
    PUBLICATION_ENVELOPE_PATH,
    PhysicalExecutionReference,
    ProfileDigests,
    PublicationEnvelope,
    RuntimeIdentity,
    SourcePublicationReference,
    canonical_digest,
    canonical_json_bytes,
    execution_profile_digest,
    object_reference,
)

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
RESULT_PUBLICATION_CONTRACT = "harbor-hf/result-publication/v1"
EntityId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")]
DatasetId = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"),
]
OwnerType = Literal["run", "trial", "execution"]
RuntimeKind = Literal["endpoint", "provider"]
RunQuality = Literal["clean", "degraded"]
ResultKind = Literal["ordinary", "composed"]
CatalogScope = Literal["primary", "audit"]
CatalogDecisionAction = Literal["promote", "withdraw"]
TaskOutcome = Literal[
    "scored",
    "agent_failed",
    "benchmark_failed",
    "infrastructure_exhausted",
    "unsupported",
]
ArtifactKind = Literal[
    "run_lock",
    "verification",
    "runtime_environment",
    "endpoint_snapshot",
    "composition",
]
TableName = Literal["runs", "trials", "executions", "metrics", "artifacts"]

_SUMMARY_PATH = "run-summary.json"
_CHECKSUMS_PATH = "checksums.json"
_TERMINAL_MARKERS = frozenset({"_SUCCESS", "_PARTIAL", "_FAILED", "_CANCELLED"})
_FORBIDDEN_ARTIFACT_PARTS = frozenset(
    {
        "artifacts.tar.gz",
        "harbor-jobs",
        "harbor.log",
        "manifest.yaml",
        "session",
        "sessions",
        "task-source",
        "trajectory",
        "trajectories",
    }
)
_ARTIFACT_PATHS: Mapping[ArtifactKind, str] = {
    "run_lock": "run.lock.json",
    "verification": "verification.json",
    "runtime_environment": "runtime-environment.json",
    "endpoint_snapshot": "endpoint.snapshot.json",
    "composition": "composition.json",
}
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_CHECKSUM_MAP = TypeAdapter(dict[str, Digest])
_TASK_OUTCOMES: tuple[TaskOutcome, ...] = (
    "scored",
    "agent_failed",
    "benchmark_failed",
    "infrastructure_exhausted",
    "unsupported",
)


class ResultPublicationError(RuntimeError):
    """Raised when evidence cannot be safely normalized or published."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CatalogDecision(FrozenModel):
    schema_version: Literal["harbor-hf/catalog-decision/v1"] = (
        "harbor-hf/catalog-decision/v1"
    )
    decision_id: EntityId
    publication_id: EntityId
    action: CatalogDecisionAction
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    created_at: AwareDatetime


class EvidenceSource(FrozenModel):
    bucket: str = Field(min_length=1)
    prefix: str = Field(min_length=1)
    run_lock_path: str = "run.lock.json"
    summary_path: Literal["run-summary.json"] = _SUMMARY_PATH

    @model_validator(mode="after")
    def paths_are_canonical(self) -> EvidenceSource:
        _validate_relative_path(self.prefix)
        _validate_relative_path(self.run_lock_path)
        return self


class RunEvidence(FrozenModel):
    run_id: EntityId
    campaign_id: EntityId
    experiment: str = Field(min_length=1)
    evaluation_id: EvaluationId
    publication_role: PublicationRole
    component_kind: ComponentKind | None
    benchmark: str = Field(min_length=1)
    benchmark_revision: str = Field(min_length=1)
    result_kind: ResultKind = "ordinary"
    outcome: Literal["complete"] = "complete"
    quality: RunQuality
    created_at: AwareDatetime
    completed_at: AwareDatetime
    model_id: EntityId
    model_repo: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    deployment_id: EntityId
    provider: str = Field(min_length=1)
    region: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    accelerator_count: int = Field(ge=0)
    agent_id: EntityId
    agent_name: str = Field(min_length=1)
    agent_revision: str = Field(min_length=1)

    @model_validator(mode="after")
    def completion_follows_creation(self) -> RunEvidence:
        if self.completed_at < self.created_at:
            raise ValueError("run completion precedes creation")
        if (self.publication_role == "component") != (self.component_kind is not None):
            raise ValueError("run component kind conflicts with publication role")
        return self


class TrialEvidence(FrozenModel):
    trial_id: EntityId
    task_name: str = Field(min_length=1)
    task_digest: Digest
    logical_attempt: int = Field(ge=1)
    selected_execution_id: EntityId | None
    outcome: TaskOutcome


class ExecutionEvidence(FrozenModel):
    execution_id: EntityId
    trial_id: EntityId
    physical_attempt: int = Field(ge=1)
    runtime_kind: RuntimeKind
    status: Literal["succeeded", "failed", "cancelled"]
    failure_category: RetryCategory | None
    started_at: AwareDatetime
    completed_at: AwareDatetime
    retry_reason: str | None = None
    remote_job_id: str | None = None

    @model_validator(mode="after")
    def completion_follows_start(self) -> ExecutionEvidence:
        if self.completed_at < self.started_at:
            raise ValueError("execution completion precedes start")
        if self.physical_attempt == 1 and self.retry_reason is not None:
            raise ValueError("first execution cannot have a retry reason")
        if (self.status == "failed") != (self.failure_category is not None):
            raise ValueError("execution failure category conflicts with its status")
        return self


def task_outcome_matches_execution(
    outcome: TaskOutcome,
    status: str,
    failure_category: RetryCategory | None,
) -> bool:
    if outcome == "unsupported":
        return False
    if outcome == "scored":
        return status == "succeeded" and failure_category is None
    if status != "failed" or failure_category is None:
        return False
    if outcome == "agent_failed":
        return failure_category == "agent"
    if outcome == "benchmark_failed":
        return failure_category == "benchmark"
    return failure_category not in {"agent", "benchmark"}


def _validate_trial_execution_reference(
    trial: TrialEvidence,
    executions: Mapping[str, object],
) -> None:
    if trial.outcome == "unsupported":
        if trial.selected_execution_id is not None:
            raise ValueError("unsupported trial has a selected execution")
        if any(
            isinstance(execution, ExecutionEvidence)
            and execution.trial_id == trial.trial_id
            for execution in executions.values()
        ):
            raise ValueError("unsupported trial has a physical execution")
        return
    selected = executions.get(trial.selected_execution_id)
    if (
        not isinstance(selected, ExecutionEvidence)
        or selected.trial_id != trial.trial_id
        or not task_outcome_matches_execution(
            trial.outcome, selected.status, selected.failure_category
        )
    ):
        raise ValueError("trial selected execution conflicts with its outcome")


class MetricEvidence(FrozenModel):
    owner_type: OwnerType
    owner_id: EntityId
    name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    aggregation: str | None = None

    @model_validator(mode="after")
    def value_is_finite(self) -> MetricEvidence:
        if not math.isfinite(self.value):
            raise ValueError("metric value must be finite")
        return self


class ArtifactEvidence(FrozenModel):
    owner_type: OwnerType
    owner_id: EntityId
    kind: ArtifactKind
    path: str
    sha256: Digest
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def path_is_safe_metadata(self) -> ArtifactEvidence:
        parts = _validate_relative_path(self.path).parts
        if any(part.lower() in _FORBIDDEN_ARTIFACT_PARTS for part in parts):
            raise ValueError(
                "raw sessions, task contents, and logs are not publishable"
            )
        if self.path != _ARTIFACT_PATHS[self.kind]:
            raise ValueError("publishable artifact kind has a noncanonical path")
        return self


class ResultEvidence(FrozenModel):
    schema_version: Literal["harbor-hf/result-evidence/v1"] = (
        "harbor-hf/result-evidence/v1"
    )
    sanitized: Literal[True]
    run: RunEvidence
    trials: list[TrialEvidence]
    executions: list[ExecutionEvidence]
    metrics: list[MetricEvidence]
    artifacts: list[ArtifactEvidence]

    @model_validator(mode="after")
    def references_are_consistent(self) -> ResultEvidence:
        trials = _unique_by_id(self.trials, "trial_id", "trial")
        executions = _unique_by_id(self.executions, "execution_id", "execution")
        for trial in self.trials:
            _validate_trial_execution_reference(trial, executions)
        if any(execution.trial_id not in trials for execution in self.executions):
            raise ValueError("execution references an unknown trial")
        owners = {
            "run": {self.run.run_id},
            "trial": set(trials),
            "execution": set(executions),
        }
        for record in [*self.metrics, *self.artifacts]:
            if record.owner_id not in owners[record.owner_type]:
                raise ValueError("result evidence references an unknown owner")
        _require_unique_measurements(self.metrics)
        _require_unique_artifacts(self.artifacts)
        degraded = any(trial.outcome != "scored" for trial in self.trials)
        if (self.run.quality == "degraded") != degraded:
            raise ValueError("run quality conflicts with its trial outcomes")
        return self


class EvidenceReader(Protocol):
    def list_files(self, *, bucket: str, prefix: str) -> list[str]: ...

    def read_bytes(self, *, bucket: str, prefix: str, path: str) -> bytes: ...


class TraceValues(TypedDict):
    publication_id: str
    run_id: str
    source_bucket: str
    source_prefix: str
    source_checksum: str
    run_lock_path: str
    run_lock_sha256: str
    control_commit: str


class TraceRow(FrozenModel):
    schema_version: str
    publication_id: EntityId
    run_id: EntityId
    source_bucket: str = Field(min_length=1)
    source_prefix: str = Field(min_length=1)
    source_checksum: Digest
    run_lock_path: str = Field(min_length=1)
    run_lock_sha256: Digest
    control_commit: Commit


class RunRow(TraceRow):
    schema_version: Literal["harbor-hf/results/runs/v1"] = "harbor-hf/results/runs/v1"
    campaign_id: EntityId
    experiment: str = Field(min_length=1)
    evaluation_id: EvaluationId
    publication_role: PublicationRole
    component_kind: ComponentKind | None
    source_publication_ids: list[EntityId]
    benchmark: str = Field(min_length=1)
    benchmark_revision: str = Field(min_length=1)
    result_kind: ResultKind
    outcome: Literal["complete"]
    quality: RunQuality
    created_at: AwareDatetime
    completed_at: AwareDatetime
    model_id: EntityId
    model_repo: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    deployment_id: EntityId
    provider: str = Field(min_length=1)
    region: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    accelerator_count: int = Field(ge=0)
    agent_id: EntityId
    agent_name: str = Field(min_length=1)
    agent_revision: str = Field(min_length=1)
    planned_trial_count: int = Field(ge=0)
    scored_trial_count: int = Field(ge=0)
    agent_failed_count: int = Field(ge=0)
    benchmark_failed_count: int = Field(ge=0)
    infrastructure_exhausted_count: int = Field(ge=0)
    unsupported_count: int = Field(ge=0)
    execution_count: int = Field(ge=0)

    @model_validator(mode="after")
    def completion_follows_creation(self) -> RunRow:
        if self.completed_at < self.created_at:
            raise ValueError("run completion precedes creation")
        outcome_count = (
            self.scored_trial_count
            + self.agent_failed_count
            + self.benchmark_failed_count
            + self.infrastructure_exhausted_count
            + self.unsupported_count
        )
        if outcome_count != self.planned_trial_count:
            raise ValueError("run task outcome counts do not match trial count")
        if (self.quality == "degraded") != (
            self.scored_trial_count < self.planned_trial_count
        ):
            raise ValueError("run quality conflicts with task outcome counts")
        if (self.publication_role == "component") != (self.component_kind is not None):
            raise ValueError("run component kind conflicts with publication role")
        if self.publication_role == "final" and self.result_kind == "composed":
            if not self.source_publication_ids:
                raise ValueError("composed final result requires source publications")
        elif self.source_publication_ids:
            raise ValueError("only composed final results may reference publications")
        return self


class TrialRow(TraceRow):
    schema_version: Literal["harbor-hf/results/trials/v1"] = (
        "harbor-hf/results/trials/v1"
    )
    trial_id: EntityId
    task_name: str = Field(min_length=1)
    task_digest: Digest
    logical_attempt: int = Field(ge=1)
    selected_execution_id: EntityId | None
    outcome: TaskOutcome


class ExecutionRow(TraceRow):
    schema_version: Literal["harbor-hf/results/executions/v1"] = (
        "harbor-hf/results/executions/v1"
    )
    execution_id: EntityId
    trial_id: EntityId
    physical_attempt: int = Field(ge=1)
    runtime_kind: RuntimeKind
    status: Literal["succeeded", "failed", "cancelled"]
    failure_category: RetryCategory | None
    started_at: AwareDatetime
    completed_at: AwareDatetime
    retry_reason: str | None
    remote_job_id: str | None

    @model_validator(mode="after")
    def values_are_consistent(self) -> ExecutionRow:
        if self.completed_at < self.started_at:
            raise ValueError("execution completion precedes start")
        if self.physical_attempt == 1 and self.retry_reason is not None:
            raise ValueError("first execution cannot have a retry reason")
        if (self.status == "failed") != (self.failure_category is not None):
            raise ValueError("execution failure category conflicts with its status")
        return self


class MetricRow(TraceRow):
    schema_version: Literal["harbor-hf/results/metrics/v1"] = (
        "harbor-hf/results/metrics/v1"
    )
    metric_id: EntityId
    owner_type: OwnerType
    owner_id: EntityId
    name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1)
    aggregation: str | None


class ArtifactRow(TraceRow):
    schema_version: Literal["harbor-hf/results/artifacts/v1"] = (
        "harbor-hf/results/artifacts/v1"
    )
    artifact_id: EntityId
    owner_type: OwnerType
    owner_id: EntityId
    kind: ArtifactKind
    path: str = Field(min_length=1)
    sha256: Digest
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)


class PublicationProvenance(FrozenModel):
    schema_version: Literal["harbor-hf/result-provenance/v1"] = (
        "harbor-hf/result-provenance/v1"
    )
    envelope_path: Literal["publication-envelope.v1.json"] = PUBLICATION_ENVELOPE_PATH
    envelope_sha256: Digest
    projection_version: str = Field(min_length=1)
    sanitizer_version: str = Field(min_length=1)
    execution_profile_sha256: Digest
    harbor_bundle_manifest_sha256s: list[Digest]
    harbor_archive_sha256s: list[Digest]

    @model_validator(mode="after")
    def bundle_references_are_paired(self) -> PublicationProvenance:
        if len(self.harbor_bundle_manifest_sha256s) != len(self.harbor_archive_sha256s):
            raise ValueError("Harbor bundle and archive references are not paired")
        return self


class ResultTables(FrozenModel):
    publication_id: str
    runs: list[RunRow]
    trials: list[TrialRow]
    executions: list[ExecutionRow]
    metrics: list[MetricRow]
    artifacts: list[ArtifactRow]
    provenance: PublicationProvenance

    @model_validator(mode="after")
    def has_one_consistent_run(self) -> ResultTables:
        if len(self.runs) != 1:
            raise ValueError("a publication must contain exactly one run row")
        rows = [
            *self.runs,
            *self.trials,
            *self.executions,
            *self.metrics,
            *self.artifacts,
        ]
        if any(row.publication_id != self.publication_id for row in rows):
            raise ValueError("publication rows have conflicting identities")
        return self


class GlobalIndexRow(FrozenModel):
    schema_version: Literal["harbor-hf/results/index/v1"] = "harbor-hf/results/index/v1"
    publication_id: EntityId
    run_id: EntityId
    campaign_id: EntityId
    evaluation_id: EvaluationId
    publication_role: PublicationRole
    component_kind: ComponentKind | None
    benchmark: str = Field(min_length=1)
    result_kind: ResultKind
    outcome: Literal["complete"]
    quality: RunQuality
    completed_at: AwareDatetime
    model_repo: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    agent_revision: str = Field(min_length=1)
    result_dataset: DatasetId
    result_revision: Commit
    source_checksum: Digest
    control_commit: Commit


class CatalogRow(FrozenModel):
    schema_version: Literal["harbor-hf/results/catalog/v1"] = (
        "harbor-hf/results/catalog/v1"
    )
    publication_id: EntityId
    run_id: EntityId
    campaign_id: EntityId
    evaluation_id: EvaluationId
    publication_role: PublicationRole
    component_kind: ComponentKind | None
    source_publication_ids: list[EntityId]
    benchmark: str = Field(min_length=1)
    benchmark_revision: str = Field(min_length=1)
    result_kind: ResultKind
    outcome: Literal["complete"]
    quality: RunQuality
    created_at: AwareDatetime
    completed_at: AwareDatetime
    model_repo: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    agent_revision: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    region: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    accelerator_count: int = Field(ge=0)
    score: float = Field(allow_inf_nan=False)
    passed_trials: int = Field(ge=0)
    planned_trial_count: int = Field(ge=0)
    scored_trial_count: int = Field(ge=0)
    agent_failed_count: int = Field(ge=0)
    benchmark_failed_count: int = Field(ge=0)
    infrastructure_exhausted_count: int = Field(ge=0)
    unsupported_count: int = Field(ge=0)
    execution_count: int = Field(ge=0)
    failed_executions: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    result_dataset: DatasetId
    result_revision: Commit
    source_checksum: Digest
    control_commit: Commit
    projection_path: str = Field(min_length=1)
    projection_sha256: Digest
    envelope_sha256: Digest
    harbor_bundle_count: int = Field(ge=1)

    @model_validator(mode="after")
    def values_are_consistent(self) -> CatalogRow:
        if not math.isfinite(self.score):
            raise ValueError("catalog score must be finite")
        if self.passed_trials > self.planned_trial_count:
            raise ValueError("catalog passed trials exceed trial count")
        outcome_count = (
            self.scored_trial_count
            + self.agent_failed_count
            + self.benchmark_failed_count
            + self.infrastructure_exhausted_count
            + self.unsupported_count
        )
        if outcome_count != self.planned_trial_count:
            raise ValueError("catalog task outcome counts do not match trial count")
        if (self.quality == "degraded") != (
            self.scored_trial_count < self.planned_trial_count
        ):
            raise ValueError("catalog quality conflicts with task outcome counts")
        if self.completed_at < self.created_at:
            raise ValueError("catalog completion precedes creation")
        _validate_relative_path(self.projection_path)
        return self


class DatasetFile(FrozenModel):
    path: str
    content: bytes


class ResultPublication(FrozenModel):
    tables: ResultTables
    files: list[DatasetFile]
    receipt_path: str
    receipt: bytes


class UnsupportedTask(FrozenModel):
    task_name: str = Field(min_length=1)
    task_digest: Digest
    logical_attempt: int = Field(default=1, ge=1)


class ResultCompositionManifest(FrozenModel):
    schema_version: Literal["harbor-hf/result-composition/v1"] = (
        "harbor-hf/result-composition/v1"
    )
    run_id: EntityId
    campaign_id: EntityId
    experiment: str = Field(min_length=1)
    evaluation_id: EvaluationId
    created_at: AwareDatetime
    completed_at: AwareDatetime
    evidence_bucket: str = Field(min_length=1)
    evidence_prefix: str = Field(min_length=1)
    sources: list[SourcePublicationReference] = Field(min_length=1)
    unsupported_tasks: list[UnsupportedTask] = Field(default_factory=list)

    @model_validator(mode="after")
    def values_are_consistent(self) -> ResultCompositionManifest:
        if self.completed_at < self.created_at:
            raise ValueError("composition completion precedes creation")
        if sum(source.role == "base" for source in self.sources) != 1:
            raise ValueError("composition requires exactly one base publication")
        selected = [
            (trial.task_name, trial.logical_attempt)
            for source in self.sources
            for trial in source.selected_trials
        ]
        unsupported = [
            (task.task_name, task.logical_attempt) for task in self.unsupported_tasks
        ]
        if len(selected) != len(set(selected)):
            raise ValueError("composition selects a task from multiple publications")
        if len(unsupported) != len(set(unsupported)):
            raise ValueError("composition has duplicate unsupported tasks")
        if set(selected).intersection(unsupported):
            raise ValueError("unsupported task is also selected from a publication")
        _validate_relative_path(self.evidence_prefix)
        return self


class ComposedResult(FrozenModel):
    manifest: ResultCompositionManifest
    envelope: PublicationEnvelope
    tables: ResultTables
    evidence_files: list[DatasetFile]


class ProjectionFileReference(FrozenModel):
    path: str = Field(min_length=1)
    sha256: Digest
    row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def path_is_relative(self) -> ProjectionFileReference:
        _validate_relative_path(self.path)
        return self


class ResultProjection(FrozenModel):
    schema_version: Literal["harbor-hf/result-projection/v1"] = (
        "harbor-hf/result-projection/v1"
    )
    publication_id: EntityId
    run_id: EntityId
    source_bucket: str = Field(min_length=1)
    source_prefix: str = Field(min_length=1)
    source_checksum: Digest
    control_commit: Commit
    envelope_path: Literal["publication-envelope.v1.json"]
    envelope_sha256: Digest
    projection_version: str = Field(min_length=1)
    sanitizer_version: str = Field(min_length=1)
    execution_profile_sha256: Digest
    harbor_bundle_manifest_sha256s: list[Digest]
    harbor_archive_sha256s: list[Digest]
    tables: dict[TableName, ProjectionFileReference]

    @model_validator(mode="after")
    def tables_are_complete(self) -> ResultProjection:
        if set(self.tables) != set(_table_names()):
            raise ValueError("projection does not reference every query table")
        return self


class RebuildRequest(FrozenModel):
    source: EvidenceSource
    control_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")


class AuditReport(FrozenModel):
    publication_id: str
    source_checksum: str
    row_counts: dict[TableName, int]


def build_result_tables(
    reader: EvidenceReader,
    source: EvidenceSource,
    *,
    control_commit: str,
) -> ResultTables:
    if not _is_commit(control_commit):
        raise ValueError("control commit must be a 40- or 64-character hex digest")
    summary, lock, source_checksum, lock_checksum, checksums = _verify_evidence(
        reader, source
    )
    provenance = _load_publication_provenance(reader, source, summary, lock, checksums)
    trace: TraceValues = {
        "publication_id": _publication_id(
            summary.run.run_id,
            source,
            source_checksum,
            lock_checksum,
            provenance.execution_profile_sha256,
        ),
        "run_id": summary.run.run_id,
        "source_bucket": source.bucket,
        "source_prefix": source.prefix,
        "source_checksum": source_checksum,
        "run_lock_path": source.run_lock_path,
        "run_lock_sha256": lock_checksum,
        "control_commit": control_commit,
    }
    publication_id = trace["publication_id"]
    run_evidence = summary.run
    outcome_counts = {
        outcome: sum(trial.outcome == outcome for trial in summary.trials)
        for outcome in _TASK_OUTCOMES
    }
    run = RunRow(
        **trace,
        campaign_id=run_evidence.campaign_id,
        experiment=run_evidence.experiment,
        evaluation_id=run_evidence.evaluation_id,
        publication_role=run_evidence.publication_role,
        component_kind=run_evidence.component_kind,
        source_publication_ids=[],
        benchmark=run_evidence.benchmark,
        benchmark_revision=run_evidence.benchmark_revision,
        result_kind=run_evidence.result_kind,
        outcome=run_evidence.outcome,
        quality=run_evidence.quality,
        created_at=run_evidence.created_at,
        completed_at=run_evidence.completed_at,
        model_id=run_evidence.model_id,
        model_repo=run_evidence.model_repo,
        model_revision=run_evidence.model_revision,
        deployment_id=run_evidence.deployment_id,
        provider=run_evidence.provider,
        region=run_evidence.region,
        hardware=run_evidence.hardware,
        accelerator_count=run_evidence.accelerator_count,
        agent_id=run_evidence.agent_id,
        agent_name=run_evidence.agent_name,
        agent_revision=run_evidence.agent_revision,
        planned_trial_count=len(summary.trials),
        scored_trial_count=outcome_counts["scored"],
        agent_failed_count=outcome_counts["agent_failed"],
        benchmark_failed_count=outcome_counts["benchmark_failed"],
        infrastructure_exhausted_count=outcome_counts["infrastructure_exhausted"],
        unsupported_count=outcome_counts["unsupported"],
        execution_count=len(summary.executions),
    )
    trials = [
        TrialRow(
            **trace,
            trial_id=record.trial_id,
            task_name=record.task_name,
            task_digest=record.task_digest,
            logical_attempt=record.logical_attempt,
            selected_execution_id=record.selected_execution_id,
            outcome=record.outcome,
        )
        for record in sorted(summary.trials, key=lambda item: item.trial_id)
    ]
    executions = [
        ExecutionRow(
            **trace,
            execution_id=record.execution_id,
            trial_id=record.trial_id,
            physical_attempt=record.physical_attempt,
            runtime_kind=record.runtime_kind,
            status=record.status,
            failure_category=record.failure_category,
            started_at=record.started_at,
            completed_at=record.completed_at,
            retry_reason=record.retry_reason,
            remote_job_id=record.remote_job_id,
        )
        for record in sorted(summary.executions, key=lambda item: item.execution_id)
    ]
    metrics = [
        MetricRow(
            **trace,
            metric_id=_metric_id(summary.run.run_id, record),
            owner_type=record.owner_type,
            owner_id=record.owner_id,
            name=record.name,
            value=record.value,
            unit=record.unit,
            aggregation=record.aggregation,
        )
        for record in sorted(summary.metrics, key=_measurement_key)
    ]
    artifacts = [
        ArtifactRow(
            **trace,
            artifact_id=_artifact_id(summary.run.run_id, record),
            owner_type=record.owner_type,
            owner_id=record.owner_id,
            kind=record.kind,
            path=record.path,
            sha256=record.sha256,
            media_type=record.media_type,
            size_bytes=record.size_bytes,
        )
        for record in sorted(summary.artifacts, key=_artifact_key)
    ]
    return ResultTables(
        publication_id=publication_id,
        runs=[run],
        trials=trials,
        executions=executions,
        metrics=metrics,
        artifacts=artifacts,
        provenance=provenance,
    )


def compose_result_tables(
    manifest: ResultCompositionManifest,
    sources: Mapping[str, ResultTables],
    *,
    control_commit: str,
) -> ComposedResult:
    if not _is_commit(control_commit):
        raise ValueError("control commit must be a 40- or 64-character hex digest")
    _validate_composition_sources(manifest, sources)
    base_reference = next(
        source for source in manifest.sources if source.role == "base"
    )
    base = sources[base_reference.publication_id]
    base_run = base.runs[0]
    _validate_composition_compatibility(manifest, sources, base_run)
    selected_rows = _select_composition_rows(manifest, sources, base)
    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    manifest_digest = _sha256_bytes(manifest_bytes)
    trace = _composition_trace(
        manifest,
        manifest_digest,
        base.provenance.execution_profile_sha256,
        control_commit,
    )
    trials, executions, metrics = _compose_child_rows(manifest, selected_rows, trace)
    _require_unique_composed_rows(trials, executions, metrics)
    run = _build_composed_run(manifest, base_run, trace, trials, executions)
    source_pairs = sorted(
        {
            pair
            for tables in sources.values()
            for pair in zip(
                tables.provenance.harbor_bundle_manifest_sha256s,
                tables.provenance.harbor_archive_sha256s,
                strict=True,
            )
        }
    )
    envelope = _build_composition_envelope(
        manifest, manifest_bytes, base_run, executions
    )
    envelope_bytes = canonical_json_bytes(envelope.model_dump(mode="json"))
    artifacts = [
        ArtifactRow(
            **trace,
            artifact_id=_composition_entity_id(
                "artifact", manifest.run_id, "composition.json"
            ),
            owner_type="run",
            owner_id=manifest.run_id,
            kind="composition",
            path="composition.json",
            sha256=manifest_digest,
            media_type="application/json",
            size_bytes=len(manifest_bytes),
        )
    ]
    tables = ResultTables(
        publication_id=trace["publication_id"],
        runs=[run],
        trials=sorted(trials, key=lambda item: item.trial_id),
        executions=sorted(executions, key=lambda item: item.execution_id),
        metrics=sorted(metrics, key=lambda item: item.metric_id),
        artifacts=artifacts,
        provenance=PublicationProvenance(
            envelope_sha256=_sha256_bytes(envelope_bytes),
            projection_version=envelope.projection_version,
            sanitizer_version=envelope.sanitizer_version,
            execution_profile_sha256=base.provenance.execution_profile_sha256,
            harbor_bundle_manifest_sha256s=[pair[0] for pair in source_pairs],
            harbor_archive_sha256s=[pair[1] for pair in source_pairs],
        ),
    )
    return ComposedResult(
        manifest=manifest,
        envelope=envelope,
        tables=tables,
        evidence_files=[
            DatasetFile(path="composition.json", content=manifest_bytes),
            DatasetFile(path=PUBLICATION_ENVELOPE_PATH, content=envelope_bytes),
        ],
    )


def build_result_publication(
    tables: ResultTables,
    *,
    extra_files: Sequence[DatasetFile] = (),
) -> ResultPublication:
    campaign_id = tables.runs[0].campaign_id
    files = [
        DatasetFile(
            path=(
                f"data/{table}/schema=v1/campaign={campaign_id}/"
                f"{tables.publication_id}.parquet"
            ),
            content=_parquet_bytes(_table_rows(tables, table), parquet_schema(table)),
        )
        for table in _table_names()
    ]
    files.append(_build_projection_file(tables, files))
    files.extend(extra_files)
    receipt_path = f"publications/{tables.publication_id}.json"
    receipt_value = {
        "schema_version": "harbor-hf/result-publication/v1",
        "publication_id": tables.publication_id,
        "run_id": tables.runs[0].run_id,
        "source_checksum": tables.runs[0].source_checksum,
        "files": {item.path: _sha256_bytes(item.content) for item in files},
    }
    return ResultPublication(
        tables=tables,
        files=files,
        receipt_path=receipt_path,
        receipt=_canonical_json(receipt_value),
    )


def build_composed_result_publication(result: ComposedResult) -> ResultPublication:
    manifest_bytes = canonical_json_bytes(result.manifest.model_dump(mode="json"))
    return build_result_publication(
        result.tables,
        extra_files=[
            DatasetFile(
                path=f"compositions/{result.tables.publication_id}.json",
                content=manifest_bytes,
            )
        ],
    )


def _validate_composition_sources(
    manifest: ResultCompositionManifest,
    sources: Mapping[str, ResultTables],
) -> None:
    expected_sources = {source.publication_id for source in manifest.sources}
    if set(sources) != expected_sources:
        raise ResultPublicationError("composition sources do not match its manifest")
    for source in manifest.sources:
        tables = sources[source.publication_id]
        run = tables.runs[0]
        if (
            tables.publication_id != source.publication_id
            or run.run_id != source.run_id
            or run.source_checksum != source.source_checksum
        ):
            raise ResultPublicationError("composition source identity conflicts")


def _select_composition_rows(
    manifest: ResultCompositionManifest,
    sources: Mapping[str, ResultTables],
    base: ResultTables,
) -> list[tuple[TrialRow, ResultTables]]:
    selected_rows: list[tuple[TrialRow, ResultTables]] = []
    selected_base_keys: set[tuple[str, int]] = set()
    base_trials = {_trial_key(trial): trial for trial in base.trials}
    for source in manifest.sources:
        tables = sources[source.publication_id]
        by_key = {_trial_key(trial): trial for trial in tables.trials}
        if len(by_key) != len(tables.trials):
            raise ResultPublicationError("composition source has duplicate trials")
        for selection in source.selected_trials:
            trial = by_key.get((selection.task_name, selection.logical_attempt))
            if trial is None:
                raise ResultPublicationError(
                    "composition selects an unknown source task"
                )
            _validate_correction_trial(source, trial, base_trials)
            selected_base_keys.add(_trial_key(trial))
            selected_rows.append((trial, tables))
    if set(base_trials) != selected_base_keys:
        raise ResultPublicationError(
            "composition does not resolve every task in the base publication"
        )
    return selected_rows


def _validate_correction_trial(
    source: SourcePublicationReference,
    trial: TrialRow,
    base_trials: Mapping[tuple[str, int], TrialRow],
) -> None:
    if source.role != "correction":
        return
    base_trial = base_trials.get(_trial_key(trial))
    if base_trial is None or base_trial.task_digest != trial.task_digest:
        raise ResultPublicationError(
            "correction task conflicts with the base publication"
        )


def _composition_trace(
    manifest: ResultCompositionManifest,
    manifest_digest: Digest,
    execution_profile_sha256: Digest,
    control_commit: str,
) -> TraceValues:
    source = EvidenceSource(
        bucket=manifest.evidence_bucket,
        prefix=manifest.evidence_prefix,
        run_lock_path="composition.json",
    )
    return {
        "publication_id": _publication_id(
            manifest.run_id,
            source,
            manifest_digest,
            manifest_digest,
            execution_profile_sha256,
        ),
        "run_id": manifest.run_id,
        "source_bucket": manifest.evidence_bucket,
        "source_prefix": manifest.evidence_prefix,
        "source_checksum": manifest_digest,
        "run_lock_path": "composition.json",
        "run_lock_sha256": manifest_digest,
        "control_commit": control_commit,
    }


def _compose_child_rows(
    manifest: ResultCompositionManifest,
    selected_rows: Sequence[tuple[TrialRow, ResultTables]],
    trace: TraceValues,
) -> tuple[list[TrialRow], list[ExecutionRow], list[MetricRow]]:
    trials: list[TrialRow] = []
    executions: list[ExecutionRow] = []
    metrics: list[MetricRow] = []
    for trial, tables in selected_rows:
        trials.append(trial.model_copy(update={**trace}))
        trial_executions = [
            execution
            for execution in tables.executions
            if execution.trial_id == trial.trial_id
        ]
        executions.extend(
            execution.model_copy(update={**trace}) for execution in trial_executions
        )
        metrics.extend(_copy_selected_metrics(trial, trial_executions, tables, trace))
    for unsupported in manifest.unsupported_tasks:
        trial, reward = _unsupported_rows(manifest.run_id, unsupported, trace)
        trials.append(trial)
        metrics.append(reward)
    return trials, executions, metrics


def _copy_selected_metrics(
    trial: TrialRow,
    executions: Sequence[ExecutionRow],
    tables: ResultTables,
    trace: TraceValues,
) -> list[MetricRow]:
    execution_ids = {execution.execution_id for execution in executions}
    return [
        metric.model_copy(update={**trace})
        for metric in tables.metrics
        if (metric.owner_type == "trial" and metric.owner_id == trial.trial_id)
        or (metric.owner_type == "execution" and metric.owner_id in execution_ids)
    ]


def _unsupported_rows(
    run_id: str,
    unsupported: UnsupportedTask,
    trace: TraceValues,
) -> tuple[TrialRow, MetricRow]:
    trial_id = _composition_entity_id(
        "trial",
        run_id,
        unsupported.task_name,
        str(unsupported.logical_attempt),
        "unsupported",
    )
    trial = TrialRow(
        **trace,
        trial_id=trial_id,
        task_name=unsupported.task_name,
        task_digest=unsupported.task_digest,
        logical_attempt=unsupported.logical_attempt,
        selected_execution_id=None,
        outcome="unsupported",
    )
    metric = MetricEvidence(
        owner_type="trial",
        owner_id=trial_id,
        name="reward",
        value=0.0,
        unit="score",
    )
    reward = MetricRow(
        **trace,
        metric_id=_metric_id(run_id, metric),
        **metric.model_dump(mode="python"),
    )
    return trial, reward


def _build_composed_run(
    manifest: ResultCompositionManifest,
    base: RunRow,
    trace: TraceValues,
    trials: Sequence[TrialRow],
    executions: Sequence[ExecutionRow],
) -> RunRow:
    counts = {
        outcome: sum(trial.outcome == outcome for trial in trials)
        for outcome in _TASK_OUTCOMES
    }
    return RunRow(
        **trace,
        campaign_id=manifest.campaign_id,
        experiment=manifest.experiment,
        evaluation_id=manifest.evaluation_id,
        publication_role="final",
        component_kind=None,
        source_publication_ids=sorted(
            source.publication_id for source in manifest.sources
        ),
        benchmark=base.benchmark,
        benchmark_revision=base.benchmark_revision,
        result_kind="composed",
        outcome="complete",
        quality="degraded" if counts["scored"] < len(trials) else "clean",
        created_at=manifest.created_at,
        completed_at=manifest.completed_at,
        model_id=base.model_id,
        model_repo=base.model_repo,
        model_revision=base.model_revision,
        deployment_id=base.deployment_id,
        provider=base.provider,
        region=base.region,
        hardware=base.hardware,
        accelerator_count=base.accelerator_count,
        agent_id=base.agent_id,
        agent_name=base.agent_name,
        agent_revision=base.agent_revision,
        planned_trial_count=len(trials),
        scored_trial_count=counts["scored"],
        agent_failed_count=counts["agent_failed"],
        benchmark_failed_count=counts["benchmark_failed"],
        infrastructure_exhausted_count=counts["infrastructure_exhausted"],
        unsupported_count=counts["unsupported"],
        execution_count=len(executions),
    )


def _build_composition_envelope(
    manifest: ResultCompositionManifest,
    manifest_bytes: bytes,
    base: RunRow,
    executions: Sequence[ExecutionRow],
) -> PublicationEnvelope:
    physical_executions = [
        PhysicalExecutionReference(
            execution_id=execution.execution_id,
            trial_id=execution.trial_id,
            physical_attempt=execution.physical_attempt,
            status=execution.status,
            failure_category=execution.failure_category,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            retry_reason=execution.retry_reason,
            remote_job_id=execution.remote_job_id,
            bundle_status="source_publication",
        )
        for execution in executions
    ]
    return PublicationEnvelope(
        run_id=manifest.run_id,
        campaign_id=manifest.campaign_id,
        created_at=manifest.created_at,
        completed_at=manifest.completed_at,
        evidence_bucket=manifest.evidence_bucket,
        evidence_prefix=manifest.evidence_prefix,
        run_lock=object_reference("composition.json", manifest_bytes),
        profiles=ProfileDigests(
            experiment=canonical_digest(
                {
                    "experiment": manifest.experiment,
                    "sources": [
                        source.model_dump(mode="json") for source in manifest.sources
                    ],
                }
            ),
            model=canonical_digest(
                {"repo": base.model_repo, "revision": base.model_revision}
            ),
            deployment=canonical_digest(
                {
                    "deployment_id": base.deployment_id,
                    "provider": base.provider,
                    "region": base.region,
                    "hardware": base.hardware,
                    "accelerator_count": base.accelerator_count,
                }
            ),
            agent=canonical_digest(
                {"name": base.agent_name, "revision": base.agent_revision}
            ),
        ),
        runtime=RuntimeIdentity(
            kind=executions[0].runtime_kind,
            provider=base.provider,
            region=base.region,
            hardware=base.hardware,
            accelerator_count=base.accelerator_count,
        ),
        cleanup_outcome=(
            "not_applicable" if executions[0].runtime_kind == "provider" else "verified"
        ),
        executions=physical_executions,
        sources=manifest.sources,
    )


def _validate_composition_compatibility(
    manifest: ResultCompositionManifest,
    sources: Mapping[str, ResultTables],
    base: RunRow,
) -> None:
    if (
        len({tables.provenance.execution_profile_sha256 for tables in sources.values()})
        != 1
    ):
        raise ResultPublicationError("composition sources are incompatible")
    fields = (
        "benchmark_revision",
        "model_repo",
        "model_revision",
        "provider",
        "region",
        "hardware",
        "accelerator_count",
        "agent_name",
        "agent_revision",
    )
    for reference in manifest.sources:
        run = sources[reference.publication_id].runs[0]
        if (
            run.result_kind != "ordinary"
            or run.evaluation_id != manifest.evaluation_id
            or run.publication_role != "component"
            or run.component_kind != reference.role
            or any(getattr(run, field) != getattr(base, field) for field in fields)
        ):
            raise ResultPublicationError("composition sources are incompatible")
    runtime_kinds = {
        execution.runtime_kind
        for tables in sources.values()
        for execution in tables.executions
    }
    if not runtime_kinds:
        raise ResultPublicationError("composition sources contain no executions")
    if len(runtime_kinds) != 1:
        raise ResultPublicationError("composition sources use mixed runtime kinds")


def _trial_key(trial: TrialRow) -> tuple[str, int]:
    return trial.task_name, trial.logical_attempt


def _composition_entity_id(kind: str, *parts: str) -> str:
    digest = hashlib.sha256(":".join(parts).encode()).hexdigest()[:24]
    return f"{kind}-{digest}"


def _require_unique_composed_rows(
    trials: Sequence[TrialRow],
    executions: Sequence[ExecutionRow],
    metrics: Sequence[MetricRow],
) -> None:
    checks = (
        ("trial", [row.trial_id for row in trials]),
        ("execution", [row.execution_id for row in executions]),
        ("metric", [row.metric_id for row in metrics]),
    )
    for kind, identities in checks:
        if len(identities) != len(set(identities)):
            raise ResultPublicationError(f"composition has duplicate {kind} identities")


def _build_projection_file(
    tables: ResultTables, table_files: list[DatasetFile]
) -> DatasetFile:
    provenance = tables.provenance
    run = tables.runs[0]
    references = {
        table: ProjectionFileReference(
            path=next(
                item.path
                for item in table_files
                if item.path.startswith(f"data/{table}/")
            ),
            sha256=_sha256_bytes(
                next(
                    item.content
                    for item in table_files
                    if item.path.startswith(f"data/{table}/")
                )
            ),
            row_count=len(_table_rows(tables, table)),
        )
        for table in _table_names()
    }
    projection = ResultProjection(
        publication_id=tables.publication_id,
        run_id=run.run_id,
        source_bucket=run.source_bucket,
        source_prefix=run.source_prefix,
        source_checksum=run.source_checksum,
        control_commit=run.control_commit,
        envelope_path=provenance.envelope_path,
        envelope_sha256=provenance.envelope_sha256,
        projection_version=provenance.projection_version,
        sanitizer_version=provenance.sanitizer_version,
        execution_profile_sha256=provenance.execution_profile_sha256,
        harbor_bundle_manifest_sha256s=(provenance.harbor_bundle_manifest_sha256s),
        harbor_archive_sha256s=provenance.harbor_archive_sha256s,
        tables=references,
    )
    return DatasetFile(
        path=f"projections/schema=v1/{tables.publication_id}.json",
        content=_canonical_json(projection.model_dump(mode="json")),
    )


def build_global_index_row(
    tables: ResultTables, *, result_dataset: str, result_revision: str
) -> GlobalIndexRow:
    run = tables.runs[0]
    return GlobalIndexRow(
        publication_id=tables.publication_id,
        run_id=run.run_id,
        campaign_id=run.campaign_id,
        evaluation_id=run.evaluation_id,
        publication_role=run.publication_role,
        component_kind=run.component_kind,
        benchmark=run.benchmark,
        result_kind=run.result_kind,
        outcome=run.outcome,
        quality=run.quality,
        completed_at=run.completed_at,
        model_repo=run.model_repo,
        model_revision=run.model_revision,
        agent_name=run.agent_name,
        agent_revision=run.agent_revision,
        result_dataset=result_dataset,
        result_revision=result_revision,
        source_checksum=run.source_checksum,
        control_commit=run.control_commit,
    )


def build_index_file(row: GlobalIndexRow) -> DatasetFile:
    return DatasetFile(
        path=f"data/index/schema=v1/{row.publication_id}.parquet",
        content=_parquet_bytes([row], index_parquet_schema()),
    )


def build_index_window_file(rows: Sequence[GlobalIndexRow], size: int) -> DatasetFile:
    if size < 1:
        raise ValueError("index window size must be positive")
    return DatasetFile(
        path=f"data/index/schema=v1/windows/{size:04d}.parquet",
        content=_parquet_bytes(rows[:size], index_parquet_schema()),
    )


def build_catalog_row(
    tables: ResultTables,
    *,
    result_dataset: str,
    result_revision: str,
    projection: DatasetFile,
) -> CatalogRow:
    run = tables.runs[0]
    rewards = _trial_reward_scores(tables)
    score = sum(rewards) / len(rewards) if rewards else 0.0
    return CatalogRow(
        publication_id=tables.publication_id,
        run_id=run.run_id,
        campaign_id=run.campaign_id,
        evaluation_id=run.evaluation_id,
        publication_role=run.publication_role,
        component_kind=run.component_kind,
        source_publication_ids=run.source_publication_ids,
        benchmark=run.benchmark,
        benchmark_revision=run.benchmark_revision,
        result_kind=run.result_kind,
        outcome=run.outcome,
        quality=run.quality,
        created_at=run.created_at,
        completed_at=run.completed_at,
        model_repo=run.model_repo,
        model_revision=run.model_revision,
        agent_name=run.agent_name,
        agent_revision=run.agent_revision,
        provider=run.provider,
        region=run.region,
        hardware=run.hardware,
        accelerator_count=run.accelerator_count,
        score=score,
        passed_trials=sum(value >= 1.0 for value in rewards),
        planned_trial_count=run.planned_trial_count,
        scored_trial_count=run.scored_trial_count,
        agent_failed_count=run.agent_failed_count,
        benchmark_failed_count=run.benchmark_failed_count,
        infrastructure_exhausted_count=run.infrastructure_exhausted_count,
        unsupported_count=run.unsupported_count,
        execution_count=run.execution_count,
        failed_executions=sum(
            execution.status == "failed" for execution in tables.executions
        ),
        duration_seconds=(run.completed_at - run.created_at).total_seconds(),
        result_dataset=result_dataset,
        result_revision=result_revision,
        source_checksum=run.source_checksum,
        control_commit=run.control_commit,
        projection_path=projection.path,
        projection_sha256=_sha256_bytes(projection.content),
        envelope_sha256=tables.provenance.envelope_sha256,
        harbor_bundle_count=len(tables.provenance.harbor_archive_sha256s),
    )


def build_catalog_window_file(
    rows: Sequence[CatalogRow], size: int, *, scope: CatalogScope
) -> DatasetFile:
    if size < 1:
        raise ValueError("catalog window size must be positive")
    return DatasetFile(
        path=f"data/catalog/schema=v1/{scope}/windows/{size:04d}.parquet",
        content=_parquet_bytes(rows[:size], catalog_parquet_schema()),
    )


def catalog_lookup_path(run_id: str) -> str:
    identity = hashlib.sha256(run_id.encode()).hexdigest()
    return f"data/catalog/schema=v1/runs/{identity}.parquet"


def catalog_publication_lookup_path(publication_id: str) -> str:
    identity = hashlib.sha256(publication_id.encode()).hexdigest()
    return f"data/catalog/schema=v1/publications/{identity}.parquet"


def build_catalog_lookup_file(row: CatalogRow) -> DatasetFile:
    return DatasetFile(
        path=catalog_lookup_path(row.run_id),
        content=_parquet_bytes([row], catalog_parquet_schema()),
    )


def build_catalog_publication_lookup_file(row: CatalogRow) -> DatasetFile:
    return DatasetFile(
        path=catalog_publication_lookup_path(row.publication_id),
        content=_parquet_bytes([row], catalog_parquet_schema()),
    )


def read_catalog_file(content: bytes) -> list[CatalogRow]:
    try:
        values = pq.read_table(
            pa.BufferReader(content), schema=catalog_parquet_schema()
        )
    except (pa.ArrowException, OSError) as error:
        raise ValueError("result catalog Parquet is invalid") from error
    return [CatalogRow.model_validate(value) for value in values.to_pylist()]


def _trial_reward_scores(tables: ResultTables) -> list[float]:
    by_trial: dict[str, list[MetricRow]] = {}
    for metric in tables.metrics:
        if metric.owner_type == "trial" and metric.unit == "score":
            by_trial.setdefault(metric.owner_id, []).append(metric)
    scores = [
        _select_reward_score(by_trial.get(trial.trial_id, []))
        for trial in tables.trials
    ]
    if any(score is None for score in scores):
        raise ResultPublicationError("trial has no score metric")
    return [float(score) for score in scores if score is not None]


def trial_reward_score(metrics: Sequence[MetricRow], trial_id: str) -> float | None:
    candidates = [
        metric
        for metric in metrics
        if metric.owner_type == "trial"
        and metric.owner_id == trial_id
        and metric.unit == "score"
    ]
    return _select_reward_score(candidates)


def _select_reward_score(candidates: Sequence[MetricRow]) -> float | None:
    if not candidates:
        return None
    preferred = next(
        (
            metric
            for name in ("reward", "score", "verifier_reward")
            for metric in candidates
            if metric.name.casefold() == name
        ),
        None,
    )
    if preferred is not None:
        return preferred.value
    named_scores = [
        metric
        for metric in candidates
        if any(
            term in metric.name.casefold() for term in ("reward", "score", "verifier")
        )
    ]
    selected = named_scores or candidates
    return sum(metric.value for metric in selected) / len(selected)


def read_index_file(content: bytes) -> list[GlobalIndexRow]:
    try:
        values = pq.read_table(pa.BufferReader(content), schema=index_parquet_schema())
    except (pa.ArrowException, OSError) as error:
        raise ValueError("global index Parquet is invalid") from error
    return [GlobalIndexRow.model_validate(value) for value in values.to_pylist()]


def rebuild_result_tables(
    reader: EvidenceReader, requests: list[RebuildRequest]
) -> list[ResultTables]:
    rebuilt = [
        build_result_tables(
            reader,
            request.source,
            control_commit=request.control_commit,
        )
        for request in sorted(
            requests,
            key=lambda item: (
                item.source.bucket,
                item.source.prefix,
                item.control_commit,
            ),
        )
    ]
    identities = [tables.publication_id for tables in rebuilt]
    if len(set(identities)) != len(identities):
        raise ResultPublicationError("rebuild contains a duplicate publication")
    return rebuilt


def audit_result_tables(
    reader: EvidenceReader,
    source: EvidenceSource,
    *,
    control_commit: str,
    observed: ResultTables,
) -> AuditReport:
    expected = build_result_tables(reader, source, control_commit=control_commit)
    if observed != expected:
        raise ResultPublicationError("published rows differ from canonical evidence")
    run = expected.runs[0]
    return AuditReport(
        publication_id=expected.publication_id,
        source_checksum=run.source_checksum,
        row_counts={
            "runs": len(expected.runs),
            "trials": len(expected.trials),
            "executions": len(expected.executions),
            "metrics": len(expected.metrics),
            "artifacts": len(expected.artifacts),
        },
    )


def result_schema_manifest() -> dict[str, object]:
    schemas: dict[str, object] = {
        name: _schema_description(parquet_schema(name)) for name in _table_names()
    }
    schemas["index"] = _schema_description(index_parquet_schema())
    schemas["catalog"] = _schema_description(catalog_parquet_schema())
    return {"schema_version": "harbor-hf/result-schemas/v1", "tables": schemas}


def parquet_schema(table: TableName) -> pa.Schema:
    return _PARQUET_SCHEMAS[table]


def index_parquet_schema() -> pa.Schema:
    return _INDEX_SCHEMA


def catalog_parquet_schema() -> pa.Schema:
    return _CATALOG_SCHEMA


def _verify_evidence(
    reader: EvidenceReader, source: EvidenceSource
) -> tuple[ResultEvidence, dict[str, JsonValue], str, str, dict[str, Digest]]:
    paths = _verified_listing(reader, source)
    checksums = _load_checksums(reader, source)
    expected_paths = set(paths) - {"_SUCCESS", _CHECKSUMS_PATH}
    if set(checksums) != expected_paths:
        raise ResultPublicationError("evidence checksum manifest is incomplete")
    _verify_object_checksums(reader, source, checksums)
    if source.summary_path not in checksums or source.run_lock_path not in checksums:
        raise ResultPublicationError("evidence omits its summary or immutable run lock")
    summary, lock = _load_summary_and_lock(reader, source)
    if lock.get("run_id") != summary.run.run_id:
        raise ResultPublicationError("evidence summary does not match its run lock")
    if (
        lock.get("evaluation_id") != summary.run.evaluation_id
        or lock.get("publication_role") != summary.run.publication_role
        or lock.get("component_kind") != summary.run.component_kind
    ):
        raise ResultPublicationError(
            "evidence publication role does not match its run lock"
        )
    _validate_summary_tasks_against_lock(summary, lock)
    _verify_artifact_evidence(reader, source, checksums, summary.artifacts)
    source_checksum = _digest(checksums)
    return (
        summary,
        lock,
        source_checksum,
        checksums[source.run_lock_path],
        checksums,
    )


def _validate_summary_tasks_against_lock(
    summary: ResultEvidence, lock: Mapping[str, JsonValue]
) -> None:
    locked = lock.get("benchmark_task_digests")
    attempts = lock.get("attempts")
    if (
        not isinstance(locked, dict)
        or not all(
            isinstance(name, str) and isinstance(digest, str)
            for name, digest in locked.items()
        )
        or not isinstance(attempts, int)
        or attempts < 1
    ):
        raise ResultPublicationError("run lock omits its planned task identities")
    expected = {
        (name, logical_attempt): digest
        for name, digest in locked.items()
        for logical_attempt in range(1, attempts + 1)
    }
    observed = {
        (trial.task_name, trial.logical_attempt): trial.task_digest
        for trial in summary.trials
    }
    if len(observed) != len(summary.trials) or observed != expected:
        raise ResultPublicationError("evidence tasks do not match its run lock")


def _load_publication_provenance(
    reader: EvidenceReader,
    source: EvidenceSource,
    summary: ResultEvidence,
    lock: Mapping[str, JsonValue],
    checksums: Mapping[str, str],
) -> PublicationProvenance:
    envelope_digest = checksums.get(PUBLICATION_ENVELOPE_PATH)
    if envelope_digest is None:
        raise ResultPublicationError("evidence has no canonical publication envelope")
    try:
        envelope = PublicationEnvelope.model_validate_json(
            reader.read_bytes(
                bucket=source.bucket,
                prefix=source.prefix,
                path=PUBLICATION_ENVELOPE_PATH,
            )
        )
    except Exception as error:
        raise ResultPublicationError("publication envelope is invalid") from error
    _validate_envelope_identity(envelope, source, summary, checksums)
    _validate_envelope_executions(envelope, summary)
    manifests = []
    archives = []
    for execution in envelope.executions:
        bundle = execution.harbor_bundle
        if bundle is None:
            continue
        if checksums.get(bundle.manifest.path) != bundle.manifest.digest:
            raise ResultPublicationError(
                "publication envelope has an unverified Harbor bundle"
            )
        if checksums.get(bundle.archive.path) != bundle.archive.digest:
            raise ResultPublicationError(
                "publication envelope has an unverified Harbor archive"
            )
        manifests.append(bundle.manifest.digest)
        archives.append(bundle.archive.digest)
    return PublicationProvenance(
        envelope_sha256=envelope_digest,
        projection_version=envelope.projection_version,
        sanitizer_version=envelope.sanitizer_version,
        execution_profile_sha256=_execution_profile_from_lock(lock),
        harbor_bundle_manifest_sha256s=manifests,
        harbor_archive_sha256s=archives,
    )


def _validate_envelope_identity(
    envelope: PublicationEnvelope,
    source: EvidenceSource,
    summary: ResultEvidence,
    checksums: Mapping[str, str],
) -> None:
    run = summary.run
    expected_prefix = source.prefix.rstrip("/")
    if (
        envelope.run_id != run.run_id
        or envelope.campaign_id != run.campaign_id
        or envelope.evidence_bucket != source.bucket
        or envelope.evidence_prefix != expected_prefix
        or envelope.run_lock.path != source.run_lock_path
        or envelope.run_lock.digest != checksums.get(source.run_lock_path)
    ):
        raise ResultPublicationError("publication envelope identity conflicts")
    runtime = envelope.runtime
    if (
        runtime.provider != run.provider
        or runtime.region != run.region
        or runtime.hardware != run.hardware
        or runtime.accelerator_count != run.accelerator_count
    ):
        raise ResultPublicationError("publication envelope runtime conflicts")


def _validate_envelope_executions(
    envelope: PublicationEnvelope,
    summary: ResultEvidence,
) -> None:
    expected = {
        execution.execution_id: (
            execution.trial_id,
            execution.physical_attempt,
            execution.status,
            execution.failure_category,
            execution.started_at,
            execution.completed_at,
            execution.retry_reason,
            execution.remote_job_id,
        )
        for execution in summary.executions
    }
    observed = {
        execution.execution_id: (
            execution.trial_id,
            execution.physical_attempt,
            execution.status,
            execution.failure_category,
            execution.started_at,
            execution.completed_at,
            execution.retry_reason,
            execution.remote_job_id,
        )
        for execution in envelope.executions
    }
    if observed != expected:
        raise ResultPublicationError(
            "publication envelope executions conflict with normalized evidence"
        )


def _verified_listing(reader: EvidenceReader, source: EvidenceSource) -> list[str]:
    paths = reader.list_files(bucket=source.bucket, prefix=source.prefix)
    if len(paths) != len(set(paths)):
        raise ResultPublicationError("evidence listing contains duplicate paths")
    for path in paths:
        _validate_relative_path(path)
    markers = _TERMINAL_MARKERS.intersection(paths)
    if markers != {"_SUCCESS"}:
        raise ResultPublicationError("evidence is not an exclusively successful run")
    if _CHECKSUMS_PATH not in paths:
        raise ResultPublicationError("evidence has no checksum manifest")
    return paths


def _load_checksums(
    reader: EvidenceReader, source: EvidenceSource
) -> dict[str, Digest]:
    try:
        return _CHECKSUM_MAP.validate_json(
            reader.read_bytes(
                bucket=source.bucket,
                prefix=source.prefix,
                path=_CHECKSUMS_PATH,
            )
        )
    except Exception as error:
        raise ResultPublicationError("evidence checksum manifest is invalid") from error


def _verify_object_checksums(
    reader: EvidenceReader,
    source: EvidenceSource,
    checksums: Mapping[str, str],
) -> None:
    for path, expected in sorted(checksums.items()):
        observed = _sha256_bytes(
            reader.read_bytes(bucket=source.bucket, prefix=source.prefix, path=path)
        )
        if observed != expected:
            raise ResultPublicationError(f"evidence checksum mismatch: {path}")


def _load_summary_and_lock(
    reader: EvidenceReader, source: EvidenceSource
) -> tuple[ResultEvidence, dict[str, JsonValue]]:
    try:
        summary = ResultEvidence.model_validate_json(
            reader.read_bytes(
                bucket=source.bucket,
                prefix=source.prefix,
                path=source.summary_path,
            )
        )
        lock = _JSON_OBJECT.validate_json(
            reader.read_bytes(
                bucket=source.bucket,
                prefix=source.prefix,
                path=source.run_lock_path,
            )
        )
    except Exception as error:
        raise ResultPublicationError(
            "evidence summary or run lock is invalid"
        ) from error
    return summary, lock


def _execution_profile_from_lock(lock: Mapping[str, JsonValue]) -> Digest:
    try:
        model = ModelProfile.model_validate(lock.get("model"))
        deployment = TypeAdapter(DeploymentTarget).validate_python(
            lock.get("deployment")
        )
        agent = AgentProfile.model_validate(lock.get("agent"))
    except Exception as error:
        raise ResultPublicationError(
            "run lock omits a valid execution profile"
        ) from error
    return execution_profile_digest(
        model=model,
        deployment=deployment,
        agent=agent,
    )


def _verify_artifact_evidence(
    reader: EvidenceReader,
    source: EvidenceSource,
    checksums: Mapping[str, str],
    artifacts: list[ArtifactEvidence],
) -> None:
    for artifact in artifacts:
        if checksums.get(artifact.path) != artifact.sha256:
            raise ResultPublicationError("artifact row is not backed by its checksum")
        content = reader.read_bytes(
            bucket=source.bucket,
            prefix=source.prefix,
            path=artifact.path,
        )
        if len(content) != artifact.size_bytes:
            raise ResultPublicationError("artifact row has the wrong evidence size")


def _publication_id(
    run_id: str,
    source: EvidenceSource,
    source_checksum: str,
    lock_checksum: str,
    execution_profile_sha256: str,
) -> str:
    value = {
        "publication_contract": RESULT_PUBLICATION_CONTRACT,
        "run_id": run_id,
        "source_bucket": source.bucket,
        "source_prefix": source.prefix,
        "source_checksum": source_checksum,
        "run_lock_sha256": lock_checksum,
        "execution_profile_sha256": execution_profile_sha256,
    }
    return f"pub-{_digest(value).removeprefix('sha256:')[:32]}"


def _metric_id(run_id: str, record: MetricEvidence) -> str:
    value = {"run_id": run_id, **record.model_dump(mode="json")}
    return f"metric-{_digest(value).removeprefix('sha256:')[:32]}"


def _artifact_id(run_id: str, record: ArtifactEvidence) -> str:
    value = {
        "run_id": run_id,
        "owner_type": record.owner_type,
        "owner_id": record.owner_id,
        "path": record.path,
        "sha256": record.sha256,
    }
    return f"artifact-{_digest(value).removeprefix('sha256:')[:32]}"


def _unique_by_id(
    records: list[TrialEvidence] | list[ExecutionEvidence],
    field: Literal["trial_id", "execution_id"],
    label: str,
) -> dict[str, TrialEvidence | ExecutionEvidence]:
    indexed = {str(getattr(record, field)): record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"result evidence contains a duplicate {label}")
    return indexed


def _require_unique_measurements(records: list[MetricEvidence]) -> None:
    keys = [_measurement_key(record) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("result evidence contains a duplicate metric")


def _require_unique_artifacts(records: list[ArtifactEvidence]) -> None:
    keys = [_artifact_key(record) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("result evidence contains a duplicate artifact")


def _measurement_key(record: MetricEvidence) -> tuple[str, str, str, str, str]:
    return (
        record.owner_type,
        record.owner_id,
        record.name,
        record.unit,
        record.aggregation or "",
    )


def _artifact_key(record: ArtifactEvidence) -> tuple[str, str, str, str, str]:
    return (
        record.owner_type,
        record.owner_id,
        record.kind,
        record.path,
        record.sha256,
    )


def _validate_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("evidence paths must be canonical relative POSIX paths")
    return path


def _is_commit(value: str) -> bool:
    return len(value) in {40, 64} and all(
        character in "0123456789abcdef" for character in value
    )


def _digest(value: object) -> str:
    return _sha256_bytes(_canonical_json(value))


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _table_names() -> tuple[TableName, ...]:
    return ("runs", "trials", "executions", "metrics", "artifacts")


def _table_rows(tables: ResultTables, table: TableName) -> Sequence[FrozenModel]:
    if table == "runs":
        return tables.runs
    if table == "trials":
        return tables.trials
    if table == "executions":
        return tables.executions
    if table == "metrics":
        return tables.metrics
    return tables.artifacts


def _parquet_bytes(rows: Sequence[FrozenModel], schema: pa.Schema) -> bytes:
    payload = [row.model_dump(mode="python") for row in rows]
    table = pa.Table.from_pylist(payload, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        version="2.6",
        data_page_version="2.0",
        write_statistics=True,
    )
    return sink.getvalue().to_pybytes()


def _schema_description(schema: pa.Schema) -> dict[str, object]:
    version = (schema.metadata or {}).get(b"harbor_hf.schema_version", b"").decode()
    return {
        "schema_version": version,
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
    }


def _field(name: str, kind: pa.DataType, *, nullable: bool = False) -> pa.Field:
    return pa.field(name, kind, nullable=nullable)


def _trace_fields() -> list[pa.Field]:
    return [
        _field("schema_version", pa.string()),
        _field("publication_id", pa.string()),
        _field("run_id", pa.string()),
        _field("source_bucket", pa.string()),
        _field("source_prefix", pa.string()),
        _field("source_checksum", pa.string()),
        _field("run_lock_path", pa.string()),
        _field("run_lock_sha256", pa.string()),
        _field("control_commit", pa.string()),
    ]


def _catalog_identity_fields() -> list[pa.Field]:
    return [
        _field("evaluation_id", pa.string()),
        _field("publication_role", pa.string()),
        _field("component_kind", pa.string(), nullable=True),
        _field("source_publication_ids", pa.list_(pa.string())),
        _field("benchmark", pa.string()),
        _field("benchmark_revision", pa.string()),
        _field("result_kind", pa.string()),
        _field("outcome", pa.string()),
        _field("quality", pa.string()),
        _field("created_at", _TIMESTAMP),
        _field("completed_at", _TIMESTAMP),
    ]


def _make_schema(version: str, fields: list[pa.Field]) -> pa.Schema:
    return pa.schema(fields, metadata={b"harbor_hf.schema_version": version.encode()})


_TIMESTAMP = pa.timestamp("us", tz="UTC")
_PARQUET_SCHEMAS: Mapping[TableName, pa.Schema] = {
    "runs": _make_schema(
        "harbor-hf/results/runs/v1",
        [
            *_trace_fields(),
            _field("campaign_id", pa.string()),
            _field("experiment", pa.string()),
            *_catalog_identity_fields(),
            _field("model_id", pa.string()),
            _field("model_repo", pa.string()),
            _field("model_revision", pa.string()),
            _field("deployment_id", pa.string()),
            _field("provider", pa.string()),
            _field("region", pa.string()),
            _field("hardware", pa.string()),
            _field("accelerator_count", pa.int64()),
            _field("agent_id", pa.string()),
            _field("agent_name", pa.string()),
            _field("agent_revision", pa.string()),
            _field("planned_trial_count", pa.int64()),
            _field("scored_trial_count", pa.int64()),
            _field("agent_failed_count", pa.int64()),
            _field("benchmark_failed_count", pa.int64()),
            _field("infrastructure_exhausted_count", pa.int64()),
            _field("unsupported_count", pa.int64()),
            _field("execution_count", pa.int64()),
        ],
    ),
    "trials": _make_schema(
        "harbor-hf/results/trials/v1",
        [
            *_trace_fields(),
            _field("trial_id", pa.string()),
            _field("task_name", pa.string()),
            _field("task_digest", pa.string()),
            _field("logical_attempt", pa.int64()),
            _field("selected_execution_id", pa.string(), nullable=True),
            _field("outcome", pa.string()),
        ],
    ),
    "executions": _make_schema(
        "harbor-hf/results/executions/v1",
        [
            *_trace_fields(),
            _field("execution_id", pa.string()),
            _field("trial_id", pa.string()),
            _field("physical_attempt", pa.int64()),
            _field("runtime_kind", pa.string()),
            _field("status", pa.string()),
            _field("failure_category", pa.string(), nullable=True),
            _field("started_at", _TIMESTAMP),
            _field("completed_at", _TIMESTAMP),
            _field("retry_reason", pa.string(), nullable=True),
            _field("remote_job_id", pa.string(), nullable=True),
        ],
    ),
    "metrics": _make_schema(
        "harbor-hf/results/metrics/v1",
        [
            *_trace_fields(),
            _field("metric_id", pa.string()),
            _field("owner_type", pa.string()),
            _field("owner_id", pa.string()),
            _field("name", pa.string()),
            _field("value", pa.float64()),
            _field("unit", pa.string()),
            _field("aggregation", pa.string(), nullable=True),
        ],
    ),
    "artifacts": _make_schema(
        "harbor-hf/results/artifacts/v1",
        [
            *_trace_fields(),
            _field("artifact_id", pa.string()),
            _field("owner_type", pa.string()),
            _field("owner_id", pa.string()),
            _field("kind", pa.string()),
            _field("path", pa.string()),
            _field("sha256", pa.string()),
            _field("media_type", pa.string()),
            _field("size_bytes", pa.int64()),
        ],
    ),
}

_INDEX_SCHEMA = _make_schema(
    "harbor-hf/results/index/v1",
    [
        _field("schema_version", pa.string()),
        _field("publication_id", pa.string()),
        _field("run_id", pa.string()),
        _field("campaign_id", pa.string()),
        _field("evaluation_id", pa.string()),
        _field("publication_role", pa.string()),
        _field("component_kind", pa.string(), nullable=True),
        _field("benchmark", pa.string()),
        _field("result_kind", pa.string()),
        _field("outcome", pa.string()),
        _field("quality", pa.string()),
        _field("completed_at", _TIMESTAMP),
        _field("model_repo", pa.string()),
        _field("model_revision", pa.string()),
        _field("agent_name", pa.string()),
        _field("agent_revision", pa.string()),
        _field("result_dataset", pa.string()),
        _field("result_revision", pa.string()),
        _field("source_checksum", pa.string()),
        _field("control_commit", pa.string()),
    ],
)

_CATALOG_SCHEMA = _make_schema(
    "harbor-hf/results/catalog/v1",
    [
        _field("schema_version", pa.string()),
        _field("publication_id", pa.string()),
        _field("run_id", pa.string()),
        _field("campaign_id", pa.string()),
        *_catalog_identity_fields(),
        _field("model_repo", pa.string()),
        _field("model_revision", pa.string()),
        _field("agent_name", pa.string()),
        _field("agent_revision", pa.string()),
        _field("provider", pa.string()),
        _field("region", pa.string()),
        _field("hardware", pa.string()),
        _field("accelerator_count", pa.int64()),
        _field("score", pa.float64()),
        _field("passed_trials", pa.int64()),
        _field("planned_trial_count", pa.int64()),
        _field("scored_trial_count", pa.int64()),
        _field("agent_failed_count", pa.int64()),
        _field("benchmark_failed_count", pa.int64()),
        _field("infrastructure_exhausted_count", pa.int64()),
        _field("unsupported_count", pa.int64()),
        _field("execution_count", pa.int64()),
        _field("failed_executions", pa.int64()),
        _field("duration_seconds", pa.float64()),
        _field("result_dataset", pa.string()),
        _field("result_revision", pa.string()),
        _field("source_checksum", pa.string()),
        _field("control_commit", pa.string()),
        _field("projection_path", pa.string()),
        _field("projection_sha256", pa.string()),
        _field("envelope_sha256", pa.string()),
        _field("harbor_bundle_count", pa.int64()),
    ],
)
