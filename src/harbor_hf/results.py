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

from harbor_hf.publication_envelope import (
    PUBLICATION_ENVELOPE_PATH,
    PublicationEnvelopeV2,
)

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
EntityId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")]
DatasetId = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"),
]
OwnerType = Literal["run", "trial", "execution"]
RuntimeKind = Literal["endpoint", "provider"]
ArtifactKind = Literal[
    "run_lock",
    "verification",
    "runtime_environment",
    "endpoint_snapshot",
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
}
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_CHECKSUM_MAP = TypeAdapter(dict[str, Digest])


class ResultPublicationError(RuntimeError):
    """Raised when evidence cannot be safely normalized or published."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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
    benchmark: str = Field(min_length=1)
    benchmark_revision: str = Field(min_length=1)
    result_kind: Literal["ordinary"] = "ordinary"
    outcome: Literal["complete"] = "complete"
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
        return self


class TrialEvidence(FrozenModel):
    trial_id: EntityId
    task_name: str = Field(min_length=1)
    task_digest: Digest
    logical_attempt: int = Field(ge=1)
    selected_execution_id: EntityId
    outcome: Literal["complete"] = "complete"


class ExecutionEvidence(FrozenModel):
    execution_id: EntityId
    trial_id: EntityId
    physical_attempt: int = Field(ge=1)
    runtime_kind: RuntimeKind
    status: Literal["succeeded", "failed_infrastructure", "cancelled"]
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
        return self


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
            selected = executions.get(trial.selected_execution_id)
            if (
                not isinstance(selected, ExecutionEvidence)
                or selected.trial_id != trial.trial_id
                or selected.status != "succeeded"
            ):
                raise ValueError("trial selected execution is not a valid success")
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
    benchmark: str = Field(min_length=1)
    benchmark_revision: str = Field(min_length=1)
    result_kind: Literal["ordinary"]
    outcome: Literal["complete"]
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
    trial_count: int = Field(ge=0)
    execution_count: int = Field(ge=0)

    @model_validator(mode="after")
    def completion_follows_creation(self) -> RunRow:
        if self.completed_at < self.created_at:
            raise ValueError("run completion precedes creation")
        return self


class TrialRow(TraceRow):
    schema_version: Literal["harbor-hf/results/trials/v1"] = (
        "harbor-hf/results/trials/v1"
    )
    trial_id: EntityId
    task_name: str = Field(min_length=1)
    task_digest: Digest
    logical_attempt: int = Field(ge=1)
    selected_execution_id: EntityId
    outcome: Literal["complete"]


class ExecutionRow(TraceRow):
    schema_version: Literal["harbor-hf/results/executions/v1"] = (
        "harbor-hf/results/executions/v1"
    )
    execution_id: EntityId
    trial_id: EntityId
    physical_attempt: int = Field(ge=1)
    runtime_kind: RuntimeKind
    status: Literal["succeeded", "failed_infrastructure", "cancelled"]
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
    schema_version: Literal["harbor-hf/result-provenance/v2"] = (
        "harbor-hf/result-provenance/v2"
    )
    envelope_path: Literal["publication-envelope.v2.json"] = PUBLICATION_ENVELOPE_PATH
    envelope_sha256: Digest
    projection_version: str = Field(min_length=1)
    sanitizer_version: str = Field(min_length=1)
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
    provenance: PublicationProvenance | None = None

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
    benchmark: str = Field(min_length=1)
    result_kind: Literal["ordinary"]
    outcome: Literal["complete"]
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
    benchmark: str = Field(min_length=1)
    benchmark_revision: str = Field(min_length=1)
    result_kind: Literal["ordinary"]
    outcome: Literal["complete"]
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
    trial_count: int = Field(ge=0)
    execution_count: int = Field(ge=0)
    infrastructure_failures: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    result_dataset: DatasetId
    result_revision: Commit
    source_checksum: Digest
    control_commit: Commit

    @model_validator(mode="after")
    def values_are_consistent(self) -> CatalogRow:
        if not math.isfinite(self.score):
            raise ValueError("catalog score must be finite")
        if self.passed_trials > self.trial_count:
            raise ValueError("catalog passed trials exceed trial count")
        if self.completed_at < self.created_at:
            raise ValueError("catalog completion precedes creation")
        return self


class CatalogRowV2(CatalogRow):
    schema_version: Literal["harbor-hf/results/catalog/v2"] = (
        "harbor-hf/results/catalog/v2"
    )
    source_format: Literal["native-v2", "legacy-v1"]
    projection_path: str | None = None
    projection_sha256: Digest | None = None
    envelope_sha256: Digest | None = None
    harbor_bundle_count: int = Field(ge=0)

    @model_validator(mode="after")
    def provenance_is_consistent(self) -> CatalogRowV2:
        values = (
            self.projection_path,
            self.projection_sha256,
            self.envelope_sha256,
        )
        if self.source_format == "native-v2":
            if any(value is None for value in values) or self.harbor_bundle_count < 1:
                raise ValueError("native v2 catalog row has incomplete provenance")
            _validate_relative_path(self.projection_path or "")
        elif any(value is not None for value in values) or self.harbor_bundle_count:
            raise ValueError("legacy v1 catalog row cannot claim v2 provenance")
        return self


CatalogEntry = CatalogRow | CatalogRowV2


class DatasetFile(FrozenModel):
    path: str
    content: bytes


class ResultPublication(FrozenModel):
    tables: ResultTables
    files: list[DatasetFile]
    receipt_path: str
    receipt: bytes


class ProjectionFileReference(FrozenModel):
    path: str = Field(min_length=1)
    sha256: Digest
    row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def path_is_relative(self) -> ProjectionFileReference:
        _validate_relative_path(self.path)
        return self


class ResultProjectionV2(FrozenModel):
    schema_version: Literal["harbor-hf/result-projection/v2"] = (
        "harbor-hf/result-projection/v2"
    )
    publication_id: EntityId
    run_id: EntityId
    source_bucket: str = Field(min_length=1)
    source_prefix: str = Field(min_length=1)
    source_checksum: Digest
    control_commit: Commit
    envelope_path: Literal["publication-envelope.v2.json"]
    envelope_sha256: Digest
    projection_version: str = Field(min_length=1)
    sanitizer_version: str = Field(min_length=1)
    harbor_bundle_manifest_sha256s: list[Digest]
    harbor_archive_sha256s: list[Digest]
    tables: dict[TableName, ProjectionFileReference]

    @model_validator(mode="after")
    def tables_are_complete(self) -> ResultProjectionV2:
        if set(self.tables) != set(_table_names()):
            raise ValueError("v2 projection does not reference every query table")
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
    summary, source_checksum, lock_checksum, checksums = _verify_evidence(
        reader, source
    )
    provenance = _load_publication_provenance(reader, source, summary, checksums)
    trace: TraceValues = {
        "publication_id": _publication_id(
            summary.run.run_id,
            source,
            source_checksum,
            lock_checksum,
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
    run = RunRow(
        **trace,
        campaign_id=run_evidence.campaign_id,
        experiment=run_evidence.experiment,
        benchmark=run_evidence.benchmark,
        benchmark_revision=run_evidence.benchmark_revision,
        result_kind=run_evidence.result_kind,
        outcome=run_evidence.outcome,
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
        trial_count=len(summary.trials),
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


def build_result_publication(tables: ResultTables) -> ResultPublication:
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
    if tables.provenance is not None:
        files.append(_build_projection_file(tables, files))
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


def _build_projection_file(
    tables: ResultTables, table_files: list[DatasetFile]
) -> DatasetFile:
    provenance = tables.provenance
    if provenance is None:
        raise ValueError("v2 projection requires publication provenance")
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
    projection = ResultProjectionV2(
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
        harbor_bundle_manifest_sha256s=(provenance.harbor_bundle_manifest_sha256s),
        harbor_archive_sha256s=provenance.harbor_archive_sha256s,
        tables=references,
    )
    return DatasetFile(
        path=f"projections/schema=v2/{tables.publication_id}.json",
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
        benchmark=run.benchmark,
        result_kind=run.result_kind,
        outcome=run.outcome,
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
    tables: ResultTables, *, result_dataset: str, result_revision: str
) -> CatalogRow:
    run = tables.runs[0]
    rewards = _trial_reward_scores(tables)
    score = sum(rewards) / len(rewards) if rewards else 0.0
    return CatalogRow(
        publication_id=tables.publication_id,
        run_id=run.run_id,
        campaign_id=run.campaign_id,
        benchmark=run.benchmark,
        benchmark_revision=run.benchmark_revision,
        result_kind=run.result_kind,
        outcome=run.outcome,
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
        trial_count=run.trial_count,
        execution_count=run.execution_count,
        infrastructure_failures=sum(
            execution.status == "failed_infrastructure"
            for execution in tables.executions
        ),
        duration_seconds=(run.completed_at - run.created_at).total_seconds(),
        result_dataset=result_dataset,
        result_revision=result_revision,
        source_checksum=run.source_checksum,
        control_commit=run.control_commit,
    )


def build_catalog_window_file(rows: Sequence[CatalogRow], size: int) -> DatasetFile:
    if size < 1:
        raise ValueError("catalog window size must be positive")
    return DatasetFile(
        path=f"data/catalog/schema=v1/windows/{size:04d}.parquet",
        content=_parquet_bytes(rows[:size], catalog_parquet_schema()),
    )


def build_catalog_row_v2(
    tables: ResultTables,
    *,
    result_dataset: str,
    result_revision: str,
    projection: DatasetFile | None,
) -> CatalogRowV2:
    base = build_catalog_row(
        tables,
        result_dataset=result_dataset,
        result_revision=result_revision,
    )
    values = base.model_dump(mode="python", exclude={"schema_version"})
    provenance = tables.provenance
    if provenance is None:
        if projection is not None:
            raise ValueError("legacy result cannot claim a v2 projection")
        return CatalogRowV2(
            **values,
            source_format="legacy-v1",
            harbor_bundle_count=0,
        )
    if projection is None:
        raise ValueError("native v2 result has no projection manifest")
    return CatalogRowV2(
        **values,
        source_format="native-v2",
        projection_path=projection.path,
        projection_sha256=_sha256_bytes(projection.content),
        envelope_sha256=provenance.envelope_sha256,
        harbor_bundle_count=len(provenance.harbor_archive_sha256s),
    )


def catalog_v2_from_legacy(row: CatalogRow) -> CatalogRowV2:
    return CatalogRowV2(
        **row.model_dump(mode="python", exclude={"schema_version"}),
        source_format="legacy-v1",
        harbor_bundle_count=0,
    )


def build_catalog_v2_window_file(
    rows: Sequence[CatalogRowV2], size: int
) -> DatasetFile:
    if size < 1:
        raise ValueError("catalog window size must be positive")
    return DatasetFile(
        path=f"data/catalog/schema=v2/windows/{size:04d}.parquet",
        content=_parquet_bytes(rows[:size], catalog_v2_parquet_schema()),
    )


def catalog_lookup_path(run_id: str) -> str:
    identity = hashlib.sha256(run_id.encode()).hexdigest()
    return f"data/catalog/schema=v1/runs/{identity}.parquet"


def catalog_v2_lookup_path(run_id: str) -> str:
    identity = hashlib.sha256(run_id.encode()).hexdigest()
    return f"data/catalog/schema=v2/runs/{identity}.parquet"


def build_catalog_lookup_file(row: CatalogRow) -> DatasetFile:
    return DatasetFile(
        path=catalog_lookup_path(row.run_id),
        content=_parquet_bytes([row], catalog_parquet_schema()),
    )


def build_catalog_v2_lookup_file(row: CatalogRowV2) -> DatasetFile:
    return DatasetFile(
        path=catalog_v2_lookup_path(row.run_id),
        content=_parquet_bytes([row], catalog_v2_parquet_schema()),
    )


def read_catalog_file(content: bytes) -> list[CatalogRow]:
    try:
        values = pq.read_table(
            pa.BufferReader(content), schema=catalog_parquet_schema()
        )
    except (pa.ArrowException, OSError) as error:
        raise ValueError("result catalog Parquet is invalid") from error
    return [CatalogRow.model_validate(value) for value in values.to_pylist()]


def read_catalog_v2_file(content: bytes) -> list[CatalogRowV2]:
    try:
        values = pq.read_table(
            pa.BufferReader(content), schema=catalog_v2_parquet_schema()
        )
    except (pa.ArrowException, OSError) as error:
        raise ValueError("v2 result catalog Parquet is invalid") from error
    return [CatalogRowV2.model_validate(value) for value in values.to_pylist()]


def _trial_reward_scores(tables: ResultTables) -> list[float]:
    by_trial: dict[str, list[MetricRow]] = {}
    for metric in tables.metrics:
        if metric.owner_type == "trial" and metric.unit == "score":
            by_trial.setdefault(metric.owner_id, []).append(metric)
    return [
        score
        for trial in tables.trials
        if (score := _select_reward_score(by_trial.get(trial.trial_id, []))) is not None
    ]


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


def catalog_v2_parquet_schema() -> pa.Schema:
    return _CATALOG_V2_SCHEMA


def _verify_evidence(
    reader: EvidenceReader, source: EvidenceSource
) -> tuple[ResultEvidence, str, str, dict[str, Digest]]:
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
    _verify_artifact_evidence(reader, source, checksums, summary.artifacts)
    source_checksum = _digest(checksums)
    return summary, source_checksum, checksums[source.run_lock_path], checksums


def _load_publication_provenance(
    reader: EvidenceReader,
    source: EvidenceSource,
    summary: ResultEvidence,
    checksums: Mapping[str, str],
) -> PublicationProvenance | None:
    envelope_digest = checksums.get(PUBLICATION_ENVELOPE_PATH)
    if envelope_digest is None:
        return None
    try:
        envelope = PublicationEnvelopeV2.model_validate_json(
            reader.read_bytes(
                bucket=source.bucket,
                prefix=source.prefix,
                path=PUBLICATION_ENVELOPE_PATH,
            )
        )
    except Exception as error:
        raise ResultPublicationError("publication envelope is invalid") from error
    _validate_envelope_identity(envelope, source, summary, checksums)
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
        harbor_bundle_manifest_sha256s=manifests,
        harbor_archive_sha256s=archives,
    )


def _validate_envelope_identity(
    envelope: PublicationEnvelopeV2,
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
) -> str:
    value = {
        "run_id": run_id,
        "source_bucket": source.bucket,
        "source_prefix": source.prefix,
        "source_checksum": source_checksum,
        "run_lock_sha256": lock_checksum,
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
            _field("benchmark", pa.string()),
            _field("benchmark_revision", pa.string()),
            _field("result_kind", pa.string()),
            _field("outcome", pa.string()),
            _field("created_at", _TIMESTAMP),
            _field("completed_at", _TIMESTAMP),
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
            _field("trial_count", pa.int64()),
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
            _field("selected_execution_id", pa.string()),
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
        _field("benchmark", pa.string()),
        _field("result_kind", pa.string()),
        _field("outcome", pa.string()),
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
        _field("benchmark", pa.string()),
        _field("benchmark_revision", pa.string()),
        _field("result_kind", pa.string()),
        _field("outcome", pa.string()),
        _field("created_at", _TIMESTAMP),
        _field("completed_at", _TIMESTAMP),
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
        _field("trial_count", pa.int64()),
        _field("execution_count", pa.int64()),
        _field("infrastructure_failures", pa.int64()),
        _field("duration_seconds", pa.float64()),
        _field("result_dataset", pa.string()),
        _field("result_revision", pa.string()),
        _field("source_checksum", pa.string()),
        _field("control_commit", pa.string()),
    ],
)

_CATALOG_V2_SCHEMA = _make_schema(
    "harbor-hf/results/catalog/v2",
    [
        *list(_CATALOG_SCHEMA),
        _field("source_format", pa.string()),
        _field("projection_path", pa.string(), nullable=True),
        _field("projection_sha256", pa.string(), nullable=True),
        _field("envelope_sha256", pa.string(), nullable=True),
        _field("harbor_bundle_count", pa.int64()),
    ],
)
