from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from harbor_hf.evidence import is_sensitive_key

ProfileId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")]
TaskName = Annotated[str, Field(min_length=1)]
_CONTROLLER_HEADROOM_SECONDS = 4800


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Metadata(StrictModel):
    name: ProfileId
    labels: dict[str, str] = Field(default_factory=dict)


class BenchmarkSpec(StrictModel):
    dataset: str = Field(min_length=1)
    task_names: list[TaskName] = Field(default_factory=lambda: ["*"], min_length=1)

    @model_validator(mode="after")
    def task_names_are_unique(self) -> BenchmarkSpec:
        if len(self.task_names) != len(set(self.task_names)):
            raise ValueError("benchmark task names must be unique")
        return self


class QuantizationSpec(StrictModel):
    method: str = Field(min_length=1)
    scheme: str = Field(min_length=1)


class WeightsSpec(StrictModel):
    format: str = Field(min_length=1)
    quantization: QuantizationSpec | None = None


class ModelProfile(StrictModel):
    id: ProfileId
    repo: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    weights: WeightsSpec


class EngineSpec(StrictModel):
    name: str = Field(min_length=1)
    image: str = Field(min_length=1)
    arguments: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    secret_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def environment_contains_no_inline_secrets(self) -> EngineSpec:
        declared = set(self.secret_names)
        inline = [
            key for key in self.environment if key in declared or is_sensitive_key(key)
        ]
        if inline:
            raise ValueError(
                "engine environment must not contain inline secret values: "
                + ", ".join(sorted(inline))
            )
        return self


class EndpointRef(StrictModel):
    namespace: str = Field(min_length=1)
    name: ProfileId
    served_model_name: str = Field(min_length=1)


class DeploymentProfile(StrictModel):
    id: ProfileId
    provider: Literal["hf-inference-endpoints"] = "hf-inference-endpoints"
    hardware: str = Field(min_length=1)
    accelerator_count: int = Field(default=1, ge=1)
    region: str = Field(min_length=1)
    engine: EngineSpec
    endpoint: EndpointRef | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def parameters_contain_no_inline_secrets(self) -> DeploymentProfile:
        _reject_sensitive_parameters(self.parameters, "deployment")
        return self


class AgentProfile(StrictModel):
    id: ProfileId
    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    revision_kind: Literal["package", "harbor-source"]
    reported_version: str | None = Field(default=None, min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def revision_metadata_is_consistent(self) -> AgentProfile:
        if self.revision_kind == "package" and self.reported_version is not None:
            raise ValueError("package agents report their package revision")
        if self.revision_kind == "harbor-source" and self.reported_version is None:
            raise ValueError("Harbor-source agents require reported_version")
        _reject_sensitive_parameters(self.parameters, "agent")
        return self


class MatrixSpec(StrictModel):
    models: list[ModelProfile] = Field(min_length=1)
    deployments: list[DeploymentProfile] = Field(min_length=1)
    agents: list[AgentProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def profile_ids_are_unique(self) -> MatrixSpec:
        for profiles in (self.models, self.deployments, self.agents):
            ids = [profile.id for profile in profiles]
            if len(ids) != len(set(ids)):
                raise ValueError(
                    "profile IDs must be unique within each matrix dimension"
                )
        return self


class ExecutionSpec(StrictModel):
    attempts: int = Field(default=1, ge=1)
    concurrent_trials: int = Field(default=1, ge=1)
    timeout_seconds: int = Field(default=3600, ge=1)


class ArtifactStoreSpec(StrictModel):
    bucket: str = Field(min_length=1)


class PublishingSpec(StrictModel):
    dataset: str = Field(min_length=1)
    index_dataset: str | None = None


class RemoteJobSpec(StrictModel):
    namespace: str = Field(min_length=1)
    image: str = Field(
        pattern=r"^.+@sha256:[0-9a-f]{64}$",
    )
    flavor: str = Field(default="cpu-basic", min_length=1)
    timeout_seconds: int = Field(default=10800, ge=1, le=85800)
    token_secret_name: Literal["HF_TOKEN"] = "HF_TOKEN"


class SourcePin(StrictModel):
    repository: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class HarborRuntimeSpec(StrictModel):
    source: SourcePin
    environment: Literal["hf-sandbox"] = "hf-sandbox"
    sandbox_flavor: str = Field(default="cpu-basic", min_length=1)
    sandbox_idle_timeout_seconds: int = Field(default=3600, ge=1, le=86400)


class RemoteExecutionSpec(StrictModel):
    job: RemoteJobSpec
    worker: SourcePin
    harbor: HarborRuntimeSpec


class ExperimentSpec(StrictModel):
    api_version: Literal["harbor-hf/v1alpha1"]
    kind: Literal["Experiment"]
    metadata: Metadata
    benchmark: BenchmarkSpec
    matrix: MatrixSpec
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    artifacts: ArtifactStoreSpec
    publishing: PublishingSpec
    remote: RemoteExecutionSpec | None = None

    @model_validator(mode="after")
    def remote_job_has_lifecycle_headroom(self) -> ExperimentSpec:
        if (
            self.remote is not None
            and self.remote.job.timeout_seconds
            < self.execution.timeout_seconds + _CONTROLLER_HEADROOM_SECONDS
        ):
            raise ValueError(
                "remote Job timeout must exceed execution timeout by at least "
                f"{_CONTROLLER_HEADROOM_SECONDS} seconds"
            )
        return self


def _reject_sensitive_parameters(value: JsonValue, owner: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if is_sensitive_key(key):
                raise ValueError(
                    f"{owner} parameters must not contain secret-like keys"
                )
            _reject_sensitive_parameters(item, owner)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive_parameters(item, owner)
