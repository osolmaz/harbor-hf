from __future__ import annotations

import hashlib
import json
import re
from fnmatch import fnmatch
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from harbor_hf.evidence import is_sensitive_key
from harbor_hf.provider_models import ProviderTarget

ProfileId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")]
EvaluationId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
PublicationRole = Literal["final", "component", "diagnostic"]
ComponentKind = Literal["base", "correction"]
TaskName = Annotated[str, Field(min_length=1)]
ContentDigest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
GitHubRepository = Annotated[
    str,
    Field(
        pattern=(
            r"^(?:https://github\.com/)?"
            r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/"
            r"[A-Za-z0-9_.-]+(?:\.git)?$"
        )
    ),
]
_CONTROLLER_HEADROOM_SECONDS = 4800
_HARBOR_PACKAGE_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Metadata(StrictModel):
    name: ProfileId
    labels: dict[str, str] = Field(default_factory=dict)


class GitHubTokenCredentials(StrictModel):
    type: Literal["github-token"] = "github-token"
    secret_name: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,120}_TOKEN$")

    @field_validator("secret_name")
    @classmethod
    def secret_is_separate_from_hugging_face_token(cls, value: str) -> str:
        if value == "HF_TOKEN":
            raise ValueError("GitHub credentials must not reuse HF_TOKEN")
        return value


class GitBenchmarkSource(StrictModel):
    type: Literal["git"] = "git"
    repository: GitHubRepository
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str = Field(min_length=1)
    credentials: GitHubTokenCredentials | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("repository", mode="before")
    @classmethod
    def canonicalize_repository(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        repository = value.removeprefix("https://github.com/")
        return repository.removesuffix(".git")

    @field_validator("path")
    @classmethod
    def path_is_safely_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != value
            or value in {"", "."}
        ):
            raise ValueError("Git benchmark path must be safely relative")
        return value


def git_benchmark_source_digest(source: GitBenchmarkSource) -> str:
    payload = json.dumps(
        source.model_dump(mode="json", exclude={"credentials"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class BenchmarkJudgeSpec(StrictModel):
    protocol: Literal["openai-compatible"] = "openai-compatible"
    api_url: AnyHttpUrl
    model: str = Field(min_length=1)
    api_key_secret_name: Literal["HF_TOKEN"] = "HF_TOKEN"

    @field_validator("api_url")
    @classmethod
    def api_url_is_secure(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if (
            value.scheme != "https"
            or value.username is not None
            or value.password
            or value.query is not None
            or value.fragment is not None
            or value.host != "router.huggingface.co"
            or value.port != 443
        ):
            raise ValueError(
                "benchmark judge API URL must be credential-free HTTPS on the "
                "trusted Hugging Face router"
            )
        return value

    @field_validator("model")
    @classmethod
    def model_is_canonical(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("benchmark judge model cannot have surrounding whitespace")
        return value


class BenchmarkSpec(StrictModel):
    dataset: str = Field(min_length=1)
    dataset_digest: ContentDigest | None = None
    source: GitBenchmarkSource | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    judge: BenchmarkJudgeSpec | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    task_names: list[TaskName] = Field(default_factory=lambda: ["*"], min_length=1)
    task_digests: dict[TaskName, ContentDigest] = Field(default_factory=dict)

    @model_validator(mode="after")
    def benchmark_contract_is_consistent(self) -> BenchmarkSpec:
        if len(self.task_names) != len(set(self.task_names)):
            raise ValueError("benchmark task names must be unique")
        if self.source is not None:
            if "@" in self.dataset:
                raise ValueError(
                    "Git-backed benchmark dataset names cannot contain a reference"
                )
            source_digest = git_benchmark_source_digest(self.source)
            if self.dataset_digest is not None and self.dataset_digest != source_digest:
                raise ValueError(
                    "benchmark dataset digest must match its immutable Git source"
                )
            self.dataset_digest = source_digest
            return self
        _, reference = _split_dataset_reference(self.dataset)
        if reference is not None and reference.startswith("sha256:"):
            if _SHA256_DIGEST.fullmatch(reference) is None:
                raise ValueError(
                    "benchmark dataset content address must be a full sha256 digest"
                )
            if self.dataset_digest is not None and reference != self.dataset_digest:
                raise ValueError(
                    "benchmark dataset digest must match its "
                    "content-addressed reference"
                )
            self.dataset_digest = reference
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
    command: list[str] = Field(default_factory=list)
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


DeploymentTarget = DeploymentProfile | ProviderTarget


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
        ambiguous_keys = [
            key
            for key in self.parameters
            if not key or key != key.strip() or "=" in key
        ]
        if ambiguous_keys:
            raise ValueError(
                "agent parameter keys must not be empty, contain '=', or have "
                "surrounding whitespace"
            )
        _reject_sensitive_parameters(self.parameters, "agent")
        return self


class MatrixRule(StrictModel):
    models: list[ProfileId] = Field(default_factory=list)
    deployments: list[ProfileId] = Field(default_factory=list)
    agents: list[ProfileId] = Field(default_factory=list)

    @model_validator(mode="after")
    def selects_at_least_one_dimension(self) -> MatrixRule:
        if not (self.models or self.deployments or self.agents):
            raise ValueError("matrix rules must select at least one dimension")
        for dimension, values in (
            ("models", self.models),
            ("deployments", self.deployments),
            ("agents", self.agents),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"matrix rule {dimension} must be unique")
        return self


class MatrixSpec(StrictModel):
    models: list[ModelProfile] = Field(min_length=1)
    deployments: list[DeploymentTarget] = Field(min_length=1)
    agents: list[AgentProfile] = Field(min_length=1)
    include: list[MatrixRule] = Field(default_factory=list)
    exclude: list[MatrixRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def profile_ids_are_unique(self) -> MatrixSpec:
        for profiles in (self.models, self.deployments, self.agents):
            ids = [profile.id for profile in profiles]
            if len(ids) != len(set(ids)):
                raise ValueError(
                    "profile IDs must be unique within each matrix dimension"
                )
        known = {
            "models": {profile.id for profile in self.models},
            "deployments": {profile.id for profile in self.deployments},
            "agents": {profile.id for profile in self.agents},
        }
        for rule in [*self.include, *self.exclude]:
            for dimension in ("models", "deployments", "agents"):
                unknown = set(getattr(rule, dimension)) - known[dimension]
                if unknown:
                    raise ValueError(
                        f"matrix rule references unknown {dimension}: "
                        + ", ".join(sorted(unknown))
                    )
        return self


class ServingProfileBinding(StrictModel):
    profile_id: ProfileId
    profile_sha256: ContentDigest
    artifact_uri: str = Field(pattern=r"^hf://buckets/[^\s]+$")
    concurrency: int = Field(ge=1)
    model_sha256: ContentDigest
    deployment_sha256: ContentDigest
    agent_sha256: ContentDigest
    benchmark_sha256: ContentDigest
    server_context_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)


class ExecutionSpec(StrictModel):
    attempts: int = Field(default=1, ge=1)
    concurrent_trials: int = Field(default=1, ge=1)
    max_trials_per_shard: int = Field(default=64, ge=1)
    max_shards_per_wave: int = Field(default=8, ge=1)
    timeout_seconds: int = Field(default=3600, ge=1)
    server_context_tokens: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    max_output_tokens: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    reasoning_required: bool = Field(
        default=False, exclude_if=lambda value: value is False
    )
    serving_profile: ServingProfileBinding | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ArtifactStoreSpec(StrictModel):
    bucket: str = Field(min_length=1)


class PublishingSpec(StrictModel):
    dataset: str = Field(min_length=1)
    index_dataset: str | None = None
    evaluation_id: EvaluationId
    role: PublicationRole
    component_kind: ComponentKind | None = None

    @model_validator(mode="after")
    def datasets_are_distinct(self) -> PublishingSpec:
        if self.index_dataset is not None and self.index_dataset == self.dataset:
            raise ValueError(
                "publishing.index_dataset must differ from publishing.dataset"
            )
        if (self.role == "component") != (self.component_kind is not None):
            raise ValueError(
                "publishing.component_kind is required only for component runs"
            )
        return self


class RemoteJobSpec(StrictModel):
    namespace: str = Field(min_length=1)
    image: str = Field(
        pattern=r"^.+@sha256:[0-9a-f]{64}$",
    )
    flavor: str = Field(default="cpu-basic", min_length=1)
    timeout_seconds: int = Field(default=10800, ge=1, le=85800)
    token_secret_name: Literal["HF_TOKEN"] = "HF_TOKEN"


class SourcePin(StrictModel):
    repository: GitHubRepository
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
        _validate_serving_profile_binding(self)
        if self.remote is None:
            return self
        _validate_remote_input_pins(self)
        if (
            self.remote.job.timeout_seconds
            < self.execution.timeout_seconds + _CONTROLLER_HEADROOM_SECONDS
        ):
            raise ValueError(
                "remote Job timeout must exceed execution timeout by at least "
                f"{_CONTROLLER_HEADROOM_SECONDS} seconds"
            )
        if (
            self.remote.harbor.sandbox_idle_timeout_seconds
            > self.remote.job.timeout_seconds
        ):
            raise ValueError("HF Sandbox timeout must not exceed remote Job timeout")
        return self


def _validate_serving_profile_binding(spec: ExperimentSpec) -> None:
    binding = spec.execution.serving_profile
    if binding is None:
        return
    if spec.execution.concurrent_trials != binding.concurrency:
        raise ValueError(
            "execution concurrent_trials must match the selected serving profile"
        )
    if any(
        len(profiles) != 1
        for profiles in (
            spec.matrix.models,
            spec.matrix.deployments,
            spec.matrix.agents,
        )
    ):
        raise ValueError("serving profile binding requires one resolved matrix cell")
    _validate_binding_identity(spec, binding)
    _validate_binding_token_limits(spec, binding)


def _validate_binding_identity(
    spec: ExperimentSpec, binding: ServingProfileBinding
) -> None:
    expected = {
        "model_sha256": _canonical_profile_digest(spec.matrix.models[0]),
        "deployment_sha256": _canonical_profile_digest(spec.matrix.deployments[0]),
        "agent_sha256": _canonical_profile_digest(spec.matrix.agents[0]),
        "benchmark_sha256": _canonical_profile_digest(
            spec.benchmark.model_dump(mode="json", exclude_none=True)
        ),
    }
    for field, value in expected.items():
        if getattr(binding, field) != value:
            raise ValueError(f"serving profile {field} does not match the experiment")


def _validate_binding_token_limits(
    spec: ExperimentSpec, binding: ServingProfileBinding
) -> None:
    for key in ("server_context_tokens", "max_output_tokens"):
        value = getattr(spec.execution, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"execution {key} must be a positive integer")
        if getattr(binding, key) != value:
            raise ValueError(f"serving profile {key} does not match execution")


def _canonical_profile_digest(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _validate_remote_input_pins(spec: ExperimentSpec) -> None:
    if spec.benchmark.source is None:
        pinned_harbor_dataset_reference(
            spec.benchmark.dataset, spec.benchmark.dataset_digest
        )
    _validate_task_pins(spec.benchmark)
    if any(
        re.fullmatch(r"[0-9a-f]{40}", model.revision) is None
        for model in spec.matrix.models
    ):
        raise ValueError("remote model revisions must be full Git commit IDs")
    endpoint_deployments = [
        deployment
        for deployment in spec.matrix.deployments
        if isinstance(deployment, DeploymentProfile)
    ]
    if any(
        re.fullmatch(r".+@sha256:[0-9a-f]{64}", deployment.engine.image) is None
        for deployment in endpoint_deployments
    ):
        raise ValueError("remote serving images must be pinned by sha256 digest")
    if any(not _is_immutable_agent_revision(agent) for agent in spec.matrix.agents):
        raise ValueError("remote agent revisions must be immutable")


def _validate_task_pins(benchmark: BenchmarkSpec) -> None:
    if not benchmark.task_digests:
        raise ValueError("remote benchmarks require resolved task digests")
    unmatched_selections = [
        selection
        for selection in benchmark.task_names
        if not any(fnmatch(task, selection) for task in benchmark.task_digests)
    ]
    unmatched_tasks = [
        task
        for task in benchmark.task_digests
        if not any(fnmatch(task, selection) for selection in benchmark.task_names)
    ]
    if unmatched_selections or unmatched_tasks:
        raise ValueError("remote task digests must exactly resolve the task selection")


def pinned_harbor_dataset_reference(dataset: str, dataset_digest: str | None) -> str:
    """Return the exact content-addressed dataset reference Harbor must execute."""
    if dataset_digest is not None and _SHA256_DIGEST.fullmatch(dataset_digest) is None:
        raise ValueError("benchmark dataset digest must be a full sha256 digest")
    name, reference = _split_dataset_reference(dataset)
    if _HARBOR_PACKAGE_NAME.fullmatch(name) is None or ".." in name:
        raise ValueError(
            "remote benchmark dataset must use a Harbor package name in org/name form"
        )
    if reference is not None and reference.startswith("sha256:"):
        if _SHA256_DIGEST.fullmatch(reference) is None:
            raise ValueError(
                "benchmark dataset content address must be a full sha256 digest"
            )
        if dataset_digest is not None and reference != dataset_digest:
            raise ValueError(
                "benchmark dataset digest must match its content-addressed reference"
            )
        return dataset
    if dataset_digest is None:
        raise ValueError("remote benchmark dataset requires an immutable sha256 digest")
    return f"{name}@{dataset_digest}"


def _split_dataset_reference(dataset: str) -> tuple[str, str | None]:
    name, separator, reference = dataset.rpartition("@")
    if not separator:
        return dataset, None
    return name, reference


def _is_immutable_agent_revision(agent: AgentProfile) -> bool:
    if agent.revision_kind == "harbor-source":
        return re.fullmatch(r"[0-9a-f]{40}", agent.revision) is not None
    return (
        re.fullmatch(
            r"v?[0-9]+(?:\.[0-9]+)*(?:[-+][0-9A-Za-z.-]+)?",
            agent.revision,
        )
        is not None
    )


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
