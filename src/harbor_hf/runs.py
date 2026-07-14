from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harbor_hf.models import (
    AgentProfile,
    BenchmarkJudgeSpec,
    DeploymentTarget,
    ExperimentSpec,
    GitBenchmarkSource,
    ModelProfile,
    RemoteExecutionSpec,
)
from harbor_hf.planner import experiment_digest, resolved_cells
from harbor_hf.provider_models import ProviderTarget

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
RUN_LOCK_V1ALPHA1 = "harbor-hf/run-lock/v1alpha1"
RUN_LOCK_V1ALPHA2 = "harbor-hf/run-lock/v1alpha2"


class Clock(Protocol):
    def __call__(self) -> datetime: ...


class HasId(Protocol):
    @property
    def id(self) -> str: ...


class RunLock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[
        "harbor-hf/run-lock/v1alpha1", "harbor-hf/run-lock/v1alpha2"
    ] = RUN_LOCK_V1ALPHA1
    run_id: str
    created_at: datetime
    experiment: str
    spec_digest: str
    benchmark_dataset: str
    benchmark_dataset_digest: str
    benchmark_source: GitBenchmarkSource | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    benchmark_judge: BenchmarkJudgeSpec | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    benchmark_tasks: list[str]
    benchmark_task_digests: dict[str, str]
    model: ModelProfile
    deployment: DeploymentTarget
    agent: AgentProfile
    attempts: int
    concurrent_trials: int
    timeout_seconds: int
    artifact_bucket: str
    artifact_prefix: str
    remote: RemoteExecutionSpec

    @model_validator(mode="after")
    def version_matches_fields(self) -> RunLock:
        if self.schema_version == RUN_LOCK_V1ALPHA1 and (
            self.benchmark_source is not None or self.benchmark_judge is not None
        ):
            raise ValueError("run-lock/v1alpha1 cannot contain source or judge fields")
        return self


def _select[Profile: HasId](
    profiles: list[Profile], profile_id: str | None, dimension: str
) -> Profile:
    if profile_id is not None:
        matches = [profile for profile in profiles if profile.id == profile_id]
        if not matches:
            raise ValueError(f"unknown {dimension} profile: {profile_id}")
        return matches[0]
    if len(profiles) != 1:
        raise ValueError(
            f"submit requires --{dimension} when the matrix has {len(profiles)} "
            f"{dimension} profiles"
        )
    return profiles[0]


def build_run_lock(
    spec: ExperimentSpec,
    *,
    model_id: str | None = None,
    deployment_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    allow_provider: bool = False,
    clock: Clock = lambda: datetime.now(UTC),
) -> RunLock:
    spec = ExperimentSpec.model_validate(spec.model_dump(mode="python"))
    if spec.remote is None:
        raise ValueError("submit requires a remote configuration")

    model = _select(spec.matrix.models, model_id, "model")
    deployment = _select(spec.matrix.deployments, deployment_id, "deployment")
    agent = _select(spec.matrix.agents, agent_id, "agent")
    selected = (model.id, deployment.id, agent.id)
    allowed = {
        (cell.model, cell.deployment, cell.agent) for cell in resolved_cells(spec)
    }
    if selected not in allowed:
        raise ValueError("selected run cell is excluded by matrix rules")
    if "version" in agent.parameters:
        raise ValueError("agent parameter 'version' is reserved by the run lock")
    if (
        agent.revision_kind == "harbor-source"
        and agent.revision != spec.remote.harbor.source.revision
    ):
        raise ValueError("Harbor-source agent revision must match the Harbor source")
    _validate_deployment_target(
        model,
        deployment,
        agent,
        spec.remote,
        allow_provider=allow_provider,
    )

    created_at = clock().astimezone(UTC)
    digest = experiment_digest(spec)
    if run_id is not None and _RUN_ID.fullmatch(run_id) is None:
        raise ValueError(
            "run ID must be one safe path component containing only letters, "
            "digits, dots, underscores, or hyphens, with at most 100 characters"
        )
    resolved_id = run_id or _new_run_id(spec.metadata.name, digest, created_at)
    return RunLock(
        schema_version=(
            RUN_LOCK_V1ALPHA2
            if spec.benchmark.source is not None or spec.benchmark.judge is not None
            else RUN_LOCK_V1ALPHA1
        ),
        run_id=resolved_id,
        created_at=created_at,
        experiment=spec.metadata.name,
        spec_digest=digest,
        benchmark_dataset=spec.benchmark.dataset,
        benchmark_dataset_digest=str(spec.benchmark.dataset_digest),
        benchmark_source=spec.benchmark.source,
        benchmark_judge=spec.benchmark.judge,
        benchmark_tasks=spec.benchmark.task_names,
        benchmark_task_digests=spec.benchmark.task_digests,
        model=model,
        deployment=deployment,
        agent=agent,
        attempts=spec.execution.attempts,
        concurrent_trials=spec.execution.concurrent_trials,
        timeout_seconds=spec.execution.timeout_seconds,
        artifact_bucket=spec.artifacts.bucket,
        artifact_prefix=f"runs/{spec.metadata.name}/{resolved_id}",
        remote=spec.remote,
    )


def harbor_process_environment(
    lock: RunLock, *, token: str, inference_base_url: str
) -> dict[str, str]:
    environment = {
        "HF_TOKEN": token,
        "OPENAI_API_KEY": token,
        "OPENAI_BASE_URL": f"{inference_base_url.rstrip('/')}/v1",
    }
    if lock.benchmark_judge is not None:
        environment.update(
            {
                "AGENT_JUDGE_API_KEY": token,
                "AGENT_JUDGE_API_URL": str(lock.benchmark_judge.api_url),
                "AGENT_JUDGE_MODEL": lock.benchmark_judge.model,
            }
        )
    return environment


def _new_run_id(name: str, digest: str, created_at: datetime) -> str:
    identity = json.dumps(
        {"name": name, "digest": digest, "created_at": created_at.isoformat()},
        sort_keys=True,
    ).encode()
    suffix = hashlib.sha256(identity).hexdigest()[:10]
    return f"{created_at:%Y%m%dT%H%M%SZ}-{suffix}"


def _validate_deployment_target(
    model: ModelProfile,
    deployment: DeploymentTarget,
    agent: AgentProfile,
    remote: RemoteExecutionSpec,
    *,
    allow_provider: bool,
) -> None:
    if isinstance(deployment, ProviderTarget):
        if not allow_provider:
            raise ValueError("Inference Provider targets require campaign execution")
        validate_provider_cell(model, deployment, agent)
        return
    if deployment.endpoint is None:
        if allow_provider:
            raise ValueError(
                "deployment wave requires a pre-existing endpoint binding; "
                "endpoint provisioning is outside this slice"
            )
        raise ValueError(
            f"deployment profile {deployment.id} requires an endpoint binding"
        )
    if deployment.endpoint.namespace != remote.job.namespace:
        raise ValueError(
            "controller Job namespace must match the endpoint namespace for leasing"
        )


def validate_provider_cell(
    model: ModelProfile,
    deployment: ProviderTarget,
    agent: AgentProfile,
) -> None:
    """Validate one resolved Inference Provider matrix cell."""
    if deployment.model != model.repo:
        raise ValueError(
            "Inference Provider target model must match the selected model profile"
        )
    if agent.name != "openclaw":
        raise ValueError("Inference Provider targets require the OpenClaw Harbor agent")
