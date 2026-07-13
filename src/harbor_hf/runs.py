from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from harbor_hf.models import (
    AgentProfile,
    DeploymentProfile,
    ExperimentSpec,
    ModelProfile,
    RemoteExecutionSpec,
)
from harbor_hf.planner import experiment_digest

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Clock(Protocol):
    def __call__(self) -> datetime: ...


class HasId(Protocol):
    id: str


class RunLock(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "harbor-hf/run-lock/v1alpha1"
    run_id: str
    created_at: datetime
    experiment: str
    spec_digest: str
    benchmark_dataset: str
    benchmark_tasks: list[str]
    model: ModelProfile
    deployment: DeploymentProfile
    agent: AgentProfile
    attempts: int
    concurrent_trials: int
    timeout_seconds: int
    artifact_prefix: str
    remote: RemoteExecutionSpec


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
    clock: Clock = lambda: datetime.now(UTC),
) -> RunLock:
    if spec.remote is None:
        raise ValueError("submit requires a remote configuration")

    model = _select(spec.matrix.models, model_id, "model")
    deployment = _select(spec.matrix.deployments, deployment_id, "deployment")
    agent = _select(spec.matrix.agents, agent_id, "agent")
    if "version" in agent.parameters:
        raise ValueError("agent parameter 'version' is reserved by the run lock")
    if deployment.endpoint is None:
        raise ValueError(
            f"deployment profile {deployment.id} requires an endpoint binding"
        )

    created_at = clock().astimezone(UTC)
    digest = experiment_digest(spec)
    if run_id is not None and _RUN_ID.fullmatch(run_id) is None:
        raise ValueError(
            "run ID must be one safe path component containing only letters, "
            "digits, dots, underscores, or hyphens"
        )
    resolved_id = run_id or _new_run_id(spec.metadata.name, digest, created_at)
    return RunLock(
        run_id=resolved_id,
        created_at=created_at,
        experiment=spec.metadata.name,
        spec_digest=digest,
        benchmark_dataset=spec.benchmark.dataset,
        benchmark_tasks=spec.benchmark.task_names,
        model=model,
        deployment=deployment,
        agent=agent,
        attempts=spec.execution.attempts,
        concurrent_trials=spec.execution.concurrent_trials,
        timeout_seconds=spec.execution.timeout_seconds,
        artifact_prefix=f"runs/{spec.metadata.name}/{resolved_id}",
        remote=spec.remote,
    )


def _new_run_id(name: str, digest: str, created_at: datetime) -> str:
    identity = json.dumps(
        {"name": name, "digest": digest, "created_at": created_at.isoformat()},
        sort_keys=True,
    ).encode()
    suffix = hashlib.sha256(identity).hexdigest()[:10]
    return f"{created_at:%Y%m%dT%H%M%SZ}-{suffix}"
