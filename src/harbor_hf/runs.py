from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harbor_hf.models import (
    AgentProfile,
    BenchmarkJudgeSpec,
    ComponentKind,
    DeploymentTarget,
    EvaluationId,
    ExperimentSpec,
    GitBenchmarkSource,
    ModelProfile,
    PublicationRole,
    RemoteExecutionSpec,
)
from harbor_hf.planner import experiment_digest, resolved_cells
from harbor_hf.provider_models import ProviderTarget

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
RUN_LOCK_V1ALPHA1 = "harbor-hf/run-lock/v1alpha1"
RUN_LOCK_V1ALPHA2 = "harbor-hf/run-lock/v1alpha2"
RUN_LOCK_V1ALPHA3 = "harbor-hf/run-lock/v1alpha3"
_GIT_CREDENTIAL_FILE_ENV = "HARBOR_HF_GIT_CREDENTIAL_FILE"
_GIT_REPOSITORY_ENV = "HARBOR_HF_GIT_REPOSITORY"
_REDACTION_SECRET_FILE_ENV = "HARBOR_HF_REDACTION_SECRET_FILE"


class Clock(Protocol):
    def __call__(self) -> datetime: ...


class HasId(Protocol):
    @property
    def id(self) -> str: ...


class RunLock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[
        "harbor-hf/run-lock/v1alpha1",
        "harbor-hf/run-lock/v1alpha2",
        "harbor-hf/run-lock/v1alpha3",
    ] = RUN_LOCK_V1ALPHA1
    run_id: str
    created_at: datetime
    experiment: str
    evaluation_id: EvaluationId
    publication_role: PublicationRole
    component_kind: ComponentKind | None
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
        if (
            self.schema_version != RUN_LOCK_V1ALPHA3
            and self.benchmark_source is not None
            and self.benchmark_source.credentials is not None
        ):
            raise ValueError(
                f"{self.schema_version} cannot contain authenticated source fields"
            )
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
            RUN_LOCK_V1ALPHA3
            if spec.benchmark.source is not None
            and spec.benchmark.source.credentials is not None
            else (
                RUN_LOCK_V1ALPHA2
                if spec.benchmark.source is not None or spec.benchmark.judge is not None
                else RUN_LOCK_V1ALPHA1
            )
        ),
        run_id=resolved_id,
        created_at=created_at,
        experiment=spec.metadata.name,
        evaluation_id=spec.publishing.evaluation_id,
        publication_role=spec.publishing.role,
        component_kind=spec.publishing.component_kind,
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


@contextmanager
def harbor_process_environment(
    lock: RunLock,
    *,
    token: str,
    inference_base_url: str,
    blocked_secret_names: Iterable[str] = (),
    redaction_secrets: Iterable[str] = (),
) -> Iterator[dict[str, str]]:
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
    blocked = set(blocked_secret_names)
    redaction_values = [value for value in redaction_secrets if value]
    source = lock.benchmark_source
    credential_path: Path | None = None
    redaction_path: Path | None = None
    try:
        if source is not None and source.credentials is not None:
            secret_name = source.credentials.secret_name
            source_token = os.environ.get(secret_name, "")
            if not source_token:
                raise ValueError(f"required secret {secret_name} is not available")
            blocked.add(secret_name)
            redaction_values.append(source_token)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="harbor-hf-git-credential-",
                delete=False,
            ) as stream:
                credential_path = Path(stream.name)
                os.fchmod(stream.fileno(), 0o600)
                stream.write(source_token)
            environment.update(
                {
                    "GIT_CONFIG_COUNT": "5",
                    "GIT_CONFIG_KEY_0": "credential.useHttpPath",
                    "GIT_CONFIG_VALUE_0": "true",
                    "GIT_CONFIG_KEY_1": (
                        f"credential.https://github.com/{source.repository}.git.helper"
                    ),
                    "GIT_CONFIG_VALUE_1": "",
                    "GIT_CONFIG_KEY_2": (
                        f"credential.https://github.com/{source.repository}.git.helper"
                    ),
                    "GIT_CONFIG_VALUE_2": "harbor-hf",
                    "GIT_CONFIG_KEY_3": (
                        f"credential.https://github.com/{source.repository}.helper"
                    ),
                    "GIT_CONFIG_VALUE_3": "",
                    "GIT_CONFIG_KEY_4": (
                        f"credential.https://github.com/{source.repository}.helper"
                    ),
                    "GIT_CONFIG_VALUE_4": "harbor-hf",
                    "GIT_TERMINAL_PROMPT": "0",
                    _GIT_CREDENTIAL_FILE_ENV: str(credential_path),
                    _GIT_REPOSITORY_ENV: source.repository,
                }
            )
        redaction_path = _write_redaction_secrets(redaction_values)
        if redaction_path is not None:
            environment[_REDACTION_SECRET_FILE_ENV] = str(redaction_path)
        for secret_name in blocked:
            environment[secret_name] = ""
        yield environment
    finally:
        if credential_path is not None:
            credential_path.unlink(missing_ok=True)
        if redaction_path is not None:
            redaction_path.unlink(missing_ok=True)


def _write_redaction_secrets(values: list[str]) -> Path | None:
    if not values:
        return None
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="harbor-hf-redaction-secrets-",
        delete=False,
    ) as stream:
        path = Path(stream.name)
        os.fchmod(stream.fileno(), 0o600)
        for value in dict.fromkeys(values):
            stream.write(value + "\n")
    return path


def require_benchmark_source_secret(lock: RunLock) -> None:
    source = lock.benchmark_source
    if source is None or source.credentials is None:
        return
    secret_name = source.credentials.secret_name
    if not os.environ.get(secret_name, ""):
        raise ValueError(f"required secret {secret_name} is not available")


def run_secret_values(lock: RunLock, token: str) -> str | tuple[str, ...]:
    values = [token]
    source = lock.benchmark_source
    if source is not None and source.credentials is not None:
        source_token = os.environ.get(source.credentials.secret_name, "")
        if source_token:
            values.append(source_token)
    unique = tuple(dict.fromkeys(value for value in values if value))
    return unique[0] if len(unique) == 1 else unique


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
