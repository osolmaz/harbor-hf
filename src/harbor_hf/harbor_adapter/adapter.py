from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from pydantic import JsonValue

from harbor_hf.harbor_adapter.errors import WorkerError
from harbor_hf.harbor_adapter.models import (
    HarborExecutionRequest,
    HarborVerificationPolicy,
    canonical_json_bytes,
    ensure_no_policy_conflicts,
    sha256_digest,
)
from harbor_hf.models import DeploymentProfile, pinned_harbor_dataset_reference
from harbor_hf.provider_models import ProviderTarget
from harbor_hf.providers import routed_provider_model
from harbor_hf.runs import RunLock


@dataclass(frozen=True)
class PreparedHarborExecution:
    request: HarborExecutionRequest
    request_path: Path
    config_path: Path
    command: list[str]


class HarborExecutionAdapter(Protocol):
    def prepare(
        self,
        lock: RunLock,
        execution_root: Path,
        jobs_dir: Path,
        base_url: str,
        harbor_source: Path,
        *,
        task_names: list[str],
        attempts: int,
        concurrency: int,
        expected_task_digests: dict[str, str],
    ) -> PreparedHarborExecution: ...


class FilesystemHarborExecutionAdapter:
    def prepare(
        self,
        lock: RunLock,
        execution_root: Path,
        jobs_dir: Path,
        base_url: str,
        harbor_source: Path,
        *,
        task_names: list[str],
        attempts: int,
        concurrency: int,
        expected_task_digests: dict[str, str],
    ) -> PreparedHarborExecution:
        request = build_execution_request(
            lock,
            jobs_dir,
            base_url,
            task_names=task_names,
            attempts=attempts,
            concurrency=concurrency,
            expected_task_digests=expected_task_digests,
        )
        request_path = execution_root / "harbor-request.json"
        config_path = execution_root / "harbor-job.json"
        _write_new(request_path, request.request_bytes())
        _write_new(config_path, request.config_bytes())
        return PreparedHarborExecution(
            request=request,
            request_path=request_path,
            config_path=config_path,
            command=render_harbor_command(harbor_source, config_path),
        )


def build_execution_request(
    lock: RunLock,
    jobs_dir: Path,
    base_url: str,
    *,
    task_names: list[str],
    attempts: int,
    concurrency: int,
    expected_task_digests: dict[str, str],
) -> HarborExecutionRequest:
    if attempts < 1 or concurrency < 1:
        raise WorkerError("Harbor attempts and concurrency must be positive")
    missing = set(task_names).difference(expected_task_digests)
    if missing:
        raise WorkerError("Harbor request contains a task outside the resolved run set")
    served_model_name = _served_model_name(lock)
    try:
        dataset_reference = pinned_harbor_dataset_reference(
            lock.benchmark_dataset, lock.benchmark_dataset_digest
        )
    except ValueError as error:
        raise WorkerError(str(error)) from error
    dataset_name, dataset_ref = dataset_reference.rsplit("@", 1)
    agent_kwargs = effective_agent_parameters(lock)
    if lock.agent.revision_kind == "package":
        agent_kwargs["version"] = lock.agent.revision
    host = urlparse(base_url).hostname or ""
    config: dict[str, JsonValue] = {
        "jobs_dir": str(jobs_dir),
        "n_attempts": attempts,
        "n_concurrent_trials": concurrency,
        "retry": {"max_retries": 0},
        "environment": {
            "type": lock.remote.harbor.environment,
            "kwargs": {
                "flavor": lock.remote.harbor.sandbox_flavor,
                "job_timeout": lock.remote.harbor.sandbox_idle_timeout_seconds,
            },
        },
        "agents": [
            {
                "name": lock.agent.name,
                "model_name": f"openai/{served_model_name}",
                "n_concurrent": concurrency,
                "extra_allowed_hosts": [host],
                "kwargs": agent_kwargs,
            }
        ],
        "datasets": [
            {
                "name": dataset_name,
                "ref": dataset_ref,
                "task_names": task_names,
            }
        ],
    }
    expected_trials = len(task_names) * attempts
    policy = HarborVerificationPolicy(
        expected_trials=expected_trials,
        expected_task_counts={task: attempts for task in task_names},
        expected_attempts_per_task=attempts,
        expected_task_names=task_names,
        expected_task_digests=expected_task_digests,
        expected_agent_name=lock.agent.name,
        expected_agent_version=_expected_agent_version(lock),
        expected_model_provider="openai",
        expected_model_name=served_model_name,
    )
    ensure_no_policy_conflicts(config, policy)
    return HarborExecutionRequest(
        harbor_revision=lock.remote.harbor.source.revision,
        harbor_config=config,
        harbor_config_digest=sha256_digest(canonical_json_bytes(config)),
        verification=policy,
    )


def render_harbor_command(harbor_source: Path, config_path: Path) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        str(harbor_source),
        "--locked",
        "--no-dev",
        "--extra",
        "hf-sandbox",
        "harbor",
        "run",
        "--config",
        str(config_path),
        "--yes",
    ]


def _served_model_name(lock: RunLock) -> str:
    deployment = lock.deployment
    if isinstance(deployment, ProviderTarget):
        return routed_provider_model(deployment)
    if not isinstance(deployment, DeploymentProfile) or deployment.endpoint is None:
        raise WorkerError("run lock has no endpoint binding")
    return deployment.endpoint.served_model_name


def _expected_agent_version(lock: RunLock) -> str:
    if lock.agent.revision_kind == "package":
        return lock.agent.revision
    assert lock.agent.reported_version is not None
    return lock.agent.reported_version


def effective_agent_parameters(lock: RunLock) -> dict[str, JsonValue]:
    parameters = deepcopy(lock.agent.parameters)
    target = lock.deployment
    if not isinstance(target, ProviderTarget):
        return parameters
    if lock.agent.name != "openclaw":
        raise WorkerError(
            "Inference Provider request controls require the OpenClaw Harbor agent"
        )
    existing = parameters.get("openclaw_config", {})
    if not isinstance(existing, dict):
        raise WorkerError("OpenClaw provider configuration must be a JSON object")
    config = deepcopy(existing)
    models = config.setdefault("models", {})
    if not isinstance(models, dict):
        raise WorkerError("OpenClaw models configuration must be a JSON object")
    providers = models.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise WorkerError("OpenClaw provider configuration must be a JSON object")
    provider = providers.setdefault("openai", {})
    if not isinstance(provider, dict):
        raise WorkerError(
            "OpenClaw OpenAI provider configuration must be a JSON object"
        )
    routed_model = routed_provider_model(target)
    request_parameters = dict(target.parameters)
    request_parameters.update(
        {
            "maxRetries": target.limits.max_attempts - 1,
            "timeoutMs": int(target.timeout_seconds * 1000),
        }
    )
    provider.update(
        {
            "api": "openai-completions",
            "timeoutSeconds": math.ceil(target.timeout_seconds),
            "models": [
                {
                    "id": routed_model,
                    "name": routed_model,
                    "params": request_parameters,
                }
            ],
        }
    )
    parameters["openclaw_config"] = config
    return parameters


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as error:
        raise WorkerError(f"Harbor execution input already exists: {path}") from error
