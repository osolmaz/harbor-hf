from __future__ import annotations

import math
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from pydantic import JsonValue

from harbor_hf.harbor_adapter.errors import HarborTrialFailure, WorkerError
from harbor_hf.harbor_adapter.models import (
    HarborExecutionRequest,
    HarborVerificationPolicy,
    HarborVerificationResult,
    canonical_json_bytes,
    ensure_no_policy_conflicts,
    sha256_digest,
)
from harbor_hf.harbor_adapter.validation import (
    load_compatibility_bundle,
    validate_compatibility_bundle,
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


@dataclass(frozen=True)
class HarborExecutionOutcome:
    exit_code: int
    verification: HarborVerificationResult | None
    compatibility_path: Path | None


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

    def execute(
        self,
        prepared: PreparedHarborExecution,
        harbor_source: Path,
        jobs_dir: Path,
        log_path: Path,
        *,
        environment: dict[str, str],
        timeout_seconds: int,
        stream_runner: Callable[..., int],
        monotonic: Callable[[], float] = time.monotonic,
        deadline: float | None = None,
    ) -> HarborExecutionOutcome: ...


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

    def execute(
        self,
        prepared: PreparedHarborExecution,
        harbor_source: Path,
        jobs_dir: Path,
        log_path: Path,
        *,
        environment: dict[str, str],
        timeout_seconds: int,
        stream_runner: Callable[..., int],
        monotonic: Callable[[], float] = time.monotonic,
        deadline: float | None = None,
    ) -> HarborExecutionOutcome:
        shared_deadline = _shared_deadline(timeout_seconds, deadline, monotonic)
        self._validate_inputs(prepared)
        try:
            exit_code = stream_runner(
                prepared.command,
                log_path,
                environment=environment,
                timeout_seconds=timeout_seconds,
            )
        finally:
            self._validate_inputs(prepared)
        has_results = any(jobs_dir.glob("*/*/result.json"))
        if exit_code != 0 and not has_results:
            return HarborExecutionOutcome(exit_code, None, None)
        export_timeout = _remaining_export_timeout(shared_deadline, monotonic)
        compatibility_path = prepared.request_path.with_name(
            "harbor-compatibility.json"
        )
        export_log = prepared.request_path.with_name("harbor-export.log")
        request_digest = sha256_digest(
            canonical_json_bytes(prepared.request.model_dump(mode="json"))
        )
        exported = self._export_compatibility(
            prepared,
            harbor_source,
            jobs_dir,
            compatibility_path,
            export_log,
            request_digest,
            environment,
            export_timeout,
            exit_code,
            stream_runner,
        )
        if not exported:
            return HarborExecutionOutcome(exit_code, None, None)
        try:
            bundle = load_compatibility_bundle(compatibility_path, prepared.request)
            verification = validate_compatibility_bundle(bundle, prepared.request)
        except HarborTrialFailure:
            raise
        except (OSError, ValueError, RuntimeError):
            if exit_code != 0:
                return HarborExecutionOutcome(exit_code, None, None)
            raise
        return HarborExecutionOutcome(exit_code, verification, compatibility_path)

    def _export_compatibility(
        self,
        prepared: PreparedHarborExecution,
        harbor_source: Path,
        jobs_dir: Path,
        compatibility_path: Path,
        export_log: Path,
        request_digest: str,
        environment: dict[str, str],
        export_timeout: int,
        harbor_exit: int,
        stream_runner: Callable[..., int],
    ) -> bool:
        try:
            exporter_exit = stream_runner(
                render_export_command(
                    harbor_source,
                    jobs_dir,
                    compatibility_path,
                    prepared.request.harbor_revision,
                    request_digest,
                ),
                export_log,
                environment=environment,
                timeout_seconds=export_timeout,
            )
        except (OSError, RuntimeError):
            self._validate_inputs(prepared)
            if harbor_exit != 0:
                return False
            raise
        self._validate_inputs(prepared)
        if exporter_exit != 0:
            if harbor_exit != 0:
                return False
            raise WorkerError(
                f"Harbor compatibility exporter exited with status {exporter_exit}"
            )
        return True

    @staticmethod
    def _validate_inputs(prepared: PreparedHarborExecution) -> None:
        if prepared.config_path.read_bytes() != prepared.request.config_bytes():
            raise WorkerError("Harbor job config changed after request preparation")
        if prepared.request_path.read_bytes() != prepared.request.request_bytes():
            raise WorkerError("Harbor execution request changed after preparation")


def _shared_deadline(
    timeout_seconds: int,
    deadline: float | None,
    monotonic: Callable[[], float],
) -> float:
    if timeout_seconds <= 0:
        raise WorkerError("Harbor execution timeout must be positive")
    started_at = monotonic()
    shared_deadline = deadline if deadline is not None else started_at + timeout_seconds
    if shared_deadline <= started_at:
        raise WorkerError("Harbor execution deadline was already reached")
    return shared_deadline


def _remaining_export_timeout(deadline: float, monotonic: Callable[[], float]) -> int:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise WorkerError("Harbor execution deadline was reached before export")
    return min(max(1, math.ceil(remaining)), 300)


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
    resolved_task_digests = {
        task: digest
        for task, digest in lock.benchmark_task_digests.items()
        if any(fnmatch(task, selector) for selector in task_names)
    }
    if (
        any(
            not any(fnmatch(task, selector) for task in lock.benchmark_task_digests)
            for selector in task_names
        )
        or expected_task_digests != resolved_task_digests
    ):
        raise WorkerError("Harbor request contains a task outside the resolved run set")
    served_model_name = _served_model_name(lock)
    if lock.benchmark_source is None:
        try:
            dataset_reference = pinned_harbor_dataset_reference(
                lock.benchmark_dataset, lock.benchmark_dataset_digest
            )
        except ValueError as error:
            raise WorkerError(str(error)) from error
        dataset_name, dataset_ref = dataset_reference.rsplit("@", 1)
        dataset: dict[str, JsonValue] = {
            "name": dataset_name,
            "ref": dataset_ref,
            "task_names": task_names,
        }
    else:
        source = lock.benchmark_source
        dataset = {
            "repo": f"{source.repository}@{source.revision}",
            "path": source.path,
            "task_names": task_names,
        }
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
        "datasets": [dataset],
    }
    expected_trials = len(expected_task_digests) * attempts
    policy = HarborVerificationPolicy(
        expected_trials=expected_trials,
        expected_task_counts={task: attempts for task in expected_task_digests},
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


def render_export_command(
    harbor_source: Path,
    jobs_dir: Path,
    output_path: Path,
    harbor_revision: str,
    request_digest: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        str(harbor_source),
        "--locked",
        "--no-dev",
        "--extra",
        "hf-sandbox",
        "python",
        str(Path(__file__).with_name("exporter.py")),
        "--jobs-dir",
        str(jobs_dir),
        "--output",
        str(output_path),
        "--harbor-revision",
        harbor_revision,
        "--request-digest",
        request_digest,
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
