from __future__ import annotations

from pathlib import Path

import pytest

from harbor_hf.harbor_adapter import build_execution_request
from harbor_hf.models import DeploymentProfile, ExperimentSpec
from harbor_hf.provider_models import (
    ExplicitProviderRoute,
    ProviderLimits,
    ProviderTarget,
)
from harbor_hf.runs import RunLock, build_run_lock
from harbor_hf.worker import (
    WorkerError,
    build_harbor_command,
    build_harbor_trial_command,
)


def _contract_lock(remote_spec: ExperimentSpec) -> RunLock:
    lock = build_run_lock(remote_spec, run_id="command-contract")
    assert isinstance(lock.deployment, DeploymentProfile)
    endpoint = lock.deployment.endpoint
    assert endpoint is not None
    return lock.model_copy(
        update={
            "benchmark_dataset": "harbor/dataset-contract@9.7",
            "benchmark_dataset_digest": "sha256:" + "9" * 64,
            "benchmark_tasks": ["task-zeta", "task-alpha"],
            "benchmark_task_digests": {
                "task-zeta": "sha256:" + "7" * 64,
                "task-alpha": "sha256:" + "8" * 64,
            },
            "attempts": 3,
            "concurrent_trials": 4,
            "agent": lock.agent.model_copy(
                update={
                    "name": "agent-contract",
                    "revision": "3.4.5",
                    "parameters": {
                        "zeta": {"nested": 7},
                        "alpha": ["value", 2],
                        "middle": "quoted-contract",
                    },
                }
            ),
            "deployment": lock.deployment.model_copy(
                update={
                    "endpoint": endpoint.model_copy(
                        update={"served_model_name": "served-contract"}
                    )
                }
            ),
            "remote": lock.remote.model_copy(
                update={
                    "harbor": lock.remote.harbor.model_copy(
                        update={
                            "environment": "environment-contract",
                            "sandbox_flavor": "flavor-contract",
                            "sandbox_idle_timeout_seconds": 4321,
                        }
                    )
                }
            ),
        }
    )


def test_run_command_is_the_complete_ordered_process_contract(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _contract_lock(remote_spec)
    jobs = tmp_path / "jobs-contract"
    source = tmp_path / "source-contract"

    command = build_harbor_command(
        lock,
        jobs,
        "https://user:password@host-contract.example:9443/api",
        source,
    )
    assert command == [
        "uv",
        "run",
        "--project",
        str(source),
        "--locked",
        "--no-dev",
        "--extra",
        "hf-sandbox",
        "harbor",
        "run",
        "--config",
        str(jobs.parent / "harbor-job.json"),
        "--yes",
    ]
    request = build_execution_request(
        lock,
        jobs,
        "https://user:password@host-contract.example:9443/api",
        task_names=list(lock.benchmark_tasks),
        attempts=3,
        concurrency=4,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    assert request.harbor_config == {
        "jobs_dir": str(jobs),
        "n_attempts": 3,
        "n_concurrent_trials": 4,
        "retry": {"max_retries": 0},
        "environment": {
            "type": "environment-contract",
            "kwargs": {"flavor": "flavor-contract", "job_timeout": 4321},
        },
        "agents": [
            {
                "name": "agent-contract",
                "model_name": "openai/served-contract",
                "n_concurrent": 4,
                "extra_allowed_hosts": ["host-contract.example"],
                "kwargs": {
                    "alpha": ["value", 2],
                    "middle": "quoted-contract",
                    "version": "3.4.5",
                    "zeta": {"nested": 7},
                },
            }
        ],
        "datasets": [
            {
                "name": "harbor/dataset-contract",
                "ref": "sha256:" + "9" * 64,
                "task_names": ["task-zeta", "task-alpha"],
            }
        ],
    }


def test_wave_trial_command_overrides_only_attempt_and_concurrency_contract(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _contract_lock(remote_spec)
    jobs = tmp_path / "trial-jobs"
    source = tmp_path / "trial-source"

    command = build_harbor_trial_command(
        lock,
        jobs,
        "https://trial-host.example/base",
        source,
        task_name="task-alpha",
    )
    assert command == [
        "uv",
        "run",
        "--project",
        str(source),
        "--locked",
        "--no-dev",
        "--extra",
        "hf-sandbox",
        "harbor",
        "run",
        "--config",
        str(jobs.parent / "harbor-job.json"),
        "--yes",
    ]
    request = build_execution_request(
        lock,
        jobs,
        "https://trial-host.example/base",
        task_names=["task-alpha"],
        attempts=1,
        concurrency=1,
        expected_task_digests={"task-alpha": lock.benchmark_task_digests["task-alpha"]},
    )
    assert request.harbor_config["n_attempts"] == 1
    assert request.harbor_config["n_concurrent_trials"] == 1
    assert request.harbor_config["datasets"] == [
        {
            "name": "harbor/dataset-contract",
            "ref": "sha256:" + "9" * 64,
            "task_names": ["task-alpha"],
        }
    ]


def test_command_uses_empty_host_only_when_base_url_has_no_hostname(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _contract_lock(remote_spec)
    request = build_execution_request(
        lock,
        tmp_path,
        "/relative-base",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    agents = request.harbor_config["agents"]
    assert isinstance(agents, list)
    assert agents[0]["extra_allowed_hosts"] == [""]


def test_command_rejects_a_lock_without_an_endpoint_binding(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _contract_lock(remote_spec)
    lock = lock.model_copy(
        update={"deployment": lock.deployment.model_copy(update={"endpoint": None})}
    )

    with pytest.raises(WorkerError, match="^run lock has no endpoint binding$"):
        build_harbor_command(lock, tmp_path, "https://unused.example", tmp_path)


def test_command_rejects_a_dataset_that_cannot_be_content_addressed(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _contract_lock(remote_spec).model_copy(
        update={"benchmark_dataset": "legacy-dataset@9.7"}
    )

    with pytest.raises(
        WorkerError,
        match="must use a Harbor package name in org/name form",
    ):
        build_harbor_command(lock, tmp_path, "https://unused.example", tmp_path)


def test_command_preserves_an_existing_content_addressed_dataset(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    digest = "sha256:" + "6" * 64
    lock = _contract_lock(remote_spec).model_copy(
        update={
            "benchmark_dataset": f"harbor/dataset-contract@{digest}",
            "benchmark_dataset_digest": digest,
        }
    )

    request = build_execution_request(
        lock,
        tmp_path,
        "https://endpoint.example",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    assert request.harbor_config["datasets"] == [
        {
            "name": "harbor/dataset-contract",
            "ref": digest,
            "task_names": ["task-zeta", "task-alpha"],
        }
    ]


def test_command_rejects_an_invalid_locked_dataset_digest(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _contract_lock(remote_spec).model_copy(
        update={"benchmark_dataset_digest": "sha256:short"}
    )

    with pytest.raises(
        WorkerError,
        match="dataset digest must be a full sha256 digest",
    ):
        build_harbor_command(lock, tmp_path, "https://unused.example", tmp_path)


def test_provider_command_applies_locked_openclaw_request_controls(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    target = ProviderTarget(
        id="provider-contract",
        model=remote_spec.matrix.models[0].repo,
        routing=ExplicitProviderRoute(provider="groq"),
        timeout_seconds=17.25,
        limits=ProviderLimits(max_concurrent_requests=4, max_attempts=3),
        parameters={"temperature": 0, "max_tokens": 4096},
    )
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(update={"deployments": [target]})
        }
    )
    lock = build_run_lock(spec, run_id="provider-command-contract", allow_provider=True)

    task_name = next(iter(lock.benchmark_task_digests))
    request = build_execution_request(
        lock,
        tmp_path / "jobs",
        "https://router.huggingface.co",
        task_names=[task_name],
        attempts=1,
        concurrency=1,
        expected_task_digests={task_name: lock.benchmark_task_digests[task_name]},
    )
    agents = request.harbor_config["agents"]
    assert isinstance(agents, list)
    config = agents[0]["kwargs"]["openclaw_config"]
    provider = config["models"]["providers"]["openai"]
    assert provider == {
        "api": "openai-completions",
        "timeoutSeconds": 18,
        "models": [
            {
                "id": f"{remote_spec.matrix.models[0].repo}:groq",
                "name": f"{remote_spec.matrix.models[0].repo}:groq",
                "params": {
                    "temperature": 0,
                    "max_tokens": 4096,
                    "maxRetries": 2,
                    "timeoutMs": 17250,
                },
            }
        ],
    }


def test_provider_command_rejects_an_agent_without_request_control_support(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    target = ProviderTarget(
        id="provider-contract", model=remote_spec.matrix.models[0].repo
    )
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(update={"deployments": [target]}),
            "execution": remote_spec.execution.model_copy(update={"attempts": 1}),
        }
    )
    lock = build_run_lock(spec, run_id="provider-command-contract", allow_provider=True)
    lock = lock.model_copy(
        update={"agent": lock.agent.model_copy(update={"name": "unsupported-agent"})}
    )

    with pytest.raises(
        WorkerError,
        match="request controls require the OpenClaw Harbor agent",
    ):
        build_harbor_trial_command(
            lock,
            tmp_path / "jobs",
            "https://router.huggingface.co",
            tmp_path / "source",
            task_name=next(iter(lock.benchmark_task_digests)),
        )
