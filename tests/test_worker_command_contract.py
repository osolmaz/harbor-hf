from __future__ import annotations

from pathlib import Path

import pytest

from harbor_hf.models import ExperimentSpec
from harbor_hf.runs import RunLock, build_run_lock
from harbor_hf.worker import (
    WorkerError,
    build_harbor_command,
    build_harbor_trial_command,
)


def _contract_lock(remote_spec: ExperimentSpec) -> RunLock:
    lock = build_run_lock(remote_spec, run_id="command-contract")
    endpoint = lock.deployment.endpoint
    assert endpoint is not None
    return lock.model_copy(
        update={
            "benchmark_dataset": "dataset-contract@9.7",
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

    assert build_harbor_command(
        lock,
        jobs,
        "https://user:password@host-contract.example:9443/api",
        source,
    ) == [
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
        "--dataset",
        "dataset-contract@9.7",
        "--n-attempts",
        "3",
        "--agent",
        "agent-contract",
        "--model",
        "openai/served-contract",
        "--env",
        "environment-contract",
        "--environment-kwarg",
        "flavor=flavor-contract",
        "--environment-kwarg",
        "job_timeout=4321",
        "--jobs-dir",
        str(jobs),
        "--n-concurrent",
        "4",
        "--n-concurrent-agents",
        "4",
        "--max-retries",
        "0",
        "--allow-agent-host",
        "host-contract.example",
        "--yes",
        "--include-task-name",
        "task-zeta",
        "--include-task-name",
        "task-alpha",
        "--agent-kwarg",
        'version="3.4.5"',
        "--agent-kwarg",
        'alpha=["value",2]',
        "--agent-kwarg",
        'middle="quoted-contract"',
        "--agent-kwarg",
        'zeta={"nested":7}',
    ]


def test_wave_trial_command_overrides_only_attempt_and_concurrency_contract(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _contract_lock(remote_spec)
    jobs = tmp_path / "trial-jobs"
    source = tmp_path / "trial-source"

    assert build_harbor_trial_command(
        lock,
        jobs,
        "https://trial-host.example/base",
        source,
        task_name="task-alpha",
    ) == [
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
        "--dataset",
        "dataset-contract@9.7",
        "--n-attempts",
        "1",
        "--agent",
        "agent-contract",
        "--model",
        "openai/served-contract",
        "--env",
        "environment-contract",
        "--environment-kwarg",
        "flavor=flavor-contract",
        "--environment-kwarg",
        "job_timeout=4321",
        "--jobs-dir",
        str(jobs),
        "--n-concurrent",
        "1",
        "--n-concurrent-agents",
        "1",
        "--max-retries",
        "0",
        "--allow-agent-host",
        "trial-host.example",
        "--yes",
        "--include-task-name",
        "task-alpha",
        "--agent-kwarg",
        'version="3.4.5"',
        "--agent-kwarg",
        'alpha=["value",2]',
        "--agent-kwarg",
        'middle="quoted-contract"',
        "--agent-kwarg",
        'zeta={"nested":7}',
    ]


def test_command_uses_empty_host_only_when_base_url_has_no_hostname(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    command = build_harbor_command(
        _contract_lock(remote_spec), tmp_path, "/relative-base", tmp_path
    )

    assert command[command.index("--allow-agent-host") + 1] == ""


def test_command_rejects_a_lock_without_an_endpoint_binding(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _contract_lock(remote_spec)
    lock = lock.model_copy(
        update={"deployment": lock.deployment.model_copy(update={"endpoint": None})}
    )

    with pytest.raises(WorkerError, match="^run lock has no endpoint binding$"):
        build_harbor_command(lock, tmp_path, "https://unused.example", tmp_path)
