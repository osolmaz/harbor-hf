from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from conftest import write_fake_compatibility_bundle

from harbor_hf.coordination import ClaimConflict, run_claim_path
from harbor_hf.harbor_adapter import build_execution_request
from harbor_hf.models import (
    DeploymentProfile,
    ExperimentSpec,
    GitBenchmarkSource,
    GitHubTokenCredentials,
    SourcePin,
)
from harbor_hf.private_artifacts import (
    PrivateArtifactRejection,
    sanitize_private_artifact_directory_files,
    sanitize_private_artifact_symlinks,
)
from harbor_hf.process import CommandRunner, ProcessError
from harbor_hf.runs import RunLock, build_run_lock
from harbor_hf.worker import (
    EndpointManager,
    WorkerError,
    _expected_agent_version,
    _expected_task_counts,
    _expected_trial_count,
    _finalize_evidence,
    _job_stage,
    _mark_watchdog_ready,
    _prepare_evidence_destination,
    _publish_evidence,
    _sanitize_direct_trial_artifacts,
    _validate_direct_private_artifacts,
    _validate_endpoint_compute,
    _validate_task_counts,
    _validate_trial_count,
    _watchdog_readiness_error,
    build_harbor_command,
    controller_environment,
    endpoint_health_route,
    endpoint_state,
    endpoint_url,
    launch_cleanup_watchdog,
    launch_cleanup_watchdog_for,
    prepare_locked_source,
    probe_runtime,
    require_executable,
    require_paused_endpoint,
    run_endpoint_watchdog,
    run_worker,
    validate_endpoint_model,
    validate_harbor_result,
    validate_run_lock,
    wait_for_runtime,
    wait_watchdog_ready,
)


class FakeClaimStore:
    def __init__(self) -> None:
        self.claims: dict[str, dict[str, str]] = {}
        self.acquired: list[tuple[str, dict[str, str]]] = []
        self.released: list[tuple[str, dict[str, str]]] = []

    def acquire(self, path: str, owner: Mapping[str, str]) -> None:
        if path in self.claims:
            raise ClaimConflict(path)
        normalized = dict(owner)
        self.claims[path] = normalized
        self.acquired.append((path, normalized))

    def release(self, path: str, owner: Mapping[str, str]) -> None:
        normalized = dict(owner)
        if self.claims.get(path) != normalized:
            raise RuntimeError("claim ownership cannot be verified")
        del self.claims[path]
        self.released.append((path, normalized))


@pytest.fixture(autouse=True)
def default_claim_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "harbor_hf.worker.HubClaimStore", lambda *_args, **_kwargs: FakeClaimStore()
    )


def snapshot(state: str, ready: int) -> dict[str, object]:
    return {
        "model": {
            "repository": "nvidia/Qwen3.6-35B-A3B-NVFP4",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "image": {"custom": {"url": "ghcr.io/example/vllm@sha256:" + "0" * 64}},
            "args": [
                "--model",
                "/repository",
                "--max-model-len",
                "65536",
                "--kv-cache-dtype",
                "fp8",
            ],
            "env": {"VLLM_USE_FLASHINFER_MOE_FP4": "1"},
            "secrets": {"HF_TOKEN": "[REDACTED]"},
        },
        "provider": {"vendor": "aws", "region": "us-east-1"},
        "compute": {
            "instanceType": "nvidia-rtx-pro-6000",
            "instanceSize": "x1",
            "scaling": {"minReplica": 0, "maxReplica": 1},
        },
        "status": {
            "state": state,
            "readyReplica": ready,
            "targetReplica": 1,
            "url": "https://endpoint.example",
        },
        "healthRoute": "/ready",
    }


class EndpointRunner:
    def __init__(self, descriptions: list[dict[str, object]]) -> None:
        self.descriptions = descriptions
        self.commands: list[list[str]] = []
        self.timeouts: list[float | None] = []

    def run_json(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        self.commands.append(list(command))
        self.timeouts.append(timeout_seconds)
        operation = command[2]
        if operation == "describe":
            return self.descriptions.pop(0)
        return snapshot("running" if operation == "resume" else "paused", 0)

    def run_text(self, command: Sequence[str]) -> str:
        raise AssertionError(command)


class CleanupFailureRunner(EndpointRunner):
    def run_json(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        if command[2] == "pause":
            self.commands.append(list(command))
            self.timeouts.append(timeout_seconds)
            raise RuntimeError("pause failed with test-token")
        return super().run_json(command, timeout_seconds=timeout_seconds)


def _prepare_source(
    _source: SourcePin, destination: Path, _runner: CommandRunner
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "uv.lock").write_text("", encoding="utf-8")


def _launch_watchdog(_lock: RunLock, _endpoint: object, _token: str) -> str:
    return "watchdog-job"


class WatchdogApiStub:
    def __init__(self) -> None:
        self.label_updates: list[dict[str, object]] = []

    def update_job_labels(self, **kwargs: object) -> object:
        self.label_updates.append(kwargs)
        return SimpleNamespace()

    def inspect_job(self, **_kwargs: object) -> object:
        raise AssertionError("unexpected job inspection")


def test_endpoint_lifecycle_and_status() -> None:
    runner = EndpointRunner([snapshot("running", 1), snapshot("paused", 0)])
    manager = EndpointManager("org", "endpoint", runner)

    manager.resume()
    assert endpoint_state(manager.wait_ready(10)) == ("running", 1, 1)
    assert endpoint_state(manager.pause_and_verify()) == ("paused", 0, 1)
    assert runner.commands == [
        [
            "hf",
            "endpoints",
            operation,
            "endpoint",
            "--namespace",
            "org",
            "--format",
            "json",
        ]
        for operation in ("resume", "describe", "pause", "describe")
    ]
    assert all(timeout is not None and 0 < timeout <= 60 for timeout in runner.timeouts)


def test_readiness_timeout() -> None:
    times = iter([0.0, 2.0])
    manager = EndpointManager(
        "org",
        "endpoint",
        EndpointRunner([snapshot("initializing", 0)]),
        sleep=lambda _: None,
        monotonic=lambda: next(times),
    )

    with pytest.raises(WorkerError, match="readiness timed out"):
        manager.wait_ready(1)


def test_readiness_retries_transient_describe_failure_within_budget() -> None:
    class FlakyRunner(EndpointRunner):
        attempts = 0

        def run_json(
            self,
            command: Sequence[str],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, object]:
            self.attempts += 1
            if self.attempts == 1:
                raise ProcessError("temporary describe failure")
            return super().run_json(command, timeout_seconds=timeout_seconds)

    sleeps: list[float] = []
    times = iter([0.0, 0.0, 0.0, 1.0])
    manager = EndpointManager(
        "org",
        "endpoint",
        FlakyRunner([snapshot("running", 1)]),
        sleep=sleeps.append,
        monotonic=lambda: next(times),
    )

    assert endpoint_state(manager.wait_ready(10, poll_seconds=2)) == (
        "running",
        1,
        1,
    )
    assert sleeps == [2]


def test_readiness_aborts_after_bounded_consecutive_describe_failures() -> None:
    class FailingRunner(EndpointRunner):
        attempts = 0

        def run_json(
            self,
            command: Sequence[str],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, object]:
            self.attempts += 1
            raise ProcessError("permanent describe failure")

    sleeps: list[float] = []
    runner = FailingRunner([])
    manager = EndpointManager(
        "org",
        "endpoint",
        runner,
        sleep=sleeps.append,
        monotonic=lambda: 0.0,
    )

    with pytest.raises(
        WorkerError,
        match=(
            "^endpoint readiness aborted after 3 consecutive provider errors: "
            "permanent describe failure$"
        ),
    ):
        manager.wait_ready(3600)

    assert runner.attempts == 3
    assert sleeps == [15, 15]


def test_endpoint_waits_through_transitional_states() -> None:
    sleeps: list[float] = []
    times = iter(float(value) for value in range(9))
    runner = EndpointRunner(
        [
            snapshot("initializing", 0),
            snapshot("running", 1),
            snapshot("pausing", 1),
            snapshot("paused", 0),
        ]
    )
    manager = EndpointManager(
        "org",
        "endpoint",
        runner,
        sleep=sleeps.append,
        monotonic=lambda: next(times),
    )

    assert endpoint_state(manager.wait_ready(10, poll_seconds=2.5))[0] == "running"
    assert endpoint_state(manager.pause_and_verify(10, poll_seconds=3.5))[0] == (
        "paused"
    )
    assert sleeps == [2.5, 3.5]


def test_endpoint_waits_for_every_target_replica() -> None:
    partial = snapshot("running", 1)
    cast(dict[str, object], partial["status"])["targetReplica"] = 2
    complete = snapshot("running", 2)
    cast(dict[str, object], complete["status"])["targetReplica"] = 2
    runner = EndpointRunner([partial, complete])

    result = EndpointManager(
        "org", "endpoint", runner, sleep=lambda _: None
    ).wait_ready(10)

    assert endpoint_state(result) == ("running", 2, 2)


def test_endpoint_parsing_rejects_incomplete_response() -> None:
    with pytest.raises(WorkerError, match="^endpoint response has no status object$"):
        endpoint_state({})
    with pytest.raises(WorkerError, match="^endpoint status is missing its URL$"):
        endpoint_url({"status": {}})
    with pytest.raises(
        WorkerError, match="^endpoint status is missing state or readyReplica$"
    ):
        endpoint_state({"status": {"state": 1, "readyReplica": "one"}})
    assert endpoint_state({"status": {"state": "running", "readyReplica": 1}}) == (
        "running",
        1,
        0,
    )
    assert (
        endpoint_url({"status": {"url": "https://endpoint.example/"}})
        == "https://endpoint.example"
    )
    assert endpoint_health_route({"healthRoute": "/ready"}) == "/ready"
    assert (
        endpoint_health_route(
            {"model": {"image": {"custom": {"healthRoute": "/status/health"}}}}
        )
        == "/status/health"
    )
    with pytest.raises(
        WorkerError, match="^endpoint response has no valid health route$"
    ):
        endpoint_health_route({"healthRoute": "//other.example/ready"})


def test_endpoint_model_must_match_lock(remote_spec: ExperimentSpec) -> None:
    lock = build_run_lock(remote_spec)
    validate_endpoint_model(lock, snapshot("running", 1))

    wrong = snapshot("running", 1)
    model = wrong["model"]
    assert isinstance(model, dict)
    cast(dict[str, object], model)["revision"] = "wrong"
    with pytest.raises(WorkerError, match="^endpoint model does not match"):
        validate_endpoint_model(lock, wrong)
    with pytest.raises(WorkerError, match="^endpoint response has no model object$"):
        validate_endpoint_model(lock, {})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "image",
            {"custom": {"url": "different"}},
            "endpoint image does not match the locked deployment",
        ),
        (
            "command",
            ["python", "-m", "different_server"],
            "endpoint command does not match the locked deployment",
        ),
        (
            "args",
            ["--max-model-len", "32768"],
            "endpoint arguments do not match the locked deployment",
        ),
        (
            "args",
            [
                "--max-model-len",
                "65536",
                "--kv-cache-dtype",
                "fp8",
                "--max-model-len",
                "32768",
            ],
            "endpoint arguments do not match the locked deployment",
        ),
        (
            "env",
            {"VLLM_USE_FLASHINFER_MOE_FP4": "0"},
            "endpoint environment does not match the locked deployment",
        ),
        (
            "secrets",
            {"OTHER_TOKEN": "[redacted]"},
            "endpoint secret names do not match the locked deployment",
        ),
    ],
)
def test_endpoint_model_requires_locked_serving_configuration(
    remote_spec: ExperimentSpec, field: str, value: object, message: str
) -> None:
    endpoint_snapshot = snapshot("running", 1)
    model = cast(dict[str, object], endpoint_snapshot["model"])
    model[field] = value

    with pytest.raises(WorkerError, match=f"^{message}$"):
        validate_endpoint_model(build_run_lock(remote_spec), endpoint_snapshot)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider", {"vendor": "aws", "region": "us-west-2"}, "compute"),
        (
            "compute",
            {
                "instanceType": "nvidia-h200",
                "instanceSize": "x1",
                "scaling": {"minReplica": 0, "maxReplica": 1},
            },
            "compute",
        ),
        (
            "compute",
            {
                "instanceType": "nvidia-rtx-pro-6000",
                "instanceSize": "x2",
                "scaling": {"minReplica": 0, "maxReplica": 1},
            },
            "compute",
        ),
        (
            "compute",
            {
                "instanceType": "nvidia-rtx-pro-6000",
                "instanceSize": "x1",
                "scaling": {"minReplica": 1, "maxReplica": 1},
            },
            "scaling",
        ),
    ],
)
def test_endpoint_compute_requires_locked_deployment(
    remote_spec: ExperimentSpec, field: str, value: object, message: str
) -> None:
    endpoint_snapshot = snapshot("running", 1)
    endpoint_snapshot[field] = value

    with pytest.raises(WorkerError, match=f"endpoint {message} does not match"):
        validate_endpoint_model(build_run_lock(remote_spec), endpoint_snapshot)


def test_endpoint_arguments_require_complete_ordered_identity(
    remote_spec: ExperimentSpec,
) -> None:
    deployment = remote_spec.matrix.deployments[0]
    assert isinstance(deployment, DeploymentProfile)
    engine = deployment.engine.model_copy(update={"arguments": ["-c", "65536"]})
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={
                    "deployments": [deployment.model_copy(update={"engine": engine})]
                }
            )
        }
    )
    endpoint_snapshot = snapshot("running", 1)
    model = cast(dict[str, object], endpoint_snapshot["model"])
    model["args"] = ["-c", "32768"]

    with pytest.raises(
        WorkerError, match="^endpoint arguments do not match the locked deployment$"
    ):
        validate_endpoint_model(build_run_lock(spec), endpoint_snapshot)


def test_endpoint_omitted_arguments_match_an_empty_lock(
    remote_spec: ExperimentSpec,
) -> None:
    deployment = remote_spec.matrix.deployments[0]
    assert isinstance(deployment, DeploymentProfile)
    engine = deployment.engine.model_copy(update={"arguments": [], "secret_names": []})
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={
                    "deployments": [deployment.model_copy(update={"engine": engine})]
                }
            )
        }
    )
    endpoint_snapshot = snapshot("paused", 0)
    model = cast(dict[str, object], endpoint_snapshot["model"])
    model.pop("args")
    model.pop("secrets")

    validate_endpoint_model(build_run_lock(spec), endpoint_snapshot)


def test_endpoint_environment_rejects_unlocked_extra_values(
    remote_spec: ExperimentSpec,
) -> None:
    endpoint_snapshot = snapshot("running", 1)
    model = cast(dict[str, object], endpoint_snapshot["model"])
    environment = cast(dict[str, object], model["env"])
    environment["UNLOCKED_SETTING"] = "1"

    with pytest.raises(
        WorkerError, match="^endpoint environment does not match the locked deployment$"
    ):
        validate_endpoint_model(build_run_lock(remote_spec), endpoint_snapshot)


def test_endpoint_compute_validation_covers_complete_identity(
    remote_spec: ExperimentSpec,
) -> None:
    lock = build_run_lock(remote_spec)
    endpoint_snapshot = snapshot("running", 1)

    _validate_endpoint_compute(lock, endpoint_snapshot)

    for missing in ("provider", "compute"):
        incomplete = snapshot("running", 1)
        incomplete.pop(missing)
        with pytest.raises(
            WorkerError,
            match="^endpoint response has no deployment compute identity$",
        ):
            _validate_endpoint_compute(lock, incomplete)

    missing_scaling = snapshot("running", 1)
    cast(dict[str, object], missing_scaling["compute"]).pop("scaling")
    with pytest.raises(
        WorkerError, match="^endpoint response has no scaling configuration$"
    ):
        _validate_endpoint_compute(lock, missing_scaling)

    wrong_region = snapshot("running", 1)
    cast(dict[str, object], wrong_region["provider"])["region"] = "us-west-2"
    with pytest.raises(
        WorkerError,
        match="^endpoint compute does not match the locked deployment$",
    ):
        _validate_endpoint_compute(lock, wrong_region)

    wrong_maximum = snapshot("running", 1)
    compute = cast(dict[str, object], wrong_maximum["compute"])
    scaling = cast(dict[str, object], compute["scaling"])
    scaling["maxReplica"] = 2
    with pytest.raises(
        WorkerError, match="^endpoint scaling does not match the locked deployment$"
    ):
        _validate_endpoint_compute(lock, wrong_maximum)


@pytest.mark.parametrize(
    ("state", "ready"),
    [("running", 1), ("paused", 1), ("initializing", 0)],
)
def test_endpoint_baseline_requires_verified_pause(state: str, ready: int) -> None:
    with pytest.raises(
        WorkerError,
        match="^endpoint must be paused with zero ready replicas before ownership$",
    ):
        require_paused_endpoint(snapshot(state, ready))

    require_paused_endpoint(snapshot("paused", 0))


def test_controller_requires_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("harbor_hf.worker.shutil.which", lambda _name: None)

    with pytest.raises(
        WorkerError, match="^required controller executable is missing: git$"
    ):
        require_executable("git")


def test_prepare_locked_source_checks_out_revision_and_requires_lock(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    remote = remote_spec.remote
    assert remote is not None
    destination = tmp_path / "nested" / "sources" / "source"

    class SourceRunner:
        def __init__(self) -> None:
            self.commands: list[list[str]] = []

        def run_text(self, command: Sequence[str]) -> str:
            self.commands.append(list(command))
            if "checkout" in command:
                destination.mkdir(parents=True)
                (destination / "uv.lock").write_text("", encoding="utf-8")
                (destination / "pyproject.toml").write_text(
                    "[project.optional-dependencies]\nhf-sandbox = []\n",
                    encoding="utf-8",
                )
            return ""

        def run_json(
            self,
            command: Sequence[str],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, object]:
            raise AssertionError(command)

    runner = SourceRunner()
    prepare_locked_source(remote.harbor.source, destination, runner)

    assert [command[1] for command in runner.commands] == ["clone", "-C", "-C"]
    assert runner.commands == [
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/harbor-framework/harbor",
            str(destination),
        ],
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--depth",
            "1",
            "origin",
            remote.harbor.source.revision,
        ],
        [
            "git",
            "-C",
            str(destination),
            "checkout",
            "--detach",
            remote.harbor.source.revision,
        ],
    ]

    with pytest.raises(WorkerError, match="source checkout already exists"):
        prepare_locked_source(remote.harbor.source, destination, runner)


def test_prepare_locked_source_rejects_checkout_without_lock(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    remote = remote_spec.remote
    assert remote is not None
    destination = tmp_path / "source"

    class SourceRunner:
        def run_text(self, command: Sequence[str]) -> str:
            destination.mkdir(parents=True, exist_ok=True)
            return ""

        def run_json(
            self,
            command: Sequence[str],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, object]:
            raise AssertionError(command)

    with pytest.raises(WorkerError, match="pinned source checkout has no uv.lock"):
        prepare_locked_source(remote.harbor.source, destination, SourceRunner())


def test_prepare_locked_source_requires_hf_sandbox_extra(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    remote = remote_spec.remote
    assert remote is not None
    destination = tmp_path / "source"

    class SourceRunner:
        def run_text(self, command: Sequence[str]) -> str:
            if "checkout" in command:
                destination.mkdir(parents=True)
                (destination / "uv.lock").write_text("", encoding="utf-8")
                (destination / "pyproject.toml").write_text(
                    "[project.optional-dependencies]\nother = []\n",
                    encoding="utf-8",
                )
            return ""

        def run_json(
            self,
            command: Sequence[str],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, object]:
            raise AssertionError(command)

    with pytest.raises(
        WorkerError,
        match="^pinned Harbor checkout does not provide the hf-sandbox extra$",
    ):
        prepare_locked_source(remote.harbor.source, destination, SourceRunner())


def test_prepare_locked_source_rejects_malformed_project_metadata(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    remote = remote_spec.remote
    assert remote is not None
    destination = tmp_path / "source"

    class SourceRunner:
        def run_text(self, command: Sequence[str]) -> str:
            if "checkout" in command:
                destination.mkdir(parents=True)
                (destination / "uv.lock").write_text("", encoding="utf-8")
                (destination / "pyproject.toml").write_text(
                    "project = []\n",
                    encoding="utf-8",
                )
            return ""

        def run_json(
            self,
            command: Sequence[str],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, object]:
            raise AssertionError(command)

    with pytest.raises(
        WorkerError,
        match="^pinned Harbor checkout does not provide the hf-sandbox extra$",
    ):
        prepare_locked_source(remote.harbor.source, destination, SourceRunner())


def test_prepare_locked_source_requires_pyproject(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    remote = remote_spec.remote
    assert remote is not None
    destination = tmp_path / "source"

    class SourceRunner:
        def run_text(self, command: Sequence[str]) -> str:
            if "checkout" in command:
                destination.mkdir(parents=True)
                (destination / "uv.lock").write_text("", encoding="utf-8")
            return ""

        def run_json(
            self,
            command: Sequence[str],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, object]:
            raise AssertionError(command)

    with pytest.raises(
        WorkerError,
        match="^pinned Harbor checkout has no pyproject.toml$",
    ):
        prepare_locked_source(remote.harbor.source, destination, SourceRunner())


def test_launch_watchdog_requires_controller_job_before_submission(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = build_run_lock(remote_spec)
    assert isinstance(lock.deployment, DeploymentProfile)
    endpoint = lock.deployment.endpoint
    assert endpoint is not None
    monkeypatch.delenv("JOB_ID", raising=False)

    with pytest.raises(
        WorkerError, match="^controller JOB_ID is required before endpoint resume$"
    ):
        launch_cleanup_watchdog(lock, endpoint, "secret")


def test_launch_watchdog_uses_independent_hf_job(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = build_run_lock(remote_spec, run_id="watchdog-run")
    assert isinstance(lock.deployment, DeploymentProfile)
    endpoint = lock.deployment.endpoint
    assert endpoint is not None
    calls: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "secret"

        def run_job(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(id="watchdog-job")

        def inspect_job(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                labels={"harbor-hf-watchdog-ready": "true"},
                status=SimpleNamespace(stage=SimpleNamespace(value="RUNNING")),
            )

    monkeypatch.setenv("JOB_ID", "controller-job")
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    assert launch_cleanup_watchdog(lock, endpoint, "secret") == "watchdog-job"
    assert calls[0] == {
        "image": "ghcr.io/astral-sh/uv@sha256:" + "0" * 64,
        "command": calls[0]["command"],
        "secrets": {"HF_TOKEN": "secret"},
        "flavor": "cpu-basic",
        "timeout": 11400,
        "labels": {
            "harbor-hf-watchdog": "watchdog-run",
            "harbor-hf-endpoint": "d026b68a5286b3887f1e9ea13d304aed",
        },
        "namespace": "osolmaz",
    }
    command = cast(list[str], calls[0]["command"])
    assert command[3:] == [
        "locked-source",
        "harbor-hf",
        "watchdog",
        "--controller-job-id",
        "controller-job",
        "--controller-namespace",
        "osolmaz",
        "--endpoint-name",
        "qwen-endpoint",
        "--endpoint-namespace",
        "osolmaz",
        "--run-id",
        "watchdog-run",
        "--token-secret-name",
        "HF_TOKEN",
        "--timeout-seconds",
        "10800",
    ]


def test_launch_watchdog_caps_timeout_and_requires_returned_id(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = remote_spec.remote
    assert remote is not None
    capped = remote_spec.model_copy(
        update={
            "remote": remote.model_copy(
                update={"job": remote.job.model_copy(update={"timeout_seconds": 85800})}
            )
        }
    )
    lock = build_run_lock(capped)
    assert isinstance(lock.deployment, DeploymentProfile)
    endpoint = lock.deployment.endpoint
    assert endpoint is not None
    calls: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "secret"

        def run_job(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(id=123)

    monkeypatch.setenv("JOB_ID", "controller-job")
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    with pytest.raises(
        WorkerError, match="^cleanup watchdog submission returned no job ID$"
    ):
        launch_cleanup_watchdog(lock, endpoint, "secret")

    assert calls[0]["timeout"] == 86400
    command = cast(list[str], calls[0]["command"])
    assert command[command.index("--timeout-seconds") + 1] == "85800"


def test_launch_watchdog_does_not_cancel_after_failed_handshake(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = build_run_lock(remote_spec)
    assert isinstance(lock.deployment, DeploymentProfile)
    endpoint = lock.deployment.endpoint
    assert endpoint is not None
    inspections: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "secret"

        def run_job(self, **_kwargs: object) -> object:
            return SimpleNamespace(id="watchdog-job")

        def inspect_job(self, **kwargs: object) -> object:
            inspections.append(kwargs)
            return SimpleNamespace(
                labels={}, status=SimpleNamespace(stage=SimpleNamespace(value="ERROR"))
            )

    monkeypatch.setenv("JOB_ID", "controller-job")
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    with pytest.raises(WorkerError, match="exited before readiness: ERROR"):
        launch_cleanup_watchdog(lock, endpoint, "secret")

    assert inspections == [{"job_id": "watchdog-job", "namespace": "osolmaz"}]


@pytest.mark.parametrize(
    "submission",
    [SimpleNamespace(), SimpleNamespace(id=object())],
)
def test_launch_watchdog_rejects_every_missing_or_nonstr_id_before_waiting(
    remote_spec: ExperimentSpec,
    monkeypatch: pytest.MonkeyPatch,
    submission: SimpleNamespace,
) -> None:
    remote = remote_spec.remote
    assert remote is not None
    deployment = remote_spec.matrix.deployments[0]
    assert isinstance(deployment, DeploymentProfile)
    endpoint = deployment.endpoint
    assert endpoint is not None

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "secret"

        def run_job(self, **_kwargs: object) -> object:
            return submission

    monkeypatch.setenv("JOB_ID", "controller-job")
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr(
        "harbor_hf.worker.wait_watchdog_ready",
        lambda *_args, **_kwargs: pytest.fail("an invalid ID must not be awaited"),
    )

    with pytest.raises(WorkerError) as captured:
        launch_cleanup_watchdog_for(remote, endpoint, "owner-one", "secret")

    assert str(captured.value) == "cleanup watchdog submission returned no job ID"


def test_launch_watchdog_waits_on_the_submitting_client_with_exact_identity(
    remote_spec: ExperimentSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = remote_spec.remote
    assert remote is not None
    deployment = remote_spec.matrix.deployments[0]
    assert isinstance(deployment, DeploymentProfile)
    endpoint = deployment.endpoint
    assert endpoint is not None
    clients: list[object] = []
    waits: list[tuple[object, str, str, int]] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "secret"
            clients.append(self)

        def run_job(self, **_kwargs: object) -> object:
            return SimpleNamespace(id="watchdog-job")

    def capture_wait(
        api: object,
        job_id: str,
        namespace: str,
        *,
        timeout_seconds: int,
    ) -> None:
        waits.append((api, job_id, namespace, timeout_seconds))

    monkeypatch.setenv("JOB_ID", "controller-job")
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr("harbor_hf.worker.wait_watchdog_ready", capture_wait)

    observed = launch_cleanup_watchdog_for(remote, endpoint, "owner-one", "secret")

    assert observed == "watchdog-job"
    assert waits == [(clients[0], "watchdog-job", "osolmaz", 300)]


def test_wait_watchdog_ready_polls_until_handshake() -> None:
    inspections: list[dict[str, object]] = []
    results = [
        SimpleNamespace(
            labels={}, status=SimpleNamespace(stage=SimpleNamespace(value="RUNNING"))
        ),
        SimpleNamespace(
            labels={"harbor-hf-watchdog-ready": "true"},
            status=SimpleNamespace(stage=SimpleNamespace(value="RUNNING")),
        ),
    ]

    class FakeApi(WatchdogApiStub):
        def inspect_job(self, **kwargs: object) -> object:
            inspections.append(kwargs)
            return results.pop(0)

    sleeps: list[float] = []
    wait_watchdog_ready(
        FakeApi(),
        "watchdog-job",
        "org",
        timeout_seconds=30,
        sleep=sleeps.append,
        monotonic=lambda: 0,
        poll_seconds=2,
    )

    assert inspections == [
        {"job_id": "watchdog-job", "namespace": "org"},
        {"job_id": "watchdog-job", "namespace": "org"},
    ]
    assert sleeps == [2]


@pytest.mark.parametrize(
    "stage", ["COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"]
)
def test_wait_watchdog_ready_rejects_early_exit(stage: str) -> None:
    class FakeApi:
        def inspect_job(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                labels={"harbor-hf-watchdog-ready": "true"},
                status=SimpleNamespace(stage=SimpleNamespace(value=stage)),
            )

    with pytest.raises(WorkerError, match=f"exited before readiness: {stage}"):
        wait_watchdog_ready(FakeApi(), "watchdog-job", "org", timeout_seconds=30)


def test_wait_watchdog_ready_reports_timeout() -> None:
    class FakeApi:
        def inspect_job(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                labels={},
                status=SimpleNamespace(stage=SimpleNamespace(value="RUNNING")),
            )

    times = iter([0.0, 0.0, 1.0])
    sleeps: list[float] = []
    with pytest.raises(WorkerError, match="^cleanup watchdog readiness timed out$"):
        wait_watchdog_ready(
            FakeApi(),
            "watchdog-job",
            "org",
            timeout_seconds=1,
            sleep=sleeps.append,
            monotonic=lambda: next(times),
        )
    assert sleeps == [5]


def test_wait_watchdog_ready_retries_provider_errors() -> None:
    calls = 0

    class FakeApi:
        def inspect_job(self, **_kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary provider error")
            return SimpleNamespace(
                labels={"harbor-hf-watchdog-ready": "true"},
                status=SimpleNamespace(stage=SimpleNamespace(value="RUNNING")),
            )

    sleeps: list[float] = []
    wait_watchdog_ready(
        FakeApi(),
        "watchdog-job",
        "org",
        timeout_seconds=30,
        sleep=sleeps.append,
        monotonic=lambda: 0,
    )

    assert calls == 2
    assert sleeps == [5]


def test_wait_watchdog_ready_bounds_provider_errors_at_deadline() -> None:
    provider_error = RuntimeError("temporary provider error")

    class FakeApi:
        def inspect_job(self, **_kwargs: object) -> object:
            raise provider_error

    times = iter([0.0, 1.0])
    with pytest.raises(
        WorkerError,
        match="^cleanup watchdog readiness timed out after provider errors$",
    ) as captured:
        wait_watchdog_ready(
            FakeApi(),
            "watchdog-job",
            "org",
            timeout_seconds=1,
            sleep=lambda _seconds: None,
            monotonic=lambda: next(times),
        )

    assert captured.value.__cause__ is provider_error


@pytest.mark.parametrize(
    "stage", ["COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"]
)
def test_endpoint_watchdog_pauses_after_controller_finishes(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    inspections: list[dict[str, object]] = []

    class FakeApi(WatchdogApiStub):
        def inspect_job(self, **kwargs: object) -> object:
            inspections.append(kwargs)
            return SimpleNamespace(
                status=SimpleNamespace(stage=SimpleNamespace(value=stage))
            )

    runner = EndpointRunner([snapshot("paused", 0)])
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("JOB_ID", "watchdog-job")
    api = FakeApi()
    claims = FakeClaimStore()
    result = run_endpoint_watchdog(
        controller_job_id="controller",
        controller_namespace="org",
        endpoint_name="endpoint",
        endpoint_namespace="org",
        run_id="run-1",
        token_secret_name="HF_TOKEN",
        timeout_seconds=60,
        api=api,
        claim_store=claims,
        runner=runner,
        monotonic=lambda: 0,
    )

    assert endpoint_state(result) == ("paused", 0, 1)
    assert claims.claims == {}
    assert len(claims.acquired) == len(claims.released) == 1
    assert inspections == [{"job_id": "controller", "namespace": "org"}]
    assert api.label_updates == [
        {
            "job_id": "watchdog-job",
            "labels": {
                "harbor-hf-watchdog": "run-1",
                "harbor-hf-endpoint": "aa3808503c913daab53ed1415fe04988",
                "harbor-hf-watchdog-ready": "true",
            },
            "namespace": "org",
        }
    ]
    assert runner.commands == [
        [
            "hf",
            "endpoints",
            operation,
            "endpoint",
            "--namespace",
            "org",
            "--format",
            "json",
        ]
        for operation in ("pause", "describe")
    ]


def test_endpoint_watchdog_survives_transient_inspection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes: list[object] = [RuntimeError("transient"), "ERROR"]
    sleeps: list[float] = []

    class FakeApi(WatchdogApiStub):
        def inspect_job(self, **_kwargs: object) -> object:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(
                status=SimpleNamespace(stage=SimpleNamespace(value=outcome))
            )

    runner = EndpointRunner([snapshot("paused", 0)])
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("JOB_ID", "watchdog-job")
    run_endpoint_watchdog(
        controller_job_id="controller",
        controller_namespace="org",
        endpoint_name="endpoint",
        endpoint_namespace="org",
        run_id="run-1",
        token_secret_name="HF_TOKEN",
        timeout_seconds=60,
        api=FakeApi(),
        claim_store=FakeClaimStore(),
        runner=runner,
        sleep=sleeps.append,
        monotonic=lambda: 0,
    )

    assert sleeps == [10]
    assert outcomes == []


def test_endpoint_watchdog_retains_lease_until_pause_when_readiness_update_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeApi(WatchdogApiStub):
        def update_job_labels(self, **_kwargs: object) -> object:
            raise RuntimeError("label update failed")

        def inspect_job(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                status=SimpleNamespace(stage=SimpleNamespace(value="COMPLETED"))
            )

    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("JOB_ID", "watchdog-job")
    claims = FakeClaimStore()
    runner = EndpointRunner([snapshot("paused", 0)])

    with pytest.raises(
        WorkerError, match="^cleanup watchdog could not confirm its readiness label$"
    ) as captured:
        run_endpoint_watchdog(
            controller_job_id="controller",
            controller_namespace="org",
            endpoint_name="endpoint",
            endpoint_namespace="org",
            run_id="run-1",
            token_secret_name="HF_TOKEN",
            timeout_seconds=60,
            api=FakeApi(),
            claim_store=claims,
            runner=runner,
            monotonic=lambda: 0,
        )

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert claims.claims == {}
    assert len(claims.released) == 1
    assert [command[2] for command in runner.commands] == ["pause", "describe"]


def test_endpoint_watchdog_rejects_competing_claim_before_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("JOB_ID", "watchdog-job")
    claims = FakeClaimStore()
    claims.acquire(
        "endpoint-leases/aa3808503c913daab53ed1415fe04988.json",
        {"controller_job_id": "other", "watchdog_job_id": "other"},
    )
    api = WatchdogApiStub()

    with pytest.raises(
        WorkerError, match="^endpoint lease is held by another watchdog$"
    ):
        run_endpoint_watchdog(
            controller_job_id="controller",
            controller_namespace="org",
            endpoint_name="endpoint",
            endpoint_namespace="org",
            run_id="run-1",
            token_secret_name="HF_TOKEN",
            timeout_seconds=60,
            api=api,
            claim_store=claims,
        )

    assert api.label_updates == []


def test_endpoint_watchdog_retains_lease_when_pause_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeApi(WatchdogApiStub):
        def inspect_job(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                status=SimpleNamespace(stage=SimpleNamespace(value="COMPLETED"))
            )

    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("JOB_ID", "watchdog-job")
    claims = FakeClaimStore()

    with pytest.raises(RuntimeError, match="^pause failed with test-token$"):
        run_endpoint_watchdog(
            controller_job_id="controller",
            controller_namespace="org",
            endpoint_name="endpoint",
            endpoint_namespace="org",
            run_id="run-1",
            token_secret_name="HF_TOKEN",
            timeout_seconds=60,
            api=FakeApi(),
            claim_store=claims,
            runner=CleanupFailureRunner([]),
            monotonic=lambda: 0,
        )

    assert len(claims.claims) == 1
    assert claims.released == []


def test_endpoint_watchdog_pauses_at_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspections: list[str] = []

    class FakeApi(WatchdogApiStub):
        def inspect_job(self, **_kwargs: object) -> object:
            inspections.append("checked")
            return SimpleNamespace(
                status=SimpleNamespace(stage=SimpleNamespace(value="RUNNING"))
            )

    times = iter([0.0, 0.0, 1.0, 2.0])
    runner = EndpointRunner([snapshot("paused", 0)])
    monkeypatch.setenv("BENCH_TOKEN", "secret")
    monkeypatch.setenv("HF_TOKEN", "stale-token")
    monkeypatch.setenv("JOB_ID", "watchdog-job")

    sleeps: list[float] = []
    run_endpoint_watchdog(
        controller_job_id="controller",
        controller_namespace="org",
        endpoint_name="endpoint",
        endpoint_namespace="org",
        run_id="run-1",
        token_secret_name="BENCH_TOKEN",
        timeout_seconds=2,
        api=FakeApi(),
        claim_store=FakeClaimStore(),
        runner=runner,
        sleep=sleeps.append,
        monotonic=lambda: next(times),
        poll_seconds=4,
    )

    assert inspections == ["checked", "checked"]
    assert sleeps == [4, 4]
    assert os.environ["HF_TOKEN"] == "secret"


def test_endpoint_watchdog_requires_configured_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_TOKEN", raising=False)

    with pytest.raises(WorkerError, match="required secret MISSING_TOKEN"):
        run_endpoint_watchdog(
            controller_job_id="controller",
            controller_namespace="org",
            endpoint_name="endpoint",
            endpoint_namespace="org",
            run_id="run-1",
            token_secret_name="MISSING_TOKEN",
            timeout_seconds=1,
        )


def test_endpoint_watchdog_requires_its_job_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.delenv("JOB_ID", raising=False)

    with pytest.raises(WorkerError, match="^watchdog JOB_ID is required$"):
        run_endpoint_watchdog(
            controller_job_id="controller",
            controller_namespace="org",
            endpoint_name="endpoint",
            endpoint_namespace="org",
            run_id="run-1",
            token_secret_name="HF_TOKEN",
            timeout_seconds=1,
            api=WatchdogApiStub(),
        )


def test_endpoint_watchdog_builds_authenticated_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[object] = []

    class FakeApi(WatchdogApiStub):
        def __init__(self, *, token: str) -> None:
            super().__init__()
            assert token == "secret"
            instances.append(self)

        def inspect_job(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                status=SimpleNamespace(stage=SimpleNamespace(value="COMPLETED"))
            )

    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("JOB_ID", "watchdog-job")
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    runner = EndpointRunner([snapshot("paused", 0)])

    run_endpoint_watchdog(
        controller_job_id="controller",
        controller_namespace="org",
        endpoint_name="endpoint",
        endpoint_namespace="org",
        run_id="run-1",
        token_secret_name="HF_TOKEN",
        timeout_seconds=1,
        runner=runner,
        monotonic=lambda: 0,
    )

    assert len(instances) == 1


def test_job_stage_reads_hf_enum_value() -> None:
    info = SimpleNamespace(
        status=SimpleNamespace(stage=SimpleNamespace(value="completed"))
    )

    assert _job_stage(info) == "COMPLETED"


def test_job_stage_reads_hf_string_value() -> None:
    assert _job_stage(SimpleNamespace(status=SimpleNamespace(stage="running"))) == (
        "RUNNING"
    )


def test_job_stage_rejects_invalid_value() -> None:
    with pytest.raises(WorkerError, match="^HF Job response has an invalid stage$"):
        _job_stage(SimpleNamespace(status=SimpleNamespace(stage=3)))


def test_cleanup_timeout() -> None:
    times = iter([0.0, 0.25, 0.5, 2.0])
    manager = EndpointManager(
        "org",
        "endpoint",
        EndpointRunner([snapshot("pausing", 1)]),
        sleep=lambda _: None,
        monotonic=lambda: next(times),
    )

    with pytest.raises(WorkerError, match="cleanup timed out"):
        manager.pause_and_verify(timeout_seconds=1)


def test_cleanup_retries_transient_pause_and_describe_failures() -> None:
    class TransientRunner:
        def __init__(self) -> None:
            self.commands: list[list[str]] = []
            self.attempts = {"pause": 0, "describe": 0}

        def run_json(
            self,
            command: Sequence[str],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, object]:
            self.commands.append(list(command))
            operation = command[2]
            self.attempts[operation] += 1
            if self.attempts[operation] == 1:
                raise ProcessError(f"transient {operation} failure")
            return snapshot("paused", 0)

        def run_text(self, command: Sequence[str]) -> str:
            raise AssertionError(command)

    runner = TransientRunner()
    sleeps: list[float] = []
    times = iter([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    manager = EndpointManager(
        "org",
        "endpoint",
        runner,
        sleep=sleeps.append,
        monotonic=lambda: next(times),
    )

    assert endpoint_state(manager.pause_and_verify(10, poll_seconds=2)) == (
        "paused",
        0,
        1,
    )
    assert runner.attempts == {"pause": 2, "describe": 2}
    assert sleeps == [2]


def test_cleanup_poll_recovers_ambiguous_pause_request() -> None:
    class AmbiguousPauseRunner:
        def run_json(
            self,
            command: Sequence[str],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, object]:
            if command[2] == "pause":
                raise ProcessError("pause response was lost")
            return snapshot("pausing", 1)

        def run_text(self, command: Sequence[str]) -> str:
            raise AssertionError(command)

    manager = EndpointManager("org", "endpoint", AmbiguousPauseRunner())

    accepted, observed, state, ready, error = manager._poll_pause(False)

    assert accepted is True
    assert observed == snapshot("pausing", 1)
    assert state == "pausing"
    assert ready == 1
    assert isinstance(error, ProcessError)
    assert str(error) == "pause response was lost"


def test_cleanup_poll_reports_transient_describe_failure() -> None:
    class DescribeFailureRunner:
        def run_json(
            self,
            command: Sequence[str],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, object]:
            if command[2] == "describe":
                raise ProcessError("describe unavailable")
            return snapshot("pausing", 1)

        def run_text(self, command: Sequence[str]) -> str:
            raise AssertionError(command)

    result = EndpointManager("org", "endpoint", DescribeFailureRunner())._poll_pause(
        False
    )

    accepted, observed, state, ready, error = result
    assert accepted is True
    assert observed is None
    assert state == "unknown"
    assert ready == -1
    assert isinstance(error, ProcessError)
    assert str(error) == "describe unavailable"


def test_cleanup_timeout_reports_last_transient_provider_error() -> None:
    class UnavailableRunner:
        def run_json(
            self,
            command: Sequence[str],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, object]:
            raise ProcessError(f"{command[2]} unavailable")

        def run_text(self, command: Sequence[str]) -> str:
            raise AssertionError(command)

    times = iter([0.0, 0.25, 0.5, 2.0])
    manager = EndpointManager(
        "org",
        "endpoint",
        UnavailableRunner(),
        sleep=lambda _: None,
        monotonic=lambda: next(times),
    )

    with pytest.raises(
        WorkerError,
        match="cleanup timed out after transient provider errors: describe unavailable",
    ):
        manager.pause_and_verify(timeout_seconds=1)


def test_harbor_command_is_pinned_and_bounded(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = build_run_lock(remote_spec)

    source = tmp_path / "harbor-source"
    command = build_harbor_command(lock, tmp_path, "https://endpoint.example", source)

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
        str(tmp_path.parent / "harbor-job.json"),
        "--yes",
    ]
    request = build_execution_request(
        lock,
        tmp_path,
        "https://endpoint.example",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    agents = request.harbor_config["agents"]
    assert isinstance(agents, list)
    assert agents[0]["kwargs"] == {
        "compaction": True,
        "thinking": "off",
        "version": "2026.7.2",
    }
    assert _expected_agent_version(lock) == "2026.7.2"


def test_harbor_source_agent_uses_reported_identity_without_version_override(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    remote = remote_spec.remote
    assert remote is not None
    agent = remote_spec.matrix.agents[0].model_copy(
        update={
            "revision": remote.harbor.source.revision,
            "revision_kind": "harbor-source",
            "reported_version": "2.0.0",
        }
    )
    spec = remote_spec.model_copy(
        update={"matrix": remote_spec.matrix.model_copy(update={"agents": [agent]})}
    )

    lock = build_run_lock(spec)
    request = build_execution_request(
        lock,
        tmp_path,
        "https://endpoint.example",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    agents = request.harbor_config["agents"]
    assert isinstance(agents, list)
    assert "version" not in agents[0]["kwargs"]
    assert _expected_agent_version(lock) == "2.0.0"


def test_package_agent_version_is_serialized_as_a_string(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    agent = remote_spec.matrix.agents[0].model_copy(update={"revision": "1"})
    spec = remote_spec.model_copy(
        update={"matrix": remote_spec.matrix.model_copy(update={"agents": [agent]})}
    )

    lock = build_run_lock(spec)
    request = build_execution_request(
        lock,
        tmp_path,
        "https://endpoint.example",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    agents = request.harbor_config["agents"]
    assert isinstance(agents, list)
    assert agents[0]["kwargs"]["version"] == "1"


def test_mark_watchdog_ready_publishes_complete_identity() -> None:
    api = WatchdogApiStub()

    _mark_watchdog_ready(
        api,
        "watchdog-job",
        "controller-namespace",
        "endpoint-namespace",
        "endpoint-name",
        "run-1",
    )

    assert api.label_updates == [
        {
            "job_id": "watchdog-job",
            "labels": {
                "harbor-hf-watchdog": "run-1",
                "harbor-hf-endpoint": "89d80c87fed8e87c598b0c6ddc685e46",
                "harbor-hf-watchdog-ready": "true",
            },
            "namespace": "controller-namespace",
        }
    ]


def test_watchdog_readiness_helper_forwards_complete_identity() -> None:
    api = WatchdogApiStub()

    error = _watchdog_readiness_error(
        api,
        "watchdog-job",
        "controller-namespace",
        "endpoint-namespace",
        "endpoint-name",
        "run-1",
    )

    assert error is None
    assert api.label_updates == [
        {
            "job_id": "watchdog-job",
            "labels": {
                "harbor-hf-watchdog": "run-1",
                "harbor-hf-endpoint": "89d80c87fed8e87c598b0c6ddc685e46",
                "harbor-hf-watchdog-ready": "true",
            },
            "namespace": "controller-namespace",
        }
    ]


def test_controller_environment_records_only_reproducibility_fields(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = build_run_lock(remote_spec)
    monkeypatch.setenv("JOB_ID", "job-123")
    monkeypatch.setenv("ACCELERATOR", "none")
    monkeypatch.setenv("CPU_CORES", "2")
    monkeypatch.setenv("MEMORY", "16Gi")

    result = controller_environment(lock)

    assert result["job_id"] == "job-123"
    assert result["namespace"] == "osolmaz"
    assert result["requested_flavor"] == "cpu-basic"
    assert result["reported_accelerator"] == "none"
    assert result["reported_cpu_cores"] == "2"
    assert result["reported_memory"] == "16Gi"
    assert set(result) == {
        "job_id",
        "namespace",
        "requested_image",
        "requested_flavor",
        "reported_accelerator",
        "reported_cpu_cores",
        "reported_memory",
        "python",
        "platform",
    }


def test_validate_harbor_result_requires_one_numeric_verifier(tmp_path: Path) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "verifier_result": {"rewards": {"reward": 0.5}},
            }
        )
    )

    assert validate_harbor_result(tmp_path) == {
        "trial_count": 1,
        "trials": [{"task_name": "task", "rewards": {"reward": 0.5}}],
    }
    nested = trial / "artifacts"
    nested.mkdir()
    (nested / "result.json").write_text("not json", encoding="utf-8")
    assert validate_harbor_result(tmp_path)["trial_count"] == 1
    (trial / "result.json").write_text(
        json.dumps(
            {"task_name": "task", "verifier_result": {"rewards": {"reward": True}}}
        )
    )
    with pytest.raises(WorkerError, match="finite numbers"):
        validate_harbor_result(tmp_path)


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf")])
def test_validate_harbor_result_rejects_nonfinite_reward(
    tmp_path: Path, reward: float
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "verifier_result": {"rewards": {"reward": reward}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match="finite numbers"):
        validate_harbor_result(tmp_path)


def test_validate_harbor_result_rejects_missing_and_multiple_trials(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        WorkerError, match="^expected exactly 1 Harbor trials, found 0$"
    ):
        validate_harbor_result(tmp_path)

    for name in ("one", "two"):
        trial = tmp_path / "job" / name
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": name,
                    "verifier_result": {"rewards": {"reward": 1}},
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(
        WorkerError, match="^expected exactly 1 Harbor trials, found 2$"
    ):
        validate_harbor_result(tmp_path)


def test_validate_harbor_result_requires_rewards(tmp_path: Path) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    result = trial / "result.json"
    result.write_text(
        json.dumps({"task_name": "task", "verifier_result": {"rewards": {}}}),
        encoding="utf-8",
    )

    with pytest.raises(
        WorkerError, match="^Harbor trial task has no verifier rewards$"
    ):
        validate_harbor_result(tmp_path)


def test_validate_harbor_result_rejects_malformed_trial_metadata(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text("{}", encoding="utf-8")

    with pytest.raises(WorkerError, match="^Harbor produced a malformed trial result$"):
        validate_harbor_result(tmp_path)


@pytest.mark.parametrize("task_name", [None, 3, "", " "])
def test_validate_harbor_result_rejects_non_string_or_empty_task_name(
    tmp_path: Path, task_name: object
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match="^Harbor produced a malformed trial result$"):
        validate_harbor_result(
            tmp_path, expected_trials=None, expected_task_names=("*",)
        )


def test_validate_harbor_result_requires_matching_trial_task_digest(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    expected = {"task": "sha256:" + "3" * 64}

    with pytest.raises(WorkerError, match="is not in the resolved task set"):
        validate_harbor_result(
            tmp_path,
            expected_task_digests={"other": "sha256:" + "3" * 64},
        )

    with pytest.raises(WorkerError, match="has no valid task lock"):
        validate_harbor_result(tmp_path, expected_task_digests=expected)

    (trial / "lock.json").write_text(
        json.dumps({"task": {"digest": "sha256:" + "4" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(WorkerError, match="task digest does not match"):
        validate_harbor_result(tmp_path, expected_task_digests=expected)

    (trial / "lock.json").write_text(
        json.dumps({"task": {"digest": expected["task"]}}),
        encoding="utf-8",
    )
    assert (
        validate_harbor_result(tmp_path, expected_task_digests=expected)["trial_count"]
        == 1
    )


def test_validate_harbor_result_rejects_trial_exception(tmp_path: Path) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "exception_info": {"exception_type": "AgentError"},
                "verifier_result": {"rewards": {"reward": 0.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match="^Harbor trial task failed with AgentError$"):
        validate_harbor_result(tmp_path)


def test_validate_harbor_result_rejects_step_exception_despite_reward(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "step_results": [
                    {"step_name": "environment_setup", "exception_info": None},
                    {
                        "step_name": "agent_setup",
                        "exception_info": {"exception_type": "AgentSetupError"},
                    },
                ],
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        WorkerError,
        match="^Harbor trial task step agent_setup failed with AgentSetupError$",
    ):
        validate_harbor_result(tmp_path)


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (
            {"exception_info": "failed"},
            "Harbor trial task step 1 failed with str",
        ),
        (
            {"exception_info": {}},
            "Harbor trial task step 1 failed with an exception",
        ),
    ],
)
def test_validate_harbor_result_describes_unnamed_step_exceptions(
    tmp_path: Path,
    step: dict[str, object],
    message: str,
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "step_results": [step],
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match=f"^{message}$"):
        validate_harbor_result(tmp_path)


@pytest.mark.parametrize(
    ("step_results", "message"),
    [
        ({}, "Harbor trial task step results failed with malformed result"),
        (["bad"], "Harbor trial task step 1 failed with malformed result"),
    ],
)
def test_validate_harbor_result_rejects_malformed_step_results(
    tmp_path: Path,
    step_results: object,
    message: str,
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "step_results": step_results,
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match=f"^{message}$"):
        validate_harbor_result(tmp_path)


def test_validate_harbor_result_enforces_agent_identity(tmp_path: Path) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "agent_info": {"name": "openclaw", "version": "wrong"},
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        WorkerError,
        match="^Harbor trial task agent identity does not match the lock$",
    ):
        validate_harbor_result(
            tmp_path,
            expected_agent_name="openclaw",
            expected_agent_version="1.2.3",
        )

    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(WorkerError, match="has no agent identity"):
        validate_harbor_result(
            tmp_path,
            expected_agent_name="openclaw",
            expected_agent_version="1.2.3",
        )

    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "step_results": [{"step_name": "agent", "exception_info": None}],
                "agent_info": {"name": "openclaw", "version": "1.2.3"},
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    assert validate_harbor_result(
        tmp_path,
        expected_agent_name="openclaw",
        expected_agent_version="1.2.3",
    ) == {
        "trial_count": 1,
        "trials": [{"task_name": "task", "rewards": {"reward": 1.0}}],
    }


def test_validate_harbor_result_enforces_model_identity(tmp_path: Path) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)

    def write_model(model_info: object) -> None:
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "task",
                    "agent_info": {
                        "name": "openclaw",
                        "version": "1.2.3",
                        "model_info": model_info,
                    },
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            ),
            encoding="utf-8",
        )

    def validate() -> dict[str, object]:
        return validate_harbor_result(
            tmp_path,
            expected_agent_name="openclaw",
            expected_agent_version="1.2.3",
            expected_model_provider="openai",
            expected_model_name="/repository",
        )

    write_model({"provider": "openai", "name": "different"})
    with pytest.raises(WorkerError, match="model identity does not match the lock"):
        validate()

    write_model(None)
    with pytest.raises(WorkerError, match="has no model identity"):
        validate()

    write_model({"provider": "openai", "name": "/repository"})
    assert validate()["trial_count"] == 1


@pytest.mark.parametrize(
    ("exception_info", "message"),
    [
        ([], "Harbor trial task failed with list"),
        ({}, "Harbor trial task failed with an exception"),
    ],
)
def test_validate_harbor_result_rejects_malformed_trial_exception(
    tmp_path: Path,
    exception_info: object,
    message: str,
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task",
                "exception_info": exception_info,
                "verifier_result": {"rewards": {"reward": 0.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match=f"^{message}$"):
        validate_harbor_result(tmp_path)


def test_validate_harbor_result_accepts_every_expected_attempt(
    tmp_path: Path,
) -> None:
    for ordinal in (1, 2):
        trial = tmp_path / "job" / str(ordinal)
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "task",
                    "verifier_result": {"rewards": {"reward": ordinal / 2}},
                }
            ),
            encoding="utf-8",
        )

    assert validate_harbor_result(tmp_path, 2) == {
        "trial_count": 2,
        "trials": [
            {"task_name": "task", "rewards": {"reward": 0.5}},
            {"task_name": "task", "rewards": {"reward": 1.0}},
        ],
    }


def test_expected_trial_count_scales_explicit_tasks_and_attempts(
    remote_spec: ExperimentSpec,
) -> None:
    benchmark = remote_spec.benchmark.model_copy(
        update={
            "task_names": ["one", "two"],
            "task_digests": {
                "one": "sha256:" + "3" * 64,
                "two": "sha256:" + "4" * 64,
            },
        }
    )
    execution = remote_spec.execution.model_copy(update={"attempts": 3})
    lock = build_run_lock(
        remote_spec.model_copy(update={"benchmark": benchmark, "execution": execution})
    )

    assert _expected_trial_count(lock) == 6
    assert _expected_task_counts(lock) == {"one": 3, "two": 3}


@pytest.mark.parametrize(
    ("pattern", "resolved_task"),
    [
        ("*", "selected"),
        ("shell-*", "shell-one"),
        ("task?", "task1"),
        ("task[12]", "task2"),
    ],
)
def test_expected_trial_count_uses_resolved_tasks_for_patterns(
    remote_spec: ExperimentSpec, pattern: str, resolved_task: str
) -> None:
    benchmark = remote_spec.benchmark.model_copy(
        update={
            "task_names": [pattern],
            "task_digests": {resolved_task: "sha256:" + "3" * 64},
        }
    )
    lock = build_run_lock(remote_spec.model_copy(update={"benchmark": benchmark}))

    assert _expected_trial_count(lock) == 1
    assert _expected_task_counts(lock) == {resolved_task: 1}


def test_expected_task_counts_preserve_exact_tasks_mixed_with_patterns(
    remote_spec: ExperimentSpec,
) -> None:
    benchmark = remote_spec.benchmark.model_copy(
        update={
            "task_names": ["exact-task", "shell-*"],
            "task_digests": {
                "exact-task": "sha256:" + "3" * 64,
                "shell-selected": "sha256:" + "4" * 64,
            },
        }
    )
    lock = build_run_lock(remote_spec.model_copy(update={"benchmark": benchmark}))

    assert _expected_trial_count(lock) == 2
    assert _expected_task_counts(lock) == {"exact-task": 1, "shell-selected": 1}


def test_validate_harbor_result_rejects_duplicate_in_place_of_requested_task(
    tmp_path: Path,
) -> None:
    for ordinal in (1, 2):
        trial = tmp_path / "job" / str(ordinal)
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "one",
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(
        WorkerError,
        match="^Harbor trial task counts do not match the requested attempts$",
    ):
        validate_harbor_result(
            tmp_path,
            expected_trials=2,
            expected_task_counts={"one": 1, "two": 1},
        )


def test_task_count_validation_accepts_exact_attempts() -> None:
    trials: list[dict[str, object]] = [
        {"task_name": "one"},
        {"task_name": "two"},
        {"task_name": "one"},
        {"task_name": "two"},
    ]

    _validate_task_counts(trials, {"one": 2, "two": 2})
    _validate_task_counts(trials, {"one": 2, "two": 2}, 2)
    _validate_task_counts(trials, None)
    _validate_task_counts(trials, None, 2, ("o*", "two"))


def test_validate_harbor_result_requires_a_wildcard_selected_trial(
    tmp_path: Path,
) -> None:
    with pytest.raises(WorkerError, match="^Harbor produced no trials$"):
        validate_harbor_result(tmp_path, expected_trials=None)


def test_validate_harbor_result_requires_every_wildcard_attempt(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "selected-task",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match="task counts do not match"):
        validate_harbor_result(
            tmp_path,
            expected_trials=None,
            expected_attempts_per_task=2,
        )


def test_validate_harbor_result_rejects_task_outside_requested_patterns(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "database-test",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match="task counts do not match"):
        validate_harbor_result(
            tmp_path,
            expected_trials=None,
            expected_attempts_per_task=1,
            expected_task_names=("shell-*",),
        )


def test_validate_harbor_result_accepts_task_matching_requested_pattern(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "shell-selected",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    assert (
        validate_harbor_result(
            tmp_path,
            expected_trials=None,
            expected_attempts_per_task=1,
            expected_task_names=("shell-*",),
        )["trial_count"]
        == 1
    )


def test_validate_harbor_result_requires_exact_task_mixed_with_wildcard(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "shell-selected",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match="task counts do not match"):
        validate_harbor_result(
            tmp_path,
            expected_trials=None,
            expected_task_counts={"exact-task": 1},
            expected_attempts_per_task=1,
        )


def test_trial_count_accepts_nonempty_wildcard_results() -> None:
    _validate_trial_count([{}], None)


def test_trial_count_accepts_matching_explicit_results() -> None:
    _validate_trial_count([{}], 1)


def test_trial_count_rejects_explicit_mismatch() -> None:
    with pytest.raises(
        WorkerError, match="^expected exactly 1 Harbor trials, found 0$"
    ):
        _validate_trial_count([], 1)


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.read_limits: list[int] = []

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        self.read_limits.append(limit)
        return self.body


def test_runtime_probe_records_json_text_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health_response = FakeResponse(b'{"ok": true}')
    version_response = FakeResponse(b"v1.2.3\xff")
    responses: list[FakeResponse | Exception] = [
        health_response,
        version_response,
        urllib.error.URLError("unavailable"),
    ]

    requests: list[tuple[urllib.request.Request, int]] = []

    def open_url(request: urllib.request.Request, *, timeout: int) -> FakeResponse:
        requests.append((request, timeout))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("urllib.request.urlopen", open_url)

    result = probe_runtime("https://endpoint.example", "token")

    probes = cast(dict[str, dict[str, object]], result["probes"])
    assert probes == {
        "health": {
            "status": "reported",
            "http_status": 200,
            "value": {"ok": True},
        },
        "version": {
            "status": "reported",
            "http_status": 200,
            "value": "v1.2.3�",
        },
        "models": {"status": "unknown", "error_type": "URLError"},
    }
    assert [request.full_url for request, _timeout in requests] == [
        "https://endpoint.example/health",
        "https://endpoint.example/version",
        "https://endpoint.example/v1/models",
    ]
    assert [request.get_header("Authorization") for request, _ in requests] == [
        "Bearer token",
        "Bearer token",
        "Bearer token",
    ]
    assert [dict(request.header_items()) for request, _ in requests] == [
        {"Authorization": "Bearer token"},
        {"Authorization": "Bearer token"},
        {"Authorization": "Bearer token"},
    ]
    assert [timeout for _request, timeout in requests] == [60, 60, 60]
    assert health_response.read_limits == [1024 * 1024]
    assert version_response.read_limits == [1024 * 1024]


def test_runtime_probe_requires_healthy_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def unhealthy(request: urllib.request.Request, **_kwargs: object) -> FakeResponse:
        requests.append(request)
        return FakeResponse(b"bad", 503)

    monkeypatch.setattr(
        "urllib.request.urlopen",
        unhealthy,
    )

    with pytest.raises(
        WorkerError, match="^endpoint health probe did not return HTTP 200$"
    ):
        probe_runtime("https://endpoint.example", "token")

    assert [request.full_url for request in requests] == [
        "https://endpoint.example/health"
    ]


def test_runtime_probe_shares_deadline_across_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [10.0]
    requests: list[tuple[str, float]] = []

    def open_url(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        requests.append((request.full_url, timeout))
        if request.full_url.endswith("/health"):
            now[0] += 0.4
            return FakeResponse(b'{"ok": true}', 200)
        now[0] += timeout
        raise TimeoutError

    monkeypatch.setattr("urllib.request.urlopen", open_url)

    result = probe_runtime(
        "https://endpoint.example",
        "token",
        request_timeout_seconds=60,
        deadline=11.0,
        monotonic=lambda: now[0],
    )

    assert [request for request, _timeout in requests] == [
        "https://endpoint.example/health",
        "https://endpoint.example/version",
    ]
    assert [timeout for _request, timeout in requests] == pytest.approx([1.0, 0.6])
    probes = cast(dict[str, dict[str, object]], result["probes"])
    assert probes["models"] == {
        "status": "unknown",
        "error_type": "TimeoutError",
    }


def test_runtime_readiness_retries_gateway_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def flaky_probe(*_args: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise WorkerError("endpoint health probe did not return HTTP 200")
        return {"probes": {"health": {"http_status": 200}}}

    sleeps: list[float] = []
    monkeypatch.setattr("harbor_hf.worker.probe_runtime", flaky_probe)

    result = wait_for_runtime(
        "https://endpoint.example",
        "token",
        "/health",
        timeout_seconds=60,
        poll_seconds=5,
        sleep=sleeps.append,
    )

    assert result == {"probes": {"health": {"http_status": 200}}}
    assert attempts == 3
    assert sleeps == [5, 5]


def test_runtime_readiness_timeout_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unhealthy(*_args: object) -> dict[str, object]:
        raise WorkerError("endpoint health probe did not return HTTP 200")

    monkeypatch.setattr("harbor_hf.worker.probe_runtime", unhealthy)

    with pytest.raises(
        WorkerError, match="^endpoint runtime did not become healthy before timeout$"
    ) as caught:
        wait_for_runtime(
            "https://endpoint.example",
            "token",
            "/health",
            timeout_seconds=0,
        )

    assert isinstance(caught.value.__cause__, WorkerError)


def test_runtime_readiness_bounds_each_probe_by_shared_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [10.0]
    captured: list[tuple[float, float | None]] = []

    def bounded_probe(
        _base_url: str,
        _token: str,
        _health_route: str,
        timeout: float,
        deadline: float | None,
        _monotonic: object,
    ) -> dict[str, object]:
        captured.append((timeout, deadline))
        now[0] += 0.6
        raise WorkerError("not ready")

    monkeypatch.setattr("harbor_hf.worker.probe_runtime", bounded_probe)

    with pytest.raises(WorkerError, match="before timeout"):
        wait_for_runtime(
            "https://endpoint.example",
            "token",
            "/health",
            timeout_seconds=1,
            poll_seconds=1,
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            monotonic=lambda: now[0],
        )

    assert captured == [(1.0, 11.0)]


def test_finalize_evidence_scrubs_and_archives(tmp_path: Path) -> None:
    jobs = tmp_path / "harbor-jobs"
    jobs.mkdir()
    (jobs / "test-token.log").write_text("contains test-token", encoding="utf-8")
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")

    _finalize_evidence(tmp_path, "test-token")

    assert (jobs / "[REDACTED].log").read_text() == "contains [REDACTED]"
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "secret_paths_redacted",
        "secrets_redacted",
    ]
    assert events[0]["count"] == 1
    assert events[1]["files"] == ["harbor-jobs/[REDACTED].log"]
    assert (tmp_path / "artifacts.tar.gz").exists()
    assert b"test-token" not in (tmp_path / "artifacts.tar.gz").read_bytes()
    checksums = json.loads((tmp_path / "checksums.json").read_text())
    assert set(checksums) == {
        "artifacts.tar.gz",
        "events.jsonl",
        "harbor-jobs/[REDACTED].log",
    }


def test_failed_finalize_recreates_rejected_jobs_symlink(tmp_path: Path) -> None:
    outside = tmp_path.with_name(f"{tmp_path.name}-outside")
    outside.mkdir()
    jobs = tmp_path / "harbor-jobs"
    jobs.symlink_to(outside, target_is_directory=True)
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "_FAILED").write_text("\n", encoding="utf-8")

    _finalize_evidence(tmp_path, "test-token")

    assert jobs.is_dir()
    assert not jobs.is_symlink()
    rejection = json.loads(
        (tmp_path / "private-artifact-rejections.json").read_text(encoding="utf-8")
    )
    assert rejection["rejections"] == [
        {"path": "harbor-jobs", "reason": "symlink", "size": None}
    ]
    assert (tmp_path / "artifacts.tar.gz").is_file()


def test_failed_finalize_recreates_rejected_jobs_file(tmp_path: Path) -> None:
    jobs = tmp_path / "harbor-jobs"
    jobs.write_text("collision", encoding="utf-8")
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "_FAILED").write_text("\n", encoding="utf-8")

    _finalize_evidence(tmp_path, "test-token")

    assert jobs.is_dir()
    rejection = json.loads(
        (tmp_path / "private-artifact-rejections.json").read_text(encoding="utf-8")
    )
    assert rejection["rejections"] == [
        {"path": "harbor-jobs", "reason": "reserved_path", "size": 9}
    ]
    assert (tmp_path / "artifacts.tar.gz").is_file()


def test_failed_finalize_removes_forged_success_marker(tmp_path: Path) -> None:
    (tmp_path / "harbor-jobs").mkdir()
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "_FAILED").write_text("\n", encoding="utf-8")
    (tmp_path / "_SUCCESS").write_text("forged", encoding="utf-8")

    _finalize_evidence(tmp_path, "test-token")

    assert not (tmp_path / "_SUCCESS").exists()
    rejection = json.loads(
        (tmp_path / "private-artifact-rejections.json").read_text(encoding="utf-8")
    )
    assert {item["path"] for item in rejection["rejections"]} == {"_SUCCESS"}


def test_direct_run_rejects_unexpected_root_directory(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
) -> None:
    (tmp_path / "harbor-jobs").mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "unbounded.log").write_text("evidence", encoding="utf-8")
    lock = build_run_lock(remote_spec, run_id="unexpected-root-directory")

    with pytest.raises(RuntimeError, match="unexpected directory: extra"):
        _validate_direct_private_artifacts(tmp_path, lock)


def test_failed_finalize_removes_unexpected_root_directory(tmp_path: Path) -> None:
    (tmp_path / "harbor-jobs").mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "unbounded.log").write_text("evidence", encoding="utf-8")
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "_FAILED").write_text("\n", encoding="utf-8")

    _finalize_evidence(tmp_path, "test-token")

    assert not extra.exists()
    rejection = json.loads(
        (tmp_path / "private-artifact-rejections.json").read_text(encoding="utf-8")
    )
    assert {item["path"] for item in rejection["rejections"]} == {"extra"}


def test_success_finalize_revalidates_late_root_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "harbor-jobs").mkdir()
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")

    def refresh(root: Path, *, strict: bool) -> None:
        assert strict is True
        with (root / "late.log").open("wb") as stream:
            stream.truncate(64 * 1024 * 1024 + 1)

    monkeypatch.setattr("harbor_hf.worker.refresh_retained_bundle", refresh)

    with pytest.raises(RuntimeError, match="file size limit: late.log"):
        _finalize_evidence(tmp_path, "test-token")


def test_direct_jobs_root_files_are_bounded(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "harbor-jobs"
    jobs.mkdir()
    oversized = jobs / "oversized.log"
    with oversized.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    lock = build_run_lock(remote_spec, run_id="jobs-root-limits")

    with pytest.raises(RuntimeError, match="file size limit: oversized.log"):
        _validate_direct_private_artifacts(tmp_path, lock)

    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "_FAILED").write_text("\n", encoding="utf-8")
    _finalize_evidence(tmp_path, "test-token")

    assert not oversized.exists()
    rejection = json.loads(
        (jobs / "private-artifact-rejections.json").read_text(encoding="utf-8")
    )
    assert rejection["rejections"] == [
        {
            "path": "oversized.log",
            "reason": "file_size",
            "size": 64 * 1024 * 1024 + 1,
        }
    ]


def test_failed_direct_run_falls_back_from_undecodable_result(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
) -> None:
    trial = tmp_path / "harbor-jobs" / "job" / "trial-fallback"
    trial.mkdir(parents=True)
    (trial / "result.json").write_bytes(b"\xff")
    (tmp_path / "run.lock.json").write_text(
        build_run_lock(remote_spec, run_id="undecodable-result").model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "_FAILED").write_text("\n", encoding="utf-8")

    _finalize_evidence(tmp_path, "test-token")

    manifest = json.loads(
        (trial / "private-artifacts.json").read_text(encoding="utf-8")
    )
    assert manifest["execution_id"] == "undecodable-result"
    assert manifest["trial_id"] == "trial-fallback"


def test_failed_direct_run_preserves_attempt_state_before_pruning(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial = tmp_path / "harbor-jobs" / "job" / "trial-fallback"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text("{", encoding="utf-8")
    (tmp_path / "run.lock.json").write_text(
        build_run_lock(remote_spec, run_id="pruned-attempt").model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / "harbor-request.json").write_text(
        json.dumps({"verification": {"expected_agent_name": "openclaw"}}),
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"event": "harbor_started"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "_FAILED").write_text("\n", encoding="utf-8")
    pruned = False

    def sanitize(
        root: Path,
        *,
        trust_existing_rejections: bool = False,
        required_directories: tuple[str, ...] = (),
        preserved_files: tuple[str, ...] = (),
        allowed_directories: tuple[str, ...] | None = None,
    ) -> list[PrivateArtifactRejection]:
        nonlocal pruned
        if root == tmp_path and not pruned:
            pruned = True
            (root / "harbor-request.json").unlink()
            (root / "events.jsonl").unlink()
        return sanitize_private_artifact_directory_files(
            root,
            trust_existing_rejections=trust_existing_rejections,
            required_directories=required_directories,
            preserved_files=preserved_files,
            allowed_directories=allowed_directories,
        )

    monkeypatch.setattr(
        "harbor_hf.worker.sanitize_private_artifact_directory_files", sanitize
    )
    monkeypatch.setattr(
        "harbor_hf.worker.refresh_retained_bundle", lambda _root, *, strict: None
    )

    _finalize_evidence(tmp_path, "test-token")

    manifest = json.loads(
        (trial / "private-artifacts.json").read_text(encoding="utf-8")
    )
    assert manifest["requirements"] == [
        {
            "name": "openclaw_session_jsonl",
            "paths": [],
            "required": True,
            "satisfied": False,
        }
    ]


def test_failed_direct_run_sanitizes_each_trial_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "harbor-jobs" / "job" / "trial-one"
    second = tmp_path / "harbor-jobs" / "job" / "trial-two"
    first.mkdir(parents=True)
    second.mkdir()
    seen: list[Path] = []
    monkeypatch.setattr(
        "harbor_hf.worker.sanitize_private_artifact_tree",
        lambda root, **_kwargs: (seen.append(root), [])[1],
    )

    _sanitize_direct_trial_artifacts(tmp_path)

    assert seen == [first, second]


def test_failed_direct_run_does_not_follow_symlinked_job_root(tmp_path: Path) -> None:
    jobs = tmp_path / "harbor-jobs"
    jobs.mkdir()
    outside_trial = tmp_path / "outside" / "trial"
    outside_trial.mkdir(parents=True)
    outside_artifact = outside_trial / "large.log"
    outside_artifact.write_bytes(b"x" * 1024)
    (jobs / "linked-job").symlink_to(outside_trial.parent, target_is_directory=True)

    _sanitize_direct_trial_artifacts(tmp_path)

    assert outside_artifact.read_bytes() == b"x" * 1024


def test_failed_direct_run_keeps_nested_symlink_rejection_with_trial(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "harbor-jobs" / "job" / "trial"
    trial.mkdir(parents=True)
    outside = tmp_path / "outside.log"
    outside.write_text("outside", encoding="utf-8")
    linked = trial / "linked.log"
    linked.symlink_to(outside)

    sanitize_private_artifact_symlinks(tmp_path, max_depth=3)
    assert linked.is_symlink()
    _sanitize_direct_trial_artifacts(tmp_path)

    rejection = json.loads(
        (trial / "private-artifact-rejections.json").read_text(encoding="utf-8")
    )
    assert rejection["rejections"] == [
        {"path": "linked.log", "reason": "symlink", "size": None}
    ]
    assert outside.read_text(encoding="utf-8") == "outside"


def test_failed_direct_run_refreshes_compatibility_after_final_pruning(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial = tmp_path / "harbor-jobs" / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        '{"agent_info":{"name":"openclaw"},"agent_execution":null}\n',
        encoding="utf-8",
    )
    (tmp_path / "run.lock.json").write_text(
        build_run_lock(remote_spec, run_id="failed-refresh").model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "_FAILED").write_text("\n", encoding="utf-8")
    refresh_calls = 0

    def refresh(root: Path, *, strict: bool) -> None:
        nonlocal refresh_calls
        assert strict is False
        refresh_calls += 1
        if refresh_calls == 1:
            with (root / "harbor-jobs" / "job" / "trial" / "late.log").open(
                "wb"
            ) as stream:
                stream.truncate(64 * 1024 * 1024 + 1)

    monkeypatch.setattr("harbor_hf.worker.refresh_retained_bundle", refresh)

    _finalize_evidence(tmp_path, "test-token")

    assert refresh_calls == 2
    assert not (trial / "late.log").exists()


def test_publish_evidence_requires_one_terminal_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "record.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "bucket" / "runs" / "run-1"

    with pytest.raises(WorkerError, match="exactly one terminal marker"):
        _publish_evidence(source, destination)

    (source / "_SUCCESS").write_text("\n", encoding="utf-8")
    (source / "_FAILED").write_text("\n", encoding="utf-8")
    with pytest.raises(WorkerError, match="exactly one terminal marker"):
        _publish_evidence(source, destination)
    (source / "_FAILED").unlink()

    original_copytree = shutil.copytree

    def tracking_copytree(
        copied_source: Path,
        copied_destination: Path,
        **kwargs: object,
    ) -> Path:
        options = cast(dict[str, Any], kwargs)
        ignore = options["ignore"]
        assert callable(ignore)
        assert ignore(copied_source, ["record.json", "_SUCCESS"]) == ["_SUCCESS"]
        result = original_copytree(copied_source, copied_destination, **options)
        assert not (copied_destination / "_SUCCESS").exists()
        return result

    monkeypatch.setattr("harbor_hf.worker.shutil.copytree", tracking_copytree)
    _prepare_evidence_destination(destination)
    _publish_evidence(source, destination)

    assert (destination / "record.json").read_text(encoding="utf-8") == "{}\n"
    assert (destination / "_SUCCESS").read_text(encoding="utf-8") == "\n"


def test_prepare_evidence_destination_is_exclusive_and_creates_parents(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "bucket" / "runs" / "experiment" / "run-1"

    _prepare_evidence_destination(destination)

    assert (destination / "_RESERVED").read_text(encoding="utf-8") == "\n"
    with pytest.raises(FileExistsError):
        _prepare_evidence_destination(destination)


def test_adopting_reserved_evidence_removes_stale_partial_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "bucket" / "runs" / "run-1"
    destination.mkdir(parents=True)
    (destination / "_RESERVED").write_text("\n", encoding="utf-8")
    (destination / "failure.json").write_text("{}\n", encoding="utf-8")
    nested = destination / "harbor-jobs"
    nested.mkdir()
    (nested / "stale.txt").write_text("stale", encoding="utf-8")

    _prepare_evidence_destination(destination, adopt_reserved=True)

    assert [path.name for path in destination.iterdir()] == ["_RESERVED"]


def test_publish_evidence_preserves_nested_terminal_markers(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "_SUCCESS").write_text("\n", encoding="utf-8")
    (nested / "_SUCCESS").write_text("task marker\n", encoding="utf-8")
    (nested / "_FAILED").write_text("task failure\n", encoding="utf-8")
    destination = tmp_path / "bucket" / "runs" / "run-1"

    _prepare_evidence_destination(destination)
    _publish_evidence(source, destination)

    assert (destination / "nested" / "_SUCCESS").read_text(encoding="utf-8") == (
        "task marker\n"
    )
    assert (destination / "nested" / "_FAILED").read_text(encoding="utf-8") == (
        "task failure\n"
    )


def test_publish_evidence_preserves_incomplete_destination_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "record.json").write_text("{}\n", encoding="utf-8")
    (source / "_FAILED").write_text("\n", encoding="utf-8")
    destination = tmp_path / "bucket" / "runs" / "run-1"

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr("harbor_hf.worker.shutil.copyfile", fail_copy)

    _prepare_evidence_destination(destination)
    with pytest.raises(OSError, match="copy failed"):
        _publish_evidence(source, destination, attempts=1)

    assert destination.exists()
    assert (destination / "_RESERVED").is_file()


def test_publish_evidence_retries_transient_copy_without_losing_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "record.json").write_text("{}\n", encoding="utf-8")
    (source / "_SUCCESS").write_text("\n", encoding="utf-8")
    destination = tmp_path / "bucket" / "runs" / "run-1"
    _prepare_evidence_destination(destination)
    original = shutil.copyfile
    calls = 0

    def flaky_copy(
        source_path: Path,
        destination_path: Path,
        *,
        follow_symlinks: bool = True,
    ) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary bucket write failure")
        return str(
            original(
                source_path,
                destination_path,
                follow_symlinks=follow_symlinks,
            )
        )

    sleeps: list[float] = []
    monkeypatch.setattr("harbor_hf.worker.shutil.copyfile", flaky_copy)

    _publish_evidence(source, destination, sleep=sleeps.append)

    assert calls == 3
    assert sleeps == [1.0]
    assert (destination / "_SUCCESS").is_file()
    assert not (destination / "_RESERVED").exists()


def _write_lock(path: Path, lock: RunLock) -> None:
    path.write_text(lock.model_dump_json(), encoding="utf-8")


def _successful_stream(
    command: Sequence[str],
    log_path: Path,
    *,
    environment: dict[str, str],
    timeout_seconds: int,
) -> int:
    if "--output" in command and "--request-digest" in command:
        write_fake_compatibility_bundle(command, log_path)
        return 0
    assert environment == {
        "HF_TOKEN": "test-token",
        "OPENAI_API_KEY": "test-token",
        "OPENAI_BASE_URL": "https://endpoint.example/v1",
    }
    assert log_path.name == "harbor.log"
    assert timeout_seconds == 60
    config_path = Path(command[command.index("--config") + 1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["agents"][0]["extra_allowed_hosts"] == ["endpoint.example"]
    jobs_dir = Path(config["jobs_dir"])
    assert any(part.startswith("harbor-hf-run-") for part in jobs_dir.parts)
    trial = jobs_dir / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "cancel-async-tasks",
                "agent_info": {
                    "name": "openclaw",
                    "version": "2026.7.2",
                    "model_info": {"provider": "openai", "name": "/repository"},
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    (trial / "lock.json").write_text(
        json.dumps({"task": {"digest": "sha256:" + "2" * 64}}),
        encoding="utf-8",
    )
    log_path.write_text("completed test-token\n", encoding="utf-8")
    return 0


def test_worker_publishes_success_after_cleanup(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="successful")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    def fake_probe(
        url: str, token: str, health_route: str, *_deadline: object
    ) -> dict[str, object]:
        assert url == "https://endpoint.example"
        assert token == "test-token"
        assert health_route == "/ready"
        return {"probes": {"health": {"http_status": 200}}}

    monkeypatch.setattr("harbor_hf.worker.probe_runtime", fake_probe)
    runner = EndpointRunner(
        [
            snapshot("paused", 0),
            snapshot("running", 1),
            snapshot("paused", 0),
        ]
    )
    claims = FakeClaimStore()
    readiness_timeouts: list[int] = []
    original_wait_ready = EndpointManager.wait_ready

    def record_wait_ready(
        manager: EndpointManager, timeout_seconds: int, poll_seconds: float = 15
    ) -> dict[str, object]:
        readiness_timeouts.append(timeout_seconds)
        return original_wait_ready(manager, timeout_seconds, poll_seconds)

    monkeypatch.setattr(EndpointManager, "wait_ready", record_wait_ready)

    root = run_worker(
        remote_manifest,
        lock_path,
        tmp_path / "output",
        runner=runner,
        stream_runner=_successful_stream,
        source_preparer=_prepare_source,
        watchdog_launcher=_launch_watchdog,
        claim_store=claims,
    )

    assert (root / "_SUCCESS").exists()
    assert list(claims.claims) == [
        run_claim_path(lock.artifact_bucket, lock.artifact_prefix)
    ]
    assert readiness_timeouts == [3600]
    assert not (root / "_FAILED").exists()
    assert json.loads((root / "verification.json").read_text()) == {
        "trial_count": 1,
        "trials": [
            {
                "task_name": "cancel-async-tasks",
                "rewards": {"reward": 1.0},
            }
        ],
    }
    assert endpoint_state(json.loads((root / "endpoint.final.json").read_text())) == (
        "paused",
        0,
        1,
    )
    assert b"test-token" not in (root / "artifacts.tar.gz").read_bytes()
    assert (root / "harbor.log").read_text() == "completed [REDACTED]\n"
    assert json.loads((root / "run.lock.json").read_text()) == lock.model_dump(
        mode="json"
    )
    expected_snapshot = snapshot("running", 1)
    cast(dict[str, object], expected_snapshot["model"])["secrets"] = "[REDACTED]"
    assert json.loads((root / "endpoint.snapshot.json").read_text()) == (
        expected_snapshot
    )
    runtime = json.loads((root / "runtime-environment.json").read_text())
    assert set(runtime) == {"controller", "endpoint"}
    assert runtime["endpoint"] == {"probes": {"health": {"http_status": 200}}}
    assert runtime["controller"]["namespace"] == "osolmaz"
    assert runtime["controller"]["requested_flavor"] == "cpu-basic"
    assert sorted(path.name for path in root.iterdir()) == [
        "_SUCCESS",
        "artifacts.tar.gz",
        "checksums.json",
        "endpoint.final.json",
        "endpoint.snapshot.json",
        "events.jsonl",
        "harbor-compatibility.json",
        "harbor-export.log",
        "harbor-job.json",
        "harbor-jobs",
        "harbor-native-bundle.json",
        "harbor-request.json",
        "harbor.log",
        "manifest.yaml",
        "run.lock.json",
        "runtime-environment.json",
        "verification.json",
    ]
    assert (root / "_SUCCESS").read_text(encoding="utf-8") == "\n"
    event_records = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    assert [
        {key: value for key, value in record.items() if key != "at"}
        for record in event_records
    ] == [
        {"event": "worker_started", "run_id": "successful"},
        {"event": "endpoint_baseline_validated"},
        {
            "event": "endpoint_lease_acquired",
            "watchdog_job_id": "watchdog-job",
        },
        {"event": "cleanup_watchdog_started", "job_id": "watchdog-job"},
        {"event": "endpoint_resume_requested"},
        {"event": "endpoint_ready", "state": "running"},
        {"event": "runtime_probed"},
        {"event": "harbor_started"},
        {"event": "harbor_finished", "exit_code": 0},
        {"event": "verification_validated"},
        {"event": "endpoint_pause_requested"},
        {
            "event": "endpoint_paused",
            "ready_replicas": 0,
            "state": "paused",
            "target_replicas": 1,
        },
        {"event": "run_succeeded"},
        {"event": "secrets_redacted", "files": ["harbor.log"]},
    ]
    checksums = json.loads((root / "checksums.json").read_text())
    assert set(checksums) == {
        "artifacts.tar.gz",
        "endpoint.final.json",
        "endpoint.snapshot.json",
        "events.jsonl",
        "harbor-compatibility.json",
        "harbor-export.log",
        "harbor-job.json",
        "harbor-jobs/job/trial/result.json",
        "harbor-jobs/job/trial/lock.json",
        "harbor-jobs/job/trial/private-artifacts.json",
        "harbor-native-bundle.json",
        "harbor.log",
        "harbor-request.json",
        "manifest.yaml",
        "run.lock.json",
        "runtime-environment.json",
        "verification.json",
    }
    assert runner.commands == [
        [
            "hf",
            "endpoints",
            operation,
            "qwen-endpoint",
            "--namespace",
            "osolmaz",
            "--format",
            "json",
        ]
        for operation in ("describe", "resume", "describe", "pause", "describe")
    ]


def test_direct_worker_fails_and_publishes_when_openclaw_session_is_missing(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="missing-session")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        "harbor_hf.worker.probe_runtime",
        lambda *_args: {"probes": {"health": {"http_status": 200}}},
    )

    def stream(
        command: Sequence[str],
        log_path: Path,
        *,
        environment: dict[str, str],
        timeout_seconds: int,
    ) -> int:
        exit_code = _successful_stream(
            command,
            log_path,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
        if "--config" in command:
            config_path = Path(command[command.index("--config") + 1])
            config = json.loads(config_path.read_text(encoding="utf-8"))
            result_path = Path(config["jobs_dir"]) / "job" / "trial" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["agent_execution"] = {"started_at": "2026-07-14T00:00:00Z"}
            result_path.write_text(json.dumps(result), encoding="utf-8")
        return exit_code

    runner = EndpointRunner(
        [snapshot("paused", 0), snapshot("running", 1), snapshot("paused", 0)]
    )

    with pytest.raises(WorkerError, match="no session JSONL"):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            stream_runner=stream,
            source_preparer=_prepare_source,
            watchdog_launcher=_launch_watchdog,
            claim_store=FakeClaimStore(),
        )

    root = tmp_path / "output" / lock.artifact_prefix
    assert (root / "_FAILED").is_file()
    manifest = json.loads(
        (root / "harbor-jobs/job/trial/private-artifacts.json").read_text()
    )
    assert manifest["execution_id"] == lock.run_id
    assert manifest["trial_id"] == "00000000-0000-0000-0000-000000000001"
    assert manifest["requirements"] == [
        {
            "name": "openclaw_session_jsonl",
            "paths": [],
            "required": True,
            "satisfied": False,
        }
    ]


def test_worker_refuses_existing_run_prefix_before_remote_work(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="existing")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    runner = EndpointRunner([])
    claims = FakeClaimStore()
    claims.acquire(
        run_claim_path(lock.artifact_bucket, lock.artifact_prefix),
        {"run_id": "other"},
    )

    with pytest.raises(WorkerError, match="^run ID is already reserved$"):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            claim_store=claims,
        )

    assert runner.commands == []


def test_failed_run_claim_release_cannot_mask_failure_and_claim_expires(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingReleaseClaims(FakeClaimStore):
        def release(self, path: str, owner: Mapping[str, str]) -> None:
            del path, owner
            raise RuntimeError("release transport failed")

    lock = build_run_lock(remote_spec, run_id="release-failed")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    destination = tmp_path / "output" / lock.artifact_prefix
    destination.mkdir(parents=True)
    claims = FailingReleaseClaims()

    with pytest.raises(FileExistsError):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=EndpointRunner([]),
            claim_store=claims,
        )

    assert len(claims.acquired) == 1
    assert "expires_at" in claims.acquired[0][1]


def test_worker_rejects_incomplete_explicit_task_set(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = remote_spec.benchmark.model_copy(
        update={
            "task_names": ["cancel-async-tasks", "second-task"],
            "task_digests": {
                "cancel-async-tasks": "sha256:" + "2" * 64,
                "second-task": "sha256:" + "3" * 64,
            },
        }
    )
    spec = remote_spec.model_copy(update={"benchmark": benchmark})
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(spec.model_dump_json(), encoding="utf-8")
    lock = build_run_lock(spec, run_id="incomplete-task-set")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        "harbor_hf.worker.probe_runtime",
        lambda _url, _token, _health_route, *_deadline: {"probes": {}},
    )
    runner = EndpointRunner(
        [snapshot("paused", 0), snapshot("running", 1), snapshot("paused", 0)]
    )

    with pytest.raises(
        WorkerError, match="^expected exactly 2 Harbor trials, found 1$"
    ):
        run_worker(
            manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            stream_runner=_successful_stream,
            source_preparer=_prepare_source,
            watchdog_launcher=_launch_watchdog,
        )

    assert [command[2] for command in runner.commands][-2:] == ["pause", "describe"]


def test_worker_failure_still_pauses_endpoint(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="failed")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        "harbor_hf.worker.probe_runtime",
        lambda _url, _token, _health_route, *_deadline: {"probes": {}},
    )
    runner = EndpointRunner(
        [snapshot("paused", 0), snapshot("running", 1), snapshot("paused", 0)]
    )

    with pytest.raises(WorkerError, match="status 7"):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            stream_runner=lambda *_args, **_kwargs: 7,
            source_preparer=_prepare_source,
            watchdog_launcher=_launch_watchdog,
        )

    root = tmp_path / "output" / lock.artifact_prefix
    assert (root / "_FAILED").exists()
    assert not (root / "_SUCCESS").exists()
    assert [command[2] for command in runner.commands][-2:] == ["pause", "describe"]
    events = [
        json.loads(line)["event"]
        for line in (root / "events.jsonl").read_text().splitlines()
    ]
    assert events[-2:] == ["endpoint_paused", "run_failed"]


def test_worker_without_endpoint_lease_never_pauses_endpoint(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="lease-lost")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    runner = EndpointRunner([snapshot("paused", 0)])

    def reject_lease(_lock: RunLock, _endpoint: object, _token: str) -> str:
        raise WorkerError("endpoint lease is held by another watchdog")

    with pytest.raises(
        WorkerError, match="^endpoint lease is held by another watchdog$"
    ):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            source_preparer=_prepare_source,
            watchdog_launcher=reject_lease,
        )

    root = tmp_path / "output" / lock.artifact_prefix
    events = [
        json.loads(line)["event"]
        for line in (root / "events.jsonl").read_text().splitlines()
    ]
    assert [command[2] for command in runner.commands] == ["describe"]
    assert events == [
        "worker_started",
        "endpoint_baseline_validated",
        "endpoint_cleanup_skipped",
        "run_failed",
    ]


def test_worker_marks_failed_when_success_evidence_cannot_finalize(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="finalization-failed")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        "harbor_hf.worker.probe_runtime",
        lambda _url, _token, _health_route, *_deadline: {"probes": {}},
    )
    staged_roots: list[Path] = []

    def fail_finalization(
        root: Path, token: str, *, strict_compatibility: bool
    ) -> None:
        staged_roots.append(root)
        assert tmp_path / "output" not in root.parents
        assert token == "test-token"
        assert strict_compatibility is True
        raise RuntimeError("archive test-token failed")

    monkeypatch.setattr("harbor_hf.worker._finalize_evidence", fail_finalization)
    runner = EndpointRunner(
        [snapshot("paused", 0), snapshot("running", 1), snapshot("paused", 0)]
    )
    claims = FakeClaimStore()

    with pytest.raises(WorkerError, match="evidence finalization failed"):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            stream_runner=_successful_stream,
            source_preparer=_prepare_source,
            watchdog_launcher=_launch_watchdog,
            claim_store=claims,
        )

    root = tmp_path / "output" / lock.artifact_prefix
    assert root.exists()
    assert (root / "_RESERVED").is_file()
    assert claims.claims == {}
    assert len(claims.released) == 1
    assert len(staged_roots) == 1
    assert not staged_roots[0].exists()


def test_worker_does_not_resume_without_independent_watchdog(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="no-watchdog")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    runner = EndpointRunner([snapshot("paused", 0)])

    def fail_watchdog(_lock: RunLock, _endpoint: object, _token: str) -> str:
        raise WorkerError("watchdog unavailable")

    with pytest.raises(WorkerError, match="watchdog unavailable"):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            source_preparer=_prepare_source,
            watchdog_launcher=fail_watchdog,
        )

    assert [command[2] for command in runner.commands] == ["describe"]


def test_worker_rejects_endpoint_drift_before_resume(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="drifted-endpoint")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    drifted = snapshot("paused", 0)
    model = cast(dict[str, object], drifted["model"])
    model["revision"] = "different"
    runner = EndpointRunner([drifted, snapshot("paused", 0)])

    with pytest.raises(WorkerError, match="endpoint model does not match"):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            source_preparer=_prepare_source,
            watchdog_launcher=_launch_watchdog,
        )

    assert [command[2] for command in runner.commands] == ["describe"]


def test_worker_rejects_live_endpoint_before_watchdog_start(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="live-endpoint")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    runner = EndpointRunner([snapshot("running", 1)])
    watchdog_calls = 0

    def launch_watchdog(_lock: RunLock, _endpoint: object, _token: str) -> str:
        nonlocal watchdog_calls
        watchdog_calls += 1
        return "watchdog-job"

    with pytest.raises(
        WorkerError,
        match="^endpoint must be paused with zero ready replicas before ownership$",
    ):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            source_preparer=_prepare_source,
            watchdog_launcher=launch_watchdog,
        )

    assert watchdog_calls == 0
    assert [command[2] for command in runner.commands] == ["describe"]


def test_cleanup_failure_prevents_success_and_redacts_failure(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="cleanup-failed")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        "harbor_hf.worker.probe_runtime",
        lambda _url, _token, _health_route, *_deadline: {"probes": {}},
    )
    runner = CleanupFailureRunner([snapshot("paused", 0), snapshot("running", 1)])

    with pytest.raises(WorkerError, match=r"^pause failed with \[REDACTED\]$"):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            stream_runner=_successful_stream,
            source_preparer=_prepare_source,
            watchdog_launcher=_launch_watchdog,
        )

    root = tmp_path / "output" / lock.artifact_prefix
    assert not (root / "_SUCCESS").exists()
    assert json.loads((root / "_FAILED").read_text()) == {
        "error_type": "RuntimeError",
        "message": "pause failed with [REDACTED]",
    }
    event_records = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    assert [
        {key: value for key, value in record.items() if key != "at"}
        for record in event_records[-4:]
    ] == [
        {"event": "endpoint_pause_requested"},
        {"event": "endpoint_cleanup_failed", "error": "RuntimeError"},
        {"event": "run_failed", "error_type": "RuntimeError"},
        {"event": "secrets_redacted", "files": ["harbor.log"]},
    ]


def test_worker_preserves_execution_and_cleanup_failures(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="execution-and-cleanup-failed")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        "harbor_hf.worker.probe_runtime",
        lambda _url, _token, _health_route, *_deadline: {"probes": {}},
    )
    runner = CleanupFailureRunner([snapshot("paused", 0), snapshot("running", 1)])

    with pytest.raises(
        WorkerError,
        match=(
            r"^Harbor exited with status 7; endpoint cleanup failed: "
            r"pause failed with \[REDACTED\]$"
        ),
    ):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            stream_runner=lambda *_args, **_kwargs: 7,
            source_preparer=_prepare_source,
            watchdog_launcher=_launch_watchdog,
        )

    root = tmp_path / "output" / lock.artifact_prefix
    assert json.loads((root / "_FAILED").read_text()) == {
        "error_type": "WorkerError",
        "message": "Harbor exited with status 7",
        "cleanup_error": {
            "error_type": "RuntimeError",
            "message": "pause failed with [REDACTED]",
        },
    }
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    assert {key: value for key, value in events[-1].items() if key != "at"} == {
        "event": "run_failed",
        "error_type": "WorkerError",
        "cleanup_error_type": "RuntimeError",
    }


def test_worker_rejects_mismatched_lock_before_remote_work(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
) -> None:
    lock = build_run_lock(remote_spec, run_id="mismatch").model_copy(
        update={"spec_digest": "sha256:" + "0" * 64}
    )
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)

    with pytest.raises(
        WorkerError, match="^manifest digest does not match the run lock$"
    ):
        run_worker(remote_manifest, lock_path, tmp_path / "output")


def test_run_lock_validation_rejects_tampered_agent_metadata(
    remote_spec: ExperimentSpec,
) -> None:
    lock = build_run_lock(remote_spec)
    reserved = lock.model_copy(
        update={"agent": lock.agent.model_copy(update={"parameters": {"version": "x"}})}
    )
    with pytest.raises(
        WorkerError,
        match="^run lock fields do not match the resolved manifest cell$",
    ):
        validate_run_lock(remote_spec, reserved)

    remote = remote_spec.remote
    assert remote is not None
    source_agent = lock.agent.model_copy(
        update={
            "revision": "0" * 40,
            "revision_kind": "harbor-source",
            "reported_version": "2.0.0",
        }
    )
    source_lock = lock.model_copy(update={"agent": source_agent})
    with pytest.raises(
        WorkerError,
        match="^run lock fields do not match the resolved manifest cell$",
    ):
        validate_run_lock(remote_spec, source_lock)


def test_run_lock_validation_reconstructs_selected_matrix_cell(
    remote_spec: ExperimentSpec,
) -> None:
    models = [
        remote_spec.matrix.models[0],
        remote_spec.matrix.models[0].model_copy(update={"id": "second-model"}),
    ]
    deployments = [
        remote_spec.matrix.deployments[0],
        remote_spec.matrix.deployments[0].model_copy(
            update={"id": "second-deployment"}
        ),
    ]
    agents = [
        remote_spec.matrix.agents[0],
        remote_spec.matrix.agents[0].model_copy(update={"id": "second-agent"}),
    ]
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={
                    "models": models,
                    "deployments": deployments,
                    "agents": agents,
                }
            )
        }
    )
    lock = build_run_lock(
        spec,
        model_id="second-model",
        deployment_id="second-deployment",
        agent_id="second-agent",
        run_id="selected-cell",
    )

    validate_run_lock(spec, lock)


def test_run_lock_validation_reports_digest_and_resolution_errors(
    remote_spec: ExperimentSpec,
) -> None:
    lock = build_run_lock(remote_spec)
    bad_digest = lock.model_copy(update={"spec_digest": "sha256:" + "0" * 64})
    with pytest.raises(
        WorkerError, match="^manifest digest does not match the run lock$"
    ):
        validate_run_lock(remote_spec, bad_digest)

    unknown_model = lock.model_copy(
        update={"model": lock.model.model_copy(update={"id": "unknown-model"})}
    )
    with pytest.raises(
        WorkerError,
        match=(
            "^run lock cannot be resolved from manifest: "
            "unknown model profile: unknown-model$"
        ),
    ):
        validate_run_lock(remote_spec, unknown_model)


def test_worker_requires_named_secret(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="missing-secret")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(
        WorkerError, match="^required secret HF_TOKEN is not available$"
    ):
        run_worker(remote_manifest, lock_path, tmp_path / "output")


def test_worker_requires_git_source_secret_before_claim_or_remote_work(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = GitBenchmarkSource(
        repository="ShellBench/public-tasks",
        revision="8" * 40,
        path="tasks/115-tasks",
        credentials=GitHubTokenCredentials(secret_name="GITHUB_TOKEN"),
    )
    raw = remote_spec.model_dump(mode="python")
    raw["benchmark"].update(
        {
            "dataset": "shellbench/public-115",
            "source": source.model_dump(mode="python"),
        }
    )
    raw["benchmark"].pop("dataset_digest", None)
    spec = ExperimentSpec.model_validate(raw)
    lock = build_run_lock(spec, run_id="missing-git-secret")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(spec.model_dump(mode="json")), encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    claims = FakeClaimStore()

    with pytest.raises(
        WorkerError, match="^required secret GITHUB_TOKEN is not available$"
    ):
        run_worker(
            manifest,
            lock_path,
            tmp_path / "output",
            runner=EndpointRunner([]),
            claim_store=claims,
        )

    assert claims.acquired == []


def test_worker_rejects_lock_without_endpoint_binding(
    remote_spec: ExperimentSpec,
    remote_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="no-endpoint")
    lock = lock.model_copy(
        update={"deployment": lock.deployment.model_copy(update={"endpoint": None})}
    )
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    with pytest.raises(WorkerError, match="^run lock fields do not match"):
        run_worker(remote_manifest, lock_path, tmp_path / "output")


def test_remote_job_rejects_custom_token_name(
    remote_spec: ExperimentSpec,
) -> None:
    payload = remote_spec.model_dump(mode="json")
    payload["remote"]["job"]["token_secret_name"] = "BENCH_TOKEN"

    with pytest.raises(ValueError, match="Input should be 'HF_TOKEN'"):
        ExperimentSpec.model_validate(payload)
