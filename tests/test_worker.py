from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from harbor_hf.models import ExperimentSpec
from harbor_hf.runs import RunLock, build_run_lock
from harbor_hf.worker import (
    EndpointManager,
    WorkerError,
    _finalize_evidence,
    build_harbor_command,
    controller_environment,
    endpoint_state,
    endpoint_url,
    probe_runtime,
    require_executable,
    run_worker,
    validate_endpoint_model,
    validate_harbor_result,
)


def snapshot(state: str, ready: int) -> dict[str, object]:
    return {
        "model": {
            "repository": "nvidia/Qwen3.6-35B-A3B-NVFP4",
            "revision": "0123456789abcdef0123456789abcdef01234567",
        },
        "status": {
            "state": state,
            "readyReplica": ready,
            "targetReplica": 1,
            "url": "https://endpoint.example",
        },
    }


class EndpointRunner:
    def __init__(self, descriptions: list[dict[str, object]]) -> None:
        self.descriptions = descriptions
        self.commands: list[list[str]] = []

    def run_json(self, command: Sequence[str]) -> dict[str, object]:
        self.commands.append(list(command))
        operation = command[2]
        if operation == "describe":
            return self.descriptions.pop(0)
        return snapshot("running" if operation == "resume" else "paused", 0)

    def run_text(self, command: Sequence[str]) -> str:
        raise AssertionError(command)


class CleanupFailureRunner(EndpointRunner):
    def run_json(self, command: Sequence[str]) -> dict[str, object]:
        if command[2] == "pause":
            self.commands.append(list(command))
            raise RuntimeError("pause failed with test-token")
        return super().run_json(command)


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


def test_endpoint_waits_through_transitional_states() -> None:
    sleeps: list[float] = []
    times = iter([0.0, 1.0, 2.0, 3.0])
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


def test_controller_requires_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("harbor_hf.worker.shutil.which", lambda _name: None)

    with pytest.raises(
        WorkerError, match="^required controller executable is missing: git$"
    ):
        require_executable("git")


def test_cleanup_timeout() -> None:
    times = iter([0.0, 2.0])
    manager = EndpointManager(
        "org",
        "endpoint",
        EndpointRunner([snapshot("pausing", 1)]),
        sleep=lambda _: None,
        monotonic=lambda: next(times),
    )

    with pytest.raises(WorkerError, match="cleanup timed out"):
        manager.pause_and_verify(timeout_seconds=1)


def test_harbor_command_is_pinned_and_bounded(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = build_run_lock(remote_spec)

    command = build_harbor_command(lock, tmp_path, "https://endpoint.example")

    assert command == [
        "uvx",
        "--from",
        "harbor[hf-sandbox] @ https://github.com/harbor-framework/harbor/"
        "archive/abcdef1234567890abcdef1234567890abcdef12.zip",
        "harbor",
        "run",
        "--dataset",
        "terminal-bench@2.0",
        "--n-tasks",
        "1",
        "--n-attempts",
        "1",
        "--agent",
        "openclaw",
        "--model",
        "openai//repository",
        "--env",
        "hf-sandbox",
        "--environment-kwarg",
        "flavor=cpu-basic",
        "--environment-kwarg",
        "job_timeout=600",
        "--jobs-dir",
        str(tmp_path),
        "--n-concurrent",
        "1",
        "--n-concurrent-agents",
        "1",
        "--max-retries",
        "0",
        "--allow-agent-host",
        "endpoint.example",
        "--yes",
        "--include-task-name",
        "cancel-async-tasks",
        "--agent-kwarg",
        "version=replace-with-commit",
        "--agent-kwarg",
        "compaction=true",
        "--agent-kwarg",
        'thinking="off"',
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

    assert validate_harbor_result(tmp_path)["rewards"] == {"reward": 0.5}
    (trial / "result.json").write_text(
        json.dumps(
            {"task_name": "task", "verifier_result": {"rewards": {"reward": True}}}
        )
    )
    with pytest.raises(WorkerError, match="numeric"):
        validate_harbor_result(tmp_path)


def test_validate_harbor_result_rejects_missing_and_multiple_trials(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        WorkerError, match="^expected exactly one Harbor trial, found 0$"
    ):
        validate_harbor_result(tmp_path)

    for name in ("one", "two"):
        trial = tmp_path / name
        trial.mkdir()
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
        WorkerError, match="^expected exactly one Harbor trial, found 2$"
    ):
        validate_harbor_result(tmp_path)


def test_validate_harbor_result_requires_rewards(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    result = trial / "result.json"
    result.write_text(
        json.dumps({"task_name": "task", "verifier_result": {"rewards": {}}}),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match="^Harbor trial has no verifier rewards$"):
        validate_harbor_result(tmp_path)


def test_validate_harbor_result_rejects_trial_exception(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
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

    with pytest.raises(WorkerError, match="^Harbor trial failed with AgentError$"):
        validate_harbor_result(tmp_path)


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
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse(b"bad", 503)
    )

    with pytest.raises(
        WorkerError, match="^endpoint health probe did not return HTTP 200$"
    ):
        probe_runtime("https://endpoint.example", "token")


def test_finalize_evidence_scrubs_and_archives(tmp_path: Path) -> None:
    jobs = tmp_path / "harbor-jobs"
    jobs.mkdir()
    (jobs / "log.txt").write_text("contains test-token", encoding="utf-8")
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")

    _finalize_evidence(tmp_path, "test-token")

    assert (jobs / "log.txt").read_text() == "contains [REDACTED]"
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["event"] == "secrets_redacted"
    assert event["files"] == ["harbor-jobs/log.txt"]
    assert (tmp_path / "artifacts.tar.gz").exists()
    checksums = json.loads((tmp_path / "checksums.json").read_text())
    assert set(checksums) == {
        "artifacts.tar.gz",
        "events.jsonl",
        "harbor-jobs/log.txt",
    }


def _write_lock(path: Path, lock: RunLock) -> None:
    path.write_text(lock.model_dump_json(), encoding="utf-8")


def _successful_stream(
    command: Sequence[str], log_path: Path, *, environment: dict[str, str]
) -> int:
    assert environment == {
        "HF_TOKEN": "test-token",
        "OPENAI_API_KEY": "test-token",
        "OPENAI_BASE_URL": "https://endpoint.example/v1",
    }
    assert log_path.name == "harbor.log"
    assert command[command.index("--allow-agent-host") + 1] == "endpoint.example"
    jobs_dir = Path(command[command.index("--jobs-dir") + 1])
    trial = jobs_dir / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "cancel-async-tasks",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
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

    def fake_probe(url: str, token: str) -> dict[str, object]:
        assert url == "https://endpoint.example"
        assert token == "test-token"
        return {"probes": {"health": {"http_status": 200}}}

    monkeypatch.setattr("harbor_hf.worker.probe_runtime", fake_probe)
    runner = EndpointRunner([snapshot("running", 1), snapshot("paused", 0)])

    root = run_worker(
        remote_manifest,
        lock_path,
        tmp_path / "output",
        runner=runner,
        stream_runner=_successful_stream,
    )

    assert (root / "_SUCCESS").exists()
    assert not (root / "_FAILED").exists()
    assert json.loads((root / "verification.json").read_text())["rewards"] == {
        "reward": 1.0
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
    assert json.loads((root / "endpoint.snapshot.json").read_text()) == snapshot(
        "running", 1
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
        "harbor-jobs",
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
        "harbor-jobs/job/trial/result.json",
        "harbor.log",
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
        for operation in ("resume", "describe", "pause", "describe")
    ]


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
        "harbor_hf.worker.probe_runtime", lambda _url, _token: {"probes": {}}
    )
    runner = EndpointRunner([snapshot("running", 1), snapshot("paused", 0)])

    with pytest.raises(WorkerError, match="status 7"):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            stream_runner=lambda *_args, **_kwargs: 7,
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
        "harbor_hf.worker.probe_runtime", lambda _url, _token: {"probes": {}}
    )
    runner = CleanupFailureRunner([snapshot("running", 1)])

    with pytest.raises(WorkerError, match=r"^pause failed with \[REDACTED\]$"):
        run_worker(
            remote_manifest,
            lock_path,
            tmp_path / "output",
            runner=runner,
            stream_runner=_successful_stream,
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

    with pytest.raises(WorkerError, match="^run lock has no endpoint binding$"):
        run_worker(remote_manifest, lock_path, tmp_path / "output")


def test_worker_maps_custom_secret_and_refuses_existing_run_prefix(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    remote = remote_spec.remote
    assert remote is not None
    custom = remote_spec.model_copy(
        update={
            "remote": remote.model_copy(
                update={
                    "job": remote.job.model_copy(
                        update={"token_secret_name": "BENCH_TOKEN"}
                    )
                }
            )
        }
    )
    manifest = tmp_path / "custom.yaml"
    manifest.write_text(
        yaml.safe_dump(custom.model_dump(mode="json", exclude_none=True)),
        encoding="utf-8",
    )
    lock = build_run_lock(custom, run_id="existing")
    lock_path = tmp_path / "lock.json"
    _write_lock(lock_path, lock)
    root = tmp_path / "output" / lock.artifact_prefix
    root.mkdir(parents=True)
    monkeypatch.setenv("BENCH_TOKEN", "custom-token")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(FileExistsError):
        run_worker(manifest, lock_path, tmp_path / "output")

    assert os.environ["HF_TOKEN"] == "custom-token"
