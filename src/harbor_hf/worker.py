from __future__ import annotations

import json
import os
import platform
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlparse

from harbor_hf.evidence import (
    append_event,
    archive_directory,
    assert_secret_absent,
    redact,
    scrub_secret,
    write_checksums,
    write_json,
)
from harbor_hf.io import load_experiment
from harbor_hf.planner import experiment_digest
from harbor_hf.process import CommandRunner, SubprocessRunner, run_streaming
from harbor_hf.runs import RunLock
from harbor_hf.submission import github_archive


class WorkerError(RuntimeError):
    """Raised when a remote benchmark run cannot complete correctly."""


class EndpointManager:
    def __init__(
        self,
        namespace: str,
        name: str,
        runner: CommandRunner,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.namespace = namespace
        self.name = name
        self.runner = runner
        self.sleep = sleep
        self.monotonic = monotonic

    def describe(self) -> dict[str, object]:
        return self.runner.run_json(self._command("describe"))

    def resume(self) -> dict[str, object]:
        return self.runner.run_json(self._command("resume"))

    def pause(self) -> dict[str, object]:
        return self.runner.run_json(self._command("pause"))

    def wait_ready(
        self, timeout_seconds: int, poll_seconds: float = 15
    ) -> dict[str, object]:
        deadline = self.monotonic() + timeout_seconds
        while True:
            snapshot = self.describe()
            state, ready, _ = endpoint_state(snapshot)
            if state == "running" and ready > 0:
                return snapshot
            if self.monotonic() >= deadline:
                raise WorkerError(
                    f"endpoint readiness timed out in state={state!r}, ready={ready}"
                )
            self.sleep(poll_seconds)

    def pause_and_verify(
        self, timeout_seconds: int = 300, poll_seconds: float = 10
    ) -> dict[str, object]:
        self.pause()
        deadline = self.monotonic() + timeout_seconds
        while True:
            snapshot = self.describe()
            state, ready, _ = endpoint_state(snapshot)
            if state == "paused" and ready == 0:
                return snapshot
            if self.monotonic() >= deadline:
                raise WorkerError(
                    f"endpoint cleanup timed out in state={state!r}, ready={ready}"
                )
            self.sleep(poll_seconds)

    def _command(self, operation: str) -> list[str]:
        return [
            "hf",
            "endpoints",
            operation,
            self.name,
            "--namespace",
            self.namespace,
            "--format",
            "json",
        ]


def endpoint_state(snapshot: Mapping[str, object]) -> tuple[str, int, int]:
    status = snapshot.get("status")
    if not isinstance(status, Mapping):
        raise WorkerError("endpoint response has no status object")
    state = status.get("state")
    ready = status.get("readyReplica")
    target = status.get("targetReplica")
    if not isinstance(state, str) or not isinstance(ready, int):
        raise WorkerError("endpoint status is missing state or readyReplica")
    return state, ready, target if isinstance(target, int) else 0


def endpoint_url(snapshot: Mapping[str, object]) -> str:
    status = snapshot.get("status")
    if not isinstance(status, Mapping):
        raise WorkerError("endpoint status is missing its URL")
    url = status.get("url")
    if not isinstance(url, str):
        raise WorkerError("endpoint status is missing its URL")
    return url.rstrip("/")


def build_harbor_command(lock: RunLock, jobs_dir: Path, base_url: str) -> list[str]:
    harbor = lock.remote.harbor
    endpoint = lock.deployment.endpoint
    if endpoint is None:
        raise WorkerError("run lock has no endpoint binding")
    command = [
        "uvx",
        "--from",
        (
            "harbor[hf-sandbox] @ "
            + github_archive(harbor.source.repository, harbor.source.revision)
        ),
        "harbor",
        "run",
        "--dataset",
        lock.benchmark_dataset,
        "--n-tasks",
        "1",
        "--n-attempts",
        str(lock.attempts),
        "--agent",
        lock.agent.name,
        "--model",
        f"openai/{endpoint.served_model_name}",
        "--env",
        harbor.environment,
        "--environment-kwarg",
        f"flavor={harbor.sandbox_flavor}",
        "--environment-kwarg",
        f"job_timeout={harbor.sandbox_idle_timeout_seconds}",
        "--jobs-dir",
        str(jobs_dir),
        "--n-concurrent",
        str(lock.concurrent_trials),
        "--n-concurrent-agents",
        str(lock.concurrent_trials),
        "--max-retries",
        "0",
        "--allow-agent-host",
        urlparse(base_url).hostname or "",
        "--yes",
    ]
    for task_name in lock.benchmark_tasks:
        command.extend(("--include-task-name", task_name))
    command.extend(("--agent-kwarg", f"version={lock.agent.revision}"))
    for key, value in sorted(lock.agent.parameters.items()):
        rendered = json.dumps(value, separators=(",", ":"))
        command.extend(("--agent-kwarg", f"{key}={rendered}"))
    return command


def run_worker(
    manifest_path: Path,
    lock_path: Path,
    output_root: Path,
    *,
    runner: CommandRunner | None = None,
    stream_runner: Callable[..., int] = run_streaming,
) -> Path:
    spec = load_experiment(manifest_path)
    lock = RunLock.model_validate_json(
        lock_path.read_text(encoding="utf-8")  # pragma: no mutate
    )
    if lock.spec_digest != experiment_digest(spec):
        raise WorkerError("manifest digest does not match the run lock")

    token_name = lock.remote.job.token_secret_name
    token = os.environ.get(token_name, "")
    if not token:
        raise WorkerError(f"required secret {token_name} is not available")
    os.environ.setdefault("HF_TOKEN", token)

    root = output_root / lock.artifact_prefix
    root.mkdir(parents=True, exist_ok=False)
    (root / "harbor-jobs").mkdir()
    shutil.copyfile(manifest_path, root / "manifest.yaml")
    write_json(root / "run.lock.json", lock.model_dump(mode="json"))
    events = root / "events.jsonl"
    process_runner = runner or SubprocessRunner()
    endpoint = lock.deployment.endpoint
    if endpoint is None:
        raise WorkerError("run lock has no endpoint binding")
    manager = EndpointManager(endpoint.namespace, endpoint.name, process_runner)
    error: Exception | None = None
    cleanup_error: Exception | None = None

    append_event(events, "worker_started", run_id=lock.run_id)
    try:
        require_executable("git")
        _execute_benchmark(root, events, lock, manager, token, stream_runner)
    except Exception as caught:
        error = caught
    finally:
        append_event(events, "endpoint_pause_requested")
        try:
            final_snapshot = manager.pause_and_verify()
            state, ready, target = endpoint_state(final_snapshot)
            write_json(root / "endpoint.final.json", redact(final_snapshot))
            append_event(
                events,
                "endpoint_paused",
                state=state,
                ready_replicas=ready,
                target_replicas=target,
            )
        except Exception as caught:
            cleanup_error = caught
            append_event(events, "endpoint_cleanup_failed", error=type(caught).__name__)

    if error is None and cleanup_error is None:
        append_event(events, "run_succeeded")
        _finalize_evidence(root, token)
        (root / "_SUCCESS").write_text("\n", encoding="utf-8")
        return root

    failure = cleanup_error or error
    assert failure is not None, "failed run has no recorded error"
    failure_message = str(failure).replace(token, "[REDACTED]")
    append_event(
        events,
        "run_failed",
        error_type=type(failure).__name__,
    )
    write_json(
        root / "_FAILED",
        {
            "error_type": type(failure).__name__,
            "message": failure_message,
        },
    )
    try:
        _finalize_evidence(root, token)
    except Exception as caught:
        message = f"{failure_message}; evidence finalization failed: {caught}"
        raise WorkerError(message) from caught
    raise WorkerError(failure_message) from failure


def _execute_benchmark(
    root: Path,
    events: Path,
    lock: RunLock,
    manager: EndpointManager,
    token: str,
    stream_runner: Callable[..., int],
) -> None:
    append_event(events, "endpoint_resume_requested")
    manager.resume()
    snapshot = manager.wait_ready(min(lock.timeout_seconds, 3600))
    validate_endpoint_model(lock, snapshot)
    append_event(events, "endpoint_ready", state=endpoint_state(snapshot)[0])
    write_json(root / "endpoint.snapshot.json", redact(snapshot))
    base_url = endpoint_url(snapshot)
    runtime = {
        "controller": controller_environment(lock),
        "endpoint": probe_runtime(base_url, token),
    }
    write_json(root / "runtime-environment.json", redact(runtime))
    append_event(events, "runtime_probed")

    jobs_dir = root / "harbor-jobs"
    harbor_command = build_harbor_command(lock, jobs_dir, base_url)
    append_event(events, "harbor_started")
    exit_code = stream_runner(
        harbor_command,
        root / "harbor.log",
        environment={
            "HF_TOKEN": token,
            "OPENAI_API_KEY": token,
            "OPENAI_BASE_URL": f"{base_url}/v1",
        },
    )
    append_event(events, "harbor_finished", exit_code=exit_code)
    if exit_code != 0:
        raise WorkerError(f"Harbor exited with status {exit_code}")
    verifier = validate_harbor_result(jobs_dir)
    write_json(root / "verification.json", verifier)
    append_event(events, "verification_validated")


def validate_endpoint_model(lock: RunLock, snapshot: Mapping[str, object]) -> None:
    model = snapshot.get("model")
    if not isinstance(model, Mapping):
        raise WorkerError("endpoint response has no model object")
    observed_repo = model.get("repository")
    observed_revision = model.get("revision")
    if observed_repo != lock.model.repo or observed_revision != lock.model.revision:
        raise WorkerError(
            "endpoint model does not match the locked repository and revision"
        )


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise WorkerError(f"required controller executable is missing: {name}")


def _finalize_evidence(root: Path, token: str) -> None:
    scrubbed = scrub_secret(root, token)
    if scrubbed:
        append_event(root / "events.jsonl", "secrets_redacted", files=scrubbed)
    assert_secret_absent(root, token)
    archive_directory(root / "harbor-jobs", root / "artifacts.tar.gz")
    write_checksums(root)


def controller_environment(lock: RunLock) -> dict[str, object]:
    return {
        "job_id": os.environ.get("JOB_ID"),
        "namespace": lock.remote.job.namespace,
        "requested_image": lock.remote.job.image,
        "requested_flavor": lock.remote.job.flavor,
        "reported_accelerator": os.environ.get("ACCELERATOR"),
        "reported_cpu_cores": os.environ.get("CPU_CORES"),
        "reported_memory": os.environ.get("MEMORY"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def probe_runtime(base_url: str, token: str) -> dict[str, object]:
    probes: dict[str, object] = {}
    for name, path in (
        ("health", "/health"),
        ("version", "/version"),
        ("models", "/v1/models"),
    ):
        request = urllib.request.Request(
            f"{base_url}{path}",
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read(1024 * 1024).decode("utf-8", errors="replace")
                try:
                    parsed: object = json.loads(body)
                except json.JSONDecodeError:
                    parsed = body
                probes[name] = {
                    "status": "reported",
                    "http_status": response.status,
                    "value": parsed,
                }
        except (urllib.error.URLError, TimeoutError) as caught:
            probes[name] = {
                "status": "unknown",
                "error_type": type(caught).__name__,
            }
    health = probes["health"]
    if not isinstance(health, Mapping) or health.get("http_status") != 200:
        raise WorkerError("endpoint health probe did not return HTTP 200")
    return {"probes": probes}


def validate_harbor_result(jobs_dir: Path) -> dict[str, object]:
    trials: list[dict[str, object]] = []
    for path in jobs_dir.rglob("result.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "task_name" in value:
            trials.append(value)
    if len(trials) != 1:
        raise WorkerError(f"expected exactly one Harbor trial, found {len(trials)}")
    verifier = trials[0].get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    if not isinstance(rewards, Mapping) or not rewards:
        raise WorkerError("Harbor trial has no verifier rewards")
    if not all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in rewards.values()
    ):
        raise WorkerError("Harbor verifier rewards must be numeric")
    return {
        "task_name": trials[0]["task_name"],
        "rewards": dict(rewards),
        "trial_count": 1,
    }
