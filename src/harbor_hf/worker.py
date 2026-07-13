from __future__ import annotations

import json
import os
import platform
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Protocol, cast
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
from harbor_hf.models import EndpointRef, ExperimentSpec, SourcePin
from harbor_hf.planner import experiment_digest
from harbor_hf.process import CommandRunner, SubprocessRunner, run_streaming
from harbor_hf.runs import RunLock
from harbor_hf.submission import github_repository, locked_source_command

_WATCHDOG_READY_LABEL = "harbor-hf-watchdog-ready"
_WATCHDOG_STARTUP_TIMEOUT_SECONDS = 300


class WorkerError(RuntimeError):
    """Raised when a remote benchmark run cannot complete correctly."""


class JobInspector(Protocol):
    def inspect_job(self, *, job_id: str, namespace: str | None = None) -> object: ...


class WatchdogApi(JobInspector, Protocol):
    def update_job_labels(
        self,
        *,
        job_id: str,
        labels: dict[str, str],
        namespace: str | None = None,
    ) -> object: ...


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


def build_harbor_command(
    lock: RunLock,
    jobs_dir: Path,
    base_url: str,
    harbor_source: Path,
) -> list[str]:
    harbor = lock.remote.harbor
    endpoint = lock.deployment.endpoint
    if endpoint is None:
        raise WorkerError("run lock has no endpoint binding")
    command = [
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
        "--dataset",
        lock.benchmark_dataset,
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
    source_preparer: Callable[[SourcePin, Path, CommandRunner], None] | None = None,
    watchdog_launcher: Callable[[RunLock, EndpointRef, str], str] | None = None,
) -> Path:
    spec = load_experiment(manifest_path)
    lock = RunLock.model_validate_json(
        lock_path.read_text(encoding="utf-8")  # pragma: no mutate
    )
    validate_run_lock(spec, lock)

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
        harbor_source = Path("/tmp/harbor-hf-sources") / (
            f"harbor-{lock.remote.harbor.source.revision}"
        )
        (source_preparer or prepare_locked_source)(
            lock.remote.harbor.source,
            harbor_source,
            process_runner,
        )
        watchdog_id = (watchdog_launcher or launch_cleanup_watchdog)(
            lock,
            endpoint,
            token,
        )
        append_event(events, "cleanup_watchdog_started", job_id=watchdog_id)
        _execute_benchmark(
            root,
            events,
            lock,
            manager,
            token,
            stream_runner,
            harbor_source,
        )
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
        _publish_success(root, events, token)
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


def validate_run_lock(spec: ExperimentSpec, lock: RunLock) -> None:
    if lock.spec_digest != experiment_digest(spec):
        raise WorkerError("manifest digest does not match the run lock")
    if "version" in lock.agent.parameters:
        raise WorkerError("agent parameter 'version' is reserved by the run lock")


def _publish_success(root: Path, events: Path, token: str) -> None:
    append_event(events, "run_succeeded")
    try:
        _finalize_evidence(root, token)
    except Exception as caught:
        append_event(
            events,
            "evidence_finalization_failed",
            error=type(caught).__name__,
        )
        write_json(
            root / "_FAILED",
            {
                "error_type": type(caught).__name__,
                "message": str(caught).replace(token, "[REDACTED]"),
            },
        )
        raise WorkerError("evidence finalization failed") from caught
    (root / "_SUCCESS").write_text("\n", encoding="utf-8")


def _execute_benchmark(
    root: Path,
    events: Path,
    lock: RunLock,
    manager: EndpointManager,
    token: str,
    stream_runner: Callable[..., int],
    harbor_source: Path,
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
    harbor_command = build_harbor_command(
        lock,
        jobs_dir,
        base_url,
        harbor_source,
    )
    append_event(events, "harbor_started")
    exit_code = stream_runner(
        harbor_command,
        root / "harbor.log",
        environment={
            "HF_TOKEN": token,
            "OPENAI_API_KEY": token,
            "OPENAI_BASE_URL": f"{base_url}/v1",
        },
        timeout_seconds=lock.timeout_seconds,
    )
    append_event(events, "harbor_finished", exit_code=exit_code)
    if exit_code != 0:
        raise WorkerError(f"Harbor exited with status {exit_code}")
    verifier = validate_harbor_result(
        jobs_dir,
        expected_trials=_expected_trial_count(lock),
    )
    write_json(root / "verification.json", verifier)
    append_event(events, "verification_validated")


def prepare_locked_source(
    source: SourcePin,
    destination: Path,
    runner: CommandRunner,
) -> None:
    if destination.exists():
        raise WorkerError(f"source checkout already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    runner.run_text(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            github_repository(source.repository),
            str(destination),
        ]
    )
    runner.run_text(
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--depth",
            "1",
            "origin",
            source.revision,
        ]
    )
    runner.run_text(
        [
            "git",
            "-C",
            str(destination),
            "checkout",
            "--detach",
            source.revision,
        ]
    )
    if not (destination / "uv.lock").is_file():
        raise WorkerError("pinned source checkout has no uv.lock")


def launch_cleanup_watchdog(lock: RunLock, endpoint: EndpointRef, token: str) -> str:
    from huggingface_hub import HfApi

    controller_job_id = os.environ.get("JOB_ID")
    if not controller_job_id:
        raise WorkerError("controller JOB_ID is required before endpoint resume")
    job_timeout_seconds = min(lock.remote.job.timeout_seconds + 600, 86400)
    command = locked_source_command(
        lock.remote.worker,
        "harbor-hf",
        "watchdog",
        "--controller-job-id",
        controller_job_id,
        "--controller-namespace",
        lock.remote.job.namespace,
        "--endpoint-name",
        endpoint.name,
        "--endpoint-namespace",
        endpoint.namespace,
        "--run-id",
        lock.run_id,
        "--token-secret-name",
        lock.remote.job.token_secret_name,
        "--timeout-seconds",
        str(lock.remote.job.timeout_seconds),
    )
    api = HfApi(token=token)
    info = api.run_job(
        image=lock.remote.job.image,
        command=command,
        secrets={lock.remote.job.token_secret_name: token},
        flavor=lock.remote.job.flavor,
        timeout=job_timeout_seconds,
        labels={"harbor-hf-watchdog": lock.run_id},
        namespace=lock.remote.job.namespace,
    )
    job_id = getattr(info, "id", None)
    if not isinstance(job_id, str) or not job_id:
        raise WorkerError("cleanup watchdog submission returned no job ID")
    try:
        wait_watchdog_ready(
            api,
            job_id,
            lock.remote.job.namespace,
            timeout_seconds=_WATCHDOG_STARTUP_TIMEOUT_SECONDS,
        )
    except Exception:
        with suppress(Exception):
            api.cancel_job(job_id=job_id, namespace=lock.remote.job.namespace)
        raise
    return job_id


def wait_watchdog_ready(
    api: JobInspector,
    job_id: str,
    namespace: str,
    *,
    timeout_seconds: int,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    poll_seconds: float = 5,
) -> None:
    deadline = monotonic() + timeout_seconds
    terminal = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"}
    while True:
        info = api.inspect_job(job_id=job_id, namespace=namespace)
        labels = getattr(info, "labels", None)
        if isinstance(labels, Mapping) and labels.get(_WATCHDOG_READY_LABEL) == "true":
            return
        stage = _job_stage(info)
        if stage in terminal:
            raise WorkerError(f"cleanup watchdog exited before readiness: {stage}")
        if monotonic() >= deadline:
            raise WorkerError("cleanup watchdog readiness timed out")
        sleep(poll_seconds)


def run_endpoint_watchdog(
    *,
    controller_job_id: str,
    controller_namespace: str,
    endpoint_name: str,
    endpoint_namespace: str,
    run_id: str,
    token_secret_name: str,
    timeout_seconds: int,
    api: WatchdogApi | None = None,
    runner: CommandRunner | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    poll_seconds: float = 10,
) -> dict[str, object]:
    token = os.environ.get(token_secret_name, "")
    if not token:
        raise WorkerError(f"required secret {token_secret_name} is not available")
    os.environ.setdefault("HF_TOKEN", token)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    watchdog_job_id = os.environ.get("JOB_ID", "")
    if not watchdog_job_id:
        raise WorkerError("watchdog JOB_ID is required")
    api.update_job_labels(
        job_id=watchdog_job_id,
        labels={
            "harbor-hf-watchdog": run_id,
            _WATCHDOG_READY_LABEL: "true",
        },
        namespace=controller_namespace,
    )
    deadline = monotonic() + timeout_seconds
    terminal = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"}
    while monotonic() < deadline:
        try:
            stage = _job_stage(
                api.inspect_job(
                    job_id=controller_job_id,
                    namespace=controller_namespace,
                )
            )
        except Exception:
            sleep(poll_seconds)
            continue
        if stage in terminal:
            break
        sleep(poll_seconds)
    manager = EndpointManager(
        endpoint_namespace,
        endpoint_name,
        runner or SubprocessRunner(),
    )
    return manager.pause_and_verify()


def _job_stage(info: object) -> str:
    from huggingface_hub import JobInfo

    job = cast(JobInfo, info)
    stage = job.status.stage
    value = getattr(stage, "value", stage)
    if not isinstance(value, str):
        raise WorkerError("HF Job response has an invalid stage")
    return value.upper()


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


def _expected_trial_count(lock: RunLock) -> int | None:
    if any(
        any(character in task for character in "*?[") for task in lock.benchmark_tasks
    ):
        return None
    return len(lock.benchmark_tasks) * lock.attempts


def validate_harbor_result(
    jobs_dir: Path, expected_trials: int | None = 1
) -> dict[str, object]:
    trials: list[dict[str, object]] = []
    for path in sorted(jobs_dir.rglob("result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "task_name" in value:
            trials.append(value)
    _validate_trial_count(trials, expected_trials)

    verified: list[dict[str, object]] = []
    for trial in trials:
        task_name = str(trial["task_name"])
        exception = trial.get("exception_info")
        if exception is not None:
            exception_type = (
                exception.get("exception_type")
                if isinstance(exception, Mapping)
                else type(exception).__name__
            )
            raise WorkerError(
                f"Harbor trial {task_name} failed with "
                f"{exception_type or 'an exception'}"
            )
        verifier = trial.get("verifier_result")
        rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
        if not isinstance(rewards, Mapping) or not rewards:
            raise WorkerError(f"Harbor trial {task_name} has no verifier rewards")
        if not all(
            isinstance(value, int | float) and not isinstance(value, bool)
            for value in rewards.values()
        ):
            raise WorkerError(f"Harbor trial {task_name} rewards must be numeric")
        verified.append({"task_name": task_name, "rewards": dict(rewards)})
    return {
        "trial_count": len(verified),
        "trials": verified,
    }


def _validate_trial_count(
    trials: list[dict[str, object]], expected_trials: int | None
) -> None:
    if expected_trials is None and not trials:
        raise WorkerError("Harbor produced no trials")
    if expected_trials is not None and len(trials) != expected_trials:
        raise WorkerError(
            f"expected exactly {expected_trials} Harbor trials, found {len(trials)}"
        )
