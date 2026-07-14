from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse

from pydantic import JsonValue

from harbor_hf.coordination import (
    ClaimConflict,
    ClaimStore,
    HubClaimStore,
    endpoint_claim_path,
    run_claim_path,
)
from harbor_hf.endpoints import EndpointSettings
from harbor_hf.evidence import (
    append_event,
    archive_directory,
    assert_secret_absent,
    redact,
    scrub_secret,
    scrub_secret_paths,
    write_checksums,
    write_json,
)
from harbor_hf.harbor_adapter import (
    FilesystemHarborExecutionAdapter,
    WorkerError,
    build_execution_request,
)
from harbor_hf.harbor_adapter import (
    HarborTrialFailure as _HarborTrialFailure,
)
from harbor_hf.harbor_adapter.adapter import (
    effective_agent_parameters,
    render_harbor_command,
)
from harbor_hf.harbor_adapter.legacy import (
    validate_harbor_result as _validate_harbor_result,
)
from harbor_hf.harbor_adapter.legacy import (
    validate_task_counts,
    validate_trial_count,
)
from harbor_hf.io import load_experiment
from harbor_hf.models import (
    DeploymentProfile,
    EndpointRef,
    ExperimentSpec,
    RemoteExecutionSpec,
    SourcePin,
)
from harbor_hf.planner import experiment_digest
from harbor_hf.process import (
    CommandRunner,
    ProcessError,
    SubprocessRunner,
    run_streaming,
)
from harbor_hf.runs import RunLock, build_run_lock
from harbor_hf.submission import (
    endpoint_lease_label_for,
    github_repository,
    locked_source_command,
)

_WATCHDOG_READY_LABEL = "harbor-hf-watchdog-ready"
_WATCHDOG_STARTUP_TIMEOUT_SECONDS = 300
_ENDPOINT_CALL_TIMEOUT_SECONDS = 60.0
_MAX_CONSECUTIVE_READINESS_ERRORS = 3

# Historical imports remain stable while new executions use the typed adapter.
HarborTrialFailure = _HarborTrialFailure
validate_harbor_result = _validate_harbor_result
_validate_task_counts = validate_task_counts
_validate_trial_count = validate_trial_count


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

    def describe(
        self, timeout_seconds: float = _ENDPOINT_CALL_TIMEOUT_SECONDS
    ) -> dict[str, object]:
        return self.runner.run_json(
            self._command("describe"), timeout_seconds=timeout_seconds
        )

    def resume(
        self, timeout_seconds: float = _ENDPOINT_CALL_TIMEOUT_SECONDS
    ) -> dict[str, object]:
        return self.runner.run_json(
            self._command("resume"), timeout_seconds=timeout_seconds
        )

    def pause(
        self, timeout_seconds: float = _ENDPOINT_CALL_TIMEOUT_SECONDS
    ) -> dict[str, object]:
        return self.runner.run_json(
            self._command("pause"), timeout_seconds=timeout_seconds
        )

    def wait_ready(
        self, timeout_seconds: int, poll_seconds: float = 15
    ) -> dict[str, object]:
        deadline = self.monotonic() + timeout_seconds
        consecutive_errors = 0
        while True:
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise WorkerError("endpoint readiness timed out before status check")
            try:
                snapshot = self.describe(min(_ENDPOINT_CALL_TIMEOUT_SECONDS, remaining))
            except ProcessError as error:
                consecutive_errors += 1
                if consecutive_errors >= _MAX_CONSECUTIVE_READINESS_ERRORS:
                    raise WorkerError(
                        "endpoint readiness aborted after "
                        f"{consecutive_errors} consecutive provider errors: {error}"
                    ) from error
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    raise WorkerError(
                        "endpoint readiness timed out after transient provider "
                        f"errors: {error}"
                    ) from error
                self.sleep(min(poll_seconds, remaining))
                continue
            consecutive_errors = 0
            state, ready, target = endpoint_state(snapshot)
            if state == "running" and target > 0 and ready >= target:
                return snapshot
            if self.monotonic() >= deadline:
                raise WorkerError(
                    "endpoint readiness timed out in "
                    f"state={state!r}, ready={ready}, target={target}"
                )
            self.sleep(poll_seconds)

    def pause_and_verify(
        self, timeout_seconds: int = 300, poll_seconds: float = 10
    ) -> dict[str, object]:
        deadline = self.monotonic() + timeout_seconds
        pause_accepted = False
        last_transient_error: ProcessError | None = None
        state = "unknown"
        ready = -1
        while True:
            pause_accepted, snapshot, state, ready, transient_error = self._poll_pause(
                pause_accepted, deadline
            )
            last_transient_error = transient_error or last_transient_error
            if snapshot is not None and state == "paused" and ready == 0:
                return snapshot
            if self.monotonic() >= deadline:
                if last_transient_error is not None:
                    raise WorkerError(
                        "endpoint cleanup timed out after transient provider errors: "
                        f"{last_transient_error}"
                    ) from last_transient_error
                raise WorkerError(
                    f"endpoint cleanup timed out in state={state!r}, ready={ready}"
                )
            self.sleep(poll_seconds)

    def _poll_pause(
        self, pause_accepted: bool, deadline: float | None = None
    ) -> tuple[bool, dict[str, object] | None, str, int, ProcessError | None]:
        transient_error: ProcessError | None = None
        if not pause_accepted:
            try:
                self.pause(self._operation_timeout(deadline))
                pause_accepted = True
            except ProcessError as caught:
                transient_error = caught
        try:
            snapshot = self.describe(self._operation_timeout(deadline))
        except ProcessError as caught:
            return pause_accepted, None, "unknown", -1, caught
        state, ready, _ = endpoint_state(snapshot)
        if state in {"pausing", "paused"}:
            pause_accepted = True
        return pause_accepted, snapshot, state, ready, transient_error

    def _operation_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return _ENDPOINT_CALL_TIMEOUT_SECONDS
        remaining = deadline - self.monotonic()
        return max(0.001, min(_ENDPOINT_CALL_TIMEOUT_SECONDS, remaining))

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


def endpoint_health_route(snapshot: Mapping[str, object]) -> str:
    candidate = snapshot.get("healthRoute")
    if not isinstance(candidate, str):
        model = snapshot.get("model")
        image = model.get("image") if isinstance(model, Mapping) else None
        custom = image.get("custom") if isinstance(image, Mapping) else None
        candidate = custom.get("healthRoute") if isinstance(custom, Mapping) else None
    if not isinstance(candidate, str):
        raise WorkerError("endpoint response has no valid health route")
    route = candidate
    parsed = urlparse(route)
    if (
        not route.startswith("/")
        or parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise WorkerError("endpoint response has no valid health route")
    return route


def build_harbor_command(
    lock: RunLock,
    jobs_dir: Path,
    base_url: str,
    harbor_source: Path,
) -> list[str]:
    return _build_harbor_command(
        lock,
        jobs_dir,
        base_url,
        harbor_source,
        task_names=lock.benchmark_tasks,
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
    )


def build_harbor_trial_command(
    lock: RunLock,
    jobs_dir: Path,
    base_url: str,
    harbor_source: Path,
    *,
    task_name: str,
) -> list[str]:
    if task_name not in lock.benchmark_task_digests:
        raise WorkerError("wave trial is not in the resolved run task set")
    return _build_harbor_command(
        lock,
        jobs_dir,
        base_url,
        harbor_source,
        task_names=[task_name],
        attempts=1,
        concurrency=1,
    )


def _endpoint_binding(lock: RunLock) -> EndpointRef:
    deployment = _endpoint_deployment(lock)
    if deployment.endpoint is None:
        raise WorkerError("run lock has no endpoint binding")
    return deployment.endpoint


def _endpoint_deployment(lock: RunLock) -> DeploymentProfile:
    deployment = lock.deployment
    if not isinstance(deployment, DeploymentProfile):
        raise WorkerError("run lock is not an Inference Endpoint target")
    return deployment


def _build_harbor_command(
    lock: RunLock,
    jobs_dir: Path,
    base_url: str,
    harbor_source: Path,
    *,
    task_names: Sequence[str],
    attempts: int,
    concurrency: int,
) -> list[str]:
    request = build_execution_request(
        lock,
        jobs_dir,
        base_url,
        task_names=list(task_names),
        attempts=attempts,
        concurrency=concurrency,
        expected_task_digests={
            task: lock.benchmark_task_digests[task] for task in task_names
        },
    )
    del request
    return render_harbor_command(harbor_source, jobs_dir.parent / "harbor-job.json")


def _effective_agent_parameters(lock: RunLock) -> dict[str, JsonValue]:
    return effective_agent_parameters(lock)


def run_worker(
    manifest_path: Path,
    lock_path: Path,
    output_root: Path,
    *,
    runner: CommandRunner | None = None,
    stream_runner: Callable[..., int] = run_streaming,
    source_preparer: Callable[[SourcePin, Path, CommandRunner], None] | None = None,
    watchdog_launcher: Callable[[RunLock, EndpointRef, str], str] | None = None,
    claim_store: ClaimStore | None = None,
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
    os.environ["HF_TOKEN"] = token

    claims = claim_store or HubClaimStore(lock.remote.job.namespace, token)
    claim_path = run_claim_path(lock.artifact_bucket, lock.artifact_prefix)
    claim_owner = {
        "artifact_bucket": lock.artifact_bucket,
        "artifact_prefix": lock.artifact_prefix,
        "run_id": lock.run_id,
        "expires_at": (
            datetime.now(UTC) + timedelta(seconds=lock.remote.job.timeout_seconds + 600)
        ).isoformat(),
    }
    try:
        claims.acquire(claim_path, claim_owner)
    except ClaimConflict as error:
        raise WorkerError("run ID is already reserved") from error

    destination = output_root / lock.artifact_prefix
    entered_worker = False
    try:
        _prepare_evidence_destination(destination, adopt_reserved=True)
        entered_worker = True
        with tempfile.TemporaryDirectory(prefix="harbor-hf-run-") as staging:
            return _run_staged_worker(
                manifest_path,
                lock,
                Path(staging) / "run",
                destination,
                token,
                runner=runner,
                stream_runner=stream_runner,
                source_preparer=source_preparer,
                watchdog_launcher=watchdog_launcher,
            )
    except Exception:
        if not entered_worker or not any(
            (destination / marker).is_file() for marker in ("_SUCCESS", "_FAILED")
        ):
            with suppress(Exception):
                claims.release(claim_path, claim_owner)
        raise


def _run_staged_worker(
    manifest_path: Path,
    lock: RunLock,
    root: Path,
    destination: Path,
    token: str,
    *,
    runner: CommandRunner | None,
    stream_runner: Callable[..., int],
    source_preparer: Callable[[SourcePin, Path, CommandRunner], None] | None,
    watchdog_launcher: Callable[[RunLock, EndpointRef, str], str] | None,
) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    (root / "harbor-jobs").mkdir()
    shutil.copyfile(manifest_path, root / "manifest.yaml")
    write_json(root / "run.lock.json", lock.model_dump(mode="json"))
    events = root / "events.jsonl"
    process_runner = runner or SubprocessRunner()
    endpoint = _endpoint_binding(lock)
    manager = EndpointManager(endpoint.namespace, endpoint.name, process_runner)
    error: Exception | None = None
    cleanup_error: Exception | None = None
    watchdog_started = False

    append_event(events, "worker_started", run_id=lock.run_id)
    try:
        require_executable("git")
        harbor_source = (
            root.parent / "sources" / (f"harbor-{lock.remote.harbor.source.revision}")
        )
        (source_preparer or prepare_locked_source)(
            lock.remote.harbor.source,
            harbor_source,
            process_runner,
        )
        baseline = manager.describe()
        validate_endpoint_model(lock, baseline)
        require_paused_endpoint(baseline)
        append_event(events, "endpoint_baseline_validated")
        watchdog_id = (watchdog_launcher or launch_cleanup_watchdog)(
            lock,
            endpoint,
            token,
        )
        watchdog_started = True
        append_event(events, "endpoint_lease_acquired", watchdog_job_id=watchdog_id)
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
        if watchdog_started:
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
                append_event(
                    events, "endpoint_cleanup_failed", error=type(caught).__name__
                )
        else:
            append_event(events, "endpoint_cleanup_skipped", reason="lease_not_owned")

    if error is None and cleanup_error is None:
        _publish_success(root, events, token)
        _publish_evidence(root, destination)
        return destination

    failure, failure_record, failure_event, reported_message = _failure_details(
        error, cleanup_error, token
    )
    append_event(events, "run_failed", **failure_event)
    write_json(root / "_FAILED", failure_record)
    try:
        _finalize_evidence(root, token)
    except Exception as caught:
        finalization_message = str(caught).replace(token, "[REDACTED]")
        message = (
            f"{reported_message}; evidence finalization failed: {finalization_message}"
        )
        raise WorkerError(message) from caught
    _publish_evidence(root, destination)
    raise WorkerError(reported_message) from failure


def _failure_details(
    error: Exception | None,
    cleanup_error: Exception | None,
    token: str,
) -> tuple[Exception, dict[str, object], dict[str, str], str]:
    failure = error or cleanup_error
    assert failure is not None, "failed run has no recorded error"
    failure_message = str(failure).replace(token, "[REDACTED]")
    record: dict[str, object] = {
        "error_type": type(failure).__name__,
        "message": failure_message,
    }
    event = {"error_type": type(failure).__name__}
    reported_message = failure_message
    if error is not None and cleanup_error is not None:
        cleanup_message = str(cleanup_error).replace(token, "[REDACTED]")
        record["cleanup_error"] = {
            "error_type": type(cleanup_error).__name__,
            "message": cleanup_message,
        }
        event["cleanup_error_type"] = type(cleanup_error).__name__
        reported_message += f"; endpoint cleanup failed: {cleanup_message}"
    return failure, record, event, reported_message


def validate_run_lock(spec: ExperimentSpec, lock: RunLock) -> None:
    if lock.spec_digest != experiment_digest(spec):
        raise WorkerError("manifest digest does not match the run lock")
    try:
        expected = build_run_lock(
            spec,
            model_id=lock.model.id,
            deployment_id=lock.deployment.id,
            agent_id=lock.agent.id,
            run_id=lock.run_id,
            clock=lambda: lock.created_at,
        )
    except ValueError as error:
        raise WorkerError(
            f"run lock cannot be resolved from manifest: {error}"
        ) from error
    if lock != expected:
        raise WorkerError("run lock fields do not match the resolved manifest cell")


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


def _publish_evidence(
    source: Path,
    destination: Path,
    *,
    attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    markers = [name for name in ("_FAILED", "_SUCCESS") if (source / name).is_file()]
    if len(markers) != 1:
        raise WorkerError("finalized evidence must have exactly one terminal marker")
    if not (destination / "_RESERVED").is_file():
        raise WorkerError("run evidence destination is not reserved")
    if attempts < 1:
        raise ValueError("publication attempts must be positive")
    source_root = source.resolve()
    terminal = markers[0]
    temporary_marker = destination / "harbor-hf-terminal.tmp"
    for attempt in range(1, attempts + 1):
        try:
            shutil.copytree(
                source,
                destination,
                dirs_exist_ok=True,
                ignore=lambda directory, names: (
                    [name for name in names if name in {"_FAILED", "_SUCCESS"}]
                    if Path(directory).resolve() == source_root
                    else []
                ),
            )
            shutil.copyfile(source / terminal, temporary_marker)
            (destination / "_RESERVED").unlink()
            temporary_marker.replace(destination / terminal)
            return
        except Exception:
            if not (destination / terminal).is_file():
                (destination / "_RESERVED").touch(exist_ok=True)
            if attempt == attempts:
                raise
            sleep(float(attempt))


def _prepare_evidence_destination(
    destination: Path, *, adopt_reserved: bool = False
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            adopt_reserved
            and (destination / "_RESERVED").is_file()
            and not any(
                (destination / marker).exists() for marker in ("_FAILED", "_SUCCESS")
            )
        ):
            for path in destination.iterdir():
                if path.name == "_RESERVED":
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            return
        raise FileExistsError(destination)
    destination.mkdir()
    try:
        (destination / "_RESERVED").write_text("\n", encoding="utf-8")
    except Exception:
        destination.rmdir()
        raise


def _execute_benchmark(
    root: Path,
    events: Path,
    lock: RunLock,
    manager: EndpointManager,
    token: str,
    stream_runner: Callable[..., int],
    harbor_source: Path,
) -> None:
    base_url = resume_and_probe_endpoint(root, events, lock, manager, token)

    jobs_dir = root / "harbor-jobs"
    adapter = FilesystemHarborExecutionAdapter()
    prepared = adapter.prepare(
        lock,
        root,
        jobs_dir,
        base_url,
        harbor_source,
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    append_event(events, "harbor_started")
    outcome = adapter.execute(
        prepared,
        harbor_source,
        jobs_dir,
        root / "harbor.log",
        environment={
            "HF_TOKEN": token,
            "OPENAI_API_KEY": token,
            "OPENAI_BASE_URL": f"{base_url}/v1",
        },
        timeout_seconds=lock.timeout_seconds,
        stream_runner=stream_runner,
    )
    append_event(events, "harbor_finished", exit_code=outcome.exit_code)
    if outcome.exit_code != 0:
        raise WorkerError(f"Harbor exited with status {outcome.exit_code}")
    if outcome.verification is None:
        raise WorkerError("Harbor produced no validated compatibility bundle")
    write_json(root / "verification.json", outcome.verification)
    append_event(events, "verification_validated")


def resume_and_probe_endpoint(
    root: Path,
    events: Path,
    lock: RunLock,
    manager: EndpointManager,
    token: str,
    *,
    readiness_timeout_seconds: int = 3600,
    compatible_locks: Sequence[RunLock] = (),
) -> str:
    _endpoint_binding(lock)
    append_event(events, "endpoint_resume_requested")
    manager.resume()
    snapshot = manager.wait_ready(readiness_timeout_seconds)
    validate_endpoint_model(lock, snapshot)
    for compatible in compatible_locks:
        validate_endpoint_model(compatible, snapshot)
    append_event(events, "endpoint_ready", state=endpoint_state(snapshot)[0])
    write_json(root / "endpoint.snapshot.json", redact(snapshot))
    base_url = endpoint_url(snapshot)
    runtime = {
        "controller": controller_environment(lock),
        "endpoint": probe_runtime(base_url, token, endpoint_health_route(snapshot)),
    }
    write_json(root / "runtime-environment.json", redact(runtime))
    append_event(events, "runtime_probed")
    return base_url


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
    pyproject = destination / "pyproject.toml"
    if not pyproject.is_file():
        raise WorkerError("pinned Harbor checkout has no pyproject.toml")
    document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = document.get("project")
    extras = (
        project.get("optional-dependencies") if isinstance(project, Mapping) else None
    )
    if not isinstance(extras, Mapping) or "hf-sandbox" not in extras:
        raise WorkerError(
            "pinned Harbor checkout does not provide the hf-sandbox extra"
        )


def launch_cleanup_watchdog(lock: RunLock, endpoint: EndpointRef, token: str) -> str:
    return launch_cleanup_watchdog_for(
        lock.remote,
        endpoint,
        lock.run_id,
        token,
    )


def launch_cleanup_watchdog_for(
    remote: RemoteExecutionSpec,
    endpoint: EndpointRef,
    owner_id: str,
    token: str,
) -> str:
    from huggingface_hub import HfApi

    controller_job_id = os.environ.get("JOB_ID")
    if not controller_job_id:
        raise WorkerError("controller JOB_ID is required before endpoint resume")
    job_timeout_seconds = min(remote.job.timeout_seconds + 600, 86400)
    command = locked_source_command(
        remote.worker,
        "harbor-hf",
        "watchdog",
        "--controller-job-id",
        controller_job_id,
        "--controller-namespace",
        remote.job.namespace,
        "--endpoint-name",
        endpoint.name,
        "--endpoint-namespace",
        endpoint.namespace,
        "--run-id",
        owner_id,
        "--token-secret-name",
        remote.job.token_secret_name,
        "--timeout-seconds",
        str(remote.job.timeout_seconds),
    )
    api = HfApi(token=token)
    info = api.run_job(
        image=remote.job.image,
        command=command,
        secrets={remote.job.token_secret_name: token},
        flavor=remote.job.flavor,
        timeout=job_timeout_seconds,
        labels={
            "harbor-hf-watchdog": owner_id,
            "harbor-hf-endpoint": endpoint_lease_label_for(
                endpoint.namespace, endpoint.name
            ),
        },
        namespace=remote.job.namespace,
    )
    job_id = getattr(info, "id", None)
    if not isinstance(job_id, str) or not job_id:
        raise WorkerError("cleanup watchdog submission returned no job ID")
    wait_watchdog_ready(
        api,
        job_id,
        remote.job.namespace,
        timeout_seconds=_WATCHDOG_STARTUP_TIMEOUT_SECONDS,
    )
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
        try:
            info = api.inspect_job(job_id=job_id, namespace=namespace)
        except Exception as error:
            if monotonic() >= deadline:
                raise WorkerError(
                    "cleanup watchdog readiness timed out after provider errors"
                ) from error
            sleep(poll_seconds)
            continue
        stage = _job_stage(info)
        if stage in terminal:
            raise WorkerError(f"cleanup watchdog exited before readiness: {stage}")
        labels = getattr(info, "labels", None)
        if isinstance(labels, Mapping) and labels.get(_WATCHDOG_READY_LABEL) == "true":
            return
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
    claim_store: ClaimStore | None = None,
    runner: CommandRunner | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    poll_seconds: float = 10,
) -> dict[str, object]:
    token = os.environ.get(token_secret_name, "")
    if not token:
        raise WorkerError(f"required secret {token_secret_name} is not available")
    os.environ["HF_TOKEN"] = token
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    watchdog_job_id = os.environ.get("JOB_ID", "")
    if not watchdog_job_id:
        raise WorkerError("watchdog JOB_ID is required")
    claims = claim_store or HubClaimStore(controller_namespace, token)
    claim_path, owner = _claim_endpoint(
        claims,
        endpoint_namespace,
        endpoint_name,
        controller_job_id,
        watchdog_job_id,
    )
    readiness_error = _watchdog_readiness_error(
        api,
        watchdog_job_id,
        controller_namespace,
        endpoint_namespace,
        endpoint_name,
        run_id,
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
    snapshot = manager.pause_and_verify()
    claims.release(claim_path, owner)
    if readiness_error is not None:
        raise WorkerError(
            "cleanup watchdog could not confirm its readiness label"
        ) from readiness_error
    return snapshot


def _watchdog_readiness_error(
    api: WatchdogApi,
    watchdog_job_id: str,
    controller_namespace: str,
    endpoint_namespace: str,
    endpoint_name: str,
    run_id: str,
) -> Exception | None:
    try:
        _mark_watchdog_ready(
            api,
            watchdog_job_id,
            controller_namespace,
            endpoint_namespace,
            endpoint_name,
            run_id,
        )
    except Exception as error:
        return error
    return None


def _claim_endpoint(
    claims: ClaimStore,
    endpoint_namespace: str,
    endpoint_name: str,
    controller_job_id: str,
    watchdog_job_id: str,
) -> tuple[str, dict[str, str]]:
    claim_path = endpoint_claim_path(endpoint_namespace, endpoint_name)
    owner = {
        "controller_job_id": controller_job_id,
        "watchdog_job_id": watchdog_job_id,
    }
    try:
        claims.acquire(claim_path, owner)
    except ClaimConflict as error:
        raise WorkerError("endpoint lease is held by another watchdog") from error
    return claim_path, owner


def _mark_watchdog_ready(
    api: WatchdogApi,
    watchdog_job_id: str,
    controller_namespace: str,
    endpoint_namespace: str,
    endpoint_name: str,
    run_id: str,
) -> None:
    api.update_job_labels(
        job_id=watchdog_job_id,
        labels={
            "harbor-hf-watchdog": run_id,
            "harbor-hf-endpoint": endpoint_lease_label_for(
                endpoint_namespace, endpoint_name
            ),
            _WATCHDOG_READY_LABEL: "true",
        },
        namespace=controller_namespace,
    )


def _job_stage(info: object) -> str:
    from huggingface_hub import JobInfo

    job = cast(JobInfo, info)
    stage = job.status.stage
    value = getattr(stage, "value", stage)
    if not isinstance(value, str):
        raise WorkerError("HF Job response has an invalid stage")
    return value.upper()


def validate_endpoint_model(lock: RunLock, snapshot: Mapping[str, object]) -> None:
    deployment = _endpoint_deployment(lock)
    model = snapshot.get("model")
    if not isinstance(model, Mapping):
        raise WorkerError("endpoint response has no model object")
    observed_repo = model.get("repository")
    observed_revision = model.get("revision")
    if observed_repo != lock.model.repo or observed_revision != lock.model.revision:
        raise WorkerError(
            "endpoint model does not match the locked repository and revision"
        )
    image = model.get("image")
    custom = image.get("custom") if isinstance(image, Mapping) else None
    observed_image = custom.get("url") if isinstance(custom, Mapping) else None
    if observed_image != deployment.engine.image:
        raise WorkerError("endpoint image does not match the locked deployment")
    observed_command = model.get("command", [])
    if (
        not isinstance(observed_command, list)
        or not all(isinstance(argument, str) for argument in observed_command)
        or observed_command != deployment.engine.command
    ):
        raise WorkerError("endpoint command does not match the locked deployment")
    observed_arguments = model.get("args", [])
    if (
        not isinstance(observed_arguments, list)
        or not all(isinstance(argument, str) for argument in observed_arguments)
        or observed_arguments != deployment.engine.arguments
    ):
        raise WorkerError("endpoint arguments do not match the locked deployment")
    observed_environment = model.get("env")
    if (
        not isinstance(observed_environment, Mapping)
        or dict(observed_environment) != deployment.engine.environment
    ):
        raise WorkerError("endpoint environment does not match the locked deployment")
    observed_secrets = model.get("secrets", {})
    if (
        not isinstance(observed_secrets, Mapping)
        or not all(isinstance(name, str) for name in observed_secrets)
        or set(observed_secrets) != set(deployment.engine.secret_names)
    ):
        raise WorkerError("endpoint secret names do not match the locked deployment")
    _validate_endpoint_compute(lock, snapshot)


def require_paused_endpoint(snapshot: Mapping[str, object]) -> None:
    state, ready, _ = endpoint_state(snapshot)
    if state != "paused" or ready != 0:
        raise WorkerError(
            "endpoint must be paused with zero ready replicas before ownership"
        )


def _validate_endpoint_compute(lock: RunLock, snapshot: Mapping[str, object]) -> None:
    deployment = _endpoint_deployment(lock)
    provider = snapshot.get("provider")
    compute = snapshot.get("compute")
    if not isinstance(provider, Mapping) or not isinstance(compute, Mapping):
        raise WorkerError("endpoint response has no deployment compute identity")
    vendor = provider.get("vendor")
    region = provider.get("region")
    observed_region = f"{vendor}-{region}"
    instance_type = compute.get("instanceType")
    normalized_hardware = (
        instance_type.removeprefix("nvidia-")
        if isinstance(instance_type, str)
        else None
    )
    instance_size = compute.get("instanceSize")
    if (
        observed_region != deployment.region
        or normalized_hardware != deployment.hardware
        or instance_size != f"x{deployment.accelerator_count}"
    ):
        raise WorkerError("endpoint compute does not match the locked deployment")
    scaling = compute.get("scaling")
    if not isinstance(scaling, Mapping):
        raise WorkerError("endpoint response has no scaling configuration")
    settings = EndpointSettings.model_validate(deployment.parameters)
    expected_scaling = {
        "minReplica": settings.min_replicas,
        "maxReplica": settings.max_replicas,
    }
    for field, expected in expected_scaling.items():
        if scaling.get(field) != expected:
            raise WorkerError("endpoint scaling does not match the locked deployment")


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise WorkerError(f"required controller executable is missing: {name}")


def _finalize_evidence(root: Path, token: str) -> None:
    redacted_paths = scrub_secret_paths(root, token)
    if redacted_paths:
        append_event(
            root / "events.jsonl",
            "secret_paths_redacted",
            count=redacted_paths,
        )
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


def probe_runtime(
    base_url: str, token: str, health_route: str = "/health"
) -> dict[str, object]:
    probes: dict[str, object] = {}
    for name, path in (
        ("health", health_route),
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


def _expected_trial_count(lock: RunLock) -> int:
    return len(lock.benchmark_task_digests) * lock.attempts


def _expected_task_counts(lock: RunLock) -> dict[str, int]:
    return {task: lock.attempts for task in lock.benchmark_task_digests}


def _expected_agent_version(lock: RunLock) -> str:
    if lock.agent.revision_kind == "package":
        return lock.agent.revision
    assert lock.agent.reported_version is not None
    return lock.agent.reported_version
