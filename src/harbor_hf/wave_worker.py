from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from harbor_hf.campaigns import (
    CampaignLock,
    CampaignTrialLock,
    WaveLock,
    WaveRunLock,
    WaveShardLock,
    build_wave_lock,
)
from harbor_hf.evidence import (
    append_event,
    archive_directory,
    assert_secret_absent,
    redact,
    scrub_secret,
    scrub_secret_paths,
    verify_checksums,
    write_checksums,
    write_json,
)
from harbor_hf.io import load_experiment
from harbor_hf.models import EndpointRef, ExperimentSpec, SourcePin
from harbor_hf.process import CommandRunner, SubprocessRunner, run_streaming
from harbor_hf.runs import RunLock
from harbor_hf.worker import (
    EndpointManager,
    WorkerError,
    build_harbor_trial_command,
    endpoint_state,
    launch_cleanup_watchdog_for,
    prepare_locked_source,
    require_executable,
    require_paused_endpoint,
    resume_and_probe_endpoint,
    validate_endpoint_model,
    validate_harbor_result,
)


class Clock(Protocol):
    def __call__(self) -> datetime: ...


class IdentifierFactory(Protocol):
    def __call__(self) -> str: ...


class StreamRunner(Protocol):
    def __call__(
        self,
        command: list[str],
        log_path: Path,
        *,
        environment: dict[str, str],
        timeout_seconds: int,
    ) -> int: ...


class SourcePreparer(Protocol):
    def __call__(
        self, source: SourcePin, destination: Path, runner: CommandRunner
    ) -> None: ...


class WatchdogLauncher(Protocol):
    def __call__(self, lock: WaveLock, endpoint: EndpointRef, token: str) -> str: ...


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LockedSubmitWaveAction(FrozenModel):
    action_id: str
    action_key: str
    kind: str = "submit-wave"
    campaign_id: str
    deployment_digest: str
    shard_ids: list[str]


class ExecutionLock(FrozenModel):
    schema_version: str = "harbor-hf/execution-lock/v1alpha1"
    execution_id: str
    created_at: datetime
    campaign_id: str
    wave_id: str
    run_id: str
    shard_id: str
    trial_id: str
    task_name: str
    task_digest: str
    logical_attempt: int
    physical_attempt: int


def validate_wave_lock(
    spec: ExperimentSpec, campaign: CampaignLock, lock: WaveLock
) -> None:
    action = LockedSubmitWaveAction(
        action_id=lock.action_id,
        action_key=lock.action_key,
        campaign_id=lock.campaign_id,
        deployment_digest=lock.deployment_digest,
        shard_ids=lock.shard_ids,
    )
    try:
        expected = build_wave_lock(campaign, spec, action)
    except ValueError as error:
        raise WorkerError(f"wave lock cannot be resolved: {error}") from error
    if lock != expected:
        raise WorkerError("wave lock fields do not match the campaign and manifest")


def run_wave_worker(
    manifest_path: Path,
    campaign_lock_path: Path,
    wave_lock_path: Path,
    output_root: Path,
    *,
    runner: CommandRunner | None = None,
    stream_runner: StreamRunner = run_streaming,
    source_preparer: SourcePreparer = prepare_locked_source,
    watchdog_launcher: WatchdogLauncher | None = None,
    identifier: IdentifierFactory = lambda: uuid.uuid4().hex,
    clock: Clock = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
) -> Path:
    spec = load_experiment(manifest_path)
    campaign = CampaignLock.model_validate_json(
        campaign_lock_path.read_text(encoding="utf-8")
    )
    lock = WaveLock.model_validate_json(wave_lock_path.read_text(encoding="utf-8"))
    validate_wave_lock(spec, campaign, lock)
    token = os.environ.get(lock.remote.job.token_secret_name, "")
    if not token:
        raise WorkerError(
            f"required secret {lock.remote.job.token_secret_name} is not available"
        )
    os.environ["HF_TOKEN"] = token

    process_runner = runner or SubprocessRunner()
    destination = output_root / lock.artifact_prefix
    _reject_terminal_wave(destination)
    with tempfile.TemporaryDirectory(prefix="harbor-hf-wave-") as staging_name:
        staging = Path(staging_name) / campaign.artifact_prefix
        _stage_campaign_records(staging, campaign, lock, token, output_root)
        return _run_staged_wave(
            manifest_path,
            campaign,
            lock,
            staging,
            output_root,
            token,
            process_runner,
            stream_runner,
            source_preparer,
            watchdog_launcher,
            identifier,
            clock,
            monotonic,
        )


def _run_staged_wave(
    manifest_path: Path,
    campaign: CampaignLock,
    lock: WaveLock,
    campaign_root: Path,
    output_root: Path,
    token: str,
    runner: CommandRunner,
    stream_runner: StreamRunner,
    source_preparer: SourcePreparer,
    watchdog_launcher: WatchdogLauncher | None,
    identifier: IdentifierFactory,
    clock: Clock,
    monotonic: Callable[[], float],
) -> Path:
    wave_root = campaign_root / "waves" / lock.wave_id
    wave_root.mkdir(parents=True)
    write_json(wave_root / "wave.lock.json", lock.model_dump(mode="json"))
    events = wave_root / "events.jsonl"
    append_event(events, "wave_started", wave_id=lock.wave_id)
    manager = EndpointManager(lock.endpoint.namespace, lock.endpoint.name, runner)
    representative = lock.runs[0].configuration
    error: Exception | None = None
    cleanup_error: Exception | None = None
    watchdog_started = False
    shard_checksums: dict[str, str] = {}
    try:
        require_executable("git")
        harbor_source = (
            campaign_root.parent
            / "sources"
            / f"harbor-{lock.remote.harbor.source.revision}"
        )
        source_preparer(lock.remote.harbor.source, harbor_source, runner)
        baseline = manager.describe()
        for run in lock.runs:
            validate_endpoint_model(run.configuration, baseline)
        require_paused_endpoint(baseline)
        append_event(events, "endpoint_baseline_validated")
        launcher = watchdog_launcher or _launch_wave_watchdog
        watchdog_id = launcher(lock, lock.endpoint, token)
        watchdog_started = True
        append_event(events, "endpoint_lease_acquired", watchdog_job_id=watchdog_id)
        append_event(events, "cleanup_watchdog_started", job_id=watchdog_id)
        deadline = monotonic() + lock.duration_seconds
        readiness_timeout = min(3600, _remaining_seconds(deadline, monotonic))
        base_url = resume_and_probe_endpoint(
            wave_root,
            events,
            representative,
            manager,
            token,
            readiness_timeout_seconds=readiness_timeout,
            compatible_locks=tuple(run.configuration for run in lock.runs[1:]),
        )
        shard_checksums = _execute_shards(
            manifest_path,
            campaign,
            lock,
            campaign_root,
            output_root,
            harbor_source,
            base_url,
            token,
            stream_runner,
            deadline,
            identifier,
            clock,
            monotonic,
        )
    except Exception as caught:
        error = caught
    finally:
        if watchdog_started:
            append_event(events, "endpoint_pause_requested")
            try:
                final_snapshot = manager.pause_and_verify()
                state, ready, target = endpoint_state(final_snapshot)
                write_json(wave_root / "endpoint.final.json", redact(final_snapshot))
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

    terminal_error = error or cleanup_error
    summary: dict[str, object] = {
        "wave_id": lock.wave_id,
        "campaign_id": campaign.campaign_id,
        "shard_checksums": shard_checksums,
        "endpoint_cleanup_verified": cleanup_error is None and watchdog_started,
    }
    if terminal_error is None:
        append_event(events, "wave_succeeded")
        write_json(wave_root / "wave-summary.json", summary)
        _finalize_unit(wave_root, token)
        (wave_root / "_SUCCESS").write_text("\n", encoding="utf-8")
        _publish_unit(wave_root, output_root / lock.artifact_prefix)
        return output_root / lock.artifact_prefix

    failure_message = str(terminal_error).replace(token, "[REDACTED]")
    summary["error_type"] = type(terminal_error).__name__
    summary["message"] = failure_message
    if error is not None and cleanup_error is not None:
        summary["cleanup_error"] = {
            "error_type": type(cleanup_error).__name__,
            "message": str(cleanup_error).replace(token, "[REDACTED]"),
        }
    append_event(events, "wave_failed", error_type=type(terminal_error).__name__)
    write_json(wave_root / "wave-summary.json", summary)
    _finalize_unit(wave_root, token)
    write_json(wave_root / "_FAILED", summary)
    _publish_unit(wave_root, output_root / lock.artifact_prefix)
    if error is not None and cleanup_error is not None:
        failure_message += "; endpoint cleanup failed: " + str(cleanup_error).replace(
            token, "[REDACTED]"
        )
    raise WorkerError(failure_message) from terminal_error


def _execute_shards(
    manifest_path: Path,
    campaign: CampaignLock,
    lock: WaveLock,
    campaign_root: Path,
    output_root: Path,
    harbor_source: Path,
    base_url: str,
    token: str,
    stream_runner: StreamRunner,
    deadline: float,
    identifier: IdentifierFactory,
    clock: Clock,
    monotonic: Callable[[], float],
) -> dict[str, str]:
    shards = [(run, shard) for run in lock.runs for shard in run.shards]
    workers = min(lock.max_concurrent_shards, len(shards))
    results: dict[str, str] = {}
    failures: list[Exception] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _execute_shard,
                manifest_path,
                campaign,
                lock,
                run,
                shard,
                campaign_root,
                output_root,
                harbor_source,
                base_url,
                token,
                stream_runner,
                deadline,
                identifier,
                clock,
                monotonic,
            ): shard.shard.shard_id
            for run, shard in shards
        }
        for future in as_completed(futures):
            shard_id = futures[future]
            try:
                results[shard_id] = future.result()
            except Exception as error:
                failures.append(error)
    if failures:
        raise failures[0]
    return dict(sorted(results.items()))


def _execute_shard(
    manifest_path: Path,
    campaign: CampaignLock,
    wave: WaveLock,
    run: WaveRunLock,
    locked_shard: WaveShardLock,
    campaign_root: Path,
    output_root: Path,
    harbor_source: Path,
    base_url: str,
    token: str,
    stream_runner: StreamRunner,
    deadline: float,
    identifier: IdentifierFactory,
    clock: Clock,
    monotonic: Callable[[], float],
) -> str:
    shard = locked_shard.shard
    shard_root = (
        campaign_root / "runs" / run.configuration.run_id / "shards" / shard.shard_id
    )
    shard_root.mkdir(parents=True, exist_ok=True)
    write_json(shard_root / "shard.lock.json", shard.model_dump(mode="json"))
    events = shard_root / "events.jsonl"
    append_event(events, "shard_started", shard_id=shard.shard_id)
    trial_checksums: dict[str, str] = {}
    for trial in shard.trials:
        destination = _trial_destination(output_root, campaign, run, trial)
        trial_root = shard_root.parent.parent / "trials" / trial.trial_id
        if _valid_terminal_trial(
            destination,
            trial,
            campaign_id=campaign.campaign_id,
            wave_id=wave.wave_id,
            run_id=run.configuration.run_id,
            shard_id=shard.shard_id,
        ):
            shutil.copytree(destination, trial_root)
            append_event(events, "trial_recovered", trial_id=trial.trial_id)
        else:
            _prepare_trial_recovery(destination, trial_root)
            _execute_trial(
                manifest_path,
                campaign,
                wave,
                run.configuration,
                shard.shard_id,
                trial,
                trial_root,
                output_root,
                harbor_source,
                base_url,
                token,
                stream_runner,
                deadline,
                identifier,
                clock,
                monotonic,
            )
            append_event(events, "trial_completed", trial_id=trial.trial_id)
        trial_checksums[trial.trial_id] = _file_digest(trial_root / "checksums.json")
    summary = {
        "campaign_id": campaign.campaign_id,
        "run_id": run.configuration.run_id,
        "shard_id": shard.shard_id,
        "trial_checksums": dict(sorted(trial_checksums.items())),
    }
    append_event(events, "shard_succeeded")
    write_json(shard_root / "shard-summary.json", summary)
    _finalize_unit(shard_root, token)
    (shard_root / "_SUCCESS").write_text("\n", encoding="utf-8")
    _publish_unit(shard_root, output_root / locked_shard.artifact_prefix)
    return _file_digest(shard_root / "checksums.json")


def _execute_trial(
    manifest_path: Path,
    campaign: CampaignLock,
    wave: WaveLock,
    run: RunLock,
    shard_id: str,
    trial: CampaignTrialLock,
    trial_root: Path,
    output_root: Path,
    harbor_source: Path,
    base_url: str,
    token: str,
    stream_runner: StreamRunner,
    deadline: float,
    identifier: IdentifierFactory,
    clock: Clock,
    monotonic: Callable[[], float],
) -> None:
    trial_root.mkdir(parents=True, exist_ok=True)
    executions = trial_root / "executions"
    executions.mkdir(exist_ok=True)
    execution_id = _execution_id(identifier)
    execution_root = executions / execution_id
    execution_root.mkdir()
    jobs_dir = execution_root / "harbor-jobs"
    jobs_dir.mkdir()
    physical_attempt = len([path for path in executions.iterdir() if path.is_dir()])
    execution = ExecutionLock(
        execution_id=execution_id,
        created_at=clock().astimezone(UTC),
        campaign_id=campaign.campaign_id,
        wave_id=wave.wave_id,
        run_id=run.run_id,
        shard_id=shard_id,
        trial_id=trial.trial_id,
        task_name=trial.task_name,
        task_digest=trial.task_digest,
        logical_attempt=trial.logical_attempt,
        physical_attempt=physical_attempt,
    )
    write_json(
        execution_root / "execution.lock.json", execution.model_dump(mode="json")
    )
    shutil.copyfile(manifest_path, execution_root / "manifest.yaml")
    events = execution_root / "events.jsonl"
    append_event(events, "execution_started", execution_id=execution_id)
    error: Exception | None = None
    try:
        command = build_harbor_trial_command(
            run,
            jobs_dir,
            base_url,
            harbor_source,
            task_name=trial.task_name,
        )
        timeout = _remaining_seconds(deadline, monotonic)
        exit_code = stream_runner(
            command,
            execution_root / "harbor.log",
            environment={
                "HF_TOKEN": token,
                "OPENAI_API_KEY": token,
                "OPENAI_BASE_URL": f"{base_url}/v1",
            },
            timeout_seconds=timeout,
        )
        append_event(events, "harbor_finished", exit_code=exit_code)
        if exit_code != 0:
            raise WorkerError(f"Harbor exited with status {exit_code}")
        verifier = validate_harbor_result(
            jobs_dir,
            expected_trials=1,
            expected_task_counts={trial.task_name: 1},
            expected_attempts_per_task=1,
            expected_task_names=[trial.task_name],
            expected_task_digests={trial.task_name: trial.task_digest},
            expected_agent_name=run.agent.name,
            expected_agent_version=_expected_agent_version(run),
            expected_model_provider="openai",
            expected_model_name=wave.endpoint.served_model_name,
        )
        write_json(execution_root / "verification.json", verifier)
        append_event(events, "execution_succeeded")
    except Exception as caught:
        error = caught
        append_event(events, "execution_failed", error_type=type(caught).__name__)
    _finalize_execution(execution_root, token)
    if error is None:
        (execution_root / "_SUCCESS").write_text("\n", encoding="utf-8")
    else:
        write_json(
            execution_root / "_FAILED",
            {
                "error_type": type(error).__name__,
                "message": str(error).replace(token, "[REDACTED]"),
            },
        )
    destination = (
        output_root
        / campaign.artifact_prefix
        / "runs"
        / run.run_id
        / "trials"
        / trial.trial_id
        / "executions"
        / execution_id
    )
    _publish_unit(execution_root, destination)
    if error is not None:
        raise error

    write_json(trial_root / "trial.lock.json", trial.model_dump(mode="json"))
    write_json(
        trial_root / "trial-summary.json",
        {
            "trial_id": trial.trial_id,
            "execution_id": execution_id,
            "execution_checksum": _file_digest(execution_root / "checksums.json"),
        },
    )
    append_event(trial_root / "events.jsonl", "trial_succeeded")
    _finalize_unit(trial_root, token)
    (trial_root / "_SUCCESS").write_text("\n", encoding="utf-8")
    _publish_unit(
        trial_root,
        output_root
        / campaign.artifact_prefix
        / "runs"
        / run.run_id
        / "trials"
        / trial.trial_id,
    )


def _stage_campaign_records(
    campaign_root: Path,
    campaign: CampaignLock,
    wave: WaveLock,
    token: str,
    output_root: Path,
) -> None:
    campaign_root.mkdir(parents=True)
    campaign_lock_path = campaign_root / "campaign.lock.json"
    write_json(campaign_lock_path, campaign.model_dump(mode="json"))
    assert_secret_absent(campaign_root, token)
    _publish_immutable_file(
        campaign_lock_path,
        output_root / campaign.artifact_prefix / "campaign.lock.json",
    )
    _publish_digest_sidecar(campaign_lock_path, output_root / campaign.artifact_prefix)
    for run in wave.runs:
        run_root = campaign_root / "runs" / run.configuration.run_id
        run_root.mkdir(parents=True)
        run_lock_path = run_root / "run.lock.json"
        write_json(run_lock_path, run.configuration.model_dump(mode="json"))
        assert_secret_absent(run_root, token)
        destination = output_root / run.artifact_prefix
        _publish_immutable_file(run_lock_path, destination / "run.lock.json")
        _publish_digest_sidecar(run_lock_path, destination)


def _valid_terminal_trial(
    path: Path,
    expected: CampaignTrialLock,
    *,
    campaign_id: str,
    wave_id: str,
    run_id: str,
    shard_id: str,
) -> bool:
    if not path.exists():
        return False
    markers = [
        name
        for name in ("_SUCCESS", "_FAILED", "_CANCELLED")
        if (path / name).is_file()
    ]
    if not markers:
        return False
    if markers != ["_SUCCESS"]:
        raise WorkerError("terminal trial evidence is not a valid success")
    _validate_terminal_trial(
        path,
        expected,
        campaign_id=campaign_id,
        wave_id=wave_id,
        run_id=run_id,
        shard_id=shard_id,
    )
    return True


def _validate_terminal_trial(
    path: Path,
    expected: CampaignTrialLock,
    *,
    campaign_id: str,
    wave_id: str,
    run_id: str,
    shard_id: str,
) -> None:
    try:
        observed = json.loads((path / "trial.lock.json").read_text(encoding="utf-8"))
        if observed != expected.model_dump(mode="json"):
            raise WorkerError("terminal trial lock does not match the wave")
        summary = json.loads((path / "trial-summary.json").read_text(encoding="utf-8"))
        execution_id = summary.get("execution_id")
        execution_checksum = summary.get("execution_checksum")
        if not isinstance(execution_id, str):
            raise WorkerError("terminal trial summary has no execution identity")
        if summary.get("trial_id") != expected.trial_id:
            raise WorkerError("terminal trial summary has the wrong trial identity")
        execution = path / "executions" / execution_id
        if not (execution / "_SUCCESS").is_file():
            raise WorkerError("terminal trial execution is not successful")
        execution_lock = ExecutionLock.model_validate_json(
            (execution / "execution.lock.json").read_text(encoding="utf-8")
        )
        _validate_execution_identity(
            execution_lock,
            execution_id,
            expected,
            campaign_id=campaign_id,
            wave_id=wave_id,
            run_id=run_id,
            shard_id=shard_id,
        )
        verify_checksums(execution)
        if execution_checksum != _file_digest(execution / "checksums.json"):
            raise WorkerError("terminal trial summary has the wrong child checksum")
        verify_checksums(path)
    except (OSError, ValueError, RuntimeError) as error:
        if isinstance(error, WorkerError):
            raise
        raise WorkerError(
            "terminal trial evidence failed checksum validation"
        ) from error


def _validate_execution_identity(
    execution: ExecutionLock,
    execution_id: str,
    expected: CampaignTrialLock,
    *,
    campaign_id: str,
    wave_id: str,
    run_id: str,
    shard_id: str,
) -> None:
    observed = (
        execution.execution_id,
        execution.campaign_id,
        execution.wave_id,
        execution.run_id,
        execution.shard_id,
        execution.trial_id,
        execution.task_name,
        execution.task_digest,
        execution.logical_attempt,
    )
    locked = (
        execution_id,
        campaign_id,
        wave_id,
        run_id,
        shard_id,
        expected.trial_id,
        expected.task_name,
        expected.task_digest,
        expected.logical_attempt,
    )
    if observed != locked:
        raise WorkerError("terminal execution identity does not match its trial")


def _prepare_trial_recovery(destination: Path, trial_root: Path) -> None:
    if any(
        (destination / marker).is_file()
        for marker in ("_SUCCESS", "_FAILED", "_CANCELLED")
    ):
        raise WorkerError("terminal trial evidence cannot be overwritten")
    trial_root.mkdir(parents=True, exist_ok=True)
    existing = destination / "executions"
    if existing.is_dir():
        shutil.copytree(existing, trial_root / "executions")


def _trial_destination(
    output_root: Path,
    campaign: CampaignLock,
    run: WaveRunLock,
    trial: CampaignTrialLock,
) -> Path:
    return (
        output_root
        / campaign.artifact_prefix
        / "runs"
        / run.configuration.run_id
        / "trials"
        / trial.trial_id
    )


def _finalize_execution(root: Path, token: str) -> None:
    _redact_unit(root, token)
    archive_directory(root / "harbor-jobs", root / "artifacts.tar.gz")
    assert_secret_absent(root, token)
    write_checksums(root)


def _finalize_unit(root: Path, token: str) -> None:
    _redact_unit(root, token)
    write_checksums(root)


def _redact_unit(root: Path, token: str) -> None:
    redacted_paths = scrub_secret_paths(root, token)
    if redacted_paths:
        append_event(
            root / "events.jsonl", "secret_paths_redacted", count=redacted_paths
        )
    scrubbed = scrub_secret(root, token)
    if scrubbed:
        append_event(root / "events.jsonl", "secrets_redacted", files=scrubbed)
    assert_secret_absent(root, token)


def _publish_unit(source: Path, destination: Path) -> None:
    markers = [
        name
        for name in ("_SUCCESS", "_FAILED", "_CANCELLED")
        if (source / name).is_file()
    ]
    if len(markers) != 1:
        raise WorkerError("finalized wave evidence must have one terminal marker")
    verify_checksums(source)
    destination.mkdir(parents=True, exist_ok=True)
    marker_paths = {source / marker for marker in markers}
    for path in sorted(source.rglob("*")):
        if path.is_dir() or path in marker_paths:
            continue
        relative = path.relative_to(source)
        _publish_immutable_file(path, destination / relative)
    _publish_immutable_file(source / markers[0], destination / markers[0])


def _publish_immutable_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise WorkerError(
                f"evidence path already has different contents: {destination}"
            )
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_digest_sidecar(source: Path, destination: Path) -> None:
    sidecar = source.with_suffix(source.suffix + ".sha256")
    sidecar.write_text(_file_digest(source) + "\n", encoding="utf-8")
    _publish_immutable_file(sidecar, destination / sidecar.name)


def _reject_terminal_wave(destination: Path) -> None:
    if any(
        (destination / marker).is_file()
        for marker in ("_SUCCESS", "_FAILED", "_CANCELLED")
    ):
        raise WorkerError("deployment wave already has terminal evidence")


def _launch_wave_watchdog(lock: WaveLock, endpoint: EndpointRef, token: str) -> str:
    return launch_cleanup_watchdog_for(lock.remote, endpoint, lock.wave_id, token)


def _execution_id(identifier: IdentifierFactory) -> str:
    value = identifier()
    if len(value) != 32 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise WorkerError(
            "execution identifier must be 32 lowercase hexadecimal digits"
        )
    return f"exec-{value}"


def _remaining_seconds(deadline: float, monotonic: Callable[[], float]) -> int:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise WorkerError("deployment wave duration bound was reached")
    return max(1, math.ceil(remaining))


def _expected_agent_version(lock: RunLock) -> str:
    if lock.agent.revision_kind == "package":
        return lock.agent.revision
    assert lock.agent.reported_version is not None
    return lock.agent.reported_version


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
