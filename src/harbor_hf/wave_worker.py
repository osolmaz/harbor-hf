from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from harbor_hf.campaigns import (
    CampaignLock,
    CampaignTrialLock,
    EndpointWaveTarget,
    ProviderWaveTarget,
    WaveLock,
    WaveRunLock,
    WaveShardLock,
    build_wave_lock,
)
from harbor_hf.control import RetryCategory
from harbor_hf.coordination import (
    ClaimConflict,
    ClaimStore,
    CoordinationError,
    HubClaimStore,
    wave_worker_claim_path,
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
from harbor_hf.harbor_adapter import (
    FilesystemHarborExecutionAdapter,
    HarborVerificationFailure,
)
from harbor_hf.harbor_adapter.exporter import refresh_retained_bundle
from harbor_hf.io import load_experiment
from harbor_hf.models import EndpointRef, ExperimentSpec, SourcePin
from harbor_hf.private_artifacts import (
    PrivateArtifactRequirementError,
    build_private_artifact_manifest,
    openclaw_execution_started,
    openclaw_execution_was_attempted,
    sanitize_private_artifact_tree,
    write_private_artifact_manifest,
)
from harbor_hf.process import CommandRunner, SubprocessRunner, run_streaming
from harbor_hf.provider_models import (
    ProviderEndpointEvidence,
    ProviderTarget,
    unavailable,
)
from harbor_hf.provider_proxy import ProviderEvidenceProxy
from harbor_hf.providers import routed_provider_model
from harbor_hf.runs import RunLock
from harbor_hf.worker import (
    EndpointManager,
    HarborTrialFailure,
    WorkerError,
    controller_environment,
    endpoint_state,
    launch_cleanup_watchdog_for,
    prepare_locked_source,
    require_executable,
    require_paused_endpoint,
    resume_and_probe_endpoint,
    validate_endpoint_model,
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


_TRIAL_FAILURE_MARKERS: tuple[tuple[RetryCategory, tuple[str, ...]], ...] = (
    ("authentication", ("authentication", "unauthorized", "forbidden")),
    ("rate-limit", ("ratelimit", "rate_limit")),
    ("quota", ("quota",)),
    (
        "transient",
        ("timeout", "connection", "serviceunavailable", "internalserver", "apierror"),
    ),
    ("configuration", ("badrequest", "notfound", "configuration")),
)

_TERMINAL_MARKERS = ("_SUCCESS", "_FAILED", "_CANCELLED")


class LockedSubmitWaveAction(FrozenModel):
    action_id: str
    action_key: str
    kind: Literal["submit-wave", "retry-shard"] = "submit-wave"
    campaign_id: str
    deployment_digest: str
    shard_ids: list[str]
    trial_ids: list[str] = Field(default_factory=list)
    estimated_cost_microusd: int | None = None


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


class _EndpointWaveLifecycle:
    def __init__(
        self,
        lock: WaveLock,
        wave_root: Path,
        events: Path,
        runner: CommandRunner,
        token: str,
        watchdog_launcher: WatchdogLauncher | None,
    ) -> None:
        if not isinstance(lock.target, EndpointWaveTarget):
            raise TypeError("endpoint lifecycle requires an endpoint target")
        self.lock = lock
        self.wave_root = wave_root
        self.events = events
        self.token = token
        self.launcher = watchdog_launcher or _launch_wave_watchdog
        endpoint = lock.target.endpoint
        self.endpoint = endpoint
        self.manager = EndpointManager(endpoint.namespace, endpoint.name, runner)
        self.owned = False

    def prepare(self, deadline: float, monotonic: Callable[[], float]) -> str:
        baseline = self.manager.describe()
        for run in self.lock.runs:
            validate_endpoint_model(run.configuration, baseline)
        require_paused_endpoint(baseline)
        append_event(self.events, "endpoint_baseline_validated")
        watchdog_id = self.launcher(self.lock, self.endpoint, self.token)
        self.owned = True
        append_event(
            self.events, "endpoint_lease_acquired", watchdog_job_id=watchdog_id
        )
        append_event(self.events, "cleanup_watchdog_started", job_id=watchdog_id)
        readiness_timeout = min(3600, _remaining_seconds(deadline, monotonic))
        return resume_and_probe_endpoint(
            self.wave_root,
            self.events,
            self.lock.runs[0].configuration,
            self.manager,
            self.token,
            readiness_timeout_seconds=readiness_timeout,
            compatible_locks=tuple(run.configuration for run in self.lock.runs[1:]),
        )

    def cleanup(self) -> Exception | None:
        if not self.owned:
            append_event(
                self.events, "endpoint_cleanup_skipped", reason="lease_not_owned"
            )
            return None
        append_event(self.events, "endpoint_pause_requested")
        try:
            final_snapshot = self.manager.pause_and_verify()
            state, ready, target = endpoint_state(final_snapshot)
            write_json(self.wave_root / "endpoint.final.json", redact(final_snapshot))
            append_event(
                self.events,
                "endpoint_paused",
                state=state,
                ready_replicas=ready,
                target_replicas=target,
            )
        except Exception as error:
            append_event(
                self.events,
                "endpoint_cleanup_failed",
                error=type(error).__name__,
            )
            return error
        return None


def validate_wave_lock(
    spec: ExperimentSpec, campaign: CampaignLock, lock: WaveLock
) -> None:
    matching_runs = [
        run for run in campaign.runs if run.deployment_digest == lock.deployment_digest
    ]
    if not matching_runs:
        raise WorkerError("wave lock references an unknown deployment")
    estimates = {run.estimated_wave_cost_microusd for run in matching_runs}
    if len(estimates) != 1:
        raise WorkerError("wave deployment has inconsistent spend estimates")
    action = LockedSubmitWaveAction(
        action_id=lock.action_id,
        action_key=lock.action_key,
        kind=lock.action_kind,
        campaign_id=lock.campaign_id,
        deployment_digest=lock.deployment_digest,
        shard_ids=lock.shard_ids,
        trial_ids=lock.trial_ids,
        estimated_cost_microusd=estimates.pop(),
    )
    try:
        expected = build_wave_lock(campaign, spec, action, endpoint=lock.endpoint)
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
    claim_store: ClaimStore | None = None,
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

    with _wave_worker_lease(lock, token, claim_store, clock):
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


@contextmanager
def _wave_worker_lease(
    lock: WaveLock,
    token: str,
    claim_store: ClaimStore | None,
    clock: Clock,
) -> Iterator[None]:
    job_id = os.environ.get("JOB_ID", "")
    if not job_id:
        raise WorkerError("wave worker claim requires JOB_ID")
    claims = claim_store
    if claims is None:
        claims = HubClaimStore(lock.remote.job.namespace, token)
    now = clock().astimezone(UTC)
    owner = {
        "campaign_id": lock.campaign_id,
        "wave_id": lock.wave_id,
        "job_id": job_id,
        "expires_at": (
            now + timedelta(seconds=lock.remote.job.timeout_seconds)
        ).isoformat(),
    }
    path = wave_worker_claim_path(lock.campaign_id, lock.wave_id)
    try:
        claims.acquire(path, owner)
    except ClaimConflict as error:
        raise WorkerError("wave worker is already active") from error
    try:
        yield
    finally:
        with suppress(CoordinationError):
            claims.release(path, owner)


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
    lifecycle = (
        _EndpointWaveLifecycle(
            lock,
            wave_root,
            events,
            runner,
            token,
            watchdog_launcher,
        )
        if isinstance(lock.target, EndpointWaveTarget)
        else None
    )
    error: Exception | None = None
    cleanup_error: Exception | None = None
    provider_proxy: ProviderEvidenceProxy | None = None
    shard_checksums: dict[str, str] = {}
    try:
        require_executable("git")
        harbor_source = (
            campaign_root.parent
            / "sources"
            / f"harbor-{lock.remote.harbor.source.revision}"
        )
        source_preparer(lock.remote.harbor.source, harbor_source, runner)
        deadline = monotonic() + lock.duration_seconds
        base_url, provider_proxy = _prepare_wave_transport(
            lock,
            wave_root,
            events,
            lifecycle,
            token,
            deadline,
            monotonic,
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
        cleanup_error = _cleanup_wave_transport(lifecycle, provider_proxy)

    terminal_error = error or cleanup_error
    summary: dict[str, object] = {
        "wave_id": lock.wave_id,
        "campaign_id": campaign.campaign_id,
        "shard_checksums": shard_checksums,
        "endpoint_cleanup_verified": (
            cleanup_error is None and lifecycle.owned
            if lifecycle is not None
            else unavailable("not_applicable").model_dump(mode="json")
        ),
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


def _prepare_wave_transport(
    lock: WaveLock,
    wave_root: Path,
    events: Path,
    lifecycle: _EndpointWaveLifecycle | None,
    token: str,
    deadline: float,
    monotonic: Callable[[], float],
) -> tuple[str, ProviderEvidenceProxy | None]:
    if lifecycle is not None:
        return lifecycle.prepare(deadline, monotonic), None
    target = _prepare_provider_target(lock, wave_root, events)
    proxy = ProviderEvidenceProxy(
        target,
        token=token,
        evidence_path=wave_root / "provider-requests.jsonl",
    )
    return proxy.start(), proxy


def _cleanup_wave_transport(
    lifecycle: _EndpointWaveLifecycle | None,
    provider_proxy: ProviderEvidenceProxy | None,
) -> Exception | None:
    if lifecycle is not None:
        return lifecycle.cleanup()
    if provider_proxy is None:
        return None
    try:
        provider_proxy.close()
    except Exception as error:
        return error
    return None


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
    with (
        ThreadPoolExecutor(max_workers=lock.max_concurrent_shards) as trial_executor,
        ThreadPoolExecutor(max_workers=workers) as shard_executor,
    ):
        futures = {
            shard_executor.submit(
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
                trial_executor=trial_executor,
            ): shard.shard.shard_id
            for run, shard in shards
        }
        for future in as_completed(futures):
            shard_id = futures[future]
            try:
                checksum = future.result()
                if checksum is not None:
                    results[shard_id] = checksum
            except Exception as error:
                failures.append(error)
    if failures:
        raise failures[0]
    return dict(sorted(results.items()))


def _prepare_provider_target(
    lock: WaveLock, wave_root: Path, events: Path
) -> ProviderTarget:
    if not isinstance(lock.target, ProviderWaveTarget):
        raise WorkerError("provider execution requires an Inference Provider target")
    target = lock.target.provider
    write_json(wave_root / "provider-target.json", target.model_dump(mode="json"))
    write_json(
        wave_root / "runtime-environment.json",
        {
            "controller": controller_environment(lock.runs[0].configuration),
            "provider": {
                "service": target.service,
                "requested_model": target.model,
                "routed_model": routed_provider_model(target),
                "routing": target.routing.model_dump(mode="json"),
                "request_controls": {
                    "max_attempts": target.limits.max_attempts,
                    "max_concurrent_requests": target.limits.max_concurrent_requests,
                    "parameters": target.parameters,
                    "timeout_seconds": target.timeout_seconds,
                },
                "transport": {
                    "kind": "loopback-evidence-proxy",
                    "evidence_path": "provider-requests.jsonl",
                },
                "endpoint": ProviderEndpointEvidence().model_dump(mode="json"),
            },
        },
    )
    append_event(
        events,
        "provider_target_validated",
        service=target.service,
        target_id=target.id,
    )
    return target


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
    *,
    trial_executor: Executor | None = None,
) -> str | None:
    with _trial_execution_pool(
        trial_executor, wave.max_concurrent_shards
    ) as selected_executor:
        return _execute_shard_with_executor(
            manifest_path,
            campaign,
            wave,
            run,
            locked_shard,
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
            selected_executor,
        )


@contextmanager
def _trial_execution_pool(
    executor: Executor | None, max_workers: int
) -> Iterator[Executor]:
    if executor is not None:
        yield executor
        return
    with ThreadPoolExecutor(max_workers=max_workers) as owned_executor:
        yield owned_executor


def _execute_shard_with_executor(
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
    trial_executor: Executor,
) -> str | None:
    shard = locked_shard.shard
    shard_root = (
        campaign_root / "runs" / run.configuration.run_id / "shards" / shard.shard_id
    )
    shard_root.mkdir(parents=True, exist_ok=True)
    write_json(shard_root / "shard.lock.json", shard.model_dump(mode="json"))
    events = shard_root / "events.jsonl"
    append_event(events, "shard_started", shard_id=shard.shard_id)
    trial_checksums: dict[str, str] = {}
    failures: list[tuple[int, Exception]] = []
    pending: dict[Future[None], tuple[int, CampaignTrialLock, Path]] = {}
    deferred = False
    selected_trial_ids = set(wave.trial_ids)
    for trial_index, trial in enumerate(shard.trials):
        destination = _trial_destination(output_root, campaign, run, trial)
        trial_root = shard_root.parent.parent / "trials" / trial.trial_id
        recovered = _valid_terminal_trial(
            destination,
            trial,
            campaign_id=campaign.campaign_id,
            wave_id=None,
            run_id=run.configuration.run_id,
            shard_id=shard.shard_id,
        )
        if recovered:
            shutil.copytree(destination, trial_root)
            append_event(events, "trial_recovered", trial_id=trial.trial_id)
        elif (
            wave.action_kind == "retry-shard"
            and trial.trial_id not in selected_trial_ids
        ):
            deferred = True
            append_event(events, "trial_deferred", trial_id=trial.trial_id)
            continue
        else:
            _prepare_trial_recovery(destination, trial_root)
            future = trial_executor.submit(
                _execute_trial,
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
            pending[future] = (trial_index, trial, trial_root)
            continue
        trial_checksums[trial.trial_id] = _file_digest(trial_root / "checksums.json")
    for future in as_completed(pending):
        trial_index, trial, trial_root = pending[future]
        try:
            future.result()
        except Exception as error:
            failures.append((trial_index, error))
            append_event(
                events,
                "trial_failed",
                trial_id=trial.trial_id,
                error_type=type(error).__name__,
            )
            continue
        append_event(events, "trial_completed", trial_id=trial.trial_id)
        trial_checksums[trial.trial_id] = _file_digest(trial_root / "checksums.json")
    if failures:
        append_event(events, "shard_failed", failed_trials=len(failures))
        raise min(failures, key=lambda item: item[0])[1]
    if deferred:
        append_event(events, "shard_deferred")
        return None
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
    failure_phase: Literal["configuration", "execution", "verification"] = (
        "configuration"
    )
    try:
        trial_base_url = (
            ProviderEvidenceProxy.scoped_base_url(base_url, trial.trial_id)
            if isinstance(wave.target, ProviderWaveTarget)
            else base_url
        )
        adapter = FilesystemHarborExecutionAdapter()
        prepared = adapter.prepare(
            run,
            execution_root,
            jobs_dir,
            trial_base_url,
            harbor_source,
            task_names=[trial.task_name],
            attempts=1,
            concurrency=1,
            expected_task_digests={trial.task_name: trial.task_digest},
        )
        failure_phase = "execution"
        timeout = _remaining_seconds(deadline, monotonic)
        append_event(events, "harbor_started")
        outcome = adapter.execute(
            prepared,
            harbor_source,
            jobs_dir,
            execution_root / "harbor.log",
            environment={
                "HF_TOKEN": token,
                "OPENAI_API_KEY": token,
                "OPENAI_BASE_URL": f"{trial_base_url}/v1",
            },
            timeout_seconds=timeout,
            stream_runner=stream_runner,
            monotonic=monotonic,
            deadline=deadline,
        )
        append_event(events, "harbor_finished", exit_code=outcome.exit_code)
        if outcome.exit_code != 0:
            raise WorkerError(f"Harbor exited with status {outcome.exit_code}")
        failure_phase = "verification"
        if outcome.verification is None:
            raise WorkerError("Harbor produced no validated compatibility bundle")
        write_json(
            execution_root / "verification.json",
            outcome.verification.model_dump(mode="json"),
        )
        build_private_artifact_manifest(execution_root, strict_session=True)
        append_event(events, "execution_succeeded")
    except Exception as caught:
        error = caught
        append_event(events, "execution_failed", error_type=type(caught).__name__)
    failure_record: dict[str, object] | None = None
    if error is not None:
        failure_record = {
            "category": _execution_failure_category(error, failure_phase),
            "error_type": type(error).__name__,
            "message": str(error).replace(token, "[REDACTED]"),
        }
        write_json(execution_root / "failure.json", failure_record)
    _finalize_execution(execution_root, token, strict_compatibility=error is None)
    if error is None:
        (execution_root / "_SUCCESS").write_text("\n", encoding="utf-8")
    else:
        assert failure_record is not None
        write_json(execution_root / "_FAILED", failure_record)
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


def _execution_failure_category(
    error: Exception,
    phase: Literal["configuration", "execution", "verification"],
) -> RetryCategory:
    if isinstance(error, PrivateArtifactRequirementError):
        return "configuration"
    if isinstance(error, HarborVerificationFailure):
        return "benchmark"
    if isinstance(error, HarborTrialFailure):
        name = error.exception_type.lower()
        for category, markers in _TRIAL_FAILURE_MARKERS:
            if any(marker in name for marker in markers):
                return category
        return "benchmark"
    if phase == "configuration":
        return "configuration"
    if phase == "execution":
        return "transient"
    return "benchmark"


def _wave_model_name(wave: WaveLock) -> str:
    if isinstance(wave.target, EndpointWaveTarget):
        return wave.target.endpoint.served_model_name
    return routed_provider_model(wave.target.provider)


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
    wave_id: str | None,
    run_id: str,
    shard_id: str,
) -> bool:
    if not path.exists():
        return False
    markers = _terminal_markers(path)
    if not markers:
        required = (
            "checksums.json",
            "trial.lock.json",
            "trial-summary.json",
        )
        if not all((path / name).is_file() for name in required):
            return False
        _validate_terminal_trial(
            path,
            expected,
            campaign_id=campaign_id,
            wave_id=wave_id,
            run_id=run_id,
            shard_id=shard_id,
        )
        with tempfile.TemporaryDirectory(
            prefix="harbor-hf-trial-marker-"
        ) as marker_directory:
            marker = Path(marker_directory) / "_SUCCESS"
            marker.write_text("\n", encoding="utf-8")
            _publish_immutable_file(marker, path / marker.name)
        verify_checksums(path)
        return True
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
    wave_id: str | None,
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
    wave_id: str | None,
    run_id: str,
    shard_id: str,
) -> None:
    observed = (
        execution.execution_id,
        execution.campaign_id,
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
        run_id,
        shard_id,
        expected.trial_id,
        expected.task_name,
        expected.task_digest,
        expected.logical_attempt,
    )
    if observed != locked or (wave_id is not None and execution.wave_id != wave_id):
        raise WorkerError("terminal execution identity does not match its trial")


def _prepare_trial_recovery(destination: Path, trial_root: Path) -> None:
    if _terminal_markers(destination):
        raise WorkerError("terminal trial evidence cannot be overwritten")
    trial_root.mkdir(parents=True, exist_ok=True)
    existing = destination / "executions"
    if not existing.exists():
        return
    terminal = _terminal_execution_directories(destination)
    recovered = trial_root / "executions"
    recovered.mkdir()
    for execution in sorted(existing.iterdir()):
        if execution.name in terminal:
            shutil.copytree(execution, recovered / execution.name)
        else:
            _remove_path(execution)


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


def _finalize_execution(
    root: Path, token: str, *, strict_compatibility: bool = True
) -> None:
    attempted = openclaw_execution_was_attempted(root)
    rejection_count = 0
    if not strict_compatibility:
        rejection_count = len(sanitize_private_artifact_tree(root))
    session_required = openclaw_execution_started(root, fallback_attempted=attempted)
    _redact_unit(root, token)
    refresh_error = refresh_retained_bundle(root, strict=strict_compatibility)
    if refresh_error is not None:
        append_event(
            root / "events.jsonl",
            "compatibility_refresh_skipped",
            error_type=refresh_error,
        )
    if not strict_compatibility:
        final_rejection_count = len(
            sanitize_private_artifact_tree(root, trust_existing_rejections=True)
        )
        if final_rejection_count > rejection_count:
            refresh_error = refresh_retained_bundle(root, strict=False)
            if refresh_error is not None:
                append_event(
                    root / "events.jsonl",
                    "compatibility_refresh_skipped",
                    error_type=refresh_error,
                )
    write_private_artifact_manifest(
        root,
        strict_session=strict_compatibility,
        session_required=session_required,
        trust_rejections=not strict_compatibility,
    )
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
    markers = _terminal_markers(source)
    if len(markers) != 1:
        raise WorkerError("finalized wave evidence must have one terminal marker")
    verify_checksums(source)
    if _terminal_markers(destination):
        raise WorkerError("terminal evidence destination cannot be overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    protected_executions = _recover_partial_publication(source, destination)
    marker_path = source / markers[0]
    files = [path for path in source.rglob("*") if path.is_file()]
    nested_markers = {
        path for path in files if path != marker_path and path.name in _TERMINAL_MARKERS
    }
    ordered = sorted(
        path for path in files if path != marker_path and path not in nested_markers
    )
    ordered.extend(
        sorted(nested_markers, key=lambda path: (-len(path.parts), str(path)))
    )
    for path in ordered:
        relative = path.relative_to(source)
        if any(
            relative.is_relative_to(protected) for protected in protected_executions
        ):
            continue
        _publish_immutable_file(path, destination / relative)
    verify_checksums(destination)
    _publish_immutable_file(marker_path, destination / marker_path.name)
    verify_checksums(destination)


def _recover_partial_publication(source: Path, destination: Path) -> set[Path]:
    """Remove interrupted content while preserving immutable child executions."""
    terminal = _terminal_execution_directories(destination)
    protected: set[Path] = set()
    for name, (execution, marker) in terminal.items():
        _require_matching_execution(source, name, execution, marker)
        protected.add(Path("executions") / name)
    _remove_unprotected_destination(destination, protected)
    return protected


def _terminal_execution_directories(
    root: Path,
) -> dict[str, tuple[Path, str]]:
    executions = root / "executions"
    if not executions.exists():
        return {}
    if executions.is_symlink() or not executions.is_dir():
        raise WorkerError("partial execution publication is invalid")
    terminal: dict[str, tuple[Path, str]] = {}
    for execution in sorted(executions.iterdir()):
        if execution.is_symlink() or not execution.is_dir():
            continue
        markers = _terminal_markers(execution)
        if markers:
            _verify_terminal_execution(execution, markers)
            terminal[execution.name] = (execution, markers[0])
    return terminal


def _verify_terminal_execution(execution: Path, markers: list[str]) -> None:
    if len(markers) != 1:
        raise WorkerError("terminal execution evidence has conflicting markers")
    try:
        verify_checksums(execution)
    except RuntimeError as error:
        raise WorkerError(
            "terminal execution evidence failed checksum validation"
        ) from error


def _require_matching_execution(
    source: Path,
    name: str,
    observed: Path,
    marker: str,
) -> None:
    expected = source / "executions" / name
    if not expected.is_dir():
        raise WorkerError("terminal execution evidence cannot be overwritten")
    expected_markers = _terminal_markers(expected)
    _verify_terminal_execution(expected, expected_markers)
    if expected_markers != [marker] or not _trees_match(observed, expected):
        raise WorkerError("terminal execution evidence cannot be overwritten")


def _remove_unprotected_destination(destination: Path, protected: set[Path]) -> None:
    for path in sorted(destination.iterdir()):
        if path.name != "executions":
            _remove_path(path)
            continue
        for execution in sorted(path.iterdir()):
            relative = Path("executions") / execution.name
            if relative not in protected:
                _remove_path(execution)
        if not any(path.iterdir()):
            path.rmdir()


def _trees_match(first: Path, second: Path) -> bool:
    first_files = {
        path.relative_to(first): _file_digest(path)
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): _file_digest(path)
        for path in second.rglob("*")
        if path.is_file()
    }
    return first_files == second_files


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _terminal_markers(root: Path) -> list[str]:
    return [name for name in _TERMINAL_MARKERS if (root / name).is_file()]


def _publish_immutable_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise WorkerError(
                f"evidence path already has different contents: {destination}"
            )
        return
    temporary = destination.with_name(
        f".harbor-hf-{uuid.uuid4().hex}-{destination.name}.tmp"
    )
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
    if _terminal_markers(destination):
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
