from __future__ import annotations

import hashlib
import json
import math
import os
import re
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
from urllib.parse import urlparse

import httpx
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
    SecretValues,
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
from harbor_hf.harbor_adapter.models import HarborCompatibilityBundle
from harbor_hf.harbor_native_bundle import write_harbor_native_bundle
from harbor_hf.io import load_experiment
from harbor_hf.judge_recorder import (
    JUDGE_RECORDER_PORT,
    JudgeEvidenceRecorder,
    JudgeRecorderError,
)
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
from harbor_hf.provider_proxy import PROVIDER_RECORDER_PORT, ProviderEvidenceProxy
from harbor_hf.providers import routed_provider_model
from harbor_hf.runs import (
    RunLock,
    harbor_process_environment,
    require_benchmark_source_secret,
    run_secret_values,
)
from harbor_hf.trial_evidence import (
    TrialEvidenceError,
    assemble_trial_evidence,
    verify_trial_evidence,
)
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
    (
        "authentication",
        (
            "authentication",
            "unauthorized",
            "forbidden",
            "status=401",
            "status=403",
            "http 401",
            "http 403",
        ),
    ),
    ("rate-limit", ("ratelimit", "rate_limit", "status=429", "http 429")),
    ("quota", ("quota",)),
    (
        "transient",
        (
            "timeout",
            "connection",
            "serviceunavailable",
            "internalserver",
            "internal server error",
            "apierror",
            "status=500",
            "status=502",
            "status=503",
            "status=504",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        ),
    ),
    (
        "configuration",
        ("badrequest", "notfound", "configuration", "status=400", "status=404"),
    ),
)
_MAX_SANDBOX_RESULT_BYTES = 1024 * 1024
_MAX_HARBOR_LOG_CLASSIFICATION_BYTES = 1024 * 1024
_MISSING_PREBUILT_IMAGE_MARKER = "hf sandbox requires a prebuilt docker image"

_TERMINAL_MARKERS = ("_SUCCESS", "_FAILED", "_CANCELLED")
_HF_JOB_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_PROVIDER_RECORDER_READY_TIMEOUT_SECONDS = 120


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
    remote_job_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


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
    if any(run.configuration.trial_evidence is None for run in lock.runs):
        raise WorkerError("wave run locks require a complete trial evidence policy")
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
    try:
        for run in lock.runs:
            require_benchmark_source_secret(run.configuration)
    except ValueError as error:
        raise WorkerError(str(error)) from error
    os.environ["HF_TOKEN"] = token

    with _wave_worker_lease(lock, token, claim_store, clock):
        process_runner = runner or SubprocessRunner()
        destination = output_root / lock.artifact_prefix
        _reject_terminal_wave(destination)
        with tempfile.TemporaryDirectory(prefix="harbor-hf-wave-") as staging_name:
            staging = Path(staging_name) / campaign.artifact_prefix
            _stage_campaign_records(
                staging,
                campaign,
                lock,
                _wave_secret_values(lock, token),
                output_root,
            )
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
    judge_recorder: JudgeEvidenceRecorder | None = None
    judge_base_url: str | None = None
    shard_checksums: dict[str, str] = {}
    secrets = _wave_secret_values(lock, token)
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
        judge_base_url, judge_recorder = _prepare_judge_transport(
            lock, events, token, deadline, monotonic
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
            provider_proxy=provider_proxy,
            judge_recorder=judge_recorder,
            judge_base_url=judge_base_url,
        )
    except Exception as caught:
        error = caught
    finally:
        cleanup_error = _cleanup_wave_transport(
            lifecycle, provider_proxy, judge_recorder
        )

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
        _finalize_unit(wave_root, secrets)
        (wave_root / "_SUCCESS").write_text("\n", encoding="utf-8")
        _publish_unit(wave_root, output_root / lock.artifact_prefix)
        return output_root / lock.artifact_prefix

    failure_message = _redact_secret_values(str(terminal_error), secrets)
    summary["error_type"] = type(terminal_error).__name__
    summary["message"] = failure_message
    if error is not None and cleanup_error is not None:
        summary["cleanup_error"] = {
            "error_type": type(cleanup_error).__name__,
            "message": _redact_secret_values(str(cleanup_error), secrets),
        }
    append_event(events, "wave_failed", error_type=type(terminal_error).__name__)
    write_json(wave_root / "wave-summary.json", summary)
    _finalize_unit(wave_root, secrets)
    write_json(wave_root / "_FAILED", summary)
    _publish_unit(wave_root, output_root / lock.artifact_prefix)
    if error is not None and cleanup_error is not None:
        failure_message += "; endpoint cleanup failed: " + _redact_secret_values(
            str(cleanup_error), secrets
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
    base_url = _provider_recorder_base_url()
    try:
        proxy.start(host="0.0.0.0", port=PROVIDER_RECORDER_PORT)
        append_event(events, "provider_recorder_listening", port=PROVIDER_RECORDER_PORT)
        _wait_for_provider_recorder(base_url, token, deadline, monotonic)
    except Exception:
        proxy.close()
        raise
    append_event(
        events,
        "provider_recorder_ready",
        host=f"{os.environ['JOB_ID']}--{PROVIDER_RECORDER_PORT}.hf.jobs",
        port=PROVIDER_RECORDER_PORT,
    )
    return base_url, proxy


def _provider_recorder_base_url() -> str:
    return _job_ingress_base_url(PROVIDER_RECORDER_PORT, "provider recorder")


def _job_ingress_base_url(port: int, label: str) -> str:
    job_id = os.environ.get("JOB_ID", "")
    if not _HF_JOB_ID.fullmatch(job_id):
        raise WorkerError(f"{label} requires a valid HF JOB_ID")
    return f"https://{job_id}--{port}.hf.jobs"


def _prepare_judge_transport(
    lock: WaveLock,
    events: Path,
    token: str,
    deadline: float,
    monotonic: Callable[[], float],
) -> tuple[str | None, JudgeEvidenceRecorder | None]:
    if not any(run.configuration.judge_required_tasks for run in lock.runs):
        return None, None
    recorder = JudgeEvidenceRecorder(token=token)
    base_url = _job_ingress_base_url(JUDGE_RECORDER_PORT, "judge recorder")
    try:
        recorder.start(port=JUDGE_RECORDER_PORT)
        append_event(events, "judge_recorder_listening", port=JUDGE_RECORDER_PORT)
        _wait_for_provider_recorder(base_url, token, deadline, monotonic)
    except Exception:
        recorder.close()
        raise
    append_event(events, "judge_recorder_ready", port=JUDGE_RECORDER_PORT)
    return base_url, recorder


def _wait_for_provider_recorder(
    base_url: str,
    token: str,
    deadline: float,
    monotonic: Callable[[], float],
    *,
    sleep: Callable[[float], None] = time.sleep,
    client: httpx.Client | None = None,
) -> None:
    if not token:
        raise WorkerError("provider recorder readiness requires an HF token")
    ready_deadline = min(
        deadline, monotonic() + _PROVIDER_RECORDER_READY_TIMEOUT_SECONDS
    )
    owned_client = client is None
    selected_client = client or httpx.Client(follow_redirects=False)
    last_failure = "no response"
    try:
        while True:
            remaining = ready_deadline - monotonic()
            if remaining <= 0:
                raise WorkerError(
                    "provider recorder ingress readiness timed out: " + last_failure
                )
            try:
                response = selected_client.get(
                    f"{base_url}/healthz",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=min(5.0, remaining),
                )
            except httpx.TransportError as error:
                last_failure = type(error).__name__
            else:
                failure = _provider_recorder_readiness_failure(response)
                if failure is None:
                    return
                last_failure = failure
            sleep(min(1.0, max(0.0, ready_deadline - monotonic())))
    finally:
        if owned_client:
            selected_client.close()


def _provider_recorder_readiness_failure(response: httpx.Response) -> str | None:
    if response.status_code in {401, 403}:
        raise WorkerError("provider recorder ingress rejected HF authentication")
    try:
        healthy = response.json() == {"status": "ok"}
    except ValueError:
        healthy = False
    if response.status_code == 200 and healthy:
        return None
    return f"HTTP {response.status_code}"


def _cleanup_wave_transport(
    lifecycle: _EndpointWaveLifecycle | None,
    provider_proxy: ProviderEvidenceProxy | None,
    judge_recorder: JudgeEvidenceRecorder | None = None,
) -> Exception | None:
    errors: list[Exception] = []
    for closer in (provider_proxy, judge_recorder):
        if closer is None:
            continue
        try:
            closer.close()
        except Exception as error:
            errors.append(error)
    if lifecycle is not None:
        try:
            lifecycle_error = lifecycle.cleanup()
            if lifecycle_error is not None:
                errors.append(lifecycle_error)
        except Exception as error:
            errors.append(error)
    return errors[0] if errors else None


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
    *,
    provider_proxy: ProviderEvidenceProxy | None = None,
    judge_recorder: JudgeEvidenceRecorder | None = None,
    judge_base_url: str | None = None,
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
                provider_proxy=provider_proxy,
                judge_recorder=judge_recorder,
                judge_base_url=judge_base_url,
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
                    "kind": "hf-job-evidence-recorder",
                    "evidence_path": "provider-requests.jsonl",
                    "ingress_host": (
                        f"{os.environ.get('JOB_ID', '')}--"
                        f"{PROVIDER_RECORDER_PORT}.hf.jobs"
                    ),
                    "port": PROVIDER_RECORDER_PORT,
                    "route_authorization": "opaque-capability",
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
    provider_proxy: ProviderEvidenceProxy | None = None,
    judge_recorder: JudgeEvidenceRecorder | None = None,
    judge_base_url: str | None = None,
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
            provider_proxy=provider_proxy,
            judge_recorder=judge_recorder,
            judge_base_url=judge_base_url,
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
    *,
    provider_proxy: ProviderEvidenceProxy | None = None,
    judge_recorder: JudgeEvidenceRecorder | None = None,
    judge_base_url: str | None = None,
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
    pending: dict[
        Future[RetryCategory | None], tuple[int, CampaignTrialLock, Path]
    ] = {}
    deferred = False
    task_local_failures = False
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
                provider_proxy=provider_proxy,
                judge_recorder=judge_recorder,
                judge_base_url=judge_base_url,
            )
            pending[future] = (trial_index, trial, trial_root)
            continue
        trial_checksums[trial.trial_id] = _file_digest(trial_root / "checksums.json")
    for future in as_completed(pending):
        trial_index, trial, trial_root = pending[future]
        task_local_failures |= _record_trial_result(
            future,
            trial_index,
            trial,
            trial_root,
            events,
            failures,
            trial_checksums,
        )
    if failures:
        append_event(events, "shard_failed", failed_trials=len(failures))
        raise min(failures, key=lambda item: item[0])[1]
    if deferred or task_local_failures:
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
    _finalize_unit(shard_root, _wave_secret_values(wave, token))
    (shard_root / "_SUCCESS").write_text("\n", encoding="utf-8")
    _publish_unit(shard_root, output_root / locked_shard.artifact_prefix)
    return _file_digest(shard_root / "checksums.json")


def _record_trial_result(
    future: Future[RetryCategory | None],
    trial_index: int,
    trial: CampaignTrialLock,
    trial_root: Path,
    events: Path,
    failures: list[tuple[int, Exception]],
    trial_checksums: dict[str, str],
) -> bool:
    try:
        category = future.result()
    except Exception as error:
        failures.append((trial_index, error))
        append_event(
            events,
            "trial_failed",
            trial_id=trial.trial_id,
            error_type=type(error).__name__,
        )
        return False
    if category is not None:
        append_event(
            events,
            "trial_failed",
            trial_id=trial.trial_id,
            category=category,
        )
        return True
    append_event(events, "trial_completed", trial_id=trial.trial_id)
    trial_checksums[trial.trial_id] = _file_digest(trial_root / "checksums.json")
    return False


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
    *,
    provider_proxy: ProviderEvidenceProxy | None = None,
    judge_recorder: JudgeEvidenceRecorder | None = None,
    judge_base_url: str | None = None,
) -> RetryCategory | None:
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
        remote_job_id=os.environ.get("JOB_ID"),
    )
    write_json(
        execution_root / "execution.lock.json", execution.model_dump(mode="json")
    )
    shutil.copyfile(manifest_path, execution_root / "manifest.yaml")
    events = execution_root / "events.jsonl"
    append_event(events, "execution_started", execution_id=execution_id)
    error: Exception | None = None
    capability: str | None = None
    judge_capability: str | None = None
    judge_route_revoked = False
    judge_api_url: str | None = None
    failure_phase: Literal["configuration", "execution", "verification"] = (
        "configuration"
    )
    try:
        trial_base_url, capability = _register_trial_provider_route(
            wave,
            provider_proxy,
            base_url,
            execution_id,
            execution_root,
            events,
        )
        judge_api_url, judge_capability = _register_trial_judge_route(
            run,
            judge_recorder,
            judge_base_url,
            execution_id,
            trial.trial_id,
            trial.task_name,
            execution_root,
            events,
            tuple(value for value in (token, capability) if value is not None),
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
            extra_environment_hosts=_judge_environment_hosts(judge_api_url),
        )
        failure_phase = "execution"
        timeout = _remaining_seconds(deadline, monotonic)
        append_event(events, "harbor_started")
        blocked_secret_names = {
            candidate.configuration.benchmark_source.credentials.secret_name
            for candidate in wave.runs
            if candidate.configuration.benchmark_source is not None
            and candidate.configuration.benchmark_source.credentials is not None
        }
        with harbor_process_environment(
            run,
            token=token,
            inference_base_url=trial_base_url,
            judge_api_url=judge_api_url,
            blocked_secret_names=blocked_secret_names,
            redaction_secrets=tuple(
                value for value in (capability, judge_capability) if value is not None
            ),
        ) as environment:
            outcome = adapter.execute(
                prepared,
                harbor_source,
                jobs_dir,
                execution_root / "harbor.log",
                environment=environment,
                timeout_seconds=timeout,
                stream_runner=stream_runner,
                monotonic=monotonic,
                deadline=deadline,
            )
        append_event(events, "harbor_finished", exit_code=outcome.exit_code)
        if outcome.exit_code != 0:
            raise WorkerError(f"Harbor exited with status {outcome.exit_code}")
        _revoke_trial_judge_route(judge_recorder, judge_capability, events)
        judge_route_revoked = True
        failure_phase = "verification"
        if outcome.verification is None:
            raise WorkerError("Harbor produced no validated compatibility bundle")
        write_json(
            execution_root / "verification.json",
            outcome.verification.model_dump(mode="json"),
        )
        evidence_secrets = _execution_secret_values(
            wave, token, capability, judge_capability
        )
        _assemble_execution_trial_evidence(
            outcome.compatibility_path,
            jobs_dir,
            run,
            campaign,
            trial,
            execution,
            execution_root,
            (
                (evidence_secrets,)
                if isinstance(evidence_secrets, str)
                else tuple(evidence_secrets)
            ),
        )
        build_private_artifact_manifest(execution_root, strict_session=True)
        append_event(events, "execution_succeeded")
    except Exception as caught:
        error = caught
        append_event(events, "execution_failed", error_type=type(caught).__name__)
    finally:
        _revoke_trial_provider_route(provider_proxy, capability, events)
        error = _finish_trial_judge_route(
            judge_recorder,
            judge_capability,
            events,
            already_revoked=judge_route_revoked,
            existing_error=error,
        )
    failure_record: dict[str, object] | None = None
    failure_category: RetryCategory | None = None
    secrets = _execution_secret_values(wave, token, capability, judge_capability)
    if error is not None:
        failure_category = _execution_failure_category(
            error, failure_phase, evidence_root=execution_root
        )
        failure_record = {
            "category": failure_category,
            "error_type": type(error).__name__,
            "message": _redact_secret_values(str(error), secrets),
        }
        write_json(execution_root / "failure.json", failure_record)
    _finalize_execution(execution_root, secrets, strict_compatibility=error is None)
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
        if failure_category in {"agent", "benchmark"}:
            return failure_category
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
    _finalize_unit(trial_root, secrets)
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
    return None


def _finish_trial_judge_route(
    recorder: JudgeEvidenceRecorder | None,
    capability: str | None,
    events: Path,
    *,
    already_revoked: bool,
    existing_error: Exception | None,
) -> Exception | None:
    if already_revoked:
        return existing_error
    try:
        _revoke_trial_judge_route(recorder, capability, events)
    except Exception as revoke_error:
        append_event(
            events,
            "judge_route_revoke_failed",
            error_type=type(revoke_error).__name__,
        )
        return existing_error or revoke_error
    return existing_error


def _judge_environment_hosts(judge_api_url: str | None) -> tuple[str, ...]:
    if judge_api_url is None:
        return ()
    host = urlparse(judge_api_url).hostname
    if not host:
        raise WorkerError("judge recorder URL has no hostname")
    return (host,)


def _register_trial_judge_route(
    run: RunLock,
    recorder: JudgeEvidenceRecorder | None,
    base_url: str | None,
    execution_id: str,
    trial_id: str,
    task_name: str,
    execution_root: Path,
    events: Path,
    known_secrets: tuple[str, ...],
) -> tuple[str | None, str | None]:
    judge = run.benchmark_judge
    if judge is None or not run.judge_required_for(task_name):
        return None, None
    if recorder is None or base_url is None or run.trial_evidence is None:
        raise WorkerError("judge-required run has no exact evidence recorder")
    destination = execution_root / "judge-records"
    capability = recorder.register_scope(
        execution_id=execution_id,
        trial_id=trial_id,
        model=judge.model,
        destination=destination,
        policy=run.trial_evidence,
        known_secrets=known_secrets,
    )
    append_event(
        events,
        "judge_route_registered",
        capability_digest=recorder.capability_digest(capability),
    )
    return recorder.scoped_url(base_url, capability), capability


def _revoke_trial_judge_route(
    recorder: JudgeEvidenceRecorder | None,
    capability: str | None,
    events: Path,
) -> None:
    if recorder is None or capability is None:
        return
    digest = recorder.capability_digest(capability)
    recorder.revoke_scope(capability)
    append_event(events, "judge_route_revoked", capability_digest=digest)


def _assemble_execution_trial_evidence(
    compatibility_path: Path | None,
    jobs_dir: Path,
    run: RunLock,
    campaign: CampaignLock,
    trial: CampaignTrialLock,
    execution: ExecutionLock,
    execution_root: Path,
    known_secrets: tuple[str, ...],
) -> None:
    if compatibility_path is None or run.trial_evidence is None:
        raise WorkerError("validated trial evidence inputs are missing")
    bundle = HarborCompatibilityBundle.model_validate_json(
        compatibility_path.read_text(encoding="utf-8")
    )
    if len(bundle.trials) != 1:
        raise WorkerError("campaign physical execution must contain one Harbor trial")
    native = bundle.trials[0]
    if native.task_name != trial.task_name or native.task_digest != trial.task_digest:
        raise WorkerError("Harbor evidence trial identity does not match campaign lock")
    native_root = jobs_dir / native.path
    assemble_trial_evidence(
        native_root,
        campaign_id=campaign.campaign_id,
        run_id=run.run_id,
        execution_id=execution.execution_id,
        trial_id=trial.trial_id,
        task_name=trial.task_name,
        task_digest=trial.task_digest,
        logical_attempt=trial.logical_attempt,
        physical_attempt=execution.physical_attempt,
        judge_expected=run.judge_required_for(trial.task_name),
        judge_model=(run.benchmark_judge.model if run.benchmark_judge else None),
        policy=run.trial_evidence,
        judge_records_dir=execution_root / "judge-records",
        known_secrets=known_secrets,
    )
    append_event(
        execution_root / "events.jsonl",
        "trial_evidence_validated",
        trial_id=trial.trial_id,
    )


def _register_trial_provider_route(
    wave: WaveLock,
    provider_proxy: ProviderEvidenceProxy | None,
    base_url: str,
    execution_id: str,
    execution_root: Path,
    events: Path,
) -> tuple[str, str | None]:
    if not isinstance(wave.target, ProviderWaveTarget):
        return base_url, None
    if provider_proxy is None:
        raise WorkerError("provider execution requires an evidence recorder")
    capability = provider_proxy.register_scope(execution_id)
    capability_digest = provider_proxy.capability_digest(capability)
    write_json(
        execution_root / "provider-route.json",
        {
            "capability_digest": capability_digest,
            "transport": "hf-job-evidence-recorder",
        },
    )
    append_event(
        events,
        "provider_route_registered",
        capability_digest=capability_digest,
    )
    return provider_proxy.scoped_base_url(base_url, capability), capability


def _revoke_trial_provider_route(
    provider_proxy: ProviderEvidenceProxy | None,
    capability: str | None,
    events: Path,
) -> None:
    if capability is None or provider_proxy is None:
        return
    capability_digest = provider_proxy.capability_digest(capability)
    provider_proxy.revoke_scope(capability)
    append_event(
        events,
        "provider_route_revoked",
        capability_digest=capability_digest,
    )


def _execution_failure_category(
    error: Exception,
    phase: Literal["configuration", "execution", "verification"],
    *,
    evidence_root: Path | None = None,
) -> RetryCategory:
    if isinstance(error, (JudgeRecorderError, TrialEvidenceError)):
        return "evidence"
    if isinstance(error, PrivateArtifactRequirementError):
        return "evidence"
    if isinstance(error, HarborVerificationFailure):
        return "benchmark"
    if isinstance(error, HarborTrialFailure):
        return _harbor_trial_failure_category(error, evidence_root)
    sandbox_category = _sandbox_failure_category(evidence_root)
    if sandbox_category is not None:
        return sandbox_category
    if phase == "configuration":
        return "configuration"
    if phase == "execution":
        return "transient"
    return "benchmark"


def _harbor_trial_failure_category(
    error: HarborTrialFailure, evidence_root: Path | None
) -> RetryCategory:
    exception_type = error.exception_type.lower()
    category = _retry_category_from_text(exception_type)
    if category is not None:
        return category
    if exception_type == "sandboxerror":
        if error.exception_message is not None:
            category = _sandbox_exception_line_category(error.exception_message.lower())
            if category is not None:
                return category
        return _sandbox_failure_category(evidence_root) or "benchmark"
    if exception_type == "nonzeroagentexitcodeerror":
        return _openclaw_transport_failure_category(evidence_root) or "agent"
    if "agent" in exception_type:
        return "agent"
    return "benchmark"


def _sandbox_failure_category(evidence_root: Path | None) -> RetryCategory | None:
    if evidence_root is None:
        return None
    resolved_root = evidence_root.resolve()
    saw_sandbox_error = False
    for result_path in sorted(evidence_root.glob("harbor-jobs/*/*/result.json")):
        exception = _sandbox_result_exception(result_path, resolved_root)
        if exception is None or exception[0].lower() != "sandboxerror":
            continue
        saw_sandbox_error = True
        if exception[1] is not None:
            category = _sandbox_exception_line_category(exception[1].lower())
            if category is not None:
                return category
    if saw_sandbox_error:
        return "benchmark"
    return _harbor_preflight_failure_category(evidence_root, resolved_root)


def _harbor_preflight_failure_category(
    evidence_root: Path, resolved_root: Path
) -> RetryCategory | None:
    log_path = evidence_root / "harbor.log"
    if not _safe_evidence_file(log_path, resolved_root):
        return None
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - _MAX_HARBOR_LOG_CLASSIFICATION_BYTES))
            tail = stream.read(_MAX_HARBOR_LOG_CLASSIFICATION_BYTES)
    except OSError:
        return None
    if _MISSING_PREBUILT_IMAGE_MARKER in tail.decode("utf-8", errors="replace").lower():
        return "benchmark"
    return None


def _sandbox_result_exception(
    result_path: Path, resolved_root: Path
) -> tuple[str, str | None] | None:
    if not _safe_evidence_file(result_path, resolved_root):
        return None
    try:
        with result_path.open("rb") as stream:
            raw = stream.read(_MAX_SANDBOX_RESULT_BYTES + 1)
        if len(raw) > _MAX_SANDBOX_RESULT_BYTES:
            return None
        result = json.loads(raw)
    except (OSError, UnicodeError, ValueError, RecursionError):
        return None
    if not isinstance(result, dict):
        return None
    exception = result.get("exception_info")
    if not isinstance(exception, dict):
        return None
    exception_type = exception.get("exception_type")
    message = exception.get("exception_message")
    if not isinstance(exception_type, str) or not isinstance(message, str | None):
        return None
    return exception_type, message


def _sandbox_exception_line_category(value: str) -> RetryCategory | None:
    if "did not become ready within" in value:
        return "transient"
    if "sandbox api error (429)" in value:
        return "rate-limit"
    if any(f"sandbox api error ({status})" in value for status in (500, 502, 503, 504)):
        return "transient"
    return None


def _retry_category_from_text(value: str) -> RetryCategory | None:
    for category, markers in _TRIAL_FAILURE_MARKERS:
        if any(marker in value for marker in markers):
            return category
    return None


def _openclaw_transport_failure_category(
    evidence_root: Path | None,
) -> RetryCategory | None:
    if evidence_root is None:
        return None
    resolved_root = evidence_root.resolve()
    for log_path in sorted(evidence_root.glob("harbor-jobs/*/*/agent/openclaw.txt")):
        if not _safe_evidence_file(log_path, resolved_root):
            continue
        category = _openclaw_log_failure_category(log_path)
        if category is not None:
            return category
    return None


def _safe_evidence_file(path: Path, resolved_root: Path) -> bool:
    return (
        not path.is_symlink()
        and path.is_file()
        and path.resolve().is_relative_to(resolved_root)
    )


def _openclaw_log_failure_category(log_path: Path) -> RetryCategory | None:
    try:
        with log_path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                stripped = line.strip()
                if stripped.startswith(
                    "[provider-transport-fetch] [model-fetch] response "
                ) or stripped.startswith("FailoverError: HTTP "):
                    category = _retry_category_from_text(stripped.lower())
                    if category is not None:
                        return category
                if stripped == "FailoverError: LLM request timed out.":
                    return "transient"
    except OSError:
        return None
    return None


def _wave_model_name(wave: WaveLock) -> str:
    if isinstance(wave.target, EndpointWaveTarget):
        return wave.target.endpoint.served_model_name
    return routed_provider_model(wave.target.provider)


def _stage_campaign_records(
    campaign_root: Path,
    campaign: CampaignLock,
    wave: WaveLock,
    secrets: SecretValues,
    output_root: Path,
) -> None:
    campaign_root.mkdir(parents=True)
    campaign_lock_path = campaign_root / "campaign.lock.json"
    write_json(campaign_lock_path, campaign.model_dump(mode="json"))
    assert_secret_absent(campaign_root, secrets)
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
        assert_secret_absent(run_root, secrets)
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
        _validate_recovered_trial_evidence(execution, execution_lock, expected)
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


def _validate_recovered_trial_evidence(
    execution_root: Path,
    execution: ExecutionLock,
    expected: CampaignTrialLock,
) -> None:
    manifests = list(execution_root.glob("harbor-jobs/*/*/evidence/manifest.json"))
    if len(manifests) != 1:
        raise WorkerError("terminal execution has no unique trial evidence manifest")
    try:
        evidence = verify_trial_evidence(manifests[0].parent.parent, deep=True)
    except TrialEvidenceError as error:
        raise WorkerError("terminal trial evidence is incomplete") from error
    observed = (
        evidence.execution_id,
        evidence.trial_id,
        evidence.task_name,
        evidence.task_digest,
        evidence.logical_attempt,
        evidence.physical_attempt,
    )
    locked = (
        execution.execution_id,
        expected.trial_id,
        expected.task_name,
        expected.task_digest,
        expected.logical_attempt,
        execution.physical_attempt,
    )
    if observed != locked:
        raise WorkerError("terminal trial evidence identity does not match its lock")


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
    root: Path, secrets: SecretValues, *, strict_compatibility: bool = True
) -> None:
    attempted = openclaw_execution_was_attempted(root)
    rejection_count = 0
    if not strict_compatibility:
        rejection_count = len(
            sanitize_private_artifact_tree(
                root,
                required_directories=("harbor-jobs",),
            )
        )
        (root / "harbor-jobs").mkdir(exist_ok=True)
    session_required = openclaw_execution_started(root, fallback_attempted=attempted)
    _redact_unit(root, secrets)
    refresh_error = refresh_retained_bundle(root, strict=strict_compatibility)
    if refresh_error is not None:
        append_event(
            root / "events.jsonl",
            "compatibility_refresh_skipped",
            error_type=refresh_error,
        )
    if not strict_compatibility:
        final_rejection_count = len(
            sanitize_private_artifact_tree(
                root,
                trust_existing_rejections=True,
                required_directories=("harbor-jobs",),
            )
        )
        (root / "harbor-jobs").mkdir(exist_ok=True)
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
    assert_secret_absent(root, secrets)
    write_harbor_native_bundle(root, required=strict_compatibility)
    write_checksums(root)


def _finalize_unit(root: Path, secrets: SecretValues) -> None:
    _redact_unit(root, secrets)
    write_checksums(root)


def _redact_unit(root: Path, secrets: SecretValues) -> None:
    redacted_paths = scrub_secret_paths(root, secrets)
    if redacted_paths:
        append_event(
            root / "events.jsonl", "secret_paths_redacted", count=redacted_paths
        )
    scrubbed = scrub_secret(root, secrets)
    if scrubbed:
        append_event(root / "events.jsonl", "secrets_redacted", files=scrubbed)
    assert_secret_absent(root, secrets)


def _wave_secret_values(lock: WaveLock, token: str) -> SecretValues:
    values = [token]
    for run in lock.runs:
        run_values = run_secret_values(run.configuration, token)
        values.extend((run_values,) if isinstance(run_values, str) else run_values)
    unique = tuple(dict.fromkeys(value for value in values if value))
    return unique[0] if len(unique) == 1 else unique


def _execution_secret_values(
    wave: WaveLock,
    token: str,
    provider_capability: str | None,
    judge_capability: str | None,
) -> SecretValues:
    return _secret_values_with(
        _secret_values_with(_wave_secret_values(wave, token), provider_capability),
        judge_capability,
    )


def _secret_values_with(secrets: SecretValues, value: str | None) -> SecretValues:
    values = [secrets] if isinstance(secrets, str) else list(secrets)
    if value:
        values.append(value)
    unique = tuple(dict.fromkeys(candidate for candidate in values if candidate))
    return unique[0] if len(unique) == 1 else unique


def _redact_secret_values(value: str, secrets: SecretValues) -> str:
    values = (secrets,) if isinstance(secrets, str) else tuple(secrets)
    for secret in values:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value


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
