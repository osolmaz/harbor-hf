from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, Self, cast

import httpx
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from harbor_hf.campaign_finalizer import (
    BucketCampaignFinalizer,
    CampaignFinalizationError,
    CampaignFinalizer,
)
from harbor_hf.campaign_observer import BucketCampaignObserver, CampaignObserver
from harbor_hf.campaigns import (
    CampaignLock,
    WaveLock,
    build_campaign_lock,
    build_campaign_plan,
    build_wave_lock,
    managed_wave_endpoint,
)
from harbor_hf.control import (
    ActionKind,
    ActionOutcomePayload,
    ActionProjection,
    ActionReservedPayload,
    CampaignCancellationWon,
    CampaignConflict,
    CampaignEvent,
    CampaignStore,
    Clock,
    EventKind,
    ExecutionOutcomePayload,
    IdentifierFactory,
    LifecyclePayload,
    TerminalPayload,
    WaveLifecyclePayload,
    new_event,
    same_event_request,
)
from harbor_hf.coordination import (
    ClaimConflict,
    ClaimStore,
    CoordinationError,
    action_claim_path,
)
from harbor_hf.endpoints import (
    AmbiguousEndpointCreate,
    AmbiguousEndpointDelete,
    AmbiguousEndpointPause,
    DesiredEndpoint,
    EndpointProvisioner,
    EndpointProvisioningError,
    EndpointSnapshot,
    EndpointVerificationTimeout,
    ProvisioningResult,
    build_desired_endpoint,
)
from harbor_hf.hf_endpoints import HuggingFaceEndpointAdapter
from harbor_hf.io import ManifestError, load_experiment_bytes
from harbor_hf.models import (
    DeploymentProfile,
    DeploymentTarget,
    ExperimentSpec,
    ModelProfile,
)
from harbor_hf.process import ProcessError, SubprocessRunner
from harbor_hf.provider_models import ProviderTarget
from harbor_hf.reconciler import (
    AdmissionUsage,
    ReconcileAction,
    ReconcileContext,
    ReconcilePlan,
    plan_reconciliation,
)
from harbor_hf.recovery import RecoveryProjection, TerminalDecision, project_recovery
from harbor_hf.submission import (
    BucketApi,
    TextRunner,
    endpoint_lease_label_for,
    submit_wave,
)

_TERMINAL_JOB_STAGES = {"CANCELED", "COMPLETED", "DELETED", "ERROR"}
_ACTION_PRIORITY = {
    "cancel-execution": 0,
    "cancel-wave": 1,
    "drain-wave": 2,
    "cleanup-wave": 3,
    "exhaust-trials": 4,
    "manual-intervention": 5,
    "publish-summary": 6,
    "publish-results": 7,
    "retry-shard": 8,
    "submit-wave": 9,
}
_OUTCOME_KINDS: dict[
    Literal["succeeded", "failed", "ambiguous"],
    Literal["action.succeeded", "action.failed", "action.ambiguous"],
] = {
    "succeeded": "action.succeeded",
    "failed": "action.failed",
    "ambiguous": "action.ambiguous",
}
_ACTION_LEASE_DURATION = timedelta(hours=2)


class CampaignApplyError(RuntimeError):
    """Raised when a campaign pass cannot be applied safely."""


class ActionExecutionError(CampaignApplyError):
    """Raised when a remote action is known not to have succeeded."""


class AmbiguousActionOutcome(CampaignApplyError):
    """Raised when a remote action may have succeeded."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RemoteWaveJob(FrozenModel):
    job_id: str
    wave_id: str
    endpoint_label: str
    stage: str
    target_label_key: Literal["harbor-hf-endpoint", "harbor-hf-provider"] = (
        "harbor-hf-endpoint"
    )

    @property
    def terminal(self) -> bool:
        return self.stage in _TERMINAL_JOB_STAGES


class AppliedAction(FrozenModel):
    action_id: str
    kind: ActionKind
    status: Literal["succeeded", "failed", "ambiguous"]
    remote_id: str | None = None
    message: str | None = None


class CampaignApplyResult(FrozenModel):
    campaign_id: str
    plan: ReconcilePlan
    applied: list[AppliedAction]


class CampaignApplyFailure(FrozenModel):
    campaign_id: str
    error_type: str
    message: str


class EndpointApplicationPort(Protocol):
    def inspect(self, desired: DesiredEndpoint) -> EndpointSnapshot | None: ...

    def create_or_adopt(self, desired: DesiredEndpoint) -> ProvisioningResult: ...

    def pause_and_verify(self, desired: DesiredEndpoint) -> EndpointSnapshot: ...


class WaveJobPort(Protocol):
    def find_wave(
        self,
        *,
        namespace: str,
        wave_id: str,
        endpoint_label: str,
        target_label_key: Literal[
            "harbor-hf-endpoint", "harbor-hf-provider"
        ] = "harbor-hf-endpoint",
    ) -> RemoteWaveJob | None: ...

    def submit(
        self,
        lock: WaveLock,
        *,
        request: bytes,
        campaign: CampaignLock,
    ) -> RemoteWaveJob: ...

    def cancel(self, job: RemoteWaveJob, *, namespace: str) -> None: ...


class CampaignPublicationPort(Protocol):
    def publish(self, campaign_id: str) -> object: ...


class HfJobsApi(Protocol):
    def list_jobs(self, **kwargs: object) -> Iterable[object]: ...

    def cancel_job(self, **kwargs: object) -> None: ...


class HuggingFaceWaveJobAdapter:
    """Narrow HF Jobs adapter with deterministic label-based adoption."""

    def __init__(
        self,
        *,
        api: HfJobsApi,
        runner: TextRunner,
        bucket_api: BucketApi,
    ) -> None:
        self.api = api
        self.runner = runner
        self.bucket_api = bucket_api

    def find_wave(
        self,
        *,
        namespace: str,
        wave_id: str,
        endpoint_label: str,
        target_label_key: Literal[
            "harbor-hf-endpoint", "harbor-hf-provider"
        ] = "harbor-hf-endpoint",
    ) -> RemoteWaveJob | None:
        expected = {
            "harbor-hf-wave": wave_id,
            target_label_key: endpoint_label,
        }
        try:
            resources = self.api.list_jobs(labels=expected, namespace=namespace)
            jobs = [_validated_remote_job(value, expected) for value in resources]
        except (HfHubHTTPError, httpx.TransportError) as error:
            raise AmbiguousActionOutcome("HF Jobs inspection failed") from error
        if len(jobs) > 1:
            active = [job for job in jobs if not job.terminal]
            if len(active) == 1:
                return active[0]
            completed = [job for job in jobs if job.stage == "COMPLETED"]
            if not active and len(completed) == 1:
                return completed[0]
            raise ActionExecutionError(
                f"multiple HF Jobs have the managed wave identity: {wave_id}"
            )
        return jobs[0] if jobs else None

    def submit(
        self,
        lock: WaveLock,
        *,
        request: bytes,
        campaign: CampaignLock,
    ) -> RemoteWaveJob:
        try:
            with tempfile.TemporaryDirectory(prefix="harbor-hf-wave-") as name:
                staging = Path(name)
                (staging / "manifest.yaml").write_bytes(request)
                _write_model(staging / "campaign.lock.json", campaign)
                _write_model(staging / "wave.lock.json", lock)
                submission = submit_wave(
                    lock,
                    input_dir=staging,
                    bucket=lock.artifact_bucket,
                    runner=self.runner,
                    bucket_api=self.bucket_api,
                )
        except ProcessError as error:
            raise AmbiguousActionOutcome(
                "HF Jobs submission ended without a definitive outcome"
            ) from error
        except ValueError as error:
            if "did not return a job ID" in str(error):
                raise AmbiguousActionOutcome(str(error)) from error
            raise ActionExecutionError(str(error)) from error
        except HfHubHTTPError as error:
            raise ActionExecutionError("HF Jobs submission preflight failed") from error
        if submission.job_id is None:
            raise AmbiguousActionOutcome("HF Jobs submission returned no job ID")
        endpoint = lock.endpoint
        provider = lock.provider_target
        if endpoint is not None:
            deployment_label = endpoint_lease_label_for(
                endpoint.namespace, endpoint.name
            )
        elif provider is not None:
            deployment_label = hashlib.sha256(provider.service.encode()).hexdigest()[
                :32
            ]
        else:
            raise ActionExecutionError("wave lock has no deployment target")
        return RemoteWaveJob(
            job_id=submission.job_id,
            wave_id=lock.wave_id,
            endpoint_label=deployment_label,
            target_label_key=(
                "harbor-hf-endpoint" if endpoint is not None else "harbor-hf-provider"
            ),
            stage="SCHEDULING",
        )

    def cancel(self, job: RemoteWaveJob, *, namespace: str) -> None:
        if job.terminal:
            return
        try:
            self.api.cancel_job(job_id=job.job_id, namespace=namespace)
        except HfHubHTTPError as error:
            status = error.response.status_code
            if status == 404:
                return
            if status == 409 or status >= 500:
                raise AmbiguousActionOutcome(
                    f"HF Job cancellation outcome is ambiguous: HTTP {status}"
                ) from error
            raise ActionExecutionError(
                f"HF Job cancellation failed: HTTP {status}"
            ) from error
        except httpx.TransportError as error:
            raise AmbiguousActionOutcome(
                "HF Job cancellation outcome is ambiguous before a response"
            ) from error


def _missing_observed_events(
    durable_events: Sequence[CampaignEvent], observed_events: Sequence[CampaignEvent]
) -> list[CampaignEvent]:
    durable_by_id = {event.event_id: event for event in durable_events}
    missing: list[CampaignEvent] = []
    for event in observed_events:
        durable = durable_by_id.get(event.event_id)
        if durable is None:
            missing.append(event)
        elif not same_event_request(durable, event):
            raise CampaignConflict(f"event conflicts: {event.event_id}")
    return missing


class CampaignReconciler:
    """Apply one bounded, stateless campaign reconciliation pass."""

    def __init__(
        self,
        store: CampaignStore,
        *,
        endpoints: EndpointApplicationPort,
        jobs: WaveJobPort,
        action_claims: ClaimStore,
        observer: CampaignObserver | None = None,
        finalizer: CampaignFinalizer | None = None,
        result_publisher: CampaignPublicationPort | None = None,
        cleanup: Callable[[], None] = lambda: None,
        clock: Clock = lambda: datetime.now(UTC),
        identifier: IdentifierFactory = lambda: uuid.uuid4().hex,
    ) -> None:
        self.store = store
        self.endpoints = endpoints
        self.jobs = jobs
        self.action_claims = action_claims
        self.observer = observer
        self.finalizer = finalizer
        self.result_publisher = result_publisher
        self._cleanup = cleanup
        self.clock = clock
        self.identifier = identifier
        self._last_observed_at: datetime | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        cleanup, self._cleanup = self._cleanup, lambda: None
        cleanup()

    def apply_campaign(
        self,
        campaign_id: str,
        *,
        context: ReconcileContext | None = None,
    ) -> CampaignApplyResult:
        lock, events = self.store.load_campaign(campaign_id)
        request = self.store.load_request(campaign_id)
        spec = _validated_request(lock, request)
        terminal_recorded = any(
            event.kind
            in {
                "campaign.completed",
                "campaign.partial",
                "campaign.failed",
                "campaign.cancelled",
            }
            for event in events
        )
        if self.observer is not None and not terminal_recorded:
            missing = _missing_observed_events(
                events, self.observer.observe(lock, spec)
            )
            changed = (
                self.store.ensure_events(campaign_id, missing) if missing else False
            )
            if changed:
                lock, events = self.store.load_campaign(campaign_id)
        self._last_observed_at = max(event.observed_at for event in events)
        reservations = _validated_reservations(
            self.store.load_action_reservations(campaign_id),
            campaign_id=campaign_id,
        )
        effective_context = _context_with_unobserved_actions(
            lock, events, reservations, context
        )
        projection, plan = plan_reconciliation(
            lock,
            events,
            context=effective_context,
            now=self.clock(),
        )
        if projection.campaign.status in {
            "completed",
            "partial",
            "failed",
            "cancelled",
        }:
            return CampaignApplyResult(campaign_id=campaign_id, plan=plan, applied=[])
        if self._recover_terminal_jobs(lock, spec, projection, reservations):
            lock, events = self.store.load_campaign(campaign_id)
            self._last_observed_at = max(event.observed_at for event in events)
            effective_context = _context_with_unobserved_actions(
                lock, events, reservations, context
            )
            projection, plan = plan_reconciliation(
                lock,
                events,
                context=effective_context,
                now=self.clock(),
            )
        if (
            plan.terminal_decision is not None
            and projection.campaign.status
            not in {"completed", "partial", "failed", "cancelled"}
            and _summary_action_succeeded(projection)
        ):
            self._record_terminal(lock, plan.terminal_decision)
            lock, events = self.store.load_campaign(campaign_id)
            effective_context = _context_with_unobserved_actions(
                lock, events, reservations, context
            )
            projection, plan = plan_reconciliation(
                lock,
                events,
                context=effective_context,
                now=self.clock(),
            )
            return CampaignApplyResult(campaign_id=campaign_id, plan=plan, applied=[])
        pending = _pending_actions(projection.campaign.actions, reservations)
        selected = sorted(
            [*pending, *self._reserve_new(lock, plan.actions)],
            key=lambda action: (_ACTION_PRIORITY[action.kind], action.action_id),
        )
        limit = effective_context.limits.action_limit
        allow_billable = projection.campaign.status in {"queued", "active"}
        applied = self._apply_selected(
            lock,
            spec,
            request,
            selected[:limit],
            projection=projection,
            terminal_decision=plan.terminal_decision,
            allow_billable=allow_billable,
        )
        return CampaignApplyResult(
            campaign_id=campaign_id,
            plan=plan,
            applied=applied,
        )

    def _apply_selected(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        request: bytes,
        selected: list[ReconcileAction],
        *,
        projection: RecoveryProjection,
        terminal_decision: TerminalDecision | None,
        allow_billable: bool,
    ) -> list[AppliedAction]:
        applied: list[AppliedAction] = []
        for action in selected:
            owner = self._action_claim_owner(lock, action)
            path = action_claim_path(lock.campaign_id, action.action_id)
            try:
                self.action_claims.acquire(path, owner)
            except ClaimConflict:
                continue
            try:
                outcome = self._apply_action(
                    lock,
                    spec,
                    request,
                    action,
                    projection=projection,
                    terminal_decision=terminal_decision,
                    allow_billable=allow_billable,
                )
            finally:
                # A durable action outcome makes a stale lease harmless; its
                # expiry remains the crash-recovery path if release is lost.
                with suppress(CoordinationError):
                    self.action_claims.release(path, owner)
            applied.append(outcome)
            if outcome.status == "ambiguous" or (
                action.kind == "publish-summary" and outcome.status == "succeeded"
            ):
                break
        return applied

    def _action_claim_owner(
        self, lock: CampaignLock, action: ReconcileAction
    ) -> dict[str, str]:
        now = self.clock().astimezone(UTC)
        return {
            "campaign_id": lock.campaign_id,
            "action_id": action.action_id,
            "reconciler_id": self.identifier(),
            "expires_at": (now + _ACTION_LEASE_DURATION).isoformat(),
        }

    def apply_all(
        self,
        *,
        context: ReconcileContext | None = None,
        campaign_ids: Sequence[str] | None = None,
    ) -> list[CampaignApplyResult | CampaignApplyFailure]:
        results: list[CampaignApplyResult | CampaignApplyFailure] = []
        selected_campaign_ids = (
            list(dict.fromkeys(campaign_ids))
            if campaign_ids is not None
            else self.store.list_campaigns()
        )
        observed: dict[str, AdmissionUsage] = {}
        admission_unknown = False
        for campaign_id in selected_campaign_ids:
            try:
                observed[campaign_id] = self._admission_usage(campaign_id)
            except Exception:
                admission_unknown = True
                observed[campaign_id] = AdmissionUsage()
        baseline = context or ReconcileContext()
        if admission_unknown:
            baseline = baseline.model_copy(
                update={
                    "usage": baseline.usage.model_copy(
                        update={
                            "global_active_waves": max(
                                baseline.usage.global_active_waves,
                                baseline.limits.global_active_waves,
                            )
                        }
                    )
                }
            )
        for campaign_id in selected_campaign_ids:
            try:
                campaign_context = baseline.model_copy(
                    update={
                        "usage": _combined_usage(
                            baseline.usage,
                            (
                                contribution
                                for owner, contribution in observed.items()
                                if owner != campaign_id
                            ),
                        )
                    }
                )
                result = self.apply_campaign(campaign_id, context=campaign_context)
                results.append(result)
                observed[campaign_id] = self._admission_usage(campaign_id)
            except Exception as error:
                results.append(
                    CampaignApplyFailure(
                        campaign_id=campaign_id,
                        error_type=type(error).__name__,
                        message=str(error),
                    )
                )
        return results

    def _admission_usage(self, campaign_id: str) -> AdmissionUsage:
        lock, events = self.store.load_campaign(campaign_id)
        projection = project_recovery(lock, events)
        reservations = _validated_reservations(
            self.store.load_action_reservations(campaign_id),
            campaign_id=campaign_id,
        )
        return _build_admission_usage(campaign_id, projection, reservations)

    def _recover_terminal_jobs(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        projection: RecoveryProjection,
        reservations: Mapping[str, ReconcileAction],
    ) -> bool:
        changed = False
        for action_id, action_state in sorted(projection.campaign.actions.items()):
            if action_state.action_kind not in {"submit-wave", "retry-shard"}:
                continue
            action = reservations.get(action_id)
            if action is None:
                raise CampaignApplyError(
                    f"action event has no reservation record: {action_id}"
                )
            wave_id = action.wave_id or f"wave-{action.action_key}"
            wave = projection.waves.get(wave_id)
            if wave is not None and wave.status == "closed":
                continue
            job = self._managed_wave_job(lock, spec, action)
            if job is None or not job.terminal:
                continue
            changed = (
                self._recover_lost_executions(lock, projection, wave_id, job) or changed
            )
            if _terminal_wave_needs_drain(wave):
                self._drain_wave(lock, action, projection)
                changed = True
        return changed

    def _recover_lost_executions(
        self,
        lock: CampaignLock,
        projection: RecoveryProjection,
        wave_id: str,
        job: RemoteWaveJob,
    ) -> bool:
        changed = False
        for execution in projection.executions.values():
            if execution.wave_id != wave_id or execution.status != "active":
                continue
            self._record_durable_event(
                lock,
                subject_type="execution",
                subject_id=execution.execution_id,
                kind="execution.failed",
                payload=ExecutionOutcomePayload(
                    trial_id=execution.trial_id,
                    physical_attempt=execution.physical_attempt,
                    category="lost",
                    message=(
                        f"HF Job {job.job_id} reached {job.stage} without "
                        "terminal execution evidence"
                    ),
                ),
                identity=f"{execution.execution_id}:lost:{job.job_id}",
            )
            changed = True
        return changed

    def _reserve_new(
        self,
        lock: CampaignLock,
        actions: list[ReconcileAction],
    ) -> list[ReconcileAction]:
        reserved: list[ReconcileAction] = []
        for action in actions:
            event = self._event(
                lock,
                "action.reserved",
                ActionReservedPayload(
                    action_id=action.action_id,
                    action_key=action.action_key,
                    action_kind=action.kind,
                    target_ids=action.target_ids,
                ),
            )
            record = cast(Mapping[str, JsonValue], action.model_dump(mode="json"))
            if self.store.reserve_action(lock.campaign_id, record, event):
                reserved.append(action)
        return reserved

    def _apply_action(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        request: bytes,
        action: ReconcileAction,
        *,
        projection: RecoveryProjection,
        terminal_decision: TerminalDecision | None,
        allow_billable: bool,
    ) -> AppliedAction:
        try:
            remote_id = self._execute(
                lock,
                spec,
                request,
                action,
                projection=projection,
                terminal_decision=terminal_decision,
                allow_billable=allow_billable,
            )
        except AmbiguousActionOutcome as error:
            return self._record_outcome(lock, action, "ambiguous", str(error))
        except _AMBIGUOUS_ENDPOINT_ERRORS as error:
            remote_id = None
            if action.kind in {"submit-wave", "retry-shard"}:
                target = _deployment_target(lock, spec, action.deployment_digest)
                if not isinstance(target, ProviderTarget):
                    remote_id = _desired_endpoint(lock, spec, action).identity.name
            return self._record_outcome(
                lock,
                action,
                "ambiguous",
                str(error),
                remote_id=remote_id,
            )
        except (
            ActionExecutionError,
            EndpointProvisioningError,
            ValueError,
        ) as error:
            return self._record_outcome(
                lock,
                action,
                "failed",
                str(error),
                remote_id=_failed_endpoint_remote_id(
                    lock, spec, action, projection, error
                ),
            )
        outcome = self._record_outcome(
            lock,
            action,
            "succeeded",
            None,
            remote_id=remote_id,
        )
        if action.kind == "publish-summary":
            if terminal_decision is None:
                raise CampaignApplyError("summary action has no terminal decision")
            self._record_terminal(lock, terminal_decision)
        return outcome

    def _execute(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        request: bytes,
        action: ReconcileAction,
        *,
        projection: RecoveryProjection,
        terminal_decision: TerminalDecision | None,
        allow_billable: bool,
    ) -> str | None:
        if action.kind in {"submit-wave", "retry-shard"}:
            return self._submit_wave(
                lock,
                spec,
                request,
                action,
                projection=projection,
                allow_submission=allow_billable,
            )
        if action.kind == "exhaust-trials":
            return self._exhaust_trials(lock, action, projection)
        return self._execute_lifecycle_action(
            lock,
            spec,
            action,
            projection=projection,
            terminal_decision=terminal_decision,
        )

    def _execute_lifecycle_action(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        action: ReconcileAction,
        *,
        projection: RecoveryProjection,
        terminal_decision: TerminalDecision | None,
    ) -> str | None:
        if action.kind == "cancel-wave":
            return self._cancel_wave(lock, spec, action)
        if action.kind == "cancel-execution":
            return self._cancel_execution(lock, spec, action, projection)
        if action.kind == "drain-wave":
            return self._drain_wave(lock, action, projection)
        if action.kind == "cleanup-wave":
            return self._cleanup_wave(lock, spec, action, projection=projection)
        if action.kind == "manual-intervention":
            return self._manual_intervention(lock, action)
        if action.kind == "publish-results":
            return self._publish_results(lock)
        if action.kind == "publish-summary":
            return self._publish_summary(lock, spec, projection, terminal_decision)
        raise ActionExecutionError(
            f"action execution is not supported by configured adapters: {action.kind}"
        )

    def _exhaust_trials(
        self,
        lock: CampaignLock,
        action: ReconcileAction,
        projection: RecoveryProjection,
    ) -> str:
        if projection.cancel_requested_at is not None:
            raise ActionExecutionError(
                "campaign cancellation superseded retry exhaustion"
            )
        _validate_spend_exhaustion_targets(lock, action)
        for trial_id in action.trial_ids:
            trial = projection.trials.get(trial_id)
            if trial is None or trial.status not in {
                "retry_wait",
                "invalid",
                "failed_infrastructure",
            }:
                raise ActionExecutionError(
                    f"spend exhaustion target is not retryable: {trial_id}"
                )
        events: list[CampaignEvent] = []
        for trial_id in action.trial_ids:
            trial = projection.trials[trial_id]
            latest = max(
                trial.executions.values(),
                key=lambda execution: execution.physical_attempt,
            )
            kind: EventKind = (
                "trial.invalid"
                if latest.category in {"agent", "benchmark"}
                else "trial.failed-infrastructure"
            )
            observed_at = self._next_observed()
            event_identity = hashlib.sha256(
                f"{lock.campaign_id}:{trial_id}:spend-cap-exhausted".encode()
            ).hexdigest()[:32]
            events.append(
                new_event(
                    subject_type="trial",
                    subject_id=trial_id,
                    kind=kind,
                    producer="reconciler",
                    payload=LifecyclePayload(
                        parent_id=trial.shard_id,
                        message="retry spend cap exhausted",
                    ),
                    clock=lambda observed_at=observed_at: observed_at,
                    identifier=lambda event_identity=event_identity: event_identity,
                )
            )
        try:
            self.store.ensure_events_unless_cancelled(lock.campaign_id, events)
        except CampaignCancellationWon as error:
            raise ActionExecutionError(str(error)) from error
        return action.action_id

    def _publish_summary(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        projection: RecoveryProjection,
        decision: TerminalDecision | None,
    ) -> str:
        if decision is None or self.finalizer is None:
            raise ActionExecutionError(
                "campaign summary finalization is not configured"
            )
        try:
            self.finalizer.finalize(lock, spec, projection, decision)
        except CampaignFinalizationError as error:
            raise ActionExecutionError(str(error)) from error
        if decision.status != "completed":
            return decision.summary_path
        if self.result_publisher is None:
            raise ActionExecutionError("automatic result publication is not configured")
        try:
            self.result_publisher.publish(lock.campaign_id)
        except (OSError, RuntimeError, ValueError) as error:
            raise ActionExecutionError(
                f"campaign result publication failed: {error}"
            ) from error
        return decision.summary_path

    def _publish_results(self, lock: CampaignLock) -> str:
        if self.result_publisher is None:
            raise ActionExecutionError("automatic result publication is not configured")
        try:
            self.result_publisher.publish(lock.campaign_id)
        except (OSError, RuntimeError, ValueError) as error:
            raise ActionExecutionError(
                f"campaign result publication failed: {error}"
            ) from error
        return lock.campaign_id

    def _submit_wave(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        request: bytes,
        action: ReconcileAction,
        *,
        projection: RecoveryProjection,
        allow_submission: bool,
    ) -> str:
        target = _deployment_target(lock, spec, action.deployment_digest)
        if isinstance(target, ProviderTarget):
            return self._submit_provider_wave(
                lock,
                spec,
                request,
                action,
                target,
                allow_submission=allow_submission,
            )
        desired = _desired_endpoint(lock, spec, action)
        endpoint = managed_wave_endpoint(lock, spec, action.deployment_digest)
        job = self._find_wave(
            spec,
            action,
            target_label_key="harbor-hf-endpoint",
            endpoint_label=endpoint_lease_label_for(endpoint.namespace, endpoint.name),
        )
        if job is not None:
            if self.endpoints.inspect(desired) is None:
                raise ActionExecutionError(
                    "managed wave Job exists without its managed endpoint"
                )
            return job.job_id
        if not allow_submission:
            raise ActionExecutionError(
                "campaign cancellation superseded the unsubmitted wave action"
            )
        if _can_recover_orphaned_endpoint(lock, action, desired, projection):
            existing = self.endpoints.inspect(desired)
            if existing is not None:
                self.endpoints.pause_and_verify(desired)
        self.endpoints.create_or_adopt(desired)
        wave = build_wave_lock(lock, spec, action, endpoint=endpoint)
        return self.jobs.submit(wave, request=request, campaign=lock).job_id

    def _submit_provider_wave(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        request: bytes,
        action: ReconcileAction,
        target: ProviderTarget,
        *,
        allow_submission: bool,
    ) -> str:
        label = hashlib.sha256(target.service.encode()).hexdigest()[:32]
        job = self._find_wave(
            spec,
            action,
            target_label_key="harbor-hf-provider",
            endpoint_label=label,
        )
        if job is not None:
            return job.job_id
        if not allow_submission:
            raise ActionExecutionError(
                "campaign cancellation superseded the unsubmitted wave action"
            )
        wave = build_wave_lock(lock, spec, action)
        return self.jobs.submit(wave, request=request, campaign=lock).job_id

    def _cancel_wave(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        action: ReconcileAction,
    ) -> str | None:
        job = self._managed_wave_job(lock, spec, action)
        if job is None:
            return None
        namespace = _remote_namespace(spec)
        self.jobs.cancel(job, namespace=namespace)
        return job.job_id

    def _cancel_execution(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        action: ReconcileAction,
        projection: RecoveryProjection,
    ) -> str:
        if len(action.target_ids) != 1:
            raise ActionExecutionError(
                "execution cancellation must target exactly one execution"
            )
        execution_id = action.target_ids[0]
        execution = projection.executions.get(execution_id)
        if execution is None or execution.status != "active":
            raise ActionExecutionError(
                f"execution cancellation target is not active: {execution_id}"
            )
        if execution.wave_id is None:
            raise ActionExecutionError(
                f"execution cancellation target has no wave: {execution_id}"
            )
        wave = projection.waves.get(execution.wave_id)
        if wave is None:
            raise ActionExecutionError(
                f"execution cancellation wave is not observed: {execution.wave_id}"
            )
        cancellation = action.model_copy(
            update={
                "deployment_digest": wave.deployment_digest,
                "wave_id": wave.wave_id,
                "shard_ids": wave.shard_ids,
            }
        )
        job = self._managed_wave_job(lock, spec, cancellation)
        if job is not None and job.terminal:
            raise AmbiguousActionOutcome(
                f"HF Job {job.job_id} became terminal; awaiting evidence observation"
            )
        remote_id = self._cancel_wave(lock, spec, cancellation)
        self._record_durable_event(
            lock,
            subject_type="execution",
            subject_id=execution_id,
            kind="execution.cancelled",
            payload=ExecutionOutcomePayload(
                trial_id=execution.trial_id,
                physical_attempt=execution.physical_attempt,
                message="cancelled by campaign reconciliation",
            ),
            identity=f"{execution_id}:cancelled:reconciler",
        )
        return remote_id or execution_id

    def _managed_wave_job(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        action: ReconcileAction,
    ) -> RemoteWaveJob | None:
        target = _deployment_target(lock, spec, action.deployment_digest)
        if isinstance(target, ProviderTarget):
            key: Literal["harbor-hf-endpoint", "harbor-hf-provider"] = (
                "harbor-hf-provider"
            )
            label = hashlib.sha256(target.service.encode()).hexdigest()[:32]
        else:
            endpoint = managed_wave_endpoint(lock, spec, action.deployment_digest)
            key = "harbor-hf-endpoint"
            label = endpoint_lease_label_for(endpoint.namespace, endpoint.name)
        return self._find_wave(
            spec,
            action,
            target_label_key=key,
            endpoint_label=label,
        )

    def _drain_wave(
        self,
        lock: CampaignLock,
        action: ReconcileAction,
        projection: RecoveryProjection,
    ) -> str:
        wave_id = action.wave_id or f"wave-{action.action_key}"
        wave = projection.waves.get(wave_id)
        if wave is None:
            payload = WaveLifecyclePayload(
                deployment_digest=action.deployment_digest,
                provider=_wave_provider(lock, action.deployment_digest),
                shard_ids=action.shard_ids,
                estimated_cost_microusd=action.estimated_cost_microusd or 0,
            )
        else:
            payload = WaveLifecyclePayload(
                deployment_digest=wave.deployment_digest,
                provider=wave.provider,
                shard_ids=wave.shard_ids,
                estimated_cost_microusd=wave.estimated_cost_microusd,
            )
        self._record_durable_event(
            lock,
            subject_type="wave",
            subject_id=wave_id,
            kind="wave.draining",
            payload=payload,
            identity=f"{wave_id}:draining:reconciler",
        )
        if projection.campaign.status == "cancel_requested":
            self._record_durable_event(
                lock,
                subject_type="campaign",
                subject_id=lock.campaign_id,
                kind="campaign.draining",
                payload=LifecyclePayload(message="campaign cancellation is draining"),
                identity=f"{lock.campaign_id}:draining:reconciler",
            )
        return wave_id

    def _manual_intervention(self, lock: CampaignLock, action: ReconcileAction) -> str:
        wave_id = action.wave_id or f"wave-{action.action_key}"
        self._record_durable_event(
            lock,
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="campaign.manual-intervention-required",
            payload=LifecyclePayload(
                parent_id=wave_id,
                message="deployment wave cleanup requires manual intervention",
            ),
            identity=f"{lock.campaign_id}:{wave_id}:manual-intervention",
        )
        return wave_id

    def _cleanup_wave(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        action: ReconcileAction,
        *,
        projection: RecoveryProjection,
    ) -> str:
        job = self._managed_wave_job(lock, spec, action)
        if job is not None and not job.terminal:
            raise AmbiguousActionOutcome(
                f"HF Job {job.job_id} is still {job.stage}; awaiting terminal cleanup"
            )
        target = _deployment_target(lock, spec, action.deployment_digest)
        if isinstance(target, ProviderTarget):
            remote_id = action.wave_id or f"wave-{action.action_key}"
        else:
            desired = _desired_endpoint(lock, spec, action)
            if self.endpoints.inspect(desired) is not None:
                self.endpoints.pause_and_verify(desired)
            remote_id = desired.identity.name
        self._record_wave_closed(lock, action, projection)
        return remote_id

    def _record_wave_closed(
        self,
        lock: CampaignLock,
        action: ReconcileAction,
        projection: RecoveryProjection,
    ) -> None:
        """Durably close the wave: a cancelled Job may never write Bucket markers."""
        wave_id = action.wave_id or f"wave-{action.action_key}"
        observed = projection.waves.get(wave_id)
        if observed is not None and observed.status == "closed":
            return
        if observed is not None:
            payload = WaveLifecyclePayload(
                deployment_digest=observed.deployment_digest,
                provider=observed.provider,
                shard_ids=observed.shard_ids,
                estimated_cost_microusd=observed.estimated_cost_microusd,
            )
        else:
            payload = WaveLifecyclePayload(
                deployment_digest=action.deployment_digest,
                provider=_wave_provider(lock, action.deployment_digest),
                shard_ids=action.shard_ids,
                estimated_cost_microusd=action.estimated_cost_microusd or 0,
            )
        kinds: tuple[Literal["wave.cleaning", "wave.closed"], ...] = ("wave.closed",)
        if observed is not None and observed.status not in {
            "draining",
            "cleaning",
            "cleanup_failed",
        }:
            kinds = ("wave.cleaning", "wave.closed")
        for kind in kinds:
            observed_at = self._next_observed()
            identity = hashlib.sha256(
                f"{lock.campaign_id}:{wave_id}:{kind}:reconciler".encode()
            ).hexdigest()[:32]
            self._ensure_durable_event(
                lock,
                new_event(
                    subject_type="wave",
                    subject_id=wave_id,
                    kind=kind,
                    producer="reconciler",
                    payload=payload,
                    clock=lambda observed_at=observed_at: observed_at,
                    identifier=lambda identity=identity: identity,
                ),
            )

    def _record_durable_event(
        self,
        lock: CampaignLock,
        *,
        subject_type: Literal["campaign", "trial", "execution", "wave"],
        subject_id: str,
        kind: EventKind,
        payload: LifecyclePayload | ExecutionOutcomePayload | WaveLifecyclePayload,
        identity: str,
    ) -> None:
        observed_at = self._next_observed()
        event_identity = hashlib.sha256(
            f"{lock.campaign_id}:{identity}".encode()
        ).hexdigest()[:32]
        self._ensure_durable_event(
            lock,
            new_event(
                subject_type=subject_type,
                subject_id=subject_id,
                kind=kind,
                producer="reconciler",
                payload=payload,
                clock=lambda: observed_at,
                identifier=lambda: event_identity,
            ),
        )

    def _ensure_durable_event(self, lock: CampaignLock, event: CampaignEvent) -> None:
        if self.store.ensure_event(lock.campaign_id, event):
            return
        _lock, events = self.store.load_campaign(lock.campaign_id)
        self._last_observed_at = max(
            event.observed_at,
            *(record.observed_at for record in events),
        )

    def _find_wave(
        self,
        spec: ExperimentSpec,
        action: ReconcileAction,
        *,
        target_label_key: Literal["harbor-hf-endpoint", "harbor-hf-provider"],
        endpoint_label: str,
    ) -> RemoteWaveJob | None:
        wave_id = action.wave_id or f"wave-{action.action_key}"
        namespace = _remote_namespace(spec)
        return self.jobs.find_wave(
            namespace=namespace,
            wave_id=wave_id,
            target_label_key=target_label_key,
            endpoint_label=endpoint_label,
        )

    def _record_terminal(
        self,
        lock: CampaignLock,
        decision: TerminalDecision,
    ) -> None:
        observed = self._next_observed()
        kind = cast(
            EventKind,
            {
                "completed": "campaign.completed",
                "partial": "campaign.partial",
                "failed": "campaign.failed",
                "cancelled": "campaign.cancelled",
            }[decision.status],
        )
        event = new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind=kind,
            producer="publisher",
            payload=TerminalPayload(
                summary_path=decision.summary_path,
                message=decision.reason,
            ),
            clock=lambda: observed,
            identifier=lambda: hashlib.sha256(
                (
                    f"{lock.campaign_id}:{decision.status}:{decision.summary_path}"
                ).encode()
            ).hexdigest()[:32],
        )
        self.store.ensure_event(lock.campaign_id, event)

    def _next_observed(self) -> datetime:
        observed = self.clock().astimezone(UTC)
        if self._last_observed_at is not None and observed <= self._last_observed_at:
            observed = self._last_observed_at + timedelta(microseconds=1)
        self._last_observed_at = observed
        return observed

    def _record_outcome(
        self,
        lock: CampaignLock,
        action: ReconcileAction,
        status: Literal["succeeded", "failed", "ambiguous"],
        message: str | None,
        *,
        remote_id: str | None = None,
    ) -> AppliedAction:
        event = self._event(
            lock,
            _OUTCOME_KINDS[status],
            ActionOutcomePayload(
                action_id=action.action_id,
                message=message,
                remote_id=remote_id,
            ),
        )
        self.store.append_event(lock.campaign_id, event)
        return AppliedAction(
            action_id=action.action_id,
            kind=action.kind,
            status=status,
            remote_id=remote_id,
            message=message,
        )

    def _event(
        self,
        lock: CampaignLock,
        kind: Literal[
            "action.reserved",
            "action.succeeded",
            "action.failed",
            "action.ambiguous",
        ],
        payload: ActionReservedPayload | ActionOutcomePayload,
    ) -> CampaignEvent:
        observed = self._next_observed()
        return new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind=kind,
            producer="reconciler",
            payload=payload,
            clock=lambda: observed,
            identifier=self.identifier,
        )


_AMBIGUOUS_ENDPOINT_ERRORS = (
    AmbiguousEndpointCreate,
    AmbiguousEndpointPause,
    AmbiguousEndpointDelete,
    EndpointVerificationTimeout,
)


def _validate_spend_exhaustion_targets(
    lock: CampaignLock, action: ReconcileAction
) -> None:
    if not action.trial_ids:
        raise ActionExecutionError("spend exhaustion action has no trials")
    if action.trial_ids != action.target_ids or len(action.trial_ids) != len(
        set(action.trial_ids)
    ):
        raise ActionExecutionError("spend exhaustion trial identity is malformed")
    matched_shards: set[str] = set()
    allowed_trials: set[str] = set()
    for run in lock.runs:
        if run.deployment_digest != action.deployment_digest:
            continue
        for shard in run.shards:
            if shard.shard_id not in action.shard_ids:
                continue
            matched_shards.add(shard.shard_id)
            allowed_trials.update(trial.trial_id for trial in shard.trials)
    if matched_shards != set(action.shard_ids) or not set(action.trial_ids).issubset(
        allowed_trials
    ):
        raise ActionExecutionError("spend exhaustion targets the wrong deployment")


def _context_with_unobserved_actions(
    lock: CampaignLock,
    events: list[CampaignEvent],
    reservations: Mapping[str, ReconcileAction],
    context: ReconcileContext | None,
) -> ReconcileContext:
    baseline = context or ReconcileContext()
    projection = project_recovery(lock, events)
    unobserved = [
        admission
        for admission in _active_action_admissions(projection, reservations)
        if admission[0] not in projection.waves
    ]
    if not unobserved:
        return baseline
    usage = _usage_from_admissions(lock.campaign_id, 0, unobserved)
    return baseline.model_copy(
        update={"usage": _combined_usage(baseline.usage, [usage])}
    )


def _build_admission_usage(
    campaign_id: str,
    projection: RecoveryProjection,
    reservations: Mapping[str, ReconcileAction],
) -> AdmissionUsage:
    admissions = [
        (
            wave.wave_id,
            wave.deployment_digest,
            wave.provider,
            wave.estimated_cost_microusd,
        )
        for wave in projection.waves.values()
        if wave.status != "closed"
    ]
    admissions.extend(_active_action_admissions(projection, reservations))
    closed_estimates = sum(
        wave.estimated_cost_microusd
        for wave in projection.waves.values()
        if wave.status == "closed"
    )
    return _usage_from_admissions(
        campaign_id,
        projection.spend_microusd + closed_estimates,
        admissions,
    )


def _usage_from_admissions(
    campaign_id: str,
    recorded_spend: int,
    admissions: Iterable[tuple[str, str, str, int]],
) -> AdmissionUsage:
    unique: dict[str, tuple[str, str, int]] = {}
    for wave_id, deployment, provider, estimate in admissions:
        unique[wave_id] = (deployment, provider, estimate)
    deployments: dict[str, int] = {}
    providers: dict[str, int] = {}
    estimated_spend = 0
    for deployment, provider, estimate in unique.values():
        deployments[deployment] = deployments.get(deployment, 0) + 1
        providers[provider] = providers.get(provider, 0) + 1
        estimated_spend += estimate
    active = len(unique)
    spend = recorded_spend + estimated_spend
    return AdmissionUsage(
        global_active_waves=active,
        deployment_active_waves=deployments,
        provider_active_waves=providers,
        campaign_active_waves={campaign_id: active} if active else {},
        campaign_spend_microusd={campaign_id: spend} if spend else {},
    )


def _active_action_admissions(
    projection: RecoveryProjection,
    reservations: Mapping[str, ReconcileAction],
) -> list[tuple[str, str, str, int]]:
    admissions: list[tuple[str, str, str, int]] = []
    for action_id, state in projection.campaign.actions.items():
        if state.action_kind not in {
            "submit-wave",
            "retry-shard",
        } or state.status not in {"reserved", "succeeded", "ambiguous"}:
            continue
        action = reservations.get(action_id)
        if action is None:
            raise CampaignApplyError(
                f"action event has no reservation record: {action_id}"
            )
        wave_id = action.wave_id or f"wave-{action.action_key}"
        observed = projection.waves.get(wave_id)
        if observed is None or observed.status != "closed":
            admissions.append(
                (
                    wave_id,
                    action.deployment_digest,
                    action.provider,
                    action.estimated_cost_microusd or 0,
                )
            )
    return admissions


def _terminal_wave_needs_drain(wave: object) -> bool:
    return wave is None or getattr(wave, "status", None) not in {
        "draining",
        "cleaning",
        "cleanup_failed",
        "closed",
    }


def _failed_endpoint_remote_id(
    lock: CampaignLock,
    spec: ExperimentSpec,
    action: ReconcileAction,
    projection: RecoveryProjection,
    error: Exception,
) -> str | None:
    if not isinstance(error, EndpointProvisioningError) or action.kind not in {
        "submit-wave",
        "retry-shard",
    }:
        return None
    target = _deployment_target(lock, spec, action.deployment_digest)
    if isinstance(target, ProviderTarget):
        return None
    desired = _desired_endpoint(lock, spec, action)
    if not _can_recover_orphaned_endpoint(lock, action, desired, projection):
        return None
    return desired.identity.name


def _can_recover_orphaned_endpoint(
    lock: CampaignLock,
    action: ReconcileAction,
    desired: DesiredEndpoint,
    projection: RecoveryProjection,
) -> bool:
    provenance_actions = {
        state.action_id
        for state in projection.campaign.actions.values()
        if state.status in {"ambiguous", "failed"}
        and state.remote_id == desired.identity.name
        and _action_targets_deployment(lock, state, action.deployment_digest)
    }
    if not provenance_actions:
        return False
    exempt_action_ids = {action.action_id} | {
        state.action_id
        for state in projection.campaign.actions.values()
        if state.action_id in provenance_actions and state.status == "failed"
    }
    if any(
        wave.deployment_digest == action.deployment_digest and wave.status != "closed"
        for wave in projection.waves.values()
    ):
        return False
    return not any(
        state.action_id != action.action_id
        and state.action_id not in exempt_action_ids
        and state.status in {"reserved", "succeeded", "ambiguous"}
        and _action_targets_deployment(lock, state, action.deployment_digest)
        and (
            (wave := projection.waves.get(f"wave-{state.action_key}")) is None
            or wave.status != "closed"
        )
        for state in projection.campaign.actions.values()
    )


def _action_targets_deployment(
    lock: CampaignLock, action: ActionProjection, deployment_digest: str
) -> bool:
    targets = set(action.target_ids)
    for run in lock.runs:
        if run.deployment_digest != deployment_digest:
            continue
        if action.action_kind == "submit-wave" and any(
            shard.shard_id in targets for shard in run.shards
        ):
            return True
        if action.action_kind == "retry-shard" and any(
            trial.trial_id in targets for shard in run.shards for trial in shard.trials
        ):
            return True
    return False


def _combined_usage(
    baseline: AdmissionUsage, contributions: Iterable[AdmissionUsage]
) -> AdmissionUsage:
    global_waves = baseline.global_active_waves
    deployments = dict(baseline.deployment_active_waves)
    providers = dict(baseline.provider_active_waves)
    campaigns = dict(baseline.campaign_active_waves)
    spend = dict(baseline.campaign_spend_microusd)

    def merge(target: dict[str, int], source: Mapping[str, int]) -> None:
        for key, value in source.items():
            target[key] = target.get(key, 0) + value

    for contribution in contributions:
        global_waves += contribution.global_active_waves
        merge(deployments, contribution.deployment_active_waves)
        merge(providers, contribution.provider_active_waves)
        merge(campaigns, contribution.campaign_active_waves)
        merge(spend, contribution.campaign_spend_microusd)
    return AdmissionUsage(
        global_active_waves=global_waves,
        deployment_active_waves=deployments,
        provider_active_waves=providers,
        campaign_active_waves=campaigns,
        campaign_spend_microusd=spend,
    )


def hugging_face_campaign_reconciler(
    namespace: str,
    *,
    store: CampaignStore | None = None,
    jobs_api: HfJobsApi | None = None,
    bucket_api: BucketApi | None = None,
    runner: TextRunner | None = None,
    endpoint_adapter: HuggingFaceEndpointAdapter | None = None,
) -> CampaignReconciler:
    """Compose production HF adapters around the campaign application layer."""
    if store is None:
        from harbor_hf.control import HubCampaignStore

        store = HubCampaignStore(namespace)
    if jobs_api is None or bucket_api is None:
        from huggingface_hub import HfApi

        api = HfApi()
        jobs_api = jobs_api or cast(HfJobsApi, api)
        bucket_api = bucket_api or cast(BucketApi, api)
    from huggingface_hub import HfApi, get_token

    from harbor_hf.bucket_evidence import (
        BucketEvidenceApi,
        BucketEvidenceWriterApi,
        HubBucketEvidenceReader,
        HubBucketEvidenceWriter,
    )
    from harbor_hf.coordination import CoordinationApi, HubClaimStore
    from harbor_hf.operations import AutomaticCampaignPublisher, DatasetRepositoryApi
    from harbor_hf.result_publisher import DatasetApi, HubDatasetPublisher

    evidence_api = HfApi()
    evidence_cache = Path(
        os.environ.get(
            "HARBOR_HF_EVIDENCE_CACHE",
            str(Path.home() / ".cache" / "harbor-hf" / "evidence"),
        )
    ).expanduser()
    reader = HubBucketEvidenceReader(
        evidence_cache,
        api=cast(BucketEvidenceApi, evidence_api),
    )
    token = get_token()
    if token is None:
        raise CampaignApplyError("HF token is required for campaign reconciliation")
    claims = HubClaimStore(
        namespace,
        token,
        api=cast(CoordinationApi, evidence_api),
    )
    writer = HubBucketEvidenceWriter(
        api=cast(BucketEvidenceWriterApi, evidence_api), claims=claims
    )
    result_publisher = AutomaticCampaignPublisher(
        namespace=namespace,
        store=store,
        reader=reader,
        publisher=HubDatasetPublisher(
            publisher_id=f"reconciler-{uuid.uuid4().hex}",
            leases=HubClaimStore(
                namespace,
                token,
                api=cast(CoordinationApi, evidence_api),
            ),
            api=cast(DatasetApi, evidence_api),
        ),
        repositories=cast(DatasetRepositoryApi, evidence_api),
    )
    adapter = endpoint_adapter or HuggingFaceEndpointAdapter()
    return CampaignReconciler(
        store,
        endpoints=EndpointProvisioner(adapter),
        jobs=HuggingFaceWaveJobAdapter(
            api=jobs_api,
            runner=runner or SubprocessRunner(),
            bucket_api=bucket_api,
        ),
        action_claims=claims,
        observer=BucketCampaignObserver(reader),
        finalizer=BucketCampaignFinalizer(reader, writer),
        result_publisher=result_publisher,
        cleanup=lambda: None,
    )


def _validated_request(lock: CampaignLock, request: bytes) -> ExperimentSpec:
    try:
        spec = load_experiment_bytes(request, source="campaign request")
    except ManifestError as error:
        raise CampaignApplyError("campaign request is not a valid manifest") from error
    try:
        expected = build_campaign_lock(
            build_campaign_plan(spec, recovery_policy=lock.recovery_policy),
            lock.campaign_id,
            clock=lambda: lock.created_at,
        )
    except ValueError as error:
        raise CampaignApplyError(
            "campaign request cannot reproduce its immutable lock"
        ) from error
    if expected != lock:
        raise CampaignApplyError("campaign request does not match its immutable lock")
    return spec


def _validated_reservations(
    values: list[dict[str, JsonValue]],
    *,
    campaign_id: str,
) -> dict[str, ReconcileAction]:
    reservations: dict[str, ReconcileAction] = {}
    for value in values:
        try:
            action = ReconcileAction.model_validate(value)
        except ValidationError as error:
            raise CampaignApplyError("action reservation is malformed") from error
        if (
            action.campaign_id != campaign_id
            or action.action_id != f"act-{action.action_key}"
        ):
            raise CampaignApplyError("action reservation identity is malformed")
        if action.action_id in reservations:
            raise CampaignApplyError("action reservation ID is duplicated")
        reservations[action.action_id] = action
    return reservations


def _pending_actions(
    actions: Mapping[str, ActionProjection],
    reservations: Mapping[str, ReconcileAction],
) -> list[ReconcileAction]:
    pending: list[ReconcileAction] = []
    for action_id, projection in sorted(actions.items()):
        if projection.status not in {"reserved", "ambiguous"}:
            continue
        reserved = reservations.get(action_id)
        if reserved is None:
            raise CampaignApplyError(
                f"action event has no reservation record: {action_id}"
            )
        if (
            reserved.action_key != projection.action_key
            or reserved.kind != projection.action_kind
            or reserved.target_ids != projection.target_ids
        ):
            raise CampaignApplyError(
                f"action reservation does not match its event: {action_id}"
            )
        pending.append(reserved)
    return pending


def _summary_action_succeeded(projection: RecoveryProjection) -> bool:
    return any(
        action.action_kind == "publish-summary" and action.status == "succeeded"
        for action in projection.campaign.actions.values()
    )


def _desired_endpoint(
    lock: CampaignLock,
    spec: ExperimentSpec,
    action: ReconcileAction,
) -> DesiredEndpoint:
    model, deployment = _profiles_for_deployment(lock, spec, action.deployment_digest)
    return build_desired_endpoint(
        namespace=_remote_namespace(spec),
        campaign_id=lock.campaign_id,
        model=model,
        deployment=deployment,
    )


def _profiles_for_deployment(
    lock: CampaignLock,
    spec: ExperimentSpec,
    deployment_digest: str,
) -> tuple[ModelProfile, DeploymentProfile]:
    matches = [run for run in lock.runs if run.deployment_digest == deployment_digest]
    pairs = {(run.model, run.deployment) for run in matches}
    if len(pairs) != 1:
        raise ActionExecutionError(
            "action deployment does not resolve to one model and deployment"
        )
    model_id, deployment_id = pairs.pop()
    model = next(profile for profile in spec.matrix.models if profile.id == model_id)
    deployment = next(
        profile for profile in spec.matrix.deployments if profile.id == deployment_id
    )
    if not isinstance(deployment, DeploymentProfile):
        raise ActionExecutionError(
            "endpoint action targets an inference provider deployment"
        )
    return model, deployment


def _wave_provider(lock: CampaignLock, deployment_digest: str) -> str:
    provider = next(
        (
            run.provider
            for run in lock.runs
            if run.deployment_digest == deployment_digest
        ),
        None,
    )
    return provider or "hf-inference-endpoints"


def _deployment_target(
    lock: CampaignLock,
    spec: ExperimentSpec,
    deployment_digest: str,
) -> DeploymentTarget:
    matches = [run for run in lock.runs if run.deployment_digest == deployment_digest]
    deployment_ids = {run.deployment for run in matches}
    if len(deployment_ids) != 1:
        raise ActionExecutionError(
            "action deployment does not resolve to one deployment target"
        )
    deployment_id = deployment_ids.pop()
    return next(
        profile for profile in spec.matrix.deployments if profile.id == deployment_id
    )


def _remote_namespace(spec: ExperimentSpec) -> str:
    if spec.remote is None:
        raise ActionExecutionError("campaign manifest has no remote execution")
    return spec.remote.job.namespace


def _validated_remote_job(
    value: object,
    expected_labels: Mapping[str, str],
) -> RemoteWaveJob:
    identifier = getattr(value, "id", None)
    labels = getattr(value, "labels", None)
    status = getattr(value, "status", None)
    stage_value = getattr(status, "stage", status)
    stage = getattr(stage_value, "value", stage_value)
    if not isinstance(identifier, str) or not identifier:
        raise ActionExecutionError("HF Job response has no valid ID")
    if not isinstance(labels, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in labels.items()
    ):
        raise ActionExecutionError("HF Job response has invalid labels")
    observed_labels = cast(Mapping[str, str], labels)
    if any(observed_labels.get(key) != item for key, item in expected_labels.items()):
        raise ActionExecutionError("HF Job response has the wrong managed labels")
    if not isinstance(stage, str) or not stage:
        raise ActionExecutionError("HF Job response has no valid stage")
    return RemoteWaveJob(
        job_id=identifier,
        wave_id=expected_labels["harbor-hf-wave"],
        endpoint_label=next(
            expected_labels[key]
            for key in ("harbor-hf-endpoint", "harbor-hf-provider")
            if key in expected_labels
        ),
        target_label_key=next(
            key
            for key in ("harbor-hf-endpoint", "harbor-hf-provider")
            if key in expected_labels
        ),
        stage=stage,
    )


def _write_model(path: Path, value: BaseModel) -> None:
    path.write_text(
        json.dumps(value.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ActionExecutionError",
    "AmbiguousActionOutcome",
    "AppliedAction",
    "CampaignApplyError",
    "CampaignApplyFailure",
    "CampaignApplyResult",
    "CampaignReconciler",
    "EndpointApplicationPort",
    "HuggingFaceWaveJobAdapter",
    "RemoteWaveJob",
    "WaveJobPort",
    "hugging_face_campaign_reconciler",
]
