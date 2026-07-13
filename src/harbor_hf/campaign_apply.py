from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, cast

import httpx
import yaml
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
    CampaignEvent,
    CampaignStore,
    Clock,
    EventKind,
    IdentifierFactory,
    TerminalPayload,
    WaveLifecyclePayload,
    new_event,
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
from harbor_hf.models import (
    DeploymentProfile,
    DeploymentTarget,
    ExperimentSpec,
    ModelProfile,
)
from harbor_hf.process import ProcessError, SubprocessRunner
from harbor_hf.provider_models import ProviderTarget
from harbor_hf.reconciler import (
    ReconcileAction,
    ReconcileContext,
    ReconcilePlan,
    plan_reconciliation,
)
from harbor_hf.recovery import RecoveryProjection, TerminalDecision
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
    "manual-intervention": 4,
    "publish-summary": 5,
    "publish-results": 6,
    "retry-shard": 7,
    "submit-wave": 8,
}
_OUTCOME_KINDS: dict[
    Literal["succeeded", "failed", "ambiguous"],
    Literal["action.succeeded", "action.failed", "action.ambiguous"],
] = {
    "succeeded": "action.succeeded",
    "failed": "action.failed",
    "ambiguous": "action.ambiguous",
}


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
            raise ActionExecutionError("HF Jobs inspection failed") from error
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


class CampaignReconciler:
    """Apply one bounded, stateless campaign reconciliation pass."""

    def __init__(
        self,
        store: CampaignStore,
        *,
        endpoints: EndpointApplicationPort,
        jobs: WaveJobPort,
        observer: CampaignObserver | None = None,
        finalizer: CampaignFinalizer | None = None,
        result_publisher: CampaignPublicationPort | None = None,
        clock: Clock = lambda: datetime.now(UTC),
        identifier: IdentifierFactory = lambda: uuid.uuid4().hex,
    ) -> None:
        self.store = store
        self.endpoints = endpoints
        self.jobs = jobs
        self.observer = observer
        self.finalizer = finalizer
        self.result_publisher = result_publisher
        self.clock = clock
        self.identifier = identifier
        self._last_observed_at: datetime | None = None

    def apply_campaign(
        self,
        campaign_id: str,
        *,
        context: ReconcileContext | None = None,
    ) -> CampaignApplyResult:
        lock, events = self.store.load_campaign(campaign_id)
        request = self.store.load_request(campaign_id)
        spec = _validated_request(lock, request)
        if self.observer is not None:
            observed = self.observer.observe(lock, spec)
            changed = False
            for event in observed:
                changed = self.store.ensure_event(campaign_id, event) or changed
            if changed:
                lock, events = self.store.load_campaign(campaign_id)
        self._last_observed_at = max(event.observed_at for event in events)
        projection, plan = plan_reconciliation(
            lock,
            events,
            context=context,
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
            projection, plan = plan_reconciliation(
                lock,
                events,
                context=context,
                now=self.clock(),
            )
        reservations = _validated_reservations(
            self.store.load_action_reservations(campaign_id),
            campaign_id=campaign_id,
        )
        pending = _pending_actions(projection.campaign.actions, reservations)
        selected = sorted(
            [*pending, *self._reserve_new(lock, plan.actions)],
            key=lambda action: (_ACTION_PRIORITY[action.kind], action.action_id),
        )
        limit = (context or ReconcileContext()).limits.action_limit
        allow_billable = projection.campaign.status in {"queued", "active"}
        applied = [
            self._apply_action(
                lock,
                spec,
                request,
                action,
                projection=projection,
                terminal_decision=plan.terminal_decision,
                allow_billable=allow_billable,
            )
            for action in selected[:limit]
        ]
        return CampaignApplyResult(
            campaign_id=campaign_id,
            plan=plan,
            applied=applied,
        )

    def apply_all(
        self,
        *,
        context: ReconcileContext | None = None,
    ) -> list[CampaignApplyResult]:
        return [
            self.apply_campaign(campaign_id, context=context)
            for campaign_id in self.store.list_campaigns()
        ]

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
            return self._record_outcome(lock, action, "ambiguous", str(error))
        except (
            ActionExecutionError,
            EndpointProvisioningError,
            ValueError,
        ) as error:
            return self._record_outcome(lock, action, "failed", str(error))
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
                allow_submission=allow_billable,
            )
        if action.kind == "cancel-wave":
            return self._cancel_wave(lock, spec, action)
        if action.kind == "cleanup-wave":
            return self._cleanup_wave(lock, spec, action, projection=projection)
        if action.kind == "publish-summary":
            return self._publish_summary(lock, spec, projection, terminal_decision)
        raise ActionExecutionError(
            f"action execution is not supported by configured adapters: {action.kind}"
        )

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

    def _submit_wave(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        request: bytes,
        action: ReconcileAction,
        *,
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
        job = self._find_wave(
            spec,
            action,
            target_label_key=key,
            endpoint_label=label,
        )
        if job is None:
            return None
        namespace = _remote_namespace(spec)
        self.jobs.cancel(job, namespace=namespace)
        return job.job_id

    def _cleanup_wave(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        action: ReconcileAction,
        *,
        projection: RecoveryProjection,
    ) -> str:
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
                estimated_cost_microusd=0,
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
            self.store.ensure_event(
                lock.campaign_id,
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
    reader = HubBucketEvidenceReader(
        Path(tempfile.mkdtemp(prefix="harbor-hf-evidence-")),
        api=cast(BucketEvidenceApi, evidence_api),
    )
    writer = HubBucketEvidenceWriter(api=cast(BucketEvidenceWriterApi, evidence_api))
    result_publisher = None
    token = get_token()
    if token is not None:
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
        observer=BucketCampaignObserver(reader),
        finalizer=BucketCampaignFinalizer(reader, writer),
        result_publisher=result_publisher,
    )


def _validated_request(lock: CampaignLock, request: bytes) -> ExperimentSpec:
    try:
        raw = yaml.safe_load(request.decode("utf-8"))
        if not isinstance(raw, dict):
            raise CampaignApplyError("campaign request must contain a YAML object")
        spec = ExperimentSpec.model_validate(raw)
    except (UnicodeDecodeError, yaml.YAMLError, ValidationError) as error:
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
    "CampaignApplyResult",
    "CampaignReconciler",
    "EndpointApplicationPort",
    "HuggingFaceWaveJobAdapter",
    "RemoteWaveJob",
    "WaveJobPort",
    "hugging_face_campaign_reconciler",
]
