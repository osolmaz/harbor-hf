from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest
import yaml
from pydantic import JsonValue

import harbor_hf.campaign_apply as campaign_apply_module
from harbor_hf.bucket_evidence import HubBucketEvidenceReader, HubBucketEvidenceWriter
from harbor_hf.campaign_apply import (
    ActionExecutionError,
    AmbiguousActionOutcome,
    CampaignApplyError,
    CampaignReconciler,
    HuggingFaceWaveJobAdapter,
    RemoteWaveJob,
)
from harbor_hf.campaign_finalizer import (
    BucketCampaignFinalizer,
    CampaignFinalizationError,
)
from harbor_hf.campaign_observer import BucketCampaignObserver
from harbor_hf.campaigns import (
    CampaignLock,
    WaveLock,
    build_campaign_lock,
    build_campaign_plan,
    build_wave_lock,
    managed_wave_endpoint,
)
from harbor_hf.control import (
    ActionOutcomePayload,
    ActionReservedPayload,
    CampaignEvent,
    CampaignSnapshot,
    CampaignSubmittedPayload,
    CancellationPayload,
    LifecyclePayload,
    new_event,
    project_campaign,
)
from harbor_hf.coordination import HubClaimStore
from harbor_hf.endpoints import (
    DesiredEndpoint,
    EndpointProvisioner,
    EndpointSnapshot,
    EndpointStatus,
    ProvisioningResult,
)
from harbor_hf.hf_endpoints import HuggingFaceEndpointAdapter
from harbor_hf.models import DeploymentProfile, ExperimentSpec
from harbor_hf.operations import AutomaticCampaignPublisher
from harbor_hf.provider_models import ExplicitProviderRoute, ProviderTarget
from harbor_hf.reconciler import (
    AdmissionLimits,
    AdmissionUsage,
    ReconcileAction,
    ReconcileContext,
    plan_reconciliation,
)
from harbor_hf.recovery import RecoveryProjection, TerminalDecision
from harbor_hf.result_publisher import HubDatasetPublisher
from harbor_hf.submission import endpoint_lease_label_for

NOW = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)


class FakeStore:
    def __init__(
        self,
        lock: CampaignLock,
        request: bytes,
        events: list[CampaignEvent],
    ) -> None:
        self.lock = lock
        self.request = request
        self.events = events
        self.reservations: dict[str, dict[str, JsonValue]] = {}
        self.win_reservations = True
        self.load_campaign_calls: list[str] = []

    def create_campaign(
        self, lock: CampaignLock, request: bytes, event: CampaignEvent
    ) -> None:
        raise AssertionError("campaign already exists")

    def load_campaign(
        self, campaign_id: str
    ) -> tuple[CampaignLock, list[CampaignEvent]]:
        self.load_campaign_calls.append(campaign_id)
        assert campaign_id == self.lock.campaign_id
        return self.lock, list(self.events)

    def load_request(self, campaign_id: str) -> bytes:
        assert campaign_id == self.lock.campaign_id
        return self.request

    def list_campaigns(self) -> list[str]:
        return [self.lock.campaign_id]

    def load_snapshot(self, campaign_id: str) -> CampaignSnapshot:
        assert campaign_id == self.lock.campaign_id
        return CampaignSnapshot(
            lock=self.lock,
            events=list(self.events),
            request=self.request,
            control_commit="test-control-commit",
        )

    def ensure_event(self, campaign_id: str, event: CampaignEvent) -> bool:
        assert campaign_id == self.lock.campaign_id
        if any(existing.event_id == event.event_id for existing in self.events):
            return False
        self.events.append(event)
        return True

    def load_action_reservations(self, campaign_id: str) -> list[dict[str, JsonValue]]:
        assert campaign_id == self.lock.campaign_id
        return list(self.reservations.values())

    def append_event(self, campaign_id: str, event: CampaignEvent) -> None:
        assert campaign_id == self.lock.campaign_id
        self.events.append(event)

    def reserve_action(
        self,
        campaign_id: str,
        action: Mapping[str, JsonValue],
        event: CampaignEvent,
    ) -> bool:
        assert campaign_id == self.lock.campaign_id
        if not self.win_reservations:
            return False
        action_id = str(action["action_id"])
        self.reservations[action_id] = dict(action)
        self.events.append(event)
        return True


class FakeEndpoints:
    def __init__(self) -> None:
        self.present = False
        self.active = False
        self.create_calls: list[DesiredEndpoint] = []
        self.inspect_calls: list[DesiredEndpoint] = []
        self.pause_calls: list[DesiredEndpoint] = []
        self.create_error: Exception | None = None
        self.pause_error: Exception | None = None

    def inspect(self, desired: DesiredEndpoint) -> EndpointSnapshot | None:
        self.inspect_calls.append(desired)
        return self._snapshot(desired) if self.present else None

    def create_or_adopt(self, desired: DesiredEndpoint) -> ProvisioningResult:
        self.create_calls.append(desired)
        if self.create_error is not None:
            raise self.create_error
        action = "adopted" if self.present else "created"
        self.present = True
        self.active = False
        return ProvisioningResult(action=action, snapshot=self._snapshot(desired))

    def pause_and_verify(self, desired: DesiredEndpoint) -> EndpointSnapshot:
        self.pause_calls.append(desired)
        if self.pause_error is not None:
            raise self.pause_error
        self.present = True
        self.active = False
        return self._snapshot(desired)

    def _snapshot(self, desired: DesiredEndpoint) -> EndpointSnapshot:
        return EndpointSnapshot(
            namespace=desired.identity.namespace,
            name=desired.identity.name,
            configuration=desired.configuration,
            status=EndpointStatus(
                state="running" if self.active else "paused",
                ready_replicas=1 if self.active else 0,
                target_replicas=1,
            ),
        )


class FakeJobs:
    def __init__(self) -> None:
        self.adopt_on_find = False
        self.find_calls: list[dict[str, str]] = []
        self.submissions: list[tuple[WaveLock, bytes, CampaignLock]] = []
        self.cancellations: list[tuple[str, str]] = []
        self.submit_error: Exception | None = None
        self.find_error: Exception | None = None

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
        self.find_calls.append(
            {
                "namespace": namespace,
                "wave_id": wave_id,
                "endpoint_label": endpoint_label,
                "target_label_key": target_label_key,
            }
        )
        if self.find_error is not None:
            raise self.find_error
        if not self.adopt_on_find:
            return None
        return RemoteWaveJob(
            job_id="abcdef012345abcdef012345",
            wave_id=wave_id,
            endpoint_label=endpoint_label,
            stage="RUNNING",
            target_label_key=target_label_key,
        )

    def submit(
        self,
        lock: WaveLock,
        *,
        request: bytes,
        campaign: CampaignLock,
    ) -> RemoteWaveJob:
        self.submissions.append((lock, request, campaign))
        if self.submit_error is not None:
            raise self.submit_error
        return RemoteWaveJob(
            job_id="0123456789abcdef01234567",
            wave_id=lock.wave_id,
            endpoint_label=self.find_calls[-1]["endpoint_label"],
            stage="SCHEDULING",
            target_label_key=cast(
                Literal["harbor-hf-endpoint", "harbor-hf-provider"],
                self.find_calls[-1]["target_label_key"],
            ),
        )

    def cancel(self, job: RemoteWaveJob, *, namespace: str) -> None:
        self.cancellations.append((job.job_id, namespace))


class RecordingFinalizer:
    def __init__(
        self,
        interactions: list[object],
        *,
        error: CampaignFinalizationError | None = None,
    ) -> None:
        self.interactions = interactions
        self.error = error

    def finalize(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        projection: RecoveryProjection,
        decision: TerminalDecision,
    ) -> None:
        self.interactions.append(("finalize", lock, spec, projection, decision))
        if self.error is not None:
            raise self.error


class RecordingCampaignPublisher:
    def __init__(
        self, interactions: list[object], *, error: Exception | None = None
    ) -> None:
        self.interactions = interactions
        self.error = error

    def publish(self, campaign_id: str) -> object:
        self.interactions.append(("publish", campaign_id))
        if self.error is not None:
            raise self.error
        return {"campaign_id": campaign_id}


class RecordingObserver:
    def __init__(self, event: CampaignEvent, interactions: list[object]) -> None:
        self.event = event
        self.interactions = interactions

    def observe(self, lock: CampaignLock, spec: ExperimentSpec) -> list[CampaignEvent]:
        self.interactions.append(("observe", lock, spec))
        return [self.event]


def _campaign(
    spec: ExperimentSpec,
) -> tuple[CampaignLock, bytes, CampaignEvent]:
    lock = build_campaign_lock(
        build_campaign_plan(spec), "campaign-one", clock=lambda: NOW
    )
    request = yaml.safe_dump(
        spec.model_dump(mode="json", exclude_none=True), sort_keys=True
    ).encode()
    submitted = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=lock.plan_digest),
        clock=lambda: NOW,
        identifier=lambda: "1" * 32,
    )
    return lock, request, submitted


def _terminal_event(
    lock: CampaignLock,
    kind: Literal["trial.complete", "trial.invalid", "trial.cancelled"],
    *,
    trial_index: int = 0,
    sequence: int = 2,
) -> CampaignEvent:
    shard = lock.runs[0].shards[0]
    return new_event(
        subject_type="trial",
        subject_id=shard.trials[trial_index].trial_id,
        kind=kind,
        producer="wave-controller",
        payload=LifecyclePayload(parent_id=shard.shard_id),
        clock=lambda: NOW + timedelta(seconds=sequence - 1),
        identifier=lambda: f"{sequence:032x}",
    )


def _provider_spec(
    remote_spec: ExperimentSpec,
) -> tuple[ExperimentSpec, ProviderTarget]:
    model = remote_spec.matrix.models[0]
    target = ProviderTarget(
        id="provider-one",
        model=model.repo,
        routing=ExplicitProviderRoute(provider="groq"),
        timeout_seconds=17,
        parameters={"temperature": 0},
    )
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(update={"deployments": [target]})
        }
    )
    return spec, target


def _reconciler(
    store: FakeStore,
    endpoints: FakeEndpoints,
    jobs: FakeJobs,
    *,
    identifier_start: int = 2,
) -> CampaignReconciler:
    identifiers = itertools.count(identifier_start)
    return CampaignReconciler(
        store,
        endpoints=endpoints,
        jobs=jobs,
        clock=lambda: NOW,
        identifier=lambda: f"{next(identifiers):032x}",
    )


def test_apply_reserves_provisions_and_submits_managed_wave(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    endpoints = FakeEndpoints()
    jobs = FakeJobs()

    result = _reconciler(store, endpoints, jobs).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "succeeded"
    assert result.applied[0].remote_id == "0123456789abcdef01234567"
    assert len(endpoints.create_calls) == 1
    assert len(jobs.submissions) == 1
    wave, submitted_request, submitted_campaign = jobs.submissions[0]
    assert submitted_request == request
    assert submitted_campaign == lock
    assert wave.endpoint is not None
    deployment = remote_spec.matrix.deployments[0]
    assert isinstance(deployment, DeploymentProfile)
    assert wave.endpoint.name.startswith("harbor-hf-")
    assert wave.endpoint != deployment.endpoint
    assert wave.endpoint.served_model_name == "/repository"
    assert [event.kind for event in store.events[-2:]] == [
        "action.reserved",
        "action.succeeded",
    ]
    assert project_campaign(lock, store.events).actions[
        result.applied[0].action_id
    ].status == ("succeeded")


def test_ambiguous_submission_is_adopted_by_labels_and_managed_endpoint(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    endpoints = FakeEndpoints()
    first_jobs = FakeJobs()
    first_jobs.submit_error = AmbiguousActionOutcome("submission timed out")

    first = _reconciler(store, endpoints, first_jobs).apply_campaign(lock.campaign_id)

    assert first.applied[0].status == "ambiguous"
    action_id = first.applied[0].action_id
    assert len(first_jobs.submissions) == 1
    recovery_jobs = FakeJobs()
    recovery_jobs.adopt_on_find = True

    recovered = _reconciler(
        store, endpoints, recovery_jobs, identifier_start=10
    ).apply_campaign(lock.campaign_id)

    assert len(recovered.applied) == 1
    assert recovered.applied[0].action_id == action_id
    assert recovered.applied[0].status == "succeeded"
    assert recovered.applied[0].remote_id == "abcdef012345abcdef012345"
    assert recovery_jobs.submissions == []
    assert len(endpoints.create_calls) == 1
    assert project_campaign(lock, store.events).actions[action_id].status == "succeeded"


def test_losing_atomic_reservation_does_not_execute_side_effect(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    store.win_reservations = False
    endpoints = FakeEndpoints()
    jobs = FakeJobs()

    result = _reconciler(store, endpoints, jobs).apply_campaign(lock.campaign_id)

    assert result.plan.action_count == 1
    assert result.applied == []
    assert endpoints.create_calls == []
    assert jobs.submissions == []


def test_reserved_cancel_and_cleanup_use_existing_remote_adapters(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    deployment = lock.runs[0].deployment_digest
    cancel = ReconcileAction(
        action_id="act-" + "2" * 24,
        action_key="2" * 24,
        kind="cancel-wave",
        campaign_id=lock.campaign_id,
        deployment_digest=deployment,
        wave_id="wave-" + "1" * 24,
        target_ids=["wave-" + "1" * 24],
    )
    cleanup = ReconcileAction(
        action_id="act-" + "3" * 24,
        action_key="3" * 24,
        kind="cleanup-wave",
        campaign_id=lock.campaign_id,
        deployment_digest=deployment,
        wave_id="wave-" + "1" * 24,
        target_ids=["wave-" + "1" * 24],
    )
    events = [
        submitted,
        new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="campaign.cancel-requested",
            producer="cli",
            payload=CancellationPayload(reason="operator"),
            clock=lambda: NOW + timedelta(seconds=1),
            identifier=lambda: "2" * 32,
        ),
        _reservation(lock, cancel, 3),
        _reservation(lock, cleanup, 4),
    ]
    store = FakeStore(lock, request, events)
    store.reservations = {
        cancel.action_id: cancel.model_dump(mode="json"),
        cleanup.action_id: cleanup.model_dump(mode="json"),
    }
    endpoints = FakeEndpoints()
    endpoints.present = True
    endpoints.active = True
    jobs = FakeJobs()
    jobs.adopt_on_find = True

    result = _reconciler(store, endpoints, jobs).apply_campaign(lock.campaign_id)

    assert [action.status for action in result.applied] == [
        "succeeded",
        "succeeded",
    ]
    assert jobs.cancellations == [("abcdef012345abcdef012345", "osolmaz")]
    endpoint = managed_wave_endpoint(lock, remote_spec, deployment)
    assert jobs.find_calls == [
        {
            "namespace": "osolmaz",
            "wave_id": cancel.wave_id,
            "endpoint_label": endpoint_lease_label_for(
                endpoint.namespace, endpoint.name
            ),
            "target_label_key": "harbor-hf-endpoint",
        }
    ]
    assert len(endpoints.pause_calls) == 1
    assert endpoints.active is False


def test_cleanup_is_idempotent_when_managed_endpoint_is_absent(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = ReconcileAction(
        action_id="act-" + "2" * 24,
        action_key="2" * 24,
        kind="cleanup-wave",
        campaign_id=lock.campaign_id,
        deployment_digest=lock.runs[0].deployment_digest,
        wave_id="wave-" + "1" * 24,
        target_ids=["wave-" + "1" * 24],
    )
    store = FakeStore(lock, request, [submitted, _reservation(lock, action, 2)])
    store.reservations[action.action_id] = action.model_dump(mode="json")
    endpoints = FakeEndpoints()

    result = _reconciler(store, endpoints, FakeJobs()).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "succeeded"
    assert isinstance(result.applied[0].remote_id, str)
    assert result.applied[0].remote_id.startswith("harbor-hf-")
    assert endpoints.pause_calls == []


def test_known_remote_failure_records_failed_outcome(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    jobs = FakeJobs()
    jobs.find_error = ActionExecutionError("list failed")

    result = _reconciler(store, FakeEndpoints(), jobs).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "failed"
    assert result.applied[0].message == "list failed"
    assert store.events[-1].kind == "action.failed"


def test_apply_rejects_request_that_does_not_match_lock(
    remote_spec: ExperimentSpec,
) -> None:
    lock, _request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, b"kind: not-an-experiment\n", [submitted])

    with pytest.raises(CampaignApplyError, match="valid manifest"):
        _reconciler(store, FakeEndpoints(), FakeJobs()).apply_campaign(lock.campaign_id)

    assert store.reservations == {}


def test_apply_all_uses_campaign_listing(remote_spec: ExperimentSpec) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])

    results = _reconciler(store, FakeEndpoints(), FakeJobs()).apply_all()

    assert [result.campaign_id for result in results] == [lock.campaign_id]


def test_apply_observes_durable_event_reloads_and_uses_refreshed_projection(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    observed = _terminal_event(lock, "trial.complete")
    store = FakeStore(lock, request, [submitted])
    interactions: list[object] = []

    result = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        observer=RecordingObserver(observed, interactions),
        finalizer=RecordingFinalizer(interactions),
        result_publisher=RecordingCampaignPublisher(interactions),
        clock=lambda: NOW,
        identifier=lambda: "3" * 32,
    ).apply_campaign(lock.campaign_id)

    assert interactions[0] == ("observe", lock, remote_spec)
    assert store.load_campaign_calls == [lock.campaign_id, lock.campaign_id]
    assert observed in store.events
    assert result.plan.terminal_decision is not None
    assert result.plan.terminal_decision.status == "completed"
    assert result.applied[0].kind == "publish-summary"


def test_apply_does_not_reload_when_observer_event_is_already_durable(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    observed = _terminal_event(lock, "trial.complete")
    store = FakeStore(lock, request, [submitted, observed])
    interactions: list[object] = []

    CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        observer=RecordingObserver(observed, interactions),
        finalizer=RecordingFinalizer(interactions),
        result_publisher=RecordingCampaignPublisher(interactions),
        clock=lambda: NOW,
        identifier=lambda: "3" * 32,
    ).apply_campaign(lock.campaign_id)

    assert store.load_campaign_calls == [lock.campaign_id]


def test_apply_context_limits_pending_actions_before_side_effects(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    cancellation = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.cancel-requested",
        producer="cli",
        payload=CancellationPayload(reason="operator"),
        clock=lambda: NOW + timedelta(seconds=2),
        identifier=lambda: "3" * 32,
    )
    store = FakeStore(
        lock,
        request,
        [submitted, _reservation(lock, action, 2), cancellation],
    )
    store.reservations[action.action_id] = action.model_dump(mode="json")
    jobs = FakeJobs()
    jobs.adopt_on_find = True

    result = _reconciler(store, FakeEndpoints(), jobs).apply_campaign(
        lock.campaign_id,
        context=ReconcileContext(limits=AdmissionLimits(action_limit=1)),
    )

    assert len(result.applied) == 1
    assert result.applied[0].kind == "cancel-wave"
    assert jobs.cancellations == [("abcdef012345abcdef012345", "osolmaz")]


def test_apply_context_admission_usage_blocks_new_billable_action(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    context = ReconcileContext(
        limits=AdmissionLimits(global_active_waves=1),
        usage=AdmissionUsage(global_active_waves=1),
    )

    result = _reconciler(store, FakeEndpoints(), FakeJobs()).apply_campaign(
        lock.campaign_id,
        context=context,
    )

    assert result.plan.actions == []
    assert result.plan.blocked[0].reason == "global-budget"
    assert result.applied == []


def test_apply_uses_configured_clock_for_reconciliation_planning(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    clock_calls: list[None] = []

    def clock() -> datetime:
        clock_calls.append(None)
        return NOW

    result = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        clock=clock,
        identifier=lambda: "3" * 32,
    ).apply_campaign(lock.campaign_id)

    assert clock_calls == [None, None, None]
    assert result.plan.actions[0].kind == "submit-wave"


def test_apply_recovers_terminal_after_summary_outcome_was_already_durable(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    complete = _terminal_event(lock, "trial.complete")
    _projection, plan = plan_reconciliation(lock, [submitted, complete], now=NOW)
    action = plan.actions[0]
    reserved = _reservation(lock, action, 3)
    succeeded = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.succeeded",
        producer="reconciler",
        payload=ActionOutcomePayload(
            action_id=action.action_id,
            remote_id=plan.terminal_decision.summary_path
            if plan.terminal_decision is not None
            else None,
        ),
        clock=lambda: NOW + timedelta(seconds=3),
        identifier=lambda: "4" * 32,
    )
    store = FakeStore(lock, request, [submitted, complete, reserved, succeeded])
    store.reservations[action.action_id] = action.model_dump(mode="json")

    result = _reconciler(store, FakeEndpoints(), FakeJobs()).apply_campaign(
        lock.campaign_id
    )

    assert result.applied == []
    assert result.plan.status == "completed"
    assert result.plan.terminal_decision is not None
    assert result.plan.terminal_decision.status == "completed"
    assert store.load_campaign_calls == [lock.campaign_id, lock.campaign_id]
    assert store.events[-1].kind == "campaign.completed"


def test_cancellation_recovers_reserved_submit_without_starting_billable_work(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    events = [
        submitted,
        _reservation(lock, action, 2),
        new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="campaign.cancel-requested",
            producer="cli",
            payload=CancellationPayload(reason="operator"),
            clock=lambda: NOW + timedelta(seconds=3),
            identifier=lambda: "3" * 32,
        ),
    ]
    store = FakeStore(lock, request, events)
    store.reservations[action.action_id] = action.model_dump(mode="json")
    endpoints = FakeEndpoints()
    jobs = FakeJobs()

    result = _reconciler(store, endpoints, jobs).apply_campaign(lock.campaign_id)

    assert [applied.kind for applied in result.applied] == [
        "cancel-wave",
        "cleanup-wave",
        "submit-wave",
    ]
    assert result.applied[-1].status == "failed"
    assert "cancellation superseded" in str(result.applied[-1].message)
    assert endpoints.create_calls == []
    assert jobs.submissions == []


def test_active_campaign_may_finish_reserved_billable_submission(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    store = FakeStore(lock, request, [submitted, _reservation(lock, action, 2)])
    store.reservations[action.action_id] = action.model_dump(mode="json")
    endpoints = FakeEndpoints()
    jobs = FakeJobs()

    result = _reconciler(store, endpoints, jobs).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "succeeded"
    assert len(endpoints.create_calls) == 1
    assert len(jobs.submissions) == 1


def test_provider_wave_uses_provider_identity_without_endpoint_side_effects(
    remote_spec: ExperimentSpec,
) -> None:
    spec, target = _provider_spec(remote_spec)
    lock, request, submitted = _campaign(spec)
    store = FakeStore(lock, request, [submitted])
    endpoints = FakeEndpoints()
    jobs = FakeJobs()

    result = _reconciler(store, endpoints, jobs).apply_campaign(lock.campaign_id)

    expected_label = hashlib.sha256(target.service.encode()).hexdigest()[:32]
    assert [action.model_dump(mode="json") for action in result.applied] == [
        {
            "action_id": result.plan.actions[0].action_id,
            "kind": "submit-wave",
            "status": "succeeded",
            "remote_id": "0123456789abcdef01234567",
            "message": None,
        }
    ]
    assert jobs.find_calls == [
        {
            "namespace": "osolmaz",
            "wave_id": result.plan.actions[0].wave_id,
            "endpoint_label": expected_label,
            "target_label_key": "harbor-hf-provider",
        }
    ]
    assert len(jobs.submissions) == 1
    wave, submitted_request, campaign = jobs.submissions[0]
    assert wave.provider_target == target
    assert wave.endpoint is None
    assert submitted_request == request
    assert campaign == lock
    assert endpoints.inspect_calls == []
    assert endpoints.create_calls == []
    assert endpoints.pause_calls == []


def test_provider_wave_adoption_and_cancellation_use_the_same_identity(
    remote_spec: ExperimentSpec,
) -> None:
    spec, target = _provider_spec(remote_spec)
    lock, request, submitted = _campaign(spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    jobs = FakeJobs()
    jobs.adopt_on_find = True
    store = FakeStore(lock, request, [submitted, _reservation(lock, action, 2)])
    store.reservations[action.action_id] = action.model_dump(mode="json")

    adopted = _reconciler(store, FakeEndpoints(), jobs).apply_campaign(lock.campaign_id)

    assert adopted.applied[0].remote_id == "abcdef012345abcdef012345"
    assert jobs.submissions == []
    assert action.wave_id is not None
    cancellation = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.cancel-requested",
        producer="cli",
        payload=CancellationPayload(reason="operator"),
        clock=lambda: NOW + timedelta(seconds=2),
        identifier=lambda: "3" * 32,
    )
    cancel = ReconcileAction(
        action_id="act-" + "4" * 24,
        action_key="4" * 24,
        kind="cancel-wave",
        campaign_id=lock.campaign_id,
        deployment_digest=lock.runs[0].deployment_digest,
        wave_id=action.wave_id,
        target_ids=[action.wave_id],
    )
    cancel_store = FakeStore(
        lock,
        request,
        [submitted, cancellation, _reservation(lock, cancel, 4)],
    )
    cancel_store.reservations[cancel.action_id] = cancel.model_dump(mode="json")
    cancel_jobs = FakeJobs()
    cancel_jobs.adopt_on_find = True

    cancelled = _reconciler(cancel_store, FakeEndpoints(), cancel_jobs).apply_campaign(
        lock.campaign_id
    )

    expected_label = hashlib.sha256(target.service.encode()).hexdigest()[:32]
    assert cancelled.applied[0].model_dump(mode="json") == {
        "action_id": cancel.action_id,
        "kind": "cancel-wave",
        "status": "succeeded",
        "remote_id": "abcdef012345abcdef012345",
        "message": None,
    }
    assert cancel_jobs.find_calls == [
        {
            "namespace": "osolmaz",
            "wave_id": action.wave_id,
            "endpoint_label": expected_label,
            "target_label_key": "harbor-hf-provider",
        }
    ]
    assert cancel_jobs.cancellations == [("abcdef012345abcdef012345", "osolmaz")]


def test_provider_reserved_submission_is_superseded_after_cancellation(
    remote_spec: ExperimentSpec,
) -> None:
    spec, _target = _provider_spec(remote_spec)
    lock, request, submitted = _campaign(spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    cancellation = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.cancel-requested",
        producer="cli",
        payload=CancellationPayload(reason="operator"),
        clock=lambda: NOW + timedelta(seconds=2),
        identifier=lambda: "3" * 32,
    )
    store = FakeStore(
        lock,
        request,
        [submitted, _reservation(lock, action, 2), cancellation],
    )
    store.reservations[action.action_id] = action.model_dump(mode="json")
    jobs = FakeJobs()

    result = _reconciler(store, FakeEndpoints(), jobs).apply_campaign(lock.campaign_id)

    submit = next(
        applied for applied in result.applied if applied.kind == "submit-wave"
    )
    assert submit.status == "failed"
    assert submit.message == (
        "campaign cancellation superseded the unsubmitted wave action"
    )
    assert jobs.submissions == []


def test_completed_campaign_finalizes_publishes_and_records_terminal_in_order(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(
        lock, request, [submitted, _terminal_event(lock, "trial.complete")]
    )
    interactions: list[object] = []
    reconciler = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        finalizer=RecordingFinalizer(interactions),
        result_publisher=RecordingCampaignPublisher(interactions),
        clock=lambda: NOW,
        identifier=lambda: "3" * 32,
    )

    result = reconciler.apply_campaign(lock.campaign_id)

    decision = result.plan.terminal_decision
    assert decision is not None
    assert decision.status == "completed"
    assert [entry[0] for entry in interactions if isinstance(entry, tuple)] == [
        "finalize",
        "publish",
    ]
    finalize = cast(tuple[object, ...], interactions[0])
    assert finalize[1] == lock
    assert finalize[2] == remote_spec
    assert isinstance(finalize[3], RecoveryProjection)
    assert finalize[4] == decision
    assert interactions[1] == ("publish", lock.campaign_id)
    assert result.applied[0].model_dump(mode="json") == {
        "action_id": result.plan.actions[0].action_id,
        "kind": "publish-summary",
        "status": "succeeded",
        "remote_id": decision.summary_path,
        "message": None,
    }
    assert [event.kind for event in store.events[-3:]] == [
        "action.reserved",
        "action.succeeded",
        "campaign.completed",
    ]
    terminal = store.events[-1]
    assert terminal.producer == "publisher"
    assert terminal.observed_at == NOW + timedelta(seconds=1, microseconds=3)
    assert (
        terminal.event_id
        == "evt-"
        + hashlib.sha256(
            f"{lock.campaign_id}:completed:{decision.summary_path}".encode()
        ).hexdigest()[:32]
    )
    assert terminal.payload.model_dump(mode="json") == {
        "summary_path": decision.summary_path,
        "summary_sha256": None,
        "message": decision.reason,
    }


def test_completed_campaign_requires_configured_automatic_publisher(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(
        lock, request, [submitted, _terminal_event(lock, "trial.complete")]
    )
    interactions: list[object] = []

    result = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        finalizer=RecordingFinalizer(interactions),
        clock=lambda: NOW,
        identifier=lambda: "3" * 32,
    ).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "failed"
    assert result.applied[0].message == (
        "automatic result publication is not configured"
    )
    assert [entry[0] for entry in interactions if isinstance(entry, tuple)] == [
        "finalize"
    ]
    assert store.events[-1].kind == "action.failed"


def test_terminal_summary_requires_configured_finalizer(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(
        lock, request, [submitted, _terminal_event(lock, "trial.complete")]
    )

    result = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        result_publisher=RecordingCampaignPublisher([]),
        clock=lambda: NOW,
        identifier=lambda: "3" * 32,
    ).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "failed"
    assert result.applied[0].message == (
        "campaign summary finalization is not configured"
    )
    assert store.events[-1].kind == "action.failed"


def test_partial_terminal_status_is_recorded_without_result_publication(
    remote_spec: ExperimentSpec,
) -> None:
    benchmark = remote_spec.benchmark.model_copy(
        update={
            "task_names": ["task-*"],
            "task_digests": {
                "task-one": "sha256:" + "3" * 64,
                "task-two": "sha256:" + "4" * 64,
            },
        }
    )
    spec = remote_spec.model_copy(
        update={
            "benchmark": benchmark,
            "execution": remote_spec.execution.model_copy(
                update={"max_trials_per_shard": 2}
            ),
        }
    )
    lock, request, submitted = _campaign(spec)
    events = [
        submitted,
        _terminal_event(lock, "trial.complete", trial_index=0, sequence=2),
        _terminal_event(lock, "trial.invalid", trial_index=1, sequence=3),
    ]
    store = FakeStore(lock, request, events)
    interactions: list[object] = []

    result = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        finalizer=RecordingFinalizer(interactions),
        result_publisher=RecordingCampaignPublisher(interactions),
        clock=lambda: NOW,
        identifier=lambda: "4" * 32,
    ).apply_campaign(lock.campaign_id)

    assert result.plan.terminal_decision is not None
    assert result.plan.terminal_decision.status == "partial"
    assert [entry[0] for entry in interactions if isinstance(entry, tuple)] == [
        "finalize"
    ]
    assert store.events[-1].kind == "campaign.partial"


@pytest.mark.parametrize(
    ("terminal_kind", "expected_status", "expected_event"),
    [
        ("trial.invalid", "failed", "campaign.failed"),
        ("trial.cancelled", "cancelled", "campaign.cancelled"),
    ],
)
def test_noncompleted_terminal_status_matrix_skips_result_publication(
    remote_spec: ExperimentSpec,
    terminal_kind: Literal["trial.invalid", "trial.cancelled"],
    expected_status: Literal["failed", "cancelled"],
    expected_event: Literal["campaign.failed", "campaign.cancelled"],
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    events = [submitted]
    if terminal_kind == "trial.cancelled":
        events.append(
            new_event(
                subject_type="campaign",
                subject_id=lock.campaign_id,
                kind="campaign.cancel-requested",
                producer="cli",
                payload=CancellationPayload(reason="operator"),
                clock=lambda: NOW + timedelta(microseconds=1),
                identifier=lambda: "2" * 32,
            )
        )
    events.append(_terminal_event(lock, terminal_kind, sequence=3))
    store = FakeStore(lock, request, events)
    interactions: list[object] = []

    result = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        finalizer=RecordingFinalizer(interactions),
        result_publisher=RecordingCampaignPublisher(interactions),
        clock=lambda: NOW,
        identifier=lambda: "4" * 32,
    ).apply_campaign(lock.campaign_id)

    assert result.plan.terminal_decision is not None
    assert result.plan.terminal_decision.status == expected_status
    assert result.applied[-1].kind == "publish-summary"
    assert result.applied[-1].status == "succeeded"
    assert [entry[0] for entry in interactions if isinstance(entry, tuple)] == [
        "finalize"
    ]
    assert store.events[-1].kind == expected_event


@pytest.mark.parametrize(
    "error",
    [
        OSError("disk unavailable"),
        RuntimeError("publisher unavailable"),
        ValueError("publication invalid"),
    ],
)
def test_completed_campaign_publication_failure_matrix_records_exact_outcome(
    remote_spec: ExperimentSpec, error: Exception
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(
        lock, request, [submitted, _terminal_event(lock, "trial.complete")]
    )
    interactions: list[object] = []

    result = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        finalizer=RecordingFinalizer(interactions),
        result_publisher=RecordingCampaignPublisher(interactions, error=error),
        clock=lambda: NOW,
        identifier=lambda: "3" * 32,
    ).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "failed"
    assert result.applied[0].remote_id is None
    assert result.applied[0].message == f"campaign result publication failed: {error}"
    assert [entry[0] for entry in interactions if isinstance(entry, tuple)] == [
        "finalize",
        "publish",
    ]
    assert [event.kind for event in store.events[-2:]] == [
        "action.reserved",
        "action.failed",
    ]
    assert not any(event.kind.startswith("campaign.") for event in store.events[2:])


def test_campaign_finalization_failure_does_not_publish_or_record_terminal(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(
        lock, request, [submitted, _terminal_event(lock, "trial.invalid")]
    )
    interactions: list[object] = []

    result = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        finalizer=RecordingFinalizer(
            interactions,
            error=CampaignFinalizationError("summary conflict"),
        ),
        result_publisher=RecordingCampaignPublisher(interactions),
        clock=lambda: NOW,
        identifier=lambda: "3" * 32,
    ).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "failed"
    assert result.applied[0].message == "summary conflict"
    assert [entry[0] for entry in interactions if isinstance(entry, tuple)] == [
        "finalize"
    ]
    assert store.events[-1].kind == "action.failed"


class FakeHfJobsApi:
    def __init__(self, jobs: list[object] | None = None) -> None:
        self.jobs = jobs or []
        self.list_arguments: list[dict[str, object]] = []
        self.cancel_arguments: list[dict[str, object]] = []

    def list_jobs(self, **kwargs: object) -> list[object]:
        self.list_arguments.append(kwargs)
        return self.jobs

    def cancel_job(self, **kwargs: object) -> None:
        self.cancel_arguments.append(kwargs)


class FakeBucketApi:
    def __init__(self) -> None:
        self.staged: dict[str, bytes] = {}

    def create_bucket(self, bucket_id: str, **kwargs: object) -> object:
        return SimpleNamespace(id=bucket_id, arguments=kwargs)

    def bucket_info(self, bucket_id: str) -> object:
        return SimpleNamespace(id=bucket_id, private=True)

    def batch_bucket_files(
        self,
        bucket_id: str,
        *,
        add: list[tuple[bytes, str]],
        **kwargs: object,
    ) -> object:
        assert bucket_id == "osolmaz/jobs-artifacts"
        assert kwargs == {}
        self.staged.update({path: content for content, path in add})
        return object()

    def create_repo(self, repo_id: str, **kwargs: object) -> object:
        return SimpleNamespace(id=repo_id, arguments=kwargs)

    def repo_info(self, repo_id: str, **kwargs: object) -> object:
        return SimpleNamespace(id=repo_id, private=True, sha="1" * 40)

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object:
        raise AssertionError((repo_id, operations, kwargs))


class InspectingRunner:
    def __init__(self) -> None:
        self.command: list[str] | None = None

    def run_text(self, command: list[str]) -> str:
        self.command = command
        return "Job started: 0123456789abcdef01234567"


@pytest.mark.parametrize("token", [None, "publication-token"])
def test_hugging_face_reconciler_factory_wires_exact_shared_adapters(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token: str | None,
) -> None:
    import huggingface_hub

    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    jobs_api = FakeHfJobsApi()
    bucket_api = FakeBucketApi()
    runner = InspectingRunner()
    endpoint_adapter = cast(HuggingFaceEndpointAdapter, object())
    evidence_api = SimpleNamespace(identity="shared-evidence-api")
    api_calls: list[dict[str, object]] = []
    cache_prefixes: list[str] = []

    def create_api(**kwargs: object) -> object:
        api_calls.append(kwargs)
        return evidence_api

    cache_root = tmp_path / "evidence-cache"
    monkeypatch.setattr(huggingface_hub, "HfApi", create_api)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: token)
    monkeypatch.setattr(
        campaign_apply_module.tempfile,
        "mkdtemp",
        lambda *, prefix: cache_prefixes.append(prefix) or str(cache_root),
    )
    monkeypatch.setattr(
        campaign_apply_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="f" * 32),
    )

    reconciler = campaign_apply_module.hugging_face_campaign_reconciler(
        "osolmaz",
        store=store,
        jobs_api=jobs_api,
        bucket_api=bucket_api,
        runner=runner,
        endpoint_adapter=endpoint_adapter,
    )

    assert api_calls == [{}]
    assert cache_prefixes == ["harbor-hf-evidence-"]
    assert reconciler.store is store
    endpoints = cast(EndpointProvisioner, reconciler.endpoints)
    jobs = cast(HuggingFaceWaveJobAdapter, reconciler.jobs)
    assert endpoints.port is endpoint_adapter
    assert jobs.api is jobs_api
    assert jobs.runner is runner
    assert jobs.bucket_api is bucket_api
    assert reconciler.observer is not None
    assert reconciler.finalizer is not None
    observer = cast(BucketCampaignObserver, reconciler.observer)
    finalizer = cast(BucketCampaignFinalizer, reconciler.finalizer)
    reader = cast(HubBucketEvidenceReader, observer.reader)
    assert reader.cache_root == cache_root
    assert reader.api is evidence_api
    assert finalizer.reader is reader
    writer = cast(HubBucketEvidenceWriter, finalizer.writer)
    assert writer.api is evidence_api
    if token is None:
        assert reconciler.result_publisher is None
    else:
        automatic = cast(AutomaticCampaignPublisher, reconciler.result_publisher)
        assert automatic is not None
        assert automatic.namespace == "osolmaz"
        assert automatic.store is store
        assert automatic.reader is reader
        assert automatic.repositories is evidence_api
        publisher = cast(HubDatasetPublisher, automatic.publisher)
        leases = cast(HubClaimStore, publisher.leases)
        assert publisher.publisher_id == "reconciler-" + "f" * 32
        assert publisher.api is evidence_api
        assert leases.repository == "osolmaz/harbor-hf-coordination"
        assert leases.token == token
        assert leases.api is evidence_api


def test_hf_wave_adapter_submits_staged_locks_through_submission_module(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    endpoint = managed_wave_endpoint(lock, remote_spec, action.deployment_digest)
    wave = build_wave_lock(lock, remote_spec, action, endpoint=endpoint)
    runner = InspectingRunner()
    bucket_api = FakeBucketApi()
    adapter = HuggingFaceWaveJobAdapter(
        api=FakeHfJobsApi(),
        runner=runner,
        bucket_api=bucket_api,
    )

    job = adapter.submit(wave, request=request, campaign=lock)

    assert job.job_id == "0123456789abcdef01234567"
    assert runner.command is not None
    staged = {
        path.rsplit("/", 1)[-1]: content for path, content in bucket_api.staged.items()
    }
    assert staged["manifest.yaml"] == request
    assert json.loads(staged["campaign.lock.json"]) == lock.model_dump(mode="json")
    assert json.loads(staged["wave.lock.json"]) == wave.model_dump(mode="json")
    volume = next(value for value in runner.command if value.endswith(":/input:ro"))
    assert volume.startswith("hf://buckets/osolmaz/jobs-artifacts/job-inputs/")


def test_hf_wave_adapter_adopts_only_exact_labels() -> None:
    labels = {
        "harbor-hf-wave": "wave-one",
        "harbor-hf-endpoint": "endpoint-one",
    }
    resource = SimpleNamespace(
        id="0123456789abcdef01234567",
        labels=labels,
        status=SimpleNamespace(stage="RUNNING"),
    )
    api = FakeHfJobsApi([resource])
    adapter = HuggingFaceWaveJobAdapter(
        api=api,
        runner=InspectingRunner(),
        bucket_api=FakeBucketApi(),
    )

    observed = adapter.find_wave(
        namespace="org",
        wave_id="wave-one",
        endpoint_label="endpoint-one",
    )

    assert observed is not None
    assert observed.job_id == resource.id
    assert api.list_arguments == [{"labels": labels, "namespace": "org"}]
    wrong = SimpleNamespace(
        id=resource.id,
        labels={**labels, "harbor-hf-endpoint": "wrong"},
        status=resource.status,
    )
    api.jobs = [wrong]
    with pytest.raises(ActionExecutionError, match="wrong managed labels"):
        adapter.find_wave(
            namespace="org",
            wave_id="wave-one",
            endpoint_label="endpoint-one",
        )


def test_hf_wave_adapter_recovers_one_active_or_completed_duplicate() -> None:
    labels = {
        "harbor-hf-wave": "wave-one",
        "harbor-hf-endpoint": "endpoint-one",
    }

    def resource(job_id: str, stage: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=job_id,
            labels=labels,
            status=SimpleNamespace(stage=stage),
        )

    canceled = resource("000000000000000000000000", "CANCELED")
    active = resource("111111111111111111111111", "RUNNING")
    completed = resource("222222222222222222222222", "COMPLETED")
    api = FakeHfJobsApi([canceled, active])
    adapter = HuggingFaceWaveJobAdapter(
        api=api,
        runner=InspectingRunner(),
        bucket_api=FakeBucketApi(),
    )

    observed = adapter.find_wave(
        namespace="org",
        wave_id="wave-one",
        endpoint_label="endpoint-one",
    )
    assert observed is not None
    assert observed.job_id == active.id

    api.jobs = [canceled, completed]
    observed = adapter.find_wave(
        namespace="org",
        wave_id="wave-one",
        endpoint_label="endpoint-one",
    )
    assert observed is not None
    assert observed.job_id == completed.id


@pytest.mark.parametrize(
    "stages",
    [("RUNNING", "SCHEDULING"), ("COMPLETED", "COMPLETED")],
)
def test_hf_wave_adapter_rejects_unresolved_duplicate_identity(
    stages: tuple[str, str],
) -> None:
    labels = {
        "harbor-hf-wave": "wave-one",
        "harbor-hf-endpoint": "endpoint-one",
    }
    api = FakeHfJobsApi(
        [
            SimpleNamespace(
                id=f"{index + 1}" * 24,
                labels=labels,
                status=SimpleNamespace(stage=stage),
            )
            for index, stage in enumerate(stages)
        ]
    )
    adapter = HuggingFaceWaveJobAdapter(
        api=api,
        runner=InspectingRunner(),
        bucket_api=FakeBucketApi(),
    )

    with pytest.raises(
        ActionExecutionError,
        match="^multiple HF Jobs have the managed wave identity: wave-one$",
    ):
        adapter.find_wave(
            namespace="org",
            wave_id="wave-one",
            endpoint_label="endpoint-one",
        )


def _reservation(
    lock: CampaignLock,
    action: ReconcileAction,
    sequence: int,
) -> CampaignEvent:
    return new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.reserved",
        producer="reconciler",
        payload=ActionReservedPayload(
            action_id=action.action_id,
            action_key=action.action_key,
            action_kind=action.kind,
            target_ids=action.target_ids,
        ),
        clock=lambda: NOW + timedelta(seconds=sequence),
        identifier=lambda: f"{sequence:032x}",
    )
