from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import httpx
import pytest
import yaml
from huggingface_hub.errors import HfHubHTTPError
from pydantic import JsonValue

import harbor_hf.campaign_apply as campaign_apply_module
from harbor_hf.bucket_evidence import HubBucketEvidenceReader, HubBucketEvidenceWriter
from harbor_hf.campaign_apply import (
    ActionExecutionError,
    AmbiguousActionOutcome,
    CampaignApplyError,
    CampaignApplyFailure,
    CampaignApplyResult,
    CampaignReconciler,
    HuggingFaceWaveJobAdapter,
    RemoteWaveJob,
    _action_targets_deployment,
    _active_action_admissions,
    _build_admission_usage,
    _can_recover_orphaned_endpoint,
    _combined_usage,
    _context_with_unobserved_actions,
    _desired_endpoint,
    _usage_from_admissions,
)
from harbor_hf.campaign_finalizer import (
    BucketCampaignFinalizer,
    CampaignFinalizationError,
    _run_evidence,
)
from harbor_hf.campaign_observer import BucketCampaignObserver
from harbor_hf.campaigns import (
    CampaignLock,
    CampaignRecoveryPolicy,
    WaveLock,
    build_campaign_lock,
    build_campaign_plan,
    build_wave_lock,
    managed_wave_endpoint,
)
from harbor_hf.control import (
    ActionOutcomePayload,
    ActionProjection,
    ActionReservedPayload,
    CampaignCancellationWon,
    CampaignEvent,
    CampaignSnapshot,
    CampaignSubmittedPayload,
    CancellationPayload,
    ExecutionOutcomePayload,
    ExecutionStartedPayload,
    LifecyclePayload,
    RetryCategory,
    TerminalPayload,
    WaveLifecyclePayload,
    new_event,
    project_campaign,
)
from harbor_hf.coordination import ClaimConflict, HubClaimStore
from harbor_hf.endpoints import (
    AmbiguousEndpointPause,
    DesiredEndpoint,
    EndpointNotPaused,
    EndpointProviderError,
    EndpointProvisioner,
    EndpointSnapshot,
    EndpointStatus,
    ProvisioningResult,
)
from harbor_hf.hf_endpoints import HuggingFaceEndpointAdapter
from harbor_hf.models import DeploymentProfile, ExperimentSpec
from harbor_hf.operations import AutomaticCampaignPublisher
from harbor_hf.provider_models import (
    ExplicitProviderRoute,
    ProviderLimits,
    ProviderTarget,
)
from harbor_hf.reconciler import (
    AdmissionLimits,
    AdmissionUsage,
    DeploymentAdmission,
    ReconcileAction,
    ReconcileContext,
    _Candidate,
    _MutableUsage,
    plan_reconciliation,
)
from harbor_hf.recovery import RecoveryProjection, TerminalDecision, project_recovery
from harbor_hf.result_publisher import HubDatasetPublisher

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

    def ensure_events_unless_cancelled(
        self, campaign_id: str, events: list[CampaignEvent]
    ) -> bool:
        assert campaign_id == self.lock.campaign_id
        if any(event.kind == "campaign.cancel-requested" for event in self.events):
            raise CampaignCancellationWon(
                "campaign cancellation superseded guarded terminal events"
            )
        changed = False
        for event in events:
            changed = self.ensure_event(campaign_id, event) or changed
        return changed

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


class FakeClaims:
    def __init__(self) -> None:
        self.held: dict[str, dict[str, str]] = {}
        self.acquire_calls: list[tuple[str, dict[str, str]]] = []
        self.release_calls: list[tuple[str, dict[str, str]]] = []
        self.conflict = False

    def acquire(self, path: str, owner: Mapping[str, str]) -> None:
        value = dict(owner)
        self.acquire_calls.append((path, value))
        if self.conflict or path in self.held:
            raise ClaimConflict(f"claim is already held: {path}")
        self.held[path] = value

    def release(self, path: str, owner: Mapping[str, str]) -> None:
        value = dict(owner)
        assert self.held.pop(path) == value
        self.release_calls.append((path, value))


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
        if self.present and self.active:
            raise EndpointNotPaused("active endpoint cannot be adopted")
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
        self.adopt_stage = "RUNNING"
        self.find_stages: list[str] = []
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
            stage=self.find_stages.pop(0) if self.find_stages else self.adopt_stage,
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
    recovery_policy: CampaignRecoveryPolicy | None = None,
) -> tuple[CampaignLock, bytes, CampaignEvent]:
    lock = build_campaign_lock(
        build_campaign_plan(spec, recovery_policy=recovery_policy),
        "campaign-one",
        clock=lambda: NOW,
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
    claims: FakeClaims | None = None,
) -> CampaignReconciler:
    identifiers = itertools.count(identifier_start)
    return CampaignReconciler(
        store,
        endpoints=endpoints,
        jobs=jobs,
        action_claims=claims or FakeClaims(),
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

    claims = FakeClaims()
    result = _reconciler(store, endpoints, jobs, claims=claims).apply_campaign(
        lock.campaign_id
    )

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
    assert len(claims.acquire_calls) == len(claims.release_calls) == 1
    path, owner = claims.acquire_calls[0]
    assert path.startswith("action-leases/") and path.endswith(".json")
    assert claims.release_calls == [(path, owner)]
    assert owner == {
        "campaign_id": lock.campaign_id,
        "action_id": result.applied[0].action_id,
        "reconciler_id": "0" * 31 + "3",
        "expires_at": (NOW + timedelta(hours=2)).isoformat(),
    }


def test_apply_skips_action_held_by_another_reconciler(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    endpoints = FakeEndpoints()
    jobs = FakeJobs()
    claims = FakeClaims()
    claims.conflict = True

    result = _reconciler(store, endpoints, jobs, claims=claims).apply_campaign(
        lock.campaign_id
    )

    assert result.applied == []
    assert endpoints.create_calls == []
    assert jobs.submissions == []
    assert len(claims.acquire_calls) == 1
    assert claims.release_calls == []
    assert store.events[-1].kind == "action.reserved"


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


def test_active_endpoint_is_paused_only_after_endpoint_provisioning_was_ambiguous(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    endpoints = FakeEndpoints()
    endpoints.present = True
    endpoints.active = True
    endpoints.create_error = AmbiguousEndpointPause("created endpoint cleanup pending")

    first = _reconciler(store, endpoints, FakeJobs()).apply_campaign(lock.campaign_id)

    assert first.applied[0].status == "ambiguous"
    assert first.applied[0].remote_id is not None
    assert endpoints.pause_calls == []

    endpoints.create_error = None
    jobs = FakeJobs()
    recovered = _reconciler(store, endpoints, jobs).apply_campaign(lock.campaign_id)

    assert recovered.applied[0].status == "succeeded"
    assert len(endpoints.pause_calls) == 1
    assert endpoints.active is False
    assert len(jobs.submissions) == 1


def test_endpoint_provenance_survives_consecutive_ambiguous_outcomes(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    endpoints = FakeEndpoints()
    jobs = FakeJobs()
    jobs.submit_error = AmbiguousActionOutcome("job submission timed out")

    first = _reconciler(store, endpoints, jobs).apply_campaign(lock.campaign_id)

    action_id = first.applied[0].action_id
    assert first.applied[0].status == "ambiguous"
    assert project_campaign(lock, store.events).actions[action_id].remote_id is None

    endpoints.active = True
    endpoints.create_error = AmbiguousEndpointPause("endpoint pause timed out")
    second = _reconciler(
        store, endpoints, FakeJobs(), identifier_start=10
    ).apply_campaign(lock.campaign_id)

    endpoint_name = second.applied[0].remote_id
    assert second.applied[0].status == "ambiguous"
    assert endpoint_name is not None
    assert (
        project_campaign(lock, store.events).actions[action_id].remote_id
        == endpoint_name
    )

    endpoints.create_error = None
    recovered_jobs = FakeJobs()
    recovered = _reconciler(
        store, endpoints, recovered_jobs, identifier_start=20
    ).apply_campaign(lock.campaign_id)

    assert recovered.applied[0].status == "succeeded"
    assert len(endpoints.pause_calls) == 1
    assert endpoints.active is False
    assert len(recovered_jobs.submissions) == 1


def test_endpoint_provenance_survives_definitive_pause_failure(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    endpoints = FakeEndpoints()
    endpoints.present = True
    endpoints.active = True
    endpoints.create_error = AmbiguousEndpointPause("created endpoint cleanup pending")

    first = _reconciler(store, endpoints, FakeJobs()).apply_campaign(lock.campaign_id)
    assert first.applied[0].status == "ambiguous"

    endpoints.create_error = None
    endpoints.pause_error = EndpointProviderError("pause was rejected")
    failed = _reconciler(
        store, endpoints, FakeJobs(), identifier_start=10
    ).apply_campaign(lock.campaign_id)

    endpoint_name = failed.applied[0].remote_id
    assert failed.applied[0].status == "failed"
    assert endpoint_name is not None

    endpoints.pause_error = None
    recovered_jobs = FakeJobs()
    recovered = _reconciler(
        store, endpoints, recovered_jobs, identifier_start=20
    ).apply_campaign(lock.campaign_id)

    assert recovered.applied[0].status == "succeeded"
    assert recovered.applied[0].remote_id is not None
    assert len(endpoints.pause_calls) == 2
    assert endpoints.active is False
    assert len(recovered_jobs.submissions) == 1


def test_active_endpoint_without_orphan_provenance_is_never_paused(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    endpoints = FakeEndpoints()
    endpoints.present = True
    endpoints.active = True

    result = _reconciler(store, endpoints, FakeJobs()).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "failed"
    assert endpoints.pause_calls == []


def test_other_ambiguous_action_blocks_orphan_endpoint_recovery(
    remote_spec: ExperimentSpec,
) -> None:
    lock, _request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    desired = _desired_endpoint(lock, remote_spec, action)
    other = action.model_copy(
        update={
            "action_id": "act-other",
            "action_key": "other",
            "wave_id": "wave-other",
        }
    )

    def ambiguous(candidate: ReconcileAction, sequence: int) -> CampaignEvent:
        return new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="action.ambiguous",
            producer="reconciler",
            payload=ActionOutcomePayload(
                action_id=candidate.action_id,
                remote_id=desired.identity.name,
            ),
            clock=lambda: NOW + timedelta(seconds=sequence),
            identifier=lambda: f"{sequence:032x}",
        )

    events = [
        submitted,
        _reservation(lock, action, 2),
        ambiguous(action, 3),
        _reservation(lock, other, 4),
        ambiguous(other, 5),
    ]
    projection = project_recovery(lock, events)

    assert not _can_recover_orphaned_endpoint(lock, action, desired, projection)


@pytest.mark.parametrize("action_kind", ["submit-wave", "retry-shard"])
def test_action_target_scan_covers_every_run_sharing_a_deployment(
    remote_spec: ExperimentSpec, action_kind: Literal["submit-wave", "retry-shard"]
) -> None:
    first_agent = remote_spec.matrix.agents[0]
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={
                    "agents": [
                        first_agent,
                        first_agent.model_copy(
                            update={"id": "second-agent", "name": "Second Agent"}
                        ),
                    ]
                }
            )
        }
    )
    lock, _request, _submitted = _campaign(spec)
    first_run, second_run = lock.runs
    target = (
        second_run.shards[0].shard_id
        if action_kind == "submit-wave"
        else second_run.shards[0].trials[0].trial_id
    )
    assert target not in {
        value
        for shard in first_run.shards
        for value in [shard.shard_id, *(trial.trial_id for trial in shard.trials)]
    }
    action = ActionProjection(
        action_id="action-two",
        action_key="key-two",
        action_kind=action_kind,
        target_ids=[target],
        status="reserved",
    )

    assert _action_targets_deployment(lock, action, second_run.deployment_digest)


def test_adopted_durable_event_advances_clock_before_following_transition(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    assert action.wave_id is not None
    payload = WaveLifecyclePayload(
        deployment_digest=action.deployment_digest,
        provider=action.provider,
        shard_ids=action.shard_ids,
        estimated_cost_microusd=action.estimated_cost_microusd or 0,
    )
    reserved = _reservation(lock, action, 1)
    succeeded = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.succeeded",
        producer="reconciler",
        payload=ActionOutcomePayload(action_id=action.action_id),
        clock=lambda: NOW + timedelta(seconds=2),
        identifier=lambda: "2" * 32,
    )
    active = new_event(
        subject_type="wave",
        subject_id=action.wave_id,
        kind="wave.active",
        producer="wave-controller",
        payload=payload,
        clock=lambda: NOW + timedelta(seconds=3),
        identifier=lambda: "3" * 32,
    )

    class ConcurrentStore(FakeStore):
        def ensure_event(self, campaign_id: str, event: CampaignEvent) -> bool:
            if event.kind == "wave.cleaning" and not any(
                record.event_id == event.event_id for record in self.events
            ):
                self.events.append(
                    event.model_copy(update={"observed_at": NOW + timedelta(hours=1)})
                )
                return False
            return super().ensure_event(campaign_id, event)

    store = ConcurrentStore(lock, request, [submitted, reserved, succeeded, active])
    projection = project_recovery(lock, store.events)
    reconciler = _reconciler(store, FakeEndpoints(), FakeJobs())
    reconciler._last_observed_at = projection.campaign.last_observed_at

    reconciler._record_wave_closed(lock, action, projection)

    recovered = project_recovery(lock, store.events)
    cleaning = next(event for event in store.events if event.kind == "wave.cleaning")
    closed = next(event for event in store.events if event.kind == "wave.closed")
    assert recovered.waves[action.wave_id].status == "closed"
    assert closed.observed_at > cleaning.observed_at


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


def test_cleanup_waits_for_terminal_job_and_recovers_on_later_pass(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    submit = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    assert submit.wave_id is not None
    store = FakeStore(
        lock,
        request,
        [submitted, *_cancelled_submitted_wave_events(lock, submit)],
    )
    store.reservations[submit.action_id] = submit.model_dump(mode="json")
    endpoints = FakeEndpoints()
    endpoints.present = True
    endpoints.active = True
    jobs = FakeJobs()
    jobs.adopt_on_find = True
    reconciler = _reconciler(store, endpoints, jobs, identifier_start=16)

    nonterminal = reconciler.apply_campaign(lock.campaign_id)

    assert [(action.kind, action.status) for action in nonterminal.applied] == [
        ("cancel-wave", "succeeded"),
        ("cleanup-wave", "ambiguous"),
    ]
    cleanup_id = nonterminal.applied[1].action_id
    assert jobs.cancellations == [("abcdef012345abcdef012345", "osolmaz")]
    assert not any(
        event.kind == "wave.closed" and event.subject_id == submit.wave_id
        for event in store.events
    )
    assert endpoints.pause_calls == []
    assert endpoints.active is True

    jobs.adopt_stage = "CANCELED"
    terminal = reconciler.apply_campaign(lock.campaign_id)

    recovered = next(
        action for action in terminal.applied if action.action_id == cleanup_id
    )
    assert (recovered.kind, recovered.status) == ("cleanup-wave", "succeeded")
    assert any(
        event.kind == "wave.closed" and event.subject_id == submit.wave_id
        for event in store.events
    )
    assert endpoints.pause_calls
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


def test_apply_all_carries_new_admission_into_the_next_campaign(
    remote_spec: ExperimentSpec,
) -> None:
    first, request, first_submitted = _campaign(remote_spec)
    second = build_campaign_lock(build_campaign_plan(remote_spec), "campaign-two")
    second_submitted = new_event(
        subject_type="campaign",
        subject_id=second.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=second.plan_digest),
        clock=lambda: NOW,
        identifier=lambda: "9" * 32,
    )

    class MultiStore(FakeStore):
        def list_campaigns(self) -> list[str]:
            return [first.campaign_id, second.campaign_id]

    class AdmissionReconciler(CampaignReconciler):
        admitted: set[str] = set()

        def _admission_usage(self, campaign_id: str) -> AdmissionUsage:
            return AdmissionUsage(global_active_waves=int(campaign_id in self.admitted))

        def apply_campaign(
            self,
            campaign_id: str,
            *,
            context: ReconcileContext | None = None,
        ) -> CampaignApplyResult:
            lock, event = (
                (first, first_submitted)
                if campaign_id == first.campaign_id
                else (second, second_submitted)
            )
            _projection, plan = plan_reconciliation(lock, [event], context=context)
            if plan.actions:
                self.admitted.add(campaign_id)
            return CampaignApplyResult(campaign_id=campaign_id, plan=plan, applied=[])

    reconciler = AdmissionReconciler(
        MultiStore(first, request, [first_submitted]),
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        action_claims=FakeClaims(),
    )

    results = reconciler.apply_all(
        context=ReconcileContext(limits=AdmissionLimits(global_active_waves=1))
    )

    first_result = cast(CampaignApplyResult, results[0])
    second_result = cast(CampaignApplyResult, results[1])
    assert len(first_result.plan.actions) == 1
    assert second_result.plan.actions == []
    assert second_result.plan.blocked[0].reason == "global-budget"


def test_apply_all_reports_one_failure_and_continues_other_campaigns(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)

    class MultiStore(FakeStore):
        def list_campaigns(self) -> list[str]:
            return ["broken-campaign", self.lock.campaign_id]

    class IsolatingReconciler(CampaignReconciler):
        def apply_campaign(
            self,
            campaign_id: str,
            *,
            context: ReconcileContext | None = None,
        ) -> CampaignApplyResult:
            if campaign_id == "broken-campaign":
                raise CampaignApplyError("malformed campaign")
            return super().apply_campaign(campaign_id, context=context)

    reconciler = IsolatingReconciler(
        MultiStore(lock, request, [submitted]),
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        action_claims=FakeClaims(),
        clock=lambda: NOW,
        identifier=lambda: "3" * 32,
    )

    results = reconciler.apply_all()

    assert results[0] == CampaignApplyFailure(
        campaign_id="broken-campaign",
        error_type="CampaignApplyError",
        message="malformed campaign",
    )
    assert results[1].campaign_id == lock.campaign_id


def test_terminal_job_without_wave_marker_fails_active_execution_and_retries(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    endpoints = FakeEndpoints()
    jobs = FakeJobs()
    reconciler = _reconciler(store, endpoints, jobs)

    first = reconciler.apply_campaign(lock.campaign_id)
    submitted_action = ReconcileAction.model_validate(
        store.reservations[first.applied[0].action_id]
    )
    assert submitted_action.wave_id is not None
    shard = lock.runs[0].shards[0]
    trial = shard.trials[0]
    store.events.extend(
        [
            new_event(
                subject_type="wave",
                subject_id=submitted_action.wave_id,
                kind="wave.active",
                producer="wave-controller",
                payload=WaveLifecyclePayload(
                    deployment_digest=submitted_action.deployment_digest,
                    provider="hf-inference-endpoints",
                    shard_ids=submitted_action.shard_ids,
                ),
                clock=lambda: NOW + timedelta(seconds=10),
                identifier=lambda: "a" * 32,
            ),
            new_event(
                subject_type="execution",
                subject_id="exec-lost",
                kind="execution.started",
                producer="wave-controller",
                payload=ExecutionStartedPayload(
                    trial_id=trial.trial_id,
                    shard_id=shard.shard_id,
                    physical_attempt=1,
                    wave_id=submitted_action.wave_id,
                ),
                clock=lambda: NOW + timedelta(seconds=11),
                identifier=lambda: "b" * 32,
            ),
        ]
    )
    jobs.adopt_on_find = True
    jobs.adopt_stage = "ERROR"

    recovered = reconciler.apply_campaign(lock.campaign_id)

    assert [(item.kind, item.status) for item in recovered.applied] == [
        ("cleanup-wave", "succeeded")
    ]
    projection, _plan = plan_reconciliation(lock, store.events)
    assert projection.executions["exec-lost"].status == "failed"
    assert projection.executions["exec-lost"].category == "lost"
    assert projection.waves[submitted_action.wave_id].status == "closed"

    jobs.adopt_on_find = False
    retried = CampaignReconciler(
        store,
        endpoints=endpoints,
        jobs=jobs,
        action_claims=FakeClaims(),
        clock=lambda: NOW + timedelta(hours=1),
        identifier=lambda: "c" * 32,
    ).apply_campaign(lock.campaign_id)

    assert [(item.kind, item.status) for item in retried.applied] == [
        ("retry-shard", "succeeded")
    ]
    assert jobs.submissions[-1][0].wave_id != submitted_action.wave_id


def test_terminal_job_recovery_never_regresses_cleanup_failed_wave(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    jobs = FakeJobs()
    reconciler = _reconciler(store, FakeEndpoints(), jobs)
    first = reconciler.apply_campaign(lock.campaign_id)
    action = ReconcileAction.model_validate(
        store.reservations[first.applied[0].action_id]
    )
    assert action.wave_id is not None
    payload = WaveLifecyclePayload(
        deployment_digest=action.deployment_digest,
        provider="hf-inference-endpoints",
        shard_ids=action.shard_ids,
    )
    store.events.extend(
        [
            new_event(
                subject_type="wave",
                subject_id=action.wave_id,
                kind="wave.active",
                producer="wave-controller",
                payload=payload,
                clock=lambda: NOW + timedelta(seconds=10),
                identifier=lambda: "a" * 32,
            ),
            new_event(
                subject_type="wave",
                subject_id=action.wave_id,
                kind="wave.cleanup-failed",
                producer="wave-controller",
                payload=payload,
                clock=lambda: NOW + timedelta(seconds=11),
                identifier=lambda: "b" * 32,
            ),
        ]
    )
    jobs.adopt_on_find = True
    jobs.adopt_stage = "ERROR"

    result = reconciler.apply_campaign(lock.campaign_id)
    projection = project_recovery(lock, store.events)

    assert [(item.kind, item.status) for item in result.applied] == [
        ("manual-intervention", "succeeded")
    ]
    assert projection.waves[action.wave_id].status == "cleanup_failed"
    wave_kinds = [
        event.kind for event in store.events if event.subject_id == action.wave_id
    ]
    tail = wave_kinds[wave_kinds.index("wave.cleanup-failed") :]
    assert "wave.draining" not in tail


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
        action_claims=FakeClaims(),
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
        action_claims=FakeClaims(),
        observer=RecordingObserver(observed, interactions),
        finalizer=RecordingFinalizer(interactions),
        result_publisher=RecordingCampaignPublisher(interactions),
        clock=lambda: NOW,
        identifier=lambda: "3" * 32,
    ).apply_campaign(lock.campaign_id)

    assert store.load_campaign_calls == [lock.campaign_id]


def test_terminal_campaign_never_observes_or_applies_late_actions(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    terminal = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.completed",
        producer="reconciler",
        payload=TerminalPayload(message="done"),
        clock=lambda: NOW + timedelta(seconds=1),
        identifier=lambda: "d" * 32,
    )
    interactions: list[object] = []
    store = FakeStore(lock, request, [submitted, terminal])

    result = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        action_claims=FakeClaims(),
        observer=RecordingObserver(
            _terminal_event(lock, "trial.complete"), interactions
        ),
    ).apply_campaign(lock.campaign_id)

    assert interactions == []
    assert result.applied == []


def test_hf_wave_adapter_treats_listing_failure_as_ambiguous() -> None:
    request = httpx.Request("GET", "https://huggingface.co/api/jobs")
    response = httpx.Response(429, request=request)

    class FailingApi(FakeHfJobsApi):
        def list_jobs(self, **kwargs: object) -> list[object]:
            del kwargs
            raise HfHubHTTPError("rate limited", response=response)

    adapter = HuggingFaceWaveJobAdapter(
        api=FailingApi([]),
        runner=InspectingRunner(),
        bucket_api=FakeBucketApi(),
    )

    with pytest.raises(AmbiguousActionOutcome, match="HF Jobs inspection failed"):
        adapter.find_wave(
            namespace="org", wave_id="wave-one", endpoint_label="endpoint-one"
        )


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


@pytest.mark.parametrize(
    ("category", "expected_status", "expected_outcome"),
    [
        ("benchmark", "invalid", "benchmark_failed"),
        ("transient", "failed_infrastructure", "infrastructure_exhausted"),
    ],
)
def test_apply_exhausts_retry_when_immutable_spend_cap_is_reached(
    remote_spec: ExperimentSpec,
    category: RetryCategory,
    expected_status: str,
    expected_outcome: str,
) -> None:
    policy = CampaignRecoveryPolicy(spend_cap_microusd=100)
    lock, request, submitted = _campaign(remote_spec, recovery_policy=policy)
    run = lock.runs[0]
    shard = run.shards[0]
    trial = shard.trials[0]
    payload = WaveLifecyclePayload(
        deployment_digest=run.deployment_digest,
        provider="hf-inference-endpoints",
        shard_ids=[shard.shard_id],
        estimated_cost_microusd=100,
    )
    events = [
        submitted,
        new_event(
            subject_type="wave",
            subject_id="wave-one",
            kind="wave.active",
            producer="wave-controller",
            payload=payload,
            clock=lambda: NOW + timedelta(seconds=1),
            identifier=lambda: "a" * 32,
        ),
        new_event(
            subject_type="execution",
            subject_id="execution-one",
            kind="execution.started",
            producer="wave-controller",
            payload=ExecutionStartedPayload(
                trial_id=trial.trial_id,
                shard_id=shard.shard_id,
                physical_attempt=1,
                wave_id="wave-one",
            ),
            clock=lambda: NOW + timedelta(seconds=2),
            identifier=lambda: "b" * 32,
        ),
        new_event(
            subject_type="execution",
            subject_id="execution-one",
            kind="execution.failed",
            producer="wave-controller",
            payload=ExecutionOutcomePayload(
                trial_id=trial.trial_id,
                physical_attempt=1,
                category=category,
            ),
            clock=lambda: NOW + timedelta(seconds=3),
            identifier=lambda: "c" * 32,
        ),
        new_event(
            subject_type="wave",
            subject_id="wave-one",
            kind="wave.cleaning",
            producer="wave-controller",
            payload=payload,
            clock=lambda: NOW + timedelta(seconds=4),
            identifier=lambda: "d" * 32,
        ),
        new_event(
            subject_type="wave",
            subject_id="wave-one",
            kind="wave.closed",
            producer="wave-controller",
            payload=payload,
            clock=lambda: NOW + timedelta(seconds=5),
            identifier=lambda: "e" * 32,
        ),
    ]
    store = FakeStore(lock, request, events)
    context = ReconcileContext(
        deployments={
            run.deployment_digest: DeploymentAdmission(estimated_wave_cost_microusd=100)
        }
    )

    identifiers = itertools.count(20)
    reconciler = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        action_claims=FakeClaims(),
        clock=lambda: NOW + timedelta(hours=1),
        identifier=lambda: f"{next(identifiers):032x}",
    )
    result = reconciler.apply_campaign(lock.campaign_id, context=context)

    assert [(item.kind, item.status) for item in result.applied] == [
        ("exhaust-trials", "succeeded")
    ]
    projection = project_recovery(lock, store.events)
    assert projection.trials[trial.trial_id].status == expected_status
    assert projection.trials[trial.trial_id].outcome == expected_outcome
    assert (
        reconciler._exhaust_trials(lock, result.plan.actions[0], projection)
        == result.plan.actions[0].action_id
    )


def test_spend_exhaustion_rejects_an_empty_trial_set(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = ReconcileAction(
        action_id="act-" + "2" * 24,
        action_key="2" * 24,
        kind="exhaust-trials",
        campaign_id=lock.campaign_id,
        deployment_digest=lock.runs[0].deployment_digest,
    )
    reconciler = CampaignReconciler(
        FakeStore(lock, request, [submitted]),
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        action_claims=FakeClaims(),
    )

    with pytest.raises(
        ActionExecutionError, match="spend exhaustion action has no trials"
    ):
        reconciler._exhaust_trials(lock, action, project_recovery(lock, [submitted]))


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"target_ids": []}, "spend exhaustion trial identity is malformed"),
        (
            {"shard_ids": ["shard-unknown"]},
            "spend exhaustion targets the wrong deployment",
        ),
    ],
)
def test_spend_exhaustion_is_bound_to_its_reserved_targets(
    remote_spec: ExperimentSpec, update: dict[str, list[str]], message: str
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    shard = lock.runs[0].shards[0]
    trial_id = shard.trials[0].trial_id
    action = ReconcileAction(
        action_id="act-" + "2" * 24,
        action_key="2" * 24,
        kind="exhaust-trials",
        campaign_id=lock.campaign_id,
        deployment_digest=lock.runs[0].deployment_digest,
        shard_ids=[shard.shard_id],
        trial_ids=[trial_id],
        target_ids=[trial_id],
    ).model_copy(update=update)
    reconciler = CampaignReconciler(
        FakeStore(lock, request, [submitted]),
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        action_claims=FakeClaims(),
    )

    with pytest.raises(ActionExecutionError, match=message):
        reconciler._exhaust_trials(lock, action, project_recovery(lock, [submitted]))


def test_campaign_cancellation_supersedes_reserved_spend_exhaustion(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    shard = lock.runs[0].shards[0]
    trial_id = shard.trials[0].trial_id
    action = ReconcileAction(
        action_id="act-" + "2" * 24,
        action_key="2" * 24,
        kind="exhaust-trials",
        campaign_id=lock.campaign_id,
        deployment_digest=lock.runs[0].deployment_digest,
        shard_ids=[shard.shard_id],
        trial_ids=[trial_id],
        target_ids=[trial_id],
    )
    cancellation = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.cancel-requested",
        producer="cli",
        payload=CancellationPayload(reason="operator"),
        clock=lambda: NOW + timedelta(seconds=2),
        identifier=lambda: "3" * 32,
    )
    manual_intervention = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.manual-intervention-required",
        producer="reconciler",
        payload=LifecyclePayload(message="cleanup failed"),
        clock=lambda: NOW + timedelta(seconds=3),
        identifier=lambda: "4" * 32,
    )
    events = [submitted, cancellation, manual_intervention]
    reconciler = CampaignReconciler(
        FakeStore(lock, request, events),
        endpoints=FakeEndpoints(),
        jobs=FakeJobs(),
        action_claims=FakeClaims(),
    )

    with pytest.raises(
        ActionExecutionError,
        match="campaign cancellation superseded retry exhaustion",
    ):
        reconciler._exhaust_trials(lock, action, project_recovery(lock, events))


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
        action_claims=FakeClaims(),
        clock=clock,
        identifier=lambda: "3" * 32,
    ).apply_campaign(lock.campaign_id)

    assert clock_calls == [None, None, None, None]
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
    run_evidence = _run_evidence(lock, wave.runs[0].configuration, NOW, quality="clean")
    assert run_evidence.model_revision == "not_observed"


def test_spend_capped_provider_wave_uses_locked_estimate_without_cli_context(
    remote_spec: ExperimentSpec,
) -> None:
    spec, target = _provider_spec(remote_spec)
    target = target.model_copy(
        update={
            "limits": ProviderLimits(
                max_attempts=2,
                max_spend_usd=Decimal("2.00"),
                estimated_wave_cost_usd=Decimal("0.75"),
            )
        }
    )
    spec = spec.model_copy(
        update={"matrix": spec.matrix.model_copy(update={"deployments": [target]})}
    )
    lock, request, submitted = _campaign(spec)
    endpoints = FakeEndpoints()
    jobs = FakeJobs()

    result = _reconciler(
        FakeStore(lock, request, [submitted]), endpoints, jobs
    ).apply_campaign(lock.campaign_id)

    assert result.plan.blocked == []
    assert result.plan.actions[0].estimated_cost_microusd == 750_000
    assert len(jobs.submissions) == 1
    wave, submitted_request, campaign = jobs.submissions[0]
    assert wave.provider_target == target
    assert wave.endpoint is None
    assert wave.estimated_cost_microusd == 750_000
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


def _cancelled_submitted_wave_events(
    lock: CampaignLock,
    action: ReconcileAction,
) -> list[CampaignEvent]:
    succeeded = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.succeeded",
        producer="reconciler",
        payload=ActionOutcomePayload(
            action_id=action.action_id,
            remote_id="0123456789abcdef01234567",
        ),
        clock=lambda: NOW + timedelta(seconds=3),
        identifier=lambda: "4" * 32,
    )
    cancellation = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.cancel-requested",
        producer="cli",
        payload=CancellationPayload(reason="operator"),
        clock=lambda: NOW + timedelta(seconds=4),
        identifier=lambda: "5" * 32,
    )
    return [
        _reservation(lock, action, 2),
        succeeded,
        cancellation,
        _terminal_event(lock, "trial.cancelled", sequence=6),
    ]


def test_provider_wave_cancellation_cleanup_closes_wave_without_endpoints(
    remote_spec: ExperimentSpec,
) -> None:
    spec, target = _provider_spec(remote_spec)
    lock, request, submitted = _campaign(spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    assert action.wave_id is not None
    store = FakeStore(
        lock,
        request,
        [submitted, *_cancelled_submitted_wave_events(lock, action)],
    )
    store.reservations[action.action_id] = action.model_dump(mode="json")
    endpoints = FakeEndpoints()
    jobs = FakeJobs()
    jobs.adopt_on_find = True
    jobs.find_stages = ["RUNNING", "RUNNING", "CANCELED"]

    result = _reconciler(store, endpoints, jobs, identifier_start=16).apply_campaign(
        lock.campaign_id
    )

    assert [(applied.kind, applied.status) for applied in result.applied] == [
        ("cancel-wave", "succeeded"),
        ("cleanup-wave", "succeeded"),
    ]
    assert result.applied[1].remote_id == action.wave_id
    assert jobs.cancellations == [("abcdef012345abcdef012345", "osolmaz")]
    assert endpoints.inspect_calls == []
    assert endpoints.create_calls == []
    assert endpoints.pause_calls == []
    closed = [event for event in store.events if event.kind == "wave.closed"]
    assert [event.subject_id for event in closed] == [action.wave_id]
    assert closed[0].producer == "reconciler"
    payload = closed[0].payload
    assert isinstance(payload, WaveLifecyclePayload)
    assert payload.provider == target.service
    assert payload.deployment_digest == action.deployment_digest
    assert payload.shard_ids == action.shard_ids
    projection, plan = plan_reconciliation(
        lock, store.events, now=NOW + timedelta(seconds=9)
    )
    assert projection.waves[action.wave_id].status == "closed"
    assert plan.terminal_decision is not None
    assert plan.terminal_decision.status == "cancelled"
    assert [planned.kind for planned in plan.actions] == ["publish-summary"]


def test_cancelled_unobserved_endpoint_wave_is_durably_closed(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    assert action.wave_id is not None
    store = FakeStore(
        lock,
        request,
        [submitted, *_cancelled_submitted_wave_events(lock, action)],
    )
    store.reservations[action.action_id] = action.model_dump(mode="json")
    endpoints = FakeEndpoints()
    endpoints.present = True
    endpoints.active = True
    jobs = FakeJobs()
    jobs.adopt_on_find = True
    jobs.find_stages = ["RUNNING", "RUNNING", "CANCELED"]

    result = _reconciler(store, endpoints, jobs, identifier_start=16).apply_campaign(
        lock.campaign_id
    )

    assert [(applied.kind, applied.status) for applied in result.applied] == [
        ("cancel-wave", "succeeded"),
        ("cleanup-wave", "succeeded"),
    ]
    assert jobs.cancellations == [("abcdef012345abcdef012345", "osolmaz")]
    assert len(endpoints.pause_calls) == 1
    assert endpoints.active is False
    closed = [event for event in store.events if event.kind == "wave.closed"]
    assert [event.subject_id for event in closed] == [action.wave_id]
    assert closed[0].producer == "reconciler"
    payload = closed[0].payload
    assert isinstance(payload, WaveLifecyclePayload)
    assert payload.provider == "hf-inference-endpoints"
    assert payload.deployment_digest == action.deployment_digest
    assert payload.shard_ids == action.shard_ids
    projection, plan = plan_reconciliation(
        lock, store.events, now=NOW + timedelta(seconds=9)
    )
    assert projection.waves[action.wave_id].status == "closed"
    assert plan.terminal_decision is not None
    assert plan.terminal_decision.status == "cancelled"
    assert [planned.kind for planned in plan.actions] == ["publish-summary"]


@pytest.mark.parametrize("terminal_race", [False, True])
def test_forced_cancellation_cancels_active_execution_and_closes_wave(
    remote_spec: ExperimentSpec, terminal_race: bool
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    submit = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    assert submit.wave_id is not None
    trial = lock.runs[0].shards[0].trials[0]
    shard = lock.runs[0].shards[0]
    wave_payload = WaveLifecyclePayload(
        deployment_digest=submit.deployment_digest,
        provider="hf-inference-endpoints",
        shard_ids=submit.shard_ids,
    )
    events = [
        submitted,
        _reservation(lock, submit, 2),
        new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="action.succeeded",
            producer="reconciler",
            payload=ActionOutcomePayload(
                action_id=submit.action_id,
                remote_id="0123456789abcdef01234567",
            ),
            clock=lambda: NOW + timedelta(seconds=2),
            identifier=lambda: "3" * 32,
        ),
        new_event(
            subject_type="wave",
            subject_id=submit.wave_id,
            kind="wave.active",
            producer="wave-controller",
            payload=wave_payload,
            clock=lambda: NOW + timedelta(seconds=3),
            identifier=lambda: "4" * 32,
        ),
        new_event(
            subject_type="execution",
            subject_id="exec-active",
            kind="execution.started",
            producer="wave-controller",
            payload=ExecutionStartedPayload(
                trial_id=trial.trial_id,
                shard_id=shard.shard_id,
                physical_attempt=1,
                wave_id=submit.wave_id,
            ),
            clock=lambda: NOW + timedelta(seconds=4),
            identifier=lambda: "5" * 32,
        ),
        new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="campaign.cancel-requested",
            producer="cli",
            payload=CancellationPayload(reason="operator"),
            clock=lambda: NOW + timedelta(seconds=5),
            identifier=lambda: "6" * 32,
        ),
    ]
    store = FakeStore(lock, request, events)
    store.reservations[submit.action_id] = submit.model_dump(mode="json")
    jobs = FakeJobs()
    jobs.adopt_on_find = True
    if terminal_race:
        jobs.find_stages = ["RUNNING", "COMPLETED"]
    else:
        jobs.find_stages = ["RUNNING", "RUNNING", "RUNNING", "RUNNING", "CANCELED"]
    identifiers = itertools.count(20)
    reconciler = CampaignReconciler(
        store,
        endpoints=FakeEndpoints(),
        jobs=jobs,
        action_claims=FakeClaims(),
        clock=lambda: (
            NOW
            + timedelta(seconds=lock.recovery_policy.cancellation_grace_seconds + 10)
        ),
        identifier=lambda: f"{next(identifiers):032x}",
    )

    result = reconciler.apply_campaign(lock.campaign_id)

    if terminal_race:
        assert [(item.kind, item.status) for item in result.applied] == [
            ("cancel-execution", "ambiguous")
        ]
        assert jobs.cancellations == []
        assert not any(event.kind == "execution.cancelled" for event in store.events)
        assert not any(event.kind == "wave.closed" for event in store.events)
        return

    assert [(item.kind, item.status) for item in result.applied] == [
        ("cancel-execution", "succeeded"),
        ("cancel-wave", "succeeded"),
        ("cleanup-wave", "succeeded"),
    ]
    assert len(jobs.cancellations) == 2
    assert any(
        event.kind == "execution.cancelled" and event.subject_id == "exec-active"
        for event in store.events
    )
    assert any(
        event.kind == "wave.closed" and event.subject_id == submit.wave_id
        for event in store.events
    )
    projection, plan = plan_reconciliation(lock, store.events)
    assert projection.executions["exec-active"].status == "cancelled"
    assert projection.waves[submit.wave_id].status == "closed"
    assert plan.terminal_decision is not None
    assert plan.terminal_decision.status == "cancelled"


def test_drain_action_records_wave_and_campaign_transitions(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(
        remote_spec, CampaignRecoveryPolicy(cancellation_grace_seconds=100)
    )
    deployment = lock.runs[0].deployment_digest
    wave_id = "wave-" + "7" * 24
    action = ReconcileAction(
        action_id="act-" + "8" * 24,
        action_key="8" * 24,
        kind="drain-wave",
        campaign_id=lock.campaign_id,
        deployment_digest=deployment,
        wave_id=wave_id,
        shard_ids=[lock.runs[0].shards[0].shard_id],
        target_ids=[wave_id],
    )
    wave = new_event(
        subject_type="wave",
        subject_id=wave_id,
        kind="wave.active",
        producer="wave-controller",
        payload=WaveLifecyclePayload(
            deployment_digest=deployment,
            provider="hf-inference-endpoints",
            shard_ids=action.shard_ids,
        ),
        clock=lambda: NOW + timedelta(seconds=1),
        identifier=lambda: "7" * 32,
    )
    cancellation = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.cancel-requested",
        producer="cli",
        payload=CancellationPayload(reason="operator"),
        clock=lambda: NOW + timedelta(seconds=2),
        identifier=lambda: "8" * 32,
    )
    store = FakeStore(
        lock,
        request,
        [submitted, wave, cancellation, _reservation(lock, action, 9)],
    )
    store.reservations[action.action_id] = action.model_dump(mode="json")

    result = _reconciler(store, FakeEndpoints(), FakeJobs()).apply_campaign(
        lock.campaign_id
    )

    assert all(item.status == "succeeded" for item in result.applied)
    assert all(item.remote_id == wave_id for item in result.applied)
    assert sum(event.kind == "wave.draining" for event in store.events) == 1
    assert sum(event.kind == "campaign.draining" for event in store.events) == 1
    projection, _plan = plan_reconciliation(lock, store.events)
    assert projection.campaign.status == "draining"
    assert projection.waves[wave_id].status == "draining"


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
        action_claims=FakeClaims(),
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
        action_claims=FakeClaims(),
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
        action_claims=FakeClaims(),
        result_publisher=RecordingCampaignPublisher([]),
        clock=lambda: NOW,
        identifier=lambda: "3" * 32,
    ).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "failed"
    assert result.applied[0].message == (
        "campaign summary finalization is not configured"
    )
    assert store.events[-1].kind == "action.failed"


def test_exhausted_task_failure_is_published_as_a_completed_scored_run(
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
        action_claims=FakeClaims(),
        finalizer=RecordingFinalizer(interactions),
        result_publisher=RecordingCampaignPublisher(interactions),
        clock=lambda: NOW,
        identifier=lambda: "4" * 32,
    ).apply_campaign(lock.campaign_id)

    assert result.plan.terminal_decision is not None
    assert result.plan.terminal_decision.status == "completed"
    assert [entry[0] for entry in interactions if isinstance(entry, tuple)] == [
        "finalize",
        "publish",
    ]
    assert store.events[-1].kind == "campaign.completed"


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
        action_claims=FakeClaims(),
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
        action_claims=FakeClaims(),
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
        action_claims=FakeClaims(),
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


def test_hugging_face_reconciler_factory_wires_exact_shared_adapters(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    token = "publication-token"
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
    action_claims = cast(HubClaimStore, reconciler.action_claims)
    assert action_claims.repository == "osolmaz/harbor-hf-coordination"


def test_hugging_face_reconciler_factory_requires_hf_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)

    with pytest.raises(CampaignApplyError, match="HF token is required"):
        campaign_apply_module.hugging_face_campaign_reconciler("osolmaz")


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


def test_usage_from_admissions_deduplicates_waves_and_accounts_every_scope() -> None:
    usage = _usage_from_admissions(
        "campaign-one",
        70,
        [
            ("wave-one", "deployment-old", "provider-old", 5),
            ("wave-two", "deployment-two", "provider-two", 20),
            ("wave-one", "deployment-one", "provider-one", 10),
            ("wave-three", "deployment-one", "provider-two", 30),
        ],
    )

    assert usage.model_dump(mode="json") == {
        "global_active_waves": 3,
        "deployment_active_waves": {"deployment-two": 1, "deployment-one": 2},
        "provider_active_waves": {"provider-two": 2, "provider-one": 1},
        "campaign_active_waves": {"campaign-one": 3},
        "campaign_spend_microusd": {"campaign-one": 130},
    }
    assert _usage_from_admissions("campaign-one", 0, []).model_dump(mode="json") == {
        "global_active_waves": 0,
        "deployment_active_waves": {},
        "provider_active_waves": {},
        "campaign_active_waves": {},
        "campaign_spend_microusd": {},
    }
    assert _usage_from_admissions("campaign-one", 7, []).campaign_spend_microusd == {
        "campaign-one": 7
    }


def test_combined_admission_usage_adds_overlapping_and_distinct_scopes() -> None:
    combined = _combined_usage(
        AdmissionUsage(
            global_active_waves=1,
            deployment_active_waves={"deployment-one": 1},
            provider_active_waves={"provider-one": 2},
            campaign_active_waves={"campaign-one": 1},
            campaign_spend_microusd={"campaign-one": 10},
        ),
        [
            AdmissionUsage(
                global_active_waves=2,
                deployment_active_waves={"deployment-one": 3},
                provider_active_waves={"provider-two": 4},
                campaign_active_waves={"campaign-one": 2},
                campaign_spend_microusd={"campaign-one": 20},
            ),
            AdmissionUsage(
                global_active_waves=5,
                deployment_active_waves={"deployment-two": 6},
                provider_active_waves={"provider-one": 7},
                campaign_active_waves={"campaign-two": 8},
                campaign_spend_microusd={"campaign-two": 30},
            ),
        ],
    )

    assert combined.model_dump(mode="json") == {
        "global_active_waves": 8,
        "deployment_active_waves": {
            "deployment-one": 4,
            "deployment-two": 6,
        },
        "provider_active_waves": {"provider-one": 9, "provider-two": 4},
        "campaign_active_waves": {"campaign-one": 3, "campaign-two": 8},
        "campaign_spend_microusd": {"campaign-one": 30, "campaign-two": 30},
    }


def test_wave_projection_usage_accounts_open_and_closed_estimates(
    remote_spec: ExperimentSpec,
) -> None:
    lock, _request, submitted = _campaign(remote_spec)
    run = lock.runs[0]
    shard_id = run.shards[0].shard_id

    def wave(sequence: int, wave_id: str, kind: str, estimate: int) -> CampaignEvent:
        return new_event(
            subject_type="wave",
            subject_id=wave_id,
            kind=cast(Literal["wave.active", "wave.closed"], kind),
            producer="wave-controller",
            payload=WaveLifecyclePayload(
                deployment_digest=run.deployment_digest,
                provider="provider-one",
                shard_ids=[shard_id],
                estimated_cost_microusd=estimate,
            ),
            clock=lambda: NOW + timedelta(seconds=sequence),
            identifier=lambda: f"{sequence:032x}",
        )

    projection = project_recovery(
        lock,
        [
            submitted,
            wave(1, "wave-closed", "wave.closed", 20),
            wave(2, "wave-open", "wave.active", 30),
        ],
    ).model_copy(update={"spend_microusd": 7})

    assert _build_admission_usage(lock.campaign_id, projection, {}).model_dump(
        mode="json"
    ) == {
        "global_active_waves": 1,
        "deployment_active_waves": {run.deployment_digest: 1},
        "provider_active_waves": {"provider-one": 1},
        "campaign_active_waves": {lock.campaign_id: 1},
        "campaign_spend_microusd": {lock.campaign_id: 57},
    }

    mutable = _MutableUsage(
        ReconcileContext(
            usage=AdmissionUsage(
                global_active_waves=2,
                deployment_active_waves={run.deployment_digest: 3},
                provider_active_waves={"provider-one": 4},
                campaign_active_waves={lock.campaign_id: 5},
                campaign_spend_microusd={lock.campaign_id: 11},
            )
        ),
        projection,
    )
    assert mutable.global_waves == 3
    assert mutable.deployments == {run.deployment_digest: 4}
    assert mutable.providers == {"provider-one": 5}
    assert mutable.campaigns == {lock.campaign_id: 6}
    assert mutable.spend == {lock.campaign_id: 68}

    defaulted = _MutableUsage(ReconcileContext(), projection)
    assert defaulted.campaigns == {lock.campaign_id: 1}
    assert defaulted.spend == {lock.campaign_id: 57}

    mutable.admit(
        _Candidate(
            kind="retry-shard",
            deployment_digest=run.deployment_digest,
            provider="provider-one",
            estimated_cost_microusd=13,
        ),
        lock.campaign_id,
    )
    assert (
        mutable.global_waves,
        mutable.deployments,
        mutable.providers,
        mutable.campaigns,
        mutable.spend,
    ) == (
        4,
        {run.deployment_digest: 5},
        {"provider-one": 6},
        {lock.campaign_id: 7},
        {lock.campaign_id: 81},
    )

    mutable.admit(
        _Candidate(
            kind="submit-wave",
            deployment_digest="deployment-two",
            provider="provider-two",
        ),
        "campaign-two",
    )
    assert mutable.campaigns["campaign-two"] == 1
    assert mutable.spend["campaign-two"] == 0


@pytest.mark.parametrize("outcome", ["action.succeeded", "action.ambiguous"])
def test_unobserved_billable_action_contributes_one_admission(
    remote_spec: ExperimentSpec, outcome: str
) -> None:
    lock, _request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    reserved = _reservation(lock, action, 1)
    finished = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind=cast(Literal["action.succeeded", "action.ambiguous"], outcome),
        producer="reconciler",
        payload=ActionOutcomePayload(action_id=action.action_id),
        clock=lambda: NOW + timedelta(seconds=2),
        identifier=lambda: "2" * 32,
    )
    projection = project_recovery(lock, [submitted, reserved, finished])
    reservations = {action.action_id: action}
    expected = [
        (
            action.wave_id,
            action.deployment_digest,
            action.provider,
            action.estimated_cost_microusd or 0,
        )
    ]

    assert _active_action_admissions(projection, reservations) == expected
    context = _context_with_unobserved_actions(
        lock,
        [submitted, reserved, finished],
        reservations,
        ReconcileContext(usage=AdmissionUsage(global_active_waves=2)),
    )
    assert context.usage.global_active_waves == 3
    assert context.usage.deployment_active_waves == {action.deployment_digest: 1}
    assert context.usage.provider_active_waves == {action.provider: 1}
    assert context.usage.campaign_active_waves == {lock.campaign_id: 1}
    assert context.usage.campaign_spend_microusd == {}


def test_reserved_billable_action_consumes_admission_before_execution(
    remote_spec: ExperimentSpec,
) -> None:
    lock, _request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    reserved = _reservation(lock, action, 1)

    assert _active_action_admissions(
        project_recovery(lock, [submitted, reserved]),
        {action.action_id: action},
    ) == [
        (
            action.wave_id,
            action.deployment_digest,
            action.provider,
            action.estimated_cost_microusd or 0,
        )
    ]


def test_active_action_admission_requires_its_reservation(
    remote_spec: ExperimentSpec,
) -> None:
    lock, _request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    projection = project_recovery(
        lock,
        [
            submitted,
            _reservation(lock, action, 1),
            new_event(
                subject_type="campaign",
                subject_id=lock.campaign_id,
                kind="action.succeeded",
                producer="reconciler",
                payload=ActionOutcomePayload(action_id=action.action_id),
                clock=lambda: NOW + timedelta(seconds=2),
                identifier=lambda: "2" * 32,
            ),
        ],
    )

    with pytest.raises(
        CampaignApplyError,
        match=f"^action event has no reservation record: {action.action_id}$",
    ):
        _active_action_admissions(projection, {})


def test_closed_action_wave_releases_reservation_admission(
    remote_spec: ExperimentSpec,
) -> None:
    lock, _request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    assert action.wave_id is not None
    run = lock.runs[0]
    projection_events = [
        submitted,
        _reservation(lock, action, 1),
        new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="action.succeeded",
            producer="reconciler",
            payload=ActionOutcomePayload(action_id=action.action_id),
            clock=lambda: NOW + timedelta(seconds=2),
            identifier=lambda: "2" * 32,
        ),
        new_event(
            subject_type="wave",
            subject_id=action.wave_id,
            kind="wave.closed",
            producer="wave-controller",
            payload=WaveLifecyclePayload(
                deployment_digest=action.deployment_digest,
                provider=action.provider,
                shard_ids=action.shard_ids,
                estimated_cost_microusd=action.estimated_cost_microusd or 0,
            ),
            clock=lambda: NOW + timedelta(seconds=3),
            identifier=lambda: "3" * 32,
        ),
    ]
    projection = project_recovery(lock, projection_events)
    reservations = {action.action_id: action}
    baseline = ReconcileContext(
        usage=AdmissionUsage(
            global_active_waves=2,
            deployment_active_waves={run.deployment_digest: 2},
        )
    )

    assert _active_action_admissions(projection, reservations) == []
    assert (
        _context_with_unobserved_actions(
            lock, projection_events, reservations, baseline
        )
        == baseline
    )


def test_nonbillable_and_unfinished_actions_do_not_consume_admission(
    remote_spec: ExperimentSpec,
) -> None:
    lock, _request, submitted = _campaign(remote_spec)
    billable = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    ignored = billable.model_copy(
        update={
            "action_id": "act-ignored",
            "action_key": "ignored",
            "kind": "publish-results",
            "wave_id": None,
        }
    )
    events = [
        submitted,
        _reservation(lock, ignored, 1),
        new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="action.succeeded",
            producer="reconciler",
            payload=ActionOutcomePayload(action_id=ignored.action_id),
            clock=lambda: NOW + timedelta(seconds=2),
            identifier=lambda: "2" * 32,
        ),
        _reservation(lock, billable, 3),
        new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="action.succeeded",
            producer="reconciler",
            payload=ActionOutcomePayload(action_id=billable.action_id),
            clock=lambda: NOW + timedelta(seconds=4),
            identifier=lambda: "4" * 32,
        ),
    ]

    assert _active_action_admissions(
        project_recovery(lock, events),
        {ignored.action_id: ignored, billable.action_id: billable},
    ) == [
        (
            billable.wave_id,
            billable.deployment_digest,
            billable.provider,
            billable.estimated_cost_microusd or 0,
        )
    ]


def test_observed_open_action_wave_is_not_double_counted(
    remote_spec: ExperimentSpec,
) -> None:
    lock, _request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    assert action.wave_id is not None
    events = [
        submitted,
        _reservation(lock, action, 1),
        new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="action.succeeded",
            producer="reconciler",
            payload=ActionOutcomePayload(action_id=action.action_id),
            clock=lambda: NOW + timedelta(seconds=2),
            identifier=lambda: "2" * 32,
        ),
        new_event(
            subject_type="wave",
            subject_id=action.wave_id,
            kind="wave.active",
            producer="wave-controller",
            payload=WaveLifecyclePayload(
                deployment_digest=action.deployment_digest,
                provider=action.provider,
                shard_ids=action.shard_ids,
                estimated_cost_microusd=action.estimated_cost_microusd or 0,
            ),
            clock=lambda: NOW + timedelta(seconds=3),
            identifier=lambda: "3" * 32,
        ),
    ]
    baseline = ReconcileContext(usage=AdmissionUsage(global_active_waves=4))

    assert (
        _context_with_unobserved_actions(
            lock, events, {action.action_id: action}, baseline
        )
        == baseline
    )


def test_retry_action_uses_its_explicit_wave_admission_identity(
    remote_spec: ExperimentSpec,
) -> None:
    lock, _request, submitted = _campaign(remote_spec)
    submit = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    retry = submit.model_copy(
        update={
            "action_id": "act-retry-explicit",
            "action_key": "retry-explicit",
            "kind": "retry-shard",
            "wave_id": "wave-explicit",
            "trial_ids": [lock.runs[0].shards[0].trials[0].trial_id],
        }
    )
    events = [
        submitted,
        _reservation(lock, retry, 1),
        new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="action.ambiguous",
            producer="reconciler",
            payload=ActionOutcomePayload(action_id=retry.action_id),
            clock=lambda: NOW + timedelta(seconds=2),
            identifier=lambda: "2" * 32,
        ),
    ]

    assert _active_action_admissions(
        project_recovery(lock, events), {retry.action_id: retry}
    ) == [
        (
            "wave-explicit",
            retry.deployment_digest,
            retry.provider,
            retry.estimated_cost_microusd or 0,
        )
    ]


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
