from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import JsonValue

from harbor_hf.campaign_apply import (
    ActionExecutionError,
    AmbiguousActionOutcome,
    CampaignApplyError,
    CampaignReconciler,
    HuggingFaceWaveJobAdapter,
    RemoteWaveJob,
)
from harbor_hf.campaigns import (
    CampaignLock,
    WaveLock,
    build_campaign_lock,
    build_campaign_plan,
    build_wave_lock,
    managed_wave_endpoint,
)
from harbor_hf.control import (
    ActionReservedPayload,
    CampaignEvent,
    CampaignSubmittedPayload,
    CancellationPayload,
    new_event,
    project_campaign,
)
from harbor_hf.endpoints import (
    DesiredEndpoint,
    EndpointSnapshot,
    EndpointStatus,
    ProvisioningResult,
)
from harbor_hf.models import DeploymentProfile, ExperimentSpec
from harbor_hf.reconciler import ReconcileAction, plan_reconciliation

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

    def create_campaign(
        self, lock: CampaignLock, request: bytes, event: CampaignEvent
    ) -> None:
        raise AssertionError("campaign already exists")

    def load_campaign(
        self, campaign_id: str
    ) -> tuple[CampaignLock, list[CampaignEvent]]:
        assert campaign_id == self.lock.campaign_id
        return self.lock, list(self.events)

    def load_request(self, campaign_id: str) -> bytes:
        assert campaign_id == self.lock.campaign_id
        return self.request

    def list_campaigns(self) -> list[str]:
        return [self.lock.campaign_id]

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
    ) -> RemoteWaveJob | None:
        self.find_calls.append(
            {
                "namespace": namespace,
                "wave_id": wave_id,
                "endpoint_label": endpoint_label,
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
        )

    def cancel(self, job: RemoteWaveJob, *, namespace: str) -> None:
        self.cancellations.append((job.job_id, namespace))


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
    def create_bucket(self, bucket_id: str, **kwargs: object) -> object:
        return SimpleNamespace(id=bucket_id, arguments=kwargs)

    def bucket_info(self, bucket_id: str) -> object:
        return SimpleNamespace(id=bucket_id, private=True)

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
        self.staged: dict[str, object] = {}

    def run_text(self, command: list[str]) -> str:
        self.command = command
        volume = next(value for value in command if value.endswith(":/input:ro"))
        input_dir = Path(volume.removesuffix(":/input:ro"))
        self.staged = {
            "request": (input_dir / "manifest.yaml").read_bytes(),
            "campaign": json.loads(
                (input_dir / "campaign.lock.json").read_text(encoding="utf-8")
            ),
            "wave": json.loads(
                (input_dir / "wave.lock.json").read_text(encoding="utf-8")
            ),
        }
        return "Job started: 0123456789abcdef01234567"


def test_hf_wave_adapter_submits_staged_locks_through_submission_module(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    endpoint = managed_wave_endpoint(lock, remote_spec, action.deployment_digest)
    wave = build_wave_lock(lock, remote_spec, action, endpoint=endpoint)
    runner = InspectingRunner()
    adapter = HuggingFaceWaveJobAdapter(
        api=FakeHfJobsApi(),
        runner=runner,
        bucket_api=FakeBucketApi(),
    )

    job = adapter.submit(wave, request=request, campaign=lock)

    assert job.job_id == "0123456789abcdef01234567"
    assert runner.command is not None
    assert runner.staged == {
        "request": request,
        "campaign": lock.model_dump(mode="json"),
        "wave": wave.model_dump(mode="json"),
    }


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
