from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError
from test_campaign_apply import (
    NOW,
    FakeBucketApi,
    FakeEndpoints,
    FakeHfJobsApi,
    FakeJobs,
    FakeStore,
    InspectingRunner,
    _campaign,
    _provider_spec,
    _reconciler,
    _reservation,
)

from harbor_hf.campaign_apply import (
    ActionExecutionError,
    AmbiguousActionOutcome,
    CampaignApplyError,
    HuggingFaceWaveJobAdapter,
    RemoteWaveJob,
)
from harbor_hf.campaigns import (
    CampaignLock,
    WaveLock,
    build_wave_lock,
    managed_wave_endpoint,
)
from harbor_hf.endpoints import (
    AmbiguousEndpointCreate,
    EndpointVerificationTimeout,
)
from harbor_hf.models import EndpointRef, ExperimentSpec
from harbor_hf.process import ProcessError
from harbor_hf.reconciler import ReconcileAction, plan_reconciliation
from harbor_hf.submission import endpoint_lease_label_for


def _http_error(status: int) -> HfHubHTTPError:
    request = httpx.Request("POST", "https://huggingface.co/api/jobs")
    return HfHubHTTPError(
        f"HTTP {status}", response=httpx.Response(status, request=request)
    )


class CancelErrorApi(FakeHfJobsApi):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def cancel_job(self, **kwargs: object) -> None:
        self.cancel_arguments.append(kwargs)
        raise self.error


class ListErrorApi(FakeHfJobsApi):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def list_jobs(self, **kwargs: object) -> list[object]:
        self.list_arguments.append(kwargs)
        raise self.error


class FailingRunner:
    def __init__(self, error: Exception | None = None, output: str = "") -> None:
        self.error = error
        self.output = output

    def run_text(self, command: list[str]) -> str:
        if self.error is not None:
            raise self.error
        return self.output


def _adapter(api: FakeHfJobsApi | None = None) -> HuggingFaceWaveJobAdapter:
    return HuggingFaceWaveJobAdapter(
        api=api or FakeHfJobsApi(),
        runner=InspectingRunner(),
        bucket_api=FakeBucketApi(),
    )


def _job(stage: str) -> RemoteWaveJob:
    return RemoteWaveJob(
        job_id="a" * 24,
        wave_id="wave-one",
        endpoint_label="endpoint-one",
        stage=stage,
    )


@pytest.mark.parametrize(
    ("stage", "terminal"),
    [
        ("CANCELED", True),
        ("COMPLETED", True),
        ("DELETED", True),
        ("ERROR", True),
        ("RUNNING", False),
        ("SCHEDULING", False),
        ("PENDING", False),
    ],
)
def test_remote_wave_job_terminal_stage_matrix(stage: str, terminal: bool) -> None:
    assert _job(stage).terminal is terminal


def test_cancel_skips_api_for_terminal_job_and_calls_exact_arguments() -> None:
    api = FakeHfJobsApi()
    adapter = _adapter(api)

    adapter.cancel(_job("COMPLETED"), namespace="org")
    assert api.cancel_arguments == []

    adapter.cancel(_job("RUNNING"), namespace="org")
    assert api.cancel_arguments == [{"job_id": "a" * 24, "namespace": "org"}]


def test_cancel_treats_missing_job_as_already_cancelled() -> None:
    api = CancelErrorApi(_http_error(404))
    adapter = _adapter(api)

    adapter.cancel(_job("RUNNING"), namespace="org")

    assert api.cancel_arguments == [{"job_id": "a" * 24, "namespace": "org"}]


@pytest.mark.parametrize("status", [409, 500, 503])
def test_cancel_conflict_and_server_errors_are_ambiguous(status: int) -> None:
    adapter = _adapter(CancelErrorApi(_http_error(status)))

    with pytest.raises(
        AmbiguousActionOutcome,
        match=f"^HF Job cancellation outcome is ambiguous: HTTP {status}$",
    ):
        adapter.cancel(_job("RUNNING"), namespace="org")


@pytest.mark.parametrize("status", [400, 403, 499])
def test_cancel_client_errors_are_known_failures(status: int) -> None:
    adapter = _adapter(CancelErrorApi(_http_error(status)))

    with pytest.raises(
        ActionExecutionError,
        match=f"^HF Job cancellation failed: HTTP {status}$",
    ):
        adapter.cancel(_job("RUNNING"), namespace="org")


def test_cancel_transport_error_is_ambiguous_before_a_response() -> None:
    adapter = _adapter(CancelErrorApi(httpx.ConnectError("boom")))

    with pytest.raises(
        AmbiguousActionOutcome,
        match="^HF Job cancellation outcome is ambiguous before a response$",
    ):
        adapter.cancel(_job("RUNNING"), namespace="org")


@pytest.mark.parametrize(
    "error",
    [_http_error(500), httpx.ConnectError("boom")],
)
def test_find_wave_inspection_failures_are_known_failures(error: Exception) -> None:
    adapter = _adapter(ListErrorApi(error))

    with pytest.raises(ActionExecutionError, match="^HF Jobs inspection failed$"):
        adapter.find_wave(
            namespace="org",
            wave_id="wave-one",
            endpoint_label="endpoint-one",
        )


def test_find_wave_returns_none_when_no_jobs_match() -> None:
    api = FakeHfJobsApi([])
    adapter = _adapter(api)

    assert (
        adapter.find_wave(
            namespace="org",
            wave_id="wave-one",
            endpoint_label="endpoint-one",
        )
        is None
    )
    assert api.list_arguments == [
        {
            "labels": {
                "harbor-hf-wave": "wave-one",
                "harbor-hf-endpoint": "endpoint-one",
            },
            "namespace": "org",
        }
    ]


def test_find_wave_uses_provider_label_key_end_to_end() -> None:
    labels = {
        "harbor-hf-wave": "wave-one",
        "harbor-hf-provider": "provider-label",
    }
    resource = SimpleNamespace(
        id="b" * 24,
        labels=labels,
        status=SimpleNamespace(stage="RUNNING"),
    )
    api = FakeHfJobsApi([resource])
    adapter = _adapter(api)

    observed = adapter.find_wave(
        namespace="org",
        wave_id="wave-one",
        endpoint_label="provider-label",
        target_label_key="harbor-hf-provider",
    )

    assert api.list_arguments == [{"labels": labels, "namespace": "org"}]
    assert observed == RemoteWaveJob(
        job_id="b" * 24,
        wave_id="wave-one",
        endpoint_label="provider-label",
        target_label_key="harbor-hf-provider",
        stage="RUNNING",
    )


@pytest.mark.parametrize(
    ("resource", "message"),
    [
        (
            SimpleNamespace(id=None, labels={}, status="RUNNING"),
            "HF Job response has no valid ID",
        ),
        (
            SimpleNamespace(id="", labels={}, status="RUNNING"),
            "HF Job response has no valid ID",
        ),
        (
            SimpleNamespace(id=123, labels={}, status="RUNNING"),
            "HF Job response has no valid ID",
        ),
        (
            SimpleNamespace(id="a" * 24, labels=None, status="RUNNING"),
            "HF Job response has invalid labels",
        ),
        (
            SimpleNamespace(
                id="a" * 24,
                labels={"harbor-hf-wave": 7},
                status="RUNNING",
            ),
            "HF Job response has invalid labels",
        ),
        (
            SimpleNamespace(
                id="a" * 24,
                labels={
                    "harbor-hf-wave": "wave-one",
                    "harbor-hf-endpoint": "endpoint-one",
                },
                status=None,
            ),
            "HF Job response has no valid stage",
        ),
        (
            SimpleNamespace(
                id="a" * 24,
                labels={
                    "harbor-hf-wave": "wave-one",
                    "harbor-hf-endpoint": "endpoint-one",
                },
                status=SimpleNamespace(stage=""),
            ),
            "HF Job response has no valid stage",
        ),
    ],
)
def test_find_wave_rejects_invalid_remote_job_shapes(
    resource: SimpleNamespace, message: str
) -> None:
    adapter = _adapter(FakeHfJobsApi([resource]))

    with pytest.raises(ActionExecutionError, match=f"^{message}$"):
        adapter.find_wave(
            namespace="org",
            wave_id="wave-one",
            endpoint_label="endpoint-one",
        )


@pytest.mark.parametrize(
    "status",
    [
        "RUNNING",
        SimpleNamespace(stage=SimpleNamespace(value="RUNNING")),
    ],
)
def test_find_wave_accepts_plain_and_enum_stage_shapes(status: object) -> None:
    resource = SimpleNamespace(
        id="a" * 24,
        labels={
            "harbor-hf-wave": "wave-one",
            "harbor-hf-endpoint": "endpoint-one",
        },
        status=status,
    )
    adapter = _adapter(FakeHfJobsApi([resource]))

    observed = adapter.find_wave(
        namespace="org",
        wave_id="wave-one",
        endpoint_label="endpoint-one",
    )

    assert observed is not None
    assert observed.stage == "RUNNING"
    assert observed.terminal is False


def _managed_wave(
    remote_spec: ExperimentSpec,
) -> tuple[CampaignLock, bytes, WaveLock, EndpointRef]:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    endpoint = managed_wave_endpoint(lock, remote_spec, action.deployment_digest)
    wave = build_wave_lock(lock, remote_spec, action, endpoint=endpoint)
    return lock, request, wave, endpoint


def test_submit_process_error_is_ambiguous(remote_spec: ExperimentSpec) -> None:
    lock, request, wave, _endpoint = _managed_wave(remote_spec)
    adapter = HuggingFaceWaveJobAdapter(
        api=FakeHfJobsApi(),
        runner=FailingRunner(error=ProcessError("worker crashed")),
        bucket_api=FakeBucketApi(),
    )

    with pytest.raises(
        AmbiguousActionOutcome,
        match="^HF Jobs submission ended without a definitive outcome$",
    ):
        adapter.submit(wave, request=request, campaign=lock)


def test_submit_without_job_id_in_output_is_ambiguous(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, wave, _endpoint = _managed_wave(remote_spec)
    adapter = HuggingFaceWaveJobAdapter(
        api=FakeHfJobsApi(),
        runner=FailingRunner(output="no identifier in this output"),
        bucket_api=FakeBucketApi(),
    )

    with pytest.raises(
        AmbiguousActionOutcome,
        match="^HF Jobs wave submission did not return a job ID$",
    ):
        adapter.submit(wave, request=request, campaign=lock)


def test_submit_preflight_value_error_is_a_known_failure(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, wave, _endpoint = _managed_wave(remote_spec)

    class PublicBucketApi(FakeBucketApi):
        def bucket_info(self, bucket_id: str) -> object:
            return SimpleNamespace(id=bucket_id, private=False)

    adapter = HuggingFaceWaveJobAdapter(
        api=FakeHfJobsApi(),
        runner=InspectingRunner(),
        bucket_api=PublicBucketApi(),
    )

    with pytest.raises(
        ActionExecutionError,
        match="^Job input bucket osolmaz/jobs-artifacts must be private$",
    ):
        adapter.submit(wave, request=request, campaign=lock)


def test_submit_preflight_hub_error_is_a_known_failure(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, wave, _endpoint = _managed_wave(remote_spec)

    class BrokenRepoApi(FakeBucketApi):
        def create_repo(self, repo_id: str, **kwargs: object) -> object:
            raise _http_error(403)

    adapter = HuggingFaceWaveJobAdapter(
        api=FakeHfJobsApi(),
        runner=InspectingRunner(),
        bucket_api=BrokenRepoApi(),
    )

    with pytest.raises(
        ActionExecutionError, match="^HF Jobs submission preflight failed$"
    ):
        adapter.submit(wave, request=request, campaign=lock)


def test_submit_returns_exact_endpoint_wave_identity(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, wave, endpoint = _managed_wave(remote_spec)
    adapter = HuggingFaceWaveJobAdapter(
        api=FakeHfJobsApi(),
        runner=InspectingRunner(),
        bucket_api=FakeBucketApi(),
    )

    job = adapter.submit(wave, request=request, campaign=lock)

    assert job == RemoteWaveJob(
        job_id="0123456789abcdef01234567",
        wave_id=wave.wave_id,
        endpoint_label=endpoint_lease_label_for(endpoint.namespace, endpoint.name),
        target_label_key="harbor-hf-endpoint",
        stage="SCHEDULING",
    )


def test_submit_returns_exact_provider_wave_identity(
    remote_spec: ExperimentSpec,
) -> None:
    spec, target = _provider_spec(remote_spec)
    lock, request, submitted = _campaign(spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    wave = build_wave_lock(lock, spec, action)
    bucket_api = FakeBucketApi()
    adapter = HuggingFaceWaveJobAdapter(
        api=FakeHfJobsApi(),
        runner=InspectingRunner(),
        bucket_api=bucket_api,
    )

    job = adapter.submit(wave, request=request, campaign=lock)

    assert job == RemoteWaveJob(
        job_id="0123456789abcdef01234567",
        wave_id=wave.wave_id,
        endpoint_label=hashlib.sha256(target.service.encode()).hexdigest()[:32],
        target_label_key="harbor-hf-provider",
        stage="SCHEDULING",
    )
    staged = {
        path.rsplit("/", 1)[-1]: content for path, content in bucket_api.staged.items()
    }
    assert staged["wave.lock.json"] == (
        json.dumps(wave.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert staged["campaign.lock.json"] == (
        json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_reservation_validation_rejects_malformed_records(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    store.reservations = {"broken": {"kind": "submit-wave"}}

    with pytest.raises(CampaignApplyError, match="^action reservation is malformed$"):
        _reconciler(store, FakeEndpoints(), FakeJobs()).apply_campaign(lock.campaign_id)


def test_reservation_validation_rejects_foreign_campaign_identity(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    record = action.model_dump(mode="json")
    record["campaign_id"] = "campaign-other"
    store = FakeStore(lock, request, [submitted])
    store.reservations = {action.action_id: record}

    with pytest.raises(
        CampaignApplyError, match="^action reservation identity is malformed$"
    ):
        _reconciler(store, FakeEndpoints(), FakeJobs()).apply_campaign(lock.campaign_id)


def test_reservation_validation_rejects_mismatched_action_id(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    record = action.model_dump(mode="json")
    record["action_id"] = "act-" + "f" * 24
    store = FakeStore(lock, request, [submitted])
    store.reservations = {action.action_id: record}

    with pytest.raises(
        CampaignApplyError, match="^action reservation identity is malformed$"
    ):
        _reconciler(store, FakeEndpoints(), FakeJobs()).apply_campaign(lock.campaign_id)


def test_reservation_validation_rejects_duplicated_action_ids(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    record = action.model_dump(mode="json")
    store = FakeStore(lock, request, [submitted])
    store.reservations = {"first": record, "second": dict(record)}

    with pytest.raises(
        CampaignApplyError, match="^action reservation ID is duplicated$"
    ):
        _reconciler(store, FakeEndpoints(), FakeJobs()).apply_campaign(lock.campaign_id)


def test_pending_action_without_reservation_record_fails_closed(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    store = FakeStore(lock, request, [submitted, _reservation(lock, action, 2)])

    with pytest.raises(
        CampaignApplyError,
        match=f"^action event has no reservation record: {action.action_id}$",
    ):
        _reconciler(store, FakeEndpoints(), FakeJobs()).apply_campaign(lock.campaign_id)


def test_pending_action_with_mismatched_reservation_fails_closed(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = plan_reconciliation(lock, [submitted], now=NOW)[1].actions[0]
    record = action.model_dump(mode="json")
    record["kind"] = "cancel-wave"
    store = FakeStore(lock, request, [submitted, _reservation(lock, action, 2)])
    store.reservations = {action.action_id: record}

    with pytest.raises(
        CampaignApplyError,
        match=f"^action reservation does not match its event: {action.action_id}$",
    ):
        _reconciler(store, FakeEndpoints(), FakeJobs()).apply_campaign(lock.campaign_id)


def test_unsupported_reserved_action_kind_records_failed_outcome(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = ReconcileAction(
        action_id="act-" + "5" * 24,
        action_key="5" * 24,
        kind="manual-intervention",
        campaign_id=lock.campaign_id,
        deployment_digest=lock.runs[0].deployment_digest,
        wave_id="wave-" + "1" * 24,
        target_ids=["wave-" + "1" * 24],
    )
    store = FakeStore(lock, request, [submitted, _reservation(lock, action, 2)])
    store.reservations = {action.action_id: action.model_dump(mode="json")}
    jobs = FakeJobs()

    result = _reconciler(store, FakeEndpoints(), jobs).apply_campaign(lock.campaign_id)

    outcome = next(
        applied for applied in result.applied if applied.action_id == action.action_id
    )
    assert outcome.status == "failed"
    assert outcome.message == (
        "action execution is not supported by configured adapters: manual-intervention"
    )
    assert action.wave_id not in {call["wave_id"] for call in jobs.find_calls}


def test_adopted_wave_without_managed_endpoint_records_failed_outcome(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    endpoints = FakeEndpoints()
    jobs = FakeJobs()
    jobs.adopt_on_find = True

    result = _reconciler(store, endpoints, jobs).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "failed"
    assert result.applied[0].message == (
        "managed wave Job exists without its managed endpoint"
    )
    assert len(endpoints.inspect_calls) == 1
    assert endpoints.create_calls == []
    assert jobs.submissions == []


def test_ambiguous_endpoint_create_records_ambiguous_outcome(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])
    endpoints = FakeEndpoints()
    endpoints.create_error = AmbiguousEndpointCreate("create may have applied")
    jobs = FakeJobs()

    result = _reconciler(store, endpoints, jobs).apply_campaign(lock.campaign_id)

    assert result.applied[0].status == "ambiguous"
    assert result.applied[0].message == "create may have applied"
    assert store.events[-1].kind == "action.ambiguous"
    assert jobs.submissions == []


def test_cleanup_verification_timeout_records_ambiguous_outcome(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = ReconcileAction(
        action_id="act-" + "6" * 24,
        action_key="6" * 24,
        kind="cleanup-wave",
        campaign_id=lock.campaign_id,
        deployment_digest=lock.runs[0].deployment_digest,
        wave_id="wave-" + "1" * 24,
        target_ids=["wave-" + "1" * 24],
    )
    store = FakeStore(lock, request, [submitted, _reservation(lock, action, 2)])
    store.reservations = {action.action_id: action.model_dump(mode="json")}
    endpoints = FakeEndpoints()
    endpoints.present = True
    endpoints.pause_error = EndpointVerificationTimeout("did not converge")

    result = _reconciler(store, endpoints, FakeJobs()).apply_campaign(lock.campaign_id)

    outcome = next(
        applied for applied in result.applied if applied.action_id == action.action_id
    )
    assert outcome.status == "ambiguous"
    assert outcome.message == "did not converge"


def test_cancel_wave_with_unresolvable_deployment_digest_fails(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = ReconcileAction(
        action_id="act-" + "7" * 24,
        action_key="7" * 24,
        kind="cancel-wave",
        campaign_id=lock.campaign_id,
        deployment_digest="0" * 64,
        wave_id="wave-" + "1" * 24,
        target_ids=["wave-" + "1" * 24],
    )
    store = FakeStore(lock, request, [submitted, _reservation(lock, action, 2)])
    store.reservations = {action.action_id: action.model_dump(mode="json")}
    jobs = FakeJobs()

    result = _reconciler(store, FakeEndpoints(), jobs).apply_campaign(lock.campaign_id)

    outcome = next(
        applied for applied in result.applied if applied.action_id == action.action_id
    )
    assert outcome.status == "failed"
    assert outcome.message == (
        "action deployment does not resolve to one deployment target"
    )
    assert action.wave_id not in {call["wave_id"] for call in jobs.find_calls}


def test_cleanup_wave_with_unresolvable_deployment_digest_fails(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = ReconcileAction(
        action_id="act-" + "8" * 24,
        action_key="8" * 24,
        kind="cleanup-wave",
        campaign_id=lock.campaign_id,
        deployment_digest="0" * 64,
        wave_id="wave-" + "1" * 24,
        target_ids=["wave-" + "1" * 24],
    )
    store = FakeStore(lock, request, [submitted, _reservation(lock, action, 2)])
    store.reservations = {action.action_id: action.model_dump(mode="json")}
    endpoints = FakeEndpoints()

    result = _reconciler(store, endpoints, FakeJobs()).apply_campaign(lock.campaign_id)

    outcome = next(
        applied for applied in result.applied if applied.action_id == action.action_id
    )
    assert outcome.status == "failed"
    assert outcome.message == (
        "action deployment does not resolve to one deployment target"
    )
    assert endpoints.inspect_calls == []


def test_cancel_wave_without_remote_job_succeeds_with_no_remote_id(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    action = ReconcileAction(
        action_id="act-" + "9" * 24,
        action_key="9" * 24,
        kind="cancel-wave",
        campaign_id=lock.campaign_id,
        deployment_digest=lock.runs[0].deployment_digest,
        wave_id="wave-" + "1" * 24,
        target_ids=["wave-" + "1" * 24],
    )
    store = FakeStore(lock, request, [submitted, _reservation(lock, action, 2)])
    store.reservations = {action.action_id: action.model_dump(mode="json")}
    jobs = FakeJobs()

    result = _reconciler(store, FakeEndpoints(), jobs).apply_campaign(lock.campaign_id)

    outcome = next(
        applied for applied in result.applied if applied.action_id == action.action_id
    )
    assert outcome.status == "succeeded"
    assert outcome.remote_id is None
    assert jobs.cancellations == []


def test_outcome_events_advance_monotonically_past_a_frozen_clock(
    remote_spec: ExperimentSpec,
) -> None:
    lock, request, submitted = _campaign(remote_spec)
    store = FakeStore(lock, request, [submitted])

    _reconciler(store, FakeEndpoints(), FakeJobs()).apply_campaign(lock.campaign_id)

    reserved, succeeded = store.events[-2:]
    assert reserved.kind == "action.reserved"
    assert succeeded.kind == "action.succeeded"
    assert reserved.observed_at == NOW + timedelta(microseconds=1)
    assert succeeded.observed_at == NOW + timedelta(microseconds=2)
