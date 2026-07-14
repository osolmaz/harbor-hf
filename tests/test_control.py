from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError
from pydantic import JsonValue, ValidationError

from harbor_hf.campaigns import CampaignLock, build_campaign_lock, build_campaign_plan
from harbor_hf.control import (
    ActionKind,
    ActionOutcomePayload,
    ActionReservedPayload,
    CampaignConflict,
    CampaignEvent,
    CampaignSnapshot,
    CampaignSubmittedPayload,
    CancellationPayload,
    ControlError,
    EventKind,
    HubCampaignStore,
    LifecyclePayload,
    TerminalPayload,
    new_event,
    project_campaign,
)
from harbor_hf.models import ExperimentSpec

NOW = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)


class FakeCampaignApi:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.generation = 1
        self.files: dict[str, bytes] = {}
        self.last_commits: dict[str, str] = {}
        self.conflicts = 0
        self.commits: list[dict[str, object]] = []
        self.info_calls: list[tuple[str, dict[str, object]]] = []
        self.list_calls: list[tuple[str, dict[str, object]]] = []
        self.download_calls: list[tuple[str, str, dict[str, object]]] = []

    @property
    def head(self) -> str:
        return f"{self.generation:040x}"

    def repo_info(self, repo_id: str, **kwargs: object) -> object:
        self.info_calls.append((repo_id, kwargs))
        assert repo_id == "org/harbor-hf-coordination"
        assert kwargs == {"repo_type": "dataset", "revision": "main"}
        return SimpleNamespace(sha=self.head)

    def get_paths_info(
        self, repo_id: str, paths: list[str] | str, **kwargs: object
    ) -> list[object]:
        assert repo_id == "org/harbor-hf-coordination"
        expand = kwargs.pop("expand", False)
        assert kwargs == {"repo_type": "dataset", "revision": self.head}
        path = paths if isinstance(paths, str) else paths[0]
        if path not in self.files:
            return []
        last_commit = SimpleNamespace(oid=self.last_commits[path]) if expand else None
        return [SimpleNamespace(path=path, last_commit=last_commit)]

    def list_repo_files(self, repo_id: str, **kwargs: object) -> list[str]:
        self.list_calls.append((repo_id, kwargs))
        assert repo_id == "org/harbor-hf-coordination"
        assert kwargs == {"repo_type": "dataset", "revision": self.head}
        return list(self.files)

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: object) -> str:
        self.download_calls.append((repo_id, filename, kwargs))
        assert repo_id == "org/harbor-hf-coordination"
        assert kwargs == {"repo_type": "dataset", "revision": self.head}
        destination = self.root / filename.replace("/", "-")
        destination.write_bytes(self.files[filename])
        return str(destination)

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object:
        assert repo_id == "org/harbor-hf-coordination"
        if self.conflicts:
            self.conflicts -= 1
            self.generation += 1
            raise _http_error(409)
        assert kwargs["parent_commit"] == self.head
        for operation in operations:
            assert isinstance(operation, CommitOperationAdd)
            payload = operation.path_or_fileobj
            assert isinstance(payload, bytes)
            self.files[operation.path_in_repo] = payload
            self.last_commits[operation.path_in_repo] = f"{self.generation + 1:040x}"
        self.commits.append(kwargs)
        self.generation += 1
        return SimpleNamespace(oid=self.head)


def _http_error(status: int) -> HfHubHTTPError:
    request = httpx.Request("POST", "https://huggingface.co/api/datasets")
    return HfHubHTTPError(
        "commit failed", response=httpx.Response(status, request=request)
    )


def _lock(remote_spec: ExperimentSpec) -> CampaignLock:
    return build_campaign_lock(
        build_campaign_plan(remote_spec), "campaign-one", clock=lambda: NOW
    )


def _submitted(lock: CampaignLock) -> CampaignEvent:
    return new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=lock.plan_digest),
        clock=lambda: NOW,
        identifier=lambda: "1" * 32,
    )


def test_event_payload_must_match_kind() -> None:
    with pytest.raises(ValidationError, match="payload does not match"):
        CampaignEvent(
            event_id="evt-" + "1" * 32,
            subject_type="campaign",
            subject_id="campaign",
            kind="campaign.submitted",
            observed_at=NOW,
            producer="cli",
            payload=TerminalPayload(message="wrong"),
        )


def test_projection_tracks_reserved_and_completed_actions(
    remote_spec: ExperimentSpec,
) -> None:
    lock = _lock(remote_spec)
    reserved = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.reserved",
        producer="reconciler",
        payload=ActionReservedPayload(
            action_id="act-one",
            action_key="key-one",
            action_kind="submit-wave",
            target_ids=[lock.runs[0].shards[0].shard_id],
        ),
        clock=lambda: NOW + timedelta(seconds=1),
        identifier=lambda: "2" * 32,
    )
    succeeded = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.succeeded",
        producer="reconciler",
        payload=ActionOutcomePayload(action_id="act-one", remote_id="job-one"),
        clock=lambda: NOW + timedelta(seconds=2),
        identifier=lambda: "3" * 32,
    )

    projection = project_campaign(
        lock, [succeeded, _submitted(lock), reserved, reserved]
    )

    assert projection.status == "active"
    assert projection.event_count == 3
    assert projection.actions["act-one"].status == "succeeded"
    assert projection.actions["act-one"].remote_id == "job-one"


@pytest.mark.parametrize(
    "events,message",
    [
        ([], "no submission"),
        ("outcome", "no reservation"),
        ("terminal", "after a terminal"),
    ],
)
def test_projection_rejects_invalid_history(
    remote_spec: ExperimentSpec, events: object, message: str
) -> None:
    lock = _lock(remote_spec)
    supplied: list[CampaignEvent]
    if events == "outcome":
        supplied = [
            _submitted(lock),
            new_event(
                subject_type="campaign",
                subject_id=lock.campaign_id,
                kind="action.failed",
                producer="reconciler",
                payload=ActionOutcomePayload(action_id="missing"),
                clock=lambda: NOW + timedelta(seconds=1),
                identifier=lambda: "2" * 32,
            ),
        ]
    elif events == "terminal":
        supplied = [
            _submitted(lock),
            new_event(
                subject_type="campaign",
                subject_id=lock.campaign_id,
                kind="campaign.completed",
                producer="reconciler",
                payload=TerminalPayload(message="done"),
                clock=lambda: NOW + timedelta(seconds=1),
                identifier=lambda: "2" * 32,
            ),
            new_event(
                subject_type="campaign",
                subject_id=lock.campaign_id,
                kind="campaign.failed",
                producer="reconciler",
                payload=TerminalPayload(message="late"),
                clock=lambda: NOW + timedelta(seconds=2),
                identifier=lambda: "3" * 32,
            ),
        ]
    else:
        supplied = []

    with pytest.raises(ControlError, match=message):
        project_campaign(lock, supplied)


def test_hub_store_creates_and_loads_campaign(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    api.conflicts = 1
    store = HubCampaignStore("org", api=api)

    store.create_campaign(lock, b"kind: Experiment\n", _submitted(lock))
    observed_lock, events = store.load_campaign(lock.campaign_id)

    assert observed_lock == lock
    assert events == [_submitted(lock)]
    assert api.files[f"campaigns/{lock.campaign_id}/request.yaml"] == (
        b"kind: Experiment\n"
    )
    assert len(api.commits) == 1
    snapshot = store.load_snapshot(lock.campaign_id)
    assert snapshot.lock == lock
    assert snapshot.request == b"kind: Experiment\n"
    assert (
        snapshot.control_commit
        == api.last_commits[f"campaigns/{lock.campaign_id}/campaign.lock.json"]
    )
    with pytest.raises(CampaignConflict, match="campaign already exists"):
        store.create_campaign(lock, b"different", _submitted(lock))


def test_hub_store_snapshot_reads_every_object_from_one_exact_revision(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    store = HubCampaignStore("org", api=api)
    request = b"kind: Experiment\nmetadata: snapshot\n"
    store.create_campaign(lock, request, _submitted(lock))
    later = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.cancel-requested",
        producer="cli",
        payload=CancellationPayload(reason="operator"),
        clock=lambda: NOW + timedelta(seconds=1),
        identifier=lambda: "0" * 32,
    )
    api.files[f"campaigns/{lock.campaign_id}/events/{later.event_id}.json"] = (
        json.dumps(later.model_dump(mode="json"), default=str).encode()
    )
    api.files[f"campaigns/{lock.campaign_id}/events/ignored.txt"] = b"not json"
    api.info_calls.clear()
    api.list_calls.clear()
    api.download_calls.clear()
    expected_head = api.head
    expected_control_commit = api.last_commits[
        f"campaigns/{lock.campaign_id}/campaign.lock.json"
    ]

    snapshot = store.load_snapshot(lock.campaign_id)

    repository = "org/harbor-hf-coordination"
    revision = {"repo_type": "dataset", "revision": expected_head}
    assert snapshot == CampaignSnapshot(
        lock=lock,
        events=[later, _submitted(lock)],
        request=request,
        control_commit=expected_control_commit,
    )
    assert api.info_calls == [
        (repository, {"repo_type": "dataset", "revision": "main"})
    ]
    assert api.list_calls == [(repository, revision)]
    assert api.download_calls == [
        (
            repository,
            f"campaigns/{lock.campaign_id}/campaign.lock.json",
            revision,
        ),
        (
            repository,
            f"campaigns/{lock.campaign_id}/events/{later.event_id}.json",
            revision,
        ),
        (
            repository,
            f"campaigns/{lock.campaign_id}/events/{_submitted(lock).event_id}.json",
            revision,
        ),
        (
            repository,
            f"campaigns/{lock.campaign_id}/request.yaml",
            revision,
        ),
    ]


def test_hub_store_reads_requests_lists_campaigns_and_loads_reservations(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    store = HubCampaignStore("org", api=api)
    store.create_campaign(lock, b"kind: Experiment\n", _submitted(lock))
    reserved = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.reserved",
        producer="reconciler",
        payload=ActionReservedPayload(
            action_id="act-one",
            action_key="key-one",
            action_kind="submit-wave",
            target_ids=["shard-one"],
        ),
        clock=lambda: NOW + timedelta(seconds=1),
        identifier=lambda: "2" * 32,
    )
    action: dict[str, JsonValue] = {
        "action_id": "act-one",
        "kind": "submit-wave",
    }
    store.reserve_action(lock.campaign_id, action, reserved)
    api.files["campaigns/nested/invalid/campaign.lock.json"] = b"{}"

    assert store.load_request(lock.campaign_id) == b"kind: Experiment\n"
    assert store.list_campaigns() == [lock.campaign_id]
    assert store.load_action_reservations(lock.campaign_id) == [action]


def test_hub_store_reserves_action_atomically_and_idempotently(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    store = HubCampaignStore("org", api=api)
    store.create_campaign(lock, b"manifest", _submitted(lock))
    event = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.reserved",
        producer="reconciler",
        payload=ActionReservedPayload(
            action_id="act-one",
            action_key="key-one",
            action_kind="submit-wave",
            target_ids=["shard-one"],
        ),
        identifier=lambda: "4" * 32,
    )
    action = {"action_id": "act-one", "kind": "submit-wave"}

    assert store.reserve_action(lock.campaign_id, action, event)
    assert not store.reserve_action(lock.campaign_id, action, event)
    conflicting = {"action_id": "act-one", "kind": "cancel-wave"}
    with pytest.raises(CampaignConflict, match="content conflicts"):
        store.reserve_action(lock.campaign_id, conflicting, event)


def test_hub_store_appends_event_and_rejects_duplicate(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    store = HubCampaignStore("org", api=FakeCampaignApi(tmp_path))
    store.create_campaign(lock, b"manifest", _submitted(lock))
    event = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.failed",
        producer="reconciler",
        payload=TerminalPayload(message="failed"),
        identifier=lambda: "5" * 32,
    )

    store.append_event(lock.campaign_id, event)
    _observed_lock, events = store.load_campaign(lock.campaign_id)

    assert event in events
    with pytest.raises(CampaignConflict, match="event already exists"):
        store.append_event(lock.campaign_id, event)


def test_hub_store_ensures_identical_event_once(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    store = HubCampaignStore("org", api=FakeCampaignApi(tmp_path))
    store.create_campaign(lock, b"manifest", _submitted(lock))
    event = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.cancel-requested",
        producer="cli",
        payload=CancellationPayload(reason="operator"),
        identifier=lambda: "6" * 32,
    )

    assert store.ensure_event(lock.campaign_id, event)
    assert not store.ensure_event(lock.campaign_id, event)
    conflicting = event.model_copy(
        update={"payload": CancellationPayload(reason="different")}
    )
    with pytest.raises(CampaignConflict, match="event conflicts"):
        store.ensure_event(lock.campaign_id, conflicting)


def test_hub_store_reports_malformed_records(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    store = HubCampaignStore("org", api=api)
    store.create_campaign(lock, b"manifest", _submitted(lock))
    api.files[f"campaigns/{lock.campaign_id}/campaign.lock.json"] = b"not-json"

    with pytest.raises(ControlError, match="cannot be read"):
        store.load_campaign(lock.campaign_id)


def test_hub_store_non_conflict_error_is_not_retried(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    class FailingApi(FakeCampaignApi):
        def create_commit(
            self, repo_id: str, operations: list[object], **kwargs: object
        ) -> object:
            raise _http_error(500)

    lock = _lock(remote_spec)

    with pytest.raises(HfHubHTTPError):
        HubCampaignStore("org", api=FailingApi(tmp_path)).create_campaign(
            lock, b"manifest", _submitted(lock)
        )


def test_hub_store_files_are_canonical_json(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    HubCampaignStore("org", api=api).create_campaign(
        lock, b"manifest", _submitted(lock)
    )
    raw = api.files[f"campaigns/{lock.campaign_id}/campaign.lock.json"]

    assert raw.endswith(b"\n")
    assert json.loads(raw) == lock.model_dump(mode="json")


def test_campaign_projection_corpus_is_stable(remote_spec: ExperimentSpec) -> None:
    lock = _lock(remote_spec)
    submitted = _submitted(lock)
    events: list[CampaignEvent] = [submitted]
    outcomes = ["action.succeeded", "action.failed", "action.ambiguous"]
    action_kinds = ["submit-wave", "cancel-wave", "publish-results"]
    for index, (outcome, action_kind) in enumerate(
        zip(outcomes, action_kinds, strict=True), 2
    ):
        action_id = f"action-{index}"
        events.append(
            new_event(
                subject_type="campaign",
                subject_id=lock.campaign_id,
                kind="action.reserved",
                producer="reconciler",
                payload=ActionReservedPayload(
                    action_id=action_id,
                    action_key=f"key-{index}",
                    action_kind=cast(ActionKind, action_kind),
                    target_ids=[f"target-{index}"],
                ),
                clock=lambda index=index: NOW + timedelta(seconds=index),
                identifier=lambda index=index: f"{index:032x}",
            )
        )
        events.append(
            new_event(
                subject_type="campaign",
                subject_id=lock.campaign_id,
                kind=cast(EventKind, outcome),
                producer="reconciler",
                payload=ActionOutcomePayload(
                    action_id=action_id,
                    message=f"message-{index}",
                    remote_id=f"remote-{index}",
                ),
                clock=lambda index=index: NOW + timedelta(seconds=index + 10),
                identifier=lambda index=index: f"{index + 10:032x}",
            )
        )
    active = project_campaign(lock, list(reversed(events)))
    cancelled = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.cancel-requested",
        producer="cli",
        payload=CancellationPayload(reason="operator"),
        clock=lambda: NOW + timedelta(seconds=2),
        identifier=lambda: "a" * 32,
    )
    draining = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.draining",
        producer="reconciler",
        payload=LifecyclePayload(message="draining"),
        clock=lambda: NOW + timedelta(seconds=3),
        identifier=lambda: "b" * 32,
    )
    manual = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.manual-intervention-required",
        producer="reconciler",
        payload=LifecyclePayload(message="cleanup failed"),
        clock=lambda: NOW + timedelta(seconds=4),
        identifier=lambda: "c" * 32,
    )
    projections = [
        active,
        project_campaign(lock, [submitted, cancelled, cancelled]),
        project_campaign(lock, [submitted, cancelled, draining, cancelled]),
        project_campaign(lock, [submitted, cancelled, draining, manual, cancelled]),
    ]
    encoded = json.dumps(
        [projection.model_dump(mode="json") for projection in projections],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert hashlib.sha256(encoded).hexdigest() == (
        "6fdb16c9dd54315174dbbf4ec9e7409f562f597271556f41fc0ba3bb78056e6a"
    )


def test_control_store_commit_corpus_is_stable(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    store = HubCampaignStore("org", api=api)
    submitted = _submitted(lock)
    reserved = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.reserved",
        producer="reconciler",
        payload=ActionReservedPayload(
            action_id="action-one",
            action_key="key-one",
            action_kind="submit-wave",
            target_ids=["shard-one"],
        ),
        clock=lambda: NOW + timedelta(seconds=1),
        identifier=lambda: "d" * 32,
    )
    outcome = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.succeeded",
        producer="reconciler",
        payload=ActionOutcomePayload(
            action_id="action-one", message="created", remote_id="remote-one"
        ),
        clock=lambda: NOW + timedelta(seconds=2),
        identifier=lambda: "e" * 32,
    )
    action = {
        "action_id": "action-one",
        "kind": "submit-wave",
        "targets": ["shard-one"],
    }

    store.create_campaign(lock, b"kind: Experiment\n", submitted)
    assert store.reserve_action(lock.campaign_id, action, reserved)
    store.append_event(lock.campaign_id, outcome)
    observed_lock, observed_events = store.load_campaign(lock.campaign_id)
    corpus = {
        "files": {
            path: payload.decode() for path, payload in sorted(api.files.items())
        },
        "commits": api.commits,
        "lock": observed_lock.model_dump(mode="json"),
        "events": [event.model_dump(mode="json") for event in observed_events],
    }
    encoded = json.dumps(
        corpus, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()

    assert hashlib.sha256(encoded).hexdigest() == (
        "5f7bfaefd5c5a0d85911efdf7e743ce03966690b8174d51fc80d1aada491859f"
    )


def test_control_projection_rejects_each_submission_mismatch(
    remote_spec: ExperimentSpec,
) -> None:
    lock = _lock(remote_spec)
    submitted = _submitted(lock)
    mismatches = [
        submitted.model_copy(update={"kind": "campaign.cancel-requested"}),
        submitted.model_copy(update={"subject_type": "run"}),
        submitted.model_copy(update={"subject_id": "different-campaign"}),
        submitted.model_copy(
            update={
                "payload": CampaignSubmittedPayload(plan_digest="sha256:" + "f" * 64)
            }
        ),
    ]

    for mismatch in mismatches:
        with pytest.raises(ControlError) as captured:
            project_campaign(lock, [mismatch])
        assert (
            str(captured.value) == "campaign submission event does not match its lock"
        )
    with pytest.raises(ControlError) as captured:
        project_campaign(lock, [])
    assert str(captured.value) == "campaign has no submission event"


def test_action_reservation_validates_kind_and_campaign_separately(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    store = HubCampaignStore("org", api=FakeCampaignApi(tmp_path))
    outcome = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.succeeded",
        producer="reconciler",
        payload=ActionOutcomePayload(action_id="action-one"),
        clock=lambda: NOW,
        identifier=lambda: "f" * 32,
    )
    reservation = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.reserved",
        producer="reconciler",
        payload=ActionReservedPayload(
            action_id="action-one",
            action_key="key-one",
            action_kind="submit-wave",
            target_ids=[],
        ),
        clock=lambda: NOW,
        identifier=lambda: "e" * 32,
    )

    for invalid in [outcome, reservation.model_copy(update={"subject_id": "wrong"})]:
        with pytest.raises(ValueError) as captured:
            store.reserve_action(lock.campaign_id, {}, invalid)
        assert str(captured.value) == (
            "action reservation requires its reservation event"
        )


def test_hub_store_ensure_event_retries_parent_conflicts_with_exact_commit(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    api.conflicts = 2
    store = HubCampaignStore("org", api=api)
    event = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.cancel-requested",
        producer="cli",
        payload=CancellationPayload(reason="operator"),
        clock=lambda: NOW,
        identifier=lambda: "7" * 32,
    )

    assert store.ensure_event(lock.campaign_id, event) is True
    assert len(api.info_calls) == 3
    assert api.commits == [
        {
            "commit_message": "chore: record campaign.cancel-requested",
            "repo_type": "dataset",
            "revision": "main",
            "parent_commit": f"{3:040x}",
        }
    ]
    event_path = f"campaigns/{lock.campaign_id}/events/{event.event_id}.json"
    assert json.loads(api.files[event_path]) == event.model_dump(mode="json")


def test_hub_store_ensure_event_has_bounded_parent_conflict_retries(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    api.conflicts = 20
    store = HubCampaignStore("org", api=api)

    with pytest.raises(ControlError) as captured:
        store.ensure_event(lock.campaign_id, _submitted(lock))

    assert str(captured.value) == "control repository remained contended"
    assert len(api.info_calls) == 8
    assert api.commits == []


def test_hub_store_reservation_commit_contains_action_and_event_atomically(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    store = HubCampaignStore("org", api=api)
    event = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.reserved",
        producer="reconciler",
        payload=ActionReservedPayload(
            action_id="act-atomic",
            action_key="atomic-key",
            action_kind="retry-shard",
            target_ids=["trial-one"],
        ),
        clock=lambda: NOW,
        identifier=lambda: "8" * 32,
    )
    action: dict[str, JsonValue] = {
        "action_id": "act-atomic",
        "kind": "retry-shard",
        "trial_ids": ["trial-one"],
    }

    assert store.reserve_action(lock.campaign_id, action, event) is True

    reservation_path = f"campaigns/{lock.campaign_id}/reservations/act-atomic.json"
    event_path = f"campaigns/{lock.campaign_id}/events/{event.event_id}.json"
    assert set(api.files) == {reservation_path, event_path}
    assert json.loads(api.files[reservation_path]) == action
    assert json.loads(api.files[event_path]) == event.model_dump(mode="json")
    assert api.commits == [
        {
            "commit_message": "chore: reserve act-atomic",
            "repo_type": "dataset",
            "revision": "main",
            "parent_commit": f"{1:040x}",
        }
    ]
    conflicting = {**action, "kind": "submit-wave"}
    with pytest.raises(CampaignConflict) as captured:
        store.reserve_action(lock.campaign_id, conflicting, event)
    assert str(captured.value) == "action reservation content conflicts"


def test_hub_store_reservation_retries_only_parent_conflicts(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    event = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.reserved",
        producer="reconciler",
        payload=ActionReservedPayload(
            action_id="act-retry",
            action_key="retry-key",
            action_kind="submit-wave",
            target_ids=[],
        ),
        clock=lambda: NOW,
        identifier=lambda: "9" * 32,
    )
    contended = FakeCampaignApi(tmp_path)
    contended.conflicts = 20

    with pytest.raises(ControlError) as captured:
        HubCampaignStore("org", api=contended).reserve_action(
            lock.campaign_id, {"action_id": "act-retry"}, event
        )

    assert str(captured.value) == "control repository remained contended"
    assert len(contended.info_calls) == 8

    class FailingApi(FakeCampaignApi):
        def create_commit(
            self, repo_id: str, operations: list[object], **kwargs: object
        ) -> object:
            raise _http_error(500)

    with pytest.raises(HfHubHTTPError):
        HubCampaignStore("org", api=FailingApi(tmp_path)).reserve_action(
            lock.campaign_id, {"action_id": "act-retry"}, event
        )


def test_hub_store_rejects_non_object_action_reservation(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    path = f"campaigns/{lock.campaign_id}/reservations/act-invalid.json"
    api.files[path] = b"[]"
    store = HubCampaignStore("org", api=api)

    with pytest.raises(ControlError) as captured:
        store.load_action_reservations(lock.campaign_id)

    assert str(captured.value) == f"action reservation must be an object: {path}"


def test_hub_store_snapshot_requires_lock_commit_identity(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _lock(remote_spec)
    api = FakeCampaignApi(tmp_path)
    store = HubCampaignStore("org", api=api)
    store.create_campaign(lock, b"manifest", _submitted(lock))
    lock_path = f"campaigns/{lock.campaign_id}/campaign.lock.json"
    api.last_commits[lock_path] = ""

    with pytest.raises(ControlError) as captured:
        store.load_snapshot(lock.campaign_id)

    assert str(captured.value) == (
        f"control record has no immutable commit: {lock_path}"
    )


def test_hub_store_snapshot_requires_one_lock_commit_record(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    class MissingCommitApi(FakeCampaignApi):
        def get_paths_info(
            self, repo_id: str, paths: list[str] | str, **kwargs: object
        ) -> list[object]:
            if kwargs.get("expand") is True:
                return []
            return super().get_paths_info(repo_id, paths, **kwargs)

    lock = _lock(remote_spec)
    api = MissingCommitApi(tmp_path)
    store = HubCampaignStore("org", api=api)
    store.create_campaign(lock, b"manifest", _submitted(lock))
    lock_path = f"campaigns/{lock.campaign_id}/campaign.lock.json"

    with pytest.raises(ControlError) as captured:
        store.load_snapshot(lock.campaign_id)

    assert str(captured.value) == (
        f"control record has no immutable commit: {lock_path}"
    )
