from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from harbor_hf.campaigns import CampaignLock
from harbor_hf.coordination import coordination_repository

_MAX_COMMIT_ATTEMPTS = 8
_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")

SubjectType = Literal["campaign", "run", "shard", "trial", "execution", "wave"]
Producer = Literal["cli", "reconciler", "wave-controller", "watchdog", "publisher"]
RetryCategory = Literal[
    "lost",
    "transient",
    "quota",
    "rate-limit",
    "ambiguous",
    "benchmark",
    "configuration",
    "authentication",
    "cleanup",
]
EventKind = Literal[
    "campaign.submitted",
    "campaign.cancel-requested",
    "campaign.shard-retry-requested",
    "campaign.draining",
    "campaign.manual-intervention-required",
    "campaign.completed",
    "campaign.partial",
    "campaign.failed",
    "campaign.cancelled",
    "run.queued",
    "run.active",
    "run.verifying",
    "run.publishing",
    "run.complete",
    "run.invalid",
    "run.failed-infrastructure",
    "run.cancelled",
    "shard.queued",
    "shard.active",
    "shard.verifying",
    "shard.publishing",
    "shard.complete",
    "shard.invalid",
    "shard.failed-infrastructure",
    "shard.cancelled",
    "trial.complete",
    "trial.invalid",
    "trial.cancelled",
    "execution.started",
    "execution.completed",
    "execution.failed",
    "execution.cancelled",
    "wave.acquiring",
    "wave.provisioning",
    "wave.ready",
    "wave.active",
    "wave.draining",
    "wave.cleaning",
    "wave.closed",
    "wave.cleanup-failed",
    "spend.recorded",
    "action.reserved",
    "action.succeeded",
    "action.failed",
    "action.ambiguous",
]
ActionKind = Literal[
    "submit-wave",
    "retry-shard",
    "cancel-execution",
    "cancel-wave",
    "drain-wave",
    "cleanup-wave",
    "publish-results",
    "publish-summary",
    "manual-intervention",
]


class ControlError(RuntimeError):
    """Raised when durable campaign state cannot be safely read or changed."""


class CampaignConflict(ControlError):
    """Raised when a campaign identity or action reservation already exists."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CampaignSubmittedPayload(FrozenModel):
    plan_digest: str


class CancellationPayload(FrozenModel):
    reason: str = Field(min_length=1)


class ShardRetryPayload(FrozenModel):
    shard_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class TerminalPayload(FrozenModel):
    summary_path: str | None = None
    summary_sha256: str | None = None
    message: str | None = None


class LifecyclePayload(FrozenModel):
    parent_id: str | None = None
    message: str | None = None


class WaveLifecyclePayload(FrozenModel):
    deployment_digest: str
    provider: str = Field(min_length=1)
    shard_ids: list[str]
    estimated_cost_microusd: int = Field(default=0, ge=0)


class ExecutionStartedPayload(FrozenModel):
    trial_id: str
    shard_id: str
    physical_attempt: int = Field(ge=1)
    wave_id: str | None = None
    estimated_cost_microusd: int = Field(default=0, ge=0)


class ExecutionOutcomePayload(FrozenModel):
    trial_id: str
    physical_attempt: int = Field(ge=1)
    category: RetryCategory | None = None
    spend_microusd: int = Field(default=0, ge=0)
    retry_after_seconds: int | None = Field(default=None, ge=0)
    message: str | None = None


class SpendRecordedPayload(FrozenModel):
    amount_microusd: int = Field(ge=0)
    source_execution_id: str | None = None


class ActionReservedPayload(FrozenModel):
    action_id: str
    action_key: str
    action_kind: ActionKind
    target_ids: list[str]


class ActionOutcomePayload(FrozenModel):
    action_id: str
    message: str | None = None
    remote_id: str | None = None


EventPayload = (
    CampaignSubmittedPayload
    | CancellationPayload
    | ShardRetryPayload
    | TerminalPayload
    | LifecyclePayload
    | WaveLifecyclePayload
    | ExecutionStartedPayload
    | ExecutionOutcomePayload
    | SpendRecordedPayload
    | ActionReservedPayload
    | ActionOutcomePayload
)

_CAMPAIGN_TERMINAL_KINDS = {
    "campaign.completed",
    "campaign.partial",
    "campaign.failed",
    "campaign.cancelled",
}
_WAVE_KINDS = {
    "wave.acquiring",
    "wave.provisioning",
    "wave.ready",
    "wave.active",
    "wave.draining",
    "wave.cleaning",
    "wave.closed",
    "wave.cleanup-failed",
}
_EXECUTION_OUTCOME_KINDS = {
    "execution.completed",
    "execution.failed",
    "execution.cancelled",
}
_EXACT_PAYLOAD_TYPES: dict[str, type[BaseModel]] = {
    "campaign.submitted": CampaignSubmittedPayload,
    "campaign.cancel-requested": CancellationPayload,
    "campaign.shard-retry-requested": ShardRetryPayload,
    "execution.started": ExecutionStartedPayload,
    "spend.recorded": SpendRecordedPayload,
    "action.reserved": ActionReservedPayload,
}


class CampaignEvent(FrozenModel):
    schema_version: Literal["harbor-hf/event/v1alpha1"] = "harbor-hf/event/v1alpha1"
    event_id: str = Field(pattern=r"^evt-[0-9a-f]{32}$")
    subject_type: SubjectType
    subject_id: str = Field(min_length=1)
    kind: EventKind
    observed_at: datetime
    producer: Producer
    payload: EventPayload

    @model_validator(mode="after")
    def payload_matches_kind(self) -> CampaignEvent:
        expected = _payload_type(self.kind)
        if not isinstance(self.payload, expected):
            raise ValueError(f"event payload does not match {self.kind}")
        prefix = self.kind.split(".", maxsplit=1)[0]
        if prefix in {"campaign", "run", "shard", "trial", "execution", "wave"}:
            if self.subject_type != prefix:
                raise ValueError(f"event subject type does not match {self.kind}")
        elif self.kind.startswith("action.") and self.subject_type != "campaign":
            raise ValueError("action events must have a campaign subject")
        return self


def _payload_type(kind: EventKind) -> type[BaseModel]:
    exact = _EXACT_PAYLOAD_TYPES.get(kind)
    if exact is not None:
        return exact
    if kind in _CAMPAIGN_TERMINAL_KINDS:
        return TerminalPayload
    if kind in _EXECUTION_OUTCOME_KINDS:
        return ExecutionOutcomePayload
    if kind in _WAVE_KINDS:
        return WaveLifecyclePayload
    if kind.startswith("action."):
        return ActionOutcomePayload
    return LifecyclePayload


class IdentifierFactory(Protocol):
    def __call__(self) -> str: ...


class Clock(Protocol):
    def __call__(self) -> datetime: ...


def new_event(
    *,
    subject_type: SubjectType,
    subject_id: str,
    kind: EventKind,
    producer: Producer,
    payload: EventPayload,
    clock: Clock = lambda: datetime.now(UTC),
    identifier: IdentifierFactory = lambda: uuid.uuid4().hex,
) -> CampaignEvent:
    return CampaignEvent(
        event_id=f"evt-{identifier()}",
        subject_type=subject_type,
        subject_id=subject_id,
        kind=kind,
        observed_at=clock().astimezone(UTC),
        producer=producer,
        payload=payload,
    )


ActionStatus = Literal["reserved", "succeeded", "failed", "ambiguous"]
CampaignStatus = Literal[
    "queued",
    "active",
    "cancel_requested",
    "draining",
    "manual_intervention",
    "completed",
    "partial",
    "failed",
    "cancelled",
]


class ActionProjection(FrozenModel):
    action_id: str
    action_key: str
    action_kind: ActionKind
    target_ids: list[str]
    status: ActionStatus
    remote_id: str | None = None
    message: str | None = None


class CampaignProjection(FrozenModel):
    campaign_id: str
    plan_digest: str
    status: CampaignStatus
    event_count: int
    last_observed_at: datetime
    actions: dict[str, ActionProjection]


def project_campaign(
    lock: CampaignLock, events: list[CampaignEvent]
) -> CampaignProjection:
    ordered = ordered_events(events)
    if not ordered:
        raise ControlError("campaign has no submission event")
    first = ordered[0]
    if (
        first.kind != "campaign.submitted"
        or first.subject_type != "campaign"
        or first.subject_id != lock.campaign_id
        or not isinstance(first.payload, CampaignSubmittedPayload)
        or first.payload.plan_digest != lock.plan_digest
    ):
        raise ControlError("campaign submission event does not match its lock")

    status: CampaignStatus = "queued"
    actions: dict[str, ActionProjection] = {}
    for event in ordered[1:]:
        if status in {"completed", "partial", "failed", "cancelled"}:
            raise ControlError("campaign has events after a terminal transition")
        if event.subject_type != "campaign":
            continue
        status = _apply_event(lock, event, cast(CampaignStatus, status), actions)
    return CampaignProjection(
        campaign_id=lock.campaign_id,
        plan_digest=lock.plan_digest,
        status=cast(CampaignStatus, status),
        event_count=len(ordered),
        last_observed_at=ordered[-1].observed_at,
        actions=actions,
    )


def _apply_event(
    lock: CampaignLock,
    event: CampaignEvent,
    status: CampaignStatus,
    actions: dict[str, ActionProjection],
) -> CampaignStatus:
    if event.subject_type != "campaign" or event.subject_id != lock.campaign_id:
        raise ControlError("campaign event has the wrong subject")
    if event.kind == "campaign.submitted":
        raise ControlError("campaign has multiple submission events")
    if event.kind in {
        "campaign.cancel-requested",
        "campaign.shard-retry-requested",
    }:
        return _apply_operator_request(event, status)
    if event.kind == "campaign.draining":
        return "draining"
    if event.kind == "campaign.manual-intervention-required":
        return "manual_intervention"
    if event.kind.startswith("campaign."):
        return cast(CampaignStatus, event.kind.removeprefix("campaign."))
    _apply_action_event(event, actions)
    return "active" if status == "queued" else status


def _apply_operator_request(
    event: CampaignEvent, status: CampaignStatus
) -> CampaignStatus:
    if event.kind == "campaign.shard-retry-requested":
        return "active" if status == "queued" else status
    if status in {"draining", "manual_intervention"}:
        return status
    return "cancel_requested"


def _apply_action_event(
    event: CampaignEvent, actions: dict[str, ActionProjection]
) -> None:
    if event.kind == "action.reserved":
        payload = cast(ActionReservedPayload, event.payload)
        if payload.action_id in actions:
            raise ControlError("action was reserved more than once")
        actions[payload.action_id] = ActionProjection(
            **payload.model_dump(mode="python"), status="reserved"
        )
        return
    payload = cast(ActionOutcomePayload, event.payload)
    action = actions.get(payload.action_id)
    if action is None:
        raise ControlError("action outcome has no reservation")
    if action.status not in {"reserved", "ambiguous"}:
        raise ControlError("action has multiple outcomes")
    outcome = cast(ActionStatus, event.kind.removeprefix("action."))
    if action.status == "ambiguous" and outcome == "ambiguous":
        return
    actions[payload.action_id] = action.model_copy(
        update={
            "status": outcome,
            "message": payload.message,
            "remote_id": payload.remote_id,
        }
    )


def ordered_events(events: list[CampaignEvent]) -> list[CampaignEvent]:
    unique: dict[str, CampaignEvent] = {}
    for event in events:
        previous = unique.get(event.event_id)
        if previous is not None and previous != event:
            raise ControlError("event ID has conflicting records")
        unique[event.event_id] = event
    return sorted(
        unique.values(), key=lambda event: (event.observed_at, event.event_id)
    )


class CampaignStoreApi(Protocol):
    def repo_info(self, repo_id: str, **kwargs: object) -> object: ...

    def get_paths_info(
        self, repo_id: str, paths: str | list[str], **kwargs: object
    ) -> list[object]: ...

    def list_repo_files(self, repo_id: str, **kwargs: object) -> list[str]: ...

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: object) -> str: ...

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object: ...


class CampaignStore(Protocol):
    def create_campaign(
        self, lock: CampaignLock, request: bytes, event: CampaignEvent
    ) -> None: ...

    def load_campaign(
        self, campaign_id: str
    ) -> tuple[CampaignLock, list[CampaignEvent]]: ...

    def load_request(self, campaign_id: str) -> bytes: ...

    def list_campaigns(self) -> list[str]: ...

    def load_action_reservations(
        self, campaign_id: str
    ) -> list[dict[str, JsonValue]]: ...

    def append_event(self, campaign_id: str, event: CampaignEvent) -> None: ...

    def ensure_event(self, campaign_id: str, event: CampaignEvent) -> bool: ...

    def load_snapshot(self, campaign_id: str) -> CampaignSnapshot: ...

    def reserve_action(
        self, campaign_id: str, action: Mapping[str, JsonValue], event: CampaignEvent
    ) -> bool: ...


@dataclass(frozen=True)
class CampaignSnapshot:
    lock: CampaignLock
    events: list[CampaignEvent]
    request: bytes
    control_commit: str


class HubCampaignStore:
    def __init__(self, namespace: str, *, api: CampaignStoreApi | None = None) -> None:
        self.repository = coordination_repository(namespace)
        self.api = api or cast(CampaignStoreApi, HfApi())

    def create_campaign(
        self, lock: CampaignLock, request: bytes, event: CampaignEvent
    ) -> None:
        if event.kind != "campaign.submitted" or event.subject_id != lock.campaign_id:
            raise ValueError("campaign creation requires its submission event")
        lock_path = _campaign_lock_path(lock.campaign_id)
        reservation_path = _campaign_reservation_path(lock.campaign_id)
        operations: list[object] = [
            CommitOperationAdd(
                path_in_repo=_campaign_request_path(lock.campaign_id),
                path_or_fileobj=request,
            ),
            CommitOperationAdd(
                path_in_repo=lock_path,
                path_or_fileobj=_json_bytes(lock.model_dump(mode="json")),
            ),
            CommitOperationAdd(
                path_in_repo=_event_path(lock.campaign_id, event.event_id),
                path_or_fileobj=_json_bytes(event.model_dump(mode="json")),
            ),
            CommitOperationAdd(
                path_in_repo=reservation_path,
                path_or_fileobj=_json_bytes(
                    {"campaign_id": lock.campaign_id, "plan_digest": lock.plan_digest}
                ),
            ),
        ]
        self._create_absent(
            lock_path,
            operations,
            f"feat: submit campaign {lock.campaign_id}",
            conflict=f"campaign already exists: {lock.campaign_id}",
        )

    def load_campaign(
        self, campaign_id: str
    ) -> tuple[CampaignLock, list[CampaignEvent]]:
        snapshot = self.load_snapshot(campaign_id)
        return snapshot.lock, snapshot.events

    def load_snapshot(self, campaign_id: str) -> CampaignSnapshot:
        head = self._head()
        lock_path = _campaign_lock_path(campaign_id)
        lock = CampaignLock.model_validate(self._read_json(lock_path, head))
        prefix = f"campaigns/{campaign_id}/events/"
        paths = sorted(
            path
            for path in self.api.list_repo_files(
                self.repository,
                repo_type="dataset",
                revision=head,
            )
            if path.startswith(prefix) and path.endswith(".json")
        )
        events = [
            CampaignEvent.model_validate(self._read_json(path, head)) for path in paths
        ]
        return CampaignSnapshot(
            lock=lock,
            events=events,
            request=self._read_bytes(_campaign_request_path(campaign_id), head),
            control_commit=self._last_commit(lock_path, head),
        )

    def load_request(self, campaign_id: str) -> bytes:
        head = self._head()
        return self._read_bytes(_campaign_request_path(campaign_id), head)

    def list_campaigns(self) -> list[str]:
        head = self._head()
        suffix = "/campaign.lock.json"
        campaign_ids = {
            path.removeprefix("campaigns/").removesuffix(suffix)
            for path in self.api.list_repo_files(
                self.repository,
                repo_type="dataset",
                revision=head,
            )
            if path.startswith("campaigns/")
            and path.endswith(suffix)
            and _CAMPAIGN_ID.fullmatch(
                path.removeprefix("campaigns/").removesuffix(suffix)
            )
            is not None
        }
        return sorted(campaign_ids)

    def load_action_reservations(self, campaign_id: str) -> list[dict[str, JsonValue]]:
        head = self._head()
        prefix = f"campaigns/{campaign_id}/reservations/"
        paths = sorted(
            path
            for path in self.api.list_repo_files(
                self.repository,
                repo_type="dataset",
                revision=head,
            )
            if path.startswith(prefix) and path.endswith(".json")
        )
        reservations: list[dict[str, JsonValue]] = []
        for path in paths:
            value = self._read_json(path, head)
            if not isinstance(value, dict) or not all(
                isinstance(key, str) for key in value
            ):
                raise ControlError(f"action reservation must be an object: {path}")
            reservations.append(cast(dict[str, JsonValue], value))
        return reservations

    def append_event(self, campaign_id: str, event: CampaignEvent) -> None:
        _validate_event_scope(campaign_id, event)
        path = _event_path(campaign_id, event.event_id)
        self._create_absent(
            path,
            [
                CommitOperationAdd(
                    path_in_repo=path,
                    path_or_fileobj=_json_bytes(event.model_dump(mode="json")),
                )
            ],
            f"chore: record {event.kind}",
            conflict=f"event already exists: {event.event_id}",
        )

    def ensure_event(self, campaign_id: str, event: CampaignEvent) -> bool:
        """Append an event once, adopting an identical concurrent request."""
        _validate_event_scope(campaign_id, event)
        path = _event_path(campaign_id, event.event_id)
        expected = event.model_dump(mode="json")
        for _attempt in range(_MAX_COMMIT_ATTEMPTS):
            head = self._head()
            if self._exists(path, head):
                if self._read_json(path, head) != expected:
                    raise CampaignConflict(f"event conflicts: {event.event_id}")
                return False
            try:
                self.api.create_commit(
                    self.repository,
                    [
                        CommitOperationAdd(
                            path_in_repo=path,
                            path_or_fileobj=_json_bytes(expected),
                        )
                    ],
                    commit_message=f"chore: record {event.kind}",
                    repo_type="dataset",
                    revision="main",
                    parent_commit=head,
                )
                return True
            except HfHubHTTPError as error:
                if not _is_parent_conflict(error):
                    raise
        raise ControlError("control repository remained contended")

    def reserve_action(
        self,
        campaign_id: str,
        action: Mapping[str, JsonValue],
        event: CampaignEvent,
    ) -> bool:
        if event.kind != "action.reserved" or event.subject_id != campaign_id:
            raise ValueError("action reservation requires its reservation event")
        payload = cast(ActionReservedPayload, event.payload)
        path = _action_reservation_path(campaign_id, payload.action_id)
        for _attempt in range(_MAX_COMMIT_ATTEMPTS):
            head = self._head()
            if self._exists(path, head):
                observed = self._read_json(path, head)
                if observed != dict(action):
                    raise CampaignConflict("action reservation content conflicts")
                return False
            try:
                self.api.create_commit(
                    self.repository,
                    [
                        CommitOperationAdd(
                            path_in_repo=path,
                            path_or_fileobj=_json_bytes(dict(action)),
                        ),
                        CommitOperationAdd(
                            path_in_repo=_event_path(campaign_id, event.event_id),
                            path_or_fileobj=_json_bytes(event.model_dump(mode="json")),
                        ),
                    ],
                    commit_message=f"chore: reserve {payload.action_id}",
                    repo_type="dataset",
                    revision="main",
                    parent_commit=head,
                )
                return True
            except HfHubHTTPError as error:
                if not _is_parent_conflict(error):
                    raise
        raise ControlError("control repository remained contended")

    def _create_absent(
        self,
        path: str,
        operations: list[object],
        message: str,
        *,
        conflict: str,
    ) -> None:
        for _attempt in range(_MAX_COMMIT_ATTEMPTS):
            head = self._head()
            if self._exists(path, head):
                raise CampaignConflict(conflict)
            try:
                self.api.create_commit(
                    self.repository,
                    operations,
                    commit_message=message,
                    repo_type="dataset",
                    revision="main",
                    parent_commit=head,
                )
                return
            except HfHubHTTPError as error:
                if not _is_parent_conflict(error):
                    raise
        raise ControlError("control repository remained contended")

    def _head(self) -> str:
        revision = getattr(
            self.api.repo_info(self.repository, repo_type="dataset", revision="main"),
            "sha",
            None,
        )
        if not isinstance(revision, str) or not revision:
            raise ControlError("control repository has no commit identity")
        return revision

    def _exists(self, path: str, revision: str) -> bool:
        return bool(
            self.api.get_paths_info(
                self.repository,
                path,
                repo_type="dataset",
                revision=revision,
            )
        )

    def _last_commit(self, path: str, revision: str) -> str:
        records = self.api.get_paths_info(
            self.repository,
            path,
            repo_type="dataset",
            revision=revision,
            expand=True,
        )
        if len(records) != 1:
            raise ControlError(f"control record has no immutable commit: {path}")
        last_commit = getattr(records[0], "last_commit", None)
        oid = getattr(last_commit, "oid", None)
        if not isinstance(oid, str) or not oid:
            raise ControlError(f"control record has no immutable commit: {path}")
        return oid

    def _read_json(self, path: str, revision: str) -> object:
        try:
            return json.loads(self._read_bytes(path, revision))
        except json.JSONDecodeError as error:
            raise ControlError(f"control record cannot be read: {path}") from error

    def _read_bytes(self, path: str, revision: str) -> bytes:
        try:
            local_path = self.api.hf_hub_download(
                self.repository,
                path,
                repo_type="dataset",
                revision=revision,
            )
            return Path(local_path).read_bytes()
        except OSError as error:
            raise ControlError(f"control record cannot be read: {path}") from error


def _campaign_request_path(campaign_id: str) -> str:
    return f"campaigns/{campaign_id}/request.yaml"


def _validate_event_scope(campaign_id: str, event: CampaignEvent) -> None:
    if _CAMPAIGN_ID.fullmatch(campaign_id) is None:
        raise ValueError("event campaign scope is invalid")
    if event.subject_type == "campaign" and event.subject_id != campaign_id:
        raise ValueError("campaign event subject does not match its scope")


def _campaign_lock_path(campaign_id: str) -> str:
    return f"campaigns/{campaign_id}/campaign.lock.json"


def _campaign_reservation_path(campaign_id: str) -> str:
    identity = hashlib.sha256(campaign_id.encode()).hexdigest()
    return f"campaign-reservations/{identity}.json"


def _event_path(campaign_id: str, event_id: str) -> str:
    return f"campaigns/{campaign_id}/events/{event_id}.json"


def _action_reservation_path(campaign_id: str, action_id: str) -> str:
    return f"campaigns/{campaign_id}/reservations/{action_id}.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _is_parent_conflict(error: HfHubHTTPError) -> bool:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) in {409, 412}
