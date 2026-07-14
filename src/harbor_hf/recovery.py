from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from harbor_hf.campaigns import CampaignLock
from harbor_hf.control import (
    CampaignEvent,
    CampaignProjection,
    CancellationPayload,
    Clock,
    ExecutionOutcomePayload,
    ExecutionStartedPayload,
    IdentifierFactory,
    RetryCategory,
    ShardRetryPayload,
    SpendRecordedPayload,
    WaveLifecyclePayload,
    new_event,
    ordered_events,
    project_campaign,
)

RunStatus = Literal[
    "planned",
    "queued",
    "active",
    "verifying",
    "publishing",
    "complete",
    "invalid",
    "failed_infrastructure",
    "cancelled",
]
ShardStatus = Literal[
    "planned",
    "queued",
    "active",
    "verifying",
    "publishing",
    "retry_wait",
    "complete",
    "invalid",
    "failed_infrastructure",
    "cancelled",
]
TrialStatus = Literal[
    "planned",
    "active",
    "retry_wait",
    "complete",
    "invalid",
    "failed_infrastructure",
    "cancelled",
]
ExecutionStatus = Literal["active", "completed", "failed", "cancelled"]
WaveStatus = Literal[
    "acquiring",
    "provisioning",
    "ready",
    "active",
    "draining",
    "cleaning",
    "closed",
    "cleanup_failed",
]
TerminalStatus = Literal["completed", "partial", "failed", "cancelled"]

_RETRYABLE_CATEGORIES = {"lost", "transient", "quota", "rate-limit", "ambiguous"}
_TERMINAL_STATUSES = {"complete", "invalid", "failed_infrastructure", "cancelled"}
_CAMPAIGN_TERMINAL = {"completed", "partial", "failed", "cancelled"}


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionProjection(FrozenModel):
    execution_id: str
    trial_id: str
    shard_id: str
    wave_id: str | None
    physical_attempt: int
    status: ExecutionStatus
    category: RetryCategory | None = None
    observed_at: datetime
    retry_after_seconds: int | None = None
    estimated_cost_microusd: int = 0
    spend_microusd: int = 0
    message: str | None = None


class TrialProjection(FrozenModel):
    trial_id: str
    shard_id: str
    logical_attempt: int
    status: TrialStatus = "planned"
    executions: dict[str, ExecutionProjection] = Field(default_factory=dict)
    retry_not_before: datetime | None = None


class ShardProjection(FrozenModel):
    shard_id: str
    run_id: str
    status: ShardStatus = "planned"
    trial_ids: list[str]
    observed_status: ShardStatus | None = None


class RunProjection(FrozenModel):
    run_id: str
    deployment_digest: str
    status: RunStatus = "planned"
    shard_ids: list[str]
    observed_status: RunStatus | None = None


class WaveProjection(FrozenModel):
    wave_id: str
    deployment_digest: str
    provider: str
    shard_ids: list[str]
    estimated_cost_microusd: int
    status: WaveStatus


class ProjectionCounts(FrozenModel):
    planned: int = 0
    active: int = 0
    retrying: int = 0
    complete: int = 0
    invalid: int = 0
    failed: int = 0
    cancelled: int = 0
    physical_retries: int = 0


class TerminalDecision(FrozenModel):
    status: TerminalStatus
    marker: Literal["_SUCCESS", "_PARTIAL", "_FAILED", "_CANCELLED"]
    summary_path: str
    marker_path: str
    reason: str
    counts: ProjectionCounts


class RecoveryProjection(FrozenModel):
    campaign: CampaignProjection
    runs: dict[str, RunProjection]
    shards: dict[str, ShardProjection]
    trials: dict[str, TrialProjection]
    executions: dict[str, ExecutionProjection]
    waves: dict[str, WaveProjection]
    spend_microusd: int
    counts: ProjectionCounts
    cancel_requested_at: datetime | None = None
    terminal_decision: TerminalDecision | None = None

    @property
    def status(self) -> str:
        return self.campaign.status


def durable_cancellation_event(
    lock: CampaignLock,
    events: list[CampaignEvent],
    reason: str,
    *,
    clock: Clock = lambda: datetime.now(UTC),
    identifier: IdentifierFactory | None = None,
) -> tuple[CampaignEvent, bool]:
    for event in ordered_events(events):
        if event.kind == "campaign.cancel-requested":
            return event, False
    return (
        new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="campaign.cancel-requested",
            producer="cli",
            payload=CancellationPayload(reason=reason),
            clock=clock,
            identifier=identifier or _cancellation_identifier(lock),
        ),
        True,
    )


def _cancellation_identifier(lock: CampaignLock) -> IdentifierFactory:
    return lambda: hashlib.sha256(f"{lock.campaign_id}:cancel".encode()).hexdigest()[
        :32
    ]


def durable_shard_retry_event(
    lock: CampaignLock,
    events: list[CampaignEvent],
    shard_id: str,
    reason: str,
    *,
    clock: Clock = lambda: datetime.now(UTC),
) -> tuple[CampaignEvent, bool]:
    """Create one immediate retry request for the shard's current execution state."""
    projection = project_recovery(lock, events)
    if projection.campaign.status in _CAMPAIGN_TERMINAL:
        raise ValueError("a terminal campaign cannot be retried")
    if projection.campaign.status in {"cancel_requested", "draining"}:
        raise ValueError("a cancelling campaign cannot be retried")
    shard = projection.shards.get(shard_id)
    if shard is None:
        raise ValueError(f"unknown campaign shard: {shard_id}")
    eligible = [
        projection.trials[trial_id]
        for trial_id in shard.trial_ids
        if projection.trials[trial_id].status == "retry_wait"
    ]
    if not eligible:
        raise ValueError("shard has no retryable logical trials")
    generation = ",".join(
        f"{trial.trial_id}:{len(trial.executions)}" for trial in eligible
    )
    identifier = hashlib.sha256(
        f"{lock.campaign_id}:{shard_id}:{generation}".encode()
    ).hexdigest()[:32]
    event_id = f"evt-{identifier}"
    for event in ordered_events(events):
        if event.event_id == event_id:
            if event.kind != "campaign.shard-retry-requested":
                raise ValueError("retry event identity conflicts")
            return event, False
    return (
        new_event(
            subject_type="campaign",
            subject_id=lock.campaign_id,
            kind="campaign.shard-retry-requested",
            producer="cli",
            payload=ShardRetryPayload(
                shard_id=shard_id,
                reason=reason,
                trial_generations={
                    trial.trial_id: len(trial.executions) for trial in eligible
                },
            ),
            clock=clock,
            identifier=lambda: identifier,
        ),
        True,
    )


def project_recovery(
    lock: CampaignLock, events: list[CampaignEvent]
) -> RecoveryProjection:
    campaign = project_campaign(lock, events)
    runs, shards, trials = _initial_projections(lock)
    executions: dict[str, ExecutionProjection] = {}
    waves: dict[str, WaveProjection] = {}
    spend = 0
    for event in ordered_events(events):
        spend += _apply_recovery_event(event, runs, shards, trials, executions, waves)
    trials = _derive_trials(lock, trials, executions)
    trials = _apply_retry_requests(lock, events, trials)
    shards = _derive_shards(shards, trials)
    runs = _derive_runs(runs, shards)
    counts = _counts(trials)
    if campaign.status == "queued" and (executions or waves):
        campaign = campaign.model_copy(update={"status": "active"})
    terminal = _terminal_decision(lock, campaign, trials, waves, counts)
    cancel_requested_at = next(
        (
            event.observed_at
            for event in ordered_events(events)
            if event.kind == "campaign.cancel-requested"
        ),
        None,
    )
    return RecoveryProjection(
        campaign=campaign,
        runs=runs,
        shards=shards,
        trials=trials,
        executions=executions,
        waves=waves,
        spend_microusd=spend,
        counts=counts,
        cancel_requested_at=cancel_requested_at,
        terminal_decision=terminal,
    )


def _apply_retry_requests(
    lock: CampaignLock,
    events: list[CampaignEvent],
    trials: dict[str, TrialProjection],
) -> dict[str, TrialProjection]:
    requested_at: dict[tuple[str, int], datetime] = {}
    ordered = ordered_events(events)
    has_legacy_requests = any(
        event.kind == "campaign.shard-retry-requested"
        and not cast(ShardRetryPayload, event.payload).trial_generations
        for event in ordered
    )
    legacy_generations = (
        _legacy_retry_generations(lock, ordered) if has_legacy_requests else {}
    )
    for event in ordered:
        if event.kind != "campaign.shard-retry-requested":
            continue
        payload = cast(ShardRetryPayload, event.payload)
        generations = payload.trial_generations or legacy_generations[event.event_id]
        for trial_id, generation in generations.items():
            key = (trial_id, generation)
            requested_at[key] = max(
                requested_at.get(key, event.observed_at), event.observed_at
            )
    return {
        trial_id: (
            trial.model_copy(
                update={
                    "retry_not_before": requested_at[
                        (trial.trial_id, len(trial.executions))
                    ]
                }
            )
            if trial.status == "retry_wait"
            and (trial.trial_id, len(trial.executions)) in requested_at
            else trial
        )
        for trial_id, trial in trials.items()
    }


def _legacy_retry_generations(
    lock: CampaignLock, events: list[CampaignEvent]
) -> dict[str, dict[str, int]]:
    """Recover generation bindings for legacy retry events in one forward pass."""
    runs, shards, trials = _initial_projections(lock)
    executions: dict[str, ExecutionProjection] = {}
    execution_ids_by_trial: dict[str, list[str]] = {trial_id: [] for trial_id in trials}
    waves: dict[str, WaveProjection] = {}
    generations: dict[str, dict[str, int]] = {}
    for event in events:
        if event.kind == "campaign.shard-retry-requested":
            payload = cast(ShardRetryPayload, event.payload)
            if not payload.trial_generations:
                shard = shards.get(payload.shard_id)
                event_generations: dict[str, int] = {}
                if shard is not None:
                    for trial_id in shard.trial_ids:
                        current = _derive_trial(
                            lock,
                            trials[trial_id],
                            [
                                executions[execution_id]
                                for execution_id in execution_ids_by_trial[trial_id]
                            ],
                        )
                        if current.status == "retry_wait":
                            event_generations[trial_id] = len(current.executions)
                generations[event.event_id] = event_generations
        _apply_recovery_event(event, runs, shards, trials, executions, waves)
        if event.kind == "execution.started":
            payload = cast(ExecutionStartedPayload, event.payload)
            execution_ids_by_trial[payload.trial_id].append(event.subject_id)
    return generations


def _apply_recovery_event(
    event: CampaignEvent,
    runs: dict[str, RunProjection],
    shards: dict[str, ShardProjection],
    trials: dict[str, TrialProjection],
    executions: dict[str, ExecutionProjection],
    waves: dict[str, WaveProjection],
) -> int:
    if event.kind.startswith("run."):
        _record_run_status(event, runs)
    elif event.kind.startswith("shard."):
        _record_shard_status(event, shards)
    elif event.kind.startswith("trial."):
        _record_trial_status(event, trials)
    elif event.kind == "execution.started":
        _start_execution(event, trials, executions)
    elif event.kind.startswith("execution."):
        return _finish_execution(event, executions)
    elif event.kind.startswith("wave."):
        _record_wave_status(event, shards, waves)
    elif event.kind == "spend.recorded":
        return cast(SpendRecordedPayload, event.payload).amount_microusd
    return 0


def retry_delay_seconds(
    lock: CampaignLock,
    category: RetryCategory,
    physical_attempt: int,
    execution_id: str,
    retry_after_seconds: int | None = None,
) -> int:
    if category not in _RETRYABLE_CATEGORIES:
        raise ValueError(f"retry category is terminal: {category}")
    policy = lock.recovery_policy
    exponent = min(physical_attempt - 1, 30)
    multiplier = 2 if category in {"quota", "rate-limit"} else 1
    raw = policy.retry_base_seconds * (2**exponent) * multiplier
    digest = hashlib.sha256(
        f"{execution_id}:{category}:{physical_attempt}".encode()
    ).digest()
    jittered = raw * (75 + digest[0] * 50 // 255) // 100
    requested = retry_after_seconds or 0
    return min(policy.retry_max_seconds, max(1, jittered, requested))


def retry_is_ready(trial: TrialProjection, now: datetime) -> bool:
    return (
        trial.status == "retry_wait"
        and trial.retry_not_before is not None
        and now.astimezone(UTC) >= trial.retry_not_before
    )


def _initial_projections(
    lock: CampaignLock,
) -> tuple[
    dict[str, RunProjection],
    dict[str, ShardProjection],
    dict[str, TrialProjection],
]:
    runs: dict[str, RunProjection] = {}
    shards: dict[str, ShardProjection] = {}
    trials: dict[str, TrialProjection] = {}
    for run in lock.runs:
        runs[run.run_id] = RunProjection(
            run_id=run.run_id,
            deployment_digest=run.deployment_digest,
            shard_ids=[shard.shard_id for shard in run.shards],
        )
        for shard in run.shards:
            shards[shard.shard_id] = ShardProjection(
                shard_id=shard.shard_id,
                run_id=run.run_id,
                trial_ids=[trial.trial_id for trial in shard.trials],
            )
            for trial in shard.trials:
                trials[trial.trial_id] = TrialProjection(
                    trial_id=trial.trial_id,
                    shard_id=shard.shard_id,
                    logical_attempt=trial.logical_attempt,
                )
    return runs, shards, trials


def _record_run_status(event: CampaignEvent, runs: dict[str, RunProjection]) -> None:
    run = runs.get(event.subject_id)
    if run is None:
        raise ValueError(f"event references unknown run: {event.subject_id}")
    status = cast(RunStatus, event.kind.removeprefix("run.").replace("-", "_"))
    runs[event.subject_id] = run.model_copy(update={"observed_status": status})


def _record_shard_status(
    event: CampaignEvent, shards: dict[str, ShardProjection]
) -> None:
    shard = shards.get(event.subject_id)
    if shard is None:
        raise ValueError(f"event references unknown shard: {event.subject_id}")
    status = cast(ShardStatus, event.kind.removeprefix("shard.").replace("-", "_"))
    shards[event.subject_id] = shard.model_copy(update={"observed_status": status})


def _record_trial_status(
    event: CampaignEvent, trials: dict[str, TrialProjection]
) -> None:
    trial = trials.get(event.subject_id)
    if trial is None:
        raise ValueError(f"event references unknown trial: {event.subject_id}")
    status = cast(TrialStatus, event.kind.removeprefix("trial."))
    trials[event.subject_id] = trial.model_copy(update={"status": status})


def _start_execution(
    event: CampaignEvent,
    trials: dict[str, TrialProjection],
    executions: dict[str, ExecutionProjection],
) -> None:
    payload = cast(ExecutionStartedPayload, event.payload)
    trial = trials.get(payload.trial_id)
    if trial is None or trial.shard_id != payload.shard_id:
        raise ValueError("execution references an unknown trial or shard")
    if event.subject_id in executions:
        raise ValueError(f"execution started more than once: {event.subject_id}")
    attempts = {
        execution.physical_attempt
        for execution in executions.values()
        if execution.trial_id == payload.trial_id
    }
    if payload.physical_attempt in attempts:
        raise ValueError("trial has duplicate physical execution numbers")
    executions[event.subject_id] = ExecutionProjection(
        execution_id=event.subject_id,
        status="active",
        observed_at=event.observed_at,
        **payload.model_dump(mode="python"),
    )


def _finish_execution(
    event: CampaignEvent, executions: dict[str, ExecutionProjection]
) -> int:
    execution = executions.get(event.subject_id)
    if execution is None:
        raise ValueError(f"execution outcome has no start: {event.subject_id}")
    if execution.status != "active":
        raise ValueError(f"execution has multiple outcomes: {event.subject_id}")
    payload = cast(ExecutionOutcomePayload, event.payload)
    if (
        payload.trial_id != execution.trial_id
        or payload.physical_attempt != execution.physical_attempt
    ):
        raise ValueError("execution outcome identity does not match its start")
    if event.kind == "execution.completed" and payload.category is not None:
        raise ValueError("completed execution cannot have a failure category")
    if event.kind == "execution.failed" and payload.category is None:
        raise ValueError("failed execution requires a failure category")
    status = cast(ExecutionStatus, event.kind.removeprefix("execution."))
    executions[event.subject_id] = execution.model_copy(
        update={
            "status": status,
            "category": payload.category,
            "observed_at": event.observed_at,
            "retry_after_seconds": payload.retry_after_seconds,
            "spend_microusd": payload.spend_microusd,
            "message": payload.message,
        }
    )
    return payload.spend_microusd


def _record_wave_status(
    event: CampaignEvent,
    shards: dict[str, ShardProjection],
    waves: dict[str, WaveProjection],
) -> None:
    payload = cast(WaveLifecyclePayload, event.payload)
    unknown = set(payload.shard_ids) - shards.keys()
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"wave references unknown shards: {names}")
    status = cast(WaveStatus, event.kind.removeprefix("wave.").replace("-", "_"))
    previous = waves.get(event.subject_id)
    identity = (
        payload.deployment_digest,
        payload.provider,
        payload.shard_ids,
        payload.estimated_cost_microusd,
    )
    if previous is not None:
        observed = (
            previous.deployment_digest,
            previous.provider,
            previous.shard_ids,
            previous.estimated_cost_microusd,
        )
        if observed != identity:
            raise ValueError("wave lifecycle identity changed")
        _validate_wave_transition(previous.status, status)
    waves[event.subject_id] = WaveProjection(
        wave_id=event.subject_id,
        status=status,
        **payload.model_dump(mode="python"),
    )


def _validate_wave_transition(previous: WaveStatus, current: WaveStatus) -> None:
    allowed: dict[WaveStatus, set[WaveStatus]] = {
        "acquiring": {"provisioning", "draining", "cleaning", "cleanup_failed"},
        "provisioning": {"ready", "draining", "cleaning", "cleanup_failed"},
        "ready": {"active", "draining", "cleaning", "cleanup_failed"},
        "active": {"draining", "cleaning", "cleanup_failed"},
        "draining": {"cleaning", "closed", "cleanup_failed"},
        "cleaning": {"closed", "cleanup_failed"},
        "cleanup_failed": {"cleaning", "closed"},
        "closed": set(),
    }
    if current != previous and current not in allowed[previous]:
        raise ValueError(f"invalid wave transition: {previous} -> {current}")


def _derive_trials(
    lock: CampaignLock,
    trials: dict[str, TrialProjection],
    executions: dict[str, ExecutionProjection],
) -> dict[str, TrialProjection]:
    by_trial: dict[str, list[ExecutionProjection]] = {key: [] for key in trials}
    for execution in executions.values():
        by_trial[execution.trial_id].append(execution)
    return {
        trial_id: _derive_trial(lock, trial, by_trial[trial_id])
        for trial_id, trial in trials.items()
    }


def _derive_trial(
    lock: CampaignLock,
    trial: TrialProjection,
    executions: list[ExecutionProjection],
) -> TrialProjection:
    ordered = sorted(executions, key=lambda value: value.physical_attempt)
    attempts = [execution.physical_attempt for execution in ordered]
    if attempts != list(range(1, len(ordered) + 1)):
        raise ValueError("physical execution attempts must be contiguous")
    completed = [
        index
        for index, execution in enumerate(ordered)
        if execution.status == "completed"
    ]
    if completed and completed[-1] != len(ordered) - 1:
        raise ValueError("a completed logical trial was physically re-executed")
    execution_map = {value.execution_id: value for value in ordered}
    if trial.status in _TERMINAL_STATUSES:
        return trial.model_copy(update={"executions": execution_map})
    if not ordered:
        return trial
    if any(execution.status == "completed" for execution in ordered):
        return trial.model_copy(
            update={"status": "complete", "executions": execution_map}
        )
    latest = ordered[-1]
    if latest.status == "active":
        status: TrialStatus = "active"
        retry_at = None
    elif latest.status == "cancelled":
        status = "cancelled"
        retry_at = None
    else:
        status, retry_at = _failed_trial_state(lock, latest, len(ordered))
    return trial.model_copy(
        update={
            "status": status,
            "executions": execution_map,
            "retry_not_before": retry_at,
        }
    )


def _failed_trial_state(
    lock: CampaignLock, execution: ExecutionProjection, execution_count: int
) -> tuple[TrialStatus, datetime | None]:
    category = execution.category
    if category == "benchmark":
        return "invalid", None
    if category not in _RETRYABLE_CATEGORIES:
        return "failed_infrastructure", None
    if execution_count >= lock.recovery_policy.max_physical_executions_per_trial:
        return "failed_infrastructure", None
    delay = retry_delay_seconds(
        lock,
        category,
        execution.physical_attempt,
        execution.execution_id,
        execution.retry_after_seconds,
    )
    return "retry_wait", execution.observed_at + timedelta(seconds=delay)


def _derive_shards(
    shards: dict[str, ShardProjection], trials: dict[str, TrialProjection]
) -> dict[str, ShardProjection]:
    return {
        shard_id: shard.model_copy(update={"status": _aggregate_shard(shard, trials)})
        for shard_id, shard in shards.items()
    }


def _aggregate_shard(
    shard: ShardProjection, trials: dict[str, TrialProjection]
) -> ShardStatus:
    statuses = [trials[trial_id].status for trial_id in shard.trial_ids]
    _validate_observed_terminal(shard.observed_status, statuses, "shard")
    if all(status == "complete" for status in statuses):
        return "complete"
    if all(status in _TERMINAL_STATUSES for status in statuses):
        if any(status == "failed_infrastructure" for status in statuses):
            return "failed_infrastructure"
        if any(status == "invalid" for status in statuses):
            return "invalid"
        return "cancelled"
    if any(status == "active" for status in statuses):
        return "active"
    if any(status == "retry_wait" for status in statuses):
        return "retry_wait"
    return shard.observed_status or "planned"


def _derive_runs(
    runs: dict[str, RunProjection], shards: dict[str, ShardProjection]
) -> dict[str, RunProjection]:
    return {
        run_id: run.model_copy(update={"status": _aggregate_run(run, shards)})
        for run_id, run in runs.items()
    }


def _aggregate_run(run: RunProjection, shards: dict[str, ShardProjection]) -> RunStatus:
    statuses = [shards[shard_id].status for shard_id in run.shard_ids]
    _validate_observed_terminal(run.observed_status, statuses, "run")
    if all(status == "complete" for status in statuses):
        return "complete"
    if all(status in _TERMINAL_STATUSES for status in statuses):
        if any(status == "failed_infrastructure" for status in statuses):
            return "failed_infrastructure"
        if any(status == "invalid" for status in statuses):
            return "invalid"
        return "cancelled"
    if any(status in {"active", "retry_wait"} for status in statuses):
        return "active"
    return run.observed_status or "planned"


def _validate_observed_terminal(
    observed: RunStatus | ShardStatus | None,
    child_statuses: list[TrialStatus] | list[ShardStatus],
    subject: str,
) -> None:
    if observed not in _TERMINAL_STATUSES:
        return
    if not all(status in _TERMINAL_STATUSES for status in child_statuses):
        raise ValueError(f"{subject} became terminal before its children")
    if observed == "complete" and not all(
        status == "complete" for status in child_statuses
    ):
        raise ValueError(f"{subject} completed with non-complete children")


def _counts(trials: dict[str, TrialProjection]) -> ProjectionCounts:
    values = list(trials.values())
    executions = sum(len(trial.executions) for trial in values)
    return ProjectionCounts(
        planned=sum(trial.status == "planned" for trial in values),
        active=sum(trial.status == "active" for trial in values),
        retrying=sum(trial.status == "retry_wait" for trial in values),
        complete=sum(trial.status == "complete" for trial in values),
        invalid=sum(trial.status == "invalid" for trial in values),
        failed=sum(trial.status == "failed_infrastructure" for trial in values),
        cancelled=sum(trial.status == "cancelled" for trial in values),
        physical_retries=max(
            0, executions - sum(bool(trial.executions) for trial in values)
        ),
    )


def _terminal_decision(
    lock: CampaignLock,
    campaign: CampaignProjection,
    trials: dict[str, TrialProjection],
    waves: dict[str, WaveProjection],
    counts: ProjectionCounts,
) -> TerminalDecision | None:
    if campaign.status in _CAMPAIGN_TERMINAL:
        status = cast(TerminalStatus, campaign.status)
        return _decision(lock, status, counts, "recorded")
    if not _cleanup_is_complete(campaign, waves):
        return None
    decision_counts = _terminal_counts(campaign, counts)
    cancelling = campaign.status in {"cancel_requested", "draining"}
    if not all(
        trial.status in _TERMINAL_STATUSES
        or (cancelling and trial.status in {"planned", "retry_wait"})
        for trial in trials.values()
    ):
        return None
    if decision_counts.complete == len(trials):
        return _decision(
            lock, "completed", decision_counts, "all logical trials completed"
        )
    if decision_counts.complete:
        return _decision(
            lock, "partial", decision_counts, "some logical trials completed"
        )
    if cancelling or decision_counts.cancelled:
        return _decision(
            lock,
            "cancelled",
            decision_counts,
            "cancellation drained and cleaned",
        )
    return _decision(
        lock, "failed", decision_counts, "no valid logical trial completed"
    )


def _terminal_counts(
    campaign: CampaignProjection, counts: ProjectionCounts
) -> ProjectionCounts:
    if campaign.status not in {"cancel_requested", "draining"}:
        return counts
    return counts.model_copy(
        update={
            "planned": 0,
            "retrying": 0,
            "cancelled": counts.cancelled + counts.planned + counts.retrying,
        }
    )


def _cleanup_is_complete(
    campaign: CampaignProjection, waves: dict[str, WaveProjection]
) -> bool:
    for action in campaign.actions.values():
        wave = waves.get(f"wave-{action.action_key}")
        if (
            action.action_kind in {"submit-wave", "retry-shard"}
            and action.status != "failed"
            and (wave is None or wave.status != "closed")
        ):
            return False
    return all(wave.status == "closed" for wave in waves.values())


def _decision(
    lock: CampaignLock,
    status: TerminalStatus,
    counts: ProjectionCounts,
    reason: str,
) -> TerminalDecision:
    markers = {
        "completed": "_SUCCESS",
        "partial": "_PARTIAL",
        "failed": "_FAILED",
        "cancelled": "_CANCELLED",
    }
    marker = cast(
        Literal["_SUCCESS", "_PARTIAL", "_FAILED", "_CANCELLED"], markers[status]
    )
    return TerminalDecision(
        status=status,
        marker=marker,
        summary_path=f"{lock.artifact_prefix}/campaign-summary.json",
        marker_path=f"{lock.artifact_prefix}/{marker}",
        reason=reason,
        counts=counts,
    )
