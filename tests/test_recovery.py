from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from harbor_hf.campaigns import (
    CampaignLock,
    CampaignRecoveryPolicy,
    build_campaign_lock,
    build_campaign_plan,
)
from harbor_hf.control import (
    ActionOutcomePayload,
    ActionReservedPayload,
    CampaignEvent,
    CampaignSubmittedPayload,
    CancellationPayload,
    EventKind,
    EventPayload,
    ExecutionOutcomePayload,
    ExecutionStartedPayload,
    LifecyclePayload,
    RetryCategory,
    SubjectType,
    TerminalPayload,
    WaveLifecyclePayload,
    new_event,
)
from harbor_hf.models import ExperimentSpec
from harbor_hf.reconciler import (
    AdmissionLimits,
    AdmissionUsage,
    DeploymentAdmission,
    ReconcileContext,
    plan_reconciliation,
)
from harbor_hf.recovery import (
    durable_cancellation_event,
    durable_shard_retry_event,
    project_recovery,
)

NOW = datetime(2026, 7, 14, tzinfo=UTC)


def _campaign(
    remote_spec: ExperimentSpec,
    *,
    tasks: int = 1,
    max_trials_per_shard: int = 64,
    max_shards_per_wave: int = 8,
    max_physical_executions_per_trial: int = 3,
    retry_base_seconds: int = 10,
    retry_max_seconds: int = 60,
    cancellation_grace_seconds: int = 0,
    spend_cap_microusd: int | None = None,
) -> tuple[CampaignLock, CampaignEvent]:
    task_digests = {
        f"task-{index}": f"sha256:{index:064x}" for index in range(1, tasks + 1)
    }
    spec = remote_spec.model_copy(
        update={
            "benchmark": remote_spec.benchmark.model_copy(
                update={"task_names": ["task-*"], "task_digests": task_digests}
            ),
            "execution": remote_spec.execution.model_copy(
                update={
                    "max_trials_per_shard": max_trials_per_shard,
                    "max_shards_per_wave": max_shards_per_wave,
                }
            ),
        }
    )
    recovery_policy = CampaignRecoveryPolicy(
        max_physical_executions_per_trial=max_physical_executions_per_trial,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        cancellation_grace_seconds=cancellation_grace_seconds,
        spend_cap_microusd=spend_cap_microusd,
    )
    lock = build_campaign_lock(
        build_campaign_plan(spec, recovery_policy=recovery_policy),
        "campaign-recovery",
        clock=lambda: NOW,
    )
    submitted = _event(
        lock,
        1,
        "campaign",
        lock.campaign_id,
        "campaign.submitted",
        CampaignSubmittedPayload(plan_digest=lock.plan_digest),
    )
    return lock, submitted


def _event(
    lock: CampaignLock,
    sequence: int,
    subject_type: SubjectType,
    subject_id: str,
    kind: EventKind,
    payload: EventPayload,
) -> CampaignEvent:
    return new_event(
        subject_type=subject_type,
        subject_id=subject_id,
        kind=kind,
        producer="reconciler",
        payload=payload,
        clock=lambda: NOW + timedelta(seconds=sequence),
        identifier=lambda: f"{sequence:032x}",
    )


def _execution_events(
    lock: CampaignLock,
    sequence: int,
    *,
    execution_id: str,
    attempt: int,
    category: RetryCategory | None,
    spend: int = 0,
) -> list[CampaignEvent]:
    shard = lock.runs[0].shards[0]
    trial = shard.trials[0]
    started = _event(
        lock,
        sequence,
        "execution",
        execution_id,
        "execution.started",
        ExecutionStartedPayload(
            trial_id=trial.trial_id,
            shard_id=shard.shard_id,
            physical_attempt=attempt,
            wave_id="wave-one",
        ),
    )
    completed_kind = "execution.completed" if category is None else "execution.failed"
    outcome = _event(
        lock,
        sequence + 1,
        "execution",
        execution_id,
        completed_kind,
        ExecutionOutcomePayload(
            trial_id=trial.trial_id,
            physical_attempt=attempt,
            category=category,
            spend_microusd=spend,
        ),
    )
    return [started, outcome]


def _wave_event(
    lock: CampaignLock, sequence: int, phase: str, *, shard_index: int = 0
) -> CampaignEvent:
    run = lock.runs[0]
    shard = run.shards[shard_index]
    return _event(
        lock,
        sequence,
        "wave",
        "wave-one",
        cast(EventKind, f"wave.{phase}"),
        WaveLifecyclePayload(
            deployment_digest=run.deployment_digest,
            provider="hf-inference-endpoints",
            shard_ids=[shard.shard_id],
            estimated_cost_microusd=100,
        ),
    )


def _cancel_event(lock: CampaignLock, sequence: int = 2) -> CampaignEvent:
    return _event(
        lock,
        sequence,
        "campaign",
        lock.campaign_id,
        "campaign.cancel-requested",
        CancellationPayload(reason="operator request"),
    )


def test_logical_trial_keeps_identity_across_physical_retry(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    first = _execution_events(
        lock,
        2,
        execution_id="execution-one",
        attempt=1,
        category="transient",
        spend=11,
    )

    projection, waiting = plan_reconciliation(
        lock, [submitted, *first], now=NOW + timedelta(seconds=4)
    )
    trial = next(iter(projection.trials.values()))

    assert trial.status == "retry_wait"
    assert list(trial.executions) == ["execution-one"]
    assert waiting.actions == []
    assert waiting.blocked[0].reason == "backoff"

    retry_at = trial.retry_not_before
    assert retry_at is not None
    _projection, ready = plan_reconciliation(lock, [submitted, *first], now=retry_at)

    assert ready.actions[0].kind == "retry-shard"
    assert ready.actions[0].trial_ids == [trial.trial_id]

    second = _execution_events(
        lock,
        20,
        execution_id="execution-two",
        attempt=2,
        category=None,
        spend=13,
    )
    completed = project_recovery(lock, [submitted, *first, *second])

    assert completed.trials[trial.trial_id].status == "complete"
    assert completed.counts.physical_retries == 1
    assert completed.spend_microusd == 24


def test_manual_retry_only_overrides_the_generation_it_requested(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    first = _execution_events(
        lock, 2, execution_id="execution-one", attempt=1, category="transient"
    )
    manual, _created = durable_shard_retry_event(
        lock,
        [submitted, *first],
        lock.runs[0].shards[0].shard_id,
        "retry now",
        clock=lambda: NOW + timedelta(seconds=5),
    )
    second = _execution_events(
        lock, 20, execution_id="execution-two", attempt=2, category="rate-limit"
    )

    with_manual = project_recovery(lock, [submitted, *first, manual, *second])
    without_manual = project_recovery(lock, [submitted, *first, *second])
    trial_id = lock.runs[0].shards[0].trials[0].trial_id

    assert with_manual.trials[trial_id].retry_not_before == (
        without_manual.trials[trial_id].retry_not_before
    )
    retry_not_before = with_manual.trials[trial_id].retry_not_before
    assert retry_not_before is not None
    assert retry_not_before > manual.observed_at


def test_reserved_retry_targets_are_excluded_from_later_retry_wave(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec, tasks=2, max_trials_per_shard=2)
    shard = lock.runs[0].shards[0]

    def failure(
        trial_index: int, sequence: int, execution_id: str
    ) -> list[CampaignEvent]:
        trial = shard.trials[trial_index]
        return [
            _event(
                lock,
                sequence,
                "execution",
                execution_id,
                "execution.started",
                ExecutionStartedPayload(
                    trial_id=trial.trial_id,
                    shard_id=shard.shard_id,
                    physical_attempt=1,
                ),
            ),
            _event(
                lock,
                sequence + 1,
                "execution",
                execution_id,
                "execution.failed",
                ExecutionOutcomePayload(
                    trial_id=trial.trial_id,
                    physical_attempt=1,
                    category="transient",
                ),
            ),
        ]

    first = failure(0, 2, "execution-one")
    _projection, first_plan = plan_reconciliation(
        lock, [submitted, *first], now=NOW + timedelta(seconds=20)
    )
    retry = next(
        action for action in first_plan.actions if action.kind == "retry-shard"
    )
    reserved = _event(
        lock,
        12,
        "campaign",
        lock.campaign_id,
        "action.reserved",
        ActionReservedPayload(
            action_id=retry.action_id,
            action_key=retry.action_key,
            action_kind=retry.kind,
            target_ids=retry.target_ids,
        ),
    )
    second = failure(1, 20, "execution-two")

    _projection, plan = plan_reconciliation(
        lock,
        [submitted, *first, reserved, *second],
        now=NOW + timedelta(seconds=40),
    )

    later = next(action for action in plan.actions if action.kind == "retry-shard")
    assert later.trial_ids == [shard.trials[1].trial_id]
    assert shard.trials[0].trial_id not in later.trial_ids


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("configuration", "failed_infrastructure"),
        ("authentication", "failed_infrastructure"),
        ("cleanup", "failed_infrastructure"),
    ],
)
def test_terminal_failure_categories_never_retry(
    remote_spec: ExperimentSpec,
    category: RetryCategory,
    expected: str,
) -> None:
    lock, submitted = _campaign(remote_spec)
    failed = _execution_events(
        lock, 2, execution_id="execution-one", attempt=1, category=category
    )

    projection, plan = plan_reconciliation(
        lock, [submitted, *failed], now=NOW + timedelta(days=1)
    )

    assert next(iter(projection.trials.values())).status == expected
    assert all(action.kind != "retry-shard" for action in plan.actions)


def test_benchmark_failure_retries_then_becomes_scored_invalid(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    first = _execution_events(
        lock, 2, execution_id="execution-one", attempt=1, category="benchmark"
    )

    projected = project_recovery(lock, [submitted, *first])
    retry_at = next(iter(projected.trials.values())).retry_not_before
    assert retry_at is not None
    _projection, retry = plan_reconciliation(lock, [submitted, *first], now=retry_at)
    assert [action.kind for action in retry.actions] == ["retry-shard"]

    exhausted_lock, exhausted_submitted = _campaign(
        remote_spec, max_physical_executions_per_trial=1
    )
    exhausted = _execution_events(
        exhausted_lock,
        2,
        execution_id="execution-one",
        attempt=1,
        category="benchmark",
    )
    exhausted_projection, exhausted_plan = plan_reconciliation(
        exhausted_lock,
        [exhausted_submitted, *exhausted],
        now=NOW + timedelta(days=1),
    )
    exhausted_trial = next(iter(exhausted_projection.trials.values()))
    assert exhausted_trial.status == "invalid"
    assert exhausted_trial.outcome == "benchmark_failed"
    assert exhausted_plan.terminal_decision is not None
    assert exhausted_plan.terminal_decision.status == "failed"


@pytest.mark.parametrize(
    "category", ["lost", "transient", "quota", "rate-limit", "ambiguous"]
)
def test_retry_budget_exhaustion_is_terminal(
    remote_spec: ExperimentSpec, category: RetryCategory
) -> None:
    lock, submitted = _campaign(remote_spec, max_physical_executions_per_trial=1)
    failed = _execution_events(
        lock, 2, execution_id="execution-one", attempt=1, category=category
    )

    projection, plan = plan_reconciliation(
        lock, [submitted, *failed], now=NOW + timedelta(days=1)
    )

    terminal_trial = next(iter(projection.trials.values()))
    assert terminal_trial.status == "failed_infrastructure"
    assert terminal_trial.outcome == "infrastructure_exhausted"
    assert all(action.kind != "retry-shard" for action in plan.actions)
    assert plan.terminal_decision is not None
    assert plan.terminal_decision.status == "failed"


def test_cancellation_converts_retry_wait_trials_to_terminal_cancelled_counts(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    failed = _execution_events(
        lock, 2, execution_id="execution-one", attempt=1, category="transient"
    )

    projection = project_recovery(
        lock, [submitted, *failed, _cancel_event(lock, sequence=5)]
    )

    assert projection.terminal_decision is not None
    assert projection.terminal_decision.status == "cancelled"
    assert projection.terminal_decision.counts.retrying == 0
    assert projection.terminal_decision.counts.cancelled == 1


def test_recorded_cancellation_keeps_normalized_retry_counts_after_reload(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    failed = _execution_events(
        lock, 2, execution_id="execution-one", attempt=1, category="transient"
    )
    cancelled = _cancel_event(lock, sequence=5)
    terminal = _event(
        lock,
        6,
        "campaign",
        lock.campaign_id,
        "campaign.cancelled",
        TerminalPayload(message="cancellation drained and cleaned"),
    )

    projection = project_recovery(lock, [submitted, *failed, cancelled, terminal])

    assert projection.terminal_decision is not None
    assert projection.terminal_decision.status == "cancelled"
    assert projection.terminal_decision.counts.retrying == 0
    assert projection.terminal_decision.counts.cancelled == 1


def test_valid_completed_trial_is_not_retried_after_reconcile_kill(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec, tasks=2, max_trials_per_shard=2)
    shard = lock.runs[0].shards[0]
    completed = _event(
        lock,
        2,
        "trial",
        shard.trials[0].trial_id,
        "trial.complete",
        LifecyclePayload(parent_id=shard.shard_id),
    )
    lost_start = _event(
        lock,
        3,
        "execution",
        "execution-lost",
        "execution.started",
        ExecutionStartedPayload(
            trial_id=shard.trials[1].trial_id,
            shard_id=shard.shard_id,
            physical_attempt=1,
        ),
    )
    lost = _event(
        lock,
        4,
        "execution",
        "execution-lost",
        "execution.failed",
        ExecutionOutcomePayload(
            trial_id=shard.trials[1].trial_id,
            physical_attempt=1,
            category="lost",
        ),
    )
    projected = project_recovery(lock, [submitted, completed, lost_start, lost])
    retry_at = projected.trials[shard.trials[1].trial_id].retry_not_before
    assert retry_at is not None

    _projection, plan = plan_reconciliation(
        lock, [submitted, completed, lost_start, lost], now=retry_at
    )

    retry = next(action for action in plan.actions if action.kind == "retry-shard")
    assert retry.trial_ids == [shard.trials[1].trial_id]
    assert shard.trials[0].trial_id not in retry.trial_ids


@pytest.mark.parametrize(
    "phase", ["acquiring", "provisioning", "ready", "active", "draining", "cleaning"]
)
def test_cancellation_at_every_wave_phase_is_cleanup_first(
    remote_spec: ExperimentSpec, phase: str
) -> None:
    lock, submitted = _campaign(remote_spec)

    _projection, plan = plan_reconciliation(
        lock, [submitted, _cancel_event(lock), _wave_event(lock, 3, phase)]
    )

    assert [action.kind for action in plan.actions[:2]] == [
        "cancel-wave",
        "cleanup-wave",
    ]
    assert all(
        action.kind not in {"submit-wave", "retry-shard"} for action in plan.actions
    )


def test_active_execution_is_cancelled_before_wave_cleanup(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    started = _execution_events(
        lock, 3, execution_id="execution-one", attempt=1, category=None
    )[0]

    _projection, plan = plan_reconciliation(
        lock,
        [submitted, _cancel_event(lock), _wave_event(lock, 5, "active"), started],
    )

    assert [action.kind for action in plan.actions[:3]] == [
        "cancel-execution",
        "cancel-wave",
        "cleanup-wave",
    ]


def test_cancellation_grace_drains_before_force_cancelling(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec, cancellation_grace_seconds=60)
    started = _execution_events(
        lock, 4, execution_id="execution-one", attempt=1, category=None
    )[0]
    events = [
        submitted,
        _cancel_event(lock),
        _wave_event(lock, 3, "active"),
        started,
    ]

    projection, draining = plan_reconciliation(
        lock, events, now=NOW + timedelta(seconds=30)
    )

    assert projection.cancel_requested_at == NOW + timedelta(seconds=2)
    assert [action.kind for action in draining.actions] == ["drain-wave"]

    _projection, forced = plan_reconciliation(
        lock, events, now=NOW + timedelta(seconds=62)
    )

    assert [action.kind for action in forced.actions[:3]] == [
        "cancel-execution",
        "cancel-wave",
        "cleanup-wave",
    ]


def test_cleanup_failure_requires_manual_intervention(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)

    projection, plan = plan_reconciliation(
        lock, [submitted, _wave_event(lock, 2, "cleanup-failed")]
    )

    assert projection.waves["wave-one"].status == "cleanup_failed"
    assert [action.kind for action in plan.actions] == ["manual-intervention"]
    assert plan.terminal_decision is None


def test_durable_cancellation_request_is_idempotent(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    first, created = durable_cancellation_event(
        lock,
        [submitted],
        "first reason",
        clock=lambda: NOW + timedelta(seconds=2),
        identifier=lambda: "a" * 32,
    )
    repeated, created_again = durable_cancellation_event(
        lock,
        [first, submitted, first],
        "different repeated reason",
        identifier=lambda: "b" * 32,
    )

    assert created
    assert not created_again
    assert repeated == first


def test_durable_cancellation_request_has_stable_identity_and_requested_time(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    observed_at = NOW + timedelta(seconds=19)

    event, created = durable_cancellation_event(
        lock,
        [submitted],
        "operator stop",
        clock=lambda: observed_at,
    )

    identity = hashlib.sha256(f"{lock.campaign_id}:cancel".encode()).hexdigest()[:32]
    assert created is True
    assert event.event_id == f"evt-{identity}"
    assert event.subject_type == "campaign"
    assert event.subject_id == lock.campaign_id
    assert event.kind == "campaign.cancel-requested"
    assert event.producer == "cli"
    assert event.observed_at == observed_at
    assert event.payload == CancellationPayload(reason="operator stop")


def test_durable_shard_retry_request_skips_current_backoff(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    failed = _execution_events(
        lock,
        2,
        execution_id="execution-one",
        attempt=1,
        category="transient",
    )
    shard_id = lock.runs[0].shards[0].shard_id
    request, created = durable_shard_retry_event(
        lock,
        [submitted, *failed],
        shard_id,
        "operator retry",
        clock=lambda: NOW + timedelta(seconds=5),
    )
    repeated, created_again = durable_shard_retry_event(
        lock,
        [submitted, *failed, request],
        shard_id,
        "different reason",
    )

    projection, plan = plan_reconciliation(
        lock, [submitted, *failed, request], now=NOW + timedelta(seconds=5)
    )

    assert created
    assert not created_again
    assert repeated == request
    assert projection.counts.retrying == 1
    assert [action.kind for action in plan.actions] == ["retry-shard"]


def test_legacy_shard_retry_request_preserves_its_event_time_generation(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    failed = _execution_events(
        lock,
        2,
        execution_id="execution-one",
        attempt=1,
        category="transient",
    )
    shard_id = lock.runs[0].shards[0].shard_id
    request, _created = durable_shard_retry_event(
        lock,
        [submitted, *failed],
        shard_id,
        "operator retry",
        clock=lambda: NOW + timedelta(seconds=5),
    )
    legacy = request.model_copy(
        update={"payload": request.payload.model_copy(update={"trial_generations": {}})}
    )

    projection, plan = plan_reconciliation(
        lock, [submitted, *failed, legacy], now=NOW + timedelta(seconds=5)
    )

    assert projection.counts.retrying == 1
    assert [action.kind for action in plan.actions] == ["retry-shard"]


def test_multiple_legacy_retry_requests_are_projected_without_recursion(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, submitted = _campaign(remote_spec)
    failed = _execution_events(
        lock,
        2,
        execution_id="execution-one",
        attempt=1,
        category="transient",
    )
    shard_id = lock.runs[0].shards[0].shard_id
    request, _created = durable_shard_retry_event(
        lock,
        [submitted, *failed],
        shard_id,
        "operator retry",
        clock=lambda: NOW + timedelta(seconds=5),
    )
    legacy = request.model_copy(
        update={"payload": request.payload.model_copy(update={"trial_generations": {}})}
    )
    repeated = legacy.model_copy(
        update={
            "event_id": "evt-" + "f" * 32,
            "observed_at": NOW + timedelta(seconds=6),
        }
    )
    monkeypatch.setattr(
        "harbor_hf.recovery.project_recovery",
        lambda *_args, **_kwargs: pytest.fail("legacy migration must not re-project"),
    )

    projection, plan = plan_reconciliation(
        lock,
        [submitted, *failed, legacy, repeated],
        now=repeated.observed_at,
    )

    assert projection.counts.retrying == 1
    assert [action.kind for action in plan.actions] == ["retry-shard"]


def test_durable_shard_retry_request_has_generation_identity_and_exact_payload(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    failed = _execution_events(
        lock,
        2,
        execution_id="execution-one",
        attempt=1,
        category="transient",
    )
    shard = lock.runs[0].shards[0]
    trial = shard.trials[0]
    observed_at = NOW + timedelta(seconds=11)

    event, created = durable_shard_retry_event(
        lock,
        [submitted, *failed],
        shard.shard_id,
        "retry now",
        clock=lambda: observed_at,
    )

    generation = f"{trial.trial_id}:1"
    identity = hashlib.sha256(
        f"{lock.campaign_id}:{shard.shard_id}:{generation}".encode()
    ).hexdigest()[:32]
    assert created is True
    assert event.event_id == f"evt-{identity}"
    assert event.kind == "campaign.shard-retry-requested"
    assert event.subject_id == lock.campaign_id
    assert event.observed_at == observed_at
    assert event.payload.model_dump(mode="json") == {
        "shard_id": shard.shard_id,
        "reason": "retry now",
        "trial_generations": {trial.trial_id: 1},
    }


def test_retry_request_rejects_generations_outside_its_event_time_shard_state(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec, tasks=2, max_trials_per_shard=1)
    failed = _execution_events(
        lock,
        2,
        execution_id="execution-one",
        attempt=1,
        category="transient",
    )
    first_shard, second_shard = lock.runs[0].shards
    first_trial = first_shard.trials[0]
    second_trial = second_shard.trials[0]
    request, _created = durable_shard_retry_event(
        lock,
        [submitted, *failed],
        first_shard.shard_id,
        "retry now",
        clock=lambda: NOW + timedelta(seconds=5),
    )

    invalid_mappings = [
        {second_trial.trial_id: 0},
        {"unknown-trial": 0},
        {first_trial.trial_id: 2},
    ]
    for mapping in invalid_mappings:
        invalid = request.model_copy(
            update={
                "payload": request.payload.model_copy(
                    update={"trial_generations": mapping}
                )
            }
        )
        with pytest.raises(
            ValueError,
            match="retry request generations do not match the requested shard state",
        ):
            project_recovery(lock, [submitted, *failed, invalid])


def test_retry_generation_binding_tolerates_cross_host_clock_skew(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    failed = _execution_events(
        lock,
        2,
        execution_id="execution-one",
        attempt=1,
        category="transient",
    )
    shard = lock.runs[0].shards[0]
    request, _created = durable_shard_retry_event(
        lock,
        [submitted, *failed],
        shard.shard_id,
        "retry now",
        clock=lambda: NOW + timedelta(seconds=4),
    )
    skewed = request.model_copy(update={"observed_at": NOW + timedelta(seconds=2)})

    projection = project_recovery(lock, [submitted, *failed, skewed])

    trial = projection.trials[shard.trials[0].trial_id]
    assert trial.status == "retry_wait"
    assert trial.retry_not_before == skewed.observed_at


def test_durable_shard_retry_identity_includes_every_eligible_trial(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec, tasks=2, max_trials_per_shard=2)
    shard = lock.runs[0].shards[0]
    failures: list[CampaignEvent] = []
    for index, trial in enumerate(shard.trials, 1):
        execution_id = f"execution-{trial.trial_id}"
        failures.extend(
            [
                _event(
                    lock,
                    index * 3,
                    "execution",
                    execution_id,
                    "execution.started",
                    ExecutionStartedPayload(
                        trial_id=trial.trial_id,
                        shard_id=shard.shard_id,
                        physical_attempt=1,
                        wave_id="wave-one",
                    ),
                ),
                _event(
                    lock,
                    index * 3 + 1,
                    "execution",
                    execution_id,
                    "execution.failed",
                    ExecutionOutcomePayload(
                        trial_id=trial.trial_id,
                        physical_attempt=1,
                        category="transient",
                    ),
                ),
            ]
        )

    event, created = durable_shard_retry_event(
        lock, [submitted, *failures], shard.shard_id, "retry both"
    )

    generation = ",".join(f"{trial.trial_id}:1" for trial in shard.trials)
    identity = hashlib.sha256(
        f"{lock.campaign_id}:{shard.shard_id}:{generation}".encode()
    ).hexdigest()[:32]
    assert created is True
    assert event.event_id == f"evt-{identity}"


def test_durable_shard_retry_rejects_conflicting_event_identity(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    failed = _execution_events(
        lock,
        2,
        execution_id="execution-one",
        attempt=1,
        category="transient",
    )
    shard = lock.runs[0].shards[0]
    retry, _created = durable_shard_retry_event(
        lock, [submitted, *failed], shard.shard_id, "retry"
    )
    conflict = _event(
        lock,
        5,
        "run",
        lock.runs[0].run_id,
        "run.queued",
        LifecyclePayload(parent_id=lock.campaign_id),
    ).model_copy(update={"event_id": retry.event_id})

    with pytest.raises(ValueError) as captured:
        durable_shard_retry_event(
            lock, [submitted, *failed, conflict], shard.shard_id, "retry"
        )

    assert str(captured.value) == "retry event identity conflicts"


@pytest.mark.parametrize(
    ("transition", "message"),
    [
        ("terminal", "a terminal campaign cannot be retried"),
        ("cancel_requested", "a cancelling campaign cannot be retried"),
        ("draining", "a cancelling campaign cannot be retried"),
        ("unknown", "unknown campaign shard: shard-unknown"),
        ("planned", "shard has no retryable logical trials"),
    ],
)
def test_durable_shard_retry_rejects_ineligible_recovery_transitions(
    remote_spec: ExperimentSpec, transition: str, message: str
) -> None:
    lock, submitted = _campaign(remote_spec)
    events = [submitted]
    shard_id = lock.runs[0].shards[0].shard_id
    if transition == "terminal":
        events.append(
            _event(
                lock,
                2,
                "campaign",
                lock.campaign_id,
                "campaign.failed",
                TerminalPayload(message="failed"),
            )
        )
    elif transition == "cancel_requested":
        events.append(_cancel_event(lock))
    elif transition == "draining":
        events.append(
            _event(
                lock,
                2,
                "campaign",
                lock.campaign_id,
                "campaign.draining",
                LifecyclePayload(parent_id=lock.campaign_id),
            )
        )
    elif transition == "unknown":
        shard_id = "shard-unknown"

    with pytest.raises(ValueError) as captured:
        durable_shard_retry_event(lock, events, shard_id, "retry")

    assert str(captured.value) == message


@pytest.mark.parametrize(
    ("limits", "usage", "cap", "reason"),
    [
        (
            AdmissionLimits(global_active_waves=1),
            AdmissionUsage(global_active_waves=1),
            None,
            "global-budget",
        ),
        (
            AdmissionLimits(deployment_active_waves=1),
            "deployment",
            None,
            "deployment-budget",
        ),
        (AdmissionLimits(provider_active_waves=1), "provider", None, "provider-budget"),
        (AdmissionLimits(campaign_active_waves=1), "campaign", None, "campaign-budget"),
        (AdmissionLimits(), AdmissionUsage(), 99, "spend-cap"),
    ],
)
def test_all_admission_budgets_are_hard_limits(
    remote_spec: ExperimentSpec,
    limits: AdmissionLimits,
    usage: AdmissionUsage | str,
    cap: int | None,
    reason: str,
) -> None:
    lock, submitted = _campaign(remote_spec, spend_cap_microusd=cap)
    digest = lock.runs[0].deployment_digest
    if usage == "deployment":
        usage = AdmissionUsage(deployment_active_waves={digest: 1})
    elif usage == "provider":
        usage = AdmissionUsage(provider_active_waves={"provider-one": 1})
    elif usage == "campaign":
        usage = AdmissionUsage(campaign_active_waves={lock.campaign_id: 1})
    assert isinstance(usage, AdmissionUsage)
    context = ReconcileContext(
        limits=limits,
        usage=usage,
        deployments={
            digest: DeploymentAdmission(
                provider="provider-one", estimated_wave_cost_microusd=100
            )
        },
    )

    _projection, plan = plan_reconciliation(lock, [submitted], context=context)

    assert plan.actions == []
    assert plan.blocked[0].reason == reason


def test_spend_cap_fails_closed_without_deployment_estimate(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec, spend_cap_microusd=1_000)

    _projection, plan = plan_reconciliation(lock, [submitted])

    assert plan.actions == []
    assert plan.blocked[0].reason == "spend-estimate-missing"


def test_cleanup_bypasses_budgets_and_action_limit_before_billable_work(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec, tasks=2, max_trials_per_shard=1)
    first_shard = lock.runs[0].shards[0]
    completed = _event(
        lock,
        2,
        "trial",
        first_shard.trials[0].trial_id,
        "trial.complete",
        LifecyclePayload(parent_id=first_shard.shard_id),
    )
    draining = _wave_event(lock, 3, "draining")
    context = ReconcileContext(
        limits=AdmissionLimits(action_limit=1, global_active_waves=1),
        usage=AdmissionUsage(global_active_waves=1),
    )

    _projection, plan = plan_reconciliation(
        lock, [submitted, completed, draining], context=context
    )

    assert [action.kind for action in plan.actions] == ["cleanup-wave"]


@pytest.mark.parametrize(
    ("trial_kinds", "cancelled", "expected", "marker"),
    [
        (["trial.complete", "trial.complete"], False, "completed", "_SUCCESS"),
        (["trial.complete", "trial.invalid"], False, "completed", "_SUCCESS"),
        (["trial.invalid", "trial.invalid"], False, "failed", "_FAILED"),
        (["trial.cancelled", "trial.cancelled"], True, "cancelled", "_CANCELLED"),
    ],
)
def test_terminal_summary_decisions(
    remote_spec: ExperimentSpec,
    trial_kinds: list[EventKind],
    cancelled: bool,
    expected: str,
    marker: str,
) -> None:
    lock, submitted = _campaign(remote_spec, tasks=2, max_trials_per_shard=2)
    shard = lock.runs[0].shards[0]
    events = [submitted]
    if cancelled:
        events.append(_cancel_event(lock))
    pairs = zip(shard.trials, trial_kinds, strict=True)
    for index, (trial, kind) in enumerate(pairs, 3):
        events.append(
            _event(
                lock,
                index,
                "trial",
                trial.trial_id,
                kind,
                LifecyclePayload(parent_id=shard.shard_id),
            )
        )

    projection, plan = plan_reconciliation(lock, events)

    assert projection.terminal_decision is not None
    assert projection.terminal_decision.status == expected
    assert projection.terminal_decision.marker == marker
    assert [action.kind for action in plan.actions] == ["publish-summary"]


def test_queued_cancellation_counts_unstarted_trials_as_cancelled(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec, tasks=2, max_trials_per_shard=2)

    projection, plan = plan_reconciliation(lock, [submitted, _cancel_event(lock)])

    assert projection.counts.planned == 2
    assert projection.counts.cancelled == 0
    assert projection.terminal_decision is not None
    assert projection.terminal_decision.status == "cancelled"
    assert projection.terminal_decision.counts.planned == 0
    assert projection.terminal_decision.counts.cancelled == 2
    assert [action.kind for action in plan.actions] == ["publish-summary"]


def test_partial_cancellation_preserves_completed_trial(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec, tasks=2, max_trials_per_shard=2)
    shard = lock.runs[0].shards[0]
    completed = _event(
        lock,
        3,
        "trial",
        shard.trials[0].trial_id,
        "trial.complete",
        LifecyclePayload(parent_id=shard.shard_id),
    )

    projection, plan = plan_reconciliation(
        lock, [submitted, _cancel_event(lock), completed]
    )

    assert projection.counts.complete == 1
    assert projection.counts.planned == 1
    assert projection.terminal_decision is not None
    assert projection.terminal_decision.status == "partial"
    assert projection.terminal_decision.counts.complete == 1
    assert projection.terminal_decision.counts.cancelled == 1
    assert projection.terminal_decision.counts.planned == 0
    assert [action.kind for action in plan.actions] == ["publish-summary"]


def test_randomized_duplicate_and_out_of_order_replay_converges(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    history = [
        submitted,
        _wave_event(lock, 2, "active"),
        *_execution_events(
            lock, 3, execution_id="execution-one", attempt=1, category=None
        ),
        _wave_event(lock, 5, "cleaning"),
        _wave_event(lock, 6, "closed"),
    ]
    expected = project_recovery(lock, history)

    for seed in range(100):
        rng = random.Random(seed)
        replay = [*history, *(rng.choice(history) for _ in range(20))]
        rng.shuffle(replay)

        assert project_recovery(lock, replay) == expected


def test_terminal_summary_waits_for_unobserved_reserved_wave(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    _projection, initial = plan_reconciliation(lock, [submitted])
    action = initial.actions[0]
    reserved = _event(
        lock,
        2,
        "campaign",
        lock.campaign_id,
        "action.reserved",
        ActionReservedPayload(
            action_id=action.action_id,
            action_key=action.action_key,
            action_kind=action.kind,
            target_ids=action.target_ids,
        ),
    )
    trial = lock.runs[0].shards[0].trials[0]
    completed = _event(
        lock,
        3,
        "trial",
        trial.trial_id,
        "trial.complete",
        LifecyclePayload(),
    )

    projection = project_recovery(lock, [submitted, reserved, completed])

    assert projection.terminal_decision is None


def test_faulted_history_rejects_wave_state_regression(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)

    with pytest.raises(ValueError) as captured:
        project_recovery(
            lock,
            [
                submitted,
                _wave_event(lock, 2, "active"),
                _wave_event(lock, 3, "ready"),
            ],
        )
    assert str(captured.value) == "invalid wave transition: active -> ready"


@pytest.mark.parametrize(
    ("terminal_phase", "synthetic_phase", "expected"),
    [
        ("closed", "draining", "closed"),
        ("closed", "cleaning", "closed"),
        ("cleanup-failed", "draining", "cleanup_failed"),
    ],
)
def test_late_terminal_evidence_supersedes_stale_reconciler_transition(
    remote_spec: ExperimentSpec,
    terminal_phase: str,
    synthetic_phase: str,
    expected: str,
) -> None:
    lock, submitted = _campaign(remote_spec)
    terminal = _wave_event(lock, 2, terminal_phase).model_copy(
        update={"producer": "wave-controller"}
    )
    synthetic = _wave_event(lock, 3, synthetic_phase)

    projection = project_recovery(lock, [submitted, terminal, synthetic])

    assert projection.waves["wave-one"].status == expected


def test_cleanup_retry_can_advance_worker_cleanup_failure(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    cleanup_failed = _wave_event(lock, 2, "cleanup-failed").model_copy(
        update={"producer": "wave-controller"}
    )
    cleanup_retry = _wave_event(lock, 3, "cleaning")

    projection = project_recovery(lock, [submitted, cleanup_failed, cleanup_retry])

    assert projection.waves["wave-one"].status == "cleaning"


def test_wave_transition_matrix_is_exhaustive(remote_spec: ExperimentSpec) -> None:
    lock, submitted = _campaign(remote_spec)
    allowed = {
        "acquiring": {"provisioning", "draining", "cleaning", "cleanup-failed"},
        "provisioning": {"ready", "draining", "cleaning", "cleanup-failed"},
        "ready": {"active", "draining", "cleaning", "cleanup-failed"},
        "active": {"draining", "cleaning", "cleanup-failed"},
        "draining": {"cleaning", "closed", "cleanup-failed"},
        "cleaning": {"closed", "cleanup-failed"},
        "cleanup-failed": {"cleaning", "closed"},
        "closed": set(),
    }
    phases = list(allowed)

    for previous in phases:
        for current in phases:
            history = [
                submitted,
                _wave_event(lock, 2, previous),
                _wave_event(lock, 3, current).model_copy(
                    update={"producer": "wave-controller"}
                ),
            ]
            if current == previous or current in allowed[previous]:
                projection = project_recovery(lock, history)
                assert projection.waves["wave-one"].status == current.replace("-", "_")
            else:
                with pytest.raises(ValueError, match="invalid wave transition"):
                    project_recovery(lock, history)


def test_faulted_history_rejects_skipped_physical_attempt(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    second = _execution_events(
        lock, 2, execution_id="execution-two", attempt=2, category="lost"
    )

    with pytest.raises(ValueError) as captured:
        project_recovery(lock, [submitted, *second])
    assert str(captured.value) == "physical execution attempts must be contiguous"


def test_faulted_history_rejects_early_parent_terminal_state(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    shard = lock.runs[0].shards[0]
    early = _event(
        lock,
        2,
        "shard",
        shard.shard_id,
        "shard.complete",
        LifecyclePayload(parent_id=lock.runs[0].run_id),
    )

    with pytest.raises(ValueError) as captured:
        project_recovery(lock, [submitted, early])
    assert str(captured.value) == "shard became terminal before its children"


def test_completed_trial_cannot_be_physically_reexecuted(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    first = _execution_events(
        lock, 2, execution_id="execution-one", attempt=1, category=None
    )
    second = _execution_events(
        lock, 4, execution_id="execution-two", attempt=2, category="lost"
    )

    with pytest.raises(ValueError) as captured:
        project_recovery(lock, [submitted, *first, *second])
    assert str(captured.value) == (
        "a completed logical trial was physically re-executed"
    )


def test_execution_start_identity_faults_are_rejected(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    shard = lock.runs[0].shards[0]
    trial = shard.trials[0]

    for trial_id, shard_id in [
        ("unknown-trial", shard.shard_id),
        (trial.trial_id, "unknown-shard"),
    ]:
        invalid = _event(
            lock,
            2,
            "execution",
            "execution-invalid",
            "execution.started",
            ExecutionStartedPayload(
                trial_id=trial_id,
                shard_id=shard_id,
                physical_attempt=1,
            ),
        )
        with pytest.raises(ValueError) as captured:
            project_recovery(lock, [submitted, invalid])
        assert str(captured.value) == ("execution references an unknown trial or shard")

    started = _execution_events(
        lock, 2, execution_id="execution-one", attempt=1, category="lost"
    )[0]
    duplicate_start = started.model_copy(
        update={
            "event_id": "evt-" + "a" * 32,
            "observed_at": NOW + timedelta(seconds=3),
        }
    )
    with pytest.raises(ValueError) as captured:
        project_recovery(lock, [submitted, started, duplicate_start])
    assert str(captured.value) == "execution started more than once: execution-one"

    duplicate_attempt = _event(
        lock,
        3,
        "execution",
        "execution-two",
        "execution.started",
        ExecutionStartedPayload(
            trial_id=trial.trial_id,
            shard_id=shard.shard_id,
            physical_attempt=1,
        ),
    )
    with pytest.raises(ValueError) as captured:
        project_recovery(lock, [submitted, started, duplicate_attempt])
    assert str(captured.value) == "trial has duplicate physical execution numbers"


def test_wave_identity_faults_are_rejected(remote_spec: ExperimentSpec) -> None:
    lock, submitted = _campaign(remote_spec)
    unknown = _wave_event(lock, 2, "active").model_copy(
        update={
            "payload": WaveLifecyclePayload(
                deployment_digest=lock.runs[0].deployment_digest,
                provider="hf-inference-endpoints",
                shard_ids=["unknown-shard"],
            )
        }
    )
    with pytest.raises(ValueError) as captured:
        project_recovery(lock, [submitted, unknown])
    assert str(captured.value) == "wave references unknown shards: unknown-shard"

    active = _wave_event(lock, 2, "active")
    changed = _wave_event(lock, 3, "draining").model_copy(
        update={
            "payload": WaveLifecyclePayload(
                deployment_digest="sha256:" + "f" * 64,
                provider="different-provider",
                shard_ids=[lock.runs[0].shards[0].shard_id],
            )
        }
    )
    with pytest.raises(ValueError) as captured:
        project_recovery(lock, [submitted, active, changed])
    assert str(captured.value) == "wave lifecycle identity changed"


def test_execution_outcome_faults_are_rejected(remote_spec: ExperimentSpec) -> None:
    lock, submitted = _campaign(remote_spec, tasks=2, max_trials_per_shard=2)
    shard = lock.runs[0].shards[0]
    trial = shard.trials[0]
    start = _event(
        lock,
        2,
        "execution",
        "execution-one",
        "execution.started",
        ExecutionStartedPayload(
            trial_id=trial.trial_id,
            shard_id=shard.shard_id,
            physical_attempt=1,
        ),
    )

    fault_payloads = [
        (
            "execution.completed",
            ExecutionOutcomePayload(
                trial_id=trial.trial_id,
                physical_attempt=1,
                category="transient",
            ),
            "completed execution cannot have a failure category",
        ),
        (
            "execution.failed",
            ExecutionOutcomePayload(trial_id=trial.trial_id, physical_attempt=1),
            "failed execution requires a failure category",
        ),
        (
            "execution.failed",
            ExecutionOutcomePayload(
                trial_id=shard.trials[1].trial_id,
                physical_attempt=1,
                category="lost",
            ),
            "execution outcome identity does not match its start",
        ),
        (
            "execution.failed",
            ExecutionOutcomePayload(
                trial_id=trial.trial_id,
                physical_attempt=2,
                category="lost",
            ),
            "execution outcome identity does not match its start",
        ),
    ]
    for sequence, (kind, payload, message) in enumerate(fault_payloads, 3):
        outcome = _event(
            lock,
            sequence,
            "execution",
            "execution-one",
            cast(EventKind, kind),
            payload,
        )
        with pytest.raises(ValueError) as captured:
            project_recovery(lock, [submitted, start, outcome])
        assert str(captured.value) == message

    missing = _event(
        lock,
        8,
        "execution",
        "missing",
        "execution.failed",
        ExecutionOutcomePayload(
            trial_id=trial.trial_id, physical_attempt=1, category="lost"
        ),
    )
    with pytest.raises(ValueError) as captured:
        project_recovery(lock, [submitted, missing])
    assert str(captured.value) == "execution outcome has no start: missing"

    first_outcome = _event(
        lock,
        9,
        "execution",
        "execution-one",
        "execution.failed",
        ExecutionOutcomePayload(
            trial_id=trial.trial_id, physical_attempt=1, category="lost"
        ),
    )
    second_outcome = first_outcome.model_copy(
        update={
            "event_id": "evt-" + "f" * 32,
            "observed_at": NOW + timedelta(seconds=10),
        }
    )
    with pytest.raises(ValueError) as captured:
        project_recovery(lock, [submitted, start, first_outcome, second_outcome])
    assert str(captured.value) == "execution has multiple outcomes: execution-one"


@pytest.mark.parametrize(
    ("scope", "reason"),
    [
        ("global", "global-budget"),
        ("deployment", "deployment-budget"),
        ("provider", "provider-budget"),
        ("campaign", "campaign-budget"),
    ],
)
def test_admission_allocates_exactly_to_each_scope_limit(
    remote_spec: ExperimentSpec, scope: str, reason: str
) -> None:
    lock, submitted = _campaign(
        remote_spec, tasks=3, max_trials_per_shard=1, max_shards_per_wave=1
    )
    digest = lock.runs[0].deployment_digest
    values = {
        "global_active_waves": 3,
        "deployment_active_waves": 3,
        "provider_active_waves": 3,
        "campaign_active_waves": 3,
    }
    values[f"{scope}_active_waves"] = 2
    limits = AdmissionLimits(**values)
    context = ReconcileContext(
        limits=limits,
        deployments={
            digest: DeploymentAdmission(
                provider="provider-one", estimated_wave_cost_microusd=10
            )
        },
    )

    _projection, plan = plan_reconciliation(lock, [submitted], context=context)

    assert len(plan.actions) == 2
    assert all(action.estimated_cost_microusd == 10 for action in plan.actions)
    assert len({action.action_id for action in plan.actions}) == 2
    assert [blocked.reason for blocked in plan.blocked] == [reason]


@pytest.mark.parametrize(
    ("scope", "reason"),
    [
        ("global", "global-budget"),
        ("deployment", "deployment-budget"),
        ("provider", "provider-budget"),
        ("campaign", "campaign-budget"),
    ],
)
def test_existing_wave_counts_toward_each_admission_scope(
    remote_spec: ExperimentSpec, scope: str, reason: str
) -> None:
    lock, submitted = _campaign(
        remote_spec, tasks=2, max_trials_per_shard=1, max_shards_per_wave=1
    )
    values = {
        "global_active_waves": 2,
        "deployment_active_waves": 2,
        "provider_active_waves": 2,
        "campaign_active_waves": 2,
    }
    values[f"{scope}_active_waves"] = 1
    context = ReconcileContext(limits=AdmissionLimits(**values))

    _projection, plan = plan_reconciliation(
        lock, [submitted, _wave_event(lock, 2, "active")], context=context
    )

    assert plan.actions == []
    assert [blocked.reason for blocked in plan.blocked] == [reason]


def test_default_admission_serializes_one_managed_deployment(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(
        remote_spec, tasks=3, max_trials_per_shard=1, max_shards_per_wave=1
    )

    _projection, plan = plan_reconciliation(lock, [submitted])

    assert len(plan.actions) == 1
    assert [blocked.reason for blocked in plan.blocked] == [
        "deployment-budget",
        "deployment-budget",
    ]


@pytest.mark.parametrize("phase", ["active", "closed"])
def test_wave_estimate_remains_charged_against_campaign_cap(
    remote_spec: ExperimentSpec,
    phase: str,
) -> None:
    lock, submitted = _campaign(
        remote_spec,
        tasks=2,
        max_trials_per_shard=1,
        max_shards_per_wave=1,
        spend_cap_microusd=100,
    )
    events = [submitted, _wave_event(lock, 2, "active")]
    if phase == "closed":
        events.extend(
            [
                _wave_event(lock, 3, "draining"),
                _wave_event(lock, 4, "cleaning"),
                _wave_event(lock, 5, "closed"),
            ]
        )

    digest = lock.runs[0].deployment_digest
    context = ReconcileContext(
        limits=AdmissionLimits(deployment_active_waves=2),
        deployments={digest: DeploymentAdmission(estimated_wave_cost_microusd=100)},
    )
    projection, plan = plan_reconciliation(lock, events, context=context)

    assert projection.waves["wave-one"].estimated_cost_microusd == 100
    assert plan.actions == []
    expected = 1 if phase == "active" else 2
    assert [blocked.reason for blocked in plan.blocked] == ["spend-cap"] * expected


def test_closed_wave_releases_all_admission_scopes(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(
        remote_spec, tasks=2, max_trials_per_shard=1, max_shards_per_wave=1
    )
    events = [
        submitted,
        _wave_event(lock, 2, "active"),
        _wave_event(lock, 3, "draining"),
        _wave_event(lock, 4, "cleaning"),
        _wave_event(lock, 5, "closed"),
    ]
    context = ReconcileContext(
        limits=AdmissionLimits(
            global_active_waves=1,
            deployment_active_waves=1,
            provider_active_waves=1,
            campaign_active_waves=1,
        )
    )

    _projection, plan = plan_reconciliation(lock, events, context=context)

    # The closed wave frees one admission slot, so one of the two pending
    # shards (including the one released by the closed wave) is admitted.
    assert [action.kind for action in plan.actions] == ["submit-wave"]
    assert [blocked.reason for blocked in plan.blocked] == ["global-budget"]


def test_cancellation_adopts_unobserved_reserved_wave(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    _projection, initial = plan_reconciliation(lock, [submitted])
    action = initial.actions[0]
    reserved = _event(
        lock,
        2,
        "campaign",
        lock.campaign_id,
        "action.reserved",
        ActionReservedPayload(
            action_id=action.action_id,
            action_key=action.action_key,
            action_kind=action.kind,
            target_ids=action.target_ids,
        ),
    )

    _projection, plan = plan_reconciliation(
        lock,
        [submitted, reserved, _cancel_event(lock, 3)],
        now=NOW + timedelta(seconds=3),
    )

    assert [item.model_dump(mode="json") for item in plan.actions] == [
        {
            "action_id": plan.actions[0].action_id,
            "action_key": plan.actions[0].action_key,
            "kind": "cancel-wave",
            "campaign_id": lock.campaign_id,
            "deployment_digest": lock.runs[0].deployment_digest,
            "provider": "",
            "wave_id": f"wave-{action.action_key}",
            "shard_ids": action.target_ids,
            "trial_ids": [],
            "target_ids": [action.action_id],
            "estimated_cost_microusd": None,
        },
        {
            "action_id": plan.actions[1].action_id,
            "action_key": plan.actions[1].action_key,
            "kind": "cleanup-wave",
            "campaign_id": lock.campaign_id,
            "deployment_digest": lock.runs[0].deployment_digest,
            "provider": "",
            "wave_id": f"wave-{action.action_key}",
            "shard_ids": action.target_ids,
            "trial_ids": [],
            "target_ids": [action.action_id],
            "estimated_cost_microusd": None,
        },
    ]


def test_failed_action_gets_new_durable_retry_identity(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    _projection, initial = plan_reconciliation(lock, [submitted])
    action = initial.actions[0]
    reserved = _event(
        lock,
        2,
        "campaign",
        lock.campaign_id,
        "action.reserved",
        ActionReservedPayload(
            action_id=action.action_id,
            action_key=action.action_key,
            action_kind=action.kind,
            target_ids=action.target_ids,
        ),
    )
    failed = _event(
        lock,
        3,
        "campaign",
        lock.campaign_id,
        "action.failed",
        ActionOutcomePayload(action_id=action.action_id, message="controller lost"),
    )

    _projection, retried = plan_reconciliation(lock, [submitted, reserved, failed])

    assert retried.actions[0].kind == "submit-wave"
    assert retried.actions[0].action_id != action.action_id


def _append_wave_cleanup_corpus(
    remote_spec: ExperimentSpec, corpus: list[object]
) -> None:
    normal_lock, normal_submitted = _campaign(remote_spec)
    normal_shard = normal_lock.runs[0].shards[0]
    normal_trial = normal_shard.trials[0]
    normal_complete = _event(
        normal_lock,
        2,
        "trial",
        normal_trial.trial_id,
        "trial.complete",
        LifecyclePayload(parent_id=normal_shard.shard_id),
    )
    for offset, phase in enumerate(
        ["active", "draining", "cleanup-failed", "closed"], 30
    ):
        events = [
            normal_submitted,
            normal_complete,
            _wave_event(normal_lock, offset, phase),
        ]
        projection, plan = plan_reconciliation(
            normal_lock, events, now=NOW + timedelta(seconds=offset)
        )
        corpus.append(
            [projection.model_dump(mode="json"), plan.model_dump(mode="json")]
        )

    grace_lock, grace_submitted = _campaign(remote_spec, cancellation_grace_seconds=60)
    grace_start = _execution_events(
        grace_lock, 4, execution_id="execution-active", attempt=1, category=None
    )[0]
    grace_events = [
        grace_submitted,
        _cancel_event(grace_lock),
        _wave_event(grace_lock, 3, "active"),
        grace_start,
    ]
    for now in (NOW + timedelta(seconds=30), NOW + timedelta(seconds=90)):
        projection, plan = plan_reconciliation(grace_lock, grace_events, now=now)
        corpus.append(
            [projection.model_dump(mode="json"), plan.model_dump(mode="json")]
        )


def test_recovery_decision_corpus_is_stable(remote_spec: ExperimentSpec) -> None:
    corpus: list[object] = []
    lock, submitted = _campaign(
        remote_spec,
        tasks=3,
        max_trials_per_shard=1,
        cancellation_grace_seconds=60,
        spend_cap_microusd=1_000,
    )
    digest = lock.runs[0].deployment_digest
    priced = ReconcileContext(
        limits=AdmissionLimits(
            action_limit=8,
            global_active_waves=6,
            deployment_active_waves=4,
            provider_active_waves=5,
            campaign_active_waves=3,
        ),
        usage=AdmissionUsage(
            global_active_waves=1,
            deployment_active_waves={digest: 1},
            provider_active_waves={"provider-one": 1},
            campaign_spend_microusd={lock.campaign_id: 75},
        ),
        deployments={
            digest: DeploymentAdmission(
                provider="provider-one", estimated_wave_cost_microusd=125
            )
        },
    )
    projection, plan = plan_reconciliation(
        lock, [submitted], context=priced, now=NOW + timedelta(seconds=10)
    )
    corpus.append([projection.model_dump(mode="json"), plan.model_dump(mode="json")])

    for offset, phase in enumerate(
        ["acquiring", "provisioning", "ready", "active", "draining", "cleaning"],
        20,
    ):
        events = [submitted, _cancel_event(lock), _wave_event(lock, offset, phase)]
        for now in (NOW + timedelta(seconds=30), NOW + timedelta(seconds=90)):
            projection, plan = plan_reconciliation(
                lock, events, context=priced, now=now
            )
            corpus.append(
                [projection.model_dump(mode="json"), plan.model_dump(mode="json")]
            )

    _append_wave_cleanup_corpus(remote_spec, corpus)

    retry_lock, retry_submitted = _campaign(remote_spec)
    for offset, category in enumerate(
        ["lost", "transient", "quota", "rate-limit", "ambiguous"], 40
    ):
        events = [
            retry_submitted,
            *_execution_events(
                retry_lock,
                offset,
                execution_id=f"execution-{category}",
                attempt=1,
                category=cast(RetryCategory, category),
                spend=offset,
            ),
        ]
        projected = project_recovery(retry_lock, events)
        retry_at = next(iter(projected.trials.values())).retry_not_before
        assert retry_at is not None
        for now in (NOW + timedelta(seconds=offset + 2), retry_at):
            projection, plan = plan_reconciliation(retry_lock, events, now=now)
            corpus.append(
                [projection.model_dump(mode="json"), plan.model_dump(mode="json")]
            )

    terminal_lock, terminal_submitted = _campaign(
        remote_spec, tasks=2, max_trials_per_shard=2
    )
    shard = terminal_lock.runs[0].shards[0]
    for case, kinds in enumerate(
        [
            ("trial.complete", "trial.complete"),
            ("trial.complete", "trial.invalid"),
            ("trial.invalid", "trial.invalid"),
            ("trial.cancelled", "trial.cancelled"),
        ],
        70,
    ):
        events = [terminal_submitted]
        if kinds[0] == "trial.cancelled":
            events.append(_cancel_event(terminal_lock))
        for index, (trial, kind) in enumerate(
            zip(shard.trials, kinds, strict=True), case
        ):
            events.append(
                _event(
                    terminal_lock,
                    index,
                    "trial",
                    trial.trial_id,
                    cast(EventKind, kind),
                    LifecyclePayload(parent_id=shard.shard_id),
                )
            )
        projection, plan = plan_reconciliation(terminal_lock, events)
        corpus.append(
            [projection.model_dump(mode="json"), plan.model_dump(mode="json")]
        )

    encoded = json.dumps(
        corpus, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "00afcdbd264af91a169af38819c1df4e4529fb63b5319d11191b142e08f22449"
    )


def test_agent_failure_exhaustion_has_explicit_task_outcome(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec, max_physical_executions_per_trial=1)
    failed = _execution_events(
        lock, 2, execution_id="execution-agent", attempt=1, category="agent"
    )

    projection = project_recovery(lock, [submitted, *failed])

    trial = next(iter(projection.trials.values()))
    assert trial.status == "invalid"
    assert trial.outcome == "agent_failed"
