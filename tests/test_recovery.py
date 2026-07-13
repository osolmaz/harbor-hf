from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from harbor_hf.campaigns import CampaignLock, build_campaign_lock, build_campaign_plan
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
from harbor_hf.recovery import durable_cancellation_event, project_recovery

NOW = datetime(2026, 7, 14, tzinfo=UTC)


def _campaign(
    remote_spec: ExperimentSpec,
    *,
    tasks: int = 1,
    max_trials_per_shard: int = 64,
    max_physical_executions_per_trial: int = 3,
    retry_base_seconds: int = 10,
    retry_max_seconds: int = 60,
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
                    "max_physical_executions_per_trial": (
                        max_physical_executions_per_trial
                    ),
                    "retry_base_seconds": retry_base_seconds,
                    "retry_max_seconds": retry_max_seconds,
                    "spend_cap_microusd": spend_cap_microusd,
                }
            ),
        }
    )
    lock = build_campaign_lock(
        build_campaign_plan(spec), "campaign-recovery", clock=lambda: NOW
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


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("benchmark", "invalid"),
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

    assert next(iter(projection.trials.values())).status == "failed_infrastructure"
    assert all(action.kind != "retry-shard" for action in plan.actions)
    assert plan.terminal_decision is not None
    assert plan.terminal_decision.status == "failed"


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
        (["trial.complete", "trial.invalid"], False, "partial", "_PARTIAL"),
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

    with pytest.raises(ValueError, match="invalid wave transition"):
        project_recovery(
            lock,
            [
                submitted,
                _wave_event(lock, 2, "active"),
                _wave_event(lock, 3, "ready"),
            ],
        )


def test_faulted_history_rejects_skipped_physical_attempt(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    second = _execution_events(
        lock, 2, execution_id="execution-two", attempt=2, category="lost"
    )

    with pytest.raises(ValueError, match="attempts must be contiguous"):
        project_recovery(lock, [submitted, *second])


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

    with pytest.raises(ValueError, match="shard became terminal before its children"):
        project_recovery(lock, [submitted, early])


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
