from datetime import UTC, datetime, timedelta

from harbor_hf.campaigns import CampaignLock, build_campaign_lock, build_campaign_plan
from harbor_hf.control import (
    ActionOutcomePayload,
    ActionReservedPayload,
    CampaignEvent,
    CampaignSubmittedPayload,
    ExecutionOutcomePayload,
    ExecutionStartedPayload,
    TerminalPayload,
    WaveLifecyclePayload,
    new_event,
)
from harbor_hf.models import AgentProfile, ExperimentSpec
from harbor_hf.reconciler import (
    AdmissionLimits,
    ReconcileAction,
    ReconcileContext,
    _estimated_retry_cost,
    plan_reconciliation,
)

NOW = datetime(2026, 7, 14, tzinfo=UTC)


def _campaign(remote_spec: ExperimentSpec) -> tuple[CampaignLock, CampaignEvent]:
    plan = build_campaign_plan(remote_spec)
    lock = build_campaign_lock(plan, "campaign-one", clock=lambda: NOW)
    submitted = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=lock.plan_digest),
        clock=lambda: NOW,
        identifier=lambda: "1" * 32,
    )
    return lock, submitted


def test_reconcile_groups_agents_by_deployment_digest(
    remote_spec: ExperimentSpec,
) -> None:
    second_agent = AgentProfile(
        id="second-agent",
        name="openclaw",
        revision="2026.7.2",
        revision_kind="package",
    )
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={"agents": [remote_spec.matrix.agents[0], second_agent]}
            ),
            "execution": remote_spec.execution.model_copy(
                update={"max_shards_per_wave": 8}
            ),
        }
    )
    lock, submitted = _campaign(spec)

    projection, plan = plan_reconciliation(lock, [submitted])

    assert projection.status == "queued"
    assert plan.action_count == 1
    assert len(plan.actions[0].shard_ids) == 2
    assert plan.actions[0].deployment_digest == lock.runs[0].deployment_digest


def test_reconcile_chunks_waves_and_is_deterministic(
    remote_spec: ExperimentSpec,
) -> None:
    tasks = {f"task-{index}": f"sha256:{index:064x}" for index in range(1, 4)}
    spec = remote_spec.model_copy(
        update={
            "benchmark": remote_spec.benchmark.model_copy(
                update={"task_names": ["task-*"], "task_digests": tasks}
            ),
            "execution": remote_spec.execution.model_copy(
                update={"max_trials_per_shard": 1, "max_shards_per_wave": 2}
            ),
        }
    )
    lock, submitted = _campaign(spec)

    context = ReconcileContext(limits=AdmissionLimits(deployment_active_waves=2))
    _projection, first = plan_reconciliation(lock, [submitted], context=context)
    _projection, second = plan_reconciliation(lock, [submitted], context=context)

    assert [len(action.shard_ids) for action in first.actions] == [2, 1]
    assert first == second
    assert first.actions[0].action_id.startswith("act-")


def test_reconcile_omits_reserved_actions(remote_spec: ExperimentSpec) -> None:
    lock, submitted = _campaign(remote_spec)
    _projection, initial = plan_reconciliation(lock, [submitted])
    action = initial.actions[0]
    reserved = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.reserved",
        producer="reconciler",
        payload=ActionReservedPayload(
            action_id=action.action_id,
            action_key=action.action_key,
            action_kind=action.kind,
            target_ids=action.shard_ids,
        ),
        clock=lambda: NOW + timedelta(seconds=1),
        identifier=lambda: "2" * 32,
    )

    _projection, observed = plan_reconciliation(lock, [submitted, reserved])

    assert observed.actions == []


def test_reconcile_terminal_campaign_has_no_actions(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _campaign(remote_spec)
    completed = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.completed",
        producer="reconciler",
        payload=TerminalPayload(message="complete"),
        clock=lambda: NOW + timedelta(seconds=1),
        identifier=lambda: "3" * 32,
    )

    projection, plan = plan_reconciliation(lock, [submitted, completed])

    assert projection.status == "completed"
    assert plan.actions == []


def _two_shard_campaign(
    remote_spec: ExperimentSpec,
) -> tuple[CampaignLock, CampaignEvent]:
    tasks = {f"task-{index}": f"sha256:{index:064x}" for index in range(1, 3)}
    spec = remote_spec.model_copy(
        update={
            "benchmark": remote_spec.benchmark.model_copy(
                update={"task_names": ["task-*"], "task_digests": tasks}
            ),
            "execution": remote_spec.execution.model_copy(
                update={"max_trials_per_shard": 1, "max_shards_per_wave": 2}
            ),
        }
    )
    return _campaign(spec)


def _submitted_wave_events(
    lock: CampaignLock, action: ReconcileAction, closed_at: datetime
) -> list[CampaignEvent]:
    reserved = new_event(
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
        clock=lambda: closed_at - timedelta(seconds=2),
        identifier=lambda: "a" * 32,
    )
    succeeded = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.succeeded",
        producer="wave-controller",
        payload=ActionOutcomePayload(action_id=action.action_id),
        clock=lambda: closed_at - timedelta(seconds=1),
        identifier=lambda: "b" * 32,
    )
    assert action.wave_id is not None
    closed = new_event(
        subject_type="wave",
        subject_id=action.wave_id,
        kind="wave.closed",
        producer="wave-controller",
        payload=WaveLifecyclePayload(
            deployment_digest=action.deployment_digest,
            provider=action.provider,
            shard_ids=action.shard_ids,
        ),
        clock=lambda: closed_at,
        identifier=lambda: "c" * 32,
    )
    return [reserved, succeeded, closed]


def test_reconcile_requeues_untouched_shards_after_wave_closes(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _two_shard_campaign(remote_spec)
    _projection, initial = plan_reconciliation(lock, [submitted], now=NOW)
    action = initial.actions[0]
    closed_at = NOW + timedelta(seconds=60)
    events = [submitted, *_submitted_wave_events(lock, action, closed_at)]

    projection, plan = plan_reconciliation(lock, events, now=closed_at)

    assert all(shard.status == "planned" for shard in projection.shards.values())
    assert [item.kind for item in plan.actions] == ["submit-wave"]
    requeued = plan.actions[0]
    assert requeued.shard_ids == action.shard_ids
    assert requeued.action_key != action.action_key


def test_reconcile_closed_wave_retains_shards_with_terminal_evidence(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _two_shard_campaign(remote_spec)
    _projection, initial = plan_reconciliation(lock, [submitted], now=NOW)
    action = initial.actions[0]
    finished_shard, untouched_shard = action.shard_ids
    trial_id = next(
        trial.trial_id
        for run in lock.runs
        for shard in run.shards
        if shard.shard_id == finished_shard
        for trial in shard.trials
    )
    started = new_event(
        subject_type="execution",
        subject_id="exec-1",
        kind="execution.started",
        producer="wave-controller",
        payload=ExecutionStartedPayload(
            trial_id=trial_id,
            shard_id=finished_shard,
            physical_attempt=1,
            wave_id=action.wave_id,
        ),
        clock=lambda: NOW + timedelta(seconds=10),
        identifier=lambda: "d" * 32,
    )
    completed = new_event(
        subject_type="execution",
        subject_id="exec-1",
        kind="execution.completed",
        producer="wave-controller",
        payload=ExecutionOutcomePayload(trial_id=trial_id, physical_attempt=1),
        clock=lambda: NOW + timedelta(seconds=20),
        identifier=lambda: "e" * 32,
    )
    events = [
        submitted,
        started,
        completed,
        *_submitted_wave_events(lock, action, NOW + timedelta(seconds=60)),
    ]

    projection, plan = plan_reconciliation(
        lock, events, now=NOW + timedelta(seconds=61)
    )

    assert projection.shards[finished_shard].status == "complete"
    assert [item.kind for item in plan.actions] == ["submit-wave"]
    assert plan.actions[0].shard_ids == [untouched_shard]


def test_reconcile_closed_wave_routes_retryable_evidence_to_retry(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _two_shard_campaign(remote_spec)
    _projection, initial = plan_reconciliation(lock, [submitted], now=NOW)
    action = initial.actions[0]
    failed_shard, untouched_shard = action.shard_ids
    trial_id = next(
        trial.trial_id
        for run in lock.runs
        for shard in run.shards
        if shard.shard_id == failed_shard
        for trial in shard.trials
    )
    started = new_event(
        subject_type="execution",
        subject_id="exec-1",
        kind="execution.started",
        producer="wave-controller",
        payload=ExecutionStartedPayload(
            trial_id=trial_id,
            shard_id=failed_shard,
            physical_attempt=1,
            wave_id=action.wave_id,
        ),
        clock=lambda: NOW + timedelta(seconds=10),
        identifier=lambda: "d" * 32,
    )
    failed = new_event(
        subject_type="execution",
        subject_id="exec-1",
        kind="execution.failed",
        producer="wave-controller",
        payload=ExecutionOutcomePayload(
            trial_id=trial_id, physical_attempt=1, category="transient"
        ),
        clock=lambda: NOW + timedelta(seconds=20),
        identifier=lambda: "e" * 32,
    )
    events = [
        submitted,
        started,
        failed,
        *_submitted_wave_events(lock, action, NOW + timedelta(seconds=60)),
    ]

    projection, plan = plan_reconciliation(lock, events, now=NOW + timedelta(hours=1))

    assert projection.shards[failed_shard].status == "retry_wait"
    assert [item.kind for item in plan.actions] == ["retry-shard"]
    by_kind = {item.kind: item for item in plan.actions}
    retry = by_kind["retry-shard"]
    assert retry.shard_ids == [failed_shard]
    assert (
        _estimated_retry_cost(lock, action.deployment_digest, 90_000_000, 1)
        == 45_000_000
    )
    assert plan.blocked[0].reason == "deployment-budget"
    assert plan.blocked[0].shard_ids == [untouched_shard]


def test_reconcile_groups_retryable_shards_by_deployment(
    remote_spec: ExperimentSpec,
) -> None:
    lock, submitted = _two_shard_campaign(remote_spec)
    _projection, initial = plan_reconciliation(lock, [submitted], now=NOW)
    action = initial.actions[0]
    trial_shards = [
        (shard.trials[0].trial_id, shard.shard_id)
        for run in lock.runs
        for shard in run.shards
    ]
    events = [submitted]
    for index, (trial_id, shard_id) in enumerate(trial_shards, start=1):
        execution_id = f"exec-{index}"
        events.extend(
            [
                new_event(
                    subject_type="execution",
                    subject_id=execution_id,
                    kind="execution.started",
                    producer="wave-controller",
                    payload=ExecutionStartedPayload(
                        trial_id=trial_id,
                        shard_id=shard_id,
                        physical_attempt=1,
                        wave_id=action.wave_id,
                    ),
                    clock=lambda index=index: NOW + timedelta(seconds=10 + index),
                    identifier=lambda index=index: f"{index + 3:032x}",
                ),
                new_event(
                    subject_type="execution",
                    subject_id=execution_id,
                    kind="execution.failed",
                    producer="wave-controller",
                    payload=ExecutionOutcomePayload(
                        trial_id=trial_id,
                        physical_attempt=1,
                        category="transient",
                    ),
                    clock=lambda index=index: NOW + timedelta(seconds=20 + index),
                    identifier=lambda index=index: f"{index + 5:032x}",
                ),
            ]
        )
    events.extend(_submitted_wave_events(lock, action, NOW + timedelta(seconds=60)))

    projection, plan = plan_reconciliation(lock, events, now=NOW + timedelta(hours=1))

    assert all(shard.status == "retry_wait" for shard in projection.shards.values())
    assert [item.kind for item in plan.actions] == ["retry-shard"]
    assert plan.actions[0].shard_ids == sorted(action.shard_ids)
    assert plan.actions[0].trial_ids == sorted(
        trial_id for trial_id, _shard_id in trial_shards
    )
    assert plan.blocked == []
