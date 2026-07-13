from datetime import UTC, datetime, timedelta

from harbor_hf.campaigns import CampaignLock, build_campaign_lock, build_campaign_plan
from harbor_hf.control import (
    ActionReservedPayload,
    CampaignEvent,
    CampaignSubmittedPayload,
    TerminalPayload,
    new_event,
)
from harbor_hf.models import AgentProfile, ExperimentSpec
from harbor_hf.reconciler import plan_reconciliation

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

    _projection, first = plan_reconciliation(lock, [submitted])
    _projection, second = plan_reconciliation(lock, [submitted])

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
