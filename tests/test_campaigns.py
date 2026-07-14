import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from harbor_hf.campaigns import (
    CampaignLock,
    CampaignPlan,
    CampaignRecoveryPolicy,
    WaveLock,
    build_campaign_lock,
    build_campaign_plan,
    build_wave_lock,
    campaign_json_schemas,
    new_campaign_id,
)
from harbor_hf.endpoints import deployment_digest
from harbor_hf.models import DeploymentProfile, ExperimentSpec, MatrixRule
from harbor_hf.reconciler import ReconcileAction, plan_reconciliation


def test_builds_content_addressed_campaign_plan(remote_spec: ExperimentSpec) -> None:
    plan = build_campaign_plan(remote_spec)

    assert plan.schema_version == "harbor-hf/campaign-plan/v1alpha1"
    assert plan.plan_digest.startswith("sha256:")
    assert plan.run_count == 1
    assert plan.shard_count == 1
    assert plan.trial_count == 1
    assert plan.runs[0].shards[0].trials[0].logical_attempt == 1
    assert plan.runs[0].deployment_digest == deployment_digest(
        remote_spec.matrix.models[0], remote_spec.matrix.deployments[0]
    )


def test_shards_ordered_task_attempts_deterministically(
    remote_spec: ExperimentSpec,
) -> None:
    tasks = {f"task-{index}": f"sha256:{index:064x}" for index in range(1, 6)}
    spec = remote_spec.model_copy(
        update={
            "benchmark": remote_spec.benchmark.model_copy(
                update={"task_names": ["task-*"], "task_digests": tasks}
            ),
            "execution": remote_spec.execution.model_copy(
                update={"attempts": 2, "max_trials_per_shard": 3}
            ),
        }
    )

    plan = build_campaign_plan(spec)

    assert plan.trial_count == 10
    assert plan.shard_count == 4
    assert [len(shard.trials) for shard in plan.runs[0].shards] == [3, 3, 3, 1]
    ordered = [
        (trial.task_name, trial.logical_attempt)
        for shard in plan.runs[0].shards
        for trial in shard.trials
    ]
    assert ordered == [
        (f"task-{index}", attempt) for index in range(1, 6) for attempt in (1, 2)
    ]


def test_plan_digest_ignores_semantically_irrelevant_input_order(
    remote_spec: ExperimentSpec,
) -> None:
    second_model = remote_spec.matrix.models[0].model_copy(update={"id": "z-model"})
    first = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={"models": [remote_spec.matrix.models[0], second_model]}
            )
        }
    )
    second = first.model_copy(
        update={
            "benchmark": first.benchmark.model_copy(
                update={
                    "task_digests": dict(reversed(first.benchmark.task_digests.items()))
                }
            ),
            "matrix": first.matrix.model_copy(
                update={"models": list(reversed(first.matrix.models))}
            ),
        }
    )

    first_plan = build_campaign_plan(first)
    second_plan = build_campaign_plan(second)

    assert first_plan.plan_digest == second_plan.plan_digest
    assert [run.cell_digest for run in first_plan.runs] == [
        run.cell_digest for run in second_plan.runs
    ]
    assert first_plan.manifest_digest != second_plan.manifest_digest


def test_matrix_rules_filter_campaign_cells(remote_spec: ExperimentSpec) -> None:
    second_model = remote_spec.matrix.models[0].model_copy(update={"id": "other-model"})
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={
                    "models": [remote_spec.matrix.models[0], second_model],
                    "include": [MatrixRule(models=["other-model"])],
                    "exclude": [],
                }
            )
        }
    )

    plan = build_campaign_plan(spec)

    assert [run.model for run in plan.runs] == ["other-model"]


def test_campaign_lock_has_stable_scoped_ids(remote_spec: ExperimentSpec) -> None:
    plan = build_campaign_plan(remote_spec)
    first = build_campaign_lock(
        plan,
        "campaign-one",
        clock=lambda: datetime(2026, 7, 14, tzinfo=UTC),
    )
    second = build_campaign_lock(
        plan,
        "campaign-one",
        clock=lambda: datetime(2026, 7, 15, tzinfo=UTC),
    )

    assert first.runs[0].run_id == second.runs[0].run_id
    assert first.runs[0].shards[0].shard_id == second.runs[0].shards[0].shard_id
    assert (
        first.runs[0].shards[0].trials[0].trial_id
        == second.runs[0].shards[0].trials[0].trial_id
    )
    assert second.created_at - first.created_at == timedelta(days=1)
    assert first.artifact_prefix == "campaigns/campaign-one"


def test_repeated_plan_submission_gets_distinct_run_ids(
    remote_spec: ExperimentSpec,
) -> None:
    plan = build_campaign_plan(remote_spec)

    first = build_campaign_lock(plan, "campaign-one")
    second = build_campaign_lock(plan, "campaign-two")

    assert first.runs[0].run_id != second.runs[0].run_id
    assert (
        first.runs[0].shards[0].trials[0].trial_id
        != second.runs[0].shards[0].trials[0].trial_id
    )


def test_campaign_id_must_be_one_safe_path_component(
    remote_spec: ExperimentSpec,
) -> None:
    with pytest.raises(ValueError, match="campaign ID must be one safe path"):
        build_campaign_lock(build_campaign_plan(remote_spec), "../unsafe")


def test_campaign_plan_rejects_inconsistent_counts(
    remote_spec: ExperimentSpec,
) -> None:
    value = build_campaign_plan(remote_spec).model_dump(mode="json")
    value["trial_count"] = 2

    with pytest.raises(ValidationError, match="counts do not match"):
        CampaignPlan.model_validate(value)


def test_campaign_plan_requires_resolved_tasks(remote_spec: ExperimentSpec) -> None:
    spec = remote_spec.model_copy(
        update={
            "benchmark": remote_spec.benchmark.model_copy(
                update={"task_names": ["*"], "task_digests": {}}
            )
        }
    )

    with pytest.raises(ValueError, match="requires resolved task digests"):
        build_campaign_plan(spec)


def test_exports_campaign_json_schemas() -> None:
    schemas = campaign_json_schemas()

    assert set(schemas) == {"campaign_plan", "campaign_lock", "wave_lock"}
    assert schemas["campaign_plan"]["title"] == "CampaignPlan"
    assert schemas["campaign_lock"]["title"] == "CampaignLock"
    assert schemas["wave_lock"]["title"] == "WaveLock"


def test_campaign_recovery_policy_is_content_addressed_and_stable(
    remote_spec: ExperimentSpec,
) -> None:
    tasks = {f"task-{index}": f"sha256:{index:064x}" for index in range(1, 6)}
    second_model = remote_spec.matrix.models[0].model_copy(update={"id": "model-two"})
    spec = remote_spec.model_copy(
        update={
            "benchmark": remote_spec.benchmark.model_copy(
                update={"task_names": ["task-*"], "task_digests": tasks}
            ),
            "matrix": remote_spec.matrix.model_copy(
                update={"models": [remote_spec.matrix.models[0], second_model]}
            ),
            "execution": remote_spec.execution.model_copy(
                update={"attempts": 2, "max_trials_per_shard": 3}
            ),
        }
    )
    policy = CampaignRecoveryPolicy(
        max_active_waves=3,
        max_physical_executions_per_trial=4,
        retry_base_seconds=17,
        retry_max_seconds=99,
        cancellation_grace_seconds=23,
        spend_cap_microusd=123_456,
    )
    default_plan = build_campaign_plan(spec)
    plan = build_campaign_plan(spec, recovery_policy=policy)
    lock = build_campaign_lock(
        plan, "campaign-policy", clock=lambda: datetime(2026, 1, 2, tzinfo=UTC)
    )

    assert plan.plan_digest != default_plan.plan_digest
    assert lock.recovery_policy == policy
    encoded = json.dumps(
        [plan.model_dump(mode="json"), lock.model_dump(mode="json")],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "42c80657f057001b86a31248875e8f13190993d0c469a6ad55149b84ce297ae5"
    )


def test_wave_lock_reuses_the_campaign_recovery_policy(
    remote_spec: ExperimentSpec,
) -> None:
    policy = CampaignRecoveryPolicy(
        max_physical_executions_per_trial=5,
        retry_base_seconds=7,
    )
    campaign = build_campaign_lock(
        build_campaign_plan(remote_spec, recovery_policy=policy),
        "campaign-policy-wave",
    )

    wave = build_wave_lock(campaign, remote_spec, _wave_action(campaign))

    assert wave.campaign_id == campaign.campaign_id
    assert campaign.recovery_policy == policy


def test_campaign_recovery_policy_rejects_unbounded_base_delay() -> None:
    with pytest.raises(ValidationError, match="retry base seconds must not exceed"):
        CampaignRecoveryPolicy(retry_base_seconds=61, retry_max_seconds=60)


def test_wave_lock_is_deterministic_and_bounded(remote_spec: ExperimentSpec) -> None:
    tasks = {
        "task-one": "sha256:" + "3" * 64,
        "task-two": "sha256:" + "4" * 64,
    }
    spec = remote_spec.model_copy(
        update={
            "benchmark": remote_spec.benchmark.model_copy(
                update={"task_names": ["task-*"], "task_digests": tasks}
            ),
            "execution": remote_spec.execution.model_copy(
                update={"max_trials_per_shard": 1, "concurrent_trials": 2}
            ),
        }
    )
    campaign = build_campaign_lock(build_campaign_plan(spec), "campaign-one")
    action = _wave_action(campaign)

    first = build_wave_lock(campaign, spec, action)
    second = build_wave_lock(campaign, spec, action)

    assert first == second
    assert first.wave_id == f"wave-{action.action_key}"
    assert first.action_id == action.action_id
    assert first.action_key == action.action_key
    assert first.campaign_id == campaign.campaign_id
    assert first.created_at == campaign.created_at
    assert first.manifest_digest == campaign.manifest_digest
    assert first.plan_digest == campaign.plan_digest
    assert first.deployment_digest == action.deployment_digest
    deployment = spec.matrix.deployments[0]
    assert isinstance(deployment, DeploymentProfile)
    assert first.endpoint == deployment.endpoint
    assert first.artifact_bucket == spec.artifacts.bucket
    assert first.artifact_prefix == (
        f"campaigns/{campaign.campaign_id}/waves/{first.wave_id}"
    )
    assert first.shard_ids == action.shard_ids
    assert first.max_shards == campaign.max_shards_per_wave
    assert first.max_concurrent_shards == 2
    assert first.duration_seconds == spec.execution.timeout_seconds
    assert sum(len(run.shards) for run in first.runs) == 2
    assert first.remote == spec.remote
    locked_run = first.runs[0]
    campaign_run = campaign.runs[0]
    assert locked_run.artifact_prefix == (
        f"campaigns/{campaign.campaign_id}/runs/{campaign_run.run_id}"
    )
    assert locked_run.configuration.run_id == campaign_run.run_id
    assert locked_run.configuration.model.id == campaign_run.model
    assert locked_run.configuration.deployment.id == campaign_run.deployment
    assert locked_run.configuration.agent.id == campaign_run.agent
    assert sorted(shard.shard.shard_id for shard in locked_run.shards) == sorted(
        action.shard_ids
    )
    assert all(shard.run_id == campaign_run.run_id for shard in locked_run.shards)
    assert all(
        shard.artifact_prefix
        == (
            f"campaigns/{campaign.campaign_id}/runs/{campaign_run.run_id}/"
            f"shards/{shard.shard.shard_id}"
        )
        for shard in locked_run.shards
    )
    assert WaveLock.model_config["frozen"] is True


def test_wave_lock_rejects_tampered_or_unbound_actions(
    remote_spec: ExperimentSpec,
) -> None:
    campaign = build_campaign_lock(build_campaign_plan(remote_spec), "campaign-one")
    action = _wave_action(campaign)
    tampered = action.model_copy(update={"action_id": "act-" + "0" * 24})

    with pytest.raises(ValueError, match="identity does not match"):
        build_wave_lock(campaign, remote_spec, tampered)

    deployment = remote_spec.matrix.deployments[0].model_copy(update={"endpoint": None})
    unbound = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={"deployments": [deployment]}
            )
        }
    )
    unbound_campaign = build_campaign_lock(
        build_campaign_plan(unbound), "campaign-unbound"
    )
    with pytest.raises(ValueError, match="pre-existing endpoint binding"):
        build_wave_lock(unbound_campaign, unbound, _wave_action(unbound_campaign))


def test_wave_lock_enforces_shard_bound(remote_spec: ExperimentSpec) -> None:
    campaign = build_campaign_lock(build_campaign_plan(remote_spec), "campaign-one")
    run = campaign.runs[0]
    shard_id = run.shards[0].shard_id
    action_key = "0" * 24
    oversized = ReconcileAction(
        action_id=f"act-{action_key}",
        action_key=action_key,
        kind="submit-wave",
        campaign_id=campaign.campaign_id,
        deployment_digest=run.deployment_digest,
        shard_ids=[
            f"{shard_id}-{index}" for index in range(campaign.max_shards_per_wave + 1)
        ],
    )

    with pytest.raises(ValueError, match="exceeds the campaign shard bound"):
        build_wave_lock(campaign, remote_spec, oversized)


def test_retry_wave_locks_only_trials_admitted_by_its_action(
    remote_spec: ExperimentSpec,
) -> None:
    campaign = build_campaign_lock(build_campaign_plan(remote_spec), "campaign-one")
    action = _wave_action(campaign)
    trial_id = campaign.runs[0].shards[0].trials[0].trial_id

    retry = action.model_copy(update={"kind": "retry-shard", "trial_ids": [trial_id]})
    lock = build_wave_lock(campaign, remote_spec, retry)
    assert lock.action_kind == "retry-shard"
    assert lock.trial_ids == [trial_id]

    with pytest.raises(ValueError, match="must admit at least one trial"):
        build_wave_lock(
            campaign,
            remote_spec,
            action.model_copy(update={"kind": "retry-shard", "trial_ids": []}),
        )
    with pytest.raises(ValueError, match="trial IDs must be unique"):
        build_wave_lock(
            campaign,
            remote_spec,
            retry.model_copy(update={"trial_ids": [trial_id, trial_id]}),
        )
    with pytest.raises(ValueError, match="outside its shards"):
        build_wave_lock(
            campaign,
            remote_spec,
            retry.model_copy(update={"trial_ids": ["trial-" + "f" * 24]}),
        )
    with pytest.raises(ValueError, match="cannot admit individual trials"):
        build_wave_lock(
            campaign,
            remote_spec,
            action.model_copy(update={"trial_ids": [trial_id]}),
        )


def test_new_campaign_id_uses_utc_plan_identity_and_bounded_nonce(
    remote_spec: ExperimentSpec,
) -> None:
    plan = build_campaign_plan(remote_spec)
    campaign_id = new_campaign_id(
        plan,
        clock=lambda: datetime(2026, 7, 14, 9, 8, 7, tzinfo=UTC),
        identifier=lambda: "0123456789abcdef",
    )

    assert campaign_id == (
        f"20260714T090807Z-{plan.plan_digest.removeprefix('sha256:')[:10]}-0123456789"
    )


def _wave_action(lock: CampaignLock) -> ReconcileAction:
    from harbor_hf.control import CampaignSubmittedPayload, new_event

    event = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=lock.plan_digest),
    )
    return plan_reconciliation(lock, [event])[1].actions[0]
