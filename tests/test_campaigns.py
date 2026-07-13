from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from harbor_hf.campaigns import (
    CampaignPlan,
    build_campaign_lock,
    build_campaign_plan,
    campaign_json_schemas,
)
from harbor_hf.endpoints import deployment_digest
from harbor_hf.models import ExperimentSpec, MatrixRule


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

    assert set(schemas) == {"campaign_plan", "campaign_lock"}
    assert schemas["campaign_plan"]["title"] == "CampaignPlan"
    assert schemas["campaign_lock"]["title"] == "CampaignLock"
