from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from fnmatch import fnmatch
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from harbor_hf.models import (
    AgentProfile,
    DeploymentProfile,
    ExperimentSpec,
    ModelProfile,
)
from harbor_hf.planner import RunCell, experiment_digest, resolved_cells

_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class Clock(Protocol):
    def __call__(self) -> datetime: ...


class IdentifierFactory(Protocol):
    def __call__(self) -> str: ...


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlannedTrial(FrozenModel):
    trial_digest: str
    task_name: str
    task_digest: str
    logical_attempt: int


class PlannedShard(FrozenModel):
    shard_digest: str
    trials: list[PlannedTrial]


class PlannedRun(FrozenModel):
    cell_digest: str
    deployment_digest: str
    model: str
    deployment: str
    agent: str
    shards: list[PlannedShard]


class CampaignPlan(FrozenModel):
    schema_version: Literal["harbor-hf/campaign-plan/v1alpha1"] = (
        "harbor-hf/campaign-plan/v1alpha1"
    )
    experiment: str
    manifest_digest: str
    plan_digest: str
    run_count: int
    shard_count: int
    trial_count: int
    max_shards_per_wave: int
    runs: list[PlannedRun]

    @model_validator(mode="after")
    def counts_match_contents(self) -> CampaignPlan:
        shard_count = sum(len(run.shards) for run in self.runs)
        trial_count = sum(
            len(shard.trials) for run in self.runs for shard in run.shards
        )
        if (self.run_count, self.shard_count, self.trial_count) != (
            len(self.runs),
            shard_count,
            trial_count,
        ):
            raise ValueError("campaign plan counts do not match its contents")
        return self


class CampaignTrialLock(FrozenModel):
    trial_id: str
    trial_digest: str
    task_name: str
    task_digest: str
    logical_attempt: int


class CampaignShardLock(FrozenModel):
    shard_id: str
    shard_digest: str
    trials: list[CampaignTrialLock]


class CampaignRunLock(FrozenModel):
    run_id: str
    cell_digest: str
    deployment_digest: str
    model: str
    deployment: str
    agent: str
    shards: list[CampaignShardLock]


class CampaignLock(FrozenModel):
    schema_version: Literal["harbor-hf/campaign-lock/v1alpha1"] = (
        "harbor-hf/campaign-lock/v1alpha1"
    )
    campaign_id: str
    created_at: datetime
    experiment: str
    manifest_digest: str
    plan_digest: str
    artifact_prefix: str
    max_shards_per_wave: int
    runs: list[CampaignRunLock]


def build_campaign_plan(spec: ExperimentSpec) -> CampaignPlan:
    tasks = _resolved_tasks(spec)
    trials = [
        PlannedTrial(
            trial_digest=_digest(
                {
                    "task_name": task_name,
                    "task_digest": task_digest,
                    "logical_attempt": logical_attempt,
                }
            ),
            task_name=task_name,
            task_digest=task_digest,
            logical_attempt=logical_attempt,
        )
        for task_name, task_digest in tasks
        for logical_attempt in range(1, spec.execution.attempts + 1)
    ]
    profiles = (
        {profile.id: profile for profile in spec.matrix.models},
        {profile.id: profile for profile in spec.matrix.deployments},
        {profile.id: profile for profile in spec.matrix.agents},
    )
    runs = [_plan_run(spec, cell, trials, profiles) for cell in resolved_cells(spec)]
    plan_payload = {
        "schema_version": "harbor-hf/campaign-plan/v1alpha1",
        "experiment": spec.metadata.model_dump(mode="json"),
        "benchmark_dataset": spec.benchmark.dataset,
        "benchmark_dataset_digest": spec.benchmark.dataset_digest,
        "execution": spec.execution.model_dump(mode="json"),
        "artifacts": spec.artifacts.model_dump(mode="json"),
        "publishing": spec.publishing.model_dump(mode="json"),
        "remote": (
            spec.remote.model_dump(mode="json") if spec.remote is not None else None
        ),
        "runs": [run.model_dump(mode="json") for run in runs],
    }
    return CampaignPlan(
        experiment=spec.metadata.name,
        manifest_digest=experiment_digest(spec),
        plan_digest=_digest(plan_payload),
        run_count=len(runs),
        shard_count=sum(len(run.shards) for run in runs),
        trial_count=sum(len(shard.trials) for run in runs for shard in run.shards),
        max_shards_per_wave=spec.execution.max_shards_per_wave,
        runs=runs,
    )


def _resolved_tasks(spec: ExperimentSpec) -> list[tuple[str, str]]:
    digests = spec.benchmark.task_digests
    if not digests:
        raise ValueError("campaign planning requires resolved task digests")
    if any(
        not any(fnmatch(task_name, selection) for task_name in digests)
        for selection in spec.benchmark.task_names
    ) or any(
        not any(
            fnmatch(task_name, selection) for selection in spec.benchmark.task_names
        )
        for task_name in digests
    ):
        raise ValueError("campaign task digests must exactly resolve task selections")
    return sorted(digests.items())


def _plan_run(
    spec: ExperimentSpec,
    cell: RunCell,
    trials: list[PlannedTrial],
    profiles: tuple[
        dict[str, ModelProfile],
        dict[str, DeploymentProfile],
        dict[str, AgentProfile],
    ],
) -> PlannedRun:
    models, deployments, agents = profiles
    deployment_digest = _digest(
        {
            "model": _dump_profile(models[cell.model]),
            "deployment": _dump_profile(deployments[cell.deployment]),
        }
    )
    cell_digest = _digest(
        {
            "model": _dump_profile(models[cell.model]),
            "deployment": _dump_profile(deployments[cell.deployment]),
            "agent": _dump_profile(agents[cell.agent]),
        }
    )
    shard_size = spec.execution.max_trials_per_shard
    shards = [
        PlannedShard(
            shard_digest=_digest(
                {
                    "cell_digest": cell_digest,
                    "trials": [trial.model_dump(mode="json") for trial in chunk],
                }
            ),
            trials=chunk,
        )
        for offset in range(0, len(trials), shard_size)
        if (chunk := trials[offset : offset + shard_size])
    ]
    return PlannedRun(
        cell_digest=cell_digest,
        deployment_digest=deployment_digest,
        model=cell.model,
        deployment=cell.deployment,
        agent=cell.agent,
        shards=shards,
    )


def _dump_profile(profile: BaseModel) -> object:
    return profile.model_dump(mode="json", exclude_none=True)


def build_campaign_lock(
    plan: CampaignPlan,
    campaign_id: str,
    *,
    clock: Clock = lambda: datetime.now(UTC),
) -> CampaignLock:
    if _CAMPAIGN_ID.fullmatch(campaign_id) is None:
        raise ValueError(
            "campaign ID must be one safe path component containing only letters, "
            "digits, dots, underscores, or hyphens, with at most 100 characters"
        )
    runs = []
    for planned_run in plan.runs:
        run_id = _short_id(
            "run", {"campaign_id": campaign_id, "cell_digest": planned_run.cell_digest}
        )
        shards = []
        for planned_shard in planned_run.shards:
            trials = [
                CampaignTrialLock(
                    trial_id=_short_id(
                        "trial",
                        {"run_id": run_id, "trial_digest": trial.trial_digest},
                    ),
                    **trial.model_dump(mode="python"),
                )
                for trial in planned_shard.trials
            ]
            shards.append(
                CampaignShardLock(
                    shard_id=_short_id(
                        "shard",
                        {
                            "run_id": run_id,
                            "shard_digest": planned_shard.shard_digest,
                        },
                    ),
                    shard_digest=planned_shard.shard_digest,
                    trials=trials,
                )
            )
        runs.append(
            CampaignRunLock(
                run_id=run_id,
                cell_digest=planned_run.cell_digest,
                deployment_digest=planned_run.deployment_digest,
                model=planned_run.model,
                deployment=planned_run.deployment,
                agent=planned_run.agent,
                shards=shards,
            )
        )
    return CampaignLock(
        campaign_id=campaign_id,
        created_at=clock().astimezone(UTC),
        experiment=plan.experiment,
        manifest_digest=plan.manifest_digest,
        plan_digest=plan.plan_digest,
        artifact_prefix=f"campaigns/{campaign_id}",
        max_shards_per_wave=plan.max_shards_per_wave,
        runs=runs,
    )


def new_campaign_id(
    plan: CampaignPlan,
    *,
    clock: Clock = lambda: datetime.now(UTC),
    identifier: IdentifierFactory = lambda: uuid.uuid4().hex,
) -> str:
    created_at = clock().astimezone(UTC)
    plan_part = plan.plan_digest.removeprefix("sha256:")[:10]
    random_part = identifier()[:10]
    return f"{created_at:%Y%m%dT%H%M%SZ}-{plan_part}-{random_part}"


def campaign_json_schemas() -> dict[str, dict[str, object]]:
    return {
        "campaign_plan": CampaignPlan.model_json_schema(),
        "campaign_lock": CampaignLock.model_json_schema(),
    }


def _short_id(prefix: str, value: object) -> str:
    return f"{prefix}-{_digest(value).removeprefix('sha256:')[:24]}"


def _digest(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"
