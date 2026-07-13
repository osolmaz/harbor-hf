from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from fnmatch import fnmatch
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harbor_hf.endpoints import deployment_digest
from harbor_hf.models import (
    AgentProfile,
    DeploymentTarget,
    EndpointRef,
    ExperimentSpec,
    ModelProfile,
    RemoteExecutionSpec,
)
from harbor_hf.planner import RunCell, experiment_digest, resolved_cells
from harbor_hf.provider_models import ProviderTarget
from harbor_hf.runs import RunLock, build_run_lock

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
    provider: str | None = Field(default=None, exclude_if=lambda value: value is None)
    max_concurrent_requests: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    spend_cap_microusd: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    shards: list[PlannedShard]


class CampaignRecoveryPolicy(FrozenModel):
    max_active_waves: int = Field(default=64, ge=1)
    max_physical_executions_per_trial: int = Field(default=3, ge=1)
    retry_base_seconds: int = Field(default=30, ge=1)
    retry_max_seconds: int = Field(default=1800, ge=1)
    cancellation_grace_seconds: int = Field(default=0, ge=0)
    spend_cap_microusd: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def retry_backoff_is_bounded(self) -> CampaignRecoveryPolicy:
        if self.retry_base_seconds > self.retry_max_seconds:
            raise ValueError("retry base seconds must not exceed retry maximum")
        return self


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
    recovery_policy: CampaignRecoveryPolicy
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
    provider: str | None = Field(default=None, exclude_if=lambda value: value is None)
    max_concurrent_requests: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    spend_cap_microusd: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
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
    recovery_policy: CampaignRecoveryPolicy
    runs: list[CampaignRunLock]


class SubmitWaveAction(Protocol):
    @property
    def action_id(self) -> str: ...

    @property
    def action_key(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def campaign_id(self) -> str: ...

    @property
    def deployment_digest(self) -> str: ...

    @property
    def shard_ids(self) -> list[str]: ...


class WaveShardLock(FrozenModel):
    artifact_prefix: str
    run_id: str
    shard: CampaignShardLock


class WaveRunLock(FrozenModel):
    artifact_prefix: str
    configuration: RunLock
    shards: list[WaveShardLock]


class EndpointWaveTarget(FrozenModel):
    kind: Literal["inference-endpoint"] = "inference-endpoint"
    endpoint: EndpointRef


class ProviderWaveTarget(FrozenModel):
    kind: Literal["inference-provider"] = "inference-provider"
    provider: ProviderTarget


WaveDeploymentTarget = Annotated[
    EndpointWaveTarget | ProviderWaveTarget,
    Field(discriminator="kind"),
]


class WaveLock(FrozenModel):
    schema_version: Literal["harbor-hf/wave-lock/v1alpha1"] = (
        "harbor-hf/wave-lock/v1alpha1"
    )
    wave_id: str
    action_id: str
    action_key: str
    campaign_id: str
    created_at: datetime
    manifest_digest: str
    plan_digest: str
    deployment_digest: str
    target: WaveDeploymentTarget
    artifact_bucket: str
    artifact_prefix: str
    max_shards: int
    max_concurrent_shards: int
    spend_cap_microusd: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    duration_seconds: int
    remote: RemoteExecutionSpec
    shard_ids: list[str]
    runs: list[WaveRunLock]

    @property
    def endpoint(self) -> EndpointRef | None:
        if isinstance(self.target, EndpointWaveTarget):
            return self.target.endpoint
        return None

    @property
    def provider_target(self) -> ProviderTarget | None:
        if isinstance(self.target, ProviderWaveTarget):
            return self.target.provider
        return None

    @model_validator(mode="after")
    def bounds_match_contents(self) -> WaveLock:
        shard_count = sum(len(run.shards) for run in self.runs)
        if shard_count < 1 or shard_count > self.max_shards:
            raise ValueError("wave shard count exceeds its locked bound")
        observed_ids = [
            shard.shard.shard_id for run in self.runs for shard in run.shards
        ]
        if (
            len(observed_ids) != len(self.shard_ids)
            or len(self.shard_ids) != len(set(self.shard_ids))
            or set(observed_ids) != set(self.shard_ids)
        ):
            raise ValueError("wave shard IDs do not match its locked contents")
        if self.max_concurrent_shards < 1:
            raise ValueError("wave concurrency must be positive")
        if self.duration_seconds < 1:
            raise ValueError("wave duration must be positive")
        return self


def build_campaign_plan(
    spec: ExperimentSpec,
    *,
    recovery_policy: CampaignRecoveryPolicy | None = None,
) -> CampaignPlan:
    selected_recovery_policy = recovery_policy or CampaignRecoveryPolicy()
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
        "recovery_policy": selected_recovery_policy.model_dump(mode="json"),
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
        recovery_policy=selected_recovery_policy,
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
        dict[str, DeploymentTarget],
        dict[str, AgentProfile],
    ],
) -> PlannedRun:
    models, deployments, agents = profiles
    resolved_deployment_digest = deployment_digest(
        models[cell.model], deployments[cell.deployment]
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
    provider, max_concurrent_requests, spend_cap_microusd = _target_admission(
        deployments[cell.deployment]
    )
    return PlannedRun(
        cell_digest=cell_digest,
        deployment_digest=resolved_deployment_digest,
        model=cell.model,
        deployment=cell.deployment,
        agent=cell.agent,
        provider=provider,
        max_concurrent_requests=max_concurrent_requests,
        spend_cap_microusd=spend_cap_microusd,
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
                provider=planned_run.provider,
                max_concurrent_requests=planned_run.max_concurrent_requests,
                spend_cap_microusd=planned_run.spend_cap_microusd,
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
        recovery_policy=plan.recovery_policy,
        runs=runs,
    )


def build_wave_lock(
    campaign: CampaignLock,
    spec: ExperimentSpec,
    action: SubmitWaveAction,
) -> WaveLock:
    """Resolve one reserved deployment wave without provisioning an endpoint."""
    expected_campaign = build_campaign_lock(
        build_campaign_plan(spec),
        campaign.campaign_id,
        clock=lambda: campaign.created_at,
    )
    if campaign != expected_campaign:
        raise ValueError("campaign lock does not match the resolved manifest")
    _validate_submit_wave_action(campaign, action)

    selected = _selected_wave_shards(campaign, action)
    run_locks: list[WaveRunLock] = []
    target: EndpointWaveTarget | ProviderWaveTarget | None = None
    for campaign_run, shards in selected:
        configuration = build_run_lock(
            spec,
            model_id=campaign_run.model,
            deployment_id=campaign_run.deployment,
            agent_id=campaign_run.agent,
            run_id=campaign_run.run_id,
            allow_provider=True,
            clock=lambda: campaign.created_at,
        )
        observed_target = _wave_target(configuration.deployment)
        if target is not None and observed_target != target:
            raise ValueError("compatible wave shards must use one exact target")
        target = observed_target
        run_locks.append(
            WaveRunLock(
                artifact_prefix=(
                    f"{campaign.artifact_prefix}/runs/{campaign_run.run_id}"
                ),
                configuration=configuration,
                shards=[
                    WaveShardLock(
                        artifact_prefix=(
                            f"{campaign.artifact_prefix}/runs/{campaign_run.run_id}/"
                            f"shards/{shard.shard_id}"
                        ),
                        run_id=campaign_run.run_id,
                        shard=shard,
                    )
                    for shard in shards
                ],
            )
        )
    if target is None or spec.remote is None:
        raise ValueError("deployment wave requires remote execution")
    provider_target = (
        target.provider if isinstance(target, ProviderWaveTarget) else None
    )
    return WaveLock(
        wave_id=deterministic_wave_id(action.action_key),
        action_id=action.action_id,
        action_key=action.action_key,
        campaign_id=campaign.campaign_id,
        created_at=campaign.created_at,
        manifest_digest=campaign.manifest_digest,
        plan_digest=campaign.plan_digest,
        deployment_digest=action.deployment_digest,
        target=target,
        artifact_bucket=spec.artifacts.bucket,
        artifact_prefix=(
            f"{campaign.artifact_prefix}/waves/"
            f"{deterministic_wave_id(action.action_key)}"
        ),
        max_shards=campaign.max_shards_per_wave,
        max_concurrent_shards=min(
            spec.execution.concurrent_trials,
            (
                provider_target.limits.max_concurrent_requests
                if provider_target is not None
                else spec.execution.concurrent_trials
            ),
        ),
        spend_cap_microusd=(
            _usd_to_microusd(provider_target.limits.max_spend_usd)
            if provider_target is not None
            else None
        ),
        duration_seconds=spec.execution.timeout_seconds,
        remote=spec.remote,
        shard_ids=action.shard_ids,
        runs=run_locks,
    )


def deterministic_wave_id(action_key: str) -> str:
    if re.fullmatch(r"[0-9a-f]{24}", action_key) is None:
        raise ValueError("wave action key must be a 24-character hexadecimal digest")
    return f"wave-{action_key}"


def _validate_submit_wave_action(
    campaign: CampaignLock, action: SubmitWaveAction
) -> None:
    if action.kind != "submit-wave" or action.campaign_id != campaign.campaign_id:
        raise ValueError("wave action does not target the campaign")
    if not action.shard_ids:
        raise ValueError("wave action must contain at least one shard")
    if len(action.shard_ids) != len(set(action.shard_ids)):
        raise ValueError("wave action shard IDs must be unique")
    if len(action.shard_ids) > campaign.max_shards_per_wave:
        raise ValueError("wave action exceeds the campaign shard bound")
    if (
        re.fullmatch(r"[0-9a-f]{24}", action.action_key) is None
        or action.action_id != f"act-{action.action_key}"
    ):
        raise ValueError("wave action identity does not match its immutable contents")


def _selected_wave_shards(
    campaign: CampaignLock, action: SubmitWaveAction
) -> list[tuple[CampaignRunLock, list[CampaignShardLock]]]:
    requested = set(action.shard_ids)
    selected: list[tuple[CampaignRunLock, list[CampaignShardLock]]] = []
    found: set[str] = set()
    for run in campaign.runs:
        shards = [shard for shard in run.shards if shard.shard_id in requested]
        if not shards:
            continue
        if run.deployment_digest != action.deployment_digest:
            raise ValueError("wave action mixes incompatible deployment digests")
        selected.append((run, shards))
        found.update(shard.shard_id for shard in shards)
    missing = requested - found
    if missing:
        raise ValueError("wave action references an unknown campaign shard")
    return selected


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
        "wave_lock": WaveLock.model_json_schema(),
    }


def _target_admission(
    target: DeploymentTarget,
) -> tuple[str | None, int | None, int | None]:
    if isinstance(target, ProviderTarget):
        return (
            target.service,
            target.limits.max_concurrent_requests,
            _usd_to_microusd(target.limits.max_spend_usd),
        )
    return None, None, None


def _wave_target(target: DeploymentTarget) -> EndpointWaveTarget | ProviderWaveTarget:
    if isinstance(target, ProviderTarget):
        return ProviderWaveTarget(provider=target)
    if target.endpoint is None:
        raise ValueError(
            "deployment wave requires a pre-existing endpoint binding; "
            "endpoint provisioning is outside this slice"
        )
    return EndpointWaveTarget(endpoint=target.endpoint)


def _usd_to_microusd(value: Decimal | None) -> int | None:
    if value is None:
        return None
    return int(value * 1_000_000)


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
