from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harbor_hf.endpoints import (
    DesiredEndpoint,
    bind_endpoint,
    build_desired_endpoint,
    endpoint_ref_for,
)
from harbor_hf.models import (
    AgentProfile,
    DeploymentTarget,
    ExperimentSpec,
    ModelProfile,
    profile_deployment_digest,
)
from harbor_hf.planner import RunCell, resolved_cells
from harbor_hf.provider_models import ProviderTarget

Sha256Digest = str
ObjectiveKind = Literal[
    "maximum_throughput",
    "maximum_goodput",
    "maximum_stable_concurrency",
    "interactive",
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProfileIdentity(FrozenModel):
    model_sha256: Sha256Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    deployment_sha256: Sha256Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    agent_sha256: Sha256Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    benchmark_sha256: Sha256Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    harbor_runtime_sha256: Sha256Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    server_context_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    reasoning_required: bool
    sample_task_count: int = Field(ge=1)
    sample_task_names: list[str] = Field(min_length=1)
    sample_tasks_sha256: Sha256Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def sample_names_match_count(self) -> ProfileIdentity:
        if len(self.sample_task_names) != self.sample_task_count:
            raise ValueError("profile sample count does not match its task names")
        if len(self.sample_task_names) != len(set(self.sample_task_names)):
            raise ValueError("profile sampled task names must be unique")
        return self


class ProfileObjective(FrozenModel):
    kind: ObjectiveKind = "maximum_goodput"
    maximum_error_rate: float = Field(default=0.0, ge=0, le=1)
    maximum_ttft_ms_p95: float | None = Field(default=None, gt=0)
    maximum_tpot_ms_p95: float | None = Field(default=None, gt=0)
    minimum_session_output_tokens_per_second: float | None = Field(default=None, ge=0)


class ProfileWorkload(FrozenModel):
    kind: Literal["benchmark"] = "benchmark"
    sample_task_count: int = Field(ge=1)
    sample_task_names: list[str] = Field(min_length=1)
    sample_tasks_sha256: Sha256Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    minimum_observations_per_point: int = Field(default=8, ge=8)
    boundary_repetitions: int = Field(default=3, ge=3)

    @model_validator(mode="after")
    def sample_names_match_count(self) -> ProfileWorkload:
        if len(self.sample_task_names) != self.sample_task_count:
            raise ValueError("profile workload count does not match its task names")
        if len(self.sample_task_names) != len(set(self.sample_task_names)):
            raise ValueError("profile workload task names must be unique")
        return self


class ProfileArtifacts(FrozenModel):
    bucket: str = Field(min_length=1)
    prefix: str = Field(min_length=1)


class ProfilePoint(FrozenModel):
    point_sha256: Sha256Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    concurrency: int = Field(ge=1)
    repetition: int = Field(ge=1)
    status: Literal["completed", "failed", "skipped"]
    planned_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    error_rate: float = Field(ge=0, le=1)
    goodput_rate: float = Field(ge=0, le=1)
    aggregate_input_tokens_per_second: float | None = Field(default=None, ge=0)
    aggregate_output_tokens_per_second: float | None = Field(default=None, ge=0)
    tasks_per_hour: float | None = Field(default=None, ge=0)
    session_output_tokens_per_second: float | None = Field(default=None, ge=0)
    ttft_ms_p95: float | None = Field(default=None, ge=0)
    ttft_ms_p50: float | None = Field(default=None, ge=0)
    ttft_ms_p99: float | None = Field(default=None, ge=0)
    tpot_ms_p95: float | None = Field(default=None, ge=0)
    tpot_ms_p50: float | None = Field(default=None, ge=0)
    tpot_ms_p99: float | None = Field(default=None, ge=0)
    latency_ms_p50: float | None = Field(default=None, ge=0)
    latency_ms_p95: float | None = Field(default=None, ge=0)
    latency_ms_p99: float | None = Field(default=None, ge=0)
    prompt_tokens_p50: float | None = Field(default=None, ge=0)
    prompt_tokens_p95: float | None = Field(default=None, ge=0)
    prompt_tokens_max: int | None = Field(default=None, ge=0)
    output_tokens_p50: float | None = Field(default=None, ge=0)
    output_tokens_p95: float | None = Field(default=None, ge=0)
    output_tokens_max: int | None = Field(default=None, ge=0)
    artifact_prefix: str = Field(min_length=1)
    failure_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def status_has_required_evidence(self) -> ProfilePoint:
        if self.status == "completed":
            if self.completed_count == 0:
                raise ValueError("completed profile points require observations")
            if self.failure_reason is not None:
                raise ValueError(
                    "completed profile points cannot have a failure reason"
                )
        elif self.failure_reason is None:
            raise ValueError("failed and skipped profile points require a reason")
        return self


class ProfileSelection(FrozenModel):
    concurrency: int = Field(ge=1)
    criterion: ObjectiveKind
    point_sha256s: list[Sha256Digest] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class ServingProfile(FrozenModel):
    schema_version: Literal["harbor-hf/serving-profile/v1"] = (
        "harbor-hf/serving-profile/v1"
    )
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    created_at: datetime
    identity: ProfileIdentity
    objective: ProfileObjective
    workload: ProfileWorkload
    candidate_concurrency: list[int] = Field(min_length=1)
    points: list[ProfilePoint] = Field(default_factory=list)
    selection: ProfileSelection | None = None
    artifacts: ProfileArtifacts

    @model_validator(mode="after")
    def ladder_and_selection_are_consistent(self) -> ServingProfile:
        if self.candidate_concurrency != sorted(set(self.candidate_concurrency)):
            raise ValueError("candidate concurrency must be sorted and unique")
        if any(
            value < 1 or value & (value - 1) for value in self.candidate_concurrency
        ):
            raise ValueError("candidate concurrency must use powers of two")
        if any(
            point.concurrency not in self.candidate_concurrency for point in self.points
        ):
            raise ValueError("profile points must be in the candidate ladder")
        if self.selection is not None and (
            self.selection.concurrency not in self.candidate_concurrency
        ):
            raise ValueError("selected concurrency must be in the candidate ladder")
        return self


class ProfilePlan(FrozenModel):
    schema_version: Literal["harbor-hf/profile-plan/v1alpha1"] = (
        "harbor-hf/profile-plan/v1alpha1"
    )
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    plan_sha256: Sha256Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    experiment: dict[str, object]
    cell: RunCell
    identity: ProfileIdentity
    model: ModelProfile
    deployment: DeploymentTarget
    agent: AgentProfile
    benchmark: dict[str, object]
    objective: ProfileObjective
    workload: ProfileWorkload
    candidate_concurrency: list[int]
    artifacts: ProfileArtifacts
    max_spend_usd: str
    estimated_profile_cost_usd: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    profile_timeout_seconds: int = Field(ge=1)
    reasoning_required: bool

    @model_validator(mode="after")
    def candidate_ladder_is_valid(self) -> ProfilePlan:
        _validate_candidate_ladder(self.candidate_concurrency)
        if self.estimated_profile_cost_usd is not None:
            if not isinstance(self.deployment, ProviderTarget):
                raise ValueError(
                    "full-profile cost estimates apply only to Inference Providers"
                )
            _positive_cost_estimate(self.estimated_profile_cost_usd)
        return self


def bind_profile_target(
    plan: ProfilePlan,
) -> tuple[ExperimentSpec, DesiredEndpoint | None]:
    spec = ExperimentSpec.model_validate(plan.experiment)
    deployment = plan.deployment
    if isinstance(deployment, ProviderTarget):
        return spec, None
    if deployment.endpoint is not None:
        return spec, None
    if spec.remote is None:
        raise ValueError("managed profile endpoints require remote execution")
    desired = build_desired_endpoint(
        namespace=spec.remote.job.namespace,
        campaign_id=f"profile-{plan.profile_id}",
        model=plan.model,
        deployment=deployment,
    )
    endpoint = endpoint_ref_for(desired, deployment, plan.model)
    return (
        bind_endpoint(
            spec,
            deployment_id=plan.cell.deployment,
            endpoint=endpoint,
        ),
        desired,
    )


def canonical_digest(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_profile_plan(
    spec: ExperimentSpec,
    *,
    profile_id: str,
    candidate_concurrency: list[int],
    max_spend_usd: str,
    profile_timeout_seconds: int,
    estimated_profile_cost_usd: str | None = None,
    sample_task_count: int = 8,
    sample_task_names: list[str] | None = None,
    objective: ProfileObjective | None = None,
) -> ProfilePlan:
    cells = resolved_cells(spec)
    spend_cap = _validate_plan_inputs(
        spec,
        cells,
        candidate_concurrency,
        max_spend_usd,
        profile_timeout_seconds,
    )
    model, deployment, agent = _resolve_profiles(spec, cells[0])
    if isinstance(deployment, ProviderTarget) and max(candidate_concurrency) > (
        deployment.limits.max_concurrent_requests
    ):
        raise ValueError(
            "profile concurrency exceeds the provider request concurrency limit"
        )
    max_spend_usd = str(spend_cap)
    profile_cost_estimate = _profile_cost_estimate(
        deployment,
        estimated_profile_cost_usd,
    )
    context = spec.execution.server_context_tokens
    output = spec.execution.max_output_tokens
    if context is None or output is None:
        raise ValueError(
            "profile planning requires execution context and output token limits"
        )
    sampled = _profile_sample_tasks(
        spec,
        deployment,
        candidate_concurrency,
        sample_task_count,
        sample_task_names,
    )
    benchmark = spec.benchmark.model_dump(mode="json", exclude_none=True)
    assert spec.remote is not None
    sample_tasks_sha256 = canonical_digest(
        {task: spec.benchmark.task_digests[task] for task in sampled}
    )
    identity = ProfileIdentity(
        model_sha256=canonical_digest(model),
        deployment_sha256=profile_deployment_digest(deployment),
        agent_sha256=canonical_digest(agent),
        benchmark_sha256=canonical_digest(benchmark),
        harbor_runtime_sha256=canonical_digest(spec.remote.harbor),
        server_context_tokens=context,
        max_output_tokens=output,
        reasoning_required=spec.execution.reasoning_required,
        sample_task_count=len(sampled),
        sample_task_names=sampled,
        sample_tasks_sha256=sample_tasks_sha256,
    )
    workload = ProfileWorkload(
        sample_task_count=len(sampled),
        sample_task_names=sampled,
        sample_tasks_sha256=sample_tasks_sha256,
    )
    artifacts = ProfileArtifacts(
        bucket=spec.artifacts.bucket,
        prefix=f"serving-profiles/{profile_id}",
    )
    resolved_objective = objective or ProfileObjective()
    if (
        resolved_objective.maximum_ttft_ms_p95 is not None
        or resolved_objective.maximum_tpot_ms_p95 is not None
    ):
        raise ValueError(
            "latency objectives require streaming measurements "
            "not collected by profiling"
        )
    core = {
        "profile_id": profile_id,
        "experiment": spec.model_dump(mode="json", exclude_none=True),
        "cell": cells[0].model_dump(mode="json"),
        "identity": identity.model_dump(mode="json"),
        "model": model.model_dump(mode="json", exclude_none=True),
        "deployment": deployment.model_dump(mode="json", exclude_none=True),
        "agent": agent.model_dump(mode="json", exclude_none=True),
        "benchmark": benchmark,
        "objective": resolved_objective.model_dump(mode="json", exclude_none=True),
        "workload": workload.model_dump(mode="json"),
        "candidate_concurrency": candidate_concurrency,
        "artifacts": artifacts.model_dump(mode="json"),
        "max_spend_usd": max_spend_usd,
        "estimated_profile_cost_usd": profile_cost_estimate,
        "profile_timeout_seconds": profile_timeout_seconds,
        "reasoning_required": spec.execution.reasoning_required,
    }
    return ProfilePlan.model_validate(
        {
            "plan_sha256": canonical_digest(core),
            **core,
        }
    )


def _profile_sample_tasks(
    spec: ExperimentSpec,
    deployment: DeploymentTarget,
    candidate_concurrency: list[int],
    sample_task_count: int,
    sample_task_names: list[str] | None,
) -> list[str]:
    tasks = sorted(spec.benchmark.task_digests)
    if not tasks:
        raise ValueError("profile planning requires resolved benchmark task digests")
    minimum = (
        max(8, 2 * max(candidate_concurrency))
        if isinstance(deployment, ProviderTarget)
        else 1
    )
    if sample_task_names is None:
        requested = max(sample_task_count, minimum)
        sampled = tasks[: min(requested, len(tasks))]
    else:
        sampled = sorted(sample_task_names)
        if len(sampled) != len(set(sampled)):
            raise ValueError("profile sampled task names must be unique")
        unknown = sorted(set(sampled) - set(tasks))
        if unknown:
            raise ValueError(
                "profile sample contains unknown benchmark tasks: " + ", ".join(unknown)
            )
    if len(sampled) < minimum:
        raise ValueError(
            "provider profiling requires at least eight tasks and twice the "
            "maximum concurrency in distinct sampled tasks"
        )
    return sampled


def _profile_cost_estimate(
    deployment: DeploymentTarget,
    value: str | None,
) -> str | None:
    if value is None:
        return None
    if not isinstance(deployment, ProviderTarget):
        raise ValueError(
            "full-profile cost estimates apply only to Inference Providers"
        )
    return str(_positive_cost_estimate(value))


def _positive_cost_estimate(value: str) -> Decimal:
    try:
        estimate = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("profile cost estimate must be finite") from error
    if not estimate.is_finite() or estimate <= 0:
        raise ValueError("profile cost estimate must be positive and finite")
    return estimate


def _validate_plan_inputs(
    spec: ExperimentSpec,
    cells: list[RunCell],
    candidate_concurrency: list[int],
    max_spend_usd: str,
    profile_timeout_seconds: int,
) -> Decimal:
    if len(cells) != 1:
        raise ValueError("profile planning requires exactly one resolved matrix cell")
    try:
        spend_cap = Decimal(max_spend_usd)
    except InvalidOperation as error:
        raise ValueError("profile planning requires a finite spend cap") from error
    if not spend_cap.is_finite() or spend_cap <= 0:
        raise ValueError("profile planning requires a positive spend cap")
    _validate_candidate_ladder(candidate_concurrency)
    if spec.remote is None:
        raise ValueError("profile planning requires remote execution settings")
    if profile_timeout_seconds + 600 > spec.remote.job.timeout_seconds:
        raise ValueError(
            "profile timeout must leave 600 seconds of remote Job lifecycle headroom"
        )
    return spend_cap


def select_profile(profile: ServingProfile) -> ServingProfile:
    if profile.selection is not None:
        raise ValueError("profile is already selected")
    for point in profile.points:
        payload = point.model_dump(
            mode="json", exclude={"point_sha256"}, exclude_none=True
        )
        if canonical_digest(payload) != point.point_sha256:
            raise ValueError("profile point digest does not match its evidence")
    all_points: dict[int, list[ProfilePoint]] = {}
    for point in profile.points:
        all_points.setdefault(point.concurrency, []).append(point)
    grouped = _eligible_groups(profile, all_points)
    if not grouped:
        raise ValueError("profile has no eligible completed points")
    if profile.objective.kind == "maximum_stable_concurrency":
        winner = max(grouped)
    else:
        scored = [
            (_score(profile.objective.kind, points), concurrency)
            for concurrency, points in grouped.items()
        ]
        winner = max(scored, key=lambda item: (item[0], -item[1]))[1]
    support = sorted(grouped[winner], key=lambda point: point.repetition)
    selection = ProfileSelection(
        concurrency=winner,
        criterion=profile.objective.kind,
        point_sha256s=[point.point_sha256 for point in support],
        rationale=(
            f"highest eligible {profile.objective.kind} score; "
            "ties prefer lower concurrency"
        ),
    )
    return profile.model_copy(update={"selection": selection})


def load_serving_profile(path: Path) -> ServingProfile:
    return ServingProfile.model_validate_json(path.read_text(encoding="utf-8"))


def new_unselected_profile(plan: ProfilePlan) -> ServingProfile:
    return ServingProfile(
        profile_id=plan.profile_id,
        created_at=datetime.now(UTC),
        identity=plan.identity,
        objective=plan.objective,
        workload=plan.workload,
        candidate_concurrency=plan.candidate_concurrency,
        artifacts=plan.artifacts,
    )


def _resolve_profiles(
    spec: ExperimentSpec, cell: RunCell
) -> tuple[ModelProfile, DeploymentTarget, AgentProfile]:
    models = {profile.id: profile for profile in spec.matrix.models}
    deployments = {profile.id: profile for profile in spec.matrix.deployments}
    agents = {profile.id: profile for profile in spec.matrix.agents}
    return models[cell.model], deployments[cell.deployment], agents[cell.agent]


def _validate_candidate_ladder(values: list[int]) -> None:
    if not values:
        raise ValueError("profile candidate concurrency must not be empty")
    if values != sorted(set(values)):
        raise ValueError("profile candidate concurrency must be sorted and unique")
    if any(value < 1 or value & (value - 1) for value in values):
        raise ValueError("profile candidate concurrency must use powers of two")


def _eligible(profile: ServingProfile, point: ProfilePoint) -> bool:
    objective = profile.objective
    return (
        point.status == "completed"
        and point.error_rate <= objective.maximum_error_rate
        and (
            objective.maximum_ttft_ms_p95 is None
            or (
                point.ttft_ms_p95 is not None
                and point.ttft_ms_p95 <= objective.maximum_ttft_ms_p95
            )
        )
        and (
            objective.maximum_tpot_ms_p95 is None
            or (
                point.tpot_ms_p95 is not None
                and point.tpot_ms_p95 <= objective.maximum_tpot_ms_p95
            )
        )
        and (
            objective.minimum_session_output_tokens_per_second is None
            or (
                point.session_output_tokens_per_second is not None
                and point.session_output_tokens_per_second
                >= objective.minimum_session_output_tokens_per_second
            )
        )
    )


def _eligible_repetition_group(
    profile: ServingProfile, points: list[ProfilePoint]
) -> list[ProfilePoint] | None:
    repetitions = [point.repetition for point in points]
    if len(repetitions) != len(set(repetitions)) or 1 not in repetitions:
        return None
    eligible = [point for point in points if _eligible(profile, point)]
    if not eligible:
        return None
    repetitions_required = (
        profile.objective.kind == "maximum_stable_concurrency" or len(points) > 1
    )
    if repetitions_required and len(eligible) != profile.workload.boundary_repetitions:
        return None
    return eligible


def _eligible_groups(
    profile: ServingProfile, all_points: dict[int, list[ProfilePoint]]
) -> dict[int, list[ProfilePoint]]:
    grouped: dict[int, list[ProfilePoint]] = {}
    for concurrency, points in all_points.items():
        eligible = _eligible_repetition_group(profile, points)
        if eligible is not None:
            grouped[concurrency] = eligible
    return grouped


def _score(kind: ObjectiveKind, points: list[ProfilePoint]) -> float:
    if kind == "maximum_throughput":
        values = [point.aggregate_output_tokens_per_second or 0 for point in points]
    elif kind in {"maximum_goodput", "interactive"}:
        values = [point.tasks_per_hour or 0 for point in points]
    else:
        values = [point.tasks_per_hour or 0 for point in points]
    return sum(values) / len(values)
