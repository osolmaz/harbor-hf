from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harbor_hf.models import (
    AgentProfile,
    DeploymentTarget,
    ExperimentSpec,
    ModelProfile,
)
from harbor_hf.planner import RunCell, resolved_cells

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
    server_context_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)


class ProfileObjective(FrozenModel):
    kind: ObjectiveKind = "maximum_goodput"
    maximum_error_rate: float = Field(default=0.0, ge=0, le=1)
    maximum_ttft_ms_p95: float | None = Field(default=None, gt=0)
    maximum_tpot_ms_p95: float | None = Field(default=None, gt=0)
    minimum_session_output_tokens_per_second: float | None = Field(default=None, ge=0)


class ProfileWorkload(FrozenModel):
    kind: Literal["benchmark"] = "benchmark"
    sample_task_count: int = Field(ge=1)
    sample_tasks_sha256: Sha256Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    minimum_observations_per_point: int = Field(default=8, ge=8)
    boundary_repetitions: int = Field(default=3, ge=3)


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
    profile_timeout_seconds: int = Field(ge=1)
    reasoning_required: bool


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
    sample_task_count: int = 8,
    objective: ProfileObjective | None = None,
) -> ProfilePlan:
    cells = resolved_cells(spec)
    if len(cells) != 1:
        raise ValueError("profile planning requires exactly one resolved matrix cell")
    if not max_spend_usd or float(max_spend_usd) <= 0:
        raise ValueError("profile planning requires a positive spend cap")
    model, deployment, agent = _resolve_profiles(spec, cells[0])
    context = spec.execution.server_context_tokens
    output = spec.execution.max_output_tokens
    if context is None or output is None:
        raise ValueError(
            "profile planning requires execution context and output token limits"
        )
    tasks = sorted(spec.benchmark.task_digests)
    if not tasks:
        raise ValueError("profile planning requires resolved benchmark task digests")
    sampled = tasks[: min(sample_task_count, len(tasks))]
    benchmark = spec.benchmark.model_dump(mode="json", exclude_none=True)
    identity = ProfileIdentity(
        model_sha256=canonical_digest(model),
        deployment_sha256=canonical_digest(deployment),
        agent_sha256=canonical_digest(agent),
        benchmark_sha256=canonical_digest(benchmark),
        server_context_tokens=context,
        max_output_tokens=output,
    )
    workload = ProfileWorkload(
        sample_task_count=len(sampled),
        sample_tasks_sha256=canonical_digest(
            {task: spec.benchmark.task_digests[task] for task in sampled}
        ),
    )
    artifacts = ProfileArtifacts(
        bucket=spec.artifacts.bucket,
        prefix=f"serving-profiles/{profile_id}",
    )
    resolved_objective = objective or ProfileObjective()
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
        "profile_timeout_seconds": profile_timeout_seconds,
        "reasoning_required": spec.execution.reasoning_required,
    }
    return ProfilePlan.model_validate(
        {
            "plan_sha256": canonical_digest(core),
            **core,
        }
    )


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
    grouped = {
        concurrency: points
        for concurrency, points in all_points.items()
        if _eligible_repetition_group(profile, points)
    }
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
) -> bool:
    repetitions = [point.repetition for point in points]
    if len(repetitions) != len(set(repetitions)) or 1 not in repetitions:
        return False
    if len(points) > 1 and set(repetitions) != set(
        range(1, profile.workload.boundary_repetitions + 1)
    ):
        return False
    return all(_eligible(profile, point) for point in points)


def _score(kind: ObjectiveKind, points: list[ProfilePoint]) -> float:
    if kind == "maximum_throughput":
        values = [point.aggregate_output_tokens_per_second or 0 for point in points]
    else:
        values = [(point.tasks_per_hour or 0) * point.goodput_rate for point in points]
    return sum(values) / len(values)
