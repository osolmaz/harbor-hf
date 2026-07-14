from __future__ import annotations

import hashlib
import json
from fnmatch import fnmatch
from itertools import product

from pydantic import BaseModel, ConfigDict

from harbor_hf.models import ExperimentSpec, MatrixRule


class RunCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    deployment: str
    agent: str


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment: str
    spec_digest: str
    run_count: int
    logical_trial_count: int | None
    cells: list[RunCell]


def experiment_digest(spec: ExperimentSpec) -> str:
    payload = spec.model_dump(mode="json", exclude_none=True)
    matrix = payload.get("matrix")
    if isinstance(matrix, dict):
        if matrix.get("include") == []:
            matrix.pop("include")
        if matrix.get("exclude") == []:
            matrix.pop("exclude")
    execution = payload.get("execution")
    if isinstance(execution, dict):
        if execution.get("max_trials_per_shard") == 64:
            execution.pop("max_trials_per_shard")
        if execution.get("max_shards_per_wave") == 8:
            execution.pop("max_shards_per_wave")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def is_task_pattern(task: str) -> bool:
    return any(character in task for character in "*?[")


def resolved_cells(spec: ExperimentSpec) -> list[RunCell]:
    candidates = [
        RunCell(model=model.id, deployment=deployment.id, agent=agent.id)
        for model, deployment, agent in product(
            sorted(spec.matrix.models, key=lambda profile: profile.id),
            sorted(spec.matrix.deployments, key=lambda profile: profile.id),
            sorted(spec.matrix.agents, key=lambda profile: profile.id),
        )
    ]
    included = [
        cell
        for cell in candidates
        if not spec.matrix.include
        or any(_rule_matches(rule, cell) for rule in spec.matrix.include)
    ]
    cells = [
        cell
        for cell in included
        if not any(_rule_matches(rule, cell) for rule in spec.matrix.exclude)
    ]
    if not cells:
        raise ValueError("matrix rules exclude every run cell")
    return cells


def _rule_matches(rule: MatrixRule, cell: RunCell) -> bool:
    return (
        (not rule.models or cell.model in rule.models)
        and (not rule.deployments or cell.deployment in rule.deployments)
        and (not rule.agents or cell.agent in rule.agents)
    )


def build_plan(spec: ExperimentSpec) -> ExperimentPlan:
    cells = resolved_cells(spec)
    task_count = sum(
        any(fnmatch(task, selection) for selection in spec.benchmark.task_names)
        for task in spec.benchmark.task_digests
    ) or (
        None
        if any(is_task_pattern(task) for task in spec.benchmark.task_names)
        else len(spec.benchmark.task_names)
    )
    logical_trial_count = (
        None
        if task_count is None
        else len(cells) * task_count * spec.execution.attempts
    )
    return ExperimentPlan(
        experiment=spec.metadata.name,
        spec_digest=experiment_digest(spec),
        run_count=len(cells),
        logical_trial_count=logical_trial_count,
        cells=cells,
    )
