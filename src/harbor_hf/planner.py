from __future__ import annotations

import hashlib
import json
from itertools import product

from pydantic import BaseModel, ConfigDict

from harbor_hf.models import ExperimentSpec


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
    canonical = json.dumps(
        spec.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def is_task_pattern(task: str) -> bool:
    return any(character in task for character in "*?[")


def build_plan(spec: ExperimentSpec) -> ExperimentPlan:
    cells = [
        RunCell(model=model.id, deployment=deployment.id, agent=agent.id)
        for model, deployment, agent in product(
            spec.matrix.models,
            spec.matrix.deployments,
            spec.matrix.agents,
        )
    ]
    task_count = (
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
