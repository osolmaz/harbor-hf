from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from harbor_hf_space.data import Snapshot
from harbor_hf_space.models import (
    ArtifactRow,
    ExecutionRow,
    GlobalIndexRow,
    MetricRow,
    RunRow,
    TrialRow,
    isoformat,
)

Cell = str | int | float
Record = dict[str, Cell]

VIEW_COLUMNS: dict[str, tuple[str, ...]] = {
    "campaigns": (
        "Campaign",
        "Benchmark",
        "Results",
        "Runs",
        "Complete",
        "Partial",
        "Latest publication",
    ),
    "runs": (
        "Result",
        "Campaign",
        "Run",
        "Benchmark",
        "Model",
        "Model revision",
        "Agent",
        "Agent revision",
        "Deployment",
        "Trials",
        "Executions",
        "Publication",
    ),
    "tasks": (
        "Result",
        "Run",
        "Task",
        "Task digest",
        "Logical attempt",
        "Outcome",
        "Score metric",
        "Score",
        "Unit",
        "Publication",
    ),
    "attempts": (
        "Result",
        "Run",
        "Task",
        "Logical attempt",
        "Physical attempt",
        "Runtime",
        "Status",
        "Selected",
        "Remote job",
        "Started",
        "Completed",
        "Publication",
    ),
    "errors": (
        "Result",
        "Run",
        "Task",
        "Execution",
        "Physical attempt",
        "Status",
        "Reason",
        "Remote job",
        "Publication",
    ),
    "throughput": (
        "Result",
        "Run",
        "Model",
        "Hardware",
        "Accelerators",
        "Metric",
        "Value",
        "Unit",
        "Aggregation",
        "Owner",
        "Publication",
    ),
    "hardware": (
        "Result",
        "Run",
        "Runtime provider",
        "Region",
        "Hardware",
        "Accelerators",
        "Deployment",
        "Model",
        "Model revision",
        "Publication",
    ),
    "cost": (
        "Result",
        "Run",
        "Metric",
        "Value",
        "Unit",
        "Aggregation",
        "Owner",
        "Publication",
    ),
    "provenance": (
        "Result",
        "Run",
        "Publication",
        "Result Dataset",
        "Result revision",
        "Index revision",
        "Control commit",
        "Evidence Bucket",
        "Evidence prefix",
        "Evidence checksum",
        "Run lock checksum",
        "Artifact kind",
        "Artifact path",
        "Artifact checksum",
    ),
}


@dataclass(frozen=True)
class ViewFilters:
    result: str = "all"
    campaign: str = ""
    run: str = ""
    search: str = ""


@dataclass(frozen=True)
class ViewSet:
    campaigns: tuple[Record, ...]
    runs: tuple[Record, ...]
    tasks: tuple[Record, ...]
    attempts: tuple[Record, ...]
    errors: tuple[Record, ...]
    throughput: tuple[Record, ...]
    hardware: tuple[Record, ...]
    cost: tuple[Record, ...]
    provenance: tuple[Record, ...]

    def table(self, name: str) -> tuple[Record, ...]:
        tables = {
            "campaigns": self.campaigns,
            "runs": self.runs,
            "tasks": self.tasks,
            "attempts": self.attempts,
            "errors": self.errors,
            "throughput": self.throughput,
            "hardware": self.hardware,
            "cost": self.cost,
            "provenance": self.provenance,
        }
        if name not in tables:
            raise KeyError(name)
        return tables[name]


def build_views(snapshot: Snapshot, filters: ViewFilters | None = None) -> ViewSet:
    selected_filters = ViewFilters() if filters is None else filters
    indexes = {
        row.publication_id: row
        for row in snapshot.index_rows
        if _matches(row, selected_filters)
    }
    publications = set(indexes)
    runs = [row for row in snapshot.runs if row.publication_id in publications]
    trials = [row for row in snapshot.trials if row.publication_id in publications]
    executions = [
        row for row in snapshot.executions if row.publication_id in publications
    ]
    metrics = [row for row in snapshot.metrics if row.publication_id in publications]
    artifacts = [
        row for row in snapshot.artifacts if row.publication_id in publications
    ]
    run_by_publication = {row.publication_id: row for row in runs}
    trial_by_id = {row.trial_id: row for row in trials}
    scores_by_trial: defaultdict[str, list[MetricRow]] = defaultdict(list)
    for metric in metrics:
        if metric.owner_type == "trial" and _is_score(metric.name):
            scores_by_trial[metric.owner_id].append(metric)
    return ViewSet(
        campaigns=tuple(_campaign_rows(indexes.values())),
        runs=tuple(_run_rows(runs, indexes)),
        tasks=tuple(_task_rows(trials, scores_by_trial, indexes)),
        attempts=tuple(_attempt_rows(executions, trial_by_id, indexes)),
        errors=tuple(_error_rows(executions, trial_by_id, indexes)),
        throughput=tuple(
            _metric_rows(
                metrics,
                run_by_publication,
                indexes,
                predicate=_is_throughput,
                cost=False,
            )
        ),
        hardware=tuple(_hardware_rows(runs, indexes)),
        cost=tuple(
            _metric_rows(
                metrics,
                run_by_publication,
                indexes,
                predicate=None,
                cost=True,
            )
        ),
        provenance=tuple(_provenance_rows(snapshot, runs, artifacts, indexes)),
    )


def summary(snapshot: Snapshot, views: ViewSet) -> str:
    visible = len(views.runs)
    return (
        f"**{visible} runs shown** from "
        f"{len(snapshot.index_rows)} indexed publications  "
        f"\nIndex `{snapshot.index_dataset}@{snapshot.index_revision}`  "
        "\nDerived display only — canonical evidence remains in the artifact Bucket."
    )


def _matches(row: GlobalIndexRow, filters: ViewFilters) -> bool:
    result = filters.result.strip().lower()
    if result != "all" and result not in {
        row.outcome,
        row.result_kind,
        f"{row.outcome}:{row.result_kind}",
    }:
        return False
    if filters.campaign.strip() and filters.campaign.strip() != row.campaign_id:
        return False
    if filters.run.strip() and filters.run.strip() != row.run_id:
        return False
    query = filters.search.strip().casefold()
    haystack = " ".join(
        (
            row.benchmark,
            row.model_repo,
            row.agent_name,
            row.run_id,
            row.campaign_id,
        )
    ).casefold()
    return not query or query in haystack


def _campaign_rows(indexes: Iterable[GlobalIndexRow]) -> list[Record]:
    grouped: dict[str, list[GlobalIndexRow]] = defaultdict(list)
    for index in indexes:
        grouped[index.campaign_id].append(index)
    records: list[Record] = []
    for campaign_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: item.completed_at, reverse=True)
        records.append(
            {
                "Campaign": campaign_id,
                "Benchmark": ", ".join(sorted({row.benchmark for row in rows})),
                "Results": ", ".join(sorted({row.result_label for row in rows})),
                "Runs": len(rows),
                "Complete": sum(row.outcome == "complete" for row in rows),
                "Partial": sum(row.outcome == "partial" for row in rows),
                "Latest publication": isoformat(ordered[0].completed_at),
            }
        )
    return records


def _run_rows(
    runs: Sequence[RunRow], indexes: Mapping[str, GlobalIndexRow]
) -> list[Record]:
    return [
        {
            "Result": indexes[row.publication_id].result_label,
            "Campaign": row.campaign_id,
            "Run": row.run_id,
            "Benchmark": row.benchmark,
            "Model": row.model_repo,
            "Model revision": row.model_revision,
            "Agent": row.agent_name,
            "Agent revision": row.agent_revision,
            "Deployment": row.deployment_id,
            "Trials": row.trial_count,
            "Executions": row.execution_count,
            "Publication": row.publication_id,
        }
        for row in runs
    ]


def _task_rows(
    trials: Sequence[TrialRow],
    scores: Mapping[str, list[MetricRow]],
    indexes: Mapping[str, GlobalIndexRow],
) -> list[Record]:
    records: list[Record] = []
    for trial in trials:
        trial_scores = scores[trial.trial_id]
        if not trial_scores:
            records.append(_task_row(trial, indexes, None))
        else:
            records.extend(_task_row(trial, indexes, score) for score in trial_scores)
    return records


def _task_row(
    trial: TrialRow,
    indexes: Mapping[str, GlobalIndexRow],
    score: MetricRow | None,
) -> Record:
    return {
        "Result": indexes[trial.publication_id].result_label,
        "Run": trial.run_id,
        "Task": trial.task_name,
        "Task digest": trial.task_digest,
        "Logical attempt": trial.logical_attempt,
        "Outcome": trial.outcome,
        "Score metric": "not reported" if score is None else score.name,
        "Score": "not reported" if score is None else score.value,
        "Unit": "" if score is None else score.unit,
        "Publication": trial.publication_id,
    }


def _attempt_rows(
    executions: Sequence[ExecutionRow],
    trials: Mapping[str, TrialRow],
    indexes: Mapping[str, GlobalIndexRow],
) -> list[Record]:
    records: list[Record] = []
    for execution in executions:
        trial = trials[execution.trial_id]
        records.append(
            {
                "Result": indexes[execution.publication_id].result_label,
                "Run": execution.run_id,
                "Task": trial.task_name,
                "Logical attempt": trial.logical_attempt,
                "Physical attempt": execution.physical_attempt,
                "Runtime": execution.runtime_kind,
                "Status": execution.status,
                "Selected": "yes"
                if trial.selected_execution_id == execution.execution_id
                else "no",
                "Remote job": execution.remote_job_id or "not reported",
                "Started": isoformat(execution.started_at),
                "Completed": isoformat(execution.completed_at),
                "Publication": execution.publication_id,
            }
        )
    return records


def _error_rows(
    executions: Sequence[ExecutionRow],
    trials: Mapping[str, TrialRow],
    indexes: Mapping[str, GlobalIndexRow],
) -> list[Record]:
    return [
        {
            "Result": indexes[row.publication_id].result_label,
            "Run": row.run_id,
            "Task": trials[row.trial_id].task_name,
            "Execution": row.execution_id,
            "Physical attempt": row.physical_attempt,
            "Status": row.status,
            "Reason": row.retry_reason or "not reported",
            "Remote job": row.remote_job_id or "not reported",
            "Publication": row.publication_id,
        }
        for row in executions
        if row.status != "succeeded" or row.retry_reason is not None
    ]


def _metric_rows(
    metrics: Sequence[MetricRow],
    runs: Mapping[str, RunRow],
    indexes: Mapping[str, GlobalIndexRow],
    *,
    predicate: Callable[[str], bool] | None,
    cost: bool,
) -> list[Record]:
    records: list[Record] = []
    for metric in metrics:
        selected = (
            _is_cost(metric.name, metric.unit)
            if cost
            else (predicate is not None and predicate(metric.name))
        )
        if not selected:
            continue
        run = runs[metric.publication_id]
        common: Record = {
            "Result": indexes[metric.publication_id].result_label,
            "Run": metric.run_id,
            "Metric": metric.name,
            "Value": metric.value,
            "Unit": metric.unit,
            "Aggregation": metric.aggregation or "not reported",
            "Owner": f"{metric.owner_type}:{metric.owner_id}",
            "Publication": metric.publication_id,
        }
        if not cost:
            common = {
                "Result": common["Result"],
                "Run": common["Run"],
                "Model": run.model_repo,
                "Hardware": run.hardware,
                "Accelerators": run.accelerator_count,
                **{
                    key: value
                    for key, value in common.items()
                    if key not in {"Result", "Run"}
                },
            }
        records.append(common)
    return records


def _hardware_rows(
    runs: Sequence[RunRow], indexes: Mapping[str, GlobalIndexRow]
) -> list[Record]:
    return [
        {
            "Result": indexes[row.publication_id].result_label,
            "Run": row.run_id,
            "Runtime provider": row.provider,
            "Region": row.region,
            "Hardware": row.hardware,
            "Accelerators": row.accelerator_count,
            "Deployment": row.deployment_id,
            "Model": row.model_repo,
            "Model revision": row.model_revision,
            "Publication": row.publication_id,
        }
        for row in runs
    ]


def _provenance_rows(
    snapshot: Snapshot,
    runs: Sequence[RunRow],
    artifacts: Sequence[ArtifactRow],
    indexes: Mapping[str, GlobalIndexRow],
) -> list[Record]:
    artifacts_by_publication: defaultdict[str, list[ArtifactRow]] = defaultdict(list)
    for artifact in artifacts:
        artifacts_by_publication[artifact.publication_id].append(artifact)
    records: list[Record] = []
    for run in runs:
        index = indexes[run.publication_id]
        observed_artifacts = artifacts_by_publication[run.publication_id]
        published_artifacts: Sequence[ArtifactRow | None] = (
            observed_artifacts if observed_artifacts else (None,)
        )
        for artifact in published_artifacts:
            records.append(
                {
                    "Result": index.result_label,
                    "Run": run.run_id,
                    "Publication": run.publication_id,
                    "Result Dataset": index.result_dataset,
                    "Result revision": index.result_revision,
                    "Index revision": snapshot.index_revision,
                    "Control commit": run.control_commit,
                    "Evidence Bucket": run.source_bucket,
                    "Evidence prefix": run.source_prefix,
                    "Evidence checksum": run.source_checksum,
                    "Run lock checksum": run.run_lock_sha256,
                    "Artifact kind": "none published"
                    if artifact is None
                    else artifact.kind,
                    "Artifact path": "" if artifact is None else artifact.path,
                    "Artifact checksum": "" if artifact is None else artifact.sha256,
                }
            )
    return records


def _is_score(name: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    return any(term in normalized for term in ("reward", "score", "verifier"))


def _is_throughput(name: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    return any(
        term in normalized
        for term in (
            "throughput",
            "goodput",
            "tokens_per_second",
            "ttft",
            "inter_token",
            "latency",
            "task_duration",
        )
    )


def _is_cost(name: str, unit: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    normalized_unit = unit.casefold()
    return any(term in normalized for term in ("cost", "spend", "price")) or (
        "usd" in normalized_unit or normalized_unit in {"$", "dollar", "dollars"}
    )
