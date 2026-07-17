from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from harbor_hf.presentation.repository import ResultRepository, ResultSnapshot
from harbor_hf.results import (
    ArtifactRow,
    CatalogRow,
    CatalogScope,
    ExecutionRow,
    MetricRow,
    RunRow,
    TrialRow,
    trial_reward_score,
)

RunCatalog = RunRow | CatalogRow
TrialKey = tuple[str, int]
RunSortField = Literal[
    "score",
    "benchmark",
    "model_repo",
    "agent_name",
    "hardware",
    "passed_trials",
    "duration_seconds",
    "completed_at",
]
SortOrder = Literal["asc", "desc"]


class ResultNotFound(LookupError):
    """Raised when a public result entity does not exist."""


@dataclass(frozen=True)
class ResultService:
    snapshot: ResultSnapshot
    title: str = "Harbor Results"
    repository: ResultRepository | None = None
    _publications: dict[str, ResultSnapshot] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )
    _historical_catalogs: dict[str, CatalogRow] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )
    _source_catalogs: dict[str, CatalogRow] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "title": self.title,
            "index_dataset": self.snapshot.index_dataset,
            "index_revision": self.snapshot.index_revision,
            "run_count": len(self._catalog_rows()),
            "audit_run_count": len(self._catalog_rows("audit")),
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "mode": "public",
            "comparison": True,
            "artifact_metadata": True,
            "artifact_content": False,
            "trajectories": False,
            "canonical_evidence": "private",
        }

    def list_runs(
        self,
        *,
        search: str = "",
        benchmark: str = "",
        model: str = "",
        hardware: str = "",
        scope: CatalogScope = "primary",
        sort: RunSortField = "score",
        order: SortOrder = "desc",
    ) -> dict[str, Any]:
        items = [self._summary(run) for run in self._catalog_rows(scope)]
        facets = {
            "benchmarks": sorted({item["benchmark"] for item in items}),
            "models": sorted({item["model_repo"] for item in items}),
            "hardware": sorted({item["hardware"] for item in items}),
            "agents": sorted({item["agent_name"] for item in items}),
        }
        needle = search.casefold().strip()
        if needle:
            fields = (
                "run_id",
                "campaign_id",
                "benchmark",
                "model_repo",
                "agent_name",
                "hardware",
            )
            items = [
                item
                for item in items
                if needle in " ".join(str(item[key]) for key in fields).casefold()
            ]
        filters = {
            "benchmark": benchmark,
            "model_repo": model,
            "hardware": hardware,
        }
        for field_name, expected in filters.items():
            if expected:
                items = [item for item in items if item[field_name] == expected]
        items.sort(key=lambda item: str(item["publication_id"]))
        items.sort(key=lambda item: item[sort], reverse=order == "desc")
        return {"items": items, "total": len(items), "facets": facets}

    def list_campaigns(self, *, scope: CatalogScope = "primary") -> dict[str, Any]:
        grouped: dict[str, list[RunCatalog]] = defaultdict(list)
        for run in self._catalog_rows(scope):
            grouped[run.campaign_id].append(run)
        items = []
        for campaign_id, runs in grouped.items():
            summaries = [self._summary(run) for run in runs]
            items.append(
                {
                    "campaign_id": campaign_id,
                    "run_count": len(runs),
                    "benchmark_count": len({run.benchmark for run in runs}),
                    "model_count": len({run.model_repo for run in runs}),
                    "completed_at": max(run.completed_at for run in runs),
                    "average_score": _average(
                        float(item["score"]) for item in summaries
                    ),
                }
            )
        items.sort(key=lambda item: item["completed_at"], reverse=True)
        return {"items": items, "total": len(items)}

    def campaign(
        self, campaign_id: str, *, scope: CatalogScope = "primary"
    ) -> dict[str, Any]:
        runs = [
            run for run in self._catalog_rows(scope) if run.campaign_id == campaign_id
        ]
        if not runs:
            raise ResultNotFound(campaign_id)
        return {
            "campaign_id": campaign_id,
            "runs": [self._summary(run) for run in runs],
        }

    def run(self, run_id: str) -> dict[str, Any]:
        detail = self._publication_service(run_id)
        run = detail._run(run_id)
        return {
            "summary": self._summary(self._catalog(run_id)),
            "sources": [self._summary(source) for source in self._sources(run)],
            "configuration": _public_row(run),
            "trials": [
                detail._trial_summary(row)
                for row in detail.snapshot.trials
                if row.run_id == run_id
            ],
            "executions": [
                _public_row(row)
                for row in detail.snapshot.executions
                if row.run_id == run_id
            ],
            "metrics": [
                _public_row(row)
                for row in detail.snapshot.metrics
                if row.run_id == run_id
            ],
            "artifacts": [
                _public_row(row)
                for row in detail.snapshot.artifacts
                if row.run_id == run_id
            ],
            "provenance": detail._provenance(run),
        }

    def _sources(self, run: RunRow) -> list[RunCatalog]:
        if not run.source_publication_ids:
            return []
        by_publication = {
            row.publication_id: row for row in self._catalog_rows("audit")
        }
        missing = [
            publication_id
            for publication_id in run.source_publication_ids
            if publication_id not in by_publication
        ]
        for publication_id in missing:
            source = self._source_catalogs.get(publication_id)
            if source is None:
                if self.repository is None:
                    raise ResultNotFound(publication_id)
                try:
                    source = self.repository.find_catalog_publication(publication_id)
                except KeyError as error:
                    raise ResultNotFound(publication_id) from error
                self._source_catalogs[publication_id] = source
            by_publication[publication_id] = source
        return [by_publication[item] for item in run.source_publication_ids]

    def compare(self, run_id: str, other_run_id: str) -> dict[str, Any]:
        left_catalog = self._catalog(run_id)
        right_catalog = self._catalog(other_run_id)
        left = self._publication_service(run_id)
        right = self._publication_service(other_run_id)
        left_trials = left._trials_by_task(run_id)
        right_trials = right._trials_by_task(other_run_id)
        tasks = []
        for task_name, logical_attempt in sorted(set(left_trials) | set(right_trials)):
            key = (task_name, logical_attempt)
            left_score = left._trial_score(left_trials.get(key))
            right_score = right._trial_score(right_trials.get(key))
            tasks.append(
                {
                    "task_name": task_name,
                    "logical_attempt": logical_attempt,
                    "left_score": left_score,
                    "right_score": right_score,
                    "delta": None
                    if left_score is None or right_score is None
                    else right_score - left_score,
                }
            )
        return {
            "compatible": (
                left_catalog.benchmark,
                left_catalog.benchmark_revision,
            )
            == (
                right_catalog.benchmark,
                right_catalog.benchmark_revision,
            ),
            "left": self._summary(left_catalog),
            "right": self._summary(right_catalog),
            "score_delta": self._catalog_score(right_catalog)
            - self._catalog_score(left_catalog),
            "tasks": tasks,
        }

    def trial(self, run_id: str, trial_id: str) -> dict[str, Any]:
        detail = self._publication_service(run_id)
        trial = self._find(detail.snapshot.trials, "trial_id", trial_id)
        if trial.run_id != run_id:
            raise ResultNotFound(trial_id)
        return {
            "trial": _public_row(trial),
            "score": detail._trial_score(trial),
            "executions": [
                _public_row(row)
                for row in detail.snapshot.executions
                if row.trial_id == trial_id
            ],
            "metrics": [
                _public_row(row)
                for row in detail.snapshot.metrics
                if row.owner_type == "trial" and row.owner_id == trial_id
            ],
            "artifacts": [
                _public_row(row)
                for row in detail.snapshot.artifacts
                if row.owner_type == "trial" and row.owner_id == trial_id
            ],
        }

    def execution(self, run_id: str, execution_id: str) -> dict[str, Any]:
        detail = self._publication_service(run_id)
        execution = self._find(detail.snapshot.executions, "execution_id", execution_id)
        if execution.run_id != run_id:
            raise ResultNotFound(execution_id)
        return {
            "execution": _public_row(execution),
            "metrics": [
                _public_row(row)
                for row in detail.snapshot.metrics
                if row.owner_type == "execution" and row.owner_id == execution_id
            ],
            "artifacts": [
                _public_row(row)
                for row in detail.snapshot.artifacts
                if row.owner_type == "execution" and row.owner_id == execution_id
            ],
        }

    def artifact(self, run_id: str, artifact_id: str) -> dict[str, Any]:
        detail = self._publication_service(run_id)
        artifact = self._find(detail.snapshot.artifacts, "artifact_id", artifact_id)
        if artifact.run_id != run_id:
            raise ResultNotFound(artifact_id)
        return _public_row(artifact)

    def _summary(self, run: RunCatalog) -> dict[str, Any]:
        if isinstance(run, CatalogRow):
            return run.model_dump(mode="json")
        return {
            "run_id": run.run_id,
            "publication_id": run.publication_id,
            "campaign_id": run.campaign_id,
            "evaluation_id": run.evaluation_id,
            "publication_role": run.publication_role,
            "component_kind": run.component_kind,
            "source_publication_ids": run.source_publication_ids,
            "benchmark": run.benchmark,
            "benchmark_revision": run.benchmark_revision,
            "model_repo": run.model_repo,
            "model_revision": run.model_revision,
            "agent_name": run.agent_name,
            "agent_revision": run.agent_revision,
            "provider": run.provider,
            "region": run.region,
            "hardware": run.hardware,
            "accelerator_count": run.accelerator_count,
            "result_kind": run.result_kind,
            "outcome": run.outcome,
            "quality": run.quality,
            "score": self._score(run),
            "passed_trials": sum(
                (self._trial_score(trial) or 0.0) >= 1.0
                for trial in self.snapshot.trials
                if trial.run_id == run.run_id
            ),
            "planned_trial_count": run.planned_trial_count,
            "scored_trial_count": run.scored_trial_count,
            "agent_failed_count": run.agent_failed_count,
            "benchmark_failed_count": run.benchmark_failed_count,
            "infrastructure_exhausted_count": (run.infrastructure_exhausted_count),
            "unsupported_count": run.unsupported_count,
            "execution_count": run.execution_count,
            "failed_executions": sum(
                execution.status == "failed"
                for execution in self.snapshot.executions
                if execution.run_id == run.run_id
            ),
            "duration_seconds": (run.completed_at - run.created_at).total_seconds(),
            "completed_at": run.completed_at,
        }

    def _trial_summary(self, trial: TrialRow) -> dict[str, Any]:
        value = _public_row(trial)
        value["score"] = self._trial_score(trial)
        value["execution_count"] = sum(
            row.trial_id == trial.trial_id for row in self.snapshot.executions
        )
        return value

    def _score(self, run: RunRow) -> float:
        if run.planned_trial_count == 0:
            return 0.0
        scores = [
            self._trial_score(trial) or 0.0
            for trial in self.snapshot.trials
            if trial.run_id == run.run_id
        ]
        return sum(scores) / run.planned_trial_count

    def _trial_score(self, trial: TrialRow | None) -> float | None:
        if trial is None:
            return None
        return trial_reward_score(self.snapshot.metrics, trial.trial_id)

    def _run(self, run_id: str) -> RunRow:
        return self._find(self.snapshot.runs, "run_id", run_id)

    def _catalog(self, run_id: str) -> RunCatalog:
        for row in self._catalog_rows("audit"):
            if row.run_id == run_id:
                return row
        historical = self._historical_catalogs.get(run_id)
        if historical is not None:
            return historical
        if self.repository is None:
            raise ResultNotFound(run_id)
        try:
            historical = self.repository.find_catalog(run_id)
        except KeyError as error:
            raise ResultNotFound(run_id) from error
        self._historical_catalogs[run_id] = historical
        return historical

    def _catalog_rows(self, scope: CatalogScope = "primary") -> tuple[RunCatalog, ...]:
        rows = (
            self.snapshot.catalog_rows
            if scope == "primary"
            else self.snapshot.audit_catalog_rows
        )
        if rows:
            return rows
        return self.snapshot.runs

    def _catalog_score(self, run: RunCatalog) -> float:
        return run.score if isinstance(run, CatalogRow) else self._score(run)

    def _publication_service(self, run_id: str) -> ResultService:
        if any(run.run_id == run_id for run in self.snapshot.runs):
            return self
        if self.repository is None:
            raise ResultNotFound(run_id)
        publication = self._publications.get(run_id)
        if publication is None:
            catalog = self._catalog(run_id)
            if not isinstance(catalog, CatalogRow):
                raise ResultNotFound(run_id)
            publication = self.repository.load_publication(catalog)
            self._publications[run_id] = publication
        return ResultService(publication, self.title)

    def _trials_by_task(self, run_id: str) -> dict[TrialKey, TrialRow]:
        return {
            (trial.task_name, trial.logical_attempt): trial
            for trial in self.snapshot.trials
            if trial.run_id == run_id
        }

    def _provenance(self, run: RunRow) -> dict[str, Any]:
        index = next(
            row
            for row in self.snapshot.index_rows
            if row.publication_id == run.publication_id
        )
        return {
            "index_dataset": self.snapshot.index_dataset,
            "index_revision": self.snapshot.index_revision,
            "result_dataset": index.result_dataset,
            "result_revision": index.result_revision,
            "source_checksum": run.source_checksum,
            "run_lock_sha256": run.run_lock_sha256,
            "control_commit": run.control_commit,
        }

    @staticmethod
    def _find[Row](rows: Iterable[Row], field_name: str, value: str) -> Row:
        for row in rows:
            if getattr(row, field_name) == value:
                return row
        raise ResultNotFound(value)


def _public_row(
    row: RunRow | TrialRow | ExecutionRow | MetricRow | ArtifactRow,
) -> dict[str, Any]:
    return row.model_dump(
        mode="json",
        exclude={"source_bucket", "source_prefix", "run_lock_path"},
    )


def _average(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0
