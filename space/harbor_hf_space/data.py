from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, ValidationError

from harbor_hf_space.config import SpaceConfig
from harbor_hf_space.models import (
    ArtifactRow,
    ExecutionRow,
    GlobalIndexRow,
    MetricRow,
    RunRow,
    TraceRow,
    TrialRow,
)

_TABLE_MODELS = {
    "runs": RunRow,
    "trials": TrialRow,
    "executions": ExecutionRow,
    "metrics": MetricRow,
    "artifacts": ArtifactRow,
}


class PresentationError(RuntimeError):
    """Raised when public result data is missing, malformed, or inconsistent."""


class DatasetReader(Protocol):
    def resolve_revision(self, dataset: str, revision: str) -> str: ...

    def list_files(self, dataset: str, revision: str) -> list[str]: ...

    def read_rows(
        self, dataset: str, revision: str, path: str
    ) -> Sequence[Mapping[str, object]]: ...


class HubApi(Protocol):
    def repo_info(
        self,
        repo_id: str,
        *,
        revision: str,
        repo_type: str,
        token: bool,
    ) -> object: ...

    def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: str,
        repo_type: str,
        token: bool,
    ) -> list[str]: ...


Download = Callable[..., str | PathLike[str]]
ParquetParser = Callable[[Path], list[Mapping[str, object]]]


class AnonymousHubReader:
    """Reads public Dataset files without consulting an HF credential."""

    def __init__(
        self,
        *,
        api: HubApi | None = None,
        download: Download = hf_hub_download,
        parse_parquet: ParquetParser | None = None,
    ) -> None:
        self._api = HfApi(token=False) if api is None else api
        self._download = download
        self._parse_parquet = _read_parquet if parse_parquet is None else parse_parquet

    def resolve_revision(self, dataset: str, revision: str) -> str:
        info = self._api.repo_info(
            dataset,
            revision=revision,
            repo_type="dataset",
            token=False,
        )
        sha = getattr(info, "sha", None)
        if not isinstance(sha, str) or not _is_commit(sha):
            raise PresentationError(f"Dataset {dataset} returned no immutable revision")
        return sha

    def list_files(self, dataset: str, revision: str) -> list[str]:
        return self._api.list_repo_files(
            dataset,
            revision=revision,
            repo_type="dataset",
            token=False,
        )

    def read_rows(
        self, dataset: str, revision: str, path: str
    ) -> list[Mapping[str, object]]:
        downloaded = self._download(
            repo_id=dataset,
            filename=path,
            repo_type="dataset",
            revision=revision,
            token=False,
        )
        return self._parse_parquet(Path(downloaded))


@dataclass(frozen=True)
class Snapshot:
    index_dataset: str
    index_revision: str
    index_rows: tuple[GlobalIndexRow, ...]
    runs: tuple[RunRow, ...]
    trials: tuple[TrialRow, ...]
    executions: tuple[ExecutionRow, ...]
    metrics: tuple[MetricRow, ...]
    artifacts: tuple[ArtifactRow, ...]


class DatasetLoader:
    def __init__(self, config: SpaceConfig, reader: DatasetReader) -> None:
        self._config = config
        self._reader = reader

    def load(self) -> Snapshot:
        index_revision = self._reader.resolve_revision(
            self._config.index_dataset, self._config.index_revision
        )
        index_rows = self._load_index(index_revision)
        rows: dict[str, list[BaseModel]] = {name: [] for name in _TABLE_MODELS}
        for index in index_rows:
            publication = self._load_publication(index)
            for table, values in publication.items():
                rows[table].extend(values)
        return Snapshot(
            index_dataset=self._config.index_dataset,
            index_revision=index_revision,
            index_rows=tuple(index_rows),
            runs=tuple(_only(rows["runs"], RunRow)),
            trials=tuple(_only(rows["trials"], TrialRow)),
            executions=tuple(_only(rows["executions"], ExecutionRow)),
            metrics=tuple(_only(rows["metrics"], MetricRow)),
            artifacts=tuple(_only(rows["artifacts"], ArtifactRow)),
        )

    def _load_index(self, revision: str) -> list[GlobalIndexRow]:
        paths = sorted(
            path
            for path in self._reader.list_files(self._config.index_dataset, revision)
            if path.startswith("data/index/schema=v1/") and path.endswith(".parquet")
        )
        if not paths:
            raise PresentationError("the configured index has no v1 Parquet rows")
        window_paths = [path for path in paths if "/windows/" in path]
        if window_paths:
            requested = self._config.max_publications
            paths = [
                min(
                    window_paths,
                    key=lambda path: (
                        _index_window_size(path) < requested,
                        abs(_index_window_size(path) - requested),
                    ),
                )
            ]
        else:
            paths = paths[: self._config.max_publications]
        parsed = [
            self._validate(GlobalIndexRow, value, path)
            for path in paths
            for value in self._reader.read_rows(
                self._config.index_dataset, revision, path
            )
        ]
        unique = _unique_publications(parsed)
        return sorted(unique, key=lambda item: item.completed_at, reverse=True)[
            : self._config.max_publications
        ]

    def _load_publication(
        self, index: GlobalIndexRow
    ) -> dict[str, Sequence[BaseModel]]:
        publication: dict[str, Sequence[BaseModel]] = {}
        for table, model in _TABLE_MODELS.items():
            path = (
                f"data/{table}/schema=v1/campaign={index.campaign_id}/"
                f"{index.publication_id}.parquet"
            )
            values = [
                self._validate(model, value, path)
                for value in self._reader.read_rows(
                    index.result_dataset, index.result_revision, path
                )
            ]
            publication[table] = values
        self._validate_trace(index, publication)
        return publication

    @staticmethod
    def _validate[Model: BaseModel](
        model: type[Model], value: object, path: str
    ) -> Model:
        try:
            return model.model_validate(value)
        except ValidationError as error:
            raise PresentationError(
                f"invalid normalized row in {path}: {error}"
            ) from error

    @staticmethod
    def _validate_trace(
        index: GlobalIndexRow, publication: Mapping[str, Sequence[BaseModel]]
    ) -> None:
        runs = _only(publication["runs"], RunRow)
        trials = _only(publication["trials"], TrialRow)
        executions = _only(publication["executions"], ExecutionRow)
        metrics = _only(publication["metrics"], MetricRow)
        artifacts = _only(publication["artifacts"], ArtifactRow)
        if len(runs) != 1:
            raise PresentationError("a publication must contain exactly one run row")
        run = runs[0]
        if (
            run.publication_id != index.publication_id
            or run.run_id != index.run_id
            or run.campaign_id != index.campaign_id
            or run.benchmark != index.benchmark
            or run.result_kind != index.result_kind
            or run.outcome != index.outcome
            or run.completed_at != index.completed_at
            or run.model_repo != index.model_repo
            or run.model_revision != index.model_revision
            or run.agent_name != index.agent_name
            or run.agent_revision != index.agent_revision
            or run.source_checksum != index.source_checksum
            or run.control_commit != index.control_commit
        ):
            raise PresentationError(
                f"publication {index.publication_id} conflicts with its index row"
            )
        all_rows: list[TraceRow] = [run, *trials, *executions, *metrics, *artifacts]
        if any(_trace(row) != _trace(run) for row in all_rows):
            raise PresentationError(
                f"publication {index.publication_id} has a mismatched trace"
            )
        _validate_relations(run, trials, executions, metrics, artifacts)


def _index_window_size(path: str) -> int:
    try:
        return int(Path(path).stem)
    except ValueError as error:
        raise PresentationError(f"invalid index window path: {path}") from error


def _read_parquet(path: Path) -> list[Mapping[str, object]]:
    import pyarrow.parquet as parquet

    values = parquet.read_table(path).to_pylist()
    rows: list[Mapping[str, object]] = []
    for value in values:
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str) for key in value
        ):
            raise PresentationError(
                f"Parquet file {path.name} contains a non-object row"
            )
        rows.append(value)
    return rows


def _is_commit(value: str) -> bool:
    return len(value) in {40, 64} and all(
        character in "0123456789abcdef" for character in value
    )


def _trace(row: TraceRow) -> tuple[str, ...]:
    return (
        row.publication_id,
        row.run_id,
        row.source_bucket,
        row.source_prefix,
        row.source_checksum,
        row.run_lock_path,
        row.run_lock_sha256,
        row.control_commit,
    )


def _validate_relations(
    run: RunRow,
    trials: Sequence[TrialRow],
    executions: Sequence[ExecutionRow],
    metrics: Sequence[MetricRow],
    artifacts: Sequence[ArtifactRow],
) -> None:
    trial_by_id = {row.trial_id: row for row in trials}
    execution_by_id = {row.execution_id: row for row in executions}
    _validate_child_counts(run, trials, executions, trial_by_id, execution_by_id)
    _validate_selected_executions(trials, execution_by_id)
    _validate_owners(run, metrics, artifacts, trial_by_id, execution_by_id)


def _validate_child_counts(
    run: RunRow,
    trials: Sequence[TrialRow],
    executions: Sequence[ExecutionRow],
    trial_by_id: Mapping[str, TrialRow],
    execution_by_id: Mapping[str, ExecutionRow],
) -> None:
    if len(trial_by_id) != len(trials):
        raise PresentationError("publication contains duplicate trial IDs")
    if len(execution_by_id) != len(executions):
        raise PresentationError("publication contains duplicate execution IDs")
    if run.trial_count != len(trials) or run.execution_count != len(executions):
        raise PresentationError("run child counts do not match normalized rows")
    if any(row.trial_id not in trial_by_id for row in executions):
        raise PresentationError("execution references an unknown trial")


def _validate_selected_executions(
    trials: Sequence[TrialRow], execution_by_id: Mapping[str, ExecutionRow]
) -> None:
    for trial in trials:
        selected = execution_by_id.get(trial.selected_execution_id)
        if (
            selected is None
            or selected.trial_id != trial.trial_id
            or selected.status != "succeeded"
        ):
            raise PresentationError("trial selected execution is not a valid success")


def _validate_owners(
    run: RunRow,
    metrics: Sequence[MetricRow],
    artifacts: Sequence[ArtifactRow],
    trial_by_id: Mapping[str, TrialRow],
    execution_by_id: Mapping[str, ExecutionRow],
) -> None:
    owners = {
        "run": {run.run_id},
        "trial": set(trial_by_id),
        "execution": set(execution_by_id),
    }
    if any(row.owner_id not in owners[row.owner_type] for row in metrics):
        raise PresentationError("metric references an unknown owner")
    if any(row.owner_id not in owners[row.owner_type] for row in artifacts):
        raise PresentationError("artifact references an unknown owner")
    if len({row.metric_id for row in metrics}) != len(metrics):
        raise PresentationError("publication contains duplicate metric IDs")
    if len({row.artifact_id for row in artifacts}) != len(artifacts):
        raise PresentationError("publication contains duplicate artifact IDs")


def _unique_publications(rows: Sequence[GlobalIndexRow]) -> list[GlobalIndexRow]:
    unique: dict[str, GlobalIndexRow] = {}
    for row in rows:
        previous = unique.get(row.publication_id)
        if previous is not None and previous != row:
            raise PresentationError(
                f"index has conflicting rows for publication {row.publication_id}"
            )
        unique[row.publication_id] = row
    return list(unique.values())


def _only[T](values: Sequence[object], expected: type[T]) -> list[T]:
    if not all(isinstance(value, expected) for value in values):
        raise PresentationError(f"expected only {expected.__name__} rows")
    return [value for value in values if isinstance(value, expected)]
