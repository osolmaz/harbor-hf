from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as parquet
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from pydantic import BaseModel, ValidationError

from harbor_hf.presentation.config import PresentationConfig
from harbor_hf.results import (
    ArtifactRow,
    CatalogEntry,
    CatalogRow,
    CatalogRowV2,
    DatasetFile,
    ExecutionRow,
    GlobalIndexRow,
    MetricRow,
    ProjectionFileReference,
    PublicationProvenance,
    ResultProjectionV2,
    RunRow,
    TableName,
    TraceRow,
    TrialRow,
    build_catalog_row_v2,
    catalog_lookup_path,
    catalog_v2_lookup_path,
)

_TABLE_MODELS: dict[TableName, type[BaseModel]] = {
    "runs": RunRow,
    "trials": TrialRow,
    "executions": ExecutionRow,
    "metrics": MetricRow,
    "artifacts": ArtifactRow,
}


class PresentationError(RuntimeError):
    """Raised when public result data is missing, malformed, or inconsistent."""


class DatasetPathNotFound(PresentationError):
    """Raised when an exact public Dataset path does not exist."""


class DatasetReader(Protocol):
    def resolve_revision(self, dataset: str, revision: str) -> str: ...

    def list_files(self, dataset: str, revision: str) -> list[str]: ...

    def read_rows(
        self, dataset: str, revision: str, path: str
    ) -> Sequence[Mapping[str, object]]: ...

    def read_bytes(self, dataset: str, revision: str, path: str) -> bytes: ...


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
    """Read public datasets without consulting ambient HF credentials."""

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
        try:
            info = self._api.repo_info(
                dataset, revision=revision, repo_type="dataset", token=False
            )
        except Exception as error:
            raise PresentationError(f"failed to resolve Dataset {dataset}") from error
        sha = getattr(info, "sha", None)
        if not isinstance(sha, str) or not _is_commit(sha):
            raise PresentationError(f"Dataset {dataset} returned no immutable revision")
        return sha

    def list_files(self, dataset: str, revision: str) -> list[str]:
        try:
            return self._api.list_repo_files(
                dataset, revision=revision, repo_type="dataset", token=False
            )
        except Exception as error:
            raise PresentationError(f"failed to list Dataset {dataset}") from error

    def read_rows(
        self, dataset: str, revision: str, path: str
    ) -> list[Mapping[str, object]]:
        try:
            downloaded = self._download(
                repo_id=dataset,
                filename=path,
                repo_type="dataset",
                revision=revision,
                token=False,
            )
            return self._parse_parquet(Path(downloaded))
        except EntryNotFoundError as error:
            raise DatasetPathNotFound(
                f"{path} does not exist in Dataset {dataset}"
            ) from error
        except Exception as error:
            raise PresentationError(
                f"failed to read {path} from Dataset {dataset}"
            ) from error

    def read_bytes(self, dataset: str, revision: str, path: str) -> bytes:
        try:
            downloaded = self._download(
                repo_id=dataset,
                filename=path,
                repo_type="dataset",
                revision=revision,
                token=False,
            )
            return Path(downloaded).read_bytes()
        except EntryNotFoundError as error:
            raise DatasetPathNotFound(
                f"{path} does not exist in Dataset {dataset}"
            ) from error
        except Exception as error:
            raise PresentationError(
                f"failed to read {path} from Dataset {dataset}"
            ) from error


@dataclass(frozen=True)
class ResultSnapshot:
    index_dataset: str
    index_revision: str
    catalog_rows: tuple[CatalogEntry, ...]
    index_rows: tuple[GlobalIndexRow, ...]
    runs: tuple[RunRow, ...]
    trials: tuple[TrialRow, ...]
    executions: tuple[ExecutionRow, ...]
    metrics: tuple[MetricRow, ...]
    artifacts: tuple[ArtifactRow, ...]


class ResultRepository:
    """Load and validate one immutable view over normalized result datasets."""

    def __init__(
        self, config: PresentationConfig, reader: DatasetReader | None = None
    ) -> None:
        self.config = config
        self._reader = AnonymousHubReader() if reader is None else reader
        self._resolved_revision: str | None = None

    def load(self) -> ResultSnapshot:
        revision = self._reader.resolve_revision(
            self.config.index_dataset, self.config.index_revision
        )
        self._resolved_revision = revision
        catalog_rows = self._load_catalog(revision)
        return ResultSnapshot(
            index_dataset=self.config.index_dataset,
            index_revision=revision,
            catalog_rows=tuple(catalog_rows),
            index_rows=(),
            runs=(),
            trials=(),
            executions=(),
            metrics=(),
            artifacts=(),
        )

    def load_publication(self, catalog: CatalogEntry) -> ResultSnapshot:
        from harbor_hf.results import ResultTables, build_catalog_row

        index = _catalog_index(catalog)
        projection, projection_file = self._load_projection(catalog)
        publication = self._load_publication(index, projection)
        rows: dict[str, list[BaseModel]] = {name: [] for name in _TABLE_MODELS}
        for table, values in publication.items():
            rows[table].extend(values)
        runs = tuple(_only(rows["runs"], RunRow))
        tables = ResultTables(
            publication_id=catalog.publication_id,
            runs=list(runs),
            trials=_only(rows["trials"], TrialRow),
            executions=_only(rows["executions"], ExecutionRow),
            metrics=_only(rows["metrics"], MetricRow),
            artifacts=_only(rows["artifacts"], ArtifactRow),
            provenance=_projection_provenance(projection),
        )
        expected = (
            build_catalog_row_v2(
                tables,
                result_dataset=catalog.result_dataset,
                result_revision=catalog.result_revision,
                projection=projection_file,
            )
            if isinstance(catalog, CatalogRowV2)
            else build_catalog_row(
                tables,
                result_dataset=catalog.result_dataset,
                result_revision=catalog.result_revision,
            )
        )
        if expected != catalog:
            raise PresentationError(
                f"publication {catalog.publication_id} conflicts with its catalog row"
            )
        return ResultSnapshot(
            index_dataset=self.config.index_dataset,
            index_revision=self._resolved_revision
            or self._reader.resolve_revision(
                self.config.index_dataset, self.config.index_revision
            ),
            catalog_rows=(catalog,),
            index_rows=(index,),
            runs=runs,
            trials=tuple(_only(rows["trials"], TrialRow)),
            executions=tuple(_only(rows["executions"], ExecutionRow)),
            metrics=tuple(_only(rows["metrics"], MetricRow)),
            artifacts=tuple(_only(rows["artifacts"], ArtifactRow)),
        )

    def find_catalog(self, run_id: str) -> CatalogEntry:
        """Resolve a stable run link outside the configured list-page window."""
        revision = self._current_revision()
        for path, model in (
            (catalog_v2_lookup_path(run_id), CatalogRowV2),
            (catalog_lookup_path(run_id), CatalogRow),
        ):
            try:
                rows = [
                    _validate(model, value, path)
                    for value in self._reader.read_rows(
                        self.config.index_dataset, revision, path
                    )
                ]
                if len(rows) != 1 or rows[0].run_id != run_id:
                    raise PresentationError(
                        f"catalog lookup for {run_id} is inconsistent"
                    )
                return rows[0]
            except (DatasetPathNotFound, KeyError):
                continue
        rows = self._load_catalog(revision, largest=True)
        for row in rows:
            if row.run_id == run_id:
                return row
        raise KeyError(run_id)

    def rebuild_catalog(self) -> list[CatalogRow]:
        """Rebuild catalog rows from legacy index and publication tables."""
        from harbor_hf.results import ResultTables, build_catalog_row

        revision = self._reader.resolve_revision(
            self.config.index_dataset, self.config.index_revision
        )
        self._resolved_revision = revision
        rows = []
        for index in self._load_index(revision):
            publication = self._load_publication(index)
            tables = ResultTables(
                publication_id=index.publication_id,
                runs=_only(publication["runs"], RunRow),
                trials=_only(publication["trials"], TrialRow),
                executions=_only(publication["executions"], ExecutionRow),
                metrics=_only(publication["metrics"], MetricRow),
                artifacts=_only(publication["artifacts"], ArtifactRow),
            )
            rows.append(
                build_catalog_row(
                    tables,
                    result_dataset=index.result_dataset,
                    result_revision=index.result_revision,
                )
            )
        return rows

    def _load_projection(
        self, catalog: CatalogEntry
    ) -> tuple[ResultProjectionV2 | None, DatasetFile | None]:
        if (
            not isinstance(catalog, CatalogRowV2)
            or catalog.source_format == "legacy-v1"
        ):
            return None, None
        path = catalog.projection_path
        expected_digest = catalog.projection_sha256
        if path is None or expected_digest is None:
            raise PresentationError("native v2 catalog has no projection reference")
        content = self._reader.read_bytes(
            catalog.result_dataset, catalog.result_revision, path
        )
        if _sha256(content) != expected_digest:
            raise PresentationError("v2 projection manifest checksum differs")
        try:
            projection = ResultProjectionV2.model_validate_json(content)
        except ValidationError as error:
            raise PresentationError("v2 projection manifest is invalid") from error
        if (
            projection.publication_id != catalog.publication_id
            or projection.run_id != catalog.run_id
            or projection.source_checksum != catalog.source_checksum
            or projection.control_commit != catalog.control_commit
            or projection.envelope_sha256 != catalog.envelope_sha256
            or len(projection.harbor_archive_sha256s) != catalog.harbor_bundle_count
        ):
            raise PresentationError("v2 projection manifest conflicts with catalog")
        return projection, DatasetFile(path=path, content=content)

    def _load_catalog(
        self, revision: str, *, largest: bool = False
    ) -> list[CatalogEntry]:
        all_paths = self._reader.list_files(self.config.index_dataset, revision)
        v2_paths = sorted(
            path
            for path in all_paths
            if path.startswith("data/catalog/schema=v2/windows/")
            and path.endswith(".parquet")
        )
        paths = v2_paths or sorted(
            path
            for path in all_paths
            if path.startswith("data/catalog/schema=v1/windows/")
            and path.endswith(".parquet")
        )
        if not paths:
            raise PresentationError(
                "the configured index has no compact v1 catalog snapshot"
            )
        requested = self.config.max_publications
        path = (
            max(paths, key=_window_size)
            if largest
            else min(
                paths,
                key=lambda value: (
                    _window_size(value) < requested,
                    abs(_window_size(value) - requested),
                ),
            )
        )
        model = CatalogRowV2 if v2_paths else CatalogRow
        rows = [
            _validate(model, value, path)
            for value in self._reader.read_rows(
                self.config.index_dataset, revision, path
            )
        ]
        unique: dict[str, CatalogEntry] = {}
        for row in rows:
            previous = unique.get(row.publication_id)
            if previous is not None and previous != row:
                raise PresentationError(
                    f"catalog has conflicting rows for publication {row.publication_id}"
                )
            unique[row.publication_id] = row
        ordered = sorted(
            unique.values(), key=lambda item: item.completed_at, reverse=True
        )
        return ordered if largest else ordered[: self.config.max_publications]

    def _current_revision(self) -> str:
        if self._resolved_revision is None:
            self._resolved_revision = self._reader.resolve_revision(
                self.config.index_dataset, self.config.index_revision
            )
        return self._resolved_revision

    def _load_index(self, revision: str) -> list[GlobalIndexRow]:
        paths = sorted(
            path
            for path in self._reader.list_files(self.config.index_dataset, revision)
            if path.startswith("data/index/schema=v1/")
            and path.endswith(".parquet")
            and "/windows/" not in path
        )
        if not paths:
            raise PresentationError("the configured index has no v1 Parquet rows")
        parsed = [
            _validate(GlobalIndexRow, value, path)
            for path in paths
            for value in self._reader.read_rows(
                self.config.index_dataset, revision, path
            )
        ]
        unique: dict[str, GlobalIndexRow] = {}
        for row in parsed:
            previous = unique.get(row.publication_id)
            if previous is not None and previous != row:
                raise PresentationError(
                    f"index has conflicting rows for publication {row.publication_id}"
                )
            unique[row.publication_id] = row
        return sorted(unique.values(), key=lambda item: item.completed_at, reverse=True)

    def _load_publication(
        self,
        index: GlobalIndexRow,
        projection: ResultProjectionV2 | None = None,
    ) -> dict[TableName, list[BaseModel]]:
        with ThreadPoolExecutor(max_workers=len(_TABLE_MODELS)) as executor:
            futures = {
                table: executor.submit(
                    self._load_table,
                    index,
                    table,
                    model,
                    projection.tables[table] if projection is not None else None,
                )
                for table, model in _TABLE_MODELS.items()
            }
            publication = {table: future.result() for table, future in futures.items()}
        _validate_publication(index, publication)
        return publication

    def _load_table(
        self,
        index: GlobalIndexRow,
        table: str,
        model: type[BaseModel],
        reference: ProjectionFileReference | None,
    ) -> list[BaseModel]:
        expected_path = (
            f"data/{table}/schema=v1/campaign={index.campaign_id}/"
            f"{index.publication_id}.parquet"
        )
        path = reference.path if reference is not None else expected_path
        if path != expected_path:
            raise PresentationError("v2 projection references a noncanonical table")
        if reference is None:
            values = self._reader.read_rows(
                index.result_dataset, index.result_revision, path
            )
        else:
            content = self._reader.read_bytes(
                index.result_dataset, index.result_revision, path
            )
            if _sha256(content) != reference.sha256:
                raise PresentationError("v2 projected table checksum differs")
            values = _read_parquet_bytes(content, path)
            if len(values) != reference.row_count:
                raise PresentationError("v2 projected table row count differs")
        return [_validate(model, value, path) for value in values]


def _validate_publication(
    index: GlobalIndexRow,
    publication: Mapping[TableName, Sequence[BaseModel]],
) -> None:
    runs = _only(publication["runs"], RunRow)
    trials = _only(publication["trials"], TrialRow)
    executions = _only(publication["executions"], ExecutionRow)
    metrics = _only(publication["metrics"], MetricRow)
    artifacts = _only(publication["artifacts"], ArtifactRow)
    if len(runs) != 1:
        raise PresentationError("a publication must contain exactly one run row")
    run = runs[0]
    _validate_index_identity(index, run)
    _validate_publication_trace(index, run, trials, executions, metrics, artifacts)
    _validate_publication_relations(run, trials, executions, metrics, artifacts)


def _validate_index_identity(index: GlobalIndexRow, run: RunRow) -> None:
    expected = (
        index.publication_id,
        index.run_id,
        index.campaign_id,
        index.benchmark,
        index.completed_at,
        index.model_repo,
        index.model_revision,
        index.agent_name,
        index.agent_revision,
        index.source_checksum,
        index.control_commit,
    )
    actual = (
        run.publication_id,
        run.run_id,
        run.campaign_id,
        run.benchmark,
        run.completed_at,
        run.model_repo,
        run.model_revision,
        run.agent_name,
        run.agent_revision,
        run.source_checksum,
        run.control_commit,
    )
    if actual != expected:
        raise PresentationError(
            f"publication {index.publication_id} conflicts with its index row"
        )


def _validate_publication_trace(
    index: GlobalIndexRow,
    run: RunRow,
    trials: Sequence[TrialRow],
    executions: Sequence[ExecutionRow],
    metrics: Sequence[MetricRow],
    artifacts: Sequence[ArtifactRow],
) -> None:
    all_rows: list[TraceRow] = [run, *trials, *executions, *metrics, *artifacts]
    if any(_trace(row) != _trace(run) for row in all_rows):
        raise PresentationError(
            f"publication {index.publication_id} has a mismatched trace"
        )


def _validate_publication_relations(
    run: RunRow,
    trials: Sequence[TrialRow],
    executions: Sequence[ExecutionRow],
    metrics: Sequence[MetricRow],
    artifacts: Sequence[ArtifactRow],
) -> None:
    trial_ids = {row.trial_id for row in trials}
    execution_ids = {row.execution_id for row in executions}
    if len(trial_ids) != len(trials) or len(execution_ids) != len(executions):
        raise PresentationError("publication contains duplicate child IDs")
    if run.trial_count != len(trials) or run.execution_count != len(executions):
        raise PresentationError("run child counts do not match normalized rows")
    if any(row.trial_id not in trial_ids for row in executions):
        raise PresentationError("execution references an unknown trial")
    _validate_selected_executions(trials, executions)
    owners = {
        "run": {run.run_id},
        "trial": trial_ids,
        "execution": execution_ids,
    }
    if any(
        row.owner_id not in owners[row.owner_type] for row in [*metrics, *artifacts]
    ):
        raise PresentationError("publication row references an unknown owner")
    if len({row.metric_id for row in metrics}) != len(metrics):
        raise PresentationError("publication contains duplicate metric IDs")
    if len({row.artifact_id for row in artifacts}) != len(artifacts):
        raise PresentationError("publication contains duplicate artifact IDs")


def _validate_selected_executions(
    trials: Sequence[TrialRow], executions: Sequence[ExecutionRow]
) -> None:
    execution_by_id = {row.execution_id: row for row in executions}
    for trial in trials:
        selected = execution_by_id.get(trial.selected_execution_id)
        if (
            selected is None
            or selected.trial_id != trial.trial_id
            or selected.status != "succeeded"
        ):
            raise PresentationError("trial selected execution is invalid")


def _catalog_index(row: CatalogEntry) -> GlobalIndexRow:
    return GlobalIndexRow(
        publication_id=row.publication_id,
        run_id=row.run_id,
        campaign_id=row.campaign_id,
        benchmark=row.benchmark,
        result_kind=row.result_kind,
        outcome=row.outcome,
        completed_at=row.completed_at,
        model_repo=row.model_repo,
        model_revision=row.model_revision,
        agent_name=row.agent_name,
        agent_revision=row.agent_revision,
        result_dataset=row.result_dataset,
        result_revision=row.result_revision,
        source_checksum=row.source_checksum,
        control_commit=row.control_commit,
    )


def _read_parquet(path: Path) -> list[Mapping[str, object]]:
    values = parquet.read_table(path).to_pylist()
    if not all(isinstance(value, Mapping) for value in values):
        raise PresentationError(f"Parquet file {path.name} contains a non-object row")
    return [value for value in values if isinstance(value, Mapping)]


def _read_parquet_bytes(content: bytes, path: str) -> list[Mapping[str, object]]:
    try:
        values = parquet.read_table(pa.BufferReader(content)).to_pylist()
    except (pa.ArrowException, OSError) as error:
        raise PresentationError(f"invalid projected Parquet file: {path}") from error
    if not all(isinstance(value, Mapping) for value in values):
        raise PresentationError(f"Parquet file {path} contains a non-object row")
    return [value for value in values if isinstance(value, Mapping)]


def _projection_provenance(
    projection: ResultProjectionV2 | None,
) -> PublicationProvenance | None:
    if projection is None:
        return None
    return PublicationProvenance(
        envelope_path=projection.envelope_path,
        envelope_sha256=projection.envelope_sha256,
        projection_version=projection.projection_version,
        sanitizer_version=projection.sanitizer_version,
        harbor_bundle_manifest_sha256s=(projection.harbor_bundle_manifest_sha256s),
        harbor_archive_sha256s=projection.harbor_archive_sha256s,
    )


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _validate[Model: BaseModel](model: type[Model], value: object, path: str) -> Model:
    try:
        return model.model_validate(value)
    except ValidationError as error:
        raise PresentationError(f"invalid normalized row in {path}: {error}") from error


def _window_size(path: str) -> int:
    try:
        return int(Path(path).stem)
    except ValueError as error:
        raise PresentationError(f"invalid index window path: {path}") from error


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


def _only[T](values: Sequence[object], expected: type[T]) -> list[T]:
    if not all(isinstance(value, expected) for value in values):
        raise PresentationError(f"expected only {expected.__name__} rows")
    return [value for value in values if isinstance(value, expected)]
