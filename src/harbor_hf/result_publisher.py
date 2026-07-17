from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, cast

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel, ConfigDict, Field

from harbor_hf.control import CampaignStoreApi
from harbor_hf.results import (
    CatalogDecision,
    CatalogRow,
    CatalogScope,
    DatasetFile,
    GlobalIndexRow,
    ResultPublication,
    build_catalog_lookup_file,
    build_catalog_row,
    build_catalog_window_file,
    build_global_index_row,
    build_index_file,
    build_index_window_file,
    read_catalog_file,
    read_index_file,
)

DatasetApi = CampaignStoreApi

_MAX_COMMIT_ATTEMPTS = 8
_MAX_REGULAR_BLOB_BYTES = 5 * 1024 * 1024
_PUBLISHER_LEASE_TTL = timedelta(minutes=15)
_INDEX_WINDOW_SIZES = tuple(2**power for power in range(12))
_LARGEST_INDEX_WINDOW = _INDEX_WINDOW_SIZES[-1]


class DatasetPublicationError(RuntimeError):
    """Raised when serialized Dataset publication cannot safely complete."""


class PublicationConflict(DatasetPublicationError):
    """Raised when a deterministic publication path has different content."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PublicationResult(FrozenModel):
    publication_id: str
    result_dataset: str
    result_revision: str
    index_dataset: str
    index_revision: str


class CatalogDecisionResult(FrozenModel):
    decision_id: str
    publication_id: str
    action: Literal["promote", "withdraw"]
    index_dataset: str
    index_revision: str


class ResultReceipt(FrozenModel):
    schema_version: Literal["harbor-hf/result-publication/v1"] = (
        "harbor-hf/result-publication/v1"
    )
    publication_id: str
    run_id: str
    source_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    files: dict[str, str]


class IndexReceipt(FrozenModel):
    schema_version: Literal["harbor-hf/index-publication/v1"] = (
        "harbor-hf/index-publication/v1"
    )
    publication_id: str
    result_dataset: str
    result_revision: str
    index_path: str
    index_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class PublisherLeaseStore(Protocol):
    def acquire(self, path: str, owner: dict[str, str]) -> None: ...

    def release(self, path: str, owner: dict[str, str]) -> None: ...


def publisher_lease_path(dataset: str) -> str:
    identity = hashlib.sha256(dataset.encode()).hexdigest()
    return f"coordination/publishers/{identity}.json"


def catalog_decision_event_path(decision_id: str) -> str:
    return _catalog_decision_event_path(decision_id)


def catalog_decision_latest_path(publication_id: str) -> str:
    return _catalog_decision_latest_path(publication_id)


class HubDatasetPublisher:
    """Publish result files through leased, parent-checked Dataset commits."""

    def __init__(
        self,
        *,
        publisher_id: str,
        leases: PublisherLeaseStore,
        api: CampaignStoreApi | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not publisher_id:
            raise ValueError("publisher ID is required")
        self.publisher_id = publisher_id
        self.leases = leases
        self.api = api or cast(CampaignStoreApi, HfApi())
        self.clock = clock

    def publish(
        self,
        publication: ResultPublication,
        *,
        result_dataset: str,
        index_dataset: str,
    ) -> PublicationResult:
        if result_dataset == index_dataset:
            raise ValueError("result and index Datasets must be distinct")
        result_revision = self._with_lease(
            result_dataset,
            lambda: self._publish_result(publication, result_dataset),
        )
        result_revision, index_revision = self._with_lease(
            index_dataset,
            lambda: self._publish_index(
                publication,
                result_dataset=result_dataset,
                result_revision=result_revision,
                index_dataset=index_dataset,
            ),
        )
        return PublicationResult(
            publication_id=publication.tables.publication_id,
            result_dataset=result_dataset,
            result_revision=result_revision,
            index_dataset=index_dataset,
            index_revision=index_revision,
        )

    def decide_catalog(
        self,
        decision: CatalogDecision,
        *,
        index_dataset: str,
    ) -> CatalogDecisionResult:
        revision = self._with_lease(
            index_dataset,
            lambda: self._record_catalog_decision(decision, index_dataset),
        )
        return CatalogDecisionResult(
            decision_id=decision.decision_id,
            publication_id=decision.publication_id,
            action=decision.action,
            index_dataset=index_dataset,
            index_revision=revision,
        )

    def _with_lease[Value](self, dataset: str, operation: Callable[[], Value]) -> Value:
        path = publisher_lease_path(dataset)
        owner = {
            "publisher_id": self.publisher_id,
            "destination": dataset,
            "expires_at": (self.clock() + _PUBLISHER_LEASE_TTL).isoformat(),
        }
        self.leases.acquire(path, owner)
        try:
            return operation()
        finally:
            self.leases.release(path, owner)

    def _publish_result(self, publication: ResultPublication, dataset: str) -> str:
        operations: list[object] = [
            _regular_blob(item.path, item.content) for item in publication.files
        ]
        operations.append(_regular_blob(publication.receipt_path, publication.receipt))
        for _attempt in range(_MAX_COMMIT_ATTEMPTS):
            head = self._head(dataset)
            if self._exists(dataset, publication.receipt_path, head):
                self._adopt_result(publication, dataset, head)
                return head
            try:
                response = self.api.create_commit(
                    dataset,
                    operations,
                    commit_message=(
                        f"feat: publish results {publication.tables.runs[0].run_id}"
                    ),
                    repo_type="dataset",
                    revision="main",
                    parent_commit=head,
                )
                return self._commit_oid(response, dataset)
            except HfHubHTTPError as error:
                if not _is_parent_conflict(error):
                    raise
        raise DatasetPublicationError("result Dataset remained contended")

    def _adopt_result(
        self, publication: ResultPublication, dataset: str, revision: str
    ) -> None:
        try:
            observed = ResultReceipt.model_validate_json(
                self._read(dataset, publication.receipt_path, revision)
            )
            expected = ResultReceipt.model_validate_json(publication.receipt)
        except Exception as error:
            raise PublicationConflict("result publication receipt conflicts") from error
        if (
            observed.schema_version != expected.schema_version
            or observed.publication_id != expected.publication_id
            or observed.run_id != expected.run_id
            or observed.source_checksum != expected.source_checksum
            or observed.files != expected.files
        ):
            raise PublicationConflict("result publication receipt conflicts")
        for path, checksum in observed.files.items():
            if not self._exists(dataset, path, revision):
                raise DatasetPublicationError(
                    "result publication receipt is incomplete"
                )
            if _sha256(self._read(dataset, path, revision)) != checksum:
                raise DatasetPublicationError("published result file is corrupted")

    def _publish_index(
        self,
        publication: ResultPublication,
        *,
        result_dataset: str,
        result_revision: str,
        index_dataset: str,
    ) -> tuple[str, str]:
        tables = publication.tables
        projection = _projection_file(publication)
        receipt_path = f"publications/{tables.publication_id}.json"
        for _attempt in range(_MAX_COMMIT_ATTEMPTS):
            head = self._head(index_dataset)
            if self._exists(index_dataset, receipt_path, head):
                receipt = self._adopt_index(
                    tables.publication_id,
                    result_dataset,
                    index_dataset,
                    receipt_path,
                    head,
                )
                row = self._read_index_row(index_dataset, receipt.index_path, head)
                expected_row = build_global_index_row(
                    tables,
                    result_dataset=result_dataset,
                    result_revision=receipt.result_revision,
                )
                if row != expected_row:
                    raise PublicationConflict("global index row conflicts")
                indexed_result_revision = receipt.result_revision
            else:
                receipt = None
                indexed_result_revision = result_revision
                row = build_global_index_row(
                    tables,
                    result_dataset=result_dataset,
                    result_revision=result_revision,
                )
            index_file = build_index_file(row)
            windows = self._index_windows(index_dataset, head, row)
            catalog = build_catalog_row(
                tables,
                result_dataset=result_dataset,
                result_revision=indexed_result_revision,
                projection=projection,
            )
            catalog_windows = [
                *self._catalog_windows(index_dataset, head, catalog, scope="audit"),
                *self._catalog_windows(index_dataset, head, catalog, scope="primary"),
            ]
            catalog_lookup = build_catalog_lookup_file(catalog)
            index_updates = [
                *windows,
                *catalog_windows,
                catalog_lookup,
            ]
            if receipt is not None and self._windows_match(
                index_dataset, head, index_updates
            ):
                return receipt.result_revision, head
            receipt = receipt or self._index_receipt(row, index_file)
            operations: list[object] = [
                _regular_blob(window.path, window.content) for window in index_updates
            ]
            if not self._exists(index_dataset, receipt_path, head):
                operations.extend(
                    (
                        _regular_blob(index_file.path, index_file.content),
                        _regular_blob(
                            receipt_path,
                            _json_bytes(receipt.model_dump(mode="json")),
                        ),
                    )
                )
            try:
                response = self.api.create_commit(
                    index_dataset,
                    operations,
                    commit_message=f"feat: index result {tables.runs[0].run_id}",
                    repo_type="dataset",
                    revision="main",
                    parent_commit=head,
                )
                return indexed_result_revision, self._commit_oid(
                    response, index_dataset
                )
            except HfHubHTTPError as error:
                if not _is_parent_conflict(error):
                    raise
        raise DatasetPublicationError("global index Dataset remained contended")

    def _index_windows(
        self, dataset: str, revision: str, row: GlobalIndexRow
    ) -> list[DatasetFile]:
        largest_path = build_index_window_file([], _LARGEST_INDEX_WINDOW).path
        if self._exists(dataset, largest_path, revision):
            existing = read_index_file(self._read(dataset, largest_path, revision))
        else:
            existing = self._individual_index_rows(dataset, revision)
        by_run = _replace_active_run(existing, row)
        ordered = sorted(
            by_run.values(),
            key=lambda item: (item.completed_at, item.publication_id),
            reverse=True,
        )[:_LARGEST_INDEX_WINDOW]
        return [build_index_window_file(ordered, size) for size in _INDEX_WINDOW_SIZES]

    def _individual_index_rows(
        self, dataset: str, revision: str
    ) -> list[GlobalIndexRow]:
        paths = self.api.list_repo_files(
            dataset, repo_type="dataset", revision=revision
        )
        rows: list[GlobalIndexRow] = []
        for path in paths:
            if (
                path.startswith("data/index/schema=v1/")
                and path.endswith(".parquet")
                and "/windows/" not in path
            ):
                rows.extend(read_index_file(self._read(dataset, path, revision)))
        return rows

    def _catalog_windows(
        self,
        dataset: str,
        revision: str,
        row: CatalogRow,
        *,
        scope: CatalogScope,
    ) -> list[DatasetFile]:
        largest_path = build_catalog_window_file(
            [], _LARGEST_INDEX_WINDOW, scope=scope
        ).path
        if self._exists(dataset, largest_path, revision):
            existing = read_catalog_file(self._read(dataset, largest_path, revision))
        else:
            index_path = build_index_window_file([], _LARGEST_INDEX_WINDOW).path
            index_rows = (
                read_index_file(self._read(dataset, index_path, revision))
                if self._exists(dataset, index_path, revision)
                else self._individual_index_rows(dataset, revision)
            )
            if index_rows:
                raise DatasetPublicationError(
                    "canonical catalog snapshot is required before publication"
                )
            existing = []
        by_run = {item.run_id: item for item in existing}
        withdrawn = (
            scope == "primary"
            and self._latest_catalog_decision(dataset, revision, row.publication_id)
            == "withdraw"
        )
        if scope == "audit" or (row.publication_role == "final" and not withdrawn):
            by_run = _replace_active_run(existing, row)
        ordered = sorted(
            by_run.values(),
            key=lambda item: (item.completed_at, item.publication_id),
            reverse=True,
        )[:_LARGEST_INDEX_WINDOW]
        return [
            build_catalog_window_file(ordered, size, scope=scope)
            for size in _INDEX_WINDOW_SIZES
        ]

    def _record_catalog_decision(self, decision: CatalogDecision, dataset: str) -> str:
        event_path = _catalog_decision_event_path(decision.decision_id)
        latest_path = _catalog_decision_latest_path(decision.publication_id)
        content = _json_bytes(decision.model_dump(mode="json"))
        for _attempt in range(_MAX_COMMIT_ATTEMPTS):
            head = self._head(dataset)
            if self._exists(dataset, event_path, head):
                if self._read(dataset, event_path, head) != content:
                    raise PublicationConflict("catalog decision ID conflicts")
                return head
            latest = self._read_catalog_decision(dataset, latest_path, head)
            if latest is not None and decision.created_at <= latest.created_at:
                raise PublicationConflict("catalog decision is not newer than latest")
            windows = self._catalog_decision_windows(dataset, head, decision)
            operations: list[object] = [
                _regular_blob(event_path, content),
                _regular_blob(latest_path, content),
                *(_regular_blob(item.path, item.content) for item in windows),
            ]
            try:
                response = self.api.create_commit(
                    dataset,
                    operations,
                    commit_message=(
                        f"{decision.action}: catalog {decision.publication_id}"
                    ),
                    repo_type="dataset",
                    revision="main",
                    parent_commit=head,
                )
                return self._commit_oid(response, dataset)
            except HfHubHTTPError as error:
                if not _is_parent_conflict(error):
                    raise
        raise DatasetPublicationError("catalog Dataset remained contended")

    def _catalog_decision_windows(
        self, dataset: str, revision: str, decision: CatalogDecision
    ) -> list[DatasetFile]:
        audit = self._read_catalog_window(dataset, revision, scope="audit")
        target = next(
            (row for row in audit if row.publication_id == decision.publication_id),
            None,
        )
        if target is None:
            raise DatasetPublicationError("catalog decision publication is unknown")
        if decision.action == "promote" and target.publication_role != "final":
            raise DatasetPublicationError("only final publications can be promoted")
        primary = {
            row.run_id: row
            for row in self._read_catalog_window(dataset, revision, scope="primary")
        }
        if decision.action == "withdraw":
            primary = {
                run_id: row
                for run_id, row in primary.items()
                if row.publication_id != decision.publication_id
            }
        else:
            primary = _replace_active_run(list(primary.values()), target)
        ordered = sorted(
            primary.values(),
            key=lambda item: (item.completed_at, item.publication_id),
            reverse=True,
        )[:_LARGEST_INDEX_WINDOW]
        return [
            build_catalog_window_file(ordered, size, scope="primary")
            for size in _INDEX_WINDOW_SIZES
        ]

    def _read_catalog_window(
        self, dataset: str, revision: str, *, scope: CatalogScope
    ) -> list[CatalogRow]:
        path = build_catalog_window_file([], _LARGEST_INDEX_WINDOW, scope=scope).path
        if not self._exists(dataset, path, revision):
            raise DatasetPublicationError("canonical catalog snapshot is required")
        return read_catalog_file(self._read(dataset, path, revision))

    def _read_catalog_decision(
        self, dataset: str, path: str, revision: str
    ) -> CatalogDecision | None:
        if not self._exists(dataset, path, revision):
            return None
        try:
            return CatalogDecision.model_validate_json(
                self._read(dataset, path, revision)
            )
        except Exception as error:
            raise DatasetPublicationError("catalog decision is invalid") from error

    def _latest_catalog_decision(
        self, dataset: str, revision: str, publication_id: str
    ) -> Literal["promote", "withdraw"] | None:
        decision = self._read_catalog_decision(
            dataset,
            _catalog_decision_latest_path(publication_id),
            revision,
        )
        return decision.action if decision is not None else None

    def _windows_match(
        self, dataset: str, revision: str, windows: list[DatasetFile]
    ) -> bool:
        return all(
            self._exists(dataset, window.path, revision)
            and self._read(dataset, window.path, revision) == window.content
            for window in windows
        )

    def _read_index_row(self, dataset: str, path: str, revision: str) -> GlobalIndexRow:
        rows = read_index_file(self._read(dataset, path, revision))
        if len(rows) != 1:
            raise DatasetPublicationError(
                "global index publication must contain exactly one row"
            )
        return rows[0]

    def _adopt_index(
        self,
        publication_id: str,
        result_dataset: str,
        index_dataset: str,
        receipt_path: str,
        revision: str,
    ) -> IndexReceipt:
        try:
            receipt = IndexReceipt.model_validate_json(
                self._read(index_dataset, receipt_path, revision)
            )
        except Exception as error:
            raise DatasetPublicationError("global index receipt is invalid") from error
        if (
            receipt.publication_id != publication_id
            or receipt.result_dataset != result_dataset
        ):
            raise PublicationConflict("global index receipt conflicts")
        if not self._exists(index_dataset, receipt.index_path, revision):
            raise DatasetPublicationError("global index receipt is incomplete")
        if (
            _sha256(self._read(index_dataset, receipt.index_path, revision))
            != receipt.index_sha256
        ):
            raise DatasetPublicationError("global index file is corrupted")
        return receipt

    def _head(self, dataset: str) -> str:
        revision = getattr(
            self.api.repo_info(dataset, repo_type="dataset", revision="main"),
            "sha",
            None,
        )
        if not isinstance(revision, str) or not revision:
            raise DatasetPublicationError("Dataset has no commit identity")
        return revision

    def _exists(self, dataset: str, path: str, revision: str) -> bool:
        return bool(
            self.api.get_paths_info(
                dataset,
                path,
                repo_type="dataset",
                revision=revision,
            )
        )

    def _read(self, dataset: str, path: str, revision: str) -> bytes:
        local_path = self.api.hf_hub_download(
            dataset,
            path,
            repo_type="dataset",
            revision=revision,
        )
        try:
            return Path(local_path).read_bytes()
        except OSError as error:
            raise DatasetPublicationError("Dataset receipt cannot be read") from error

    def _commit_oid(self, response: object, dataset: str) -> str:
        oid = getattr(response, "oid", None)
        if isinstance(oid, str) and oid:
            return oid
        return self._head(dataset)

    @staticmethod
    def _index_receipt(row: GlobalIndexRow, index_file: DatasetFile) -> IndexReceipt:
        return IndexReceipt(
            publication_id=row.publication_id,
            result_dataset=row.result_dataset,
            result_revision=row.result_revision,
            index_path=index_file.path,
            index_sha256=_sha256(index_file.content),
        )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _catalog_decision_event_path(decision_id: str) -> str:
    return f"data/catalog/schema=v1/decisions/events/{decision_id}.json"


def _catalog_decision_latest_path(publication_id: str) -> str:
    return f"data/catalog/schema=v1/decisions/latest/{publication_id}.json"


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _replace_active_run[Row: GlobalIndexRow | CatalogRow](
    existing: list[Row], row: Row
) -> dict[str, Row]:
    by_run = {
        item.run_id: item
        for item in sorted(
            existing,
            key=lambda item: (item.completed_at, item.publication_id),
        )
    }
    by_run[row.run_id] = row
    return by_run


def _projection_file(publication: ResultPublication) -> DatasetFile:
    path = f"projections/schema=v1/{publication.tables.publication_id}.json"
    projection = next(
        (item for item in publication.files if item.path == path),
        None,
    )
    if projection is None:
        raise DatasetPublicationError(
            "canonical result publication has no projection manifest"
        )
    return projection


def _is_parent_conflict(error: HfHubHTTPError) -> bool:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) in {409, 412}


def _regular_blob(path: str, content: bytes) -> CommitOperationAdd:
    if len(content) > _MAX_REGULAR_BLOB_BYTES:
        raise DatasetPublicationError(
            f"generated publication file exceeds the regular blob limit: {path}"
        )
    operation = CommitOperationAdd(path_in_repo=path, path_or_fileobj=content)
    operation._upload_mode = "regular"
    return operation
