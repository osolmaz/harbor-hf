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
    CatalogRow,
    CatalogRowV2,
    DatasetFile,
    GlobalIndexRow,
    ResultPublication,
    build_catalog_lookup_file,
    build_catalog_row,
    build_catalog_row_v2,
    build_catalog_v2_lookup_file,
    build_catalog_v2_window_file,
    build_catalog_window_file,
    build_global_index_row,
    build_index_file,
    build_index_window_file,
    catalog_v2_from_legacy,
    read_catalog_file,
    read_catalog_v2_file,
    read_index_file,
)

DatasetApi = CampaignStoreApi

_MAX_COMMIT_ATTEMPTS = 8
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


class CatalogMigrationResult(FrozenModel):
    index_dataset: str
    index_revision: str
    publication_count: int = Field(ge=0)


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

    def migrate_catalog_v2(self, index_dataset: str) -> CatalogMigrationResult:
        return self._with_lease(
            index_dataset,
            lambda: self._migrate_catalog_v2(index_dataset),
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

    def _migrate_catalog_v2(self, dataset: str) -> CatalogMigrationResult:
        for _attempt in range(_MAX_COMMIT_ATTEMPTS):
            head = self._head(dataset)
            rows = self._catalog_v2_migration_rows(dataset, head)
            files = [
                *(
                    build_catalog_v2_window_file(rows, size)
                    for size in _INDEX_WINDOW_SIZES
                ),
                *(build_catalog_v2_lookup_file(row) for row in rows),
            ]
            if self._windows_match(dataset, head, files):
                return CatalogMigrationResult(
                    index_dataset=dataset,
                    index_revision=head,
                    publication_count=len(rows),
                )
            try:
                response = self.api.create_commit(
                    dataset,
                    [
                        CommitOperationAdd(
                            path_in_repo=item.path,
                            path_or_fileobj=item.content,
                        )
                        for item in files
                    ],
                    commit_message="feat: migrate result catalog to v2",
                    repo_type="dataset",
                    revision="main",
                    parent_commit=head,
                )
                return CatalogMigrationResult(
                    index_dataset=dataset,
                    index_revision=self._commit_oid(response, dataset),
                    publication_count=len(rows),
                )
            except HfHubHTTPError as error:
                if not _is_parent_conflict(error):
                    raise
        raise DatasetPublicationError("v2 catalog migration remained contended")

    def _catalog_v2_migration_rows(
        self, dataset: str, revision: str
    ) -> list[CatalogRowV2]:
        legacy_path = build_catalog_window_file([], _LARGEST_INDEX_WINDOW).path
        if not self._exists(dataset, legacy_path, revision):
            raise DatasetPublicationError("index Dataset has no v1 result catalog")
        paths = self.api.list_repo_files(
            dataset, repo_type="dataset", revision=revision
        )
        legacy = read_catalog_file(self._read(dataset, legacy_path, revision))
        legacy.extend(
            self._catalog_lookup_rows(
                dataset,
                revision,
                paths,
                prefix="data/catalog/schema=v1/runs/",
                reader=read_catalog_file,
                label="v1",
            )
        )
        by_publication = {
            row.publication_id: catalog_v2_from_legacy(row) for row in legacy
        }
        v2_path = build_catalog_v2_window_file([], _LARGEST_INDEX_WINDOW).path
        if self._exists(dataset, v2_path, revision):
            by_publication.update(
                {
                    row.publication_id: row
                    for row in read_catalog_v2_file(
                        self._read(dataset, v2_path, revision)
                    )
                }
            )
        for row in self._catalog_lookup_rows(
            dataset,
            revision,
            paths,
            prefix="data/catalog/schema=v2/runs/",
            reader=read_catalog_v2_file,
            label="v2",
        ):
            by_publication[row.publication_id] = row
        return sorted(
            by_publication.values(),
            key=lambda item: (item.completed_at, item.publication_id),
            reverse=True,
        )

    def _catalog_lookup_rows[Row](
        self,
        dataset: str,
        revision: str,
        paths: list[str],
        *,
        prefix: str,
        reader: Callable[[bytes], list[Row]],
        label: str,
    ) -> list[Row]:
        results: list[Row] = []
        for path in paths:
            if not (path.startswith(prefix) and path.endswith(".parquet")):
                continue
            rows = reader(self._read(dataset, path, revision))
            if len(rows) != 1:
                raise DatasetPublicationError(
                    f"{label} catalog lookup must contain exactly one row"
                )
            results.append(rows[0])
        return results

    def _publish_result(self, publication: ResultPublication, dataset: str) -> str:
        operations: list[object] = [
            CommitOperationAdd(
                path_in_repo=item.path,
                path_or_fileobj=item.content,
            )
            for item in publication.files
        ]
        operations.append(
            CommitOperationAdd(
                path_in_repo=publication.receipt_path,
                path_or_fileobj=publication.receipt,
            )
        )
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
        projection = next(
            (
                item
                for item in publication.files
                if item.path == f"projections/schema=v2/{tables.publication_id}.json"
            ),
            None,
        )
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
            )
            catalog_windows = self._catalog_windows(index_dataset, head, catalog)
            catalog_lookup = build_catalog_lookup_file(catalog)
            catalog_v2 = build_catalog_row_v2(
                tables,
                result_dataset=result_dataset,
                result_revision=indexed_result_revision,
                projection=projection,
            )
            catalog_v2_windows = self._catalog_v2_windows(
                index_dataset, head, catalog_v2
            )
            catalog_v2_lookup = build_catalog_v2_lookup_file(catalog_v2)
            index_updates = [
                *windows,
                *catalog_windows,
                catalog_lookup,
                *catalog_v2_windows,
                catalog_v2_lookup,
            ]
            if receipt is not None and self._windows_match(
                index_dataset, head, index_updates
            ):
                return receipt.result_revision, head
            receipt = receipt or self._index_receipt(row, index_file)
            operations: list[object] = [
                CommitOperationAdd(
                    path_in_repo=window.path,
                    path_or_fileobj=window.content,
                )
                for window in index_updates
            ]
            if not self._exists(index_dataset, receipt_path, head):
                operations.extend(
                    (
                        CommitOperationAdd(
                            path_in_repo=index_file.path,
                            path_or_fileobj=index_file.content,
                        ),
                        CommitOperationAdd(
                            path_in_repo=receipt_path,
                            path_or_fileobj=_json_bytes(
                                receipt.model_dump(mode="json")
                            ),
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
            existing = self._legacy_index_rows(dataset, revision)
        by_publication = {item.publication_id: item for item in existing}
        by_publication[row.publication_id] = row
        ordered = sorted(
            by_publication.values(),
            key=lambda item: (item.completed_at, item.publication_id),
            reverse=True,
        )[:_LARGEST_INDEX_WINDOW]
        return [build_index_window_file(ordered, size) for size in _INDEX_WINDOW_SIZES]

    def _legacy_index_rows(self, dataset: str, revision: str) -> list[GlobalIndexRow]:
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
        self, dataset: str, revision: str, row: CatalogRow
    ) -> list[DatasetFile]:
        largest_path = build_catalog_window_file([], _LARGEST_INDEX_WINDOW).path
        if self._exists(dataset, largest_path, revision):
            existing = read_catalog_file(self._read(dataset, largest_path, revision))
        else:
            index_path = build_index_window_file([], _LARGEST_INDEX_WINDOW).path
            index_rows = (
                read_index_file(self._read(dataset, index_path, revision))
                if self._exists(dataset, index_path, revision)
                else self._legacy_index_rows(dataset, revision)
            )
            if index_rows:
                raise DatasetPublicationError(
                    "result catalog migration is required before publication"
                )
            existing = []
        by_publication = {item.publication_id: item for item in existing}
        by_publication[row.publication_id] = row
        ordered = sorted(
            by_publication.values(),
            key=lambda item: (item.completed_at, item.publication_id),
            reverse=True,
        )[:_LARGEST_INDEX_WINDOW]
        return [
            build_catalog_window_file(ordered, size) for size in _INDEX_WINDOW_SIZES
        ]

    def _catalog_v2_windows(
        self, dataset: str, revision: str, row: CatalogRowV2
    ) -> list[DatasetFile]:
        largest_path = build_catalog_v2_window_file([], _LARGEST_INDEX_WINDOW).path
        if self._exists(dataset, largest_path, revision):
            existing = read_catalog_v2_file(self._read(dataset, largest_path, revision))
        else:
            legacy_path = build_catalog_window_file([], _LARGEST_INDEX_WINDOW).path
            existing = (
                [
                    catalog_v2_from_legacy(item)
                    for item in read_catalog_file(
                        self._read(dataset, legacy_path, revision)
                    )
                ]
                if self._exists(dataset, legacy_path, revision)
                else []
            )
        by_publication = {item.publication_id: item for item in existing}
        by_publication[row.publication_id] = row
        ordered = sorted(
            by_publication.values(),
            key=lambda item: (item.completed_at, item.publication_id),
            reverse=True,
        )[:_LARGEST_INDEX_WINDOW]
        return [
            build_catalog_v2_window_file(ordered, size) for size in _INDEX_WINDOW_SIZES
        ]

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


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _is_parent_conflict(error: HfHubHTTPError) -> bool:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) in {409, 412}
