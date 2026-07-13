from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, cast

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel, ConfigDict, Field

from harbor_hf.results import (
    DatasetFile,
    GlobalIndexRow,
    ResultPublication,
    ResultTables,
    build_global_index_row,
    build_index_file,
)

_MAX_COMMIT_ATTEMPTS = 8


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


class IndexReceipt(FrozenModel):
    schema_version: Literal["harbor-hf/index-publication/v1"] = (
        "harbor-hf/index-publication/v1"
    )
    publication_id: str
    result_dataset: str
    result_revision: str
    index_path: str
    index_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class DatasetApi(Protocol):
    def repo_info(self, repo_id: str, **kwargs: object) -> object: ...

    def get_paths_info(
        self, repo_id: str, paths: str | list[str], **kwargs: object
    ) -> list[object]: ...

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: object) -> str: ...

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object: ...


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
        api: DatasetApi | None = None,
    ) -> None:
        if not publisher_id:
            raise ValueError("publisher ID is required")
        self.publisher_id = publisher_id
        self.leases = leases
        self.api = api or cast(DatasetApi, HfApi())

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
                publication.tables,
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

    def _with_lease[Value](self, dataset: str, operation: Callable[[], Value]) -> Value:
        path = publisher_lease_path(dataset)
        owner = {"publisher_id": self.publisher_id, "destination": dataset}
        self.leases.acquire(path, owner)
        try:
            return operation()
        finally:
            self.leases.release(path, owner)

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
        observed = self._read(dataset, publication.receipt_path, revision)
        if observed != publication.receipt:
            raise PublicationConflict("result publication receipt conflicts")
        for item in publication.files:
            if not self._exists(dataset, item.path, revision):
                raise DatasetPublicationError(
                    "result publication receipt is incomplete"
                )
            if _sha256(self._read(dataset, item.path, revision)) != _sha256(
                item.content
            ):
                raise DatasetPublicationError("published result file is corrupted")

    def _publish_index(
        self,
        tables: ResultTables,
        *,
        result_dataset: str,
        result_revision: str,
        index_dataset: str,
    ) -> tuple[str, str]:
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
                return receipt.result_revision, head
            row = build_global_index_row(
                tables,
                result_dataset=result_dataset,
                result_revision=result_revision,
            )
            index_file = build_index_file(row)
            receipt = self._index_receipt(row, index_file)
            try:
                response = self.api.create_commit(
                    index_dataset,
                    [
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
                    ],
                    commit_message=f"feat: index result {tables.runs[0].run_id}",
                    repo_type="dataset",
                    revision="main",
                    parent_commit=head,
                )
                return result_revision, self._commit_oid(response, index_dataset)
            except HfHubHTTPError as error:
                if not _is_parent_conflict(error):
                    raise
        raise DatasetPublicationError("global index Dataset remained contended")

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
