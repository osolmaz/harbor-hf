from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, NoReturn, Protocol, cast

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationDelete, HfApi
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harbor_hf.models import ComponentKind, EvaluationId, PublicationRole
from harbor_hf.result_publisher import (
    IndexReceipt,
    PublisherLeaseStore,
    _regular_blob,
    publisher_lease_path,
)
from harbor_hf.results import (
    ArtifactRow,
    CatalogRow,
    DatasetFile,
    Digest,
    ExecutionRow,
    GlobalIndexRow,
    MetricRow,
    ProjectionFileReference,
    PublicationProvenance,
    ResultCompositionManifest,
    ResultProjection,
    ResultPublication,
    ResultTables,
    RunRow,
    TrialRow,
    build_catalog_lookup_file,
    build_catalog_publication_lookup_file,
    build_catalog_row,
    build_catalog_window_file,
    build_global_index_row,
    build_index_file,
    build_index_window_file,
    build_result_publication,
    read_catalog_file,
)

_WINDOW_SIZES = tuple(2**power for power in range(12))
_LEASE_TTL = timedelta(minutes=30)
_LEGACY_CATALOG_PREFIX = "data/catalog/schema=v1/windows/"
_RESULT_MARKER_PREFIX = "cutovers"
_INDEX_MARKER_PREFIX = "data/catalog/schema=v1/cutovers"


class CatalogCutoverError(RuntimeError):
    """Raised when an explicit catalog cutover cannot be applied safely."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CatalogClassification(FrozenModel):
    publication_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    evaluation_id: EvaluationId
    role: PublicationRole
    execution_profile_sha256: Digest
    component_kind: ComponentKind | None = None
    source_publication_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def values_are_consistent(self) -> CatalogClassification:
        if (self.role == "component") != (self.component_kind is not None):
            raise ValueError("component kind conflicts with publication role")
        if self.role != "final" and self.source_publication_ids:
            raise ValueError(
                "only final publications may reference source publications"
            )
        if len(set(self.source_publication_ids)) != len(self.source_publication_ids):
            raise ValueError("source publication IDs must be unique")
        return self


class CatalogCutoverPlan(FrozenModel):
    schema_version: Literal["harbor-hf/catalog-cutover/v1"] = (
        "harbor-hf/catalog-cutover/v1"
    )
    cutover_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    result_dataset: str = Field(pattern=r"^[^/]+/[^/]+$")
    index_dataset: str = Field(pattern=r"^[^/]+/[^/]+$")
    source_catalog_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    source_catalog_path: str = Field(min_length=1)
    expected_result_head: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    expected_index_head: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    classifications: list[CatalogClassification] = Field(min_length=1)

    @model_validator(mode="after")
    def identities_are_unique(self) -> CatalogCutoverPlan:
        if self.result_dataset == self.index_dataset:
            raise ValueError("result and index Datasets must be distinct")
        identities = [item.publication_id for item in self.classifications]
        if len(set(identities)) != len(identities):
            raise ValueError("cutover classifications contain duplicate publications")
        known = set(identities)
        if any(
            source not in known
            for item in self.classifications
            for source in item.source_publication_ids
        ):
            raise ValueError("cutover source publication is not classified")
        return self


class CatalogCutoverResult(FrozenModel):
    cutover_id: str
    result_dataset: str
    result_revision: str
    index_dataset: str
    index_revision: str
    primary_publications: int
    audit_publications: int


class CutoverDatasetApi(Protocol):
    def repo_info(self, repo_id: str, **kwargs: object) -> object: ...

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: object) -> str: ...

    def list_repo_files(self, repo_id: str, **kwargs: object) -> list[str]: ...

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object: ...


class HubCatalogCutover:
    """Apply one explicit, parent-checked V1 catalog cutover."""

    def __init__(
        self,
        *,
        publisher_id: str,
        leases: PublisherLeaseStore,
        api: CutoverDatasetApi | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not publisher_id:
            raise ValueError("publisher ID is required")
        self.publisher_id = publisher_id
        self.leases = leases
        self.api = api or cast(CutoverDatasetApi, HfApi())
        self.clock = clock

    def apply(self, plan: CatalogCutoverPlan) -> CatalogCutoverResult:
        datasets = sorted({plan.result_dataset, plan.index_dataset})
        acquired: list[tuple[str, dict[str, str]]] = []
        try:
            for dataset in datasets:
                acquired.append((dataset, self._acquire(dataset)))
            return self._apply_locked(plan)
        finally:
            for dataset, owner in reversed(acquired):
                self.leases.release(publisher_lease_path(dataset), owner)

    def _acquire(self, dataset: str) -> dict[str, str]:
        owner = {
            "publisher_id": self.publisher_id,
            "destination": dataset,
            "expires_at": (self.clock() + _LEASE_TTL).isoformat(),
        }
        self.leases.acquire(publisher_lease_path(dataset), owner)
        return owner

    def _apply_locked(self, plan: CatalogCutoverPlan) -> CatalogCutoverResult:
        index_head = self._head(plan.index_dataset)
        if index_head != plan.expected_index_head:
            return self._completed_cutover(plan, index_head)
        result_head = self._head(plan.result_dataset)
        result_marker = f"{_RESULT_MARKER_PREFIX}/{plan.cutover_id}.json"
        if result_head != plan.expected_result_head and not self._marker_matches(
            plan.result_dataset, result_marker, result_head, plan
        ):
            self._raise_moved(
                plan.result_dataset, plan.expected_result_head, result_head
            )
        legacy = self._legacy_catalog(plan, plan.source_catalog_revision)
        expected_head_catalog = self._legacy_catalog(plan, plan.expected_index_head)
        classified = {item.publication_id: item for item in plan.classifications}
        referenced = {
            source
            for item in plan.classifications
            for source in item.source_publication_ids
        }
        discovered = set(legacy) | set(expected_head_catalog)
        if not discovered.issubset(classified) or set(classified) - discovered != (
            referenced - discovered
        ):
            raise CatalogCutoverError(
                "cutover must classify both catalogs and any missing sources"
            )
        publications = [
            self._migrate_publication(plan, classification, classified)
            for classification in classified.values()
        ]
        if result_head == plan.expected_result_head:
            result_revision = self._commit_results(plan, publications)
        elif self._marker_matches(
            plan.result_dataset,
            result_marker,
            result_head,
            plan,
        ):
            self._verify_result_files(plan, publications, result_head)
            result_revision = result_head
        else:
            self._raise_moved(
                plan.result_dataset, plan.expected_result_head, result_head
            )
        self._require_head(plan.index_dataset, plan.expected_index_head)
        index_revision = self._commit_index(plan, publications, result_revision)
        return CatalogCutoverResult(
            cutover_id=plan.cutover_id,
            result_dataset=plan.result_dataset,
            result_revision=result_revision,
            index_dataset=plan.index_dataset,
            index_revision=index_revision,
            primary_publications=sum(
                item.role == "final" for item in plan.classifications
            ),
            audit_publications=len(plan.classifications),
        )

    def _completed_cutover(
        self, plan: CatalogCutoverPlan, index_revision: str
    ) -> CatalogCutoverResult:
        marker = f"{_INDEX_MARKER_PREFIX}/{plan.cutover_id}.json"
        if not self._marker_matches(plan.index_dataset, marker, index_revision, plan):
            self._raise_moved(
                plan.index_dataset, plan.expected_index_head, index_revision
            )
        audit = self._catalog_at(plan, index_revision, scope="audit")
        primary = self._catalog_at(plan, index_revision, scope="primary")
        expected = {item.publication_id: item for item in plan.classifications}
        observed = {item.publication_id: item for item in audit}
        if set(observed) != set(expected) or any(
            (
                row.evaluation_id,
                row.publication_role,
                row.component_kind,
                row.source_publication_ids,
            )
            != (
                expected[publication_id].evaluation_id,
                expected[publication_id].role,
                expected[publication_id].component_kind,
                expected[publication_id].source_publication_ids,
            )
            for publication_id, row in observed.items()
        ):
            raise CatalogCutoverError("completed cutover audit catalog conflicts")
        expected_primary = {
            item.publication_id for item in plan.classifications if item.role == "final"
        }
        if {item.publication_id for item in primary} != expected_primary:
            raise CatalogCutoverError("completed cutover primary catalog conflicts")
        result_revisions = {item.result_revision for item in audit}
        if len(result_revisions) != 1:
            raise CatalogCutoverError("completed cutover has mixed result revisions")
        result_revision = result_revisions.pop()
        if not self._marker_matches(
            plan.result_dataset,
            f"{_RESULT_MARKER_PREFIX}/{plan.cutover_id}.json",
            result_revision,
            plan,
        ):
            raise CatalogCutoverError("completed cutover result marker conflicts")
        return CatalogCutoverResult(
            cutover_id=plan.cutover_id,
            result_dataset=plan.result_dataset,
            result_revision=result_revision,
            index_dataset=plan.index_dataset,
            index_revision=index_revision,
            primary_publications=len(primary),
            audit_publications=len(audit),
        )

    def _catalog_at(
        self,
        plan: CatalogCutoverPlan,
        revision: str,
        *,
        scope: Literal["primary", "audit"],
    ) -> list[CatalogRow]:
        path = f"data/catalog/schema=v1/{scope}/windows/2048.parquet"
        try:
            return read_catalog_file(self._read(plan.index_dataset, path, revision))
        except ValueError as error:
            raise CatalogCutoverError(
                f"completed cutover {scope} catalog is invalid"
            ) from error

    def _verify_result_files(
        self,
        plan: CatalogCutoverPlan,
        publications: Sequence[ResultPublication],
        revision: str,
    ) -> None:
        expected = [
            file
            for publication in publications
            for file in (
                *publication.files,
                DatasetFile(
                    path=publication.receipt_path,
                    content=publication.receipt,
                ),
            )
        ]
        for file in expected:
            if self._read(plan.result_dataset, file.path, revision) != file.content:
                raise CatalogCutoverError(
                    f"recovered result file conflicts: {file.path}"
                )

    def _marker_matches(
        self,
        dataset: str,
        path: str,
        revision: str,
        plan: CatalogCutoverPlan,
    ) -> bool:
        files = self.api.list_repo_files(
            dataset, repo_type="dataset", revision=revision
        )
        return path in files and self._read(dataset, path, revision) == _json_bytes(
            plan.model_dump(mode="json")
        )

    def _legacy_catalog(
        self, plan: CatalogCutoverPlan, revision: str
    ) -> dict[str, Mapping[str, object]]:
        content = self._read(
            plan.index_dataset,
            plan.source_catalog_path,
            revision,
        )
        try:
            rows = pq.read_table(pa.BufferReader(content)).to_pylist()
        except (pa.ArrowException, OSError) as error:
            raise CatalogCutoverError("source catalog is invalid") from error
        values = {
            str(row["publication_id"]): row
            for row in rows
            if isinstance(row, Mapping) and row.get("publication_id")
        }
        if len(values) != len(rows):
            raise CatalogCutoverError(
                "source catalog has invalid publication identities"
            )
        return values

    def _migrate_publication(
        self,
        plan: CatalogCutoverPlan,
        classification: CatalogClassification,
        classifications: Mapping[str, CatalogClassification],
    ) -> ResultPublication:
        dataset = plan.result_dataset
        revision = plan.expected_result_head
        projection_path = f"projections/schema=v1/{classification.publication_id}.json"
        try:
            projection_value = json.loads(
                self._read(dataset, projection_path, revision)
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CatalogCutoverError("source projection is invalid") from error
        if not isinstance(projection_value, dict):
            raise CatalogCutoverError("source projection is invalid")
        observed_profile = projection_value.get("execution_profile_sha256")
        if (
            observed_profile is not None
            and observed_profile != classification.execution_profile_sha256
        ):
            raise CatalogCutoverError(
                "source projection execution profile conflicts with classification"
            )
        projection_value["execution_profile_sha256"] = (
            classification.execution_profile_sha256
        )
        projection = ResultProjection.model_validate(projection_value)
        if projection.publication_id != classification.publication_id:
            raise CatalogCutoverError("source projection publication conflicts")
        rows = {
            table: self._parquet_rows(dataset, reference, revision)
            for table, reference in projection.tables.items()
        }
        run_value = rows["runs"][0]
        run_value.update(
            {
                "evaluation_id": classification.evaluation_id,
                "publication_role": classification.role,
                "component_kind": classification.component_kind,
                "source_publication_ids": classification.source_publication_ids,
            }
        )
        run_value.setdefault(
            "unsupported_count",
            sum(value.get("outcome") == "unsupported" for value in rows["trials"]),
        )
        tables = ResultTables(
            publication_id=classification.publication_id,
            runs=[RunRow.model_validate(run_value)],
            trials=[TrialRow.model_validate(value) for value in rows["trials"]],
            executions=[
                ExecutionRow.model_validate(value) for value in rows["executions"]
            ],
            metrics=[MetricRow.model_validate(value) for value in rows["metrics"]],
            artifacts=[
                ArtifactRow.model_validate(value) for value in rows["artifacts"]
            ],
            provenance=PublicationProvenance(
                envelope_sha256=projection.envelope_sha256,
                projection_version=projection.projection_version,
                sanitizer_version=projection.sanitizer_version,
                execution_profile_sha256=projection.execution_profile_sha256,
                harbor_bundle_manifest_sha256s=(
                    projection.harbor_bundle_manifest_sha256s
                ),
                harbor_archive_sha256s=projection.harbor_archive_sha256s,
            ),
        )
        extras: list[DatasetFile] = []
        if tables.runs[0].result_kind == "composed":
            path = f"compositions/{classification.publication_id}.json"
            manifest_bytes = self._read(dataset, path, revision)
            manifest = ResultCompositionManifest.model_validate_json(manifest_bytes)
            if sorted(classification.source_publication_ids) != sorted(
                source.publication_id for source in manifest.sources
            ):
                raise CatalogCutoverError(
                    "composed source publications conflict with classification"
                )
            if any(
                classifications[source.publication_id].role != "component"
                or classifications[source.publication_id].component_kind != source.role
                or classifications[source.publication_id].evaluation_id
                != classification.evaluation_id
                for source in manifest.sources
            ):
                raise CatalogCutoverError(
                    "composed source classifications conflict with manifest"
                )
            extras.append(
                DatasetFile(
                    path=path,
                    content=manifest_bytes,
                )
            )
        return build_result_publication(tables, extra_files=extras)

    def _parquet_rows(
        self,
        dataset: str,
        reference: ProjectionFileReference,
        revision: str,
    ) -> list[dict[str, object]]:
        content = self._read(dataset, reference.path, revision)
        if _sha256(content) != reference.sha256:
            raise CatalogCutoverError(
                f"normalized table checksum differs: {reference.path}"
            )
        try:
            rows = pq.read_table(pa.BufferReader(content)).to_pylist()
        except (pa.ArrowException, OSError) as error:
            raise CatalogCutoverError(
                f"normalized table is invalid: {reference.path}"
            ) from error
        if len(rows) != reference.row_count:
            raise CatalogCutoverError(
                f"normalized table row count differs: {reference.path}"
            )
        return rows

    def _commit_results(
        self, plan: CatalogCutoverPlan, publications: Sequence[ResultPublication]
    ) -> str:
        operations: list[object] = [
            _regular_blob(file.path, file.content)
            for publication in publications
            for file in publication.files
        ]
        operations.extend(
            _regular_blob(publication.receipt_path, publication.receipt)
            for publication in publications
        )
        operations.append(
            _regular_blob(
                f"cutovers/{plan.cutover_id}.json",
                _json_bytes(plan.model_dump(mode="json")),
            )
        )
        response = self.api.create_commit(
            plan.result_dataset,
            operations,
            commit_message=f"refactor: cut over catalog {plan.cutover_id}",
            repo_type="dataset",
            revision="main",
            parent_commit=plan.expected_result_head,
        )
        return self._commit_oid(response, plan.result_dataset)

    def _commit_index(
        self,
        plan: CatalogCutoverPlan,
        publications: Sequence[ResultPublication],
        result_revision: str,
    ) -> str:
        index_rows = [
            build_global_index_row(
                publication.tables,
                result_dataset=plan.result_dataset,
                result_revision=result_revision,
            )
            for publication in publications
        ]
        catalog_rows = [
            build_catalog_row(
                publication.tables,
                result_dataset=plan.result_dataset,
                result_revision=result_revision,
                projection=_projection_file(publication),
            )
            for publication in publications
        ]
        index_rows.sort(
            key=lambda item: (item.completed_at, item.publication_id), reverse=True
        )
        catalog_rows.sort(
            key=lambda item: (item.completed_at, item.publication_id), reverse=True
        )
        primary = [row for row in catalog_rows if row.publication_role == "final"]
        files = self._index_files(
            plan, index_rows, catalog_rows, primary, result_revision
        )
        legacy_paths = [
            path
            for path in self.api.list_repo_files(
                plan.index_dataset,
                repo_type="dataset",
                revision=plan.expected_index_head,
            )
            if path.startswith(_LEGACY_CATALOG_PREFIX) and path.endswith(".parquet")
        ]
        operations: list[object] = [
            *(_regular_blob(file.path, file.content) for file in files),
            *(CommitOperationDelete(path_in_repo=path) for path in legacy_paths),
        ]
        response = self.api.create_commit(
            plan.index_dataset,
            operations,
            commit_message=f"refactor: cut over catalog {plan.cutover_id}",
            repo_type="dataset",
            revision="main",
            parent_commit=plan.expected_index_head,
        )
        return self._commit_oid(response, plan.index_dataset)

    def _index_files(
        self,
        plan: CatalogCutoverPlan,
        index_rows: Sequence[GlobalIndexRow],
        audit: Sequence[CatalogRow],
        primary: Sequence[CatalogRow],
        result_revision: str,
    ) -> list[DatasetFile]:
        files: list[DatasetFile] = []
        for row in index_rows:
            index_file = build_index_file(row)
            receipt = IndexReceipt(
                publication_id=row.publication_id,
                result_dataset=plan.result_dataset,
                result_revision=result_revision,
                index_path=index_file.path,
                index_sha256=_sha256(index_file.content),
            )
            files.extend(
                [
                    index_file,
                    DatasetFile(
                        path=f"publications/{row.publication_id}.json",
                        content=_json_bytes(receipt.model_dump(mode="json")),
                    ),
                ]
            )
        files.extend(
            build_index_window_file(index_rows, size) for size in _WINDOW_SIZES
        )
        files.extend(
            build_catalog_window_file(audit, size, scope="audit")
            for size in _WINDOW_SIZES
        )
        files.extend(
            build_catalog_window_file(primary, size, scope="primary")
            for size in _WINDOW_SIZES
        )
        files.extend(build_catalog_lookup_file(row) for row in audit)
        files.extend(build_catalog_publication_lookup_file(row) for row in audit)
        files.append(
            DatasetFile(
                path=f"data/catalog/schema=v1/cutovers/{plan.cutover_id}.json",
                content=_json_bytes(plan.model_dump(mode="json")),
            )
        )
        return files

    def _require_head(self, dataset: str, expected: str) -> None:
        observed = self._head(dataset)
        if observed != expected:
            self._raise_moved(dataset, expected, observed)

    def _head(self, dataset: str) -> str:
        observed = getattr(
            self.api.repo_info(dataset, repo_type="dataset", revision="main"),
            "sha",
            None,
        )
        if not isinstance(observed, str) or not observed:
            raise CatalogCutoverError(f"Dataset {dataset} returned no identity")
        return observed

    @staticmethod
    def _raise_moved(dataset: str, expected: str, observed: str) -> NoReturn:
        raise CatalogCutoverError(
            f"Dataset {dataset} moved: expected {expected}, observed {observed}"
        )

    def _read(self, dataset: str, path: str, revision: str) -> bytes:
        local_path = self.api.hf_hub_download(
            dataset, path, repo_type="dataset", revision=revision
        )
        try:
            return Path(local_path).read_bytes()
        except OSError as error:
            raise CatalogCutoverError(f"cannot read Dataset path: {path}") from error

    def _commit_oid(self, response: object, dataset: str) -> str:
        oid = getattr(response, "oid", None)
        if isinstance(oid, str) and oid:
            return oid
        observed = getattr(
            self.api.repo_info(dataset, repo_type="dataset", revision="main"),
            "sha",
            None,
        )
        if not isinstance(observed, str):
            raise CatalogCutoverError("Dataset commit has no identity")
        return observed


def _projection_file(publication: ResultPublication) -> DatasetFile:
    expected = f"projections/schema=v1/{publication.tables.publication_id}.json"
    return next(file for file in publication.files if file.path == expected)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"
