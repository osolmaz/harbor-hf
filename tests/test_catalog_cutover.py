from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from harbor_hf.catalog_cutover import (
    CatalogClassification,
    CatalogCutoverError,
    CatalogCutoverPlan,
    HubCatalogCutover,
)
from harbor_hf.result_publisher import publisher_lease_path
from harbor_hf.results import (
    PublicationProvenance,
    ResultTables,
    RunRow,
    build_catalog_row,
    build_result_publication,
    catalog_publication_lookup_path,
    read_catalog_file,
)

NOW = datetime(2026, 7, 17, 1, 2, 3, tzinfo=UTC)
RESULT_HEAD = "1" * 40
INDEX_HEAD = "2" * 40


class FakeLeases:
    def __init__(self) -> None:
        self.held: set[str] = set()

    def acquire(self, path: str, owner: dict[str, str]) -> None:
        del owner
        assert path not in self.held
        self.held.add(path)

    def release(self, path: str, owner: dict[str, str]) -> None:
        del owner
        self.held.remove(path)


class FailingSecondLease(FakeLeases):
    def acquire(self, path: str, owner: dict[str, str]) -> None:
        if self.held:
            raise RuntimeError("simulated lease contention")
        super().acquire(path, owner)


class FakeApi:
    def __init__(self, root: Path, publication_files: dict[str, bytes]) -> None:
        self.root = root
        self.heads = {"org/results": RESULT_HEAD, "org/index": INDEX_HEAD}
        self.snapshots = {
            ("org/results", RESULT_HEAD): dict(publication_files),
            ("org/index", INDEX_HEAD): {},
        }
        self.commit_count = 0

    def repo_info(self, repo_id: str, **kwargs: object) -> object:
        assert kwargs == {"repo_type": "dataset", "revision": "main"}
        return SimpleNamespace(sha=self.heads[repo_id])

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: object) -> str:
        revision = str(kwargs["revision"])
        content = self.snapshots[(repo_id, revision)][filename]
        destination = self.root / repo_id.replace("/", "-") / revision / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return str(destination)

    def list_repo_files(self, repo_id: str, **kwargs: object) -> list[str]:
        revision = str(kwargs["revision"])
        return list(self.snapshots[(repo_id, revision)])

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object:
        parent = str(kwargs["parent_commit"])
        assert parent == self.heads[repo_id]
        files = dict(self.snapshots[(repo_id, parent)])
        for operation in operations:
            if isinstance(operation, CommitOperationAdd):
                assert isinstance(operation.path_or_fileobj, bytes)
                files[operation.path_in_repo] = operation.path_or_fileobj
            else:
                assert isinstance(operation, CommitOperationDelete)
                files.pop(operation.path_in_repo)
        self.commit_count += 1
        revision = f"{self.commit_count + 2:040x}"
        self.snapshots[(repo_id, revision)] = files
        self.heads[repo_id] = revision
        return SimpleNamespace(oid=revision)


class FailingIndexApi(FakeApi):
    def __init__(self, root: Path, publication_files: dict[str, bytes]) -> None:
        super().__init__(root, publication_files)
        self.fail_index_once = True

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object:
        if repo_id == "org/index" and self.fail_index_once:
            self.fail_index_once = False
            raise RuntimeError("simulated index interruption")
        return super().create_commit(repo_id, operations, **kwargs)


def _publication() -> tuple[ResultTables, dict[str, bytes]]:
    run = RunRow(
        publication_id="publication-one",
        run_id="run-one",
        source_bucket="hf://buckets/private-evidence",
        source_prefix="campaigns/campaign-one/runs/run-one",
        source_checksum="sha256:" + "1" * 64,
        run_lock_path="run.lock.json",
        run_lock_sha256="sha256:" + "2" * 64,
        control_commit="3" * 40,
        campaign_id="campaign-one",
        experiment="experiment-one",
        evaluation_id="old-evaluation",
        publication_role="diagnostic",
        component_kind=None,
        source_publication_ids=[],
        benchmark="shellbench/public-115",
        benchmark_revision="sha256:" + "4" * 64,
        result_kind="ordinary",
        outcome="complete",
        quality="clean",
        created_at=NOW,
        completed_at=NOW + timedelta(minutes=1),
        model_id="model-one",
        model_repo="org/model",
        model_revision="5" * 40,
        deployment_id="deployment-one",
        provider="huggingface",
        region="us-east-1",
        hardware="h200",
        accelerator_count=1,
        agent_id="agent-one",
        agent_name="openclaw",
        agent_revision="1.0.0",
        planned_trial_count=0,
        scored_trial_count=0,
        agent_failed_count=0,
        benchmark_failed_count=0,
        infrastructure_exhausted_count=0,
        unsupported_count=0,
        execution_count=0,
    )
    tables = ResultTables(
        publication_id=run.publication_id,
        runs=[run],
        trials=[],
        executions=[],
        metrics=[],
        artifacts=[],
        provenance=PublicationProvenance(
            envelope_sha256="sha256:" + "6" * 64,
            projection_version="harbor-hf/results-projection/v1",
            sanitizer_version="harbor-hf/public-results/v1",
            execution_profile_sha256="sha256:" + "7" * 64,
            harbor_bundle_manifest_sha256s=["sha256:" + "8" * 64],
            harbor_archive_sha256s=["sha256:" + "9" * 64],
        ),
    )
    publication = build_result_publication(tables)
    return tables, {
        **{file.path: file.content for file in publication.files},
        publication.receipt_path: publication.receipt,
    }


def _plan() -> CatalogCutoverPlan:
    return CatalogCutoverPlan(
        cutover_id="primary-catalog-20260717",
        result_dataset="org/results",
        index_dataset="org/index",
        source_catalog_revision=INDEX_HEAD,
        source_catalog_path="data/catalog/schema=v1/windows/2048.parquet",
        expected_result_head=RESULT_HEAD,
        expected_index_head=INDEX_HEAD,
        classifications=[
            CatalogClassification(
                publication_id="publication-one",
                evaluation_id="evaluation-one",
                role="final",
                execution_profile_sha256="sha256:" + "7" * 64,
            )
        ],
    )


def _prepared_api[Api: FakeApi](tmp_path: Path, kind: type[Api]) -> Api:
    tables, files = _publication()
    projection = next(
        file
        for file in build_result_publication(tables).files
        if file.path.startswith("projections/")
    )
    legacy = build_catalog_row(
        tables,
        result_dataset="org/results",
        result_revision=RESULT_HEAD,
        projection=projection,
    )
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist([legacy.model_dump(mode="python")]), sink)
    api = kind(tmp_path, files)
    legacy_path = "data/catalog/schema=v1/windows/2048.parquet"
    api.snapshots[("org/index", INDEX_HEAD)][legacy_path] = sink.getvalue().to_pybytes()
    return api


def test_cutover_rewrites_v1_and_switches_scoped_catalogs(tmp_path: Path) -> None:
    api = _prepared_api(tmp_path, FakeApi)
    legacy_path = "data/catalog/schema=v1/windows/2048.parquet"
    leases = FakeLeases()

    result = HubCatalogCutover(
        publisher_id="cutover-one", leases=leases, api=api
    ).apply(_plan())

    assert result.primary_publications == 1
    assert result.audit_publications == 1
    assert not leases.held
    index_files = api.snapshots[("org/index", result.index_revision)]
    assert legacy_path not in index_files
    primary = read_catalog_file(
        index_files["data/catalog/schema=v1/primary/windows/2048.parquet"]
    )
    audit = read_catalog_file(
        index_files["data/catalog/schema=v1/audit/windows/2048.parquet"]
    )
    assert primary == audit
    assert primary[0].evaluation_id == "evaluation-one"
    assert primary[0].publication_role == "final"
    assert catalog_publication_lookup_path("publication-one") in index_files
    assert publisher_lease_path("org/results") not in leases.held


def test_cutover_repairs_source_projection_without_execution_profile(
    tmp_path: Path,
) -> None:
    api = _prepared_api(tmp_path, FakeApi)
    path = "projections/schema=v1/publication-one.json"
    projection = json.loads(api.snapshots[("org/results", RESULT_HEAD)][path])
    del projection["execution_profile_sha256"]
    api.snapshots[("org/results", RESULT_HEAD)][path] = (
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )

    result = HubCatalogCutover(
        publisher_id="cutover-one", leases=FakeLeases(), api=api
    ).apply(_plan())

    assert result.primary_publications == 1


def test_cutover_derives_missing_unsupported_count(tmp_path: Path) -> None:
    api = _prepared_api(tmp_path, FakeApi)
    projection_path = "projections/schema=v1/publication-one.json"
    projection = json.loads(
        api.snapshots[("org/results", RESULT_HEAD)][projection_path]
    )
    runs = projection["tables"]["runs"]
    table = pq.read_table(
        pa.BufferReader(api.snapshots[("org/results", RESULT_HEAD)][runs["path"]])
    )
    values = table.to_pylist()
    del values[0]["unsupported_count"]
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(values), sink)
    content = sink.getvalue().to_pybytes()
    api.snapshots[("org/results", RESULT_HEAD)][runs["path"]] = content
    runs["sha256"] = "sha256:" + hashlib.sha256(content).hexdigest()
    api.snapshots[("org/results", RESULT_HEAD)][projection_path] = (
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )

    result = HubCatalogCutover(
        publisher_id="cutover-one", leases=FakeLeases(), api=api
    ).apply(_plan())

    assert result.primary_publications == 1


def test_cutover_rejects_execution_profile_conflict(tmp_path: Path) -> None:
    api = _prepared_api(tmp_path, FakeApi)
    plan = _plan().model_copy(
        update={
            "classifications": [
                _plan()
                .classifications[0]
                .model_copy(update={"execution_profile_sha256": "sha256:" + "a" * 64})
            ]
        }
    )

    with pytest.raises(
        CatalogCutoverError,
        match="source projection execution profile conflicts with classification",
    ):
        HubCatalogCutover(
            publisher_id="cutover-one", leases=FakeLeases(), api=api
        ).apply(plan)


def test_cutover_recovers_after_result_commit_and_is_idempotent(
    tmp_path: Path,
) -> None:
    api = _prepared_api(tmp_path, FailingIndexApi)
    cutover = HubCatalogCutover(
        publisher_id="cutover-one", leases=FakeLeases(), api=api
    )

    with pytest.raises(RuntimeError, match="simulated index interruption"):
        cutover.apply(_plan())

    result = cutover.apply(_plan())
    commit_count = api.commit_count
    repeated = cutover.apply(_plan())

    assert result == repeated
    assert commit_count == 2
    assert api.commit_count == commit_count


def test_cutover_refuses_a_moved_dataset(tmp_path: Path) -> None:
    _tables, files = _publication()
    api = FakeApi(tmp_path, files)
    api.heads["org/results"] = "f" * 40
    api.snapshots[("org/results", "f" * 40)] = dict(files)

    with pytest.raises(CatalogCutoverError, match="moved"):
        HubCatalogCutover(
            publisher_id="cutover-one", leases=FakeLeases(), api=api
        ).apply(_plan())


def test_cutover_refuses_publication_added_at_expected_head(tmp_path: Path) -> None:
    api = _prepared_api(tmp_path, FakeApi)
    path = "data/catalog/schema=v1/windows/2048.parquet"
    original = read_catalog_file(api.snapshots[("org/index", INDEX_HEAD)][path])[0]
    added = original.model_copy(
        update={"publication_id": "publication-two", "run_id": "run-two"}
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pylist(
            [original.model_dump(mode="python"), added.model_dump(mode="python")]
        ),
        sink,
    )
    source_revision = "3" * 40
    api.snapshots[("org/index", source_revision)] = {
        path: api.snapshots[("org/index", INDEX_HEAD)][path]
    }
    api.snapshots[("org/index", INDEX_HEAD)][path] = sink.getvalue().to_pybytes()
    plan = _plan().model_copy(update={"source_catalog_revision": source_revision})

    with pytest.raises(CatalogCutoverError, match="classify both catalogs"):
        HubCatalogCutover(
            publisher_id="cutover-one", leases=FakeLeases(), api=api
        ).apply(plan)


def test_cutover_releases_first_lease_when_second_is_contended() -> None:
    leases = FailingSecondLease()

    with pytest.raises(RuntimeError, match="lease contention"):
        HubCatalogCutover(publisher_id="cutover-one", leases=leases).apply(_plan())

    assert not leases.held


def test_production_cutover_manifest_is_complete() -> None:
    plan = CatalogCutoverPlan.model_validate_json(
        Path("docs/catalog-cutovers/2026-07-17-primary-catalog.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(plan.classifications) == 17
    assert sum(item.role == "final" for item in plan.classifications) == 5
    assert sum(item.role == "component" for item in plan.classifications) == 10
    assert sum(item.role == "diagnostic" for item in plan.classifications) == 2
