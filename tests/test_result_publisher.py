from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError

from harbor_hf.result_publisher import (
    DatasetPublicationError,
    HubDatasetPublisher,
    PublicationConflict,
    _regular_blob,
    catalog_decision_event_path,
    catalog_decision_latest_path,
    publisher_lease_path,
)
from harbor_hf.results import (
    CatalogDecision,
    PublicationProvenance,
    ResultPublication,
    ResultTables,
    RunRow,
    build_catalog_lookup_file,
    build_catalog_publication_lookup_file,
    build_global_index_row,
    build_index_file,
    build_result_publication,
    read_catalog_file,
    read_index_file,
)

NOW = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)
SECRET_SESSION = b"raw session must remain private"
SHELLBENCH_TASK = b"public ShellBench task instructions must not be published"


class FakeLeases:
    def __init__(self) -> None:
        self.held: set[str] = set()
        self.events: list[tuple[str, str, dict[str, str]]] = []

    def acquire(self, path: str, owner: dict[str, str]) -> None:
        assert path not in self.held
        self.held.add(path)
        self.events.append(("acquire", path, owner))

    def release(self, path: str, owner: dict[str, str]) -> None:
        assert path in self.held
        self.held.remove(path)
        self.events.append(("release", path, owner))


class FakeDatasetApi:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.generations = {
            "org/results": 1,
            "org/index": 1,
        }
        self.files: dict[str, dict[str, bytes]] = {
            "org/results": {},
            "org/index": {},
        }
        self.commits: list[tuple[str, dict[str, object]]] = []
        self.conflicts: dict[str, int] = {}
        self.fail_index_once = False

    def head(self, repo_id: str) -> str:
        return f"{self.generations[repo_id]:040x}"

    def repo_info(self, repo_id: str, **kwargs: object) -> object:
        assert kwargs == {"repo_type": "dataset", "revision": "main"}
        return SimpleNamespace(sha=self.head(repo_id))

    def get_paths_info(
        self, repo_id: str, paths: str | list[str], **kwargs: object
    ) -> list[object]:
        assert kwargs == {"repo_type": "dataset", "revision": self.head(repo_id)}
        path = paths if isinstance(paths, str) else paths[0]
        return [SimpleNamespace(path=path)] if path in self.files[repo_id] else []

    def list_repo_files(self, repo_id: str, **kwargs: object) -> list[str]:
        assert kwargs == {"repo_type": "dataset", "revision": self.head(repo_id)}
        return list(self.files[repo_id])

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: object) -> str:
        assert kwargs == {"repo_type": "dataset", "revision": self.head(repo_id)}
        destination = self.root / repo_id.replace("/", "-") / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[repo_id][filename])
        return str(destination)

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object:
        if self.fail_index_once and repo_id == "org/index":
            self.fail_index_once = False
            raise RuntimeError("interrupted before index commit")
        if self.conflicts.get(repo_id, 0):
            self.conflicts[repo_id] -= 1
            self.generations[repo_id] += 1
            raise _http_error(409)
        assert kwargs["parent_commit"] == self.head(repo_id)
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == "main"
        for operation in operations:
            assert isinstance(operation, CommitOperationAdd)
            assert isinstance(operation.path_or_fileobj, bytes)
            assert operation._upload_mode == "regular"
            self.files[repo_id][operation.path_in_repo] = operation.path_or_fileobj
        self.generations[repo_id] += 1
        self.commits.append((repo_id, kwargs))
        return SimpleNamespace(oid=self.head(repo_id))


def _http_error(status: int) -> HfHubHTTPError:
    request = httpx.Request("POST", "https://huggingface.co/api/datasets")
    return HfHubHTTPError(
        "commit failed", response=httpx.Response(status, request=request)
    )


def test_regular_blob_rejects_oversized_generated_files() -> None:
    with pytest.raises(DatasetPublicationError, match="regular blob limit"):
        _regular_blob("data/oversized.parquet", b"x" * (5 * 1024 * 1024 + 1))


@pytest.fixture
def publication() -> ResultPublication:
    trace = {
        "publication_id": "pub-" + "1" * 32,
        "run_id": "run-one",
        "source_bucket": "hf://buckets/private-evidence",
        "source_prefix": "campaigns/campaign-one/runs/run-one",
        "source_checksum": "sha256:" + "2" * 64,
        "run_lock_path": "run.lock.json",
        "run_lock_sha256": "sha256:" + "3" * 64,
        "control_commit": "4" * 40,
    }
    run = RunRow.model_validate(
        {
            **trace,
            "campaign_id": "campaign-one",
            "experiment": "experiment-one",
            "evaluation_id": "evaluation-one",
            "publication_role": "final",
            "component_kind": None,
            "source_publication_ids": [],
            "benchmark": "shellbench",
            "benchmark_revision": "sha256:" + "5" * 64,
            "result_kind": "ordinary",
            "outcome": "complete",
            "quality": "clean",
            "created_at": NOW,
            "completed_at": NOW + timedelta(minutes=1),
            "model_id": "model-one",
            "model_repo": "org/model",
            "model_revision": "6" * 40,
            "deployment_id": "deployment-one",
            "provider": "huggingface",
            "region": "aws-us-east-1",
            "hardware": "a100",
            "accelerator_count": 1,
            "agent_id": "agent-one",
            "agent_name": "example-agent",
            "agent_revision": "1.2.3",
            "planned_trial_count": 0,
            "scored_trial_count": 0,
            "agent_failed_count": 0,
            "benchmark_failed_count": 0,
            "infrastructure_exhausted_count": 0,
            "unsupported_count": 0,
            "execution_count": 0,
        }
    )
    tables = ResultTables(
        publication_id=trace["publication_id"],
        runs=[run],
        trials=[],
        executions=[],
        metrics=[],
        artifacts=[],
        provenance=PublicationProvenance(
            envelope_sha256="sha256:" + "7" * 64,
            projection_version="harbor-hf/results-projection/v1",
            sanitizer_version="harbor-hf/public-results/v1",
            execution_profile_sha256="sha256:" + "6" * 64,
            harbor_bundle_manifest_sha256s=["sha256:" + "8" * 64],
            harbor_archive_sha256s=["sha256:" + "9" * 64],
        ),
    )
    return build_result_publication(tables)


def test_serializes_result_and_index_with_parent_checked_leases(
    publication: ResultPublication, tmp_path: Path
) -> None:
    api = FakeDatasetApi(tmp_path)
    api.conflicts = {"org/results": 1, "org/index": 1}
    leases = FakeLeases()
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=leases, api=api
    )

    result = publisher.publish(
        publication,
        result_dataset="org/results",
        index_dataset="org/index",
    )

    assert result.result_revision == "0" * 39 + "3"
    assert result.index_revision == "0" * 39 + "3"
    assert [repo for repo, _kwargs in api.commits] == ["org/results", "org/index"]
    assert not leases.held
    assert [event[:2] for event in leases.events] == [
        ("acquire", publisher_lease_path("org/results")),
        ("release", publisher_lease_path("org/results")),
        ("acquire", publisher_lease_path("org/index")),
        ("release", publisher_lease_path("org/index")),
    ]
    committed = b"".join(
        content for files in api.files.values() for content in files.values()
    )
    assert SECRET_SESSION not in committed
    assert SHELLBENCH_TASK not in committed
    index_path = f"data/index/schema=v1/{publication.tables.publication_id}.parquet"
    index = pq.read_table(
        pa.BufferReader(api.files["org/index"][index_path])
    ).to_pylist()
    assert index[0]["result_revision"] == result.result_revision
    assert "task_name" not in index[0]
    windows = sorted(
        path
        for path in api.files["org/index"]
        if path.startswith("data/index/") and "/windows/" in path
    )
    assert windows == [
        f"data/index/schema=v1/windows/{2**power:04d}.parquet" for power in range(12)
    ]
    assert [
        row.publication_id
        for row in read_index_file(api.files["org/index"][windows[-1]])
    ] == [publication.tables.publication_id]
    catalog_path = "data/catalog/schema=v1/primary/windows/2048.parquet"
    catalog = read_catalog_file(api.files["org/index"][catalog_path])
    assert catalog[0].run_id == publication.tables.runs[0].run_id
    assert catalog[0].score == 0.0
    lookup = build_catalog_lookup_file(catalog[0])
    assert read_catalog_file(api.files["org/index"][lookup.path]) == catalog
    publication_lookup = build_catalog_publication_lookup_file(catalog[0])
    assert read_catalog_file(api.files["org/index"][publication_lookup.path]) == catalog
    assert catalog[0].projection_path.startswith("projections/schema=v1/")
    assert catalog[0].harbor_bundle_count == 1


def test_publishes_canonical_projection_and_catalog(
    publication: ResultPublication, tmp_path: Path
) -> None:
    api = FakeDatasetApi(tmp_path)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=FakeLeases(), api=api
    )

    publisher.publish(
        publication,
        result_dataset="org/results",
        index_dataset="org/index",
    )

    projection_path = f"projections/schema=v1/{publication.tables.publication_id}.json"
    projection = json.loads(api.files["org/results"][projection_path])
    assert projection["schema_version"] == "harbor-hf/result-projection/v1"
    assert set(projection["tables"]) == {
        "runs",
        "trials",
        "executions",
        "metrics",
        "artifacts",
    }
    catalog = read_catalog_file(
        api.files["org/index"]["data/catalog/schema=v1/primary/windows/2048.parquet"]
    )[0]
    assert catalog.projection_path == projection_path
    assert catalog.harbor_bundle_count == 1


@pytest.mark.parametrize(
    ("role", "component_kind"),
    [("component", "base"), ("diagnostic", None)],
)
def test_nonfinal_publications_only_enter_audit_catalog(
    publication: ResultPublication,
    tmp_path: Path,
    role: str,
    component_kind: str | None,
) -> None:
    run = publication.tables.runs[0].model_copy(
        update={"publication_role": role, "component_kind": component_kind}
    )
    candidate = build_result_publication(
        publication.tables.model_copy(update={"runs": [run]})
    )
    api = FakeDatasetApi(tmp_path)

    HubDatasetPublisher(
        publisher_id="publisher-one", leases=FakeLeases(), api=api
    ).publish(candidate, result_dataset="org/results", index_dataset="org/index")

    primary = read_catalog_file(
        api.files["org/index"]["data/catalog/schema=v1/primary/windows/2048.parquet"]
    )
    audit = read_catalog_file(
        api.files["org/index"]["data/catalog/schema=v1/audit/windows/2048.parquet"]
    )
    assert primary == []
    assert [row.publication_id for row in audit] == [candidate.tables.publication_id]


def test_catalog_decisions_withdraw_and_restore_final_publication(
    publication: ResultPublication, tmp_path: Path
) -> None:
    api = FakeDatasetApi(tmp_path)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=FakeLeases(), api=api
    )
    publisher.publish(
        publication, result_dataset="org/results", index_dataset="org/index"
    )
    publication_id = publication.tables.publication_id
    withdraw = CatalogDecision(
        decision_id="decision-withdraw",
        publication_id=publication_id,
        action="withdraw",
        actor="operator@example.com",
        reason="superseded evaluation",
        created_at=NOW + timedelta(minutes=2),
    )

    withdrawn = publisher.decide_catalog(withdraw, index_dataset="org/index")

    primary_path = "data/catalog/schema=v1/primary/windows/2048.parquet"
    audit_path = "data/catalog/schema=v1/audit/windows/2048.parquet"
    assert read_catalog_file(api.files["org/index"][primary_path]) == []
    assert len(read_catalog_file(api.files["org/index"][audit_path])) == 1
    assert withdrawn.index_revision == api.head("org/index")
    assert catalog_decision_event_path(withdraw.decision_id) in api.files["org/index"]
    assert catalog_decision_latest_path(publication_id) in api.files["org/index"]

    promote = withdraw.model_copy(
        update={
            "decision_id": "decision-promote",
            "action": "promote",
            "reason": "approved evaluation",
            "created_at": NOW + timedelta(minutes=3),
        }
    )
    publisher.decide_catalog(promote, index_dataset="org/index")

    assert [
        row.publication_id
        for row in read_catalog_file(api.files["org/index"][primary_path])
    ] == [publication_id]
    duplicate = publisher.decide_catalog(promote, index_dataset="org/index")
    assert duplicate.index_revision == api.head("org/index")


def test_catalog_decision_rejects_component_promotion(
    publication: ResultPublication, tmp_path: Path
) -> None:
    run = publication.tables.runs[0].model_copy(
        update={"publication_role": "component", "component_kind": "base"}
    )
    component = build_result_publication(
        publication.tables.model_copy(update={"runs": [run]})
    )
    api = FakeDatasetApi(tmp_path)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=FakeLeases(), api=api
    )
    publisher.publish(
        component, result_dataset="org/results", index_dataset="org/index"
    )
    decision = CatalogDecision(
        decision_id="decision-promote-component",
        publication_id=component.tables.publication_id,
        action="promote",
        actor="operator@example.com",
        reason="invalid request",
        created_at=NOW + timedelta(minutes=2),
    )

    with pytest.raises(DatasetPublicationError, match="only final"):
        publisher.decide_catalog(decision, index_dataset="org/index")


def test_duplicate_publication_is_a_no_op(
    publication: ResultPublication, tmp_path: Path
) -> None:
    api = FakeDatasetApi(tmp_path)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=FakeLeases(), api=api
    )

    first = publisher.publish(
        publication,
        result_dataset="org/results",
        index_dataset="org/index",
    )
    second = publisher.publish(
        publication,
        result_dataset="org/results",
        index_dataset="org/index",
    )

    assert second == first
    assert len(api.commits) == 2


def test_adoption_requires_canonical_catalog_when_history_is_missing(
    publication: ResultPublication, tmp_path: Path
) -> None:
    api = FakeDatasetApi(tmp_path)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=FakeLeases(), api=api
    )
    publisher.publish(
        publication,
        result_dataset="org/results",
        index_dataset="org/index",
    )
    for path in list(api.files["org/index"]):
        if "/windows/" in path:
            del api.files["org/index"][path]

    with pytest.raises(DatasetPublicationError, match="canonical catalog"):
        publisher.publish(
            publication,
            result_dataset="org/results",
            index_dataset="org/index",
        )

    assert len(api.commits) == 2
    assert not [path for path in api.files["org/index"] if "/windows/" in path]


def test_duplicate_publication_rejects_different_canonical_result_bytes(
    publication: ResultPublication, tmp_path: Path
) -> None:
    api = FakeDatasetApi(tmp_path)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=FakeLeases(), api=api
    )
    publisher.publish(
        publication,
        result_dataset="org/results",
        index_dataset="org/index",
    )
    moved_run = publication.tables.runs[0].model_copy(
        update={"control_commit": "7" * 40}
    )
    moved_publication = build_result_publication(
        publication.tables.model_copy(update={"runs": [moved_run]})
    )

    assert moved_publication.tables.publication_id == publication.tables.publication_id
    assert moved_publication.receipt != publication.receipt
    with pytest.raises(PublicationConflict, match="result publication receipt"):
        publisher.publish(
            moved_publication,
            result_dataset="org/results",
            index_dataset="org/index",
        )
    assert len(api.commits) == 2


def test_duplicate_publication_rejects_different_canonical_index_row(
    publication: ResultPublication, tmp_path: Path
) -> None:
    api = FakeDatasetApi(tmp_path)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=FakeLeases(), api=api
    )
    publisher.publish(
        publication,
        result_dataset="org/results",
        index_dataset="org/index",
    )
    receipt_path = f"publications/{publication.tables.publication_id}.json"
    receipt = json.loads(api.files["org/index"][receipt_path])
    index_path = receipt["index_path"]
    stale_run = publication.tables.runs[0].model_copy(
        update={"agent_revision": "stale"}
    )
    stale_row = build_global_index_row(
        publication.tables.model_copy(update={"runs": [stale_run]}),
        result_dataset="org/results",
        result_revision=receipt["result_revision"],
    )
    stale_file = build_index_file(stale_row)
    api.files["org/index"][index_path] = stale_file.content
    receipt["index_sha256"] = "sha256:" + hashlib.sha256(stale_file.content).hexdigest()
    api.files["org/index"][receipt_path] = json.dumps(receipt).encode()

    with pytest.raises(PublicationConflict, match="global index row"):
        publisher.publish(
            publication,
            result_dataset="org/results",
            index_dataset="org/index",
        )


def test_retry_after_interruption_adopts_result_and_finishes_index(
    publication: ResultPublication, tmp_path: Path
) -> None:
    api = FakeDatasetApi(tmp_path)
    api.fail_index_once = True
    leases = FakeLeases()
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=leases, api=api
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        publisher.publish(
            publication,
            result_dataset="org/results",
            index_dataset="org/index",
        )

    assert [repo for repo, _kwargs in api.commits] == ["org/results"]
    assert not leases.held
    result = publisher.publish(
        publication,
        result_dataset="org/results",
        index_dataset="org/index",
    )
    assert [repo for repo, _kwargs in api.commits] == ["org/results", "org/index"]
    assert result.result_revision == "0" * 39 + "2"


def test_conflicting_receipt_is_rejected(
    publication: ResultPublication, tmp_path: Path
) -> None:
    api = FakeDatasetApi(tmp_path)
    api.files["org/results"][publication.receipt_path] = json.dumps(
        {"publication_id": "different"}
    ).encode()
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=FakeLeases(), api=api
    )

    with pytest.raises(PublicationConflict, match="receipt conflicts"):
        publisher.publish(
            publication,
            result_dataset="org/results",
            index_dataset="org/index",
        )


@pytest.mark.parametrize("dataset", ["org/results", "org/index"])
def test_duplicate_publication_detects_corrupted_parquet(
    publication: ResultPublication, tmp_path: Path, dataset: str
) -> None:
    api = FakeDatasetApi(tmp_path)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=FakeLeases(), api=api
    )
    publisher.publish(
        publication,
        result_dataset="org/results",
        index_dataset="org/index",
    )
    parquet_path = (
        publication.files[0].path
        if dataset == "org/results"
        else f"data/index/schema=v1/{publication.tables.publication_id}.parquet"
    )
    api.files[dataset][parquet_path] = b"corrupted"

    message = "published result file" if dataset == "org/results" else "global index"
    with pytest.raises(RuntimeError, match=message):
        publisher.publish(
            publication,
            result_dataset="org/results",
            index_dataset="org/index",
        )


def test_rejects_invalid_publisher_and_same_destination(
    publication: ResultPublication, tmp_path: Path
) -> None:
    api = FakeDatasetApi(tmp_path)
    with pytest.raises(ValueError, match="publisher ID"):
        HubDatasetPublisher(publisher_id="", leases=FakeLeases(), api=api)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one", leases=FakeLeases(), api=api
    )
    with pytest.raises(ValueError, match="must be distinct"):
        publisher.publish(
            publication,
            result_dataset="org/results",
            index_dataset="org/results",
        )


def test_non_parent_commit_error_is_not_retried(
    publication: ResultPublication, tmp_path: Path
) -> None:
    class FailingApi(FakeDatasetApi):
        def create_commit(
            self, repo_id: str, operations: list[object], **kwargs: object
        ) -> object:
            raise _http_error(500)

    publisher = HubDatasetPublisher(
        publisher_id="publisher-one",
        leases=FakeLeases(),
        api=FailingApi(tmp_path),
    )
    with pytest.raises(HfHubHTTPError):
        publisher.publish(
            publication,
            result_dataset="org/results",
            index_dataset="org/index",
        )
