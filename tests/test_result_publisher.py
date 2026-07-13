from __future__ import annotations

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
    HubDatasetPublisher,
    PublicationConflict,
    publisher_lease_path,
)
from harbor_hf.results import (
    ResultPublication,
    ResultTables,
    RunRow,
    build_result_publication,
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
            self.files[repo_id][operation.path_in_repo] = operation.path_or_fileobj
        self.generations[repo_id] += 1
        self.commits.append((repo_id, kwargs))
        return SimpleNamespace(oid=self.head(repo_id))


def _http_error(status: int) -> HfHubHTTPError:
    request = httpx.Request("POST", "https://huggingface.co/api/datasets")
    return HfHubHTTPError(
        "commit failed", response=httpx.Response(status, request=request)
    )


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
            "benchmark": "shellbench",
            "benchmark_revision": "sha256:" + "5" * 64,
            "result_kind": "ordinary",
            "outcome": "complete",
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
            "trial_count": 0,
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
    parquet_path = next(
        path for path in api.files[dataset] if path.endswith(".parquet")
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
