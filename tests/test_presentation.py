from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from harbor_hf.presentation.api import create_app
from harbor_hf.presentation.config import PresentationConfig
from harbor_hf.presentation.repository import (
    AnonymousHubReader,
    PresentationError,
    ResultRepository,
    ResultSnapshot,
)
from harbor_hf.presentation.service import ResultService
from harbor_hf.results import (
    ArtifactRow,
    ExecutionRow,
    GlobalIndexRow,
    MetricRow,
    ResultTables,
    RunRow,
    TrialRow,
    build_catalog_row,
)

NOW = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)
INDEX_REVISION = "a" * 40
RESULT_REVISION = "b" * 40
CONTROL_COMMIT = "c" * 40
DIGEST = "sha256:" + "d" * 64
LOCK_DIGEST = "sha256:" + "e" * 64
TASK_DIGEST = "sha256:" + "f" * 64


@dataclass
class RepoInfo:
    sha: str


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, bool]] = []

    def repo_info(
        self,
        repo_id: str,
        *,
        revision: str,
        repo_type: str,
        token: bool,
    ) -> RepoInfo:
        self.calls.append((repo_id, revision, repo_type, token))
        return RepoInfo(INDEX_REVISION)

    def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: str,
        repo_type: str,
        token: bool,
    ) -> list[str]:
        self.calls.append((repo_id, revision, repo_type, token))
        return ["data/index/schema=v1/windows/0008.parquet"]


class FakeReader:
    def __init__(self, snapshot: ResultSnapshot) -> None:
        self.snapshot = snapshot
        self.read_calls: list[str] = []
        self.rows: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        for size in (4, 8):
            catalog_path = f"data/catalog/schema=v1/windows/{size:04d}.parquet"
            self.rows[("org/index", INDEX_REVISION, catalog_path)] = [
                row.model_dump(mode="python") for row in snapshot.catalog_rows
            ]
            index_path = f"data/index/schema=v1/windows/{size:04d}.parquet"
            self.rows[("org/index", INDEX_REVISION, index_path)] = [
                row.model_dump(mode="python") for row in snapshot.index_rows
            ]
        for index in snapshot.index_rows:
            for table in ("runs", "trials", "executions", "metrics", "artifacts"):
                values = [
                    row
                    for row in getattr(snapshot, table)
                    if row.publication_id == index.publication_id
                ]
                path = (
                    f"data/{table}/schema=v1/campaign={index.campaign_id}/"
                    f"{index.publication_id}.parquet"
                )
                self.rows[("org/results", RESULT_REVISION, path)] = [
                    row.model_dump(mode="python") for row in values
                ]

    def resolve_revision(self, dataset: str, revision: str) -> str:
        assert (dataset, revision) == ("org/index", "main")
        return INDEX_REVISION

    def list_files(self, dataset: str, revision: str) -> list[str]:
        assert (dataset, revision) == ("org/index", INDEX_REVISION)
        return [
            "data/catalog/schema=v1/windows/0004.parquet",
            "data/catalog/schema=v1/windows/0008.parquet",
            "data/index/schema=v1/windows/0004.parquet",
            "data/index/schema=v1/windows/0008.parquet",
        ]

    def read_rows(
        self, dataset: str, revision: str, path: str
    ) -> Sequence[Mapping[str, object]]:
        self.read_calls.append(path)
        return self.rows[(dataset, revision, path)]


@pytest.fixture
def snapshot() -> ResultSnapshot:
    rows = [_publication(1, 1.0), _publication(2, 0.0)]
    catalogs = [
        build_catalog_row(
            ResultTables(
                publication_id=row[0].publication_id,
                runs=[row[1]],
                trials=[row[2]],
                executions=[row[3]],
                metrics=[row[4]],
                artifacts=[row[5]],
            ),
            result_dataset="org/results",
            result_revision=RESULT_REVISION,
        )
        for row in rows
    ]
    return ResultSnapshot(
        index_dataset="org/index",
        index_revision=INDEX_REVISION,
        catalog_rows=tuple(catalogs),
        index_rows=tuple(row[0] for row in rows),
        runs=tuple(row[1] for row in rows),
        trials=tuple(row[2] for row in rows),
        executions=tuple(row[3] for row in rows),
        metrics=tuple(row[4] for row in rows),
        artifacts=tuple(row[5] for row in rows),
    )


def test_config_is_public_and_bounded() -> None:
    config = PresentationConfig.from_env(
        {
            "HARBOR_HF_INDEX_DATASET": "org/index",
            "HARBOR_HF_INDEX_REVISION": "release-v1",
            "HARBOR_HF_MAX_PUBLICATIONS": "64",
            "HARBOR_HF_SPACE_TITLE": "Evaluation operations",
        }
    )
    assert config == PresentationConfig(
        "org/index", "release-v1", 64, "Evaluation operations"
    )
    with pytest.raises(ValueError, match="namespace/name"):
        PresentationConfig.from_env({})
    with pytest.raises(ValueError, match="between 1 and 4096"):
        PresentationConfig.from_env(
            {"HARBOR_HF_INDEX_DATASET": "org/index", "HARBOR_HF_MAX_PUBLICATIONS": "0"}
        )


def test_anonymous_reader_never_uses_ambient_token(tmp_path: Path) -> None:
    api = FakeApi()
    parquet = tmp_path / "rows.parquet"
    parquet.touch()
    downloads: list[dict[str, object]] = []

    def download(**values: object) -> str:
        downloads.append(values)
        return str(parquet)

    reader = AnonymousHubReader(
        api=api,
        download=download,
        parse_parquet=lambda path: [{"path": path.name}],
    )
    assert reader.resolve_revision("org/index", "main") == INDEX_REVISION
    assert reader.list_files("org/index", INDEX_REVISION)[0].endswith("0008.parquet")
    assert reader.read_rows("org/index", INDEX_REVISION, "rows.parquet") == [
        {"path": "rows.parquet"}
    ]
    assert all(call[-1] is False for call in api.calls)
    assert downloads[0]["token"] is False
    assert downloads[0]["revision"] == INDEX_REVISION


def test_repository_loads_exact_revisions_and_validates_trace(
    snapshot: ResultSnapshot,
) -> None:
    reader = FakeReader(snapshot)
    repository = ResultRepository(
        PresentationConfig("org/index", max_publications=2), reader
    )
    loaded = repository.load()
    assert len(loaded.catalog_rows) == 2
    assert not loaded.runs
    assert reader.read_calls[0].endswith("windows/0004.parquet")
    detail = repository.load_publication(loaded.catalog_rows[0])
    assert len(detail.runs) == 1
    rebuilt = repository.rebuild_catalog()
    assert {row.run_id: row.score for row in rebuilt} == {
        "run-1": 1.0,
        "run-2": 0.0,
    }
    assert all(
        revision in {INDEX_REVISION, RESULT_REVISION}
        for _dataset, revision, _path in reader.rows
    )

    run_path = "data/runs/schema=v1/campaign=campaign-1/publication-1.parquet"
    reader.rows[("org/results", RESULT_REVISION, run_path)][0]["source_checksum"] = (
        "sha256:" + "0" * 64
    )
    with pytest.raises(PresentationError, match="conflicts with its index row"):
        repository.load_publication(
            next(row for row in loaded.catalog_rows if row.run_id == "run-1")
        )


def test_api_exposes_comparison_details_and_denies_private_content(
    snapshot: ResultSnapshot, tmp_path: Path
) -> None:
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<h1>Harbor Results</h1>")
    client = TestClient(create_app(ResultService(snapshot), web_dir=web))

    health = client.get("/api/v1/health")
    assert health.json()["run_count"] == 2
    assert health.headers["etag"]
    assert (
        client.get(
            "/api/v1/health", headers={"If-None-Match": health.headers["etag"]}
        ).status_code
        == 304
    )
    runs = client.get("/api/v1/runs?benchmark=shellbench").json()
    assert runs["total"] == 2
    assert {item["run_id"]: item["score"] for item in runs["items"]} == {
        "run-1": 1.0,
        "run-2": 0.0,
    }
    assert runs["facets"]["models"] == ["org/model-1", "org/model-2"]
    first_page = client.get("/api/v1/runs?limit=1").json()
    assert len(first_page["items"]) == 1
    assert (
        client.get(f"/api/v1/runs?limit=1&cursor={first_page['next_cursor']}").json()[
            "items"
        ][0]["run_id"]
        != first_page["items"][0]["run_id"]
    )
    invalid_cursor = client.get("/api/v1/runs?cursor=invalid")
    assert invalid_cursor.status_code == 400
    assert invalid_cursor.json()["error"]["code"] == "request_rejected"
    assert client.get("/api/v1/campaigns").json()["total"] == 2
    assert client.get("/api/v1/campaigns/campaign-1").json()["runs"][0]["score"] == 1.0

    detail = client.get("/api/v1/runs/run-1").json()
    assert detail["trials"][0]["score"] == 1.0
    assert "source_bucket" not in detail["configuration"]
    assert detail["provenance"]["result_revision"] == RESULT_REVISION
    comparison = client.get("/api/v1/runs/run-1/compare/run-2").json()
    assert comparison["compatible"] is True
    assert comparison["score_delta"] == -1.0
    assert len(comparison["tasks"]) == 1
    assert (
        client.get("/api/v1/compare?run_id=run-1&run_id=run-2").json()["score_delta"]
        == -1.0
    )
    assert client.get("/api/v1/runs/run-1/trials").json()["total"] == 1
    assert client.get("/api/v1/runs/run-1/metrics").json()["total"] == 1

    assert client.get("/api/v1/trials/trial-1").json()["score"] == 1.0
    assert client.get("/api/v1/trials/trial-1/executions").json()["total"] == 1
    assert client.get("/api/v1/executions/execution-1").status_code == 200
    assert client.get("/api/v1/artifacts/artifact-1").status_code == 200
    assert client.get("/api/v1/artifacts/artifact-1/content").status_code == 403
    assert client.get("/api/v1/executions/execution-1/trajectory").status_code == 403
    missing = client.get("/api/v1/runs/missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    assert "Harbor Results" in client.get("/runs/run-1").text


def test_openapi_contract_matches_snapshot() -> None:
    expected = json.loads(Path("docs/api-v1.openapi.json").read_text())
    assert create_app().openapi() == expected


def _publication(
    number: int, score: float
) -> tuple[GlobalIndexRow, RunRow, TrialRow, ExecutionRow, MetricRow, ArtifactRow]:
    trace: dict[str, Any] = {
        "publication_id": f"publication-{number}",
        "run_id": f"run-{number}",
        "source_bucket": "hf://buckets/org/private-results",
        "source_prefix": f"campaigns/campaign-{number}/runs/run-{number}",
        "source_checksum": DIGEST,
        "run_lock_path": "run.lock.json",
        "run_lock_sha256": LOCK_DIGEST,
        "control_commit": CONTROL_COMMIT,
    }
    completed = NOW + timedelta(minutes=number)
    index = GlobalIndexRow(
        publication_id=trace["publication_id"],
        run_id=trace["run_id"],
        campaign_id=f"campaign-{number}",
        benchmark="shellbench",
        result_kind="ordinary",
        outcome="complete",
        completed_at=completed,
        model_repo=f"org/model-{number}",
        model_revision=str(number) * 40,
        agent_name="openclaw",
        agent_revision="2026.7.1",
        result_dataset="org/results",
        result_revision=RESULT_REVISION,
        source_checksum=DIGEST,
        control_commit=CONTROL_COMMIT,
    )
    run = RunRow(
        **trace,
        campaign_id=f"campaign-{number}",
        experiment="viewer-test",
        benchmark="shellbench",
        benchmark_revision=TASK_DIGEST,
        result_kind="ordinary",
        outcome="complete",
        created_at=NOW,
        completed_at=completed,
        model_id=f"model-{number}",
        model_repo=f"org/model-{number}",
        model_revision=str(number) * 40,
        deployment_id=f"deployment-{number}",
        provider="huggingface",
        region="us-east-1",
        hardware="h200",
        accelerator_count=1,
        agent_id="openclaw",
        agent_name="openclaw",
        agent_revision="2026.7.1",
        trial_count=1,
        execution_count=1,
    )
    trial = TrialRow(
        **trace,
        trial_id=f"trial-{number}",
        task_name="task-shared",
        task_digest=TASK_DIGEST,
        logical_attempt=1,
        selected_execution_id=f"execution-{number}",
        outcome="complete",
    )
    execution = ExecutionRow(
        **trace,
        execution_id=f"execution-{number}",
        trial_id=f"trial-{number}",
        physical_attempt=1,
        runtime_kind="endpoint",
        status="succeeded",
        started_at=NOW,
        completed_at=completed,
        retry_reason=None,
        remote_job_id=f"job-{number}",
    )
    metric = MetricRow(
        **trace,
        metric_id=f"metric-{number}",
        owner_type="trial",
        owner_id=f"trial-{number}",
        name="reward",
        value=score,
        unit="score",
        aggregation=None,
    )
    artifact = ArtifactRow(
        **trace,
        artifact_id=f"artifact-{number}",
        owner_type="run",
        owner_id=f"run-{number}",
        kind="run_lock",
        path="run.lock.json",
        sha256=LOCK_DIGEST,
        media_type="application/json",
        size_bytes=100,
    )
    return index, run, trial, execution, metric, artifact
