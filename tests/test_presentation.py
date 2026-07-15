from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import harbor_hf.presentation.api as presentation_api
from harbor_hf.presentation.api import ServiceHolder, create_app
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
    build_catalog_lookup_file,
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
        for row in snapshot.catalog_rows:
            lookup = build_catalog_lookup_file(row)
            self.rows[("org/index", INDEX_REVISION, lookup.path)] = [
                row.model_dump(mode="python")
            ]
        for index in snapshot.index_rows:
            index_path = f"data/index/schema=v1/{index.publication_id}.parquet"
            self.rows[("org/index", INDEX_REVISION, index_path)] = [
                index.model_dump(mode="python")
            ]
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
            *[
                f"data/index/schema=v1/{row.publication_id}.parquet"
                for row in self.snapshot.index_rows
            ],
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
    with pytest.raises(ValueError, match="between 1 and 2048"):
        PresentationConfig.from_env(
            {"HARBOR_HF_INDEX_DATASET": "org/index", "HARBOR_HF_MAX_PUBLICATIONS": "0"}
        )
    with pytest.raises(ValueError, match="between 5 and 3600"):
        PresentationConfig.from_env(
            {"HARBOR_HF_INDEX_DATASET": "org/index", "HARBOR_HF_REFRESH_SECONDS": "1"}
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


def test_repository_rejects_malformed_rows_and_broken_relations(
    snapshot: ResultSnapshot,
) -> None:
    reader = FakeReader(snapshot)
    repository = ResultRepository(
        PresentationConfig("org/index", max_publications=2), reader
    )
    catalog = next(
        row for row in repository.load().catalog_rows if row.run_id == "run-1"
    )
    metric_path = "data/metrics/schema=v1/campaign=campaign-1/publication-1.parquet"
    metric = reader.rows[("org/results", RESULT_REVISION, metric_path)][0]
    metric["value"] = float("nan")
    with pytest.raises(PresentationError, match="invalid normalized row"):
        repository.load_publication(catalog)

    metric["value"] = 1.0
    execution_path = (
        "data/executions/schema=v1/campaign=campaign-1/publication-1.parquet"
    )
    execution = reader.rows[("org/results", RESULT_REVISION, execution_path)][0]
    execution["trial_id"] = "unknown-trial"
    with pytest.raises(PresentationError, match="unknown trial"):
        repository.load_publication(catalog)

    execution["trial_id"] = "trial-1"
    execution["status"] = "failed_infrastructure"
    with pytest.raises(PresentationError, match="selected execution"):
        repository.load_publication(catalog)


def test_service_refreshes_mutable_revision_after_ttl(
    snapshot: ResultSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    newer = replace(snapshot, index_revision="9" * 40)
    snapshots = iter((snapshot, newer))

    class RefreshingRepository:
        def __init__(self, _config: PresentationConfig) -> None:
            pass

        def load(self) -> ResultSnapshot:
            return next(snapshots)

    times = iter((10.0, 10.0, 12.0, 16.0, 16.0))
    monkeypatch.setattr(presentation_api, "ResultRepository", RefreshingRepository)
    monkeypatch.setattr(presentation_api, "monotonic", lambda: next(times))
    holder = ServiceHolder(config=PresentationConfig("org/index", refresh_seconds=5))

    first = holder.get()
    assert holder.get() is first
    assert holder.get().snapshot.index_revision == "9" * 40


def test_service_keeps_cached_snapshot_when_refresh_fails(
    snapshot: ResultSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[ResultSnapshot | PresentationError] = [
        snapshot,
        PresentationError("temporary Hub failure"),
    ]

    class FailingRefreshRepository:
        def __init__(self, _config: PresentationConfig) -> None:
            pass

        def load(self) -> ResultSnapshot:
            value = loads.pop(0)
            if isinstance(value, PresentationError):
                raise value
            return value

    times = iter((10.0, 10.0, 16.0, 16.0))
    monkeypatch.setattr(presentation_api, "ResultRepository", FailingRefreshRepository)
    monkeypatch.setattr(presentation_api, "monotonic", lambda: next(times))
    holder = ServiceHolder(config=PresentationConfig("org/index", refresh_seconds=5))

    first = holder.get()
    assert holder.get() is first


def test_historical_links_resolve_outside_list_window(
    snapshot: ResultSnapshot,
) -> None:
    reader = FakeReader(snapshot)
    repository = ResultRepository(
        PresentationConfig("org/index", max_publications=1), reader
    )
    service = ResultService(repository.load(), repository=repository)
    client = TestClient(create_app(service))

    assert [item["run_id"] for item in client.get("/api/v1/runs").json()["items"]] == [
        "run-2"
    ]
    assert client.get("/api/v1/runs/run-1").status_code == 200
    assert client.get("/api/v1/runs/run-1/trials/trial-1").status_code == 200
    assert client.get("/api/v1/runs/run-1/executions/execution-1").status_code == 200
    assert client.get("/api/v1/runs/run-1/trials/missing").status_code == 404
    assert len([path for path in reader.read_calls if "/windows/" not in path]) == 6


def test_catalog_uses_valid_nonstandard_reward_names() -> None:
    index, run, trial, execution, metric, artifact = _publication(1, 1.0)
    del index
    renamed = metric.model_copy(update={"name": "task_success", "value": 0.75})
    tables = ResultTables(
        publication_id=run.publication_id,
        runs=[run],
        trials=[trial],
        executions=[execution],
        metrics=[renamed],
        artifacts=[artifact],
    )

    catalog = build_catalog_row(
        tables,
        result_dataset="org/results",
        result_revision=RESULT_REVISION,
    )

    assert catalog.score == 0.75
    assert catalog.passed_trials == 0


def test_catalog_prefers_primary_reward_over_secondary_metrics() -> None:
    index, run, trial, execution, metric, artifact = _publication(1, 1.0)
    del index
    speed = metric.model_copy(
        update={"metric_id": "metric-speed", "name": "speed", "value": 0.2}
    )
    tables = ResultTables(
        publication_id=run.publication_id,
        runs=[run],
        trials=[trial],
        executions=[execution],
        metrics=[speed, metric],
        artifacts=[artifact],
    )

    catalog = build_catalog_row(
        tables,
        result_dataset="org/results",
        result_revision=RESULT_REVISION,
    )

    assert catalog.score == 1.0
    assert catalog.passed_trials == 1


def test_detail_and_comparison_use_nonstandard_reward_names(
    snapshot: ResultSnapshot,
) -> None:
    metrics = tuple(
        metric.model_copy(update={"name": "task_success", "value": 1.25})
        if metric.run_id == "run-1"
        else metric
        for metric in snapshot.metrics
    )
    service = ResultService(replace(snapshot, catalog_rows=(), metrics=metrics))

    detail = service.run("run-1")
    comparison = service.compare("run-1", "run-2")

    assert detail["summary"]["score"] == 1.25
    assert detail["summary"]["passed_trials"] == 1
    assert detail["trials"][0]["score"] == 1.25
    assert comparison["tasks"][0]["left_score"] == 1.25


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

    assert client.get("/api/v1/runs/run-1/trials/trial-1").json()["score"] == 1.0
    assert (
        client.get("/api/v1/runs/run-1/trials/trial-1/executions").json()["total"] == 1
    )
    assert client.get("/api/v1/runs/run-1/executions/execution-1").status_code == 200
    assert client.get("/api/v1/runs/run-1/artifacts/artifact-1").status_code == 200
    assert (
        client.get("/api/v1/runs/run-1/artifacts/artifact-1/content").status_code == 403
    )
    assert (
        client.get("/api/v1/runs/run-1/executions/execution-1/trajectory").status_code
        == 403
    )
    missing = client.get("/api/v1/runs/missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    unknown_api = client.get("/api/unknown")
    assert unknown_api.status_code == 404
    assert unknown_api.json()["error"]["code"] == "request_rejected"
    assert "Harbor Results" in client.get("/runs/run-1").text


def test_etag_changes_with_response_representation(snapshot: ResultSnapshot) -> None:
    first = TestClient(create_app(ResultService(snapshot, title="First"))).get(
        "/api/v1/health"
    )
    second = TestClient(create_app(ResultService(snapshot, title="Second"))).get(
        "/api/v1/health"
    )

    assert first.json()["title"] == "First"
    assert second.json()["title"] == "Second"
    assert first.headers["etag"] != second.headers["etag"]


def test_comparison_keeps_logical_attempts_and_checks_benchmark_revision(
    snapshot: ResultSnapshot,
) -> None:
    trials = list(snapshot.trials)
    executions = list(snapshot.executions)
    metrics = list(snapshot.metrics)
    for number in (1, 2):
        original_trial = next(row for row in trials if row.run_id == f"run-{number}")
        original_execution = next(
            row for row in executions if row.run_id == f"run-{number}"
        )
        original_metric = next(row for row in metrics if row.run_id == f"run-{number}")
        trials.append(
            original_trial.model_copy(
                update={
                    "trial_id": f"trial-{number}-attempt-2",
                    "logical_attempt": 2,
                    "selected_execution_id": f"execution-{number}-attempt-2",
                }
            )
        )
        executions.append(
            original_execution.model_copy(
                update={
                    "execution_id": f"execution-{number}-attempt-2",
                    "trial_id": f"trial-{number}-attempt-2",
                }
            )
        )
        metrics.append(
            original_metric.model_copy(
                update={
                    "metric_id": f"metric-{number}-attempt-2",
                    "owner_id": f"trial-{number}-attempt-2",
                }
            )
        )
    runs = tuple(
        row.model_copy(
            update={
                "trial_count": 2,
                "execution_count": 2,
                **(
                    {"benchmark_revision": "sha256:" + "0" * 64}
                    if row.run_id == "run-2"
                    else {}
                ),
            }
        )
        for row in snapshot.runs
    )
    expanded = ResultSnapshot(
        index_dataset=snapshot.index_dataset,
        index_revision=snapshot.index_revision,
        catalog_rows=(),
        index_rows=snapshot.index_rows,
        runs=runs,
        trials=tuple(trials),
        executions=tuple(executions),
        metrics=tuple(metrics),
        artifacts=snapshot.artifacts,
    )

    comparison = ResultService(expanded).compare("run-1", "run-2")
    assert comparison["compatible"] is False
    assert [item["logical_attempt"] for item in comparison["tasks"]] == [1, 2]


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
