from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from harbor_hf_space.config import SpaceConfig
from harbor_hf_space.data import (
    AnonymousHubReader,
    DatasetLoader,
    PresentationError,
    Snapshot,
)
from harbor_hf_space.ui import _refresh_views, create_app
from harbor_hf_space.views import ViewFilters, build_views

_INDEX_REVISION = "a" * 40
_RESULT_REVISION = "b" * 40
_CONTROL_COMMIT = "c" * 40
_SOURCE_CHECKSUM = f"sha256:{'d' * 64}"
_LOCK_CHECKSUM = f"sha256:{'e' * 64}"
_TASK_DIGEST = f"sha256:{'f' * 64}"
_ARTIFACT_CHECKSUM = f"sha256:{'1' * 64}"


class FakeReader:
    def __init__(self, publications: list[tuple[str, str]]) -> None:
        self.publications = publications
        self.rows: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        index_rows: list[dict[str, object]] = []
        for offset, (kind, outcome) in enumerate(publications, start=1):
            publication = f"publication-{offset}"
            campaign = f"campaign-{offset}"
            run = f"run-{offset}"
            index_rows.append(
                _index_row(publication, campaign, run, kind, outcome, offset)
            )
            for table, rows in _result_rows(
                publication, campaign, run, kind, outcome, offset
            ).items():
                path = (
                    f"data/{table}/schema=v1/campaign={campaign}/{publication}.parquet"
                )
                self.rows[("org/results", _RESULT_REVISION, path)] = rows
        self.rows[
            ("org/index", _INDEX_REVISION, "data/index/schema=v1/all.parquet")
        ] = index_rows

    def resolve_revision(self, dataset: str, revision: str) -> str:
        assert dataset == "org/index"
        assert revision == "main"
        return _INDEX_REVISION

    def list_files(self, dataset: str, revision: str) -> list[str]:
        assert (dataset, revision) == ("org/index", _INDEX_REVISION)
        return ["README.md", "data/index/schema=v1/all.parquet"]

    def read_rows(
        self, dataset: str, revision: str, path: str
    ) -> Sequence[Mapping[str, object]]:
        return self.rows[(dataset, revision, path)]


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
        return RepoInfo(sha=_INDEX_REVISION)

    def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: str,
        repo_type: str,
        token: bool,
    ) -> list[str]:
        self.calls.append((repo_id, revision, repo_type, token))
        return ["data/index/schema=v1/one.parquet"]


@dataclass
class StaticLoader:
    snapshot: Snapshot

    def load(self) -> Snapshot:
        return self.snapshot


class FailingLoader:
    def load(self) -> Snapshot:
        raise PresentationError("fixture is unavailable")


def test_space_config_accepts_only_bounded_public_configuration() -> None:
    config = SpaceConfig.from_env(
        {
            "HARBOR_HF_INDEX_DATASET": "org/index",
            "HARBOR_HF_INDEX_REVISION": "release-v1",
            "HARBOR_HF_MAX_PUBLICATIONS": "20",
            "HARBOR_HF_SPACE_TITLE": "Evaluation operations",
        }
    )

    assert config == SpaceConfig(
        index_dataset="org/index",
        index_revision="release-v1",
        max_publications=20,
        title="Evaluation operations",
    )
    with pytest.raises(ValueError, match="namespace/name"):
        SpaceConfig.from_env({})
    with pytest.raises(ValueError, match="between 1 and 2000"):
        SpaceConfig.from_env(
            {
                "HARBOR_HF_INDEX_DATASET": "org/index",
                "HARBOR_HF_MAX_PUBLICATIONS": "0",
            }
        )


def test_hub_reader_forces_anonymous_exact_revision_reads(tmp_path: Path) -> None:
    api = FakeApi()
    download_calls: list[dict[str, object]] = []
    parquet = tmp_path / "row.parquet"
    parquet.touch()

    def download(**values: object) -> str:
        download_calls.append(values)
        return str(parquet)

    reader = AnonymousHubReader(
        api=api,
        download=download,
        parse_parquet=lambda path: [{"path": path.name}],
    )

    assert reader.resolve_revision("org/index", "main") == _INDEX_REVISION
    assert reader.list_files("org/index", _INDEX_REVISION) == [
        "data/index/schema=v1/one.parquet"
    ]
    assert reader.read_rows("org/index", _INDEX_REVISION, "data.parquet") == [
        {"path": "row.parquet"}
    ]
    assert api.calls == [
        ("org/index", "main", "dataset", False),
        ("org/index", _INDEX_REVISION, "dataset", False),
    ]
    assert download_calls == [
        {
            "repo_id": "org/index",
            "filename": "data.parquet",
            "repo_type": "dataset",
            "revision": _INDEX_REVISION,
            "token": False,
        }
    ]


def test_loader_builds_all_operational_views_with_explicit_labels() -> None:
    reader = FakeReader(
        [("ordinary", "complete"), ("composite", "partial"), ("manual", "complete")]
    )
    snapshot = DatasetLoader(SpaceConfig(index_dataset="org/index"), reader).load()
    views = build_views(snapshot)

    assert {row["Result"] for row in views.runs} == {
        "COMPLETE · ORDINARY",
        "PARTIAL · COMPOSITE",
        "COMPLETE · MANUAL",
    }
    assert len(views.campaigns) == 3
    assert len(views.tasks) == 3
    assert len(views.attempts) == 6
    assert len(views.errors) == 6
    assert len(views.throughput) == 3
    assert len(views.hardware) == 3
    assert len(views.cost) == 3
    assert len(views.provenance) == 3
    assert views.tasks[0]["Task digest"] == _TASK_DIGEST
    assert views.provenance[0]["Index revision"] == _INDEX_REVISION
    assert views.provenance[0]["Result revision"] == _RESULT_REVISION

    partial = build_views(snapshot, ViewFilters(result="partial"))
    assert [row["Result"] for row in partial.runs] == ["PARTIAL · COMPOSITE"]
    searched = build_views(snapshot, ViewFilters(search="model-3"))
    assert [row["Run"] for row in searched.runs] == ["run-3"]


def test_loader_fails_closed_on_conflicting_provenance() -> None:
    reader = FakeReader([("ordinary", "complete")])
    path = "data/runs/schema=v1/campaign=campaign-1/publication-1.parquet"
    reader.rows[("org/results", _RESULT_REVISION, path)][0]["source_checksum"] = (
        f"sha256:{'0' * 64}"
    )

    with pytest.raises(PresentationError, match="conflicts with its index row"):
        DatasetLoader(SpaceConfig(index_dataset="org/index"), reader).load()


def test_ui_builds_without_network_and_refreshes_injected_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARBOR_HF_INDEX_DATASET", raising=False)
    assert type(create_app()).__name__ == "Blocks"

    snapshot = DatasetLoader(
        SpaceConfig(index_dataset="org/index"),
        FakeReader([("ordinary", "complete")]),
    ).load()
    app = create_app(
        config=SpaceConfig(index_dataset="org/index"),
        loader=StaticLoader(snapshot),
    )
    rendered = _refresh_views(StaticLoader(snapshot), "all", "", "", "")
    status = rendered[0]
    runs = rendered[2]

    assert type(app).__name__ == "Blocks"
    assert isinstance(status, str)
    assert "**1 runs shown**" in status
    assert isinstance(runs, list)
    first_run = runs[0]
    assert isinstance(first_run, list)
    assert first_run[0] == "COMPLETE · ORDINARY"


def test_ui_refresh_fails_closed() -> None:
    rendered = _refresh_views(FailingLoader(), "all", "", "", "")
    status = rendered[0]

    assert isinstance(status, str)
    assert "Dataset read failed" in status
    assert all(table == [] for table in rendered[1:])


def _index_row(
    publication: str,
    campaign: str,
    run: str,
    kind: str,
    outcome: str,
    offset: int,
) -> dict[str, object]:
    return {
        "schema_version": "harbor-hf/results/index/v1",
        "publication_id": publication,
        "run_id": run,
        "campaign_id": campaign,
        "benchmark": "shellbench",
        "result_kind": kind,
        "outcome": outcome,
        "completed_at": f"2026-07-{offset:02d}T00:00:00Z",
        "model_repo": f"org/model-{offset}",
        "model_revision": str(offset) * 40,
        "agent_name": "terminus",
        "agent_revision": "1.0.0",
        "result_dataset": "org/results",
        "result_revision": _RESULT_REVISION,
        "source_checksum": _SOURCE_CHECKSUM,
        "control_commit": _CONTROL_COMMIT,
    }


def _result_rows(
    publication: str,
    campaign: str,
    run: str,
    kind: str,
    outcome: str,
    offset: int,
) -> dict[str, list[dict[str, object]]]:
    trace: dict[str, object] = {
        "publication_id": publication,
        "run_id": run,
        "source_bucket": "org/evidence",
        "source_prefix": f"campaigns/{campaign}/runs/{run}",
        "source_checksum": _SOURCE_CHECKSUM,
        "run_lock_path": "run.lock.json",
        "run_lock_sha256": _LOCK_CHECKSUM,
        "control_commit": _CONTROL_COMMIT,
    }
    trial_id = f"trial-{offset}"
    failed_execution = f"execution-{offset}-1"
    selected_execution = f"execution-{offset}-2"
    return {
        "runs": [
            {
                **trace,
                "schema_version": "harbor-hf/results/runs/v1",
                "campaign_id": campaign,
                "experiment": "hosted-evaluation",
                "benchmark": "shellbench",
                "benchmark_revision": "2" * 40,
                "result_kind": kind,
                "outcome": outcome,
                "created_at": f"2026-06-{offset:02d}T00:00:00Z",
                "completed_at": f"2026-07-{offset:02d}T00:00:00Z",
                "model_id": f"model-{offset}",
                "model_repo": f"org/model-{offset}",
                "model_revision": str(offset) * 40,
                "deployment_id": f"deployment-{offset}",
                "provider": "aws",
                "region": "us-east-1",
                "hardware": "nvidia-a100",
                "accelerator_count": 1,
                "agent_id": "terminus-1",
                "agent_name": "terminus",
                "agent_revision": "1.0.0",
                "trial_count": 1,
                "execution_count": 2,
            }
        ],
        "trials": [
            {
                **trace,
                "schema_version": "harbor-hf/results/trials/v1",
                "trial_id": trial_id,
                "task_name": f"task-{offset}",
                "task_digest": _TASK_DIGEST,
                "logical_attempt": 1,
                "selected_execution_id": selected_execution,
                "outcome": "complete",
            }
        ],
        "executions": [
            {
                **trace,
                "schema_version": "harbor-hf/results/executions/v1",
                "execution_id": failed_execution,
                "trial_id": trial_id,
                "physical_attempt": 1,
                "runtime_kind": "endpoint",
                "status": "failed_infrastructure",
                "started_at": "2026-07-01T00:00:00Z",
                "completed_at": "2026-07-01T00:01:00Z",
                "retry_reason": "remote job lost",
                "remote_job_id": "job-failed",
            },
            {
                **trace,
                "schema_version": "harbor-hf/results/executions/v1",
                "execution_id": selected_execution,
                "trial_id": trial_id,
                "physical_attempt": 2,
                "runtime_kind": "endpoint",
                "status": "succeeded",
                "started_at": "2026-07-01T00:02:00Z",
                "completed_at": "2026-07-01T00:03:00Z",
                "retry_reason": "remote job lost",
                "remote_job_id": "job-selected",
            },
        ],
        "metrics": [
            {
                **trace,
                "schema_version": "harbor-hf/results/metrics/v1",
                "metric_id": f"metric-{offset}-score",
                "owner_type": "trial",
                "owner_id": trial_id,
                "name": "verifier_reward",
                "value": 1.0,
                "unit": "reward",
                "aggregation": None,
            },
            {
                **trace,
                "schema_version": "harbor-hf/results/metrics/v1",
                "metric_id": f"metric-{offset}-throughput",
                "owner_type": "run",
                "owner_id": run,
                "name": "aggregate_throughput",
                "value": 12.5,
                "unit": "tokens/s",
                "aggregation": "mean",
            },
            {
                **trace,
                "schema_version": "harbor-hf/results/metrics/v1",
                "metric_id": f"metric-{offset}-cost",
                "owner_type": "run",
                "owner_id": run,
                "name": "estimated_cost",
                "value": 2.5,
                "unit": "USD",
                "aggregation": "sum",
            },
        ],
        "artifacts": [
            {
                **trace,
                "schema_version": "harbor-hf/results/artifacts/v1",
                "artifact_id": f"artifact-{offset}",
                "owner_type": "run",
                "owner_id": run,
                "kind": "verification",
                "path": "checksums.json",
                "sha256": _ARTIFACT_CHECKSUM,
                "media_type": "application/json",
                "size_bytes": 100,
            }
        ],
    }
