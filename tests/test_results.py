from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from harbor_hf.results import (
    ArtifactRow,
    EvidenceSource,
    ExecutionRow,
    GlobalIndexRow,
    MetricRow,
    RebuildRequest,
    ResultEvidence,
    ResultPublicationError,
    RunRow,
    TableName,
    TraceRow,
    TrialRow,
    audit_result_tables,
    build_global_index_row,
    build_result_publication,
    build_result_tables,
    index_parquet_schema,
    parquet_schema,
    rebuild_result_tables,
    result_schema_manifest,
)

NOW = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)
CONTROL_COMMIT = "c" * 40
SECRET_SESSION = b"raw session must remain private"
SHELLBENCH_TASK = b"public ShellBench task instructions must not be published"


class MemoryEvidence:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def list_files(self, *, bucket: str, prefix: str) -> list[str]:
        assert bucket == "hf://buckets/private-evidence"
        assert prefix == "campaigns/campaign-one/runs/run-one"
        return list(reversed(self.files))

    def read_bytes(self, *, bucket: str, prefix: str, path: str) -> bytes:
        assert bucket == "hf://buckets/private-evidence"
        assert prefix == "campaigns/campaign-one/runs/run-one"
        return self.files[path]


@pytest.fixture
def source() -> EvidenceSource:
    return EvidenceSource(
        bucket="hf://buckets/private-evidence",
        prefix="campaigns/campaign-one/runs/run-one",
    )


@pytest.fixture
def summary_value() -> dict[str, object]:
    return sample_summary()


def sample_summary() -> dict[str, object]:
    return {
        "schema_version": "harbor-hf/result-evidence/v1",
        "sanitized": True,
        "run": {
            "run_id": "run-one",
            "campaign_id": "campaign-one",
            "experiment": "experiment-one",
            "benchmark": "shellbench",
            "benchmark_revision": "sha256:" + "b" * 64,
            "result_kind": "ordinary",
            "outcome": "complete",
            "created_at": NOW.isoformat(),
            "completed_at": (NOW + timedelta(minutes=5)).isoformat(),
            "model_id": "model-one",
            "model_repo": "org/model",
            "model_revision": "a" * 40,
            "deployment_id": "deployment-one",
            "provider": "huggingface",
            "region": "aws-us-east-1",
            "hardware": "a100",
            "accelerator_count": 1,
            "agent_id": "agent-one",
            "agent_name": "example-agent",
            "agent_revision": "1.2.3",
        },
        "trials": [
            {
                "trial_id": "trial-two",
                "task_name": "task-two",
                "task_digest": "sha256:" + "2" * 64,
                "logical_attempt": 1,
                "selected_execution_id": "execution-three",
                "outcome": "complete",
            },
            {
                "trial_id": "trial-one",
                "task_name": "task-one",
                "task_digest": "sha256:" + "1" * 64,
                "logical_attempt": 1,
                "selected_execution_id": "execution-two",
                "outcome": "complete",
            },
        ],
        "executions": [
            {
                "execution_id": "execution-three",
                "trial_id": "trial-two",
                "physical_attempt": 1,
                "runtime_kind": "provider",
                "status": "succeeded",
                "started_at": (NOW + timedelta(minutes=2)).isoformat(),
                "completed_at": (NOW + timedelta(minutes=3)).isoformat(),
                "retry_reason": None,
                "remote_job_id": None,
            },
            {
                "execution_id": "execution-one",
                "trial_id": "trial-one",
                "physical_attempt": 1,
                "runtime_kind": "endpoint",
                "status": "failed_infrastructure",
                "started_at": NOW.isoformat(),
                "completed_at": (NOW + timedelta(minutes=1)).isoformat(),
                "retry_reason": None,
                "remote_job_id": "job-one",
            },
            {
                "execution_id": "execution-two",
                "trial_id": "trial-one",
                "physical_attempt": 2,
                "runtime_kind": "endpoint",
                "status": "succeeded",
                "started_at": (NOW + timedelta(minutes=1)).isoformat(),
                "completed_at": (NOW + timedelta(minutes=2)).isoformat(),
                "retry_reason": "provider_timeout",
                "remote_job_id": "job-two",
            },
        ],
        "metrics": [
            {
                "owner_type": "trial",
                "owner_id": "trial-two",
                "name": "reward",
                "value": 0.0,
                "unit": "score",
                "aggregation": None,
            },
            {
                "owner_type": "execution",
                "owner_id": "execution-two",
                "name": "request_latency",
                "value": 1.25,
                "unit": "seconds",
                "aggregation": "mean",
            },
        ],
        "artifacts": [
            {
                "owner_type": "run",
                "owner_id": "run-one",
                "kind": "verification",
                "path": "verification.json",
                "sha256": "sha256:" + "9" * 64,
                "media_type": "application/json",
                "size_bytes": 42,
            }
        ],
    }


@pytest.fixture
def evidence(summary_value: dict[str, object]) -> MemoryEvidence:
    return _evidence(summary_value)


def _evidence(summary: dict[str, object], marker: str = "_SUCCESS") -> MemoryEvidence:
    normalized = json.loads(json.dumps(summary))
    verification = _json_bytes({"trial_count": 2})
    normalized["artifacts"][0]["sha256"] = _sha256(verification)
    normalized["artifacts"][0]["size_bytes"] = len(verification)
    files = {
        "run.lock.json": _json_bytes({"run_id": "run-one", "cell_digest": "x"}),
        "run-summary.json": _json_bytes(normalized),
        "verification.json": verification,
        "trials/trial-one/executions/execution-two/session.json": SECRET_SESSION,
        "trials/trial-one/task-source/instruction.md": SHELLBENCH_TASK,
    }
    checksums = {path: _sha256(value) for path, value in files.items()}
    files["checksums.json"] = _json_bytes(checksums)
    files[marker] = b"\n"
    return MemoryEvidence(files)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _refresh_checksums(evidence: MemoryEvidence) -> None:
    checksums = {
        path: _sha256(value)
        for path, value in evidence.files.items()
        if path not in {"checksums.json", "_SUCCESS", "_PARTIAL"}
    }
    evidence.files["checksums.json"] = _json_bytes(checksums)


def test_builds_deterministic_traceable_rows_and_parquet(
    evidence: MemoryEvidence, source: EvidenceSource
) -> None:
    first = build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)
    second = build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)
    first_publication = build_result_publication(first)
    second_publication = build_result_publication(second)

    assert first == second
    assert first_publication == second_publication
    assert [row.trial_id for row in first.trials] == ["trial-one", "trial-two"]
    assert [row.execution_id for row in first.executions] == [
        "execution-one",
        "execution-three",
        "execution-two",
    ]
    assert first.metrics[0].value == 1.25
    assert first.metrics[1].value == 0.0
    trace = {
        (
            row.publication_id,
            row.source_checksum,
            row.run_lock_sha256,
            row.control_commit,
        )
        for rows in (
            first.runs,
            first.trials,
            first.executions,
            first.metrics,
            first.artifacts,
        )
        for row in rows
    }
    assert len(trace) == 1
    assert len(first_publication.files) == 5
    for item in first_publication.files:
        table = pq.read_table(pa.BufferReader(item.content))
        assert table.schema.metadata is not None
        assert b"harbor_hf.schema_version" in table.schema.metadata
        assert SECRET_SESSION not in item.content
        assert SHELLBENCH_TASK not in item.content


def test_publication_identity_is_stable_across_later_control_events(
    evidence: MemoryEvidence, source: EvidenceSource
) -> None:
    before = build_result_tables(evidence, source, control_commit="a" * 40)
    after = build_result_tables(evidence, source, control_commit="b" * 40)

    assert before.publication_id == after.publication_id
    assert before.runs[0].control_commit == "a" * 40
    assert after.runs[0].control_commit == "b" * 40


def test_global_index_is_discovery_only(
    evidence: MemoryEvidence, source: EvidenceSource
) -> None:
    tables = build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)

    row = build_global_index_row(
        tables,
        result_dataset="org/shellbench-results",
        result_revision="d" * 40,
    )

    assert row.result_revision == "d" * 40
    assert "trial" not in type(row).model_fields
    assert "artifact" not in type(row).model_fields
    assert "task_name" not in type(row).model_fields
    assert "source_prefix" not in type(row).model_fields


def test_audit_and_rebuild_equal_canonical_rows(
    evidence: MemoryEvidence, source: EvidenceSource
) -> None:
    tables = build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)

    report = audit_result_tables(
        evidence,
        source,
        control_commit=CONTROL_COMMIT,
        observed=tables,
    )
    rebuilt = rebuild_result_tables(
        evidence,
        [RebuildRequest(source=source, control_commit=CONTROL_COMMIT)],
    )

    assert report.publication_id == tables.publication_id
    assert report.row_counts == {
        "runs": 1,
        "trials": 2,
        "executions": 3,
        "metrics": 2,
        "artifacts": 1,
    }
    assert rebuilt == [tables]
    tampered = tables.model_copy(
        update={"runs": [tables.runs[0].model_copy(update={"trial_count": 999})]}
    )
    with pytest.raises(ResultPublicationError, match="differ"):
        audit_result_tables(
            evidence,
            source,
            control_commit=CONTROL_COMMIT,
            observed=tampered,
        )
    with pytest.raises(ResultPublicationError, match="duplicate publication"):
        rebuild_result_tables(
            evidence,
            [
                RebuildRequest(source=source, control_commit=CONTROL_COMMIT),
                RebuildRequest(source=source, control_commit=CONTROL_COMMIT),
            ],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("checksum", "checksum mismatch"),
        ("partial", "exclusively successful"),
        ("unlisted", "checksum manifest is incomplete"),
        ("wrong-lock", "does not match its run lock"),
    ],
)
def test_rejects_incomplete_or_tampered_raw_evidence(
    evidence: MemoryEvidence,
    source: EvidenceSource,
    mutation: str,
    message: str,
) -> None:
    if mutation == "checksum":
        evidence.files["verification.json"] = b"tampered"
    elif mutation == "partial":
        evidence.files["_PARTIAL"] = evidence.files.pop("_SUCCESS")
    elif mutation == "unlisted":
        evidence.files["unlisted.json"] = b"{}"
    else:
        evidence.files["run.lock.json"] = _json_bytes({"run_id": "other"})
        checksums = {
            path: _sha256(value)
            for path, value in evidence.files.items()
            if path not in {"checksums.json", "_SUCCESS"}
        }
        evidence.files["checksums.json"] = _json_bytes(checksums)

    with pytest.raises(ResultPublicationError, match=message):
        build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)


def test_rejects_unsanitized_or_expansive_summary(
    summary_value: dict[str, object], source: EvidenceSource
) -> None:
    unsanitized = dict(summary_value)
    unsanitized["sanitized"] = False
    with pytest.raises(ResultPublicationError, match="summary or run lock"):
        build_result_tables(
            _evidence(unsanitized), source, control_commit=CONTROL_COMMIT
        )

    expansive = dict(summary_value)
    expansive["task_instructions"] = "forbidden public task body"
    with pytest.raises(ResultPublicationError, match="summary or run lock"):
        build_result_tables(_evidence(expansive), source, control_commit=CONTROL_COMMIT)


def test_rejects_raw_session_artifact_metadata(
    summary_value: dict[str, object],
) -> None:
    summary = json.loads(json.dumps(summary_value))
    summary["artifacts"][0]["path"] = "sessions/raw.json"

    with pytest.raises(ValidationError, match="not publishable"):
        ResultEvidence.model_validate(summary)


@pytest.mark.parametrize("field", ["sha256", "size_bytes"])
def test_artifact_rows_must_match_checksummed_evidence(
    evidence: MemoryEvidence,
    source: EvidenceSource,
    field: str,
) -> None:
    summary = json.loads(evidence.files["run-summary.json"])
    artifact = summary["artifacts"][0]
    artifact[field] = "sha256:" + "0" * 64 if field == "sha256" else 999
    evidence.files["run-summary.json"] = _json_bytes(summary)
    _refresh_checksums(evidence)

    with pytest.raises(ResultPublicationError, match="artifact row"):
        build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)


def test_rejects_invalid_control_commit_and_paths(
    evidence: MemoryEvidence, source: EvidenceSource
) -> None:
    with pytest.raises(ValueError, match="control commit"):
        build_result_tables(evidence, source, control_commit="mutable-main")
    with pytest.raises(ValidationError, match="canonical relative"):
        EvidenceSource(bucket="bucket", prefix="../escape")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("selected", "selected execution"),
        ("execution-owner", "unknown trial"),
        ("metric-owner", "unknown owner"),
        ("duplicate-metric", "duplicate metric"),
        ("duplicate-artifact", "duplicate artifact"),
    ],
)
def test_rejects_inconsistent_summary_references(
    summary_value: dict[str, object], mutation: str, message: str
) -> None:
    summary = json.loads(json.dumps(summary_value))
    if mutation == "selected":
        summary["trials"][0]["selected_execution_id"] = "execution-one"
    elif mutation == "execution-owner":
        summary["executions"][1]["trial_id"] = "trial-missing"
    elif mutation == "metric-owner":
        summary["metrics"][0]["owner_id"] = "trial-missing"
    elif mutation == "duplicate-metric":
        summary["metrics"].append(summary["metrics"][0])
    else:
        summary["artifacts"].append(summary["artifacts"][0])

    with pytest.raises(ValidationError, match=message):
        ResultEvidence.model_validate(summary)


def test_frozen_parquet_schema_matches_golden() -> None:
    golden = Path("tests/golden/result-schemas-v1.json")

    assert result_schema_manifest() == json.loads(golden.read_text(encoding="utf-8"))
    models: list[tuple[TableName, type[TraceRow]]] = [
        ("runs", RunRow),
        ("trials", TrialRow),
        ("executions", ExecutionRow),
        ("metrics", MetricRow),
        ("artifacts", ArtifactRow),
    ]
    for table, model in models:
        assert parquet_schema(table).names == list(model.model_fields)
    assert index_parquet_schema().names == list(GlobalIndexRow.model_fields)
