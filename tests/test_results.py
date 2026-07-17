from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from harbor_hf.publication_envelope import (
    SourcePublicationReference,
    SourceTrialSelection,
    execution_profile_digest,
)
from harbor_hf.results import (
    ArtifactRow,
    EvidenceSource,
    ExecutionRow,
    GlobalIndexRow,
    MetricRow,
    PublicationProvenance,
    RebuildRequest,
    ResultCompositionManifest,
    ResultEvidence,
    ResultPublicationError,
    ResultTables,
    RunRow,
    TableName,
    TraceRow,
    TrialRow,
    UnsupportedTask,
    audit_result_tables,
    build_composed_result_publication,
    build_global_index_row,
    build_result_publication,
    build_result_tables,
    compose_result_tables,
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
            "evaluation_id": "evaluation-one",
            "publication_role": "component",
            "component_kind": "base",
            "benchmark": "shellbench",
            "benchmark_revision": "sha256:" + "b" * 64,
            "result_kind": "ordinary",
            "outcome": "complete",
            "quality": "clean",
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
                "outcome": "scored",
            },
            {
                "trial_id": "trial-one",
                "task_name": "task-one",
                "task_digest": "sha256:" + "1" * 64,
                "logical_attempt": 1,
                "selected_execution_id": "execution-two",
                "outcome": "scored",
            },
        ],
        "executions": [
            {
                "execution_id": "execution-three",
                "trial_id": "trial-two",
                "physical_attempt": 1,
                "runtime_kind": "provider",
                "status": "succeeded",
                "failure_category": None,
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
                "status": "failed",
                "failure_category": "transient",
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
                "failure_category": None,
                "started_at": (NOW + timedelta(minutes=1)).isoformat(),
                "completed_at": (NOW + timedelta(minutes=2)).isoformat(),
                "retry_reason": "provider_timeout",
                "remote_job_id": "job-two",
            },
        ],
        "metrics": [
            {
                "owner_type": "trial",
                "owner_id": "trial-one",
                "name": "reward",
                "value": 1.0,
                "unit": "score",
                "aggregation": None,
            },
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
        "run.lock.json": _json_bytes(
            {
                "run_id": "run-one",
                "evaluation_id": "evaluation-one",
                "publication_role": "component",
                "component_kind": "base",
                "cell_digest": "x",
                "attempts": 1,
                "model": {
                    "id": "model-one",
                    "repo": "org/model",
                    "revision": "a" * 40,
                    "weights": {"format": "safetensors"},
                },
                "deployment": {
                    "id": "deployment-one",
                    "provider": "hf-inference-endpoints",
                    "hardware": "a100",
                    "accelerator_count": 1,
                    "region": "aws-us-east-1",
                    "engine": {
                        "name": "vllm",
                        "image": "vllm/vllm-openai:latest",
                    },
                },
                "agent": {
                    "id": "agent-one",
                    "name": "example-agent",
                    "revision": "1.2.3",
                    "revision_kind": "package",
                },
                "benchmark_task_digests": {
                    "task-one": "sha256:" + "1" * 64,
                    "task-two": "sha256:" + "2" * 64,
                },
            }
        ),
        "run-summary.json": _json_bytes(normalized),
        "verification.json": verification,
        "trials/trial-one/executions/execution-two/session.json": SECRET_SESSION,
        "trials/trial-one/task-source/instruction.md": SHELLBENCH_TASK,
    }
    checksums = {path: _sha256(value) for path, value in files.items()}
    files["checksums.json"] = _json_bytes(checksums)
    files[marker] = b"\n"
    evidence = MemoryEvidence(files)
    _add_envelope(evidence)
    return evidence


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


def _add_envelope(evidence: MemoryEvidence) -> None:
    summary = json.loads(evidence.files["run-summary.json"])
    run_lock = evidence.files["run.lock.json"]
    executions = []
    for record in summary["executions"]:
        bundle = None
        if record["status"] == "succeeded":
            prefix = f"trials/{record['trial_id']}/executions/{record['execution_id']}"
            manifest_path = f"{prefix}/harbor-native-bundle.json"
            archive_path = f"{prefix}/artifacts.tar.gz"
            manifest = f"manifest for {record['execution_id']}".encode()
            archive = f"archive for {record['execution_id']}".encode()
            evidence.files[manifest_path] = manifest
            evidence.files[archive_path] = archive
            bundle = {
                "manifest": {
                    "path": manifest_path,
                    "digest": _sha256(manifest),
                    "size_bytes": len(manifest),
                },
                "archive": {
                    "path": archive_path,
                    "digest": _sha256(archive),
                    "size_bytes": len(archive),
                },
                "harbor_revision": "a" * 40,
                "harbor_version": "0.1.0",
                "compatibility_schema": "harbor-hf/harbor-compatibility/v1alpha3",
                "request_digest": "sha256:" + "a" * 64,
                "document_count": 2,
            }
        executions.append(
            {
                "execution_id": record["execution_id"],
                "trial_id": record["trial_id"],
                "physical_attempt": record["physical_attempt"],
                "status": record["status"],
                "failure_category": record["failure_category"],
                "started_at": record["started_at"],
                "completed_at": record["completed_at"],
                "retry_reason": record["retry_reason"],
                "remote_job_id": record["remote_job_id"],
                "bundle_status": "verified" if bundle else "not_available",
                "harbor_bundle": bundle,
            }
        )
    envelope = {
        "schema_version": "harbor-hf/publication-envelope/v1",
        "run_id": summary["run"]["run_id"],
        "campaign_id": summary["run"]["campaign_id"],
        "created_at": summary["run"]["created_at"],
        "completed_at": summary["run"]["completed_at"],
        "evidence_bucket": "hf://buckets/private-evidence",
        "evidence_prefix": "campaigns/campaign-one/runs/run-one",
        "run_lock": {
            "path": "run.lock.json",
            "digest": _sha256(run_lock),
            "size_bytes": len(run_lock),
        },
        "profiles": {
            "experiment": "sha256:" + "1" * 64,
            "model": "sha256:" + "2" * 64,
            "deployment": "sha256:" + "3" * 64,
            "agent": "sha256:" + "4" * 64,
        },
        "runtime": {
            "kind": "endpoint",
            "provider": summary["run"]["provider"],
            "region": summary["run"]["region"],
            "hardware": summary["run"]["hardware"],
            "accelerator_count": summary["run"]["accelerator_count"],
        },
        "sanitizer_version": "harbor-hf/public-results/v1",
        "projection_version": "harbor-hf/results-projection/v1",
        "cleanup_outcome": "verified",
        "executions": executions,
    }
    evidence.files["publication-envelope.v1.json"] = _json_bytes(envelope)
    _refresh_checksums(evidence)


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
    assert [row.value for row in first.metrics] == [1.25, 1.0, 0.0]
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
    assert len(first_publication.files) == 6
    for item in first_publication.files:
        if not item.path.startswith("data/"):
            continue
        table = pq.read_table(pa.BufferReader(item.content))
        assert table.schema.metadata is not None
        assert b"harbor_hf.schema_version" in table.schema.metadata
        assert SECRET_SESSION not in item.content
        assert SHELLBENCH_TASK not in item.content


def test_exhausted_trial_failure_is_a_zero_score_result(
    source: EvidenceSource,
) -> None:
    summary = sample_summary()
    trials = summary["trials"]
    assert isinstance(trials, list)
    failed_trial = cast(dict[str, object], trials[0])
    failed_trial["outcome"] = "benchmark_failed"
    run = summary["run"]
    assert isinstance(run, dict)
    cast(dict[str, object], run)["quality"] = "degraded"
    executions = summary["executions"]
    assert isinstance(executions, list)
    failed_execution = cast(dict[str, object], executions[0])
    failed_execution["status"] = "failed"
    failed_execution["failure_category"] = "benchmark"

    tables = build_result_tables(
        _evidence(summary), source, control_commit=CONTROL_COMMIT
    )

    observed = next(row for row in tables.trials if row.trial_id == "trial-two")
    assert observed.outcome == "benchmark_failed"
    reward = next(
        row
        for row in tables.metrics
        if row.owner_id == "trial-two" and row.name == "reward"
    )
    assert reward.value == 0.0


def test_rejects_task_outcome_that_conflicts_with_failure_category() -> None:
    summary = sample_summary()
    cast(dict[str, object], summary["run"])["quality"] = "degraded"
    trial = cast(list[dict[str, object]], summary["trials"])[0]
    trial["outcome"] = "agent_failed"
    execution = cast(list[dict[str, object]], summary["executions"])[0]
    execution["status"] = "failed"
    execution["failure_category"] = "benchmark"

    with pytest.raises(ValidationError, match="conflicts with its outcome"):
        ResultEvidence.model_validate(summary)


def test_builds_projection_bound_to_native_envelope(
    evidence: MemoryEvidence, source: EvidenceSource
) -> None:
    _add_envelope(evidence)

    tables = build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)
    publication = build_result_publication(tables)

    assert tables.provenance is not None
    assert len(tables.provenance.harbor_archive_sha256s) == 2
    projection_file = next(
        item for item in publication.files if item.path.startswith("projections/")
    )
    projection = json.loads(projection_file.content)
    assert projection["envelope_sha256"] == tables.provenance.envelope_sha256
    assert projection["source_checksum"] == tables.runs[0].source_checksum
    assert set(projection["tables"]) == {
        "runs",
        "trials",
        "executions",
        "metrics",
        "artifacts",
    }
    published = b"".join(item.content for item in publication.files)
    assert SECRET_SESSION not in published
    assert SHELLBENCH_TASK not in published


def test_rejects_envelope_with_unverified_bundle(
    evidence: MemoryEvidence, source: EvidenceSource
) -> None:
    _add_envelope(evidence)
    envelope = json.loads(evidence.files["publication-envelope.v1.json"])
    succeeded = next(
        record for record in envelope["executions"] if record["harbor_bundle"]
    )
    succeeded["harbor_bundle"]["archive"]["digest"] = "sha256:" + "0" * 64
    evidence.files["publication-envelope.v1.json"] = _json_bytes(envelope)
    _refresh_checksums(evidence)

    with pytest.raises(ResultPublicationError, match="unverified Harbor archive"):
        build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)


def test_rejects_legacy_success_without_native_provenance(
    evidence: MemoryEvidence, source: EvidenceSource
) -> None:
    _add_envelope(evidence)
    envelope = json.loads(evidence.files["publication-envelope.v1.json"])
    succeeded = next(
        record for record in envelope["executions"] if record["status"] == "succeeded"
    )
    succeeded["bundle_status"] = "legacy_unavailable"
    succeeded["harbor_bundle"] = None
    evidence.files["publication-envelope.v1.json"] = _json_bytes(envelope)
    _refresh_checksums(evidence)

    with pytest.raises(ResultPublicationError, match="envelope is invalid"):
        build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)


def test_rejects_evidence_without_canonical_envelope(
    evidence: MemoryEvidence, source: EvidenceSource
) -> None:
    del evidence.files["publication-envelope.v1.json"]
    _refresh_checksums(evidence)

    with pytest.raises(
        ResultPublicationError, match="no canonical publication envelope"
    ):
        build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)


def test_rejects_envelope_execution_that_conflicts_with_summary(
    evidence: MemoryEvidence, source: EvidenceSource
) -> None:
    _add_envelope(evidence)
    envelope = json.loads(evidence.files["publication-envelope.v1.json"])
    envelope["executions"][0]["trial_id"] = "trial-imposter"
    evidence.files["publication-envelope.v1.json"] = _json_bytes(envelope)
    _refresh_checksums(evidence)

    with pytest.raises(ResultPublicationError, match="executions conflict"):
        build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)


def test_rejects_envelope_failure_category_that_conflicts_with_summary(
    evidence: MemoryEvidence, source: EvidenceSource
) -> None:
    _add_envelope(evidence)
    envelope = json.loads(evidence.files["publication-envelope.v1.json"])
    failed = next(
        record for record in envelope["executions"] if record["status"] == "failed"
    )
    failed["failure_category"] = "benchmark"
    evidence.files["publication-envelope.v1.json"] = _json_bytes(envelope)
    _refresh_checksums(evidence)

    with pytest.raises(ResultPublicationError, match="executions conflict"):
        build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)


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
        "metrics": 3,
        "artifacts": 1,
    }
    assert rebuilt == [tables]
    tampered = tables.model_copy(
        update={
            "runs": [tables.runs[0].model_copy(update={"planned_trial_count": 999})]
        }
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


def test_rejects_summary_that_omits_a_locked_task(
    summary_value: dict[str, object], source: EvidenceSource
) -> None:
    summary = json.loads(json.dumps(summary_value))
    summary["trials"] = summary["trials"][:1]
    selected_trial = summary["trials"][0]["trial_id"]
    summary["executions"] = [
        execution
        for execution in summary["executions"]
        if execution["trial_id"] == selected_trial
    ]
    summary["metrics"] = [
        metric
        for metric in summary["metrics"]
        if metric["owner_type"] == "run" or metric["owner_id"] == selected_trial
    ]

    with pytest.raises(ResultPublicationError, match="tasks do not match"):
        build_result_tables(_evidence(summary), source, control_commit=CONTROL_COMMIT)


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


def test_unsupported_trial_has_no_execution(
    summary_value: dict[str, object],
) -> None:
    summary = json.loads(json.dumps(summary_value))
    unsupported = summary["trials"][0]
    unsupported["selected_execution_id"] = None
    unsupported["outcome"] = "unsupported"
    summary["run"]["quality"] = "degraded"
    unsupported_trial_id = unsupported["trial_id"]
    summary["executions"] = [
        execution
        for execution in summary["executions"]
        if execution["trial_id"] != unsupported_trial_id
    ]

    assert ResultEvidence.model_validate(summary).trials[0].outcome == "unsupported"

    summary_with_execution = json.loads(json.dumps(summary_value))
    summary_with_execution["trials"][0]["selected_execution_id"] = None
    summary_with_execution["trials"][0]["outcome"] = "unsupported"
    summary_with_execution["run"]["quality"] = "degraded"
    with pytest.raises(ValidationError, match="physical execution"):
        ResultEvidence.model_validate(summary_with_execution)


def test_composes_correction_and_unsupported_trials(
    evidence: MemoryEvidence,
    source: EvidenceSource,
) -> None:
    manifest, sources = _composition_inputs(evidence, source)

    result = compose_result_tables(manifest, sources, control_commit=CONTROL_COMMIT)
    repeated = compose_result_tables(manifest, sources, control_commit=CONTROL_COMMIT)

    assert result == repeated
    run = result.tables.runs[0]
    assert run.result_kind == "composed"
    assert run.planned_trial_count == 3
    assert run.scored_trial_count == 1
    assert run.benchmark_failed_count == 1
    assert run.unsupported_count == 1
    assert run.execution_count == 2
    trials = {trial.task_name: trial for trial in result.tables.trials}
    assert trials["task-one"].outcome == "benchmark_failed"
    assert trials["task-two"].outcome == "scored"
    assert trials["task-three"].outcome == "unsupported"
    assert trials["task-three"].selected_execution_id is None
    assert {row.execution_id for row in result.tables.executions} == {
        "execution-three",
        "execution-correction",
    }
    assert all(
        execution.bundle_status == "source_publication"
        for execution in result.envelope.executions
    )

    publication = build_composed_result_publication(result)
    composition_path = f"compositions/{result.tables.publication_id}.json"
    assert composition_path in {item.path for item in publication.files}
    receipt = json.loads(publication.receipt)
    assert composition_path in receipt["files"]


def test_composition_accepts_different_profile_labels(
    evidence: MemoryEvidence,
    source: EvidenceSource,
) -> None:
    manifest, sources = _composition_inputs(evidence, source)
    correction_reference = next(
        item for item in manifest.sources if item.role == "correction"
    )
    correction = sources[correction_reference.publication_id]
    relabeled_run = correction.runs[0].model_copy(
        update={
            "model_id": "model-correction",
            "deployment_id": "deployment-correction",
            "agent_id": "agent-correction",
        }
    )
    relabeled = correction.model_copy(update={"runs": [relabeled_run]})

    result = compose_result_tables(
        manifest,
        {**sources, correction_reference.publication_id: relabeled},
        control_commit=CONTROL_COMMIT,
    )

    base_run = sources[manifest.sources[0].publication_id].runs[0]
    composed_run = result.tables.runs[0]
    assert composed_run.model_id == base_run.model_id
    assert composed_run.deployment_id == base_run.deployment_id
    assert composed_run.agent_id == base_run.agent_id
    assert composed_run.evaluation_id == base_run.evaluation_id
    assert "evaluation_id" not in manifest.model_dump(mode="json")


def test_execution_profile_digest_ignores_labels_binding_and_cost_estimates() -> None:
    profiles = {
        "model": {"id": "model-base", "repo": "org/model", "revision": "abc"},
        "deployment": {
            "id": "deployment-base",
            "endpoint": {
                "namespace": "org",
                "name": "endpoint-base",
                "served_model_name": "/repository",
            },
            "limits": {
                "max_concurrent_requests": 16,
                "max_attempts": 3,
                "max_spend_usd": "250",
                "estimated_wave_cost_usd": "150",
            },
            "engine": {"arguments": ["--max-model-len", "65536"]},
        },
        "agent": {
            "id": "agent-base",
            "name": "openclaw",
            "parameters": {"thinking": "high"},
        },
    }
    expected = execution_profile_digest(**profiles)
    relabeled = {
        **profiles,
        "model": {**profiles["model"], "id": "model-correction"},
        "deployment": {
            **profiles["deployment"],
            "id": "deployment-correction",
            "endpoint": {
                "namespace": "org",
                "name": "endpoint-correction",
                "served_model_name": "/repository",
            },
            "limits": {
                **profiles["deployment"]["limits"],
                "max_spend_usd": "100",
                "estimated_wave_cost_usd": "50",
            },
        },
        "agent": {**profiles["agent"], "id": "agent-correction"},
    }
    changed = {
        **relabeled,
        "agent": {
            **relabeled["agent"],
            "parameters": {"thinking": "off"},
        },
    }
    changed_served_name = {
        **relabeled,
        "deployment": {
            **relabeled["deployment"],
            "endpoint": {
                **relabeled["deployment"]["endpoint"],
                "served_model_name": "different-model",
            },
        },
    }
    changed_concurrency = {
        **relabeled,
        "deployment": {
            **relabeled["deployment"],
            "limits": {
                **relabeled["deployment"]["limits"],
                "max_concurrent_requests": 8,
            },
        },
    }

    assert execution_profile_digest(**relabeled) == expected
    assert execution_profile_digest(**changed) != expected
    assert execution_profile_digest(**changed_served_name) != expected
    assert execution_profile_digest(**changed_concurrency) != expected


def test_composition_rejects_unresolved_base_trial(
    evidence: MemoryEvidence,
    source: EvidenceSource,
) -> None:
    manifest, sources = _composition_inputs(evidence, source)
    base_reference = next(item for item in manifest.sources if item.role == "base")
    incomplete = manifest.model_copy(update={"sources": [base_reference]})

    with pytest.raises(ResultPublicationError, match="resolve every task"):
        compose_result_tables(
            incomplete,
            {base_reference.publication_id: sources[base_reference.publication_id]},
            control_commit=CONTROL_COMMIT,
        )


def test_composition_rejects_incompatible_correction(
    evidence: MemoryEvidence,
    source: EvidenceSource,
) -> None:
    manifest, sources = _composition_inputs(evidence, source)
    correction_reference = next(
        item for item in manifest.sources if item.role == "correction"
    )
    correction = sources[correction_reference.publication_id]
    incompatible_run = correction.runs[0].model_copy(
        update={"model_revision": "d" * 40}
    )
    incompatible = correction.model_copy(update={"runs": [incompatible_run]})

    with pytest.raises(ResultPublicationError, match="incompatible"):
        compose_result_tables(
            manifest,
            {**sources, correction_reference.publication_id: incompatible},
            control_commit=CONTROL_COMMIT,
        )


def test_composition_rejects_incompatible_execution_profile(
    evidence: MemoryEvidence,
    source: EvidenceSource,
) -> None:
    manifest, sources = _composition_inputs(evidence, source)
    correction_reference = next(
        item for item in manifest.sources if item.role == "correction"
    )
    correction = sources[correction_reference.publication_id]
    incompatible = correction.model_copy(
        update={
            "provenance": correction.provenance.model_copy(
                update={"execution_profile_sha256": "sha256:" + "7" * 64}
            )
        }
    )

    with pytest.raises(ResultPublicationError, match="incompatible"):
        compose_result_tables(
            manifest,
            {**sources, correction_reference.publication_id: incompatible},
            control_commit=CONTROL_COMMIT,
        )


def test_composition_rejects_mixed_runtime_kinds(
    evidence: MemoryEvidence,
    source: EvidenceSource,
) -> None:
    manifest, sources = _composition_inputs(evidence, source)
    correction_reference = next(
        item for item in manifest.sources if item.role == "correction"
    )
    correction = sources[correction_reference.publication_id]
    provider_execution = correction.executions[0].model_copy(
        update={"runtime_kind": "provider"}
    )
    mixed = correction.model_copy(update={"executions": [provider_execution]})

    with pytest.raises(ResultPublicationError, match="mixed runtime kinds"):
        compose_result_tables(
            manifest,
            {**sources, correction_reference.publication_id: mixed},
            control_commit=CONTROL_COMMIT,
        )


def _composition_inputs(
    evidence: MemoryEvidence,
    source: EvidenceSource,
) -> tuple[ResultCompositionManifest, dict[str, ResultTables]]:
    base = build_result_tables(evidence, source, control_commit=CONTROL_COMMIT)
    base = base.model_copy(
        update={
            "executions": [
                execution.model_copy(update={"runtime_kind": "endpoint"})
                for execution in base.executions
            ]
        }
    )
    base_run = base.runs[0]
    base_trial = next(trial for trial in base.trials if trial.task_name == "task-one")
    correction_trace = {
        "publication_id": "pub-correction",
        "run_id": "run-correction",
        "source_bucket": base_run.source_bucket,
        "source_prefix": "campaigns/campaign-correction/runs/run-correction",
        "source_checksum": "sha256:" + "4" * 64,
        "run_lock_path": "run.lock.json",
        "run_lock_sha256": "sha256:" + "5" * 64,
        "control_commit": CONTROL_COMMIT,
    }
    correction_run = RunRow.model_validate(
        {
            **base_run.model_dump(mode="python"),
            **correction_trace,
            "campaign_id": "campaign-correction",
            "component_kind": "correction",
            "quality": "degraded",
            "completed_at": NOW + timedelta(minutes=7),
            "planned_trial_count": 1,
            "scored_trial_count": 0,
            "benchmark_failed_count": 1,
            "execution_count": 1,
        }
    )
    correction_trial = TrialRow.model_validate(
        {
            **base_trial.model_dump(mode="python"),
            **correction_trace,
            "trial_id": "trial-correction",
            "selected_execution_id": "execution-correction",
            "outcome": "benchmark_failed",
        }
    )
    correction_execution = ExecutionRow.model_validate(
        {
            **correction_trace,
            "execution_id": "execution-correction",
            "trial_id": correction_trial.trial_id,
            "physical_attempt": 1,
            "runtime_kind": "endpoint",
            "status": "failed",
            "failure_category": "benchmark",
            "started_at": NOW + timedelta(minutes=5),
            "completed_at": NOW + timedelta(minutes=6),
            "retry_reason": None,
            "remote_job_id": "job-correction",
        }
    )
    correction_metric = MetricRow.model_validate(
        {
            **correction_trace,
            "metric_id": "metric-correction",
            "owner_type": "trial",
            "owner_id": correction_trial.trial_id,
            "name": "reward",
            "value": 0.0,
            "unit": "score",
            "aggregation": None,
        }
    )
    correction = ResultTables(
        publication_id=correction_trace["publication_id"],
        runs=[correction_run],
        trials=[correction_trial],
        executions=[correction_execution],
        metrics=[correction_metric],
        artifacts=[],
        provenance=PublicationProvenance.model_validate(base.provenance),
    )
    base_reference = SourcePublicationReference(
        role="base",
        publication_id=base.publication_id,
        run_id=base_run.run_id,
        result_dataset="org/results",
        result_revision="a" * 40,
        source_checksum=base_run.source_checksum,
        selected_trials=[SourceTrialSelection(task_name="task-two", logical_attempt=1)],
    )
    correction_reference = SourcePublicationReference(
        role="correction",
        publication_id=correction.publication_id,
        run_id=correction_run.run_id,
        result_dataset="org/results",
        result_revision="b" * 40,
        source_checksum=correction_run.source_checksum,
        selected_trials=[SourceTrialSelection(task_name="task-one", logical_attempt=1)],
    )
    manifest = ResultCompositionManifest(
        run_id="run-composed",
        campaign_id="campaign-composed",
        experiment="experiment-composed",
        created_at=base_run.created_at,
        completed_at=correction_run.completed_at,
        evidence_bucket=source.bucket,
        evidence_prefix="compositions/run-composed",
        sources=[base_reference, correction_reference],
        unsupported_tasks=[
            UnsupportedTask(
                task_name="task-three",
                task_digest="sha256:" + "3" * 64,
            )
        ],
    )
    return manifest, {
        base.publication_id: base,
        correction.publication_id: correction,
    }


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
