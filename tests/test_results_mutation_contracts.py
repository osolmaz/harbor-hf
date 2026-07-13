from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

import pyarrow as pa
import pytest

from harbor_hf.results import (
    EvidenceSource,
    ResultPublicationError,
    _canonical_json,
    _field,
    _is_commit,
    _make_schema,
    _schema_description,
    _trace_fields,
    _validate_relative_path,
    build_global_index_row,
    build_index_file,
    build_result_publication,
    build_result_tables,
)

NOW = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)
SOURCE = EvidenceSource(
    bucket="hf://buckets/private-evidence",
    prefix="campaigns/campaign-mutation/runs/run-mutation",
)
CONTROL_COMMIT = "c" * 40


class RecordingEvidence:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.calls: list[tuple[str, str, str, str | None]] = []

    def list_files(self, *, bucket: str, prefix: str) -> list[str]:
        self.calls.append(("list", bucket, prefix, None))
        return list(reversed(self.files))

    def read_bytes(self, *, bucket: str, prefix: str, path: str) -> bytes:
        self.calls.append(("read", bucket, prefix, path))
        return self.files[path]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _reader() -> RecordingEvidence:
    artifact = _json_bytes({"verified": True, "score": 0.75})
    summary = {
        "schema_version": "harbor-hf/result-evidence/v1",
        "sanitized": True,
        "run": {
            "run_id": "run-mutation",
            "campaign_id": "campaign-mutation",
            "experiment": "experiment-mutation",
            "benchmark": "shellbench",
            "benchmark_revision": "sha256:" + "b" * 64,
            "result_kind": "ordinary",
            "outcome": "complete",
            "created_at": NOW.isoformat(),
            "completed_at": (NOW + timedelta(minutes=5)).isoformat(),
            "model_id": "model-mutation",
            "model_repo": "org/model",
            "model_revision": "a" * 40,
            "deployment_id": "deployment-mutation",
            "provider": "huggingface",
            "region": "aws-us-east-1",
            "hardware": "a100",
            "accelerator_count": 2,
            "agent_id": "agent-mutation",
            "agent_name": "mutation-agent",
            "agent_revision": "1.2.3",
        },
        "trials": [
            {
                "trial_id": "trial-z",
                "task_name": "task-z",
                "task_digest": "sha256:" + "2" * 64,
                "logical_attempt": 2,
                "selected_execution_id": "execution-z",
                "outcome": "complete",
            },
            {
                "trial_id": "trial-a",
                "task_name": "task-a",
                "task_digest": "sha256:" + "1" * 64,
                "logical_attempt": 1,
                "selected_execution_id": "execution-b",
                "outcome": "complete",
            },
        ],
        "executions": [
            {
                "execution_id": "execution-z",
                "trial_id": "trial-z",
                "physical_attempt": 1,
                "runtime_kind": "provider",
                "status": "succeeded",
                "started_at": (NOW + timedelta(minutes=3)).isoformat(),
                "completed_at": (NOW + timedelta(minutes=4)).isoformat(),
                "retry_reason": None,
                "remote_job_id": None,
            },
            {
                "execution_id": "execution-a",
                "trial_id": "trial-a",
                "physical_attempt": 1,
                "runtime_kind": "endpoint",
                "status": "failed_infrastructure",
                "started_at": NOW.isoformat(),
                "completed_at": (NOW + timedelta(minutes=1)).isoformat(),
                "retry_reason": None,
                "remote_job_id": "job-a",
            },
            {
                "execution_id": "execution-b",
                "trial_id": "trial-a",
                "physical_attempt": 2,
                "runtime_kind": "endpoint",
                "status": "succeeded",
                "started_at": (NOW + timedelta(minutes=1)).isoformat(),
                "completed_at": (NOW + timedelta(minutes=2)).isoformat(),
                "retry_reason": "provider_timeout",
                "remote_job_id": "job-b",
            },
        ],
        "metrics": [
            {
                "owner_type": "trial",
                "owner_id": "trial-z",
                "name": "reward",
                "value": 0.0,
                "unit": "score",
                "aggregation": None,
            },
            {
                "owner_type": "execution",
                "owner_id": "execution-b",
                "name": "latency",
                "value": 1.25,
                "unit": "seconds",
                "aggregation": "mean",
            },
        ],
        "artifacts": [
            {
                "owner_type": "run",
                "owner_id": "run-mutation",
                "kind": "verification",
                "path": "verification.json",
                "sha256": _sha256(artifact),
                "media_type": "application/json",
                "size_bytes": len(artifact),
            }
        ],
    }
    files = {
        "run.lock.json": _json_bytes(
            {"run_id": "run-mutation", "cell_digest": "sha256:" + "d" * 64}
        ),
        "run-summary.json": _json_bytes(summary),
        "verification.json": artifact,
        "logs/controller.jsonl": b'{"event":"complete"}\n',
    }
    files["checksums.json"] = _json_bytes(
        {path: _sha256(content) for path, content in files.items()}
    )
    files["_SUCCESS"] = b"\n"
    return RecordingEvidence(files)


def test_full_result_rows_publication_and_index_have_canonical_hashes() -> None:
    reader = _reader()
    tables = build_result_tables(reader, SOURCE, control_commit=CONTROL_COMMIT)
    publication = build_result_publication(tables)
    index_row = build_global_index_row(
        tables,
        result_dataset="org/results",
        result_revision="d" * 40,
    )
    index_file = build_index_file(index_row)

    assert _canonical_hash(tables.model_dump(mode="json")) == (
        "9446ffafff3a041cf6e5875cc00395788114e4752e9e3d19d24e1d8c118dff8d"
    )
    assert (
        _canonical_hash(
            {
                "publication_id": publication.tables.publication_id,
                "receipt_path": publication.receipt_path,
                "receipt": publication.receipt.decode(),
                "files": [
                    {
                        "path": item.path,
                        "sha256": _sha256(item.content),
                        "size": len(item.content),
                    }
                    for item in publication.files
                ],
                "index_row": index_row.model_dump(mode="json"),
                "index_path": index_file.path,
                "index_sha256": _sha256(index_file.content),
                "index_size": len(index_file.content),
            }
        )
        == "eb742baf36a4ca0a9b7bff3694596b386f956f17f22e99928fc0717d60d2bd36"
    )
    assert [_sha256(item.content) for item in publication.files] == [
        "sha256:19bbad897a36cbb8657b9f5c51d3c16979ad9ce0d298d8642afbeae8b6478748",
        "sha256:90a7648b47a7cf22e853e6409c44d6bd14cf9b95ac326ae24604ff56e6287927",
        "sha256:a1e4281ad853ef176dcf238a08cc636756ae369a5fd4bd40c17cb6716f3925eb",
        "sha256:1fbb5ab3a2a2cec3e497063b7689d2106dd46d9e4c4aacb7d05d071fc51cb2a4",
        "sha256:151542d24d200a96801dbbb64dca8dbe7cf51f9ff2a7b09ad0144f5bb7d9758c",
    ]

    common = ("hf://buckets/private-evidence", SOURCE.prefix)
    assert reader.calls == [
        ("list", *common, None),
        ("read", *common, "checksums.json"),
        ("read", *common, "logs/controller.jsonl"),
        ("read", *common, "run-summary.json"),
        ("read", *common, "run.lock.json"),
        ("read", *common, "verification.json"),
        ("read", *common, "run-summary.json"),
        ("read", *common, "run.lock.json"),
        ("read", *common, "verification.json"),
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda files: files.update({"_FAILED": b"\n"}), "exclusively successful"),
        (lambda files: files.pop("_SUCCESS"), "exclusively successful"),
        (lambda files: files.pop("checksums.json"), "no checksum manifest"),
        (
            lambda files: files.update({"checksums.json": b"[]\n"}),
            "checksum manifest is invalid",
        ),
        (
            lambda files: files.update({"checksums.json": b'{"missing":"nope"}\n'}),
            "checksum manifest is invalid",
        ),
        (
            lambda files: files.update({"run.lock.json": b'{"run_id":"wrong"}\n'}),
            "checksum mismatch",
        ),
    ],
)
def test_evidence_rejection_matrix_checks_every_manifest_boundary(
    mutate: Callable[[dict[str, bytes]], object], message: str
) -> None:
    reader = _reader()
    mutate(reader.files)

    with pytest.raises(ResultPublicationError) as captured:
        build_result_tables(reader, SOURCE, control_commit=CONTROL_COMMIT)

    assert message in str(captured.value)


def test_schema_construction_and_canonical_helpers_have_complete_outputs() -> None:
    fields = _trace_fields()
    assert [(field.name, str(field.type), field.nullable) for field in fields] == [
        ("schema_version", "string", False),
        ("publication_id", "string", False),
        ("run_id", "string", False),
        ("source_bucket", "string", False),
        ("source_prefix", "string", False),
        ("source_checksum", "string", False),
        ("run_lock_path", "string", False),
        ("run_lock_sha256", "string", False),
        ("control_commit", "string", False),
    ]
    assert _field("optional", pa.int64(), nullable=True) == pa.field(
        "optional", pa.int64(), nullable=True
    )
    schema = _make_schema("mutation/v1", [*fields, _field("count", pa.int64())])
    assert _schema_description(schema) == {
        "schema_version": "mutation/v1",
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
    }
    assert _canonical_json({"snowman": "☃", "b": 2, "a": 1}) == (
        b'{"a":1,"b":2,"snowman":"\\u2603"}\n'
    )


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("a" * 40, True),
        ("f" * 64, True),
        ("a" * 39, False),
        ("a" * 41, False),
        ("g" * 40, False),
        ("A" * 40, False),
        ("-" * 40, False),
    ],
)
def test_commit_validation_matrix(value: str, valid: bool) -> None:
    assert _is_commit(value) is valid


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("artifact.json", True),
        ("nested/artifact.json", True),
        ("", False),
        ("/absolute", False),
        ("../escape", False),
        ("nested/../escape", False),
        ("nested//artifact", False),
        ("nested\\artifact", False),
        ("./artifact", False),
        ("nested/./artifact", False),
    ],
)
def test_relative_path_validation_matrix(value: str, valid: bool) -> None:
    if valid:
        assert _validate_relative_path(value) == PurePosixPath(value)
    else:
        with pytest.raises(ValueError) as captured:
            _validate_relative_path(value)
        assert str(captured.value) == (
            "evidence paths must be canonical relative POSIX paths"
        )
