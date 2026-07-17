from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError

from harbor_hf.evidence import (
    append_event,
    verify_checksums,
    write_checksums,
    write_json,
)
from harbor_hf.result_publisher import (
    DatasetPublicationError,
    HubDatasetPublisher,
    PublicationConflict,
)
from harbor_hf.results import (
    PublicationProvenance,
    ResultPublication,
    ResultTables,
    RunRow,
    build_result_publication,
)

NOW = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _publication() -> ResultPublication:
    trace = {
        "publication_id": "pub-" + "1" * 32,
        "run_id": "run-publication",
        "source_bucket": "hf://buckets/private-evidence",
        "source_prefix": "campaigns/campaign-publication/runs/run-publication",
        "source_checksum": "sha256:" + "2" * 64,
        "run_lock_path": "run.lock.json",
        "run_lock_sha256": "sha256:" + "3" * 64,
        "control_commit": "4" * 40,
    }
    run = RunRow.model_validate(
        {
            **trace,
            "campaign_id": "campaign-publication",
            "experiment": "experiment-publication",
            "evaluation_id": "evaluation-publication",
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
            "model_id": "model-publication",
            "model_repo": "org/model",
            "model_revision": "6" * 40,
            "deployment_id": "deployment-publication",
            "provider": "huggingface",
            "region": "aws-us-east-1",
            "hardware": "a100",
            "accelerator_count": 1,
            "agent_id": "agent-publication",
            "agent_name": "publication-agent",
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
    return build_result_publication(
        ResultTables(
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
    )


class RecordingLeases:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    def acquire(self, path: str, owner: dict[str, str]) -> None:
        self.calls.append(("acquire", path, owner))

    def release(self, path: str, owner: dict[str, str]) -> None:
        self.calls.append(("release", path, owner))


class RecordingDatasetApi:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.generations = {"org/results": 1, "org/index": 1}
        self.files: dict[str, dict[str, bytes]] = {
            "org/results": {},
            "org/index": {},
        }
        self.calls: list[dict[str, object]] = []

    def head(self, repo_id: str) -> str:
        return f"{self.generations[repo_id]:040x}"

    def repo_info(self, repo_id: str, **kwargs: object) -> object:
        self.calls.append({"method": "repo_info", "repo": repo_id, **kwargs})
        return SimpleNamespace(sha=self.head(repo_id))

    def get_paths_info(
        self, repo_id: str, paths: str | list[str], **kwargs: object
    ) -> list[object]:
        path = paths if isinstance(paths, str) else paths[0]
        self.calls.append(
            {"method": "get_paths_info", "repo": repo_id, "path": path, **kwargs}
        )
        return [SimpleNamespace(path=path)] if path in self.files[repo_id] else []

    def list_repo_files(self, repo_id: str, **kwargs: object) -> list[str]:
        self.calls.append({"method": "list_repo_files", "repo": repo_id, **kwargs})
        return list(self.files[repo_id])

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: object) -> str:
        self.calls.append(
            {"method": "hf_hub_download", "repo": repo_id, "path": filename, **kwargs}
        )
        destination = self.root / repo_id.replace("/", "-") / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[repo_id][filename])
        return str(destination)

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object:
        recorded_operations: list[dict[str, str]] = []
        for operation in operations:
            assert isinstance(operation, CommitOperationAdd)
            assert isinstance(operation.path_or_fileobj, bytes)
            self.files[repo_id][operation.path_in_repo] = operation.path_or_fileobj
            recorded_operations.append(
                {
                    "path": operation.path_in_repo,
                    "sha256": _sha256(operation.path_or_fileobj),
                }
            )
        self.calls.append(
            {
                "method": "create_commit",
                "repo": repo_id,
                "operations": recorded_operations,
                **kwargs,
            }
        )
        self.generations[repo_id] += 1
        return SimpleNamespace(oid=self.head(repo_id))


def test_publication_and_idempotent_adoption_have_complete_side_effect_logs(
    tmp_path: Path,
) -> None:
    publication = _publication()
    api = RecordingDatasetApi(tmp_path)
    leases = RecordingLeases()
    publisher = HubDatasetPublisher(
        publisher_id="publisher-mutation", leases=leases, api=api, clock=lambda: NOW
    )

    first = publisher.publish(
        publication, result_dataset="org/results", index_dataset="org/index"
    )
    second = publisher.publish(
        publication, result_dataset="org/results", index_dataset="org/index"
    )
    corpus = {
        "first": first.model_dump(mode="json"),
        "second": second.model_dump(mode="json"),
        "leases": leases.calls,
        "api": api.calls,
        "files": {
            repo: {path: _sha256(value) for path, value in sorted(files.items())}
            for repo, files in sorted(api.files.items())
        },
    }

    assert _hash(corpus) == (
        "a4ea3ad5740ff047dcabc191cc936b709cec2092fc1889f5413090490cc7d38c"
    )


def test_publisher_releases_lease_after_unrecoverable_commit_error(
    tmp_path: Path,
) -> None:
    class FailingApi(RecordingDatasetApi):
        def create_commit(
            self, repo_id: str, operations: list[object], **kwargs: object
        ) -> object:
            request = httpx.Request("POST", "https://huggingface.co/api/datasets")
            raise HfHubHTTPError(
                "commit failed",
                response=httpx.Response(500, request=request),
            )

    leases = RecordingLeases()
    publisher = HubDatasetPublisher(
        publisher_id="publisher-mutation",
        leases=leases,
        api=FailingApi(tmp_path),
        clock=lambda: NOW,
    )

    with pytest.raises(HfHubHTTPError):
        publisher.publish(
            _publication(), result_dataset="org/results", index_dataset="org/index"
        )

    assert leases.calls == [
        (
            "acquire",
            "coordination/publishers/eb1ce4ea9e1b25394e2cff859f3d086d3afab316fc7519d7b3f901940d22e697.json",
            {
                "publisher_id": "publisher-mutation",
                "destination": "org/results",
                "expires_at": "2026-07-14T01:17:03+00:00",
            },
        ),
        (
            "release",
            "coordination/publishers/eb1ce4ea9e1b25394e2cff859f3d086d3afab316fc7519d7b3f901940d22e697.json",
            {
                "publisher_id": "publisher-mutation",
                "destination": "org/results",
                "expires_at": "2026-07-14T01:17:03+00:00",
            },
        ),
    ]


def test_evidence_serialization_and_checksum_verification_are_complete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    write_json(root / "nested" / "summary.json", {"z": 1, "a": [2, 3]})
    (root / "artifact.bin").write_bytes(b"artifact payload")
    append_event(root / "events.jsonl", "completed", run_id="run-one", count=2)
    event = json.loads((root / "events.jsonl").read_text())
    event["at"] = "<timestamp>"
    checksums = write_checksums(root)
    observed = verify_checksums(root)
    manifest = json.loads((root / "checksums.json").read_text())
    for value in (checksums, observed, manifest):
        value["events.jsonl"] = "<event-digest>"
    corpus = {
        "summary": (root / "nested" / "summary.json").read_text(),
        "event": event,
        "checksums": checksums,
        "manifest": manifest,
        "observed": observed,
    }

    assert _hash(corpus) == (
        "dfa12fcd44ac9ffa2f130bccc6c09fe63b0ac0dc73ada785e4974d040e081b4f"
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "evidence has no valid checksum manifest"),
        ("invalid-json", "evidence has no valid checksum manifest"),
        ("list", "evidence checksum manifest is malformed"),
        ("bad-path", "evidence checksum manifest does not cover exact contents"),
        ("bad-digest", "evidence checksum manifest is malformed"),
        ("incomplete", "evidence checksum manifest does not cover exact contents"),
        ("extra", "evidence checksum manifest does not cover exact contents"),
        ("tampered", "evidence checksum mismatch: artifact.txt"),
    ],
)
def test_checksum_rejection_matrix_has_exact_errors(
    tmp_path: Path, case: str, message: str
) -> None:
    root = tmp_path / case
    root.mkdir()
    artifact = root / "artifact.txt"
    artifact.write_text("original", encoding="utf-8")
    write_checksums(root)
    manifest = root / "checksums.json"
    if case == "missing":
        manifest.unlink()
    elif case == "invalid-json":
        manifest.write_text("not-json", encoding="utf-8")
    elif case == "list":
        write_json(manifest, [])
    elif case == "bad-path":
        write_json(manifest, {1: _sha256(b"original")})
    elif case == "bad-digest":
        write_json(manifest, {"artifact.txt": "nope"})
    elif case == "incomplete":
        write_json(manifest, {})
    elif case == "extra":
        write_json(
            manifest,
            {
                "artifact.txt": _sha256(b"original"),
                "missing.txt": _sha256(b"missing"),
            },
        )
    else:
        artifact.write_text("tampered", encoding="utf-8")

    with pytest.raises(RuntimeError) as captured:
        verify_checksums(root)

    assert str(captured.value) == message


@pytest.mark.parametrize(
    ("case", "error_type", "message"),
    [
        ("result-receipt", PublicationConflict, "result publication receipt conflicts"),
        (
            "result-missing",
            DatasetPublicationError,
            "result publication receipt is incomplete",
        ),
        (
            "result-corrupt",
            DatasetPublicationError,
            "published result file is corrupted",
        ),
        (
            "index-invalid",
            DatasetPublicationError,
            "global index receipt is invalid",
        ),
        ("index-publication", PublicationConflict, "global index receipt conflicts"),
        ("index-dataset", PublicationConflict, "global index receipt conflicts"),
        (
            "index-missing",
            DatasetPublicationError,
            "global index receipt is incomplete",
        ),
        (
            "index-corrupt",
            DatasetPublicationError,
            "global index file is corrupted",
        ),
    ],
)
def test_publication_adoption_rejection_matrix_has_exact_errors(
    tmp_path: Path,
    case: str,
    error_type: type[Exception],
    message: str,
) -> None:
    publication = _publication()
    api = RecordingDatasetApi(tmp_path)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-mutation", leases=RecordingLeases(), api=api
    )
    publisher.publish(
        publication, result_dataset="org/results", index_dataset="org/index"
    )
    index_receipt_path = f"publications/{publication.tables.publication_id}.json"
    if case == "result-receipt":
        api.files["org/results"][publication.receipt_path] = b"different\n"
    elif case == "result-missing":
        del api.files["org/results"][publication.files[0].path]
    elif case == "result-corrupt":
        api.files["org/results"][publication.files[0].path] = b"corrupt"
    elif case == "index-invalid":
        api.files["org/index"][index_receipt_path] = b"not-json"
    else:
        receipt = json.loads(api.files["org/index"][index_receipt_path])
        if case == "index-publication":
            receipt["publication_id"] = "pub-different"
        elif case == "index-dataset":
            receipt["result_dataset"] = "org/different"
        elif case == "index-missing":
            del api.files["org/index"][receipt["index_path"]]
        else:
            api.files["org/index"][receipt["index_path"]] = b"corrupt"
        api.files["org/index"][index_receipt_path] = (
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

    with pytest.raises(error_type) as captured:
        publisher.publish(
            publication, result_dataset="org/results", index_dataset="org/index"
        )

    assert str(captured.value) == message


def test_publication_rejects_missing_dataset_commit_identity(tmp_path: Path) -> None:
    class MissingHeadApi(RecordingDatasetApi):
        def repo_info(self, repo_id: str, **kwargs: object) -> object:
            return SimpleNamespace(sha=None)

    publisher = HubDatasetPublisher(
        publisher_id="publisher-mutation",
        leases=RecordingLeases(),
        api=MissingHeadApi(tmp_path),
    )

    with pytest.raises(DatasetPublicationError) as captured:
        publisher.publish(
            _publication(), result_dataset="org/results", index_dataset="org/index"
        )

    assert str(captured.value) == "Dataset has no commit identity"
