from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harbor_hf.harbor_native_bundle import (
    HARBOR_NATIVE_BUNDLE_PATH,
    HarborNativeBundle,
    NativeBundleError,
    build_harbor_native_bundle,
    write_harbor_native_bundle,
)


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "execution"
    job = root / "harbor-jobs" / "job-one"
    trial = job / "trial-one"
    trial.mkdir(parents=True)
    files = {
        job / "lock.json": b'{"job":"lock"}\n',
        job / "result.json": b'{"job":"result"}\n',
        trial / "lock.json": b'{"trial":"lock"}\n',
        trial / "result.json": b'{"trial":"result"}\n',
    }
    for path, content in files.items():
        path.write_bytes(content)
    (root / "artifacts.tar.gz").write_bytes(b"deterministic archive")
    compatibility = {
        "schema_version": "harbor-hf/harbor-compatibility/v1alpha3",
        "harbor_revision": "a" * 40,
        "harbor_version": "0.1.0",
        "request_digest": "sha256:" + "b" * 64,
        "jobs": [
            {
                "path": "job-one",
                "lock_digest": _digest(files[job / "lock.json"]),
                "result_digest": _digest(files[job / "result.json"]),
                "total_trials": 1,
                "completed_trials": 1,
                "errored_trials": 0,
            }
        ],
        "trials": [
            {
                "path": "job-one/trial-one",
                "trial_id": "trial-one",
                "trial_name": "trial-one",
                "lock_digest": _digest(files[trial / "lock.json"]),
                "result_digest": _digest(files[trial / "result.json"]),
                "task_name": "task-one",
                "task_digest": "sha256:" + "c" * 64,
                "agent_name": "openclaw",
                "agent_version": "1.0.0",
                "model_provider": "openai",
                "model_name": "model",
                "timing": {
                    "trial": {"started_at": None, "finished_at": None},
                    "steps": [],
                },
                "usage": {},
                "artifacts": [],
            }
        ],
    }
    (root / "harbor-compatibility.json").write_text(
        json.dumps(compatibility), encoding="utf-8"
    )
    return root


def test_builds_minimal_deterministic_native_manifest(tmp_path: Path) -> None:
    root = _root(tmp_path)

    first = write_harbor_native_bundle(root, required=True)
    first_bytes = (root / HARBOR_NATIVE_BUNDLE_PATH).read_bytes()
    second = write_harbor_native_bundle(root, required=True)

    assert first == second
    assert (root / HARBOR_NATIVE_BUNDLE_PATH).read_bytes() == first_bytes
    assert first is not None
    assert first.contract_status == "compatibility"
    assert [document.kind for document in first.documents] == [
        "job_lock",
        "job_result",
        "trial_lock",
        "trial_result",
    ]
    assert "task_name" not in first_bytes.decode()
    assert "reward" not in first_bytes.decode()
    assert HarborNativeBundle.model_validate_json(first_bytes) == first


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("document", "native Harbor document digest differs"),
        ("archive", "native bundle path is not a file"),
        ("compatibility", "Harbor compatibility export is invalid"),
    ],
)
def test_rejects_missing_or_tampered_inputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root = _root(tmp_path)
    if mutation == "document":
        (root / "harbor-jobs/job-one/result.json").write_text(
            "tampered", encoding="utf-8"
        )
    elif mutation == "archive":
        (root / "artifacts.tar.gz").unlink()
    else:
        (root / "harbor-compatibility.json").write_text("[]", encoding="utf-8")

    with pytest.raises(NativeBundleError, match=message):
        build_harbor_native_bundle(root)


def test_rejects_symlinked_native_document(tmp_path: Path) -> None:
    root = _root(tmp_path)
    result = root / "harbor-jobs/job-one/result.json"
    target = tmp_path / "result.json"
    target.write_bytes(result.read_bytes())
    result.unlink()
    result.symlink_to(target)

    with pytest.raises(NativeBundleError, match="symlink"):
        build_harbor_native_bundle(root)


def test_failed_execution_can_omit_invalid_bundle(tmp_path: Path) -> None:
    root = tmp_path / "failed"
    root.mkdir()

    assert write_harbor_native_bundle(root, required=False) is None
    assert not (root / HARBOR_NATIVE_BUNDLE_PATH).exists()
    with pytest.raises(NativeBundleError):
        write_harbor_native_bundle(root, required=True)


def test_manifest_rejects_duplicate_documents(tmp_path: Path) -> None:
    value = build_harbor_native_bundle(_root(tmp_path)).model_dump(mode="json")
    value["documents"].append(value["documents"][0])

    with pytest.raises(ValidationError, match="duplicate documents"):
        HarborNativeBundle.model_validate(value)
