from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harbor_hf.private_artifacts import (
    PrivateArtifactEntry,
    PrivateArtifactManifest,
    build_private_artifact_manifest,
    write_private_artifact_manifest,
)
from harbor_hf.results import ArtifactEvidence


def _execution_root(
    root: Path, *, started: bool = True, with_session: bool = True
) -> Path:
    (root / "execution.lock.json").write_text(
        json.dumps({"execution_id": "exec-one", "trial_id": "trial-one"}) + "\n",
        encoding="utf-8",
    )
    (root / "events.jsonl").write_text("{}\n", encoding="utf-8")
    trial = root / "harbor-jobs" / "job-one" / "trial-one"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "agent_info": {"name": "openclaw"},
                "agent_execution": {
                    "started_at": "2026-07-14T00:00:00Z" if started else None
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (agent / "openclaw.txt").write_text("agent output\n", encoding="utf-8")
    (agent / "trajectory.json").write_text("{}\n", encoding="utf-8")
    (trial / "verifier").mkdir()
    (trial / "verifier" / "reward.txt").write_text("1\n", encoding="utf-8")
    if with_session:
        sessions = agent / "openclaw-sessions"
        sessions.mkdir()
        (sessions / "session-one.jsonl").write_text("{}\n", encoding="utf-8")
    return root


def test_private_manifest_classifies_and_checksums_complete_execution(
    tmp_path: Path,
) -> None:
    root = _execution_root(tmp_path)

    manifest = write_private_artifact_manifest(root, strict_session=True)
    written = PrivateArtifactManifest.model_validate_json(
        (root / "private-artifacts.json").read_text(encoding="utf-8")
    )

    assert written == manifest
    assert manifest.requirements[0].required is True
    assert manifest.requirements[0].satisfied is True
    kinds = {entry.path: entry.kind for entry in manifest.entries}
    assert (
        kinds["harbor-jobs/job-one/trial-one/agent/openclaw-sessions/session-one.jsonl"]
        == "session"
    )
    assert kinds["harbor-jobs/job-one/trial-one/agent/trajectory.json"] == (
        "trajectory"
    )
    assert kinds["harbor-jobs/job-one/trial-one/verifier/reward.txt"] == "verifier"
    assert "private-artifacts.json" not in kinds
    assert all(entry.classification == "private" for entry in manifest.entries)


def test_successful_openclaw_execution_requires_session_jsonl(tmp_path: Path) -> None:
    root = _execution_root(tmp_path, with_session=False)

    with pytest.raises(RuntimeError, match="no session JSONL"):
        build_private_artifact_manifest(root, strict_session=True)

    retained = build_private_artifact_manifest(root, strict_session=False)
    assert retained.requirements[0].required is True
    assert retained.requirements[0].satisfied is False


def test_session_is_not_required_before_agent_execution_starts(tmp_path: Path) -> None:
    root = _execution_root(tmp_path, started=False, with_session=False)

    manifest = build_private_artifact_manifest(root, strict_session=True)

    assert manifest.requirements[0].required is False
    assert manifest.requirements[0].satisfied is False


def test_multi_step_agent_timing_requires_session_jsonl(tmp_path: Path) -> None:
    root = _execution_root(tmp_path, started=False, with_session=False)
    result_path = root / "harbor-jobs" / "job-one" / "trial-one" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["step_results"] = [
        {"agent_execution": {"started_at": "2026-07-14T00:00:00Z"}}
    ]
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no session JSONL"):
        build_private_artifact_manifest(root, strict_session=True)


def test_private_manifest_enforces_file_and_bundle_size_limits(tmp_path: Path) -> None:
    root = _execution_root(tmp_path)

    with pytest.raises(RuntimeError, match="file size limit"):
        build_private_artifact_manifest(
            root, strict_session=True, max_file_bytes=1, max_bundle_bytes=10_000
        )
    with pytest.raises(RuntimeError, match="bundle exceeds size limit"):
        build_private_artifact_manifest(
            root,
            strict_session=True,
            max_file_bytes=10_000,
            max_bundle_bytes=1,
        )


def test_private_manifest_rejects_symlinks(tmp_path: Path) -> None:
    root = _execution_root(tmp_path)
    (root / "linked").symlink_to(root / "events.jsonl")

    with pytest.raises(RuntimeError, match="cannot contain symlinks"):
        build_private_artifact_manifest(root, strict_session=True)


def test_private_models_reject_unsafe_or_inconsistent_manifests() -> None:
    with pytest.raises(ValidationError, match="safely relative"):
        PrivateArtifactEntry(
            path="../session.jsonl",
            size=1,
            digest="sha256:" + "1" * 64,
            kind="session",
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        PrivateArtifactManifest(
            execution_id="exec-one",
            trial_id="trial-one",
            total_bytes=2,
            entries=[
                PrivateArtifactEntry(
                    path=path,
                    size=1,
                    digest="sha256:" + digest * 64,
                    kind="other",
                )
                for path, digest in (("z", "1"), ("a", "2"))
            ],
            requirements=[],
        )


def test_private_manifest_cannot_cross_public_result_boundary() -> None:
    with pytest.raises(ValidationError, match="noncanonical path"):
        ArtifactEvidence(
            owner_type="execution",
            owner_id="exec-one",
            kind="verification",
            path="private-artifacts.json",
            sha256="sha256:" + "1" * 64,
            media_type="application/json",
            size_bytes=1,
        )
