from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harbor_hf.private_artifacts import (
    PrivateArtifactEntry,
    PrivateArtifactManifest,
    build_private_artifact_manifest,
    openclaw_execution_started,
    sanitize_private_artifact_directory_files,
    sanitize_private_artifact_tree,
    validate_private_artifact_directory_files,
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


@pytest.mark.parametrize("content", ["", "{}\n{", '"not-an-object"\n'])
def test_session_requirement_rejects_unusable_jsonl(
    tmp_path: Path, content: str
) -> None:
    root = _execution_root(tmp_path)
    session = next(root.rglob("session-one.jsonl"))
    session.write_text(content, encoding="utf-8")

    with pytest.raises(RuntimeError, match="no session JSONL"):
        build_private_artifact_manifest(root, strict_session=True)

    retained = build_private_artifact_manifest(root, strict_session=False)
    assert retained.requirements[0].satisfied is False


def test_trajectory_sidecar_does_not_satisfy_session_requirement(
    tmp_path: Path,
) -> None:
    root = _execution_root(tmp_path, with_session=False)
    sessions = (
        root / "harbor-jobs" / "job-one" / "trial-one" / "agent" / "openclaw-sessions"
    )
    sessions.mkdir()
    sidecar = sessions / "session-one.trajectory.jsonl"
    sidecar.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no session JSONL"):
        build_private_artifact_manifest(root, strict_session=True)

    retained = build_private_artifact_manifest(root, strict_session=False)
    kinds = {entry.path: entry.kind for entry in retained.entries}
    assert kinds[str(sidecar.relative_to(root))] == "trajectory"
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


def test_null_step_results_still_requires_session_jsonl(tmp_path: Path) -> None:
    root = _execution_root(tmp_path, with_session=False)
    result_path = root / "harbor-jobs" / "job-one" / "trial-one" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["step_results"] = None
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no session JSONL"):
        build_private_artifact_manifest(root, strict_session=True)


def test_started_harbor_openclaw_request_requires_session_without_result(
    tmp_path: Path,
) -> None:
    root = _execution_root(tmp_path, started=False, with_session=False)
    (root / "harbor-jobs" / "job-one" / "trial-one" / "result.json").unlink()
    (root / "harbor-request.json").write_text(
        json.dumps({"verification": {"expected_agent_name": "openclaw"}}) + "\n",
        encoding="utf-8",
    )
    (root / "events.jsonl").write_text(
        json.dumps({"event": "harbor_started"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no session JSONL"):
        build_private_artifact_manifest(root, strict_session=True)


def test_malformed_result_uses_explicit_attempted_fallback(tmp_path: Path) -> None:
    root = _execution_root(tmp_path, started=False, with_session=False)
    result_path = root / "harbor-jobs" / "job-one" / "trial-one" / "result.json"
    result_path.write_text("{", encoding="utf-8")

    assert openclaw_execution_started(root, fallback_attempted=True) is True


def test_private_manifest_sorts_serialized_relative_paths(tmp_path: Path) -> None:
    root = _execution_root(tmp_path)
    nested = root / "foo" / "bar"
    nested.parent.mkdir()
    nested.write_text("nested\n", encoding="utf-8")
    (root / "foo.txt").write_text("sibling\n", encoding="utf-8")

    manifest = build_private_artifact_manifest(root, strict_session=True)

    paths = [entry.path for entry in manifest.entries]
    assert paths == sorted(paths)
    assert paths.index("foo.txt") < paths.index("foo/bar")


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


def test_private_artifact_sanitizer_records_and_removes_rejected_files(
    tmp_path: Path,
) -> None:
    root = _execution_root(tmp_path)
    (root / "linked").symlink_to(root / "events.jsonl")
    oversized = root / "harbor-jobs" / "oversized.log"
    oversized.write_text("x" * 200, encoding="utf-8")

    rejected = sanitize_private_artifact_tree(root, max_file_bytes=150)

    assert [(item.path, item.reason) for item in rejected] == [
        ("harbor-jobs/oversized.log", "file_size"),
        ("linked", "symlink"),
    ]
    assert not oversized.exists()
    assert not (root / "linked").exists()
    with pytest.raises(RuntimeError, match="controller-reserved"):
        build_private_artifact_manifest(root, strict_session=False)
    retained = build_private_artifact_manifest(
        root, strict_session=False, trust_rejections=True
    )
    assert retained.rejections == rejected


def test_directory_file_limits_do_not_charge_child_trial_bundles(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    trial = job / "trial"
    trial.mkdir(parents=True)
    (job / "job.log").write_text("small", encoding="utf-8")
    (trial / "session.jsonl").write_text("x" * 100, encoding="utf-8")

    validate_private_artifact_directory_files(
        job, max_file_bytes=10, max_bundle_bytes=10
    )

    (job / "job.log").write_text("x" * 20, encoding="utf-8")
    with pytest.raises(RuntimeError, match="file size limit: job.log"):
        validate_private_artifact_directory_files(
            job, max_file_bytes=10, max_bundle_bytes=10
        )

    rejected = sanitize_private_artifact_directory_files(
        job, max_file_bytes=10, max_bundle_bytes=10
    )
    assert [(item.path, item.reason) for item in rejected] == [("job.log", "file_size")]
    assert not (job / "job.log").exists()
    assert (trial / "session.jsonl").stat().st_size == 100


def test_private_manifest_rejects_controller_reserved_paths(tmp_path: Path) -> None:
    root = _execution_root(tmp_path)
    (root / "private-artifacts.json").write_text("x" * 200, encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="controller-reserved path: private-artifacts.json",
    ):
        build_private_artifact_manifest(
            root,
            strict_session=True,
            max_file_bytes=100,
        )


def test_private_artifact_sanitizer_removes_reserved_path_collisions(
    tmp_path: Path,
) -> None:
    root = _execution_root(tmp_path)
    forged_manifest = root / "private-artifacts.json"
    forged_manifest.write_text("forged", encoding="utf-8")
    forged_checksums = root / "checksums.json"
    forged_checksums.mkdir()
    (forged_checksums / "payload").write_text("hidden", encoding="utf-8")

    rejected = sanitize_private_artifact_tree(root)

    assert [(item.path, item.reason, item.size) for item in rejected] == [
        ("checksums.json", "reserved_path", None),
        ("private-artifacts.json", "reserved_path", 6),
    ]
    assert not forged_manifest.exists()
    assert not forged_checksums.exists()
    retained = build_private_artifact_manifest(
        root,
        strict_session=True,
        trust_rejections=True,
    )
    assert retained.rejections == rejected


def test_private_artifact_sanitizer_removes_forged_rejection_symlink_first(
    tmp_path: Path,
) -> None:
    target = tmp_path / "malformed-target"
    target.write_text("not json", encoding="utf-8")
    root = tmp_path / "evidence"
    root.mkdir()
    rejection = root / "private-artifact-rejections.json"
    rejection.symlink_to(target)

    rejected = sanitize_private_artifact_tree(root)

    assert [(item.path, item.reason) for item in rejected] == [
        ("private-artifact-rejections.json", "symlink")
    ]
    assert not rejection.is_symlink()
    assert target.read_text(encoding="utf-8") == "not json"


def test_private_artifact_sanitizer_replaces_untrusted_rejection_record(
    tmp_path: Path,
) -> None:
    rejection = tmp_path / "private-artifact-rejections.json"
    rejection.write_text(
        json.dumps(
            {"rejections": [{"path": "forged", "reason": "symlink", "size": None}]}
        ),
        encoding="utf-8",
    )

    rejected = sanitize_private_artifact_tree(tmp_path)

    assert [(item.path, item.reason) for item in rejected] == [
        ("private-artifact-rejections.json", "reserved_path")
    ]


def test_private_artifact_sanitizer_can_preserve_its_previous_rejections(
    tmp_path: Path,
) -> None:
    linked = tmp_path / "linked"
    target = tmp_path / "target"
    target.write_text("retained", encoding="utf-8")
    linked.symlink_to(target)

    first = sanitize_private_artifact_tree(tmp_path)
    second = sanitize_private_artifact_tree(tmp_path, trust_existing_rejections=True)

    assert second == first


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
