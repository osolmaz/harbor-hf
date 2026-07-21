from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from harbor_hf.private_artifacts import (
    PrivateArtifactEntry,
    PrivateArtifactManifest,
    build_private_artifact_manifest,
    openclaw_execution_started,
    sanitize_private_artifact_directory_files,
    sanitize_private_artifact_special_files,
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


def test_session_requirement_rejects_over_nested_json(tmp_path: Path) -> None:
    root = _execution_root(tmp_path)
    session = next(root.rglob("session-one.jsonl"))
    session.write_text('{"nested":' * 20_000 + "{}" + "}" * 20_000 + "\n")

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


def test_private_manifest_allows_large_direct_workspace_archive(
    tmp_path: Path,
) -> None:
    root = _execution_root(tmp_path)
    evidence = root / "evidence"
    evidence.mkdir()
    archive = evidence / "workspace.tar.zst"
    archive.write_bytes(b"x" * 2048)

    manifest = build_private_artifact_manifest(
        root,
        strict_session=True,
        max_file_bytes=1024,
        max_bundle_bytes=1024 * 1024,
    )

    assert any(entry.path == "evidence/workspace.tar.zst" for entry in manifest.entries)


def test_private_manifest_rejects_symlinks(tmp_path: Path) -> None:
    root = _execution_root(tmp_path)
    (root / "linked").symlink_to(root / "events.jsonl")

    with pytest.raises(RuntimeError, match="cannot contain symlinks"):
        build_private_artifact_manifest(root, strict_session=True)


def test_private_artifact_sanitizer_rejects_special_files(tmp_path: Path) -> None:
    root = _execution_root(tmp_path)
    fifo = root / "runtime.pipe"
    os.mkfifo(fifo)

    with pytest.raises(RuntimeError, match="unsupported file type: runtime.pipe"):
        build_private_artifact_manifest(root, strict_session=True)

    rejected = sanitize_private_artifact_tree(root)

    assert ("runtime.pipe", "special_file") in {
        (item.path, item.reason) for item in rejected
    }
    assert not fifo.exists()


def test_special_file_sanitizer_preserves_regular_profile_evidence(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "session.jsonl"
    regular.write_text("{}\n", encoding="utf-8")
    fifo = tmp_path / "runtime.pipe"
    os.mkfifo(fifo)

    rejected = sanitize_private_artifact_special_files(tmp_path)

    assert [(item.path, item.reason) for item in rejected] == [
        ("runtime.pipe", "special_file")
    ]
    assert regular.read_text(encoding="utf-8") == "{}\n"
    assert not fifo.exists()
    assert json.loads(
        (tmp_path / "private-artifact-rejections.json").read_text(encoding="utf-8")
    )["rejections"] == [
        {"path": "runtime.pipe", "reason": "special_file", "size": None}
    ]


def test_private_artifact_sanitizer_records_and_removes_rejected_files(
    tmp_path: Path,
) -> None:
    root = _execution_root(tmp_path)
    (root / "linked").symlink_to(root / "events.jsonl")
    oversized = root / "harbor-jobs" / "oversized.log"
    oversized.write_text("x" * 2000, encoding="utf-8")

    rejected = sanitize_private_artifact_tree(root, max_file_bytes=1024)

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


def test_rejection_record_is_bounded_and_reports_omissions(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    for index in range(20):
        (root / f"linked-{index:02d}").symlink_to(outside)

    rejected = sanitize_private_artifact_tree(root, max_file_bytes=300)

    record_path = root / "private-artifact-rejections.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record_path.stat().st_size <= 300
    assert record["omitted_count"] > 0
    assert record["omitted_count"] + len(record["rejections"]) == 20
    assert [item.model_dump(mode="json") for item in rejected] == record["rejections"]


def test_rejection_record_preserves_omissions_across_trusted_passes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    for index in range(20):
        (root / f"linked-{index:02d}").symlink_to(outside)

    first = sanitize_private_artifact_tree(root, max_file_bytes=300)
    first_record = json.loads(
        (root / "private-artifact-rejections.json").read_text(encoding="utf-8")
    )
    second = sanitize_private_artifact_tree(
        root,
        max_file_bytes=300,
        trust_existing_rejections=True,
    )
    second_record_path = root / "private-artifact-rejections.json"
    second_record = json.loads(second_record_path.read_text(encoding="utf-8"))

    assert second == first
    assert second_record == first_record
    assert second_record["omitted_count"] > 0
    assert second_record_path.stat().st_size <= 300


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

    (job / "job.log").write_text("x" * 400, encoding="utf-8")
    with pytest.raises(RuntimeError, match="file size limit: job.log"):
        validate_private_artifact_directory_files(
            job, max_file_bytes=10, max_bundle_bytes=10
        )

    rejected = sanitize_private_artifact_directory_files(
        job, max_file_bytes=300, max_bundle_bytes=300
    )
    assert [(item.path, item.reason) for item in rejected] == [("job.log", "file_size")]
    assert not (job / "job.log").exists()
    assert (trial / "session.jsonl").stat().st_size == 100


def test_directory_sanitizer_reserves_space_for_its_rejection_record(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "job.log").write_text("x" * 180, encoding="utf-8")
    (job / "oversized.log").write_text("x" * 400, encoding="utf-8")

    first = sanitize_private_artifact_directory_files(
        job,
        max_file_bytes=300,
        max_bundle_bytes=350,
    )
    second = sanitize_private_artifact_directory_files(
        job,
        max_file_bytes=300,
        max_bundle_bytes=350,
        trust_existing_rejections=True,
    )

    direct_files = [candidate for candidate in job.iterdir() if candidate.is_file()]
    assert sum(candidate.stat().st_size for candidate in direct_files) <= 350
    assert (job / "private-artifact-rejections.json").stat().st_size <= 300
    assert all(item.path != "private-artifact-rejections.json" for item in second)
    assert second == first


def test_private_artifact_file_count_is_bounded(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    for name in ("one", "two", "three"):
        (root / name).touch()

    with pytest.raises(RuntimeError, match="file count limit"):
        build_private_artifact_manifest(
            root,
            strict_session=False,
            execution_id="execution-one",
            trial_id="trial-one",
            session_required=False,
            max_file_count=2,
        )

    rejected = sanitize_private_artifact_tree(root, max_file_count=2)
    manifest = write_private_artifact_manifest(
        root,
        strict_session=False,
        execution_id="execution-one",
        trial_id="trial-one",
        session_required=False,
        trust_rejections=True,
        max_file_count=2,
    )

    assert len(manifest.entries) == 2
    assert [(item.path, item.reason) for item in rejected] == [("one", "bundle_size")]


def test_private_artifact_manifest_is_bounded(tmp_path: Path) -> None:
    (tmp_path / "retained").touch()

    with pytest.raises(RuntimeError, match="manifest exceeds file size limit"):
        write_private_artifact_manifest(
            tmp_path,
            strict_session=False,
            execution_id="execution-one",
            trial_id="trial-one",
            session_required=False,
            max_file_bytes=256,
        )


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
