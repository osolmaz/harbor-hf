from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import harbor_hf.trial_evidence as trial_evidence
from harbor_hf.judge_recorder import JudgeEvidenceRecorder
from harbor_hf.models import TrialEvidencePolicy
from harbor_hf.trial_evidence import (
    TrialEvidenceError,
    assemble_trial_evidence,
    package_workspace,
    restore_workspace,
    verify_trial_evidence,
    verify_workspace_package,
    write_trial_evidence_schemas,
)


def _policy(**changes: int) -> TrialEvidencePolicy:
    values = {
        "workspace_root": "/app",
        "workspace_max_nodes": 1000,
        "workspace_max_file_bytes": 1024 * 1024,
        "workspace_max_total_bytes": 8 * 1024 * 1024,
        "workspace_max_archive_bytes": 8 * 1024 * 1024,
        "workspace_capture_timeout_seconds": 60,
        "judge_max_request_bytes": 1024 * 1024,
        "judge_max_response_bytes": 1024 * 1024,
        "judge_max_calls_per_execution": 4,
    }
    values.update(changes)
    return TrialEvidencePolicy.model_validate(values)


def _trial(tmp_path: Path, *, judge: bool = False) -> Path:
    root = tmp_path / "trial"
    workspace = root / "artifacts" / "workspace" / "app"
    workspace.mkdir(parents=True)
    (workspace / "output").mkdir()
    (workspace / "output" / "answer.txt").write_text("answer\n")
    (workspace / "empty").mkdir()
    os.symlink("output/answer.txt", workspace / "answer-link")
    agent = root / "agent"
    agent.mkdir()
    (agent / "session.jsonl").write_text('{"role":"user"}\n')
    (agent / "trajectory.jsonl").write_text('{"event":"done"}\n')
    (agent / "agent.txt").write_text("done\n")
    verifier = root / "verifier"
    verifier.mkdir()
    (verifier / "scorecard.json").write_text('{"passed":true}\n')
    (verifier / "reward.txt").write_text("1\n")
    (verifier / "test-stdout.txt").write_text("ok\n")
    if judge:
        exchange = root / "evidence" / "judge" / "judge-0001"
        exchange.mkdir(parents=True)
        (exchange / "exchange.json").write_text('{"exchange_id":"judge-0001"}\n')
    return root


def test_workspace_package_is_deterministic_and_restorable(tmp_path: Path) -> None:
    source = tmp_path / "app"
    source.mkdir()
    (source / "a").mkdir()
    (source / "a" / "x.txt").write_text("x")
    os.symlink("a/x.txt", source / "link")
    policy = _policy()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = package_workspace(source, first_root / "evidence", policy=policy)
    second = package_workspace(source, second_root / "evidence", policy=policy)
    assert first.evidence.archive.sha256 == second.evidence.archive.sha256
    assert first.evidence.file_index.sha256 == second.evidence.file_index.sha256
    restored = tmp_path / "restored"
    restore_workspace(first_root, first.evidence, restored)
    assert (restored / "app" / "a" / "x.txt").read_text() == "x"
    assert os.readlink(restored / "app" / "link") == "a/x.txt"
    verify_workspace_package(first_root, first.evidence, deep=True)


@pytest.mark.parametrize("target", ["../outside", "missing", "loop"])
def test_workspace_rejects_unsafe_symlink_targets(tmp_path: Path, target: str) -> None:
    source = tmp_path / "app"
    source.mkdir()
    (tmp_path / "outside").write_text("outside")
    os.symlink(target, source / "link")
    if target == "loop":
        os.unlink(source / "link")
        os.symlink("link", source / "link")
    root = tmp_path / "trial"
    root.mkdir()
    with pytest.raises(TrialEvidenceError, match="symlink"):
        package_workspace(source, root / "evidence", policy=_policy())


def test_workspace_materializes_hardlinks_as_regular_files(tmp_path: Path) -> None:
    source = tmp_path / "app"
    source.mkdir()
    (source / "first").write_text("value")
    os.link(source / "first", source / "second")
    root = tmp_path / "trial"
    root.mkdir()
    package = package_workspace(source, root / "evidence", policy=_policy())
    files = [entry for entry in package.entries if entry.type == "file"]
    assert [entry.path for entry in files] == ["first", "second"]
    assert package.evidence.regular_file_bytes == 10
    restored = tmp_path / "restored"
    restore_workspace(root, package.evidence, restored)
    assert (restored / "app" / "first").read_text() == "value"
    assert (restored / "app" / "second").read_text() == "value"
    assert (restored / "app" / "first").stat().st_ino != (
        restored / "app" / "second"
    ).stat().st_ino


def test_workspace_limits_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "app"
    source.mkdir()
    (source / "large").write_bytes(b"xx")
    root = tmp_path / "trial"
    root.mkdir()
    with pytest.raises(TrialEvidenceError, match="byte limit"):
        package_workspace(
            source,
            root / "evidence",
            policy=_policy(workspace_max_total_bytes=1, workspace_max_file_bytes=1),
        )


def test_workspace_capture_timeout_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "app"
    source.mkdir()
    (source / "answer.txt").write_text("answer")
    root = tmp_path / "trial"
    root.mkdir()
    times = iter([0.0, 2.0])
    monkeypatch.setattr(trial_evidence, "monotonic", lambda: next(times))
    with pytest.raises(TrialEvidenceError, match="timeout"):
        package_workspace(
            source,
            root / "evidence",
            policy=_policy(workspace_capture_timeout_seconds=1),
        )


def test_special_workspace_file_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "app"
    source.mkdir()
    os.mkfifo(source / "fifo")
    root = tmp_path / "trial"
    root.mkdir()
    with pytest.raises(TrialEvidenceError, match="special file"):
        package_workspace(source, root / "evidence", policy=_policy())


def test_assemble_rejects_capability_in_any_trial_file(tmp_path: Path) -> None:
    root = _trial(tmp_path)
    capability = "opaque-execution-capability"
    (root / "agent" / "agent.txt").write_text(f"route={capability}\n")
    with pytest.raises(TrialEvidenceError, match="known secret"):
        assemble_trial_evidence(
            root,
            campaign_id="campaign",
            run_id="run",
            execution_id="execution",
            trial_id="trial",
            task_name="task",
            task_digest="sha256:" + "a" * 64,
            logical_attempt=1,
            physical_attempt=1,
            judge_expected=False,
            judge_model=None,
            policy=_policy(),
            known_secrets=(capability,),
        )


def test_assemble_and_deep_verify_trial(tmp_path: Path) -> None:
    root = _trial(tmp_path)
    manifest = assemble_trial_evidence(
        root,
        campaign_id="campaign",
        run_id="run",
        execution_id="execution",
        trial_id="trial",
        task_name="task",
        task_digest="sha256:" + "a" * 64,
        logical_attempt=2,
        physical_attempt=1,
        judge_expected=False,
        judge_model=None,
        policy=_policy(),
    )
    assert manifest.completion.status == "complete"
    assert not (root / "artifacts" / "workspace" / "app").exists()
    assert verify_trial_evidence(root, deep=True) == manifest


def test_assemble_requires_judge_recorder_summary(tmp_path: Path) -> None:
    root = _trial(tmp_path)
    with pytest.raises(TrialEvidenceError, match="summary is invalid"):
        assemble_trial_evidence(
            root,
            campaign_id=None,
            run_id="run",
            execution_id="execution",
            trial_id="trial",
            task_name="task",
            task_digest="sha256:" + "a" * 64,
            logical_attempt=1,
            physical_attempt=1,
            judge_expected=True,
            judge_model="judge/model",
            policy=_policy(),
        )


def test_assemble_rejects_judge_recorder_with_rejected_calls(tmp_path: Path) -> None:
    root = _trial(tmp_path)
    records = tmp_path / "judge-records"
    recorder = JudgeEvidenceRecorder(token="token")
    capability = recorder.register_scope(
        execution_id="execution",
        trial_id="trial",
        model="judge/model",
        destination=records,
        policy=_policy(),
    )
    recorder.revoke_scope(capability)
    recorder.close()
    summary_path = records / "recorder.json"
    summary = json.loads(summary_path.read_text())
    summary["rejected_call_count"] = 1
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(TrialEvidenceError, match="rejected one or more calls"):
        assemble_trial_evidence(
            root,
            campaign_id=None,
            run_id="run",
            execution_id="execution",
            trial_id="trial",
            task_name="task",
            task_digest="sha256:" + "a" * 64,
            logical_attempt=1,
            physical_attempt=1,
            judge_expected=True,
            judge_model="judge/model",
            policy=_policy(),
            judge_records_dir=records,
        )


def test_assemble_accepts_judge_recorder_with_zero_calls(tmp_path: Path) -> None:
    root = _trial(tmp_path)
    records = tmp_path / "judge-records"
    recorder = JudgeEvidenceRecorder(token="token")
    capability = recorder.register_scope(
        execution_id="execution",
        trial_id="trial",
        model="judge/model",
        destination=records,
        policy=_policy(),
    )
    recorder.revoke_scope(capability)
    recorder.close()
    manifest = assemble_trial_evidence(
        root,
        campaign_id=None,
        run_id="run",
        execution_id="execution",
        trial_id="trial",
        task_name="task",
        task_digest="sha256:" + "a" * 64,
        logical_attempt=1,
        physical_attempt=1,
        judge_expected=True,
        judge_model="judge/model",
        policy=_policy(),
        judge_records_dir=records,
    )
    assert manifest.judge.expected is True
    assert manifest.judge.exchanges == []
    assert manifest.judge.recorder_summary is not None
    assert manifest.verifier.judge_selection is None
    assert verify_trial_evidence(root, deep=True) == manifest


def test_schema_files_are_strict_json(tmp_path: Path) -> None:
    write_trial_evidence_schemas(tmp_path)
    names = {path.name for path in tmp_path.iterdir()}
    assert names == {
        "judge-recorder-summary-v1.schema.json",
        "judge-selection-v1.schema.json",
        "trial-evidence-v1.schema.json",
        "workspace-file-v1.schema.json",
    }
    for path in tmp_path.iterdir():
        assert isinstance(json.loads(path.read_text()), dict)
