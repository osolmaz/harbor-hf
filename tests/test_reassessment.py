from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from harbor_hf.reassessment import (
    ReassessmentError,
    ReassessmentPlan,
    ReassessmentTrial,
    _assert_secrets_absent,
    _checksums,
    _prepare_verifier_tests,
    _publish_success,
    _retain_failed_attempt,
    _reward,
    _task_config,
    _write_fixed_zero,
    reassessment_plan_digest,
)


def _plan_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "harbor-hf/reassessment-plan/v1",
        "reassessment_id": "reassessment-test",
        "created_at": datetime(2026, 7, 23, tzinfo=UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": {
            "campaign_id": "campaign",
            "run_id": "run",
            "publication_id": "publication",
            "source_checksum": "sha256:" + "a" * 64,
            "result_revision": "b" * 40,
            "index_revision": "c" * 40,
        },
        "judge": {
            "provider": "openai-api",
            "api_url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-5.6-luna",
            "reasoning_effort": "xhigh",
            "strip_temperature": True,
            "api_key_secret_name": "OPENAI_API_KEY",
        },
        "verifier_judge_timeout_seconds": 900,
        "harbor_hf_revision": "d" * 40,
        "benchmark_repository": "ShellBench/public-tasks",
        "benchmark_revision": "e" * 40,
        "runtime_image": "hf.co/spaces/example/runtime",
        "output_prefix": "reassessments/test",
        "judge_policy": {
            "workspace_root": "/app",
            "workspace_max_nodes": 1000,
            "workspace_max_file_bytes": 1048576,
            "workspace_max_total_bytes": 8388608,
            "workspace_max_archive_bytes": 8388608,
            "workspace_capture_timeout_seconds": 60,
            "judge_max_request_bytes": 1048576,
            "judge_max_response_bytes": 1048576,
            "judge_timeout_seconds": 300,
            "judge_max_calls_per_execution": 4,
        },
        "trials": [
            {
                "trial_id": "trial-" + "1" * 24,
                "task_name": "task",
                "task_digest": "sha256:" + "2" * 64,
                "logical_attempt": 1,
                "source_execution_id": "exec-" + "3" * 32,
                "source_trial_path": "runs/run/trials/trial-" + "1" * 24,
                "source_outcome": "scored",
                "source_reward": 1.0,
                "action": "rejudge",
            }
        ],
    }
    payload["plan_digest"] = reassessment_plan_digest(payload)
    return payload


def test_plan_digest_and_identity_are_fail_closed() -> None:
    payload = _plan_payload()
    plan = ReassessmentPlan.model_validate_json(json.dumps(payload))
    assert plan.judge.model == "gpt-5.6-luna"
    payload["output_prefix"] = "reassessments/tampered"
    with pytest.raises(ValueError, match="digest mismatch"):
        ReassessmentPlan.model_validate_json(json.dumps(payload))


def test_fixed_zero_requires_agent_failure() -> None:
    base = _plan_payload()["trials"]
    assert isinstance(base, list)
    first = base[0]
    assert isinstance(first, dict)
    trial = dict(first)
    trial.update(source_outcome="agent_failed", action="fixed_zero", source_reward=0.0)
    parsed = ReassessmentTrial.model_validate(trial)
    assert parsed.action == "fixed_zero"
    trial["source_reward"] = 1.0
    with pytest.raises(ValueError, match="nonzero"):
        ReassessmentTrial.model_validate(trial)


def test_reward_prefers_strict_score_and_bounds_values(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "agent_judge_results.json").write_text(
        json.dumps({"normal_plus_safety_score": 1.0, "reward": 0.0})
    )
    assert _reward(verifier) == 1.0
    (verifier / "agent_judge_results.json").write_text(
        json.dumps({"normal_plus_safety_score": 2.0})
    )
    with pytest.raises(ReassessmentError, match="bounded reward"):
        _reward(verifier)
    (verifier / "agent_judge_results.json").write_text(
        json.dumps({"status": "infra_error", "reward": 0.0})
    )
    with pytest.raises(ReassessmentError, match="infrastructure error"):
        _reward(verifier)


def test_secret_scan_and_checksums_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "safe.txt").write_text("safe")
    checksums = _checksums(tmp_path)
    assert checksums == {
        "safe.txt": "sha256:"
        + "8b3369944dd2a3fab39e32d1aeb1f763946a458ae3e6368a46432adc8f3a0860"
    }
    _assert_secrets_absent(tmp_path, ("secret",))
    (tmp_path / "unsafe.txt").write_text("contains-secret")
    with pytest.raises(ReassessmentError, match="known secret"):
        _assert_secrets_absent(tmp_path, ("secret",))


def test_verifier_timeout_transform_is_recorded(tmp_path: Path) -> None:
    task = tmp_path / "task"
    tests = task / "tests"
    tests.mkdir(parents=True)
    (tests / "judge.py").write_text(
        "urllib.request.urlopen(request, timeout=120)\n"
        "urllib.request.urlopen(local_request, timeout=5)\n"
    )
    transformed, metadata = _prepare_verifier_tests(task, 900, "a" * 40)
    try:
        content = (transformed / "judge.py").read_text()
        assert "timeout=900" in content
        assert "timeout=5" in content
        assert metadata["timeout_replacement_count"] == 1
        assert metadata["source_tree_digest"] != metadata["effective_tree_digest"]
    finally:
        shutil.rmtree(transformed.parent)


def test_failed_attempt_is_preserved_before_retry_success(tmp_path: Path) -> None:
    final = tmp_path / "trial"
    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "recorder.json").write_text(
        '{"rejected_error_types":["TrialEvidenceError"]}\n'
    )
    _retain_failed_attempt(
        staging=failed,
        final=final,
        execution_id="rejudge-" + "1" * 32,
        error=ReassessmentError("unsafe detail"),
        known_secrets=("secret",),
    )
    attempt = final / "attempts" / ("rejudge-" + "1" * 32)
    assert (attempt / "_FAILED").is_file()
    assert "unsafe detail" not in (attempt / "failure.json").read_text()

    success = tmp_path / "success"
    success.mkdir()
    (success / "result.json").write_text("{}\n")
    (success / "_SUCCESS").write_text("")
    _publish_success(success, final)
    assert (final / "_SUCCESS").is_file()
    assert attempt.is_dir()


def test_task_config_uses_harbor_default_verifier_command(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    task = tasks / "task"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        '[environment]\ndocker_image = "hf.co/spaces/example/runtime"\n'
        "[verifier]\ntimeout_sec = 600\n"
    )
    raw_trials = _plan_payload()["trials"]
    assert isinstance(raw_trials, list)
    first = raw_trials[0]
    assert isinstance(first, dict)
    trial = ReassessmentTrial.model_validate(first)
    _, image, command = _task_config(tasks, trial)
    assert image == "hf.co/spaces/example/runtime"
    assert command == "bash tests/test.sh"


def test_write_fixed_zero_is_append_only(tmp_path: Path) -> None:
    payload = _plan_payload()
    raw_trials = payload["trials"]
    assert isinstance(raw_trials, list)
    first = raw_trials[0]
    assert isinstance(first, dict)
    raw_trial = dict(first)
    raw_trial.update(
        source_outcome="agent_failed", action="fixed_zero", source_reward=0.0
    )
    payload["trials"] = [raw_trial]
    payload.pop("plan_digest")
    payload["plan_digest"] = reassessment_plan_digest(payload)
    plan = ReassessmentPlan.model_validate_json(json.dumps(payload))
    trial = plan.trials[0]
    source = tmp_path / "source"
    source_execution = source / "executions" / trial.source_execution_id
    source_execution.mkdir(parents=True)
    (source_execution / "checksums.json").write_text("{}\n")
    output = tmp_path / "output"

    _write_fixed_zero(output, trial, source, plan)
    final = output / "trials" / trial.trial_id
    assert (final / "_SUCCESS").is_file()
    assert json.loads((final / "result.json").read_text())["reward"] == 0.0
    before = (final / "checksums.json").read_bytes()
    _write_fixed_zero(output, trial, source, plan)
    assert (final / "checksums.json").read_bytes() == before
