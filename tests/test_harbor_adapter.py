from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harbor_hf.evidence import scrub_secret, scrub_secret_paths
from harbor_hf.harbor_adapter import (
    FilesystemHarborExecutionAdapter,
    HarborExecutionRequest,
    HarborTrialFailure,
    HarborVerificationFailure,
    WorkerError,
    build_execution_request,
    resolve_native_trial_root,
)
from harbor_hf.harbor_adapter.exporter import (
    _openclaw_session_usage,
    _usage_with_openclaw_fallback,
    refresh_bundle_artifacts,
)
from harbor_hf.harbor_adapter.models import HarborCompatibilityBundle, sha256_digest
from harbor_hf.harbor_adapter.validation import validate_compatibility_bundle
from harbor_hf.models import ExperimentSpec
from harbor_hf.runs import RunLock, build_run_lock

GOLDEN_CONTRACT = Path(__file__).parent / "golden" / "harbor-adapter-contract-v1.json"


def test_resolve_native_trial_root_rejects_escaping_paths(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    trial = jobs / "job" / "trial"
    trial.mkdir(parents=True)

    assert resolve_native_trial_root(jobs, "job/trial") == trial
    for value in ("../outside", "/absolute", "job/../trial", "job//trial"):
        with pytest.raises(WorkerError, match="safe relative"):
            resolve_native_trial_root(jobs, value)

    linked = jobs / "linked"
    linked.symlink_to(trial, target_is_directory=True)
    with pytest.raises(WorkerError, match="symbolic link"):
        resolve_native_trial_root(jobs, "linked")


def _session_record(
    *,
    role: str = "assistant",
    input_tokens: object = 10,
    cache_read: object = 4,
    cache_write: object = 2,
    output_tokens: object = 3,
    cost: object = 0.25,
) -> str:
    return json.dumps(
        {
            "message": {
                "role": role,
                "usage": {
                    "input": input_tokens,
                    "cacheRead": cache_read,
                    "cacheWrite": cache_write,
                    "output": output_tokens,
                    "cost": {"total": cost},
                },
            }
        }
    )


def test_openclaw_session_usage_aggregates_raw_assistant_messages(
    tmp_path: Path,
) -> None:
    sessions = tmp_path / "agent" / "openclaw-sessions"
    sessions.mkdir(parents=True)
    (sessions / "one.jsonl").write_text(
        _session_record() + "\n" + _session_record(input_tokens=5, cost=0.5) + "\n",
        encoding="utf-8",
    )

    assert _openclaw_session_usage(tmp_path) == (27, 12, 6, 0.75)


def test_openclaw_session_usage_ignores_invalid_and_trajectory_records(
    tmp_path: Path,
) -> None:
    sessions = tmp_path / "agent" / "openclaw-sessions"
    sessions.mkdir(parents=True)
    (sessions / "one.jsonl").write_text(
        _session_record()
        + "\nnot-json\n"
        + _session_record(role="user")
        + "\n"
        + _session_record(input_tokens=True)
        + "\n"
        + _session_record(cache_write=-1)
        + "\n"
        + _session_record(cost="unknown")
        + "\n",
        encoding="utf-8",
    )
    (sessions / "one.trajectory.jsonl").write_text(
        _session_record(input_tokens=1000) + "\n", encoding="utf-8"
    )

    assert _openclaw_session_usage(tmp_path) == (32, 12, 6, None)


def test_openclaw_session_usage_does_not_duplicate_legacy_session(
    tmp_path: Path,
) -> None:
    sessions = tmp_path / "agent" / "openclaw-sessions"
    sessions.mkdir(parents=True)
    (sessions / "one.jsonl").write_text(_session_record() + "\n", encoding="utf-8")
    (tmp_path / "agent" / "openclaw.session.jsonl").write_text(
        _session_record(input_tokens=1000) + "\n", encoding="utf-8"
    )

    assert _openclaw_session_usage(tmp_path) == (16, 6, 3, 0.25)


def test_openclaw_session_usage_uses_legacy_session_as_fallback(
    tmp_path: Path,
) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "openclaw.session.jsonl").write_text(
        _session_record() + "\n", encoding="utf-8"
    )

    assert _openclaw_session_usage(tmp_path) == (16, 6, 3, 0.25)


def test_openclaw_session_usage_only_fills_missing_native_totals(
    tmp_path: Path,
) -> None:
    sessions = tmp_path / "agent" / "openclaw-sessions"
    sessions.mkdir(parents=True)
    (sessions / "one.jsonl").write_text(_session_record() + "\n", encoding="utf-8")

    assert _usage_with_openclaw_fallback((20, None, 8, None), tmp_path) == (
        20,
        6,
        8,
        0.25,
    )


def _request(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> tuple[RunLock, HarborExecutionRequest]:
    lock = build_run_lock(remote_spec, run_id="adapter-contract")
    request = build_execution_request(
        lock,
        tmp_path / "jobs",
        "https://endpoint.example",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    return lock, request


def _bundle(
    request: HarborExecutionRequest, **trial_updates: object
) -> HarborCompatibilityBundle:
    policy = request.verification
    task_name = next(iter(policy.expected_task_digests or {}))
    trial: dict[str, object] = {
        "path": "job/trial",
        "trial_id": "00000000-0000-0000-0000-000000000001",
        "trial_name": "trial-contract",
        "lock_digest": "sha256:" + "3" * 64,
        "result_digest": "sha256:" + "4" * 64,
        "task_name": task_name,
        "task_digest": (policy.expected_task_digests or {})[task_name],
        "agent_name": policy.expected_agent_name,
        "agent_version": policy.expected_agent_version,
        "model_provider": policy.expected_model_provider,
        "model_name": policy.expected_model_name,
        "exception_type": None,
        "exception_message": None,
        "step_exceptions": [],
        "rewards": {"reward": 1.0},
        "timing": {
            "trial": {"started_at": None, "finished_at": None},
            "environment_setup": None,
            "agent_setup": None,
            "agent_execution": None,
            "verifier": None,
            "steps": [],
        },
        "usage": {
            "input_tokens": 12,
            "cache_tokens": 4,
            "output_tokens": 8,
            "cost_usd": None,
        },
        "artifacts": [
            {
                "path": "result.json",
                "size": 10,
                "digest": "sha256:" + "5" * 64,
                "kind": "result",
                "classification": "private",
            }
        ],
    }
    trial.update(trial_updates)
    request_digest = request.model_dump_json()
    from harbor_hf.harbor_adapter.models import canonical_json_bytes, sha256_digest

    return HarborCompatibilityBundle.model_validate(
        {
            "harbor_revision": request.harbor_revision,
            "harbor_version": "0.17.1",
            "request_digest": sha256_digest(
                canonical_json_bytes(json.loads(request_digest))
            ),
            "jobs": [],
            "trials": [trial],
        }
    )


def test_adapter_prepares_one_immutable_harbor_config(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock, request = _request(remote_spec, tmp_path)
    prepared = FilesystemHarborExecutionAdapter().prepare(
        lock,
        tmp_path,
        tmp_path / "jobs",
        "https://endpoint.example",
        tmp_path / "harbor",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )

    assert prepared.request == request
    assert prepared.config_path.read_bytes() == request.config_bytes()
    assert prepared.request_path.read_bytes() == request.request_bytes()
    assert prepared.command[-3:] == [
        "--config",
        str(prepared.config_path),
        "--yes",
    ]
    with pytest.raises(WorkerError, match="execution input already exists"):
        FilesystemHarborExecutionAdapter().prepare(
            lock,
            tmp_path,
            tmp_path / "jobs",
            "https://endpoint.example",
            tmp_path / "harbor",
            task_names=list(lock.benchmark_tasks),
            attempts=lock.attempts,
            concurrency=lock.concurrent_trials,
            expected_task_digests=dict(lock.benchmark_task_digests),
        )


def test_request_digest_rejects_tampering(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    _, request = _request(remote_spec, tmp_path)
    value = request.model_dump(mode="json")
    value["harbor_config"]["n_attempts"] = 9

    with pytest.raises(ValidationError, match="digest does not match"):
        HarborExecutionRequest.model_validate(value)


def test_adapter_models_reject_unknown_schema_versions(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    _, request = _request(remote_spec, tmp_path)
    request_value = request.model_dump(mode="json")
    request_value["schema_version"] = "harbor-hf/harbor-execution-request/v2"
    bundle_value = _bundle(request).model_dump(mode="json")
    bundle_value["schema_version"] = "harbor-hf/harbor-compatibility/v2"

    with pytest.raises(ValidationError, match="schema_version"):
        HarborExecutionRequest.model_validate(request_value)
    with pytest.raises(ValidationError, match="schema_version"):
        HarborCompatibilityBundle.model_validate(bundle_value)


def test_compatibility_reader_accepts_v1alpha2_without_exception_messages(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    _, request = _request(remote_spec, tmp_path)
    bundle_value = _bundle(request).model_dump(mode="json")
    bundle_value["schema_version"] = "harbor-hf/harbor-compatibility/v1alpha2"
    for trial in bundle_value["trials"]:
        trial.pop("exception_message", None)
        for step in trial["step_exceptions"]:
            step.pop("exception_message", None)

    bundle = HarborCompatibilityBundle.model_validate(bundle_value)

    assert bundle.schema_version == "harbor-hf/harbor-compatibility/v1alpha2"
    assert bundle.trials[0].exception_message is None


def test_compatibility_v1alpha2_rejects_exception_messages(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    _, request = _request(remote_spec, tmp_path)
    bundle_value = _bundle(request).model_dump(mode="json")
    bundle_value["schema_version"] = "harbor-hf/harbor-compatibility/v1alpha2"
    bundle_value["trials"][0]["exception_message"] = None

    with pytest.raises(ValueError, match="v1alpha2 cannot contain"):
        HarborCompatibilityBundle.model_validate(bundle_value)


def test_typed_bundle_preserves_existing_verification_result(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    _, request = _request(remote_spec, tmp_path)

    assert validate_compatibility_bundle(_bundle(request), request).model_dump(
        mode="json"
    ) == {
        "trial_count": 1,
        "trials": [{"task_name": "cancel-async-tasks", "rewards": {"reward": 1.0}}],
    }


def test_typed_bundle_resolves_internal_task_name_by_locked_digest(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    _, request = _request(remote_spec, tmp_path)

    result = validate_compatibility_bundle(
        _bundle(request, task_name="task/internal-canonical-name"), request
    )

    assert result.model_dump(mode="json") == {
        "trial_count": 1,
        "trials": [{"task_name": "cancel-async-tasks", "rewards": {"reward": 1.0}}],
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"task_digest": "sha256:" + "9" * 64}, "task digest"),
        ({"agent_version": "wrong"}, "agent identity"),
        ({"model_name": "wrong"}, "model identity"),
        ({"rewards": None}, "no verifier rewards"),
    ],
)
def test_typed_bundle_rejects_policy_mismatches(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    _, request = _request(remote_spec, tmp_path)

    with pytest.raises(HarborVerificationFailure, match=message):
        validate_compatibility_bundle(_bundle(request, **updates), request)


def test_adapter_revalidates_inputs_before_failed_return(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock, _request_value = _request(remote_spec, tmp_path)
    adapter = FilesystemHarborExecutionAdapter()
    prepared = adapter.prepare(
        lock,
        tmp_path,
        tmp_path / "jobs",
        "https://endpoint.example",
        tmp_path / "harbor",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )

    def mutate_input(*_args: object, **_kwargs: object) -> int:
        prepared.config_path.write_text("{}\n", encoding="utf-8")
        return 7

    with pytest.raises(WorkerError, match="config changed"):
        adapter.execute(
            prepared,
            tmp_path / "harbor",
            tmp_path / "jobs",
            tmp_path / "harbor.log",
            environment={},
            timeout_seconds=30,
            stream_runner=mutate_input,
        )


def test_adapter_revalidates_inputs_after_runner_exception(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock, _request_value = _request(remote_spec, tmp_path)
    adapter = FilesystemHarborExecutionAdapter()
    prepared = adapter.prepare(
        lock,
        tmp_path,
        tmp_path / "jobs",
        "https://endpoint.example",
        tmp_path / "harbor",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )

    def fail(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("runner failed")

    with pytest.raises(RuntimeError, match="runner failed"):
        adapter.execute(
            prepared,
            tmp_path / "harbor",
            tmp_path / "jobs",
            tmp_path / "harbor.log",
            environment={},
            timeout_seconds=30,
            stream_runner=fail,
        )

    def mutate_then_fail(*_args: object, **_kwargs: object) -> int:
        prepared.request_path.write_text("{}\n", encoding="utf-8")
        raise RuntimeError("runner failed")

    with pytest.raises(WorkerError, match="request changed"):
        adapter.execute(
            prepared,
            tmp_path / "harbor",
            tmp_path / "jobs",
            tmp_path / "harbor.log",
            environment={},
            timeout_seconds=30,
            stream_runner=mutate_then_fail,
        )


def test_adapter_export_uses_only_remaining_shared_deadline(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock, _request_value = _request(remote_spec, tmp_path)
    adapter = FilesystemHarborExecutionAdapter()
    jobs_dir = tmp_path / "jobs"
    prepared = adapter.prepare(
        lock,
        tmp_path,
        jobs_dir,
        "https://endpoint.example",
        tmp_path / "harbor",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    timeouts: list[int] = []

    def run(*_args: object, **kwargs: object) -> int:
        timeout = kwargs["timeout_seconds"]
        assert isinstance(timeout, int)
        timeouts.append(timeout)
        if len(timeouts) == 1:
            now[0] += 4.25
            result = jobs_dir / "job" / "trial" / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text("{}\n", encoding="utf-8")
            return 0
        return 1

    now = [100.0]

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    with pytest.raises(WorkerError, match="deadline was reached"):
        adapter.execute(
            prepared,
            tmp_path / "harbor",
            jobs_dir,
            tmp_path / "harbor.log",
            environment={},
            timeout_seconds=10,
            stream_runner=run,
            monotonic=monotonic,
            sleep=sleep,
            deadline=110.0,
        )

    assert timeouts == [10, 6, 5, 3]


def test_adapter_retries_transient_export_failure(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock, _request_value = _request(remote_spec, tmp_path)
    adapter = FilesystemHarborExecutionAdapter()
    jobs_dir = tmp_path / "jobs"
    prepared = adapter.prepare(
        lock,
        tmp_path,
        jobs_dir,
        "https://endpoint.example",
        tmp_path / "harbor",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    calls = 0

    def run(_command: object, log_path: Path, **_kwargs: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            result = jobs_dir / "job" / "trial" / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text("{}\n", encoding="utf-8")
            return 0
        if calls == 2:
            log_path.write_text(
                "transient bucket visibility failure\n", encoding="utf-8"
            )
            return 1
        prepared.request_path.with_name("harbor-compatibility.json").write_text(
            _bundle(prepared.request).model_dump_json() + "\n", encoding="utf-8"
        )
        return 0

    outcome = adapter.execute(
        prepared,
        tmp_path / "harbor",
        jobs_dir,
        tmp_path / "harbor.log",
        environment={},
        timeout_seconds=30,
        stream_runner=run,
        sleep=lambda _seconds: None,
    )

    assert calls == 3
    assert outcome.verification is not None
    assert outcome.verification.trial_count == 1
    assert (
        "transient bucket visibility failure"
        in (tmp_path / "harbor-export.log").read_text()
    )


def test_adapter_retries_successful_but_incomplete_export(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock, _request_value = _request(remote_spec, tmp_path)
    adapter = FilesystemHarborExecutionAdapter()
    jobs_dir = tmp_path / "jobs"
    prepared = adapter.prepare(
        lock,
        tmp_path,
        jobs_dir,
        "https://endpoint.example",
        tmp_path / "harbor",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    calls = 0

    def run(_command: object, _log_path: Path, **_kwargs: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            result = jobs_dir / "job" / "trial" / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text("{}\n", encoding="utf-8")
            return 0
        bundle = _bundle(prepared.request)
        if calls == 2:
            bundle = bundle.model_copy(update={"trials": []})
        prepared.request_path.with_name("harbor-compatibility.json").write_text(
            bundle.model_dump_json() + "\n", encoding="utf-8"
        )
        return 0

    outcome = adapter.execute(
        prepared,
        tmp_path / "harbor",
        jobs_dir,
        tmp_path / "harbor.log",
        environment={},
        timeout_seconds=30,
        stream_runner=run,
        sleep=lambda _seconds: None,
    )

    assert calls == 3
    assert outcome.verification is not None
    assert outcome.verification.trial_count == 1
    export_log = (tmp_path / "harbor-export.log").read_text()
    assert "== exporter attempt 1 ==" in export_log
    assert "== exporter attempt 2 ==" in export_log


def test_adapter_revalidates_inputs_after_failed_export(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock, _request_value = _request(remote_spec, tmp_path)
    adapter = FilesystemHarborExecutionAdapter()
    jobs_dir = tmp_path / "jobs"
    prepared = adapter.prepare(
        lock,
        tmp_path,
        jobs_dir,
        "https://endpoint.example",
        tmp_path / "harbor",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    calls = 0

    def run(*_args: object, **_kwargs: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            result = jobs_dir / "job" / "trial" / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text("{}\n", encoding="utf-8")
            return 7
        prepared.request_path.write_text("{}\n", encoding="utf-8")
        return 1

    with pytest.raises(WorkerError, match="request changed"):
        adapter.execute(
            prepared,
            tmp_path / "harbor",
            jobs_dir,
            tmp_path / "harbor.log",
            environment={},
            timeout_seconds=30,
            stream_runner=run,
        )

    assert calls == 2


def test_adapter_preserves_harbor_failure_when_export_raises(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock, _request_value = _request(remote_spec, tmp_path)
    adapter = FilesystemHarborExecutionAdapter()
    jobs_dir = tmp_path / "jobs"
    prepared = adapter.prepare(
        lock,
        tmp_path,
        jobs_dir,
        "https://endpoint.example",
        tmp_path / "harbor",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    calls = 0

    def run(*_args: object, **_kwargs: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            result = jobs_dir / "job" / "trial" / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text("{}\n", encoding="utf-8")
            return 7
        raise RuntimeError("export timed out")

    outcome = adapter.execute(
        prepared,
        tmp_path / "harbor",
        jobs_dir,
        tmp_path / "harbor.log",
        environment={},
        timeout_seconds=30,
        stream_runner=run,
    )

    assert calls == 2
    assert outcome.exit_code == 7
    assert outcome.verification is None
    assert outcome.compatibility_path is None


def test_adapter_does_not_start_export_after_shared_deadline(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock, _request_value = _request(remote_spec, tmp_path)
    adapter = FilesystemHarborExecutionAdapter()
    jobs_dir = tmp_path / "jobs"
    prepared = adapter.prepare(
        lock,
        tmp_path,
        jobs_dir,
        "https://endpoint.example",
        tmp_path / "harbor",
        task_names=list(lock.benchmark_tasks),
        attempts=lock.attempts,
        concurrency=lock.concurrent_trials,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )
    calls = 0

    def run(*_args: object, **_kwargs: object) -> int:
        nonlocal calls
        calls += 1
        result = jobs_dir / "job" / "trial" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text("{}\n", encoding="utf-8")
        return 0

    times = iter([100.0, 110.0])
    with pytest.raises(WorkerError, match="deadline was reached"):
        adapter.execute(
            prepared,
            tmp_path / "harbor",
            jobs_dir,
            tmp_path / "harbor.log",
            environment={},
            timeout_seconds=10,
            stream_runner=run,
            monotonic=lambda: next(times),
            deadline=110.0,
        )

    assert calls == 1


def test_compatibility_inventory_refreshes_after_redaction(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "job"
    trial_dir = job_dir / "secret-trial"
    trial_dir.mkdir(parents=True)
    (job_dir / "lock.json").write_text('{"token":"secret"}\n', encoding="utf-8")
    (job_dir / "result.json").write_text('{"token":"secret"}\n', encoding="utf-8")
    (trial_dir / "lock.json").write_text('{"token":"secret"}\n', encoding="utf-8")
    (trial_dir / "result.json").write_text('{"token":"secret"}\n', encoding="utf-8")
    (trial_dir / "secret-output.txt").write_text("secret\n", encoding="utf-8")
    output = tmp_path / "harbor-compatibility.json"
    output.write_text(
        json.dumps(
            {
                "jobs": [{"path": "job"}],
                "trials": [{"path": "job/secret-trial"}],
            }
        ),
        encoding="utf-8",
    )

    scrub_secret_paths(tmp_path, "secret")
    scrub_secret(tmp_path, "secret")
    refresh_bundle_artifacts(jobs_dir, output)

    bundle = json.loads(output.read_text(encoding="utf-8"))
    trial = bundle["trials"][0]
    retained = jobs_dir / trial["path"]
    assert trial["path"] == "job/[REDACTED]-trial"
    assert {entry["path"] for entry in trial["artifacts"]} == {
        "[REDACTED]-output.txt",
        "lock.json",
        "result.json",
    }
    for entry in trial["artifacts"]:
        path = retained / entry["path"]
        assert entry["size"] == path.stat().st_size
        assert entry["digest"] == sha256_digest(path.read_bytes())


def test_compatibility_inventory_excludes_raw_workspace(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "job"
    trial_dir = job_dir / "trial"
    workspace = trial_dir / "artifacts" / "workspace" / "app"
    workspace.mkdir(parents=True)
    (job_dir / "lock.json").write_text("{}\n", encoding="utf-8")
    (job_dir / "result.json").write_text("{}\n", encoding="utf-8")
    (trial_dir / "lock.json").write_text("{}\n", encoding="utf-8")
    (trial_dir / "result.json").write_text("{}\n", encoding="utf-8")
    (workspace / "answer.txt").write_text("answer\n", encoding="utf-8")
    (workspace / "answer-link").symlink_to("answer.txt")
    output = tmp_path / "harbor-compatibility.json"
    output.write_text(
        json.dumps(
            {
                "jobs": [{"path": "job"}],
                "trials": [{"path": "job/trial"}],
            }
        ),
        encoding="utf-8",
    )

    refresh_bundle_artifacts(jobs_dir, output)

    artifacts = json.loads(output.read_text())["trials"][0]["artifacts"]
    assert {entry["path"] for entry in artifacts} == {"lock.json", "result.json"}


def test_typed_bundle_reports_trial_and_multistep_failures(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    _, request = _request(remote_spec, tmp_path)

    with pytest.raises(HarborTrialFailure, match="failed with AgentError"):
        validate_compatibility_bundle(
            _bundle(request, exception_type="AgentError"), request
        )
    with pytest.raises(
        HarborTrialFailure, match="step verifier failed with VerifierError"
    ):
        validate_compatibility_bundle(
            _bundle(
                request,
                step_exceptions=[
                    {"step_name": "verifier", "exception_type": "VerifierError"}
                ],
            ),
            request,
        )


def test_wildcard_request_counts_resolved_tasks_not_patterns(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = build_run_lock(remote_spec, run_id="wildcard-contract").model_copy(
        update={
            "benchmark_tasks": ["task-*"],
            "benchmark_task_digests": {
                "task-one": "sha256:" + "6" * 64,
                "task-two": "sha256:" + "7" * 64,
            },
            "attempts": 2,
        }
    )
    request = build_execution_request(
        lock,
        tmp_path / "jobs",
        "https://endpoint.example",
        task_names=["task-*"],
        attempts=2,
        concurrency=1,
        expected_task_digests=dict(lock.benchmark_task_digests),
    )

    assert request.verification.expected_trials == 4
    assert request.verification.expected_task_counts == {
        "task-one": 2,
        "task-two": 2,
    }
    datasets = request.harbor_config["datasets"]
    assert isinstance(datasets, list)
    dataset = datasets[0]
    assert isinstance(dataset, dict)
    assert dataset["task_names"] == ["task-*"]


def test_literal_bracketed_task_name_is_not_treated_as_a_pattern(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    deprecated = "[DEPRECATED] duplicate-task"
    lock = build_run_lock(remote_spec, run_id="literal-task-name").model_copy(
        update={
            "benchmark_tasks": [deprecated],
            "benchmark_task_digests": {
                deprecated: "sha256:" + "6" * 64,
                "D duplicate-task": "sha256:" + "7" * 64,
            },
        }
    )

    request = build_execution_request(
        lock,
        tmp_path / "jobs",
        "https://endpoint.example",
        task_names=[deprecated],
        attempts=1,
        concurrency=1,
        expected_task_digests={deprecated: "sha256:" + "6" * 64},
    )

    assert request.verification.expected_task_digests == {
        deprecated: "sha256:" + "6" * 64
    }
    datasets = request.harbor_config["datasets"]
    assert isinstance(datasets, list)
    dataset = datasets[0]
    assert isinstance(dataset, dict)
    assert dataset["task_names"] == [deprecated]


@pytest.mark.parametrize(
    "expected_task_digests",
    [
        {"task-one": "sha256:" + "6" * 64},
        {"task-one": "sha256:" + "9" * 64, "task-two": "sha256:" + "7" * 64},
        {"task-outside": "sha256:" + "9" * 64},
    ],
)
def test_request_rejects_task_maps_that_do_not_match_the_lock(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    expected_task_digests: dict[str, str],
) -> None:
    lock = build_run_lock(remote_spec, run_id="task-boundary").model_copy(
        update={
            "benchmark_tasks": ["task-*"],
            "benchmark_task_digests": {
                "task-one": "sha256:" + "6" * 64,
                "task-two": "sha256:" + "7" * 64,
            },
        }
    )

    with pytest.raises(WorkerError, match="outside the resolved run set"):
        build_execution_request(
            lock,
            tmp_path / "jobs",
            "https://endpoint.example",
            task_names=["task-*"],
            attempts=1,
            concurrency=1,
            expected_task_digests=expected_task_digests,
        )


def test_golden_adapter_scenarios_remain_compatible(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    _, request = _request(remote_spec, tmp_path)
    corpus = json.loads(GOLDEN_CONTRACT.read_text(encoding="utf-8"))

    assert corpus["schema_version"] == ("harbor-hf/harbor-adapter-contract-corpus/v1")
    assert [scenario["name"] for scenario in corpus["scenarios"]] == [
        "successful-trial",
        "handled-trial-failure",
        "infrastructure-failure",
        "physical-retry",
        "successful-multi-step-trial",
    ]
    for scenario in corpus["scenarios"]:
        if scenario["bundle"] is None:
            assert scenario["process_exit"] != 0
            assert scenario["expected"] == "process-failure"
            continue
        bundle = _bundle(request, **scenario["trial_updates"])
        if scenario["expected"] == "verified":
            assert validate_compatibility_bundle(bundle, request).trial_count == 1
        else:
            with pytest.raises(HarborTrialFailure, match=str(scenario["expected"])):
                validate_compatibility_bundle(bundle, request)
