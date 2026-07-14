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
)
from harbor_hf.harbor_adapter.exporter import refresh_bundle_artifacts
from harbor_hf.harbor_adapter.models import HarborCompatibilityBundle, sha256_digest
from harbor_hf.harbor_adapter.validation import validate_compatibility_bundle
from harbor_hf.models import ExperimentSpec
from harbor_hf.runs import RunLock, build_run_lock

GOLDEN_CONTRACT = Path(__file__).parent / "golden" / "harbor-adapter-contract-v1.json"


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
            result = jobs_dir / "job" / "trial" / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text("{}\n", encoding="utf-8")
            return 0
        return 1

    times = iter([100.0, 104.25])
    with pytest.raises(WorkerError, match="exporter exited"):
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

    assert timeouts == [10, 6]


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
