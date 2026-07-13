from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import yaml

from harbor_hf.campaigns import (
    CampaignLock,
    WaveLock,
    build_campaign_lock,
    build_campaign_plan,
    build_wave_lock,
)
from harbor_hf.control import CampaignSubmittedPayload, new_event
from harbor_hf.evidence import verify_checksums, write_checksums
from harbor_hf.models import EndpointRef, ExperimentSpec, SourcePin
from harbor_hf.process import CommandRunner
from harbor_hf.reconciler import plan_reconciliation
from harbor_hf.wave_worker import WorkerError, run_wave_worker


def endpoint_snapshot(state: str, ready: int) -> dict[str, object]:
    return {
        "model": {
            "repository": "nvidia/Qwen3.6-35B-A3B-NVFP4",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "image": {
                "custom": {
                    "url": "ghcr.io/example/vllm@sha256:" + "0" * 64,
                }
            },
            "args": [
                "--model",
                "/repository",
                "--max-model-len",
                "65536",
                "--kv-cache-dtype",
                "fp8",
            ],
            "env": {"VLLM_USE_FLASHINFER_MOE_FP4": "1"},
            "secrets": {"HF_TOKEN": "configured"},
        },
        "provider": {"vendor": "aws", "region": "us-east-1"},
        "compute": {
            "instanceType": "nvidia-rtx-pro-6000",
            "instanceSize": "x1",
            "scaling": {"minReplica": 0, "maxReplica": 1},
        },
        "status": {
            "state": state,
            "readyReplica": ready,
            "targetReplica": 1,
            "url": "https://endpoint.example",
        },
        "healthRoute": "/ready",
    }


class EndpointRunner:
    def __init__(self, descriptions: list[dict[str, object]]) -> None:
        self.descriptions = descriptions
        self.commands: list[list[str]] = []

    def run_json(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        del timeout_seconds
        self.commands.append(list(command))
        operation = command[2]
        if operation == "describe":
            return self.descriptions.pop(0)
        return endpoint_snapshot("running" if operation == "resume" else "paused", 0)

    def run_text(
        self, command: Sequence[str], *, timeout_seconds: float | None = None
    ) -> str:
        raise AssertionError((command, timeout_seconds))


class IdentifierSequence:
    def __init__(self, start: int = 1) -> None:
        self.value = start
        self.lock = threading.Lock()

    def __call__(self) -> str:
        with self.lock:
            value = self.value
            self.value += 1
        return f"{value:032x}"


class HarborStream:
    def __init__(
        self,
        task_digests: dict[str, str],
        *,
        expected_calls: int,
        exit_code: int = 0,
        synchronize: bool = False,
    ) -> None:
        self.task_digests = task_digests
        self.barrier = threading.Barrier(expected_calls) if synchronize else None
        self.exit_code = exit_code
        self.commands: list[list[str]] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def __call__(
        self,
        command: list[str],
        log_path: Path,
        *,
        environment: dict[str, str],
        timeout_seconds: int,
    ) -> int:
        assert environment["OPENAI_BASE_URL"] == "https://endpoint.example/v1"
        assert timeout_seconds > 0
        with self.lock:
            self.commands.append(command)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2)
            log_path.write_text("completed test-token\n", encoding="utf-8")
            if self.exit_code != 0:
                return self.exit_code
            task_name = command[command.index("--include-task-name") + 1]
            jobs_dir = Path(command[command.index("--jobs-dir") + 1])
            trial = jobs_dir / "job" / "trial"
            trial.mkdir(parents=True)
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": task_name,
                        "agent_info": {
                            "name": "openclaw",
                            "version": "2026.7.2",
                            "model_info": {
                                "provider": "openai",
                                "name": "/repository",
                            },
                        },
                        "verifier_result": {"rewards": {"reward": 1.0}},
                    }
                ),
                encoding="utf-8",
            )
            (trial / "lock.json").write_text(
                json.dumps({"task": {"digest": self.task_digests[task_name]}}),
                encoding="utf-8",
            )
            return 0
        finally:
            with self.lock:
                self.active -= 1


def prepare_source(source: SourcePin, destination: Path, runner: CommandRunner) -> None:
    del source, runner
    destination.mkdir(parents=True)
    (destination / "uv.lock").write_text("", encoding="utf-8")


def launch_watchdog(lock: WaveLock, endpoint: EndpointRef, token: str) -> str:
    del lock, endpoint, token
    return "watchdog-job"


def test_wave_runs_two_attempt_shards_under_one_endpoint_startup(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=2, concurrency=2
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(
        "harbor_hf.worker.probe_runtime",
        lambda _url, _token, _route: {"probes": {"health": {"http_status": 200}}},
    )
    endpoint = EndpointRunner(
        [
            endpoint_snapshot("paused", 0),
            endpoint_snapshot("running", 1),
            endpoint_snapshot("paused", 0),
        ]
    )
    harbor = HarborStream(
        spec.benchmark.task_digests, expected_calls=2, synchronize=True
    )
    output = tmp_path / "output"

    destination = run_wave_worker(
        manifest,
        campaign_path,
        wave_path,
        output,
        runner=endpoint,
        stream_runner=harbor,
        source_preparer=prepare_source,
        watchdog_launcher=launch_watchdog,
        identifier=IdentifierSequence(),
    )

    assert destination == output / wave.artifact_prefix
    assert (destination / "_SUCCESS").is_file()
    assert not (destination / "_FAILED").exists()
    assert harbor.max_active == 2
    assert len(harbor.commands) == 2
    assert all(
        command[command.index("--n-attempts") + 1] == "1"
        and command[command.index("--n-concurrent") + 1] == "1"
        for command in harbor.commands
    )
    assert [command[2] for command in endpoint.commands] == [
        "describe",
        "resume",
        "describe",
        "pause",
        "describe",
    ]
    assert _event_payloads(destination / "events.jsonl") == [
        {"event": "wave_started", "wave_id": wave.wave_id},
        {"event": "endpoint_baseline_validated"},
        {"event": "endpoint_lease_acquired", "watchdog_job_id": "watchdog-job"},
        {"event": "cleanup_watchdog_started", "job_id": "watchdog-job"},
        {"event": "endpoint_resume_requested"},
        {"event": "endpoint_ready", "state": "running"},
        {"event": "runtime_probed"},
        {"event": "endpoint_pause_requested"},
        {
            "event": "endpoint_paused",
            "state": "paused",
            "ready_replicas": 0,
            "target_replicas": 1,
        },
        {"event": "wave_succeeded"},
    ]
    assert json.loads((destination / "wave.lock.json").read_text()) == (
        wave.model_dump(mode="json")
    )
    assert sorted(path.name for path in destination.iterdir()) == [
        "_SUCCESS",
        "checksums.json",
        "endpoint.final.json",
        "endpoint.snapshot.json",
        "events.jsonl",
        "runtime-environment.json",
        "wave-summary.json",
        "wave.lock.json",
    ]
    run = wave.runs[0]
    run_root = output / run.artifact_prefix
    assert json.loads((run_root / "run.lock.json").read_text()) == (
        run.configuration.model_dump(mode="json")
    )
    trial_roots = sorted((run_root / "trials").iterdir())
    assert len(trial_roots) == 2
    attempts = set()
    expected_trials = {
        trial.trial_id: trial for shard in run.shards for trial in shard.shard.trials
    }
    for trial_root in trial_roots:
        verify_checksums(trial_root)
        trial = expected_trials[trial_root.name]
        assert sorted(path.name for path in trial_root.iterdir()) == [
            "_SUCCESS",
            "checksums.json",
            "events.jsonl",
            "executions",
            "trial-summary.json",
            "trial.lock.json",
        ]
        assert json.loads((trial_root / "trial.lock.json").read_text()) == (
            trial.model_dump(mode="json")
        )
        assert _event_payloads(trial_root / "events.jsonl") == [
            {"event": "trial_succeeded"}
        ]
        execution_root = next((trial_root / "executions").iterdir())
        verify_checksums(execution_root)
        assert sorted(path.name for path in execution_root.iterdir()) == [
            "_SUCCESS",
            "artifacts.tar.gz",
            "checksums.json",
            "events.jsonl",
            "execution.lock.json",
            "harbor-jobs",
            "harbor.log",
            "manifest.yaml",
            "verification.json",
        ]
        execution = json.loads(
            (execution_root / "execution.lock.json").read_text(encoding="utf-8")
        )
        attempts.add(execution["logical_attempt"])
        assert execution == {
            "schema_version": "harbor-hf/execution-lock/v1alpha1",
            "execution_id": execution_root.name,
            "created_at": execution["created_at"],
            "campaign_id": campaign.campaign_id,
            "wave_id": wave.wave_id,
            "run_id": run.configuration.run_id,
            "shard_id": execution["shard_id"],
            "trial_id": trial.trial_id,
            "task_name": trial.task_name,
            "task_digest": trial.task_digest,
            "logical_attempt": trial.logical_attempt,
            "physical_attempt": 1,
        }
        assert _event_payloads(execution_root / "events.jsonl") == [
            {"event": "execution_started", "execution_id": execution_root.name},
            {"event": "harbor_finished", "exit_code": 0},
            {"event": "execution_succeeded"},
            {"event": "secrets_redacted", "files": ["harbor.log"]},
        ]
        assert json.loads((execution_root / "verification.json").read_text()) == {
            "trial_count": 1,
            "trials": [{"task_name": trial.task_name, "rewards": {"reward": 1.0}}],
        }
        trial_summary = json.loads((trial_root / "trial-summary.json").read_text())
        assert trial_summary == {
            "trial_id": trial.trial_id,
            "execution_id": execution_root.name,
            "execution_checksum": trial_summary["execution_checksum"],
        }
        assert b"test-token" not in (execution_root / "artifacts.tar.gz").read_bytes()
        assert (execution_root / "harbor.log").read_text(encoding="utf-8") == (
            "completed [REDACTED]\n"
        )
    assert attempts == {1, 2}
    for shard in run.shards:
        shard_root = output / shard.artifact_prefix
        verify_checksums(shard_root)
        assert sorted(path.name for path in shard_root.iterdir()) == [
            "_SUCCESS",
            "checksums.json",
            "events.jsonl",
            "shard-summary.json",
            "shard.lock.json",
        ]
        assert json.loads((shard_root / "shard.lock.json").read_text()) == (
            shard.shard.model_dump(mode="json")
        )
        shard_events = _event_payloads(shard_root / "events.jsonl")
        assert shard_events[0] == {
            "event": "shard_started",
            "shard_id": shard.shard.shard_id,
        }
        assert shard_events[-1] == {"event": "shard_succeeded"}
        assert not (shard_root / "trials").exists()
    verify_checksums(destination)
    summary = json.loads(
        (destination / "wave-summary.json").read_text(encoding="utf-8")
    )
    assert summary == {
        "wave_id": wave.wave_id,
        "campaign_id": campaign.campaign_id,
        "shard_checksums": summary["shard_checksums"],
        "endpoint_cleanup_verified": True,
    }
    assert set(summary["shard_checksums"]) == set(wave.shard_ids)
    assert json.loads(
        (output / campaign.artifact_prefix / "campaign.lock.json").read_text()
    ) == campaign.model_dump(mode="json")


def test_wave_failure_still_pauses_and_publishes_failed_execution(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr("harbor_hf.worker.probe_runtime", lambda *_args: {"probes": {}})
    endpoint = EndpointRunner(
        [
            endpoint_snapshot("paused", 0),
            endpoint_snapshot("running", 1),
            endpoint_snapshot("paused", 0),
        ]
    )
    harbor = HarborStream(spec.benchmark.task_digests, expected_calls=1, exit_code=7)
    output = tmp_path / "output"

    with pytest.raises(WorkerError, match="Harbor exited with status 7"):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            output,
            runner=endpoint,
            stream_runner=harbor,
            source_preparer=prepare_source,
            watchdog_launcher=launch_watchdog,
            identifier=IdentifierSequence(),
        )

    assert [command[2] for command in endpoint.commands][-2:] == [
        "pause",
        "describe",
    ]
    assert (output / wave.artifact_prefix / "_FAILED").is_file()
    wave_root = output / wave.artifact_prefix
    failure = json.loads((wave_root / "_FAILED").read_text())
    assert failure == {
        "wave_id": wave.wave_id,
        "campaign_id": "campaign-one",
        "shard_checksums": {},
        "endpoint_cleanup_verified": True,
        "error_type": "WorkerError",
        "message": "Harbor exited with status 7",
    }
    assert _event_payloads(wave_root / "events.jsonl")[-3:] == [
        {"event": "endpoint_pause_requested"},
        {
            "event": "endpoint_paused",
            "state": "paused",
            "ready_replicas": 0,
            "target_replicas": 1,
        },
        {"event": "wave_failed", "error_type": "WorkerError"},
    ]
    run = wave.runs[0]
    trial_id = run.shards[0].shard.trials[0].trial_id
    executions = output / run.artifact_prefix / "trials" / trial_id / "executions"
    execution = next(executions.iterdir())
    assert (execution / "_FAILED").is_file()
    assert json.loads((execution / "_FAILED").read_text()) == {
        "error_type": "WorkerError",
        "message": "Harbor exited with status 7",
    }
    assert _event_payloads(execution / "events.jsonl") == [
        {"event": "execution_started", "execution_id": execution.name},
        {"event": "harbor_finished", "exit_code": 7},
        {"event": "execution_failed", "error_type": "WorkerError"},
        {"event": "secrets_redacted", "files": ["harbor.log"]},
    ]
    assert not (execution.parent.parent / "_SUCCESS").exists()


def test_wave_never_resumes_or_pauses_without_watchdog_lease(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spec, _campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    endpoint = EndpointRunner([endpoint_snapshot("paused", 0)])

    def reject(lock: WaveLock, endpoint: EndpointRef, token: str) -> str:
        del lock, endpoint, token
        raise WorkerError("endpoint lease is held by another watchdog")

    with pytest.raises(WorkerError, match="lease is held"):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            tmp_path / "output",
            runner=endpoint,
            source_preparer=prepare_source,
            watchdog_launcher=reject,
        )

    assert [command[2] for command in endpoint.commands] == ["describe"]
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "output" / wave.artifact_prefix / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert "endpoint_cleanup_skipped" in events


@pytest.mark.parametrize(
    ("state", "ready", "message"),
    [
        ("running", 1, "must be paused"),
        ("paused", 1, "must be paused"),
    ],
)
def test_wave_rejects_non_paused_endpoint_before_watchdog(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    ready: int,
    message: str,
) -> None:
    _spec, _campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    endpoint = EndpointRunner([endpoint_snapshot(state, ready)])
    watchdog_called = False

    def unexpected_watchdog(lock: WaveLock, endpoint: EndpointRef, token: str) -> str:
        nonlocal watchdog_called
        del lock, endpoint, token
        watchdog_called = True
        return "unexpected"

    with pytest.raises(WorkerError, match=message):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            tmp_path / "output",
            runner=endpoint,
            source_preparer=prepare_source,
            watchdog_launcher=unexpected_watchdog,
        )

    assert not watchdog_called
    assert [command[2] for command in endpoint.commands] == ["describe"]
    wave_root = tmp_path / "output" / wave.artifact_prefix
    assert json.loads((wave_root / "_FAILED").read_text())["message"] == (
        "endpoint must be paused with zero ready replicas before ownership"
    )
    assert _event_payloads(wave_root / "events.jsonl")[-2:] == [
        {"event": "endpoint_cleanup_skipped", "reason": "lease_not_owned"},
        {"event": "wave_failed", "error_type": "WorkerError"},
    ]


def test_wave_rejects_endpoint_model_mismatch_before_watchdog(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spec, _campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    snapshot = endpoint_snapshot("paused", 0)
    model = cast(dict[str, object], snapshot["model"])
    model["revision"] = "f" * 40
    endpoint = EndpointRunner([snapshot])

    def unexpected_watchdog(lock: WaveLock, endpoint: EndpointRef, token: str) -> str:
        del lock, endpoint, token
        pytest.fail("watchdog must not start")

    with pytest.raises(WorkerError, match="model does not match"):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            tmp_path / "output",
            runner=endpoint,
            source_preparer=prepare_source,
            watchdog_launcher=unexpected_watchdog,
        )

    assert [command[2] for command in endpoint.commands] == ["describe"]


def test_wave_cleanup_failure_overrides_success_and_redacts_secret(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr("harbor_hf.worker.probe_runtime", lambda *_args: {"probes": {}})

    def fail_cleanup(_manager: object) -> dict[str, object]:
        raise WorkerError("pause failed with test-token")

    monkeypatch.setattr(
        "harbor_hf.wave_worker.EndpointManager.pause_and_verify", fail_cleanup
    )
    output = tmp_path / "output"

    with pytest.raises(WorkerError, match=r"pause failed with \[REDACTED\]"):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            output,
            runner=EndpointRunner(
                [endpoint_snapshot("paused", 0), endpoint_snapshot("running", 1)]
            ),
            stream_runner=HarborStream(spec.benchmark.task_digests, expected_calls=1),
            source_preparer=prepare_source,
            watchdog_launcher=launch_watchdog,
            identifier=IdentifierSequence(),
        )

    wave_root = output / wave.artifact_prefix
    failure = json.loads((wave_root / "_FAILED").read_text())
    assert failure["endpoint_cleanup_verified"] is False
    assert failure["error_type"] == "WorkerError"
    assert failure["message"] == "pause failed with [REDACTED]"
    assert _event_payloads(wave_root / "events.jsonl")[-2:] == [
        {"event": "endpoint_cleanup_failed", "error": "WorkerError"},
        {"event": "wave_failed", "error_type": "WorkerError"},
    ]
    assert "test-token" not in (wave_root / "events.jsonl").read_text()


def test_wave_reports_primary_and_cleanup_failures(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr("harbor_hf.worker.probe_runtime", lambda *_args: {"probes": {}})

    def fail_cleanup(_manager: object) -> dict[str, object]:
        raise WorkerError("cleanup test-token")

    monkeypatch.setattr(
        "harbor_hf.wave_worker.EndpointManager.pause_and_verify", fail_cleanup
    )
    output = tmp_path / "output"

    with pytest.raises(
        WorkerError,
        match="Harbor exited with status 9; endpoint cleanup failed: cleanup ",
    ):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            output,
            runner=EndpointRunner(
                [endpoint_snapshot("paused", 0), endpoint_snapshot("running", 1)]
            ),
            stream_runner=HarborStream(
                spec.benchmark.task_digests, expected_calls=1, exit_code=9
            ),
            source_preparer=prepare_source,
            watchdog_launcher=launch_watchdog,
            identifier=IdentifierSequence(),
        )

    failure = json.loads((output / wave.artifact_prefix / "_FAILED").read_text())
    assert failure["message"] == "Harbor exited with status 9"
    assert failure["cleanup_error"] == {
        "error_type": "WorkerError",
        "message": "cleanup [REDACTED]",
    }


def test_wave_validates_lock_and_secret_before_remote_work(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spec, _campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    endpoint = EndpointRunner([])

    with pytest.raises(WorkerError, match="required secret HF_TOKEN"):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            tmp_path / "missing-token",
            runner=endpoint,
        )
    assert endpoint.commands == []

    monkeypatch.setenv("HF_TOKEN", "test-token")
    tampered = wave.model_copy(update={"duration_seconds": wave.duration_seconds + 1})
    wave_path.write_text(tampered.model_dump_json(), encoding="utf-8")
    with pytest.raises(WorkerError, match="fields do not match"):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            tmp_path / "tampered",
            runner=endpoint,
        )
    assert endpoint.commands == []


@pytest.mark.parametrize("marker", ["_SUCCESS", "_FAILED", "_CANCELLED"])
def test_wave_never_overwrites_terminal_wave_evidence(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    _spec, _campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    destination = tmp_path / "output" / wave.artifact_prefix
    destination.mkdir(parents=True)
    (destination / marker).write_text("\n", encoding="utf-8")
    endpoint = EndpointRunner([])

    with pytest.raises(WorkerError, match="already has terminal evidence"):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            tmp_path / "output",
            runner=endpoint,
        )
    assert endpoint.commands == []


def test_wave_duration_bound_stops_admission_and_still_pauses(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spec, _campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr("harbor_hf.worker.probe_runtime", lambda *_args: {"probes": {}})
    endpoint = EndpointRunner(
        [
            endpoint_snapshot("paused", 0),
            endpoint_snapshot("running", 1),
            endpoint_snapshot("paused", 0),
        ]
    )
    observed = iter((0.0, 0.0, 61.0))

    def monotonic() -> float:
        return next(observed, 61.0)

    def unexpected_stream(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("duration-bound wave must not admit a Harbor trial")

    with pytest.raises(WorkerError, match="duration bound was reached"):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            tmp_path / "output",
            runner=endpoint,
            stream_runner=unexpected_stream,
            source_preparer=prepare_source,
            watchdog_launcher=launch_watchdog,
            monotonic=monotonic,
        )

    assert [command[2] for command in endpoint.commands][-2:] == [
        "pause",
        "describe",
    ]
    assert not (tmp_path / "output" / wave.artifact_prefix / "_SUCCESS").exists()


def test_wave_recovery_skips_checksum_valid_terminal_trial(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=2, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr("harbor_hf.worker.probe_runtime", lambda *_args: {"probes": {}})
    first_output = tmp_path / "first"
    run_wave_worker(
        manifest,
        campaign_path,
        wave_path,
        first_output,
        runner=EndpointRunner(
            [
                endpoint_snapshot("paused", 0),
                endpoint_snapshot("running", 1),
                endpoint_snapshot("paused", 0),
            ]
        ),
        stream_runner=HarborStream(spec.benchmark.task_digests, expected_calls=2),
        source_preparer=prepare_source,
        watchdog_launcher=launch_watchdog,
        identifier=IdentifierSequence(),
    )
    recovery = tmp_path / "recovery"
    shutil.copytree(first_output, recovery)
    campaign_root = recovery / campaign.artifact_prefix
    shutil.rmtree(campaign_root / "waves")
    run = wave.runs[0]
    run_root = recovery / run.artifact_prefix
    shutil.rmtree(run_root / "shards")
    second_trial = run.shards[1].shard.trials[0]
    shutil.rmtree(run_root / "trials" / second_trial.trial_id)
    harbor = HarborStream(spec.benchmark.task_digests, expected_calls=1)

    run_wave_worker(
        manifest,
        campaign_path,
        wave_path,
        recovery,
        runner=EndpointRunner(
            [
                endpoint_snapshot("paused", 0),
                endpoint_snapshot("running", 1),
                endpoint_snapshot("paused", 0),
            ]
        ),
        stream_runner=harbor,
        source_preparer=prepare_source,
        watchdog_launcher=launch_watchdog,
        identifier=IdentifierSequence(start=100),
    )

    assert len(harbor.commands) == 1
    first_trial = run.shards[0].shard.trials[0]
    first_executions = (
        recovery / run.artifact_prefix / "trials" / first_trial.trial_id / "executions"
    )
    assert len(list(first_executions.iterdir())) == 1
    events = (recovery / run.shards[0].artifact_prefix / "events.jsonl").read_text()
    assert "trial_recovered" in events


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("second-marker", "not a valid success"),
        ("trial-summary", "wrong trial identity"),
        ("execution-lock", "execution identity does not match"),
        ("execution-content", "checksum validation"),
    ],
)
def test_wave_recovery_rejects_invalid_terminal_trial(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    message: str,
) -> None:
    spec, campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr("harbor_hf.worker.probe_runtime", lambda *_args: {"probes": {}})
    output = tmp_path / "output"
    run_wave_worker(
        manifest,
        campaign_path,
        wave_path,
        output,
        runner=EndpointRunner(
            [
                endpoint_snapshot("paused", 0),
                endpoint_snapshot("running", 1),
                endpoint_snapshot("paused", 0),
            ]
        ),
        stream_runner=HarborStream(spec.benchmark.task_digests, expected_calls=1),
        source_preparer=prepare_source,
        watchdog_launcher=launch_watchdog,
        identifier=IdentifierSequence(),
    )
    campaign_root = output / campaign.artifact_prefix
    shutil.rmtree(campaign_root / "waves")
    run = wave.runs[0]
    shutil.rmtree(output / run.artifact_prefix / "shards")
    trial = run.shards[0].shard.trials[0]
    trial_root = output / run.artifact_prefix / "trials" / trial.trial_id
    execution_root = next((trial_root / "executions").iterdir())
    if corruption == "second-marker":
        (trial_root / "_FAILED").write_text("\n", encoding="utf-8")
    elif corruption == "trial-summary":
        summary_path = trial_root / "trial-summary.json"
        summary = json.loads(summary_path.read_text())
        summary["trial_id"] = "trial-wrong"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        write_checksums(trial_root)
    elif corruption == "execution-lock":
        lock_path = execution_root / "execution.lock.json"
        execution = json.loads(lock_path.read_text())
        execution["wave_id"] = "wave-" + "f" * 24
        lock_path.write_text(json.dumps(execution), encoding="utf-8")
        write_checksums(execution_root)
        write_checksums(trial_root)
    else:
        with (execution_root / "harbor.log").open("a", encoding="utf-8") as stream:
            stream.write("changed\n")

    endpoint = EndpointRunner(
        [
            endpoint_snapshot("paused", 0),
            endpoint_snapshot("running", 1),
            endpoint_snapshot("paused", 0),
        ]
    )
    with pytest.raises(WorkerError, match=message):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            output,
            runner=endpoint,
            stream_runner=lambda *_args, **_kwargs: pytest.fail(
                "invalid terminal trial must not be rerun"
            ),
            source_preparer=prepare_source,
            watchdog_launcher=launch_watchdog,
            identifier=IdentifierSequence(start=100),
        )

    assert [command[2] for command in endpoint.commands][-2:] == [
        "pause",
        "describe",
    ]


def _wave_inputs(
    remote_spec: ExperimentSpec,
    root: Path,
    *,
    attempts: int,
    concurrency: int,
) -> tuple[ExperimentSpec, CampaignLock, WaveLock, Path, Path, Path]:
    spec = remote_spec.model_copy(
        update={
            "execution": remote_spec.execution.model_copy(
                update={
                    "attempts": attempts,
                    "concurrent_trials": concurrency,
                    "max_trials_per_shard": 1,
                }
            )
        }
    )
    campaign = build_campaign_lock(build_campaign_plan(spec), "campaign-one")
    submitted = new_event(
        subject_type="campaign",
        subject_id=campaign.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=campaign.plan_digest),
    )
    action = plan_reconciliation(campaign, [submitted])[1].actions[0]
    wave = build_wave_lock(campaign, spec, action)
    manifest = root / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True)),
        encoding="utf-8",
    )
    campaign_path = root / "campaign.lock.json"
    campaign_path.write_text(campaign.model_dump_json(), encoding="utf-8")
    wave_path = root / "wave.lock.json"
    wave_path.write_text(wave.model_dump_json(), encoding="utf-8")
    return spec, campaign, wave, manifest, campaign_path, wave_path


def _event_payloads(path: Path) -> list[dict[str, object]]:
    return [
        {key: value for key, value in json.loads(line).items() if key != "at"}
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
