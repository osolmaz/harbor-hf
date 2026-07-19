from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from conftest import write_fake_compatibility_bundle

from harbor_hf.campaign_finalizer import BucketCampaignFinalizer
from harbor_hf.campaign_observer import BucketCampaignObserver
from harbor_hf.campaigns import (
    CampaignLock,
    CampaignTrialLock,
    ProviderWaveTarget,
    WaveLock,
    build_campaign_lock,
    build_campaign_plan,
    build_wave_lock,
)
from harbor_hf.control import CampaignSubmittedPayload, new_event
from harbor_hf.evidence import assert_secret_absent, verify_checksums, write_checksums
from harbor_hf.harbor_adapter import HarborTrialFailure, HarborVerificationFailure
from harbor_hf.models import (
    EndpointRef,
    ExperimentSpec,
    GitBenchmarkSource,
    GitHubTokenCredentials,
    SourcePin,
)
from harbor_hf.process import CommandRunner
from harbor_hf.provider_models import ProviderLimits, ProviderTarget
from harbor_hf.provider_proxy import ProviderEvidenceProxy
from harbor_hf.reconciler import (
    DeploymentAdmission,
    ReconcileContext,
    plan_reconciliation,
)
from harbor_hf.recovery import project_recovery
from harbor_hf.results import EvidenceSource, build_result_tables
from harbor_hf.wave_worker import (
    WorkerError,
    _execute_shard,
    _execution_failure_category,
    _execution_id,
    _file_digest,
    _finalize_execution,
    _launch_wave_watchdog,
    _remaining_seconds,
    _sandbox_failure_category,
    _valid_terminal_trial,
    _wave_model_name,
    run_wave_worker,
)


def test_verification_failure_is_terminal_benchmark_evidence() -> None:
    error = HarborVerificationFailure("task digest does not match")

    assert _execution_failure_category(error, "execution") == "benchmark"


def test_wrapped_endpoint_server_error_without_log_remains_agent_failure() -> None:
    error = HarborTrialFailure(
        "agent failed",
        "NonZeroAgentExitCodeError",
        'provider response status=500: "500 Internal Server Error"',
    )

    assert _execution_failure_category(error, "execution") == "agent"


def test_plain_nonzero_agent_exit_is_agent_failure() -> None:
    error = HarborTrialFailure(
        "agent failed", "NonZeroAgentExitCodeError", "command exited with status 1"
    )

    assert _execution_failure_category(error, "execution") == "agent"


def test_benchmark_exception_message_does_not_trigger_transport_retry() -> None:
    error = HarborTrialFailure(
        "verifier failed", "AssertionError", "expected timeout handling"
    )

    assert _execution_failure_category(error, "execution") == "benchmark"


def test_agent_output_keywords_do_not_trigger_transport_retry() -> None:
    error = HarborTrialFailure(
        "agent failed",
        "NonZeroAgentExitCodeError",
        "command asked to diagnose a timeout and quota issue; exit status 1",
    )

    assert _execution_failure_category(error, "execution") == "agent"


def test_openclaw_terminal_transport_log_makes_wrapped_exit_retryable(
    tmp_path: Path,
) -> None:
    log = tmp_path / "harbor-jobs" / "job" / "trial" / "agent" / "openclaw.txt"
    log.parent.mkdir(parents=True)
    log.write_text(
        'model request failed\nFailoverError: HTTP 500: "500 Internal Server Error"\n',
        encoding="utf-8",
    )
    error = HarborTrialFailure("agent failed", "NonZeroAgentExitCodeError")

    assert (
        _execution_failure_category(error, "execution", evidence_root=tmp_path)
        == "transient"
    )


def test_openclaw_structured_transport_timeout_is_retryable(tmp_path: Path) -> None:
    log = tmp_path / "harbor-jobs" / "job" / "trial" / "agent" / "openclaw.txt"
    log.parent.mkdir(parents=True)
    log.write_text(
        "[provider-transport-fetch] [model-fetch] response provider=openai "
        "status=503 elapsedMs=2363 contentType=application/json\n"
        "FailoverError: LLM request timed out.\n",
        encoding="utf-8",
    )
    error = HarborTrialFailure("agent failed", "NonZeroAgentExitCodeError")

    assert (
        _execution_failure_category(error, "execution", evidence_root=tmp_path)
        == "transient"
    )


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        (
            "huggingface_hub.errors.SandboxError: Sandbox job job-one did not "
            "become ready within 120s.",
            "transient",
        ),
        (
            "huggingface_hub.errors.SandboxError: Sandbox API error (503): "
            "service unavailable",
            "transient",
        ),
        (
            "huggingface_hub.errors.SandboxError: Sandbox API error (429): "
            "rate limited",
            "rate-limit",
        ),
        (
            "huggingface_hub.errors.SandboxError: Sandbox API error (400): "
            "failed to spawn '/bin/bash': No such file or directory",
            "benchmark",
        ),
    ],
)
def test_sandbox_failure_uses_trusted_exception_evidence(
    tmp_path: Path, detail: str, expected: str
) -> None:
    error = HarborTrialFailure("sandbox failed", "SandboxError", detail)

    assert (
        _execution_failure_category(error, "execution", evidence_root=tmp_path)
        == expected
    )


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        (
            "huggingface_hub.errors.SandboxError: Sandbox job job-one did not "
            "become ready within 120s.",
            "transient",
        ),
        (
            "huggingface_hub.errors.SandboxError: Sandbox API error (400): "
            "failed to spawn '/bin/bash': No such file or directory",
            "benchmark",
        ),
    ],
)
def test_wrapped_harbor_exit_uses_sandbox_exception_evidence(
    tmp_path: Path, detail: str, expected: str
) -> None:
    result = tmp_path / "harbor-jobs" / "job" / "trial" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_type": "SandboxError",
                    "exception_message": detail,
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        _execution_failure_category(
            WorkerError("Harbor exited with status 1"),
            "execution",
            evidence_root=tmp_path,
        )
        == expected
    )


def test_wrapped_harbor_exit_ignores_sandbox_markers_without_sandbox_error(
    tmp_path: Path,
) -> None:
    result = tmp_path / "harbor-jobs" / "job" / "trial" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "exception_info": {
                    "exception_type": "RuntimeError",
                    "exception_message": "SandboxError: Sandbox API error (503)",
                }
            }
        ),
        encoding="utf-8",
    )

    assert _sandbox_failure_category(tmp_path) is None


def test_wrapped_harbor_exit_uses_preflight_harbor_log_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "harbor.log").write_text(
        "ValueError: HF Sandbox requires a prebuilt Docker image.\n",
        encoding="utf-8",
    )

    assert (
        _execution_failure_category(
            WorkerError("Harbor exited with status 1"),
            "execution",
            evidence_root=tmp_path,
        )
        == "benchmark"
    )


def test_wrapped_harbor_exit_ignores_non_utf8_result(tmp_path: Path) -> None:
    result = tmp_path / "harbor-jobs" / "job" / "trial" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_bytes(b"\xff")

    assert _sandbox_failure_category(tmp_path) is None


def test_wrapped_harbor_exit_ignores_excessively_nested_result(tmp_path: Path) -> None:
    result = tmp_path / "harbor-jobs" / "job" / "trial" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")

    assert _sandbox_failure_category(tmp_path) is None


def test_wrapped_harbor_exit_ignores_overlong_integer_result(tmp_path: Path) -> None:
    result = tmp_path / "harbor-jobs" / "job" / "trial" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"value":' + "1" * 5000 + "}", encoding="utf-8")

    assert _sandbox_failure_category(tmp_path) is None


def test_openclaw_nonterminal_status_log_does_not_reclassify_agent_exit(
    tmp_path: Path,
) -> None:
    log = tmp_path / "harbor-jobs" / "job" / "trial" / "agent" / "openclaw.txt"
    log.parent.mkdir(parents=True)
    log.write_text("provider response status=500\n", encoding="utf-8")
    error = HarborTrialFailure("agent failed", "NonZeroAgentExitCodeError")

    assert (
        _execution_failure_category(error, "execution", evidence_root=tmp_path)
        == "agent"
    )


def test_failed_execution_retains_malformed_compatibility_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "harbor-jobs").mkdir()
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "execution.lock.json").write_text(
        '{"execution_id":"execution-one","trial_id":"trial-one"}\n',
        encoding="utf-8",
    )
    (tmp_path / "harbor-compatibility.json").write_text("{", encoding="utf-8")

    _finalize_execution(tmp_path, "test-token", strict_compatibility=False)

    verify_checksums(tmp_path)
    assert (tmp_path / "artifacts.tar.gz").is_file()
    assert _event_payloads(tmp_path / "events.jsonl") == [
        {
            "event": "compatibility_refresh_skipped",
            "error_type": "JSONDecodeError",
        }
    ]


def test_failed_execution_recreates_rejected_jobs_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    jobs = tmp_path / "harbor-jobs"
    jobs.symlink_to(outside, target_is_directory=True)
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "execution.lock.json").write_text(
        '{"execution_id":"execution-one","trial_id":"trial-one"}\n',
        encoding="utf-8",
    )

    _finalize_execution(tmp_path, "test-token", strict_compatibility=False)

    assert jobs.is_dir()
    assert not jobs.is_symlink()
    assert (tmp_path / "artifacts.tar.gz").is_file()
    rejection = json.loads(
        (tmp_path / "private-artifact-rejections.json").read_text(encoding="utf-8")
    )
    assert rejection["rejections"] == [
        {"path": "harbor-jobs", "reason": "symlink", "size": None}
    ]


def test_failed_execution_recreates_rejected_jobs_file(tmp_path: Path) -> None:
    jobs = tmp_path / "harbor-jobs"
    jobs.write_text("collision", encoding="utf-8")
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "execution.lock.json").write_text(
        '{"execution_id":"execution-one","trial_id":"trial-one"}\n',
        encoding="utf-8",
    )

    _finalize_execution(tmp_path, "test-token", strict_compatibility=False)

    assert jobs.is_dir()
    assert (tmp_path / "artifacts.tar.gz").is_file()
    rejection = json.loads(
        (tmp_path / "private-artifact-rejections.json").read_text(encoding="utf-8")
    )
    assert rejection["rejections"] == [
        {"path": "harbor-jobs", "reason": "reserved_path", "size": 9}
    ]


def test_failed_execution_prunes_unsafe_evidence_and_still_finalizes(
    tmp_path: Path,
) -> None:
    jobs = tmp_path / "harbor-jobs"
    jobs.mkdir()
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "execution.lock.json").write_text(
        '{"execution_id":"execution-one","trial_id":"trial-one"}\n',
        encoding="utf-8",
    )
    (jobs / "linked").symlink_to(tmp_path / "execution.lock.json")
    oversized = jobs / "oversized.log"
    with oversized.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)

    _finalize_execution(tmp_path, "test-token", strict_compatibility=False)

    verify_checksums(tmp_path)
    manifest = json.loads((tmp_path / "private-artifacts.json").read_text())
    assert [(item["path"], item["reason"]) for item in manifest["rejections"]] == [
        ("harbor-jobs/linked", "symlink"),
        ("harbor-jobs/oversized.log", "file_size"),
    ]
    assert not (jobs / "linked").exists()
    assert not oversized.exists()
    assert (tmp_path / "artifacts.tar.gz").is_file()


def test_failed_execution_preserves_attempt_state_before_sanitizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "harbor-jobs").mkdir()
    (tmp_path / "execution.lock.json").write_text(
        '{"execution_id":"execution-one","trial_id":"trial-one"}\n',
        encoding="utf-8",
    )
    (tmp_path / "harbor-request.json").write_text(
        '{"verification":{"expected_agent_name":"openclaw"}}\n',
        encoding="utf-8",
    )
    events = tmp_path / "events.jsonl"
    events.write_text('{"event":"harbor_started"}\n', encoding="utf-8")

    def trim_attempt_event(root: Path, **_kwargs: object) -> list[object]:
        (root / "events.jsonl").unlink(missing_ok=True)
        return []

    monkeypatch.setattr(
        "harbor_hf.wave_worker.sanitize_private_artifact_tree", trim_attempt_event
    )

    _finalize_execution(tmp_path, "test-token", strict_compatibility=False)

    manifest = json.loads((tmp_path / "private-artifacts.json").read_text())
    assert manifest["requirements"] == [
        {
            "name": "openclaw_session_jsonl",
            "paths": [],
            "required": True,
            "satisfied": False,
        }
    ]


def test_failed_execution_sanitizes_result_before_session_probe(tmp_path: Path) -> None:
    trial = tmp_path / "harbor-jobs" / "job" / "trial"
    trial.mkdir(parents=True)
    (tmp_path / "execution.lock.json").write_text(
        '{"execution_id":"execution-one","trial_id":"trial-one"}\n',
        encoding="utf-8",
    )
    (tmp_path / "harbor-request.json").write_text(
        '{"verification":{"expected_agent_name":"openclaw"}}\n',
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text(
        '{"event":"harbor_started"}\n', encoding="utf-8"
    )
    outside = tmp_path.parent / "oversized-result.json"
    with outside.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    (trial / "result.json").symlink_to(outside)

    _finalize_execution(tmp_path, "test-token", strict_compatibility=False)

    manifest = json.loads((tmp_path / "private-artifacts.json").read_text())
    assert manifest["requirements"][0]["required"] is True
    assert ("harbor-jobs/job/trial/result.json", "symlink") in {
        (item["path"], item["reason"]) for item in manifest["rejections"]
    }
    assert outside.stat().st_size == 64 * 1024 * 1024 + 1


def test_failed_execution_refreshes_compatibility_after_final_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = tmp_path / "harbor-jobs"
    jobs.mkdir()
    (tmp_path / "execution.lock.json").write_text(
        '{"execution_id":"execution-one","trial_id":"trial-one"}\n',
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    refresh_calls = 0

    def refresh(root: Path, *, strict: bool) -> None:
        nonlocal refresh_calls
        assert strict is False
        refresh_calls += 1
        if refresh_calls == 1:
            with (root / "harbor-jobs" / "late.log").open("wb") as stream:
                stream.truncate(64 * 1024 * 1024 + 1)

    monkeypatch.setattr("harbor_hf.wave_worker.refresh_retained_bundle", refresh)

    _finalize_execution(tmp_path, "test-token", strict_compatibility=False)

    assert refresh_calls == 2
    assert not (jobs / "late.log").exists()


def test_success_rejects_malformed_compatibility_evidence(tmp_path: Path) -> None:
    (tmp_path / "harbor-jobs").mkdir()
    (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "harbor-compatibility.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError):
        _finalize_execution(tmp_path, "test-token")


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


class LocalEvidence:
    def __init__(self, root: Path) -> None:
        self.root = root

    def list_files(self, *, bucket: str, prefix: str) -> list[str]:
        del bucket
        source = self.root / prefix
        if not source.exists():
            return []
        return sorted(
            str(path.relative_to(source))
            for path in source.rglob("*")
            if path.is_file()
        )

    def read_bytes(self, *, bucket: str, prefix: str, path: str) -> bytes:
        del bucket
        return (self.root / prefix / path).read_bytes()

    def write_immutable(self, *, bucket: str, path: str, content: bytes) -> bool:
        del bucket
        destination = self.root / path
        if destination.exists():
            assert destination.read_bytes() == content
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return True


class HarborStream:
    def __init__(
        self,
        task_digests: dict[str, str],
        *,
        expected_calls: int,
        exit_code: int = 0,
        synchronize: bool = False,
        expected_base_url: str | None = "https://endpoint.example/v1",
        expected_model_name: str = "/repository",
        agent_started: bool = False,
        failure_exception: tuple[str, str] | None = None,
    ) -> None:
        self.task_digests = task_digests
        self.barrier = threading.Barrier(expected_calls) if synchronize else None
        self.exit_code = exit_code
        self.expected_base_url = expected_base_url
        self.expected_model_name = expected_model_name
        self.agent_started = agent_started
        self.failure_exception = failure_exception
        self.commands: list[list[str]] = []
        self.configs: list[dict[str, Any]] = []
        self.base_urls: list[str] = []
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
        if "--output" in command and "--request-digest" in command:
            write_fake_compatibility_bundle(command, log_path)
            return 0
        base_url = environment["OPENAI_BASE_URL"]
        if self.expected_base_url is not None:
            assert base_url == self.expected_base_url
        else:
            assert base_url.startswith("https://test-wave-job--8000.hf.jobs/scopes/")
            assert base_url.endswith("/v1")
        assert timeout_seconds > 0
        config_path = Path(command[command.index("--config") + 1])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with self.lock:
            self.commands.append(command)
            self.configs.append(config)
            self.base_urls.append(base_url)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2)
            log_path.write_text("completed test-token\n", encoding="utf-8")
            task_name = config["datasets"][0]["task_names"][0]
            jobs_dir = Path(config["jobs_dir"])
            trial = jobs_dir / "job" / "trial"
            trial.mkdir(parents=True)
            if self.exit_code != 0:
                if self.failure_exception is not None:
                    exception_type, exception_message = self.failure_exception
                    (trial / "result.json").write_text(
                        json.dumps(
                            {
                                "exception_info": {
                                    "exception_type": exception_type,
                                    "exception_message": exception_message,
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
                return self.exit_code
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": task_name,
                        "agent_info": {
                            "name": "openclaw",
                            "version": "2026.7.2",
                            "model_info": {
                                "provider": "openai",
                                "name": self.expected_model_name,
                            },
                        },
                        "agent_execution": (
                            {"started_at": "2026-07-14T00:00:00Z"}
                            if self.agent_started
                            else None
                        ),
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
        lambda _url, _token, _route, *_deadline: {
            "probes": {"health": {"http_status": 200}}
        },
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
        config["n_attempts"] == 1 and config["n_concurrent_trials"] == 1
        for config in harbor.configs
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
            "harbor-compatibility.json",
            "harbor-export.log",
            "harbor-job.json",
            "harbor-jobs",
            "harbor-native-bundle.json",
            "harbor-request.json",
            "harbor.log",
            "manifest.yaml",
            "private-artifacts.json",
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
            "remote_job_id": "test-wave-job",
        }
        assert _event_payloads(execution_root / "events.jsonl") == [
            {"event": "execution_started", "execution_id": execution_root.name},
            {"event": "harbor_started"},
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


def test_provider_wave_runs_shards_without_endpoint_lifecycle(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, campaign, wave, manifest, campaign_path, wave_path = _provider_wave_inputs(
        remote_spec,
        tmp_path,
        attempts=2,
        concurrency=4,
        provider_concurrency=2,
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    runner = EndpointRunner([])
    routed_model = f"{spec.matrix.models[0].repo}:fastest"
    harbor = HarborStream(
        spec.benchmark.task_digests,
        expected_calls=2,
        synchronize=True,
        expected_base_url=None,
        expected_model_name=routed_model,
    )

    def reject_watchdog(lock: WaveLock, endpoint: EndpointRef, token: str) -> str:
        del lock, endpoint, token
        raise AssertionError("provider waves must not launch an endpoint watchdog")

    output = tmp_path / "output"
    destination = run_wave_worker(
        manifest,
        campaign_path,
        wave_path,
        output,
        runner=runner,
        stream_runner=harbor,
        source_preparer=prepare_source,
        watchdog_launcher=reject_watchdog,
        identifier=IdentifierSequence(),
    )

    assert isinstance(wave.target, ProviderWaveTarget)
    assert wave.max_concurrent_shards == 2
    assert wave.spend_cap_microusd == 2_500_000
    assert runner.commands == []
    assert harbor.max_active == 2
    assert len(set(harbor.base_urls)) == 2
    assert all(
        base_url.startswith("https://test-wave-job--8000.hf.jobs/scopes/")
        for base_url in harbor.base_urls
    )
    assert all(
        trial.trial_id not in " ".join(harbor.base_urls)
        for trial in wave.runs[0].shards[0].shard.trials
    )
    capabilities = [
        base_url.split("/scopes/", maxsplit=1)[1].removesuffix("/v1")
        for base_url in harbor.base_urls
    ]
    assert_secret_absent(output, capabilities)
    route_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output.rglob("provider-route.json")
    ]
    assert {record["capability_digest"] for record in route_records} == {
        ProviderEvidenceProxy.capability_digest(capability)
        for capability in capabilities
    }
    assert all(
        config["agents"][0]["model_name"] == f"openai/{routed_model}"
        for config in harbor.configs
    )
    assert sorted(path.name for path in destination.iterdir()) == [
        "_SUCCESS",
        "checksums.json",
        "events.jsonl",
        "provider-requests.jsonl",
        "provider-target.json",
        "runtime-environment.json",
        "wave-summary.json",
        "wave.lock.json",
    ]
    runtime = json.loads((destination / "runtime-environment.json").read_text())
    assert runtime["provider"]["request_controls"] == {
        "max_attempts": 2,
        "max_concurrent_requests": 2,
        "parameters": {},
        "timeout_seconds": 60.0,
    }
    assert runtime["provider"]["transport"] == {
        "evidence_path": "provider-requests.jsonl",
        "ingress_host": "test-wave-job--8000.hf.jobs",
        "kind": "hf-job-evidence-recorder",
        "port": 8000,
        "route_authorization": "opaque-capability",
    }
    assert (destination / "provider-requests.jsonl").read_text() == ""
    endpoint = runtime["provider"]["endpoint"]
    assert endpoint["endpoint_name"]["status"] == "not_applicable"
    assert endpoint["endpoint_status"]["status"] == "not_applicable"
    assert endpoint["ready_replicas"]["status"] == "not_applicable"
    for field in ("region", "hardware", "engine", "precision"):
        assert endpoint[field]["status"] == "not_reported"
    assert _event_payloads(destination / "events.jsonl") == [
        {"event": "wave_started", "wave_id": wave.wave_id},
        {
            "event": "provider_target_validated",
            "service": "hf-inference-providers",
            "target_id": "hf-provider",
        },
        {"event": "provider_recorder_listening", "port": 8000},
        {
            "event": "provider_recorder_ready",
            "host": "test-wave-job--8000.hf.jobs",
            "port": 8000,
        },
        {"event": "wave_succeeded"},
    ]
    summary = json.loads((destination / "wave-summary.json").read_text())
    assert summary["endpoint_cleanup_verified"] == {
        "status": "not_applicable",
        "value": None,
        "detail": None,
    }
    assert not (destination / "endpoint.snapshot.json").exists()
    assert not (destination / "endpoint.final.json").exists()


def test_terminal_wave_evidence_closes_and_normalizes_campaign(
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
        lambda _url, _token, _route, *_deadline: {
            "probes": {"health": {"http_status": 200}}
        },
    )
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
        stream_runner=HarborStream(
            spec.benchmark.task_digests, expected_calls=2, synchronize=True
        ),
        source_preparer=prepare_source,
        watchdog_launcher=launch_watchdog,
        identifier=IdentifierSequence(),
    )
    submitted = new_event(
        subject_type="campaign",
        subject_id=campaign.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=campaign.plan_digest),
        clock=lambda: campaign.created_at,
    )
    evidence = LocalEvidence(output)
    observed = BucketCampaignObserver(evidence).observe(campaign, spec)
    projection = project_recovery(campaign, [submitted, *observed])

    assert projection.terminal_decision is not None
    assert projection.terminal_decision.status == "completed"
    assert {event.kind for event in observed} >= {
        "wave.active",
        "wave.cleaning",
        "wave.closed",
        "execution.started",
        "execution.completed",
    }

    BucketCampaignFinalizer(evidence, evidence).finalize(
        campaign,
        spec,
        projection,
        projection.terminal_decision,
    )
    run = campaign.runs[0]
    tables = build_result_tables(
        evidence,
        EvidenceSource(
            bucket=spec.artifacts.bucket,
            prefix=f"{campaign.artifact_prefix}/runs/{run.run_id}",
        ),
        control_commit="c" * 40,
    )

    assert len(tables.runs) == 1
    assert len(tables.trials) == 2
    assert len(tables.executions) == 2
    assert [metric.value for metric in tables.metrics] == [1.0, 1.0]
    assert (output / campaign.artifact_prefix / "_SUCCESS").is_file()


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
        "category": "transient",
        "error_type": "WorkerError",
        "message": "Harbor exited with status 7",
    }
    assert json.loads((execution / "failure.json").read_text()) == {
        "category": "transient",
        "error_type": "WorkerError",
        "message": "Harbor exited with status 7",
    }
    assert "failure.json" in json.loads((execution / "checksums.json").read_text())
    assert _event_payloads(execution / "events.jsonl") == [
        {"event": "execution_started", "execution_id": execution.name},
        {"event": "harbor_started"},
        {"event": "harbor_finished", "exit_code": 7},
        {"event": "execution_failed", "error_type": "WorkerError"},
        {"event": "secrets_redacted", "files": ["harbor.log"]},
    ]


def test_task_local_failure_does_not_abort_wave(
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
    harbor = HarborStream(
        spec.benchmark.task_digests,
        expected_calls=1,
        exit_code=1,
        failure_exception=(
            "SandboxError",
            "HF Sandbox requires a prebuilt Docker image for this task",
        ),
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

    assert (destination / "_SUCCESS").is_file()
    assert not (destination / "_FAILED").exists()
    run = wave.runs[0]
    trial_id = run.shards[0].shard.trials[0].trial_id
    executions = output / run.artifact_prefix / "trials" / trial_id / "executions"
    execution = next(executions.iterdir())
    assert json.loads((execution / "failure.json").read_text()) == {
        "category": "benchmark",
        "error_type": "WorkerError",
        "message": "Harbor exited with status 1",
    }
    shard_root = (
        output
        / _campaign.artifact_prefix
        / "runs"
        / run.configuration.run_id
        / "shards"
        / run.shards[0].shard.shard_id
    )
    assert not (shard_root / "_SUCCESS").exists()


def test_missing_required_session_publishes_terminal_failed_evidence(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _campaign, wave, manifest, campaign_path, wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr("harbor_hf.worker.probe_runtime", lambda *_args: {"probes": {}})
    output = tmp_path / "output"

    with pytest.raises(WorkerError, match="no session JSONL"):
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
            stream_runner=HarborStream(
                spec.benchmark.task_digests,
                expected_calls=1,
                agent_started=True,
            ),
            source_preparer=prepare_source,
            watchdog_launcher=launch_watchdog,
            identifier=IdentifierSequence(),
        )

    run = wave.runs[0]
    trial_id = run.shards[0].shard.trials[0].trial_id
    executions = output / run.artifact_prefix / "trials" / trial_id / "executions"
    execution = next(executions.iterdir())
    failure = json.loads((execution / "_FAILED").read_text(encoding="utf-8"))
    private = json.loads(
        (execution / "private-artifacts.json").read_text(encoding="utf-8")
    )
    assert failure == {
        "category": "configuration",
        "error_type": "PrivateArtifactRequirementError",
        "message": "successful OpenClaw execution has no session JSONL",
    }
    assert private["requirements"] == [
        {
            "name": "openclaw_session_jsonl",
            "paths": [],
            "required": True,
            "satisfied": False,
        }
    ]
    verify_checksums(execution)
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


def test_wave_requires_git_source_secret_before_remote_work(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = GitBenchmarkSource(
        repository="ShellBench/public-tasks",
        revision="8" * 40,
        path="tasks/115-tasks",
        credentials=GitHubTokenCredentials(secret_name="GITHUB_TOKEN"),
    )
    raw = remote_spec.model_dump(mode="python")
    raw["benchmark"].update(
        {
            "dataset": "shellbench/public-115",
            "source": source.model_dump(mode="python"),
        }
    )
    raw["benchmark"].pop("dataset_digest", None)
    private_spec = ExperimentSpec.model_validate(raw)
    _spec, _campaign, _wave, manifest, campaign_path, wave_path = _wave_inputs(
        private_spec, tmp_path, attempts=1, concurrency=1
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    endpoint = EndpointRunner([])

    with pytest.raises(WorkerError, match="required secret GITHUB_TOKEN"):
        run_wave_worker(
            manifest,
            campaign_path,
            wave_path,
            tmp_path / "missing-git-token",
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
    replacement = wave.model_copy(
        update={
            "wave_id": "wave-" + "e" * 24,
            "action_id": "act-" + "e" * 24,
            "action_key": "e" * 24,
            "artifact_prefix": f"{campaign.artifact_prefix}/waves/wave-" + "e" * 24,
        }
    )
    replacement_path = tmp_path / "replacement-wave.lock.json"
    replacement_path.write_text(replacement.model_dump_json(), encoding="utf-8")

    run_wave_worker(
        manifest,
        campaign_path,
        replacement_path,
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


def test_retry_wave_accepts_a_valid_success_from_an_earlier_wave(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    run = wave.runs[0]
    shard = run.shards[0].shard
    trial = shard.trials[0]
    trial_root = output / run.artifact_prefix / "trials" / trial.trial_id

    assert _valid_terminal_trial(
        trial_root,
        trial,
        campaign_id=campaign.campaign_id,
        wave_id=None,
        run_id=run.configuration.run_id,
        shard_id=shard.shard_id,
    )
    with pytest.raises(WorkerError, match="execution identity does not match"):
        _valid_terminal_trial(
            trial_root,
            trial,
            campaign_id=campaign.campaign_id,
            wave_id="wave-from-another-submit-action",
            run_id=run.configuration.run_id,
            shard_id=shard.shard_id,
        )


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
        execution["campaign_id"] = "campaign-wrong"
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


@pytest.mark.parametrize("value", ["0" * 31, "0" * 33, "A" * 32, "X" * 32, "g" * 32])
def test_execution_identifier_rejects_every_noncanonical_shape(value: str) -> None:
    with pytest.raises(WorkerError) as captured:
        _execution_id(lambda: value)

    assert str(captured.value) == (
        "execution identifier must be 32 lowercase hexadecimal digits"
    )


def test_wave_scalar_helpers_have_exact_boundary_contracts(tmp_path: Path) -> None:
    assert _execution_id(lambda: "0123456789abcdef" * 2) == (
        "exec-0123456789abcdef0123456789abcdef"
    )
    assert _remaining_seconds(10.0, lambda: 8.01) == 2
    assert _remaining_seconds(10.0, lambda: 9.99) == 1
    with pytest.raises(WorkerError) as captured:
        _remaining_seconds(10.0, lambda: 10.0)
    assert str(captured.value) == "deployment wave duration bound was reached"

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"wave-contract\n")
    assert _file_digest(payload) == (
        "sha256:8b6a391b539bf23c01dfed62246c6e04cb057e7bd5c119318589b64df6c1b413"
    )


def test_wave_target_and_watchdog_helpers_forward_exact_locked_identity(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "endpoint").mkdir()
    (tmp_path / "provider").mkdir()
    _spec, _campaign, endpoint_wave, *_paths = _wave_inputs(
        remote_spec, tmp_path / "endpoint", attempts=1, concurrency=1
    )
    _provider_spec, _provider_campaign, provider_wave, *_provider_paths = (
        _provider_wave_inputs(
            remote_spec,
            tmp_path / "provider",
            attempts=1,
            concurrency=1,
            provider_concurrency=1,
        )
    )
    assert endpoint_wave.endpoint is not None
    calls: list[tuple[object, object, str, str]] = []

    def launch(remote: object, endpoint: object, owner_id: str, token: str) -> str:
        calls.append((remote, endpoint, owner_id, token))
        return "watchdog-contract"

    monkeypatch.setattr("harbor_hf.wave_worker.launch_cleanup_watchdog_for", launch)

    assert _wave_model_name(endpoint_wave) == endpoint_wave.endpoint.served_model_name
    assert _wave_model_name(provider_wave) == ("nvidia/Qwen3.6-35B-A3B-NVFP4:fastest")
    assert (
        _launch_wave_watchdog(endpoint_wave, endpoint_wave.endpoint, "secret")
        == "watchdog-contract"
    )
    assert calls == [
        (
            endpoint_wave.remote,
            endpoint_wave.endpoint,
            endpoint_wave.wave_id,
            "secret",
        )
    ]


def test_shard_continues_after_one_trial_fails(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, wave = _two_trial_wave(remote_spec)
    run = wave.runs[0]
    shard = run.shards[0]
    barrier = threading.Barrier(2)
    calls: list[tuple[str, float]] = []
    calls_lock = threading.Lock()

    def execute(*args: object, **kwargs: object) -> None:
        del kwargs
        trial = cast(CampaignTrialLock, args[5])
        trial_id = trial.trial_id
        trial_root = cast(Path, args[6])
        deadline = cast(float, args[12])
        with calls_lock:
            calls.append((trial_id, deadline))
        barrier.wait(timeout=5)
        if trial_id == shard.shard.trials[0].trial_id:
            raise WorkerError("first trial failed")
        (trial_root / "checksums.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("harbor_hf.wave_worker._execute_trial", execute)
    output = tmp_path / "output"
    campaign_root = tmp_path / "staging" / campaign.artifact_prefix

    with pytest.raises(WorkerError, match="^first trial failed$"):
        _execute_shard(
            tmp_path / "manifest.yaml",
            campaign,
            wave,
            run,
            shard,
            campaign_root,
            output,
            tmp_path / "harbor",
            "https://endpoint.example",
            "test-token",
            lambda *_args, **_kwargs: 0,
            100.0,
            IdentifierSequence(),
            lambda: datetime.now(UTC),
            lambda: 0.0,
        )

    assert sorted(trial_id for trial_id, _deadline in calls) == sorted(
        trial.trial_id for trial in shard.shard.trials
    )
    assert {deadline for _trial_id, deadline in calls} == {100.0}
    events = _event_payloads(
        campaign_root
        / "runs"
        / run.configuration.run_id
        / "shards"
        / shard.shard.shard_id
        / "events.jsonl"
    )
    assert events[0]["event"] == "shard_started"
    assert {event["event"] for event in events[1:3]} == {
        "trial_failed",
        "trial_completed",
    }
    assert events[3]["event"] == "shard_failed"


def test_one_shard_runs_multiple_trials_at_configured_concurrency(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, wave = _two_trial_wave(remote_spec)
    run = wave.runs[0]
    shard = run.shards[0]
    barrier = threading.Barrier(2)
    calls: list[tuple[str, float]] = []
    calls_lock = threading.Lock()

    def execute(*args: object, **kwargs: object) -> None:
        del kwargs
        trial = cast(CampaignTrialLock, args[5])
        trial_root = cast(Path, args[6])
        deadline = cast(float, args[12])
        with calls_lock:
            calls.append((trial.trial_id, deadline))
        barrier.wait(timeout=5)
        (trial_root / "checksums.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("harbor_hf.wave_worker._execute_trial", execute)
    campaign_root = tmp_path / "staging" / campaign.artifact_prefix

    checksum = _execute_shard(
        tmp_path / "manifest.yaml",
        campaign,
        wave,
        run,
        shard,
        campaign_root,
        tmp_path / "output",
        tmp_path / "harbor",
        "https://endpoint.example",
        "test-token",
        lambda *_args, **_kwargs: 0,
        100.0,
        IdentifierSequence(),
        lambda: datetime.now(UTC),
        lambda: 0.0,
    )

    assert checksum is not None
    assert sorted(trial_id for trial_id, _deadline in calls) == sorted(
        trial.trial_id for trial in shard.shard.trials
    )
    assert len(calls) == len(shard.shard.trials)
    assert {deadline for _trial_id, deadline in calls} == {100.0}
    events = _event_payloads(
        campaign_root
        / "runs"
        / run.configuration.run_id
        / "shards"
        / shard.shard.shard_id
        / "events.jsonl"
    )
    assert events[0]["event"] == "shard_started"
    assert [event["event"] for event in events[1:3]] == [
        "trial_completed",
        "trial_completed",
    ]
    assert events[3]["event"] == "shard_succeeded"


def test_retry_shard_executes_only_admitted_trials(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, initial = _two_trial_wave(remote_spec)
    shard = initial.runs[0].shards[0].shard
    selected = shard.trials[1]
    action = (
        plan_reconciliation(
            campaign,
            [
                new_event(
                    subject_type="campaign",
                    subject_id=campaign.campaign_id,
                    kind="campaign.submitted",
                    producer="cli",
                    payload=CampaignSubmittedPayload(plan_digest=campaign.plan_digest),
                )
            ],
        )[1]
        .actions[0]
        .model_copy(update={"kind": "retry-shard", "trial_ids": [selected.trial_id]})
    )
    wave = build_wave_lock(campaign, _two_trial_spec(remote_spec), action)
    run = wave.runs[0]
    locked_shard = run.shards[0]
    calls: list[str] = []

    def execute(*args: object, **kwargs: object) -> None:
        del kwargs
        trial = cast(CampaignTrialLock, args[5])
        trial_id = trial.trial_id
        trial_root = cast(Path, args[6])
        calls.append(trial_id)
        (trial_root / "checksums.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("harbor_hf.wave_worker._execute_trial", execute)
    campaign_root = tmp_path / "staging" / campaign.artifact_prefix
    result = _execute_shard(
        tmp_path / "manifest.yaml",
        campaign,
        wave,
        run,
        locked_shard,
        campaign_root,
        tmp_path / "output",
        tmp_path / "harbor",
        "https://endpoint.example",
        "test-token",
        lambda *_args, **_kwargs: 0,
        100.0,
        IdentifierSequence(),
        lambda: datetime.now(UTC),
        lambda: 0.0,
    )

    assert result is None
    assert calls == [selected.trial_id]
    shard_root = (
        campaign_root
        / "runs"
        / run.configuration.run_id
        / "shards"
        / locked_shard.shard.shard_id
    )
    assert not (shard_root / "_SUCCESS").exists()
    assert [
        event["event"] for event in _event_payloads(shard_root / "events.jsonl")
    ] == [
        "shard_started",
        "trial_deferred",
        "trial_completed",
        "shard_deferred",
    ]


def _two_trial_spec(remote_spec: ExperimentSpec) -> ExperimentSpec:
    return remote_spec.model_copy(
        update={
            "benchmark": remote_spec.benchmark.model_copy(
                update={
                    "task_names": ["task-a", "task-b"],
                    "task_digests": {
                        "task-a": "sha256:" + "a" * 64,
                        "task-b": "sha256:" + "b" * 64,
                    },
                }
            ),
            "execution": remote_spec.execution.model_copy(
                update={
                    "attempts": 1,
                    "concurrent_trials": 2,
                    "max_trials_per_shard": 2,
                }
            ),
        }
    )


def _two_trial_wave(remote_spec: ExperimentSpec) -> tuple[CampaignLock, WaveLock]:
    spec = _two_trial_spec(remote_spec)
    campaign = build_campaign_lock(build_campaign_plan(spec), "campaign-two-trials")
    submitted = new_event(
        subject_type="campaign",
        subject_id=campaign.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=campaign.plan_digest),
    )
    action = plan_reconciliation(campaign, [submitted])[1].actions[0]
    return campaign, build_wave_lock(campaign, spec, action)


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


def _provider_wave_inputs(
    remote_spec: ExperimentSpec,
    root: Path,
    *,
    attempts: int,
    concurrency: int,
    provider_concurrency: int,
) -> tuple[ExperimentSpec, CampaignLock, WaveLock, Path, Path, Path]:
    model = remote_spec.matrix.models[0]
    target = ProviderTarget(
        id="hf-provider",
        model=model.repo,
        limits=ProviderLimits(
            max_concurrent_requests=provider_concurrency,
            max_attempts=2,
            max_spend_usd=Decimal("2.50"),
            estimated_wave_cost_usd=Decimal("1.00"),
        ),
    )
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(update={"deployments": [target]}),
            "execution": remote_spec.execution.model_copy(
                update={
                    "attempts": attempts,
                    "concurrent_trials": concurrency,
                    "max_trials_per_shard": 1,
                }
            ),
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
    context = ReconcileContext(
        deployments={
            campaign.runs[0].deployment_digest: DeploymentAdmission(
                estimated_wave_cost_microusd=1_000_000
            )
        }
    )
    action = plan_reconciliation(campaign, [submitted], context=context)[1].actions[0]
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
