from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from test_wave_worker import _provider_wave_inputs, _wave_inputs

import harbor_hf.wave_worker as wave_worker
import harbor_hf.worker as worker
from harbor_hf.models import DeploymentProfile, EndpointRef, ExperimentSpec, SourcePin
from harbor_hf.process import CommandRunner
from harbor_hf.provider_models import unavailable
from harbor_hf.runs import RunLock, build_run_lock
from harbor_hf.wave_worker import _EndpointWaveLifecycle
from harbor_hf.worker import WorkerError, _prepare_evidence_destination


def _events(path: Path) -> list[dict[str, object]]:
    return [
        {key: value for key, value in json.loads(line).items() if key != "at"}
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class _UnusedRunner:
    def run_json(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        raise AssertionError((command, timeout_seconds))

    def run_text(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        raise AssertionError((command, timeout_seconds))


def test_staged_worker_success_has_exact_ordered_side_effects(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="staged-worker-contract")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("manifest-contract\n", encoding="utf-8")
    root = tmp_path / "staging" / "run"
    destination = tmp_path / "published" / "run"
    _prepare_evidence_destination(destination)
    runner = _UnusedRunner()
    calls: list[tuple[object, ...]] = []
    baseline: dict[str, object] = {"snapshot": "baseline"}
    final: dict[str, object] = {
        "status": {
            "state": "paused",
            "readyReplica": 0,
            "targetReplica": 1,
        },
        "apiToken": "must-redact",
    }

    class Manager:
        def __init__(
            self, namespace: str, name: str, process_runner: CommandRunner
        ) -> None:
            calls.append(("manager", namespace, name, process_runner))

        def describe(self) -> dict[str, object]:
            calls.append(("describe",))
            return baseline

        def pause_and_verify(self) -> dict[str, object]:
            calls.append(("pause_and_verify",))
            return final

    monkeypatch.setattr(worker, "EndpointManager", Manager)
    monkeypatch.setattr(
        worker, "require_executable", lambda name: calls.append(("require", name))
    )
    monkeypatch.setattr(
        worker,
        "validate_endpoint_model",
        lambda candidate, snapshot: calls.append(("validate", candidate, snapshot)),
    )
    monkeypatch.setattr(
        worker,
        "require_paused_endpoint",
        lambda snapshot: calls.append(("require_paused", snapshot)),
    )
    monkeypatch.setattr(
        worker,
        "_execute_benchmark",
        lambda *args: calls.append(("execute", *args)),
    )
    monkeypatch.setattr(
        worker,
        "_finalize_evidence",
        lambda candidate, token, **options: calls.append(
            ("finalize", candidate, token, options)
        ),
    )

    def prepare(
        source: SourcePin, destination_path: Path, process_runner: CommandRunner
    ) -> None:
        calls.append(("source", source, destination_path, process_runner))

    def launch(candidate: RunLock, endpoint: EndpointRef, token: str) -> str:
        calls.append(("watchdog", candidate, endpoint, token))
        return "watchdog-contract"

    result = worker._run_staged_worker(
        manifest,
        lock,
        root,
        destination,
        "contract-token",
        runner=runner,
        stream_runner=lambda *args, **kwargs: 0,
        source_preparer=prepare,
        watchdog_launcher=launch,
    )

    assert isinstance(lock.deployment, DeploymentProfile)
    endpoint = lock.deployment.endpoint
    assert endpoint is not None
    harbor_source = (
        root.parent / "sources" / (f"harbor-{lock.remote.harbor.source.revision}")
    )
    assert result == destination
    assert calls[:7] == [
        ("manager", endpoint.namespace, endpoint.name, runner),
        ("require", "git"),
        ("source", lock.remote.harbor.source, harbor_source, runner),
        ("describe",),
        ("validate", lock, baseline),
        ("require_paused", baseline),
        ("watchdog", lock, endpoint, "contract-token"),
    ]
    assert calls[7][0] == "execute"
    assert calls[7][1:6] == (
        root,
        root / "events.jsonl",
        lock,
        calls[7][4],
        "contract-token",
    )
    assert calls[8:] == [
        ("pause_and_verify",),
        ("finalize", root, "contract-token", {"strict_compatibility": True}),
    ]
    assert (root / "harbor-jobs").is_dir()
    assert (root / "manifest.yaml").read_text(encoding="utf-8") == (
        "manifest-contract\n"
    )
    assert json.loads((root / "run.lock.json").read_text(encoding="utf-8")) == (
        lock.model_dump(mode="json")
    )
    assert json.loads((root / "endpoint.final.json").read_text(encoding="utf-8")) == {
        "apiToken": "[REDACTED]",
        "status": final["status"],
    }
    assert _events(destination / "events.jsonl") == [
        {"event": "worker_started", "run_id": lock.run_id},
        {"event": "endpoint_baseline_validated"},
        {
            "event": "endpoint_lease_acquired",
            "watchdog_job_id": "watchdog-contract",
        },
        {"event": "cleanup_watchdog_started", "job_id": "watchdog-contract"},
        {"event": "endpoint_pause_requested"},
        {
            "event": "endpoint_paused",
            "state": "paused",
            "ready_replicas": 0,
            "target_replicas": 1,
        },
        {"event": "run_succeeded"},
    ]
    assert (destination / "_SUCCESS").read_text(encoding="utf-8") == "\n"
    assert not (destination / "_RESERVED").exists()


def test_staged_worker_failure_before_lease_skips_cleanup_and_redacts_publication(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = build_run_lock(remote_spec, run_id="staged-worker-failure")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("manifest\n", encoding="utf-8")
    root = tmp_path / "stage" / "run"
    destination = tmp_path / "output" / "run"
    _prepare_evidence_destination(destination)
    finalized: list[tuple[Path, str]] = []

    class Manager:
        def __init__(self, *args: object) -> None:
            pass

        def pause_and_verify(self) -> dict[str, object]:
            raise AssertionError("cleanup is forbidden before watchdog ownership")

    monkeypatch.setattr(worker, "EndpointManager", Manager)
    monkeypatch.setattr(
        worker,
        "require_executable",
        lambda name: (_ for _ in ()).throw(ValueError("bad contract-token")),
    )
    monkeypatch.setattr(
        worker,
        "_finalize_evidence",
        lambda candidate, token, **_options: finalized.append((candidate, token)),
    )

    with pytest.raises(WorkerError) as captured:
        worker._run_staged_worker(
            manifest,
            lock,
            root,
            destination,
            "contract-token",
            runner=_UnusedRunner(),
            stream_runner=lambda *args, **kwargs: 0,
            source_preparer=None,
            watchdog_launcher=None,
        )

    assert str(captured.value) == "bad [REDACTED]"
    assert isinstance(captured.value.__cause__, ValueError)
    assert finalized == [(root, "contract-token")]
    assert _events(destination / "events.jsonl") == [
        {"event": "worker_started", "run_id": lock.run_id},
        {"event": "endpoint_cleanup_skipped", "reason": "lease_not_owned"},
        {"event": "run_failed", "error_type": "ValueError"},
    ]
    assert json.loads((destination / "_FAILED").read_text(encoding="utf-8")) == {
        "error_type": "ValueError",
        "message": "bad [REDACTED]",
    }
    assert not (destination / "_SUCCESS").exists()


def test_staged_provider_wave_finalizes_then_publishes_exact_success(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spec, campaign, wave, manifest, _campaign_path, _wave_path = _provider_wave_inputs(
        remote_spec,
        tmp_path,
        attempts=1,
        concurrency=1,
        provider_concurrency=1,
    )
    campaign_root = tmp_path / "stage" / campaign.artifact_prefix
    output_root = tmp_path / "output"
    calls: list[tuple[object, ...]] = []
    shard_kwargs: dict[str, object] = {}
    proxy = object()
    checksums = {"shard-z": "sha256:z", "shard-a": "sha256:a"}

    monkeypatch.setattr(
        wave_worker,
        "require_executable",
        lambda name: calls.append(("require", name)),
    )

    def prepare_transport(*args: object) -> tuple[str, object]:
        calls.append(("transport", *args))
        return "http://127.0.0.1:4321", proxy

    def execute_shards(*args: object, **kwargs: object) -> dict[str, str]:
        calls.append(("shards", *args))
        shard_kwargs.update(kwargs)
        return checksums

    def cleanup(
        lifecycle: object, provider_proxy: object, judge_recorder: object
    ) -> None:
        calls.append(("cleanup", lifecycle, provider_proxy, judge_recorder))
        return None

    def finalize(root: Path, token: str) -> None:
        calls.append(("finalize", root, token, (root / "_SUCCESS").exists()))
        assert (root / "wave-summary.json").is_file()

    def publish(source: Path, destination: Path) -> None:
        calls.append(("publish", source, destination, (source / "_SUCCESS").is_file()))

    monkeypatch.setattr(wave_worker, "_prepare_wave_transport", prepare_transport)
    monkeypatch.setattr(
        wave_worker, "_prepare_judge_transport", lambda *args: (None, None)
    )
    monkeypatch.setattr(wave_worker, "_execute_shards", execute_shards)
    monkeypatch.setattr(wave_worker, "_cleanup_wave_transport", cleanup)
    monkeypatch.setattr(wave_worker, "_finalize_unit", finalize)
    monkeypatch.setattr(wave_worker, "_publish_unit", publish)

    def source_preparer(
        source: SourcePin, destination: Path, runner: CommandRunner
    ) -> None:
        calls.append(("source", source, destination, runner))

    result = wave_worker._run_staged_wave(
        manifest,
        campaign,
        wave,
        campaign_root,
        output_root,
        "contract-token",
        _UnusedRunner(),
        lambda *args, **kwargs: 0,
        source_preparer,
        None,
        lambda: "0" * 32,
        lambda: wave.created_at,
        lambda: 100.0,
    )

    wave_root = campaign_root / "waves" / wave.wave_id
    assert result == output_root / wave.artifact_prefix
    assert [call[0] for call in calls] == [
        "require",
        "source",
        "transport",
        "shards",
        "cleanup",
        "finalize",
        "publish",
    ]
    assert calls[0] == ("require", "git")
    assert calls[1][1] == wave.remote.harbor.source
    assert calls[1][2] == (
        campaign_root.parent
        / "sources"
        / f"harbor-{wave.remote.harbor.source.revision}"
    )
    assert calls[4] == ("cleanup", None, proxy, None)
    assert shard_kwargs == {
        "provider_proxy": proxy,
        "judge_recorder": None,
        "judge_base_url": None,
    }
    assert calls[5] == ("finalize", wave_root, "contract-token", False)
    assert calls[6] == (
        "publish",
        wave_root,
        output_root / wave.artifact_prefix,
        True,
    )
    assert _events(wave_root / "events.jsonl") == [
        {"event": "wave_started", "wave_id": wave.wave_id},
        {"event": "wave_succeeded"},
    ]
    assert json.loads((wave_root / "wave-summary.json").read_text()) == {
        "wave_id": wave.wave_id,
        "campaign_id": campaign.campaign_id,
        "shard_checksums": checksums,
        "endpoint_cleanup_verified": unavailable("not_applicable").model_dump(
            mode="json"
        ),
    }
    assert (wave_root / "_SUCCESS").read_text(encoding="utf-8") == "\n"
    assert not (wave_root / "_FAILED").exists()


def test_staged_endpoint_wave_preserves_primary_and_cleanup_failures(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spec, campaign, wave, manifest, _campaign_path, _wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=1, concurrency=1
    )
    campaign_root = tmp_path / "stage" / campaign.artifact_prefix
    output_root = tmp_path / "output"
    calls: list[tuple[object, ...]] = []

    class Lifecycle:
        owned = True

        def cleanup(self) -> Exception:
            calls.append(("cleanup",))
            return RuntimeError("cleanup contract-token")

    lifecycle = Lifecycle()
    monkeypatch.setattr(
        wave_worker,
        "_EndpointWaveLifecycle",
        lambda *args: lifecycle,
    )
    monkeypatch.setattr(wave_worker, "require_executable", lambda name: None)
    monkeypatch.setattr(
        wave_worker,
        "_prepare_wave_transport",
        lambda *args: (_ for _ in ()).throw(ValueError("primary contract-token")),
    )
    monkeypatch.setattr(
        wave_worker,
        "_finalize_unit",
        lambda root, token: calls.append(("finalize", root, token)),
    )
    monkeypatch.setattr(
        wave_worker,
        "_publish_unit",
        lambda source, destination: calls.append(("publish", source, destination)),
    )

    with pytest.raises(WorkerError) as captured:
        wave_worker._run_staged_wave(
            manifest,
            campaign,
            wave,
            campaign_root,
            output_root,
            "contract-token",
            _UnusedRunner(),
            lambda *args, **kwargs: 0,
            lambda source, destination, runner: None,
            None,
            lambda: "0" * 32,
            lambda: wave.created_at,
            lambda: 100.0,
        )

    wave_root = campaign_root / "waves" / wave.wave_id
    assert str(captured.value) == (
        "primary [REDACTED]; endpoint cleanup failed: cleanup [REDACTED]"
    )
    assert isinstance(captured.value.__cause__, ValueError)
    assert calls == [
        ("cleanup",),
        ("finalize", wave_root, "contract-token"),
        ("publish", wave_root, output_root / wave.artifact_prefix),
    ]
    summary = {
        "wave_id": wave.wave_id,
        "campaign_id": campaign.campaign_id,
        "shard_checksums": {},
        "endpoint_cleanup_verified": False,
        "error_type": "ValueError",
        "message": "primary [REDACTED]",
        "cleanup_error": {
            "error_type": "RuntimeError",
            "message": "cleanup [REDACTED]",
        },
    }
    assert json.loads((wave_root / "wave-summary.json").read_text()) == summary
    assert json.loads((wave_root / "_FAILED").read_text()) == summary
    assert _events(wave_root / "events.jsonl") == [
        {"event": "wave_started", "wave_id": wave.wave_id},
        {"event": "wave_failed", "error_type": "ValueError"},
    ]
    assert not (wave_root / "_SUCCESS").exists()


def test_endpoint_wave_prepare_validates_every_run_before_lease_and_resume(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spec, _campaign, wave, _manifest, _campaign_path, _wave_path = _wave_inputs(
        remote_spec, tmp_path, attempts=2, concurrency=1
    )
    wave_root = tmp_path / "wave"
    wave_root.mkdir()
    events = wave_root / "events.jsonl"
    calls: list[tuple[object, ...]] = []
    baseline: dict[str, object] = {"baseline": True}

    class Manager:
        def describe(self) -> dict[str, object]:
            calls.append(("describe",))
            return baseline

    def launch(lock: object, endpoint: object, token: str) -> str:
        calls.append(("launch", lock, endpoint, token))
        return "watchdog-wave"

    lifecycle = _EndpointWaveLifecycle(
        wave,
        wave_root,
        events,
        _UnusedRunner(),
        "contract-token",
        launch,
    )
    lifecycle.manager = cast(Any, Manager())
    monkeypatch.setattr(
        wave_worker,
        "validate_endpoint_model",
        lambda run, snapshot: calls.append(("validate", run, snapshot)),
    )
    monkeypatch.setattr(
        wave_worker,
        "require_paused_endpoint",
        lambda snapshot: calls.append(("paused", snapshot)),
    )

    def resume(
        root: Path,
        event_path: Path,
        run: RunLock,
        manager: object,
        token: str,
        *,
        readiness_timeout_seconds: int,
        compatible_locks: Sequence[RunLock],
    ) -> str:
        calls.append(
            (
                "resume",
                root,
                event_path,
                run,
                manager,
                token,
                readiness_timeout_seconds,
                compatible_locks,
            )
        )
        return "https://endpoint.example"

    monkeypatch.setattr(wave_worker, "resume_and_probe_endpoint", resume)

    assert lifecycle.prepare(1010.2, lambda: 1000.0) == ("https://endpoint.example")

    assert lifecycle.owned is True
    assert calls == [
        ("describe",),
        *[("validate", run.configuration, baseline) for run in wave.runs],
        ("paused", baseline),
        ("launch", wave, lifecycle.endpoint, "contract-token"),
        (
            "resume",
            wave_root,
            events,
            wave.runs[0].configuration,
            lifecycle.manager,
            "contract-token",
            11,
            tuple(run.configuration for run in wave.runs[1:]),
        ),
    ]
    assert _events(events) == [
        {"event": "endpoint_baseline_validated"},
        {
            "event": "endpoint_lease_acquired",
            "watchdog_job_id": "watchdog-wave",
        },
        {"event": "cleanup_watchdog_started", "job_id": "watchdog-wave"},
    ]
