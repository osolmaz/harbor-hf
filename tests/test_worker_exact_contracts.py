from __future__ import annotations

import inspect
import re
import shutil
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace, TracebackType

import pytest

from harbor_hf import worker
from harbor_hf.models import ExperimentSpec
from harbor_hf.runs import build_run_lock
from harbor_hf.worker import (
    EndpointManager,
    WorkerError,
    build_harbor_trial_command,
    endpoint_health_route,
    endpoint_state,
    endpoint_url,
    require_executable,
)


class RecordingRunner:
    def __init__(self, results: Sequence[dict[str, object]]) -> None:
        self.results = list(results)
        self.calls: list[tuple[list[str], float | None]] = []

    def run_json(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        self.calls.append((list(command), timeout_seconds))
        return self.results.pop(0)

    def run_text(self, command: Sequence[str]) -> str:
        raise AssertionError("run_text must not be used by EndpointManager")


class SequenceClock:
    def __init__(self, values: Sequence[float]) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        if len(self.values) > 1:
            return self.values.pop(0)
        return self.values[0]


def _snapshot(state: str, ready: int, target: int = 1) -> dict[str, object]:
    return {"status": {"state": state, "readyReplica": ready, "targetReplica": target}}


def test_endpoint_manager_sends_exact_cli_commands_with_default_timeout() -> None:
    runner = RecordingRunner([{"a": 1}, {"b": 2}, {"c": 3}])
    manager = EndpointManager("ns-exact", "name-exact", runner)

    assert manager.describe() == {"a": 1}
    assert manager.resume() == {"b": 2}
    assert manager.pause() == {"c": 3}
    expected = [
        (
            [
                "hf",
                "endpoints",
                operation,
                "name-exact",
                "--namespace",
                "ns-exact",
                "--format",
                "json",
            ],
            60.0,
        )
        for operation in ("describe", "resume", "pause")
    ]
    assert runner.calls == expected


def test_wait_ready_forwards_remaining_bounded_timeouts_and_default_poll() -> None:
    runner = RecordingRunner(
        [_snapshot("initializing", 5, 1), _snapshot("running", 2, 2)]
    )
    sleeps: list[float] = []
    manager = EndpointManager(
        "ns",
        "ep",
        runner,
        sleep=sleeps.append,
        monotonic=SequenceClock([0.0, 10.0, 20.0, 50.0]),
    )

    snapshot = manager.wait_ready(90)

    assert snapshot == _snapshot("running", 2, 2)
    assert [timeout for _, timeout in runner.calls] == [60.0, 40.0]
    assert sleeps == [15]


def test_wait_ready_requires_positive_target_before_returning() -> None:
    runner = RecordingRunner([_snapshot("running", 1, 0)])
    manager = EndpointManager(
        "ns",
        "ep",
        runner,
        sleep=lambda _: None,
        monotonic=SequenceClock([0.0, 10.0, 95.0]),
    )

    with pytest.raises(
        WorkerError,
        match=re.escape(
            "endpoint readiness timed out in state='running', ready=1, target=0"
        ),
    ):
        manager.wait_ready(90)


def test_wait_ready_raises_before_describing_at_exhausted_deadline() -> None:
    runner = RecordingRunner([])
    manager = EndpointManager(
        "ns",
        "ep",
        runner,
        sleep=lambda _: None,
        monotonic=SequenceClock([0.0, 90.0]),
    )

    with pytest.raises(
        WorkerError, match="^endpoint readiness timed out before status check$"
    ):
        manager.wait_ready(90)
    assert runner.calls == []


def test_pause_and_verify_returns_snapshot_only_at_zero_ready() -> None:
    runner = RecordingRunner([{}, _snapshot("paused", 0, 0)])
    manager = EndpointManager(
        "ns",
        "ep",
        runner,
        sleep=lambda _: None,
        monotonic=SequenceClock([0.0, 1.0, 2.0, 3.0]),
    )

    assert manager.pause_and_verify() == _snapshot("paused", 0, 0)
    assert [command[2] for command, _ in runner.calls] == ["pause", "describe"]


def test_pause_and_verify_times_out_with_exact_state_message() -> None:
    runner = RecordingRunner([{}, _snapshot("paused", 1, 1)])
    manager = EndpointManager(
        "ns",
        "ep",
        runner,
        sleep=lambda _: None,
        monotonic=SequenceClock([0.0, 1.0, 2.0, 1000.0]),
    )

    with pytest.raises(
        WorkerError,
        match=re.escape("endpoint cleanup timed out in state='paused', ready=1"),
    ):
        manager.pause_and_verify()


def test_operation_timeout_clamps_between_floor_and_call_ceiling() -> None:
    manager = EndpointManager("ns", "ep", RecordingRunner([]), monotonic=lambda: 10.0)

    assert manager._operation_timeout(None) == 60.0
    assert manager._operation_timeout(100.0) == 60.0
    assert manager._operation_timeout(40.0) == 30.0
    assert manager._operation_timeout(10.0) == 0.001
    assert manager._operation_timeout(5.0) == 0.001


def test_endpoint_state_returns_exact_tuple_and_defaults_target() -> None:
    assert endpoint_state(_snapshot("pausing", 3, 5)) == ("pausing", 3, 5)
    assert endpoint_state(
        {"status": {"state": "paused", "readyReplica": 2, "targetReplica": "2"}}
    ) == ("paused", 2, 0)
    assert endpoint_state({"status": {"state": "paused", "readyReplica": 2}}) == (
        "paused",
        2,
        0,
    )


def test_endpoint_state_rejects_malformed_status_with_exact_messages() -> None:
    with pytest.raises(WorkerError, match="^endpoint response has no status object$"):
        endpoint_state({"status": "running"})
    for status in (
        {"state": 5, "readyReplica": 1},
        {"state": "running", "readyReplica": "1"},
        {"readyReplica": 1},
        {"state": "running"},
    ):
        with pytest.raises(
            WorkerError, match="^endpoint status is missing state or readyReplica$"
        ):
            endpoint_state({"status": status})


def test_endpoint_url_strips_only_trailing_slashes() -> None:
    assert (
        endpoint_url({"status": {"url": "https://h.example/x//"}})
        == "https://h.example/x"
    )
    assert (
        endpoint_url({"status": {"url": "https://h.example/x"}})
        == "https://h.example/x"
    )
    for snapshot in ({}, {"status": {"url": 5}}, {"status": {}}):
        with pytest.raises(WorkerError, match="^endpoint status is missing its URL$"):
            endpoint_url(snapshot)


def test_endpoint_health_route_prefers_top_level_then_custom_image() -> None:
    assert endpoint_health_route({"healthRoute": "/health-top"}) == "/health-top"
    assert (
        endpoint_health_route(
            {
                "healthRoute": 5,
                "model": {"image": {"custom": {"healthRoute": "/nested"}}},
            }
        )
        == "/nested"
    )


@pytest.mark.parametrize(
    "route",
    [
        "health",
        "https://h.example/health",
        "//netloc/health",
        "/health?probe=1",
        "/health#fragment",
        "",
    ],
)
def test_endpoint_health_route_rejects_every_nonlocal_shape(route: str) -> None:
    with pytest.raises(
        WorkerError, match="^endpoint response has no valid health route$"
    ):
        endpoint_health_route({"healthRoute": route})


def test_endpoint_health_route_rejects_missing_nested_route() -> None:
    for snapshot in (
        {},
        {"model": {"image": {"custom": {}}}},
        {"model": {"image": {"custom": {"healthRoute": 7}}}},
        {"model": {"image": "custom"}},
        {"model": "image"},
    ):
        with pytest.raises(
            WorkerError, match="^endpoint response has no valid health route$"
        ):
            endpoint_health_route(snapshot)


def test_trial_command_rejects_task_outside_resolved_set(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = build_run_lock(remote_spec, run_id="exact-contract")

    with pytest.raises(
        WorkerError, match="^wave trial is not in the resolved run task set$"
    ):
        build_harbor_trial_command(
            lock,
            tmp_path,
            "https://unused.example",
            tmp_path,
            task_name="task-not-resolved",
        )


def test_job_stage_uppercases_reported_value() -> None:
    info = SimpleNamespace(status=SimpleNamespace(stage="running"))
    assert worker._job_stage(info) == "RUNNING"


def test_require_executable_reports_exact_missing_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(
        WorkerError,
        match="^required controller executable is missing: tool-exact$",
    ):
        require_executable("tool-exact")


class _Response:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body
        self.reads: list[int] = []

    def read(self, size: int) -> bytes:
        self.reads.append(size)
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False


def test_probe_runtime_sends_exact_authenticated_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        "https://probe.example/hp": _Response(200, b'{"ok": true}'),
        "https://probe.example/version": _Response(201, b"not-json"),
        "https://probe.example/v1/models": _Response(203, b'["m"]'),
    }
    captured: list[tuple[str, str | None, float]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        url = request.full_url
        header = request.get_header("Authorization")
        captured.append((url, header, timeout))
        return responses[url]

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = worker.probe_runtime("https://probe.example", "token-exact", "/hp")

    assert captured == [
        ("https://probe.example/hp", "Bearer token-exact", 60),
        ("https://probe.example/version", "Bearer token-exact", 60),
        ("https://probe.example/v1/models", "Bearer token-exact", 60),
    ]
    assert all(response.reads == [1024 * 1024] for response in responses.values())
    assert result == {
        "probes": {
            "health": {"status": "reported", "http_status": 200, "value": {"ok": True}},
            "version": {"status": "reported", "http_status": 201, "value": "not-json"},
            "models": {"status": "reported", "http_status": 203, "value": ["m"]},
        }
    }


def test_worker_defaults_and_constants_are_exact() -> None:
    def default(func: Callable[..., object], name: str) -> object:
        return inspect.signature(func).parameters[name].default

    assert default(EndpointManager.describe, "timeout_seconds") == 60.0
    assert default(EndpointManager.resume, "timeout_seconds") == 60.0
    assert default(EndpointManager.pause, "timeout_seconds") == 60.0
    assert default(EndpointManager.wait_ready, "poll_seconds") == 15
    assert default(EndpointManager.pause_and_verify, "timeout_seconds") == 300
    assert default(EndpointManager.pause_and_verify, "poll_seconds") == 10
    assert default(worker.wait_watchdog_ready, "poll_seconds") == 5
    assert default(worker.run_endpoint_watchdog, "poll_seconds") == 10
    assert (
        default(worker.resume_and_probe_endpoint, "readiness_timeout_seconds") == 3600
    )
    assert default(worker.probe_runtime, "health_route") == "/health"
    assert default(worker.validate_harbor_result, "expected_trials") == 1
    assert worker._WATCHDOG_READY_LABEL == "harbor-hf-watchdog-ready"
    assert worker._WATCHDOG_STARTUP_TIMEOUT_SECONDS == 300
    assert worker._ENDPOINT_CALL_TIMEOUT_SECONDS == 60.0
