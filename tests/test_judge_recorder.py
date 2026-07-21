from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import httpx
import pytest

from harbor_hf.judge_recorder import (
    JudgeEvidenceRecorder,
    JudgeRecorderError,
    verify_judge_exchange,
    verify_judge_recorder_summary,
)
from harbor_hf.models import TrialEvidencePolicy


def _policy() -> TrialEvidencePolicy:
    return TrialEvidencePolicy(
        workspace_root="/app",
        workspace_max_nodes=1000,
        workspace_max_file_bytes=1024 * 1024,
        workspace_max_total_bytes=8 * 1024 * 1024,
        workspace_max_archive_bytes=8 * 1024 * 1024,
        workspace_capture_timeout_seconds=60,
        judge_max_request_bytes=1024 * 1024,
        judge_max_response_bytes=1024 * 1024,
        judge_timeout_seconds=300,
        judge_max_calls_per_execution=4,
    )


def _request(url: str, body: dict[str, object]) -> tuple[int, bytes, Message[str, str]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer verifier-placeholder",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers


def test_records_exact_bodies_and_enforces_model(tmp_path: Path) -> None:
    observed: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json", "X-Request-ID": "req-1"},
            content=b'{"choices":[{"message":{"content":"ok"}}]}',
        )

    recorder = JudgeEvidenceRecorder(
        token="real-secret-token",
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        capability_factory=lambda: "a" * 32,
    )
    base = recorder.start(host="127.0.0.1", port=0)
    destination = tmp_path / "judge"
    capability = recorder.register_scope(
        execution_id="exec",
        trial_id="trial",
        model="locked/judge",
        destination=destination,
        policy=_policy(),
    )
    try:
        status, content, headers = _request(
            recorder.scoped_url(base, capability),
            {"model": "wrong", "messages": [{"role": "user", "content": "grade"}]},
        )
    finally:
        recorder.close()
    assert status == 200
    assert content == b'{"choices":[{"message":{"content":"ok"}}]}'
    assert headers["X-Harbor-Judge-Exchange-ID"] == "judge-0001"
    assert json.loads(observed[0].content)["model"] == "locked/judge"
    assert observed[0].extensions["timeout"] == {
        "connect": 300,
        "read": 300,
        "write": 300,
        "pool": 300,
    }
    exchange = destination / "judge-0001"
    metadata = json.loads((exchange / "exchange.json").read_text())
    assert metadata["transformation"] == "model_enforced"
    assert metadata["upstream_request_id"] == "req-1"
    assert (exchange / "request-received.bin").read_bytes() != (
        exchange / "request-forwarded.bin"
    ).read_bytes()
    assert "real-secret-token" not in "".join(
        path.read_text(errors="ignore") for path in exchange.iterdir()
    )
    assert verify_judge_exchange(exchange).exchange_id == "judge-0001"
    unexpected = exchange / "unexpected"
    unexpected.mkdir()
    with pytest.raises(JudgeRecorderError, match="unsupported entry"):
        verify_judge_exchange(exchange)
    unexpected.rmdir()
    (exchange / "response-delivered.bin").write_bytes(b"tampered")
    with pytest.raises(JudgeRecorderError, match="digest mismatch"):
        verify_judge_exchange(exchange)


def test_rejects_streaming_without_upstream_call(tmp_path: Path) -> None:
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"{}")

    recorder = JudgeEvidenceRecorder(
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        capability_factory=lambda: "b" * 32,
    )
    base = recorder.start(host="127.0.0.1", port=0)
    capability = recorder.register_scope(
        execution_id="exec",
        trial_id="trial",
        model="judge",
        destination=tmp_path / "judge",
        policy=_policy(),
    )
    try:
        status, _, _ = _request(
            recorder.scoped_url(base, capability),
            {"model": "judge", "stream": True, "messages": []},
        )
    finally:
        recorder.close()
    assert status == 502
    assert calls == 0


def test_rejects_untrusted_routing_fields_without_upstream_call(tmp_path: Path) -> None:
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"{}")

    recorder = JudgeEvidenceRecorder(
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        capability_factory=lambda: "r" * 32,
    )
    base = recorder.start(host="127.0.0.1", port=0)
    capability = recorder.register_scope(
        execution_id="exec",
        trial_id="trial",
        model="judge",
        destination=tmp_path / "judge",
        policy=_policy(),
    )
    try:
        status, _, _ = _request(
            recorder.scoped_url(base, capability),
            {"model": "judge", "messages": [], "provider": "untrusted"},
        )
    finally:
        recorder.close()

    assert status == 502
    assert calls == 0


def test_known_secret_in_prompt_fails_closed(tmp_path: Path) -> None:
    recorder = JudgeEvidenceRecorder(
        token="upstream-token",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"{}")
            )
        ),
        capability_factory=lambda: "c" * 32,
    )
    base = recorder.start(host="127.0.0.1", port=0)
    capability = recorder.register_scope(
        execution_id="exec",
        trial_id="trial",
        model="judge",
        destination=tmp_path / "judge",
        policy=_policy(),
        known_secrets=("do-not-store",),
    )
    try:
        status, body, _ = _request(
            recorder.scoped_url(base, capability),
            {
                "model": "judge",
                "messages": [{"role": "user", "content": "do-not-store"}],
            },
        )
    finally:
        recorder.revoke_scope(capability)
        recorder.close()
    assert status == 502
    assert json.loads(body) == {"error": "judge recorder rejected request"}
    summary = verify_judge_recorder_summary(tmp_path / "judge" / "recorder.json")
    assert summary.exchange_count == 0
    assert summary.rejected_call_count == 1


def test_records_transport_errors_and_delivered_response(tmp_path: Path) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unavailable", request=request)

    recorder = JudgeEvidenceRecorder(
        token="token",
        client=httpx.Client(transport=httpx.MockTransport(upstream)),
        capability_factory=lambda: "e" * 32,
    )
    base = recorder.start(host="127.0.0.1", port=0)
    destination = tmp_path / "judge"
    capability = recorder.register_scope(
        execution_id="exec",
        trial_id="trial",
        model="judge",
        destination=destination,
        policy=_policy(),
    )
    try:
        status, body, headers = _request(
            recorder.scoped_url(base, capability),
            {"model": "judge", "messages": []},
        )
    finally:
        recorder.close()
    assert status == 502
    assert json.loads(body) == {"error": "judge exchange failed"}
    assert headers["X-Harbor-Judge-Exchange-ID"] == "judge-0001"
    exchange = verify_judge_exchange(destination / "judge-0001")
    assert exchange.outcome == "transport_error"
    assert exchange.upstream_http_status is None
    assert exchange.response_upstream is None
    assert exchange.delivered_http_status == 502
    assert exchange.error_type == "ConnectError"
    assert exchange.error_message == "upstream judge transport failed"
    assert (destination / "judge-0001" / "response-delivered.bin").read_bytes() == body


def test_absolute_deadline_stops_streamed_judge_response(tmp_path: Path) -> None:
    times = iter([0.0, 11.0])
    recorder = JudgeEvidenceRecorder(
        token="token",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, stream=httpx.ByteStream(b"{}"), request=request
                )
            )
        ),
        capability_factory=lambda: "g" * 32,
        deadline=10.0,
        monotonic=lambda: next(times),
    )
    base = recorder.start(host="127.0.0.1", port=0)
    destination = tmp_path / "judge"
    capability = recorder.register_scope(
        execution_id="exec",
        trial_id="trial",
        model="judge",
        destination=destination,
        policy=_policy(),
    )
    try:
        status, _, _ = _request(
            recorder.scoped_url(base, capability),
            {"model": "judge", "messages": []},
        )
    finally:
        recorder.close()

    assert status == 502
    exchange = verify_judge_exchange(destination / "judge-0001")
    assert exchange.outcome == "recorder_error"
    assert exchange.error_type == "JudgeRecorderError"
    assert exchange.error_message == "judge evidence recorder failed"


def test_response_limit_stops_and_records_failed_exchange(tmp_path: Path) -> None:
    recorder = JudgeEvidenceRecorder(
        token="token",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=b"oversized", request=request
                )
            )
        ),
        capability_factory=lambda: "f" * 32,
    )
    base = recorder.start(host="127.0.0.1", port=0)
    destination = tmp_path / "judge"
    capability = recorder.register_scope(
        execution_id="exec",
        trial_id="trial",
        model="judge",
        destination=destination,
        policy=_policy().model_copy(update={"judge_max_response_bytes": 4}),
    )
    try:
        status, _, _ = _request(
            recorder.scoped_url(base, capability),
            {"model": "judge", "messages": []},
        )
    finally:
        recorder.revoke_scope(capability)
        recorder.close()
    assert status == 502
    exchange = verify_judge_exchange(destination / "judge-0001")
    assert exchange.outcome == "recorder_error"
    assert exchange.response_upstream is None
    assert exchange.error_type == "JudgeRecorderError"


def test_decoded_response_limit_blocks_compression_expansion(tmp_path: Path) -> None:
    compressed = gzip.compress(b"x" * 4096)
    recorder = JudgeEvidenceRecorder(
        token="token",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"Content-Encoding": "gzip"},
                    content=compressed,
                    request=request,
                )
            )
        ),
        capability_factory=lambda: "g" * 32,
    )
    base = recorder.start(host="127.0.0.1", port=0)
    destination = tmp_path / "judge"
    capability = recorder.register_scope(
        execution_id="exec",
        trial_id="trial",
        model="judge",
        destination=destination,
        policy=_policy().model_copy(update={"judge_max_response_bytes": 128}),
    )
    try:
        status, _, _ = _request(
            recorder.scoped_url(base, capability),
            {"model": "judge", "messages": []},
        )
    finally:
        recorder.revoke_scope(capability)
        recorder.close()
    assert len(compressed) < 128
    assert status == 502
    exchange = verify_judge_exchange(destination / "judge-0001")
    assert exchange.outcome == "recorder_error"
    assert exchange.error_type == "JudgeRecorderError"


def test_revoked_scope_is_unavailable(tmp_path: Path) -> None:
    recorder = JudgeEvidenceRecorder(
        token="token",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"{}")
            )
        ),
        capability_factory=lambda: "d" * 32,
    )
    base = recorder.start(host="127.0.0.1", port=0)
    capability = recorder.register_scope(
        execution_id="exec",
        trial_id="trial",
        model="judge",
        destination=tmp_path / "judge",
        policy=_policy(),
    )
    recorder.revoke_scope(capability)
    try:
        status, _, _ = _request(
            recorder.scoped_url(base, capability), {"model": "judge", "messages": []}
        )
    finally:
        recorder.close()
    assert status == 404
