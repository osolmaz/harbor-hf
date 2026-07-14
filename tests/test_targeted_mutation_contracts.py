from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import JsonValue
from test_campaign_terminal_evidence_mutation_contracts import (
    _campaign,
    _execution_lock_for_wave,
    _wave,
)
from test_endpoints import FakePort, FakeTime, _desired, _snapshot
from test_hf_endpoints import _http_error

from harbor_hf.campaign_observer import (
    _execution_control_events,
    _legacy_failure_category,
    _wave_events,
)
from harbor_hf.endpoints import (
    AmbiguousEndpointCreate,
    AmbiguousEndpointDelete,
    AmbiguousEndpointPause,
    EndpointConfigurationMismatch,
    EndpointIdentityMismatch,
    EndpointProviderError,
    EndpointProvisioner,
    EndpointProvisioningError,
    EndpointSnapshot,
    effective_configuration_mismatches,
    verify_exact_endpoint,
)
from harbor_hf.hf_endpoints import (
    _access_type,
    _mapping,
    _provider_call,
    _scaling_measure,
    _string_list,
    _string_mapping,
)
from harbor_hf.models import ExperimentSpec
from harbor_hf.provider_models import (
    ExplicitProviderRoute,
    ProviderChatRequest,
    ProviderLimits,
    ProviderMessage,
    ProviderTarget,
)
from harbor_hf.provider_proxy import (
    _MAX_EVIDENCE_RESPONSE_BYTES,
    _MAX_REQUEST_BYTES,
    ProviderProxyError,
    _forwarded_payload,
    _json_object,
    _provider_message,
    _provider_request,
    _relay_response,
)
from harbor_hf.providers import (
    _classify_status,
    _elapsed_ms,
    _quota_evidence,
    _usage_evidence,
    observe_provider_response,
)
from harbor_hf.result_publisher import (
    HubDatasetPublisher,
    IndexReceipt,
    ResultReceipt,
    publisher_lease_path,
)
from harbor_hf.results import (
    GlobalIndexRow,
    build_index_window_file,
    read_index_file,
)
from harbor_hf.wave_worker import ExecutionLock


def _provider_target() -> ProviderTarget:
    return ProviderTarget(
        id="provider-contract",
        model="org/model",
        routing=ExplicitProviderRoute(provider="groq"),
        timeout_seconds=11,
        limits=ProviderLimits(max_concurrent_requests=2, max_attempts=2),
        parameters={"temperature": 0.25, "seed": 17},
    )


@pytest.mark.parametrize("content", [b"[]", b"null", b'"value"', b"3"])
def test_provider_proxy_accepts_only_json_objects(content: bytes) -> None:
    with pytest.raises(
        ProviderProxyError, match="provider request must be a JSON object"
    ):
        _json_object(content)


def test_provider_proxy_forwarding_precedence_and_stream_usage_contract() -> None:
    target = _provider_target()

    forwarded = _forwarded_payload(
        target,
        {
            "model": "caller/model",
            "temperature": 0.75,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": False},
        },
    )

    assert forwarded == {
        "seed": 17,
        "temperature": 0.75,
        "model": "org/model:groq",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


@pytest.mark.parametrize(
    ("error_type", "message", "expected"),
    [
        ("AuthenticationError", None, "authentication"),
        (None, "401 unauthorized", "authentication"),
        (None, "request forbidden", "authentication"),
        ("RateLimitError", None, "rate-limit"),
        (None, "rate_limit exceeded", "rate-limit"),
        (None, "provider status=429", "rate-limit"),
        ("QuotaError", "allocation consumed", "quota"),
        ("TimeoutError", None, "transient"),
        (None, "connection reset", "transient"),
        (None, "provider status=503", "transient"),
        ("ConfigurationError", None, "configuration"),
        ("BadRequestError", None, "configuration"),
        (None, "endpoint NotFound", "configuration"),
        (None, None, "benchmark"),
        (17, {"unexpected": True}, "benchmark"),
    ],
)
def test_legacy_failure_categories_preserve_historical_classification(
    error_type: object, message: object, expected: str
) -> None:
    assert _legacy_failure_category(error_type, message) == expected


def test_observer_wave_and_execution_event_projection_is_exact(
    remote_spec: ExperimentSpec,
) -> None:
    campaign = _campaign(remote_spec)
    wave = _wave(campaign, remote_spec).model_copy(
        update={"estimated_cost_microusd": 765_432}
    )
    records: list[dict[str, object]] = [
        {"event": "wave_started", "at": "2026-07-14T09:10:00+08:00"},
        {"event": "wave_succeeded", "at": "2026-07-14T01:15:00+00:00"},
        {"event": "endpoint_pause_requested", "at": "2026-07-14T01:16:00+00:00"},
    ]

    wave_events = _wave_events(campaign, wave, "_SUCCESS", records)

    assert [event.kind for event in wave_events] == [
        "wave.active",
        "wave.cleaning",
        "wave.closed",
    ]
    assert [event.observed_at for event in wave_events] == [
        datetime(2026, 7, 14, 1, 10, tzinfo=UTC),
        datetime(2026, 7, 14, 1, 16, tzinfo=UTC),
        datetime(2026, 7, 14, 1, 16, tzinfo=UTC),
    ]
    assert wave_events[0].payload.model_dump(mode="json") == {
        "deployment_digest": wave.deployment_digest,
        "provider": "hf-inference-endpoints",
        "shard_ids": wave.shard_ids,
        "estimated_cost_microusd": 765_432,
    }
    assert wave_events[2].event_id == (
        "evt-"
        + hashlib.sha256(
            f"{campaign.campaign_id}:{wave.wave_id}:closed:_SUCCESS".encode()
        ).hexdigest()[:32]
    )

    trial = campaign.runs[0].shards[0].trials[0]
    execution = ExecutionLock.model_validate_json(
        _execution_lock_for_wave(campaign, wave, trial.trial_id, "execution-contract")
    )
    execution_events = _execution_control_events(
        campaign,
        execution,
        "_FAILED",
        [
            {"event": "execution_started", "at": "2026-07-14T01:11:00+00:00"},
            {"event": "execution_failed", "at": "2026-07-14T01:12:00+00:00"},
        ],
        "provider connection reset",
        "transient",
    )
    assert [event.kind for event in execution_events] == [
        "execution.started",
        "execution.failed",
    ]
    assert execution_events[0].payload.model_dump(mode="json") == {
        "trial_id": trial.trial_id,
        "shard_id": execution.shard_id,
        "physical_attempt": 1,
        "wave_id": wave.wave_id,
        "estimated_cost_microusd": 0,
    }
    assert execution_events[1].payload.model_dump(mode="json") == {
        "trial_id": trial.trial_id,
        "physical_attempt": 1,
        "category": "transient",
        "spend_microusd": 0,
        "retry_after_seconds": None,
        "message": "provider connection reset",
    }


def test_provider_proxy_request_normalization_is_exact() -> None:
    payload: dict[str, JsonValue] = {
        "model": "caller/model",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "answer"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": False},
        "temperature": 0,
        "max_tokens": 32,
    }

    request = _provider_request(payload, "provider-41")

    assert request.model_dump(mode="json") == {
        "request_id": "provider-41",
        "messages": [
            {
                "role": "system",
                "content": "system",
                "tool_call_id": None,
                "tool_calls": [],
            },
            {
                "role": "user",
                "content": "question",
                "tool_call_id": None,
                "tool_calls": [],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_call_id": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function_name": "lookup",
                        "arguments": '{"q":1}',
                    }
                ],
            },
            {
                "role": "tool",
                "content": "answer",
                "tool_call_id": "call-1",
                "tool_calls": [],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "parameters": {"temperature": 0, "max_tokens": 32},
        "stream": True,
    }


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "provider message must be an object"),
        ({"role": "user", "tool_calls": {}}, "tool calls must be a list"),
        (
            {"role": "assistant", "tool_calls": ["bad"]},
            "provider tool call must be an object",
        ),
        (
            {"role": "assistant", "tool_calls": [{"id": "one"}]},
            "provider tool call function must be an object",
        ),
    ],
)
def test_provider_proxy_message_shape_errors_are_specific(
    value: JsonValue, message: str
) -> None:
    with pytest.raises(ProviderProxyError, match=message):
        _provider_message(value)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "messages must be a list"),
        ({"messages": {}, "tools": []}, "messages must be a list"),
        ({"messages": [], "tools": {}}, "tools must be a list"),
    ],
)
def test_provider_proxy_request_shape_errors_are_specific(
    payload: dict[str, JsonValue], message: str
) -> None:
    with pytest.raises(ProviderProxyError, match=message):
        _provider_request(payload, "provider-1")


class _RecordingOutput(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


class _Handler:
    def __init__(self, content: bytes = b"", content_length: str | None = None) -> None:
        self.rfile = io.BytesIO(content)
        self.wfile = _RecordingOutput()
        self.headers = (
            {} if content_length is None else {"Content-Length": content_length}
        )
        self.responses: list[int] = []
        self.response_headers: list[tuple[str, str]] = []
        self.ended = 0
        self.close_connection = False

    def send_response(self, status: int) -> None:
        self.responses.append(status)

    def send_header(self, name: str, value: str) -> None:
        self.response_headers.append((name, value))

    def end_headers(self) -> None:
        self.ended += 1


def test_provider_proxy_request_reader_enforces_exact_size_boundary(
    tmp_path: Path,
) -> None:
    from harbor_hf.provider_proxy import ProviderEvidenceProxy

    proxy = ProviderEvidenceProxy(
        _provider_target(), token="token", evidence_path=tmp_path / "evidence.jsonl"
    )
    assert proxy._read_request(cast(Any, _Handler(b"{}", "2"))) == {}

    for length in (None, "bad", "-1", str(_MAX_REQUEST_BYTES + 1)):
        with pytest.raises(ProviderProxyError, match="invalid request size"):
            proxy._read_request(cast(Any, _Handler(b"{}", length)))

    proxy.client.close()


def test_provider_proxy_response_relay_forwards_allowlist_and_all_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([10.125])
    monkeypatch.setattr("harbor_hf.provider_proxy.perf_counter", lambda: next(ticks))
    handler = _Handler()
    content = b"first-second"
    response = httpx.Response(
        206,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": "request-1",
            "Retry-After": "3",
            "X-Private": "secret",
        },
        stream=httpx.ByteStream(content),
    )

    captured, first_byte_ms = _relay_response(cast(Any, handler), response, 10.0)

    assert captured == content
    assert first_byte_ms == pytest.approx(125)
    assert handler.responses == [206]
    assert handler.response_headers == [
        ("content-type", "application/json"),
        ("x-request-id", "request-1"),
        ("retry-after", "3"),
        ("Connection", "close"),
    ]
    assert handler.ended == 1
    assert handler.wfile.getvalue() == content
    assert handler.wfile.flushes == 1
    assert handler.close_connection is True


def test_provider_proxy_response_relay_caps_evidence_without_truncating_client() -> (
    None
):
    handler = _Handler()
    content = b"x" * (_MAX_EVIDENCE_RESPONSE_BYTES + 1)
    response = httpx.Response(200, stream=httpx.ByteStream(content))

    captured, first_byte_ms = _relay_response(cast(Any, handler), response, 0)

    assert captured == b""
    assert first_byte_ms is not None
    assert handler.wfile.getvalue() == content


def test_provider_proxy_attempt_identity_budget_and_recording_are_deterministic(
    tmp_path: Path,
) -> None:
    from harbor_hf.provider_proxy import ProviderEvidenceProxy

    evidence_path = tmp_path / "nested" / "evidence.jsonl"
    proxy = ProviderEvidenceProxy(
        _provider_target(), token="token", evidence_path=evidence_path
    )
    payload: dict[str, JsonValue] = {"messages": [{"role": "user", "content": "same"}]}

    first, first_attempt = proxy._request(payload)
    second, second_attempt = proxy._request(payload)
    different, different_attempt = proxy._request(
        {"messages": [{"role": "user", "content": "different"}]}
    )

    assert (first.request_id, first_attempt) == ("provider-1", 1)
    assert (second.request_id, second_attempt) == ("provider-2", 2)
    assert (different.request_id, different_attempt) == ("provider-3", 1)
    with pytest.raises(ProviderProxyError, match="attempt budget is exhausted"):
        proxy._request(payload)
    assert proxy._request_counter == 4
    assert sorted(proxy._attempts.values()) == [1, 2]

    evidence_path.parent.mkdir(parents=True)
    proxy._record({"z": "é", "a": 1})
    proxy._record({"line": 2})
    assert evidence_path.read_bytes() == (b'{"a":1,"z":"\\u00e9"}\n{"line":2}\n')
    proxy.client.close()


def test_provider_proxy_json_response_writer_sets_framing_and_closes() -> None:
    from harbor_hf.provider_proxy import ProviderEvidenceProxy

    handler = _Handler()
    ProviderEvidenceProxy._send_json(
        cast(Any, handler), 418, {"z": "last", "a": "first"}
    )

    content = b'{"a": "first", "z": "last"}'
    assert handler.responses == [418]
    assert handler.response_headers == [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(content))),
        ("Connection", "close"),
    ]
    assert handler.ended == 1
    assert handler.wfile.getvalue() == content
    assert handler.close_connection is True


def test_provider_proxy_forward_observes_exact_upstream_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harbor_hf.provider_proxy import ProviderEvidenceProxy

    calls: list[dict[str, object]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "authorization": request.headers["authorization"],
                "payload": json.loads(request.content),
            }
        )
        return httpx.Response(
            202,
            headers={"content-type": "application/json", "x-request-id": "up-1"},
            stream=httpx.ByteStream(b'{"accepted":true}'),
        )

    client = httpx.Client(transport=httpx.MockTransport(upstream))
    proxy = ProviderEvidenceProxy(
        _provider_target(),
        token="token-value",
        evidence_path=tmp_path / "evidence.jsonl",
        client=client,
    )
    ticks = iter([20.0, 20.007, 20.025])
    monkeypatch.setattr("harbor_hf.provider_proxy.perf_counter", lambda: next(ticks))
    handler = _Handler()

    observed = proxy._forward(cast(Any, handler), {"model": "org/model:groq"})

    assert calls == [
        {
            "method": "POST",
            "url": "https://router.huggingface.co/v1/chat/completions",
            "authorization": "Bearer token-value",
            "payload": {"model": "org/model:groq"},
        }
    ]
    assert observed.status_code == 202
    assert dict(observed.headers) == {
        "content-type": "application/json",
        "x-request-id": "up-1",
    }
    assert observed.content == b'{"accepted":true}'
    assert observed.total_ms == 25.0
    assert observed.first_byte_ms == pytest.approx(7)
    assert handler.wfile.getvalue() == b'{"accepted":true}'
    client.close()


@pytest.mark.parametrize(
    ("error", "status", "body"),
    [
        (httpx.ReadTimeout("slow"), 504, {"error": "provider request timed out"}),
        (httpx.ConnectError("down"), 502, {"error": "provider transport failed"}),
    ],
)
def test_provider_proxy_forward_maps_pre_response_transport_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: httpx.HTTPError,
    status: int,
    body: dict[str, str],
) -> None:
    from harbor_hf.provider_proxy import ProviderEvidenceProxy

    def upstream(request: httpx.Request) -> httpx.Response:
        error.request = request
        raise error

    client = httpx.Client(transport=httpx.MockTransport(upstream))
    proxy = ProviderEvidenceProxy(
        _provider_target(),
        token="token",
        evidence_path=tmp_path / "evidence.jsonl",
        client=client,
    )
    ticks = iter([30.0, 30.01])
    monkeypatch.setattr("harbor_hf.provider_proxy.perf_counter", lambda: next(ticks))
    handler = _Handler()

    observed = proxy._forward(cast(Any, handler), {"messages": []})

    assert observed.status_code == status
    assert dict(observed.headers) == {}
    assert observed.content == b""
    assert observed.total_ms == 10.0
    assert observed.first_byte_ms is None
    assert handler.responses == [status]
    assert json.loads(handler.wfile.getvalue()) == body
    client.close()


def test_provider_status_classification_has_exact_http_boundaries() -> None:
    assert {
        code: _classify_status(code) for code in (399, 400, 428, 429, 499, 500, 599)
    } == {
        399: ("provider_error", "ambiguous", "http_399"),
        400: ("rejected", "not_completed", "http_400"),
        428: ("rejected", "not_completed", "http_428"),
        429: ("throttled", "not_completed", "rate_limited"),
        499: ("rejected", "not_completed", "http_499"),
        500: ("provider_error", "ambiguous", "http_500"),
        599: ("provider_error", "ambiguous", "http_599"),
    }
    assert _elapsed_ms(10.0, 10.1234567) == 123.457
    assert _elapsed_ms(10.0, 9.0) == 0.0


def test_provider_usage_and_quota_evidence_preserve_every_boundary() -> None:
    assert _usage_evidence(
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 6}
    ).model_dump(mode="json") == {
        "input_tokens": {"status": "observed", "value": 2, "detail": None},
        "output_tokens": {"status": "observed", "value": 3, "detail": None},
        "total_tokens": {
            "status": "malformed",
            "value": None,
            "detail": "total_tokens does not equal input plus output",
        },
    }
    assert _quota_evidence(
        httpx.Headers(
            {
                "x-ratelimit-limit-requests": "10",
                "x-ratelimit-remaining-requests": "-1",
                "x-ratelimit-limit-tokens": "many",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-requests": "1s",
                "x-ratelimit-reset": "2s",
            }
        )
    ).model_dump(mode="json") == {
        "request_limit": {"status": "observed", "value": 10, "detail": None},
        "requests_remaining": {
            "status": "malformed",
            "value": None,
            "detail": ("x-ratelimit-remaining-requests must be a non-negative integer"),
        },
        "token_limit": {
            "status": "malformed",
            "value": None,
            "detail": "x-ratelimit-limit-tokens must be a non-negative integer",
        },
        "tokens_remaining": {"status": "observed", "value": 0, "detail": None},
        "reset": {"status": "observed", "value": "1s", "detail": None},
    }


def test_observed_provider_response_keeps_exact_stream_error_evidence() -> None:
    request = ProviderChatRequest(
        request_id="request-1",
        messages=[ProviderMessage(role="user", content="hello")],
        stream=True,
    )
    malformed = observe_provider_response(
        _provider_target(),
        request,
        attempt=2,
        status_code=200,
        headers=httpx.Headers({"x-request-id": "upstream-1"}),
        content=(
            b'data: {"id":"response-1","model":"reported/model",'
            b'"choices":[{"finish_reason":"length","delta":'
            b'{"content":"partial","tool_calls":[]}}]}\n'
        ),
        total_ms=19.5,
        time_to_first_token_ms=4.25,
    )

    assert malformed.status == "malformed_response"
    assert malformed.remote_outcome == "ambiguous"
    assert malformed.response_id.model_dump(mode="json") == {
        "status": "observed",
        "value": "response-1",
        "detail": None,
    }
    assert malformed.finish_reason.value == "length"
    assert malformed.error_code == "invalid_stream"
    assert malformed.evidence.request.model_dump(mode="json") == {
        "request_id": "request-1",
        "provider_request_id": {
            "status": "not_reported",
            "value": None,
            "detail": None,
        },
        "streaming": True,
        "message_count": 1,
        "tool_count": 0,
    }
    assert malformed.evidence.model.model_dump(mode="json") == {
        "requested": "org/model",
        "routed": "org/model:groq",
        "response": {"status": "observed", "value": "reported/model", "detail": None},
    }
    assert malformed.evidence.retry.model_dump(mode="json") == {
        "attempt": 2,
        "max_attempts": 2,
        "disposition": "inspect",
        "retry_after": {"status": "not_reported", "value": None, "detail": None},
    }
    assert malformed.evidence.latency.model_dump(mode="json") == {
        "total_ms": {"status": "observed", "value": 19.5, "detail": None},
        "time_to_first_token_ms": {
            "status": "not_observed",
            "value": None,
            "detail": None,
        },
    }


def _endpoint_provisioner(port: FakePort) -> EndpointProvisioner:
    clock = FakeTime()
    return EndpointProvisioner(
        port,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def test_endpoint_identity_and_nested_configuration_contracts_are_exact(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    missing_tag = desired.configuration.model_copy(
        update={"tags": ["benchmark", *desired.identity.tags[1:]]}
    )
    foreign = EndpointSnapshot(
        namespace="foreign",
        name=desired.identity.name,
        configuration=missing_tag,
        status=_snapshot(desired).status,
    )
    with pytest.raises(
        EndpointIdentityMismatch, match="deterministic managed identity"
    ):
        verify_exact_endpoint(desired, foreign)

    changed_model = desired.configuration.model.model_copy(
        update={
            "environment": {"A": "1", "B": "2"},
            "arguments": ["--different", "2"],
        }
    )
    changed = desired.configuration.model_copy(
        update={"model": changed_model, "cache_http_responses": False}
    )
    mismatches = effective_configuration_mismatches(desired.configuration, changed)
    assert [item.model_dump(mode="json") for item in mismatches] == [
        {
            "path": "configuration.cache_http_responses",
            "expected": "true",
            "observed": "false",
        },
        {
            "path": "configuration.model.arguments",
            "expected": (
                '["--model","/repository","--max-model-len","65536",'
                '"--kv-cache-dtype","fp8"]'
            ),
            "observed": '["--different","2"]',
        },
        {
            "path": "configuration.model.environment.A",
            "expected": "null",
            "observed": '"1"',
        },
        {
            "path": "configuration.model.environment.B",
            "expected": "null",
            "observed": '"2"',
        },
        {
            "path": "configuration.model.environment.VLLM_USE_FLASHINFER_MOE_FP4",
            "expected": '"1"',
            "observed": "null",
        },
    ]
    with pytest.raises(EndpointConfigurationMismatch) as captured:
        verify_exact_endpoint(desired, _snapshot(desired, configuration=changed))
    assert captured.value.mismatches == mismatches


def test_endpoint_create_pause_and_delete_have_exact_side_effect_sequences(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    running = _snapshot(desired, state="running", ready=1, target=2)
    paused = _snapshot(desired, state="paused", ready=0, target=2)
    created_port = FakePort(
        inspections=[None, running, paused],
        create_result=running,
        pause_result=running,
    )
    created = _endpoint_provisioner(created_port).create_or_adopt(
        desired, timeout_seconds=5, poll_seconds=1
    )
    identity = desired.identity.name
    assert created.action == "created"
    assert created.snapshot == paused
    assert created_port.calls == [
        f"inspect:{identity}",
        f"create:{identity}",
        f"inspect:{identity}",
        f"pause:{identity}",
        f"inspect:{identity}",
    ]

    delete_port = FakePort(
        inspections=[paused, None],
        delete_error=AmbiguousEndpointDelete("uncertain"),
    )
    assert _endpoint_provisioner(delete_port).delete(desired) is True
    assert delete_port.calls == [
        f"inspect:{identity}",
        f"delete:{identity}",
        f"inspect:{identity}",
    ]


def test_ambiguous_endpoint_create_is_adopted_then_verified_paused(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    paused = _snapshot(desired, target=2)
    port = FakePort(
        inspections=[None, None, paused, paused],
        create_result=AmbiguousEndpointCreate("uncertain"),
    )

    result = _endpoint_provisioner(port).create_or_adopt(
        desired, timeout_seconds=5, poll_seconds=1
    )

    identity = desired.identity.name
    assert result.action == "adopted"
    assert result.snapshot == paused
    assert port.calls == [
        f"inspect:{identity}",
        f"create:{identity}",
        f"inspect:{identity}",
        f"inspect:{identity}",
        f"inspect:{identity}",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("public", "public"), ("authenticated", "authenticated"), ("private", "private")],
)
def test_hf_endpoint_access_types_are_preserved(value: str, expected: str) -> None:
    assert _access_type(value) == expected


@pytest.mark.parametrize("value", [None, "", "PUBLIC", "protected", 1, True])
def test_hf_endpoint_access_type_rejects_noncontract_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _access_type(value)


def test_hf_endpoint_boundary_collection_parsers_are_strict() -> None:
    assert dict(_mapping({"key": 1}, "root")) == {"key": 1}
    assert _string_list(["a", "b"], "items") == ["a", "b"]
    assert _string_mapping({"a": "1", "b": "2"}, "values") == {
        "a": "1",
        "b": "2",
    }
    for value in (None, [], {1: "value"}):
        with pytest.raises(TypeError):
            _mapping(value, "root")
    for value in (None, {}, ["a", 2], "a"):
        with pytest.raises(TypeError):
            _string_list(value, "items")
    for value in (None, [], {"a": 1}):
        with pytest.raises(TypeError):
            _string_mapping(value, "values")


def test_hf_endpoint_scaling_measure_contract_is_exact() -> None:
    assert _scaling_measure(None) == (None, None)
    assert _scaling_measure({"pendingRequests": 2}) == ("pendingRequests", 2.0)
    assert _scaling_measure({"hardwareUsage": 0.75}) == ("hardwareUsage", 0.75)
    assert _scaling_measure({"pendingRequests": None}) == (None, None)
    for value in (
        {},
        {"pendingRequests": 1, "hardwareUsage": 2},
        {"unknown": 1},
        {"pendingRequests": True},
        {"pendingRequests": "1"},
    ):
        with pytest.raises((TypeError, ValueError)):
            _scaling_measure(value)


@pytest.mark.parametrize(
    ("operation", "status", "error_type", "message"),
    [
        ("create", 400, EndpointProviderError, "create failed: HTTP 400"),
        ("pause", 409, AmbiguousEndpointPause, "pause outcome is ambiguous: HTTP 409"),
        (
            "create",
            500,
            AmbiguousEndpointCreate,
            "create outcome is ambiguous: HTTP 500",
        ),
        (
            "delete",
            404,
            AmbiguousEndpointDelete,
            "delete outcome is ambiguous: HTTP 404",
        ),
        ("delete", 403, EndpointProviderError, "delete failed: HTTP 403"),
    ],
)
def test_hf_endpoint_provider_http_error_classification_is_exact(
    operation: str,
    status: int,
    error_type: type[Exception],
    message: str,
) -> None:
    def request() -> None:
        raise _http_error(status)

    ambiguous: type[EndpointProvisioningError] = {
        "create": AmbiguousEndpointCreate,
        "pause": AmbiguousEndpointPause,
        "delete": AmbiguousEndpointDelete,
    }[operation]
    with pytest.raises(error_type, match=message):
        _provider_call(operation, request, ambiguous=ambiguous)


@pytest.mark.parametrize(
    ("operation", "ambiguous"),
    [
        ("create", AmbiguousEndpointCreate),
        ("pause", AmbiguousEndpointPause),
        ("delete", AmbiguousEndpointDelete),
    ],
)
def test_hf_endpoint_transport_errors_are_always_ambiguous(
    operation: str, ambiguous: type[EndpointProvisioningError]
) -> None:
    def request() -> None:
        raise httpx.ConnectError("connection lost")

    with pytest.raises(ambiguous, match="ambiguous before a response"):
        _provider_call(operation, request, ambiguous=ambiguous)


class _LeaseStub:
    def acquire(self, path: str, owner: dict[str, str]) -> None:
        del path, owner

    def release(self, path: str, owner: dict[str, str]) -> None:
        del path, owner


class _ApiStub:
    def list_repo_files(self, repo_id: str, **kwargs: object) -> list[str]:
        del repo_id, kwargs
        return []


def _index_row(
    publication_id: str, completed_at: datetime, revision: str
) -> GlobalIndexRow:
    return GlobalIndexRow(
        publication_id=publication_id,
        run_id=f"run-{publication_id}",
        campaign_id="campaign-one",
        benchmark="shellbench",
        result_kind="ordinary",
        outcome="complete",
        completed_at=completed_at,
        model_repo="org/model",
        model_revision="a" * 40,
        agent_name="agent",
        agent_revision="1.2.3",
        result_dataset="org/results",
        result_revision=revision,
        source_checksum="sha256:" + "b" * 64,
        control_commit="c" * 40,
    )


def test_result_index_windows_are_deduplicated_sorted_and_power_sized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    old = _index_row("pub-old", now, "1" * 40)
    same_time_later_id = _index_row("pub-z", now + timedelta(minutes=1), "2" * 40)
    replaced = _index_row("pub-new", now + timedelta(minutes=2), "3" * 40)
    replacement = _index_row("pub-new", now + timedelta(minutes=3), "4" * 40)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one",
        leases=_LeaseStub(),
        api=cast(Any, _ApiStub()),
    )
    monkeypatch.setattr(publisher, "_exists", lambda *args: False)
    monkeypatch.setattr(
        publisher,
        "_legacy_index_rows",
        lambda *args: [same_time_later_id, replaced, old],
    )

    windows = publisher._index_windows("org/index", "d" * 40, replacement)

    assert [item.path for item in windows] == [
        f"data/index/schema=v1/windows/{2**power:04d}.parquet" for power in range(12)
    ]
    expected_ids = ["pub-new", "pub-z", "pub-old"]
    assert [row.publication_id for row in read_index_file(windows[0].content)] == [
        "pub-new"
    ]
    assert [row.publication_id for row in read_index_file(windows[1].content)] == [
        "pub-new",
        "pub-z",
    ]
    assert [row.publication_id for row in read_index_file(windows[2].content)] == (
        expected_ids
    )
    assert [row.result_revision for row in read_index_file(windows[-1].content)] == [
        "4" * 40,
        "2" * 40,
        "1" * 40,
    ]


def test_result_index_window_prefers_consolidated_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    existing = [_index_row("pub-existing", now, "1" * 40)]
    consolidated = build_index_window_file(existing, 2048)
    publisher = HubDatasetPublisher(
        publisher_id="publisher-one",
        leases=_LeaseStub(),
        api=cast(Any, _ApiStub()),
    )
    monkeypatch.setattr(
        publisher,
        "_exists",
        lambda dataset, path, revision: (
            dataset == "org/index"
            and path == consolidated.path
            and revision == "d" * 40
        ),
    )
    monkeypatch.setattr(
        publisher,
        "_read",
        lambda dataset, path, revision: consolidated.content,
    )
    monkeypatch.setattr(
        publisher,
        "_legacy_index_rows",
        lambda *args: pytest.fail("legacy files must not be read"),
    )

    windows = publisher._index_windows(
        "org/index",
        "d" * 40,
        _index_row("pub-new", now + timedelta(seconds=1), "2" * 40),
    )

    assert [row.publication_id for row in read_index_file(windows[-1].content)] == [
        "pub-new",
        "pub-existing",
    ]


def test_publication_receipts_and_lease_identity_are_canonical() -> None:
    row = _index_row("pub-receipt", datetime(2026, 7, 14, tzinfo=UTC), "1" * 40)
    index_file = build_index_window_file([row], 1)
    receipt = HubDatasetPublisher._index_receipt(row, index_file)
    assert receipt == IndexReceipt(
        publication_id="pub-receipt",
        result_dataset="org/results",
        result_revision="1" * 40,
        index_path="data/index/schema=v1/windows/0001.parquet",
        index_sha256=("sha256:" + hashlib.sha256(index_file.content).hexdigest()),
    )
    assert publisher_lease_path("org/results") == (
        "coordination/publishers/"
        "eb1ce4ea9e1b25394e2cff859f3d086d3afab316fc7519d7b3f901940d22e697.json"
    )
    result = ResultReceipt(
        publication_id="pub-receipt",
        run_id="run-one",
        source_checksum="sha256:" + "a" * 64,
        files={"data/runs.parquet": "sha256:" + "b" * 64},
    )
    assert json.loads(result.model_dump_json()) == {
        "schema_version": "harbor-hf/result-publication/v1",
        "publication_id": "pub-receipt",
        "run_id": "run-one",
        "source_checksum": "sha256:" + "a" * 64,
        "files": {"data/runs.parquet": "sha256:" + "b" * 64},
    }


def test_result_index_file_rejects_invalid_bytes_and_nonpositive_windows() -> None:
    with pytest.raises(ValueError, match="window size must be positive"):
        build_index_window_file([], 0)
    with pytest.raises(ValueError, match="global index Parquet is invalid"):
        read_index_file(b"not parquet")
