from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import JsonValue, ValidationError

from harbor_hf.provider_models import (
    ExplicitProviderRoute,
    ProviderLimits,
    ProviderTarget,
)
from harbor_hf.provider_proxy import (
    ProviderEvidenceProxy,
    ProviderProxyError,
    _forwarded_payload,
    _json_object,
    _provider_message,
    _provider_request,
)


def test_proxy_forwards_stream_and_records_content_free_provider_evidence(
    tmp_path: Path,
) -> None:
    observed: list[dict[str, object]] = []
    stream = (
        b'data: {"id":"response-one","model":"org/model",'
        b'"choices":[{"delta":{"content":"answer"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":11,"completion_tokens":4,'
        b'"total_tokens":15}}\n\n'
        b"data: [DONE]\n\n"
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(
            {
                "authorization": request.headers.get("authorization"),
                "payload": json.loads(request.content),
            }
        )
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "x-request-id": "provider-request-one",
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "99",
            },
            stream=httpx.ByteStream(stream),
        )

    target = ProviderTarget(
        id="provider-one",
        model="org/model",
        routing=ExplicitProviderRoute(provider="groq"),
        limits=ProviderLimits(max_attempts=2),
        parameters={"temperature": 0},
    )
    evidence_path = tmp_path / "provider-requests.jsonl"
    upstream_client = httpx.Client(transport=httpx.MockTransport(upstream))
    proxy = ProviderEvidenceProxy(
        target,
        token="secret-token",
        evidence_path=evidence_path,
        client=upstream_client,
    )
    base_url = proxy.start()
    scoped_url = proxy.scoped_base_url(base_url, "trial-one")
    try:
        response = httpx.post(
            f"{scoped_url}/v1/chat/completions",
            json={
                "model": "ignored/model",
                "messages": [{"role": "user", "content": "private benchmark prompt"}],
                "stream": True,
            },
            timeout=5,
        )
    finally:
        proxy.close()
        upstream_client.close()

    assert response.status_code == 200
    assert response.content == stream
    assert observed == [
        {
            "authorization": "Bearer secret-token",
            "payload": {
                "messages": [{"content": "private benchmark prompt", "role": "user"}],
                "model": "org/model:groq",
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": 0,
            },
        }
    ]
    raw_evidence = evidence_path.read_text(encoding="utf-8")
    assert "private benchmark prompt" not in raw_evidence
    assert "answer" not in raw_evidence
    assert "secret-token" not in raw_evidence
    evidence = json.loads(raw_evidence)
    assert evidence["status"] == "succeeded"
    assert evidence["evidence"]["request"] == {
        "message_count": 1,
        "provider_request_id": {
            "detail": None,
            "status": "observed",
            "value": "provider-request-one",
        },
        "request_id": "provider-1",
        "streaming": True,
        "tool_count": 0,
    }
    assert evidence["evidence"]["usage"] == {
        "input_tokens": {"detail": None, "status": "observed", "value": 11},
        "output_tokens": {"detail": None, "status": "observed", "value": 4},
        "total_tokens": {"detail": None, "status": "observed", "value": 15},
    }
    ttft = evidence["evidence"]["latency"]["time_to_first_token_ms"]
    assert ttft["status"] == "observed"
    assert ttft["value"] >= 0


def test_proxy_request_translation_preserves_openai_tool_contract() -> None:
    target = ProviderTarget(
        id="provider-one",
        model="org/model",
        routing=ExplicitProviderRoute(provider="groq"),
        parameters={"temperature": 0},
    )
    payload = cast(
        dict[str, JsonValue],
        {
            "model": "ignored/model",
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"x":1}'},
                        }
                    ],
                },
                {"role": "tool", "content": "result", "tool_call_id": "call-one"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "stream": True,
            "max_tokens": 64,
        },
    )

    request = _provider_request(payload, "request-one")

    assert request.model_dump(mode="json") == {
        "request_id": "request-one",
        "messages": [
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
                        "id": "call-one",
                        "type": "function",
                        "function_name": "shell",
                        "arguments": '{"x":1}',
                    }
                ],
            },
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "call-one",
                "tool_calls": [],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a command",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "parameters": {"max_tokens": 64},
        "stream": True,
    }
    assert _forwarded_payload(target, payload) == {
        **payload,
        "model": "org/model:groq",
        "temperature": 0,
        "stream_options": {"include_usage": True},
    }


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (b"[]", "JSON object"),
        (b'[{"x":1}]', "JSON object"),
        (b"not-json", "Expecting value"),
    ],
)
def test_proxy_rejects_non_object_json(value: bytes, message: str) -> None:
    with pytest.raises((ProviderProxyError, json.JSONDecodeError), match=message):
        _json_object(value)


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"role": "assistant", "content": None, "tool_calls": {}},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": ["invalid"],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call", "function": "invalid"}],
        },
    ],
)
def test_proxy_rejects_malformed_openai_messages(value: object) -> None:
    with pytest.raises((ProviderProxyError, ValidationError)):
        _provider_message(cast(JsonValue, value))


def test_proxy_lifecycle_and_http_failure_matrix(tmp_path: Path) -> None:
    upstream_calls: list[dict[str, object]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        upstream_calls.append(payload)
        content = payload["messages"][0]["content"]
        if content == "PRIVATE_THROTTLE":
            return httpx.Response(
                429,
                headers={"retry-after": "2"},
                stream=httpx.ByteStream(b'{"error":"slow down"}'),
            )
        if content == "PRIVATE_TIMEOUT":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(
                b'{"id":"complete-one","model":"org/model",'
                b'"choices":[{"finish_reason":"tool_calls","message":'
                b'{"role":"assistant","content":null,"tool_calls":['
                b'{"id":"call-one","type":"function","function":'
                b'{"name":"shell","arguments":"{}"}}]}}],'
                b'"usage":{"prompt_tokens":2,"completion_tokens":3,'
                b'"total_tokens":5}}'
            ),
        )

    target = ProviderTarget(
        id="provider-one",
        model="org/model",
        limits=ProviderLimits(max_attempts=2),
    )
    evidence_path = tmp_path / "provider-requests.jsonl"
    upstream_client = httpx.Client(transport=httpx.MockTransport(upstream))
    with pytest.raises(ValueError, match="token must not be empty"):
        ProviderEvidenceProxy(target, token="", evidence_path=evidence_path)
    proxy = ProviderEvidenceProxy(
        target,
        token="secret-token",
        evidence_path=evidence_path,
        client=upstream_client,
    )
    base_url = proxy.start()
    throttled_url = proxy.scoped_base_url(base_url, "trial-throttled")
    complete_url = proxy.scoped_base_url(base_url, "trial-complete")
    timeout_url = proxy.scoped_base_url(base_url, "trial-timeout")
    with pytest.raises(ProviderProxyError, match="already running"):
        proxy.start()
    try:
        invalid_route = httpx.post(f"{base_url}/unsupported", json={}, timeout=5)
        invalid_message = httpx.post(
            f"{complete_url}/v1/chat/completions", json={"messages": {}}, timeout=5
        )
        responses = [
            httpx.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": "ignored",
                    "messages": [{"role": "user", "content": content}],
                },
                timeout=5,
            )
            for url, content in (
                (throttled_url, "PRIVATE_THROTTLE"),
                (throttled_url, "PRIVATE_THROTTLE"),
                (throttled_url, "PRIVATE_THROTTLE"),
                (complete_url, "PRIVATE_COMPLETE"),
                (timeout_url, "PRIVATE_TIMEOUT"),
            )
        ]
    finally:
        proxy.close()
        proxy.close()
        upstream_client.close()

    assert (invalid_route.status_code, invalid_route.json()) == (
        404,
        {"error": "unsupported provider route"},
    )
    assert invalid_message.status_code == 400
    assert [response.status_code for response in responses] == [429, 429, 400, 200, 504]
    assert responses[2].json() == {
        "error": "provider request attempt budget is exhausted"
    }
    assert len(upstream_calls) == 4
    evidence = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert [record["status"] for record in evidence] == [
        "throttled",
        "throttled",
        "succeeded",
        "provider_error",
    ]
    assert [record["evidence"]["retry"]["attempt"] for record in evidence] == [
        1,
        2,
        1,
        1,
    ]
    assert [record["evidence"]["retry"]["disposition"] for record in evidence] == [
        "retry",
        "no_retry",
        "no_retry",
        "inspect",
    ]
    serialized = json.dumps(evidence, sort_keys=True)
    for private in (
        "PRIVATE_THROTTLE",
        "PRIVATE_COMPLETE",
        "PRIVATE_TIMEOUT",
        "shell",
        "secret-token",
    ):
        assert private not in serialized


def test_proxy_preserves_gzip_encoding_and_decodes_evidence(tmp_path: Path) -> None:
    payload = (
        b'{"id":"complete-one","model":"org/model","choices":['
        b'{"finish_reason":"stop","message":{"role":"assistant",'
        b'"content":"answer"}}]}'
    )

    def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
            },
            stream=httpx.ByteStream(gzip.compress(payload)),
        )

    client = httpx.Client(transport=httpx.MockTransport(upstream))
    evidence = tmp_path / "evidence.jsonl"
    proxy = ProviderEvidenceProxy(
        ProviderTarget(id="provider", model="org/model"),
        token="token",
        evidence_path=evidence,
        client=client,
    )
    base_url = proxy.start()
    try:
        response = httpx.post(
            f"{proxy.scoped_base_url(base_url, 'trial')}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "prompt"}]},
        )
    finally:
        proxy.close()
        client.close()

    assert response.content == payload
    assert response.headers["content-encoding"] == "gzip"
    assert json.loads(evidence.read_text())["status"] == "succeeded"


def test_mid_relay_transport_error_keeps_provider_status_and_partial_body(
    tmp_path: Path,
) -> None:
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b"partial"
            raise httpx.ReadError("upstream disconnected")

    def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BrokenStream())

    client = httpx.Client(transport=httpx.MockTransport(upstream))
    proxy = ProviderEvidenceProxy(
        ProviderTarget(id="provider", model="org/model"),
        token="token",
        evidence_path=tmp_path / "evidence.jsonl",
        client=client,
    )

    class Handler:
        responses: list[int] = []
        response_headers: list[tuple[str, str]] = []
        wfile = type(
            "Output", (), {"write": lambda *_: None, "flush": lambda *_: None}
        )()
        close_connection = False

        def send_response(self, status: int) -> None:
            self.responses.append(status)

        def send_header(self, name: str, value: str) -> None:
            self.response_headers.append((name, value))

        def end_headers(self) -> None:
            pass

    observed = proxy._forward(cast(Any, Handler()), {"messages": []})
    client.close()

    assert observed.status_code == 200
    assert observed.content == b"partial"
