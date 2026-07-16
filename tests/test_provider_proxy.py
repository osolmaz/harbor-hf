from __future__ import annotations

import gzip
import json
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import JsonValue, ValidationError

import harbor_hf.provider_proxy as provider_proxy
from harbor_hf.provider_models import (
    ExplicitProviderRoute,
    ProviderLimits,
    ProviderTarget,
)
from harbor_hf.provider_proxy import (
    ProviderEvidenceProxy,
    ProviderProxyError,
    _decode_evidence_response,
    _forwarded_payload,
    _json_object,
    _provider_message,
    _provider_request,
    _relay_response,
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
    base_url = proxy.start(host="127.0.0.1", port=0)
    capability = proxy.register_scope("trial-one")
    scoped_url = proxy.scoped_base_url(base_url, capability)
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


def test_proxy_health_and_capability_lifecycle_isolate_trials(
    tmp_path: Path,
) -> None:
    upstream_calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": "response-one",
                "model": "org/model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "answer"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    generated = iter(("A" * 22, "B" * 22))
    client = httpx.Client(transport=httpx.MockTransport(upstream))
    proxy = ProviderEvidenceProxy(
        ProviderTarget(
            id="provider",
            model="org/model",
            limits=ProviderLimits(max_attempts=1),
        ),
        token="secret-token",
        evidence_path=tmp_path / "evidence.jsonl",
        client=client,
        capability_factory=generated.__next__,
    )
    base_url = proxy.start(host="127.0.0.1", port=0)
    first = proxy.register_scope("execution-one")
    second = proxy.register_scope("execution-two")
    first_url = proxy.scoped_base_url(base_url, first)
    second_url = proxy.scoped_base_url(base_url, second)
    payload = {"messages": [{"role": "user", "content": "private"}]}
    try:
        health = httpx.get(f"{base_url}/healthz", timeout=5)
        unknown = httpx.post(
            f"{base_url}/scopes/{'C' * 22}/v1/chat/completions",
            json=payload,
            timeout=5,
        )
        first_response = httpx.post(
            f"{first_url}/v1/chat/completions", json=payload, timeout=5
        )
        second_response = httpx.post(
            f"{second_url}/v1/chat/completions", json=payload, timeout=5
        )
        proxy.revoke_scope(first)
        revoked = httpx.post(
            f"{first_url}/v1/chat/completions", json=payload, timeout=5
        )
    finally:
        proxy.close()
        client.close()

    assert (health.status_code, health.json()) == (200, {"status": "ok"})
    assert unknown.status_code == 404
    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert revoked.status_code == 404
    assert len(upstream_calls) == 2
    assert "execution-one" not in first_url
    assert "execution-two" not in second_url
    assert proxy.capability_digest(first).startswith("sha256:")


def test_proxy_forwards_multipart_message_content(tmp_path: Path) -> None:
    observed: list[dict[str, object]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "response-one",
                "model": "org/model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "summary"},
                    }
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(upstream))
    proxy = ProviderEvidenceProxy(
        ProviderTarget(id="provider", model="org/model"),
        token="secret-token",
        evidence_path=tmp_path / "evidence.jsonl",
        client=client,
    )
    base_url = proxy.start(host="127.0.0.1", port=0)
    capability = proxy.register_scope("compaction")
    content = [{"type": "text", "text": "summarize the conversation"}]
    try:
        response = httpx.post(
            f"{proxy.scoped_base_url(base_url, capability)}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": content}]},
            timeout=5,
        )
    finally:
        proxy.close()
        client.close()

    assert response.status_code == 200
    assert observed == [
        {
            "messages": [{"role": "user", "content": content}],
            "model": "org/model:fastest",
        }
    ]


@pytest.mark.parametrize(
    "content",
    [[], [{}], [{"type": ""}], [{"text": "missing type"}]],
)
def test_proxy_rejects_malformed_multipart_message_content(
    content: list[dict[str, str]],
) -> None:
    with pytest.raises(ValidationError):
        _provider_message(cast(JsonValue, {"role": "user", "content": content}))


def test_proxy_rejects_invalid_scope_and_capability(tmp_path: Path) -> None:
    def reject_upstream(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid routes must not reach the provider")

    client = httpx.Client(transport=httpx.MockTransport(reject_upstream))
    proxy = ProviderEvidenceProxy(
        ProviderTarget(id="provider", model="org/model"),
        token="secret-token",
        evidence_path=tmp_path / "evidence.jsonl",
        client=client,
        capability_factory=lambda: "short",
    )
    try:
        with pytest.raises(ValueError, match="request scope is invalid"):
            proxy.register_scope("invalid/scope")
        with pytest.raises(ProviderProxyError, match="route capability is invalid"):
            proxy.register_scope("valid-scope")
        with pytest.raises(ValueError, match="bind address is invalid"):
            proxy.start(host="", port=8000)
    finally:
        proxy.close()
        client.close()


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
                    "reasoning_content": "I should call the shell tool.",
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
                "reasoning_content": "I should call the shell tool.",
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
    base_url = proxy.start(host="127.0.0.1", port=0)
    throttled_url = proxy.scoped_base_url(
        base_url, proxy.register_scope("trial-throttled")
    )
    complete_url = proxy.scoped_base_url(
        base_url, proxy.register_scope("trial-complete")
    )
    timeout_url = proxy.scoped_base_url(base_url, proxy.register_scope("trial-timeout"))
    with pytest.raises(ProviderProxyError, match="already running"):
        proxy.start(host="127.0.0.1", port=0)
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


@pytest.mark.parametrize(
    ("encoding", "encoded"),
    [
        ("gzip", gzip.compress),
        ("deflate", zlib.compress),
    ],
)
def test_proxy_preserves_encoding_and_decodes_evidence(
    tmp_path: Path, encoding: str, encoded: Callable[[bytes], bytes]
) -> None:
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
                "content-encoding": encoding,
            },
            stream=httpx.ByteStream(encoded(payload)),
        )

    client = httpx.Client(transport=httpx.MockTransport(upstream))
    evidence = tmp_path / "evidence.jsonl"
    proxy = ProviderEvidenceProxy(
        ProviderTarget(id="provider", model="org/model"),
        token="token",
        evidence_path=evidence,
        client=client,
    )
    base_url = proxy.start(host="127.0.0.1", port=0)
    capability = proxy.register_scope("trial")
    try:
        response = httpx.post(
            f"{proxy.scoped_base_url(base_url, capability)}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "prompt"}]},
        )
    finally:
        proxy.close()
        client.close()

    assert response.content == payload
    assert response.headers["content-encoding"] == encoding
    assert json.loads(evidence.read_text())["status"] == "succeeded"


@pytest.mark.parametrize(
    ("encoding", "delimiter", "compression"),
    [("gzip", b"\n", "gzip"), ("deflate", b"\n", "raw"), (None, b"\r", None)],
)
def test_ttft_probe_decodes_compression_and_accepts_bare_cr(
    encoding: str | None, delimiter: bytes, compression: str | None
) -> None:
    stream = delimiter.join(
        [
            b'data: {"choices":[{"delta":{"content":"answer"}}]}',
            b"",
            b"data: [DONE]",
            b"",
        ]
    )
    if compression == "gzip":
        content = gzip.compress(stream)
    elif compression == "raw":
        compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
        content = compressor.compress(stream) + compressor.flush()
    else:
        content = stream
    headers = {"content-encoding": encoding} if encoding is not None else {}

    class FragmentedStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            for value in content:
                yield bytes([value])

    response = httpx.Response(
        200,
        headers=headers,
        stream=FragmentedStream(),
    )

    class Handler:
        close_connection = False
        wfile = type(
            "Output", (), {"write": lambda *_: None, "flush": lambda *_: None}
        )()

        def send_response(self, _status: int) -> None:
            pass

        def send_header(self, _name: str, _value: str) -> None:
            pass

        def end_headers(self) -> None:
            pass

    captured, semantic_output_ms = _relay_response(cast(Any, Handler()), response, 0)

    assert bytes(captured) == content
    assert semantic_output_ms is not None


def test_evidence_decoder_bounds_high_ratio_compressed_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_proxy, "_MAX_EVIDENCE_RESPONSE_BYTES", 64)
    headers = httpx.Headers({"content-encoding": "gzip"})

    content = gzip.compress(b"x" * 4096)
    decoded_headers, decoded, invalid = _decode_evidence_response(headers, content)

    assert decoded_headers == headers
    assert decoded == content
    assert invalid is True


def test_evidence_decoder_preserves_unknown_or_malformed_encoding() -> None:
    for encoding, content in (("custom", b"encoded"), ("gzip", b"truncated")):
        headers = httpx.Headers({"content-encoding": encoding})
        observed_headers, observed_content, invalid = _decode_evidence_response(
            headers, content
        )

        assert observed_headers["content-encoding"] == encoding
        assert observed_content == content
        assert invalid is True


@pytest.mark.parametrize("encoding", ["br", "zstd"])
def test_evidence_decoder_rejects_encodings_without_bounded_decoders(
    encoding: str,
) -> None:
    headers = httpx.Headers({"content-encoding": encoding})

    observed_headers, observed_content, invalid = _decode_evidence_response(
        headers, b"provider-controlled encoded bytes"
    )

    assert observed_headers == headers
    assert observed_content == b"provider-controlled encoded bytes"
    assert invalid is True


@pytest.mark.parametrize(
    ("encoding", "encoded"),
    [("gzip", gzip.compress), ("deflate", zlib.compress)],
)
def test_evidence_decoder_rejects_compressed_body_without_eof(
    encoding: str, encoded: Callable[[bytes], bytes]
) -> None:
    headers = httpx.Headers({"content-encoding": encoding})
    content = encoded(b'{"choices":[]}')[:-1]

    observed_headers, observed_content, invalid = _decode_evidence_response(
        headers, content
    )

    assert observed_headers == headers
    assert observed_content == content
    assert invalid is True


def test_evidence_decoder_rejects_trailing_gzip_member() -> None:
    headers = httpx.Headers({"content-encoding": "gzip"})
    content = gzip.compress(b'{"first":true}') + gzip.compress(b'{"second":true}')

    observed_headers, observed_content, invalid = _decode_evidence_response(
        headers, content
    )

    assert observed_headers == headers
    assert observed_content == content
    assert invalid is True


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
    assert observed.transport_interrupted is True
