from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import httpx

from harbor_hf.provider_models import (
    ExplicitProviderRoute,
    ProviderChatRequest,
    ProviderLimits,
    ProviderMessage,
    ProviderTarget,
    ProviderTool,
    ProviderToolCall,
    ProviderToolFunction,
)
from harbor_hf.providers import HfInferenceProviderAdapter


class TickClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


def _hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _target() -> ProviderTarget:
    return ProviderTarget(
        id="provider-mutation",
        model="owner/model",
        routing=ExplicitProviderRoute(provider="groq"),
        timeout_seconds=17,
        limits=ProviderLimits(max_concurrent_requests=3, max_attempts=2),
        parameters={"top_p": 0.9, "seed": 7},
    )


def _request(*, stream: bool = False) -> ProviderChatRequest:
    prior_call = ProviderToolCall(
        id="call-prior",
        function_name="lookup",
        arguments='{"query":"weather"}',
    )
    return ProviderChatRequest(
        request_id="request-mutation",
        messages=[
            ProviderMessage(role="system", content="Be precise"),
            ProviderMessage(role="user", content="Weather in Paris?"),
            ProviderMessage(role="assistant", tool_calls=[prior_call]),
            ProviderMessage(
                role="tool", tool_call_id="call-prior", content="Sunny, 25 C"
            ),
        ],
        tools=[
            ProviderTool(
                function=ProviderToolFunction(
                    name="lookup",
                    description="Look up weather",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                )
            )
        ],
        parameters={"temperature": 0, "max_tokens": 64},
        stream=stream,
    )


def _adapter(
    handler: Callable[[httpx.Request], httpx.Response],
) -> HfInferenceProviderAdapter:
    return HfInferenceProviderAdapter(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=TickClock(),
    )


def test_complete_provider_request_and_response_evidence_are_canonical() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "authorized": request.headers.get("authorization")
                == "Bearer test-token",
                "payload": json.loads(request.content),
            }
        )
        return httpx.Response(
            200,
            headers={
                "x-amzn-requestid": "provider-request-mutation",
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "98",
                "x-ratelimit-limit-tokens": "10000",
                "x-ratelimit-remaining-tokens": "9900",
                "x-ratelimit-reset": "2s",
                "retry-after": "5",
            },
            json={
                "id": "completion-mutation",
                "model": "owner/model-reported",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-next",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"query":"forecast"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                },
            },
        )

    result = _adapter(handler).chat_completion(
        _target(), _request(), token="test-token", attempt=2
    )
    corpus = {"calls": calls, "result": result.model_dump(mode="json")}

    assert _hash(corpus) == (
        "b89dbdafd48bed0098b3b86feaee3165b86297555be2dc10fe04abcaca34331b"
    )


def test_stream_fragments_and_terminal_evidence_are_canonical() -> None:
    stream = "\n".join(
        [
            'data: {"id":"stream-mutation","model":"owner/model-reported",'
            '"choices":[{"delta":{"content":"Hel","tool_calls":[]},'
            '"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"lo ","tool_calls":['
            '{"index":0,"id":"call-stream","type":"function",'
            '"function":{"name":"look","arguments":"{\\"query\\":"}}]},'
            '"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"name":"up","arguments":"\\"Paris\\"}"}}]},'
            '"finish_reason":"tool_calls"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":8,'
            '"completion_tokens":3,"total_tokens":11}}',
            "data: [DONE]",
            "",
        ]
    )

    result = _adapter(lambda request: httpx.Response(200, text=stream)).chat_completion(
        _target(), _request(stream=True), token="test-token"
    )

    assert _hash(result.model_dump(mode="json")) == (
        "2fbb106b6c52d05a8335e445c309592aed2993d531bba992a4cc682485b62a0d"
    )


def test_provider_failure_evidence_matrix_is_canonical() -> None:
    cases: list[dict[str, object]] = []
    for status, attempt, headers in [
        (400, 1, {}),
        (429, 1, {"retry-after": "3", "x-ratelimit-remaining-requests": "0"}),
        (429, 2, {"retry-after": "3"}),
        (503, 1, {"x-request-id": "failed-request"}),
        (503, 2, {}),
    ]:
        result = _adapter(
            lambda request, status=status, headers=headers: httpx.Response(
                status, headers=headers, json={"error": "provider failure"}
            )
        ).chat_completion(_target(), _request(), token="test-token", attempt=attempt)
        cases.append(
            {
                "status_code": status,
                "attempt": attempt,
                "result": result.model_dump(mode="json"),
            }
        )

    malformed = _adapter(
        lambda request: httpx.Response(200, json={"choices": []})
    ).chat_completion(_target(), _request(), token="test-token")
    cases.append({"malformed": malformed.model_dump(mode="json")})

    assert _hash(cases) == (
        "01c5cc277240be829b04c2efee82faf67fd260cabddfa2ffaae8428468495be3"
    )


def test_timeout_and_malformed_stream_matrix_has_complete_evidence() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    results = [
        _adapter(timeout)
        .chat_completion(_target(), _request(stream=stream), token="test-token")
        .model_dump(mode="json")
        for stream in [False, True]
    ]
    for stream in [
        "",
        ": keepalive\n",
        "data: [DONE]\n",
        'data: {"choices":[]}\ndata: [DONE]\n',
        'event: message\ndata: {"choices":[]}\n',
        "data: not-json\n",
    ]:
        results.append(
            _adapter(lambda request, stream=stream: httpx.Response(200, text=stream))
            .chat_completion(_target(), _request(stream=True), token="test-token")
            .model_dump(mode="json")
        )

    assert _hash(results) == (
        "99a9d494840651830854fb283ff56f56a83bdee56ac28b314a642bdda6f30e49"
    )


def test_malformed_quota_and_usage_evidence_matrix_is_complete() -> None:
    cases: list[object] = []
    headers = {
        "x-ratelimit-limit-requests": "many",
        "x-ratelimit-remaining-requests": "-1",
        "x-ratelimit-limit-tokens": "0",
        "x-ratelimit-remaining-tokens": "7",
        "x-ratelimit-reset-requests": "1s",
    }
    for usage in [
        None,
        {},
        {"prompt_tokens": True, "completion_tokens": -1, "total_tokens": "3"},
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 9},
    ]:
        body: dict[str, object] = {
            "id": "completion-evidence",
            "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
        }
        if usage is not None:
            body["usage"] = usage
        result = _adapter(
            lambda request, body=body: httpx.Response(200, headers=headers, json=body)
        ).chat_completion(_target(), _request(), token="test-token")
        cases.append(result.model_dump(mode="json"))

    assert _hash(cases) == (
        "98f7fea3ecbc2f4ad9a8964bc1208debab7b354fbb27004619da357ecb54cd16"
    )
