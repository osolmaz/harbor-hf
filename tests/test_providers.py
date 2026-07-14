import hashlib
import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from harbor_hf.provider_models import (
    ExplicitProviderRoute,
    PolicyRoute,
    ProviderCallResult,
    ProviderChatRequest,
    ProviderLimits,
    ProviderMessage,
    ProviderTarget,
    ProviderTool,
    ProviderToolFunction,
)
from harbor_hf.providers import HfInferenceProviderAdapter, observe_provider_response

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class TickClock:
    def __init__(self, tick: float = 0.01) -> None:
        self.value = 100.0 - tick
        self.tick = tick

    def __call__(self) -> float:
        self.value += self.tick
        return self.value


def _target(
    *,
    attempts: int = 2,
    route: PolicyRoute | ExplicitProviderRoute | None = None,
) -> ProviderTarget:
    return ProviderTarget(
        id="provider-target",
        model="openai/gpt-oss-120b",
        routing=route or PolicyRoute(),
        limits=ProviderLimits(max_attempts=attempts),
    )


def _request(*, stream: bool = False, tools: bool = False) -> ProviderChatRequest:
    definitions = []
    if tools:
        definitions = [
            ProviderTool(
                function=ProviderToolFunction(
                    name="get_weather",
                    description="Get current weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                )
            )
        ]
    return ProviderChatRequest(
        request_id="request-1",
        messages=[ProviderMessage(role="user", content="weather in Paris")],
        tools=definitions,
        stream=stream,
        parameters={"temperature": 0},
    )


def _adapter(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    clock: Callable[[], float] | None = None,
) -> HfInferenceProviderAdapter:
    return HfInferenceProviderAdapter(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=clock or TickClock(),
    )


def _result_digest(result: ProviderCallResult) -> str:
    value = result.model_dump(mode="json")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_records_successful_response_usage_quota_and_latency() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _JSON_OBJECT.validate_json(request.content)
        assert request.url == "https://router.huggingface.co/v1/chat/completions"
        assert request.headers["authorization"].startswith("Bearer ")
        assert payload["model"] == "openai/gpt-oss-120b:fastest"
        assert payload["stream"] is False
        return httpx.Response(
            200,
            headers={
                "x-request-id": "provider-request-123",
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "99",
                "x-ratelimit-limit-tokens": "10000",
                "x-ratelimit-remaining-tokens": "9980",
                "x-ratelimit-reset-requests": "2s",
            },
            json={
                "id": "completion-1",
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Sunny"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                },
            },
        )

    result = _adapter(handler).chat_completion(
        _target(), _request(), token="mock-token"
    )

    assert result.status == "succeeded"
    assert result.remote_outcome == "completed"
    assert result.message is not None
    assert result.message.content == "Sunny"
    assert result.response_id.value == "completion-1"
    assert result.evidence.request.provider_request_id.value == "provider-request-123"
    assert result.evidence.model.requested == "openai/gpt-oss-120b"
    assert result.evidence.model.routed == "openai/gpt-oss-120b:fastest"
    assert result.evidence.model.response.value == "openai/gpt-oss-120b"
    assert result.evidence.routing.requested_value == "fastest"
    assert result.evidence.routing.selected_provider.status == "not_reported"
    assert result.evidence.quota.requests_remaining.value == 99
    assert result.evidence.usage.input_tokens.value == 12
    assert result.evidence.usage.output_tokens.value == 3
    assert result.evidence.usage.total_tokens.value == 15
    assert result.evidence.latency.total_ms.value == 10.0
    assert result.evidence.latency.time_to_first_token_ms.status == "not_applicable"


def test_explicit_provider_route_preserves_tool_request_and_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _JSON_OBJECT.validate_json(request.content)
        assert payload["model"] == "openai/gpt-oss-120b:groq"
        tools = payload["tools"]
        assert isinstance(tools, list)
        assert tools[0]["function"]["name"] == "get_weather"
        return httpx.Response(
            200,
            json={
                "id": "completion-tools",
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    result = _adapter(handler).chat_completion(
        _target(route=ExplicitProviderRoute(provider="groq")),
        _request(tools=True),
        token="mock-token",
    )

    assert result.status == "succeeded"
    assert result.message is not None
    assert result.message.content is None
    assert result.message.tool_calls[0].function_name == "get_weather"
    assert result.message.tool_calls[0].arguments == '{"city":"Paris"}'
    assert result.evidence.routing.requested_kind == "provider"
    assert result.evidence.routing.requested_value == "groq"
    assert result.evidence.request.tool_count == 1
    assert result.evidence.usage.input_tokens.status == "not_reported"


def test_streaming_contract_records_ttft_and_terminal_usage() -> None:
    stream = "\n".join(
        [
            'data: {"id":"stream-1","model":"owner/model","choices":'
            '[{"delta":{"content":"Hel"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":4,'
            '"completion_tokens":2,"total_tokens":6}}',
            "data: [DONE]",
            "",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = _JSON_OBJECT.validate_json(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        return httpx.Response(200, text=stream)

    result = _adapter(handler).chat_completion(
        _target(), _request(stream=True), token="mock-token"
    )

    assert result.status == "succeeded"
    assert result.message is not None
    assert result.message.content == "Hello"
    assert result.evidence.latency.time_to_first_token_ms.value == 10.0
    assert result.evidence.latency.total_ms.value == 40.0
    assert result.evidence.usage.total_tokens.value == 6


def test_throttling_exposes_retry_and_quota_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={
                "retry-after": "3",
                "x-ratelimit-limit-requests": "20",
                "x-ratelimit-remaining-requests": "0",
            },
            json={"error": "rate limited"},
        )

    result = _adapter(handler).chat_completion(
        _target(attempts=2), _request(), token="mock-token", attempt=1
    )

    assert result.status == "throttled"
    assert result.remote_outcome == "not_completed"
    assert result.error_code == "rate_limited"
    assert result.evidence.retry.disposition == "retry"
    assert result.evidence.retry.retry_after.value == "3"
    assert result.evidence.quota.request_limit.value == 20
    assert result.evidence.quota.requests_remaining.value == 0
    assert result.evidence.usage.total_tokens.status == "not_reported"


def test_exhausted_throttle_does_not_recommend_another_attempt() -> None:
    result = _adapter(lambda request: httpx.Response(429)).chat_completion(
        _target(attempts=1), _request(), token="mock-token", attempt=1
    )

    assert result.evidence.retry.disposition == "no_retry"


def test_timeout_is_ambiguous_and_requires_inspection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("provider did not finish", request=request)

    result = _adapter(handler).chat_completion(
        _target(), _request(stream=True), token="mock-token"
    )

    assert result.status == "timed_out"
    assert result.remote_outcome == "ambiguous"
    assert result.error_code == "timeout"
    assert result.evidence.retry.disposition == "inspect"
    assert result.evidence.latency.time_to_first_token_ms.status == "not_observed"


def test_interrupted_response_body_is_ambiguous_even_after_success_headers() -> None:
    result = observe_provider_response(
        _target(),
        _request(),
        attempt=1,
        status_code=200,
        headers=httpx.Headers({"x-request-id": "upstream-one"}),
        content=b'{"id":"partial"',
        total_ms=12.5,
        transport_interrupted=True,
    )

    assert result.status == "provider_error"
    assert result.remote_outcome == "ambiguous"
    assert result.error_code == "transport_interrupted"
    assert result.evidence.retry.disposition == "inspect"
    assert result.evidence.request.provider_request_id.value == "upstream-one"


@pytest.mark.parametrize(
    ("stream", "ttft", "digest"),
    [
        (
            False,
            None,
            "7d1ace571f577618de94d57c4bc98ad7c0542569a134eafb1d07ff804e204cf7",
        ),
        (
            True,
            None,
            "72983856756099c61558f6b3ddfeaa985a5ad83268e6532b09c7e19cd9345a9e",
        ),
        (
            True,
            4.5,
            "583b967c55a8cf8caa52640188c5f248793aed9eb617c5de4b8f07a1510f7c0f",
        ),
    ],
)
def test_interrupted_response_evidence_contract_is_stable(
    stream: bool, ttft: float | None, digest: str
) -> None:
    result = observe_provider_response(
        _target(),
        _request(stream=stream),
        attempt=1,
        status_code=200,
        headers=httpx.Headers({"x-request-id": "upstream-one"}),
        content=b'{"id":"partial"',
        total_ms=12.5,
        time_to_first_token_ms=ttft,
        transport_interrupted=True,
    )

    assert _result_digest(result) == digest


@pytest.mark.parametrize(
    ("stream", "digest"),
    [
        (
            False,
            "9b2037399c55e1e33404fdedf98219f546158070989091643a102f5259eb572e",
        ),
        (
            True,
            "cd13ccb2499ba232366919824f55604bd36d7e6a0a18191228f8b640fef11945",
        ),
    ],
)
def test_malformed_content_encoding_is_ambiguous_and_stable(
    stream: bool, digest: str
) -> None:
    result = observe_provider_response(
        _target(),
        _request(stream=stream),
        attempt=1,
        status_code=200,
        headers=httpx.Headers({"content-encoding": "gzip"}),
        content=b"not-gzip",
        total_ms=12.5,
    )

    assert result.status == "malformed_response"
    assert result.remote_outcome == "ambiguous"
    assert result.error_code == "invalid_content_encoding"
    assert _result_digest(result) == digest


@pytest.mark.parametrize(
    ("usage", "field", "detail"),
    [
        (
            {"prompt_tokens": "many", "completion_tokens": 2, "total_tokens": 6},
            "input_tokens",
            "prompt_tokens must be a non-negative integer",
        ),
        (
            {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 9},
            "total_tokens",
            "total_tokens does not equal input plus output",
        ),
    ],
)
def test_malformed_usage_does_not_discard_valid_completion(
    usage: dict[str, JsonValue], field: str, detail: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "completion-usage",
                "choices": [{"message": {"role": "assistant", "content": "answer"}}],
                "usage": usage,
            },
        )

    result = _adapter(handler).chat_completion(
        _target(), _request(), token="mock-token"
    )

    assert result.status == "succeeded"
    usage_field = getattr(result.evidence.usage, field)
    assert usage_field.status == "malformed"
    assert usage_field.detail == detail


def test_malformed_completion_is_completed_but_not_retried_blindly() -> None:
    result = _adapter(
        lambda request: httpx.Response(200, json={"choices": []})
    ).chat_completion(_target(), _request(), token="mock-token")

    assert result.status == "malformed_response"
    assert result.remote_outcome == "completed"
    assert result.error_code == "invalid_completion"
    assert result.evidence.retry.disposition == "inspect"


def test_truncated_stream_is_ambiguous() -> None:
    result = _adapter(
        lambda request: httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"partial"}}]}\n',
        )
    ).chat_completion(_target(), _request(stream=True), token="mock-token")

    assert result.status == "malformed_response"
    assert result.remote_outcome == "ambiguous"
    assert result.error_code == "invalid_stream"
    assert result.evidence.retry.disposition == "inspect"


def test_observed_sse_preserves_unicode_line_separator_inside_json() -> None:
    content = (
        'data: {"id":"one","choices":[{"delta":{"content":"left'
        + "\u2028"
        + 'right"}}]}\n\n'
        + 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        + "data: [DONE]\n\n"
    ).encode()

    result = observe_provider_response(
        _target(),
        _request(stream=True),
        attempt=1,
        status_code=200,
        headers=httpx.Headers({"content-type": "text/event-stream"}),
        content=content,
        total_ms=10,
    )

    assert result.status == "succeeded"
    assert result.message is not None
    assert result.message.content == "left\u2028right"


def test_reasoning_only_budget_exhaustion_is_a_valid_stream_completion() -> None:
    stream = (
        'data: {"choices":[{"delta":{"reasoning_content":"thinking"},'
        '"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
        "data: [DONE]\n\n"
    )
    result = _adapter(lambda request: httpx.Response(200, text=stream)).chat_completion(
        _target(), _request(stream=True), token="mock-token"
    )

    assert result.status == "succeeded"
    assert result.finish_reason.value == "length"
    assert result.message is not None
    assert result.message.content == ""


def test_provider_evidence_never_guesses_endpoint_runtime_details() -> None:
    result = _adapter(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
        )
    ).chat_completion(_target(), _request(), token="mock-token")

    endpoint = result.evidence.endpoint
    assert endpoint.endpoint_name.status == "not_applicable"
    assert endpoint.endpoint_status.status == "not_applicable"
    assert endpoint.ready_replicas.status == "not_applicable"
    assert endpoint.region.status == "not_reported"
    assert endpoint.hardware.status == "not_reported"
    assert endpoint.engine.status == "not_reported"
    assert endpoint.precision.status == "not_reported"
