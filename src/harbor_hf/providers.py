from __future__ import annotations

import re
from collections.abc import Callable
from time import perf_counter
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from harbor_hf.provider_models import (
    EvidenceValue,
    ExplicitProviderRoute,
    ProviderCallResult,
    ProviderChatRequest,
    ProviderEndpointEvidence,
    ProviderEvidence,
    ProviderLatencyEvidence,
    ProviderMessage,
    ProviderModelEvidence,
    ProviderQuotaEvidence,
    ProviderRequestEvidence,
    ProviderRetryEvidence,
    ProviderRoutingEvidence,
    ProviderTarget,
    ProviderToolCall,
    ProviderUsageEvidence,
    malformed,
    observed,
    unavailable,
)

HF_INFERENCE_PROVIDER_BASE_URL = "https://router.huggingface.co"
_CHAT_COMPLETIONS_URL = f"{HF_INFERENCE_PROVIDER_BASE_URL}/v1/chat/completions"
_REQUEST_ID_HEADERS = ("x-request-id", "x-amzn-requestid", "request-id")


class BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _RawToolFunction(BoundaryModel):
    name: str
    arguments: str


class _RawToolCall(BoundaryModel):
    id: str
    type: Literal["function"]
    function: _RawToolFunction


class _RawMessage(BoundaryModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    reasoning: str | None = None
    tool_calls: list[_RawToolCall] = Field(default_factory=list)


class _RawChoice(BoundaryModel):
    finish_reason: str | None = None
    message: _RawMessage


class _RawCompletion(BoundaryModel):
    id: str | None = None
    model: str | None = None
    choices: list[_RawChoice] = Field(min_length=1)
    usage: dict[str, JsonValue] | None = None


class _RawStreamToolFunction(BoundaryModel):
    name: str | None = None
    arguments: str | None = None


class _RawStreamToolCall(BoundaryModel):
    index: int = Field(ge=0)
    id: str | None = None
    type: Literal["function"] | None = None
    function: _RawStreamToolFunction | None = None


class _RawDelta(BoundaryModel):
    content: str | None = None
    reasoning_content: str | None = None
    reasoning: str | None = None
    tool_calls: list[_RawStreamToolCall] = Field(default_factory=list)


class _RawStreamChoice(BoundaryModel):
    finish_reason: str | None = None
    delta: _RawDelta


class _RawStreamChunk(BoundaryModel):
    id: str | None = None
    model: str | None = None
    choices: list[_RawStreamChoice] = Field(default_factory=list)
    usage: dict[str, JsonValue] | None = None


class _StreamToolState:
    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments = ""

    def add(self, value: _RawStreamToolCall) -> None:
        if value.id is not None:
            self.id = value.id
        if value.function is None:
            return
        if value.function.name is not None:
            self.name += value.function.name
        if value.function.arguments is not None:
            self.arguments += value.function.arguments

    def build(self) -> ProviderToolCall:
        return ProviderToolCall(
            id=self.id,
            function_name=self.name,
            arguments=self.arguments,
        )


class _StreamState:
    def __init__(self) -> None:
        self.response_id: str | None = None
        self.response_model: str | None = None
        self.content: list[str] = []
        self.saw_reasoning = False
        self.tools: dict[int, _StreamToolState] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, JsonValue] | None = None
        self.saw_payload = False
        self.saw_done = False
        self.first_token_ms: float | None = None

    def add(self, chunk: _RawStreamChunk, elapsed_ms: float) -> None:
        self.response_id = self.response_id or chunk.id
        self.response_model = self.response_model or chunk.model
        self.usage = chunk.usage or self.usage
        for choice in chunk.choices:
            if choice.finish_reason is not None:
                self.finish_reason = choice.finish_reason
            delta = choice.delta
            if delta.content:
                self.content.append(delta.content)
                self._record_first_token(elapsed_ms)
            if delta.reasoning_content or delta.reasoning:
                self.saw_reasoning = True
                self._record_first_token(elapsed_ms)
            for tool in delta.tool_calls:
                state = self.tools.setdefault(tool.index, _StreamToolState())
                state.add(tool)
                self._record_first_token(elapsed_ms)
        self.saw_payload = self.saw_payload or bool(chunk.choices)

    def _record_first_token(self, elapsed_ms: float) -> None:
        if self.first_token_ms is None:
            self.first_token_ms = elapsed_ms

    def message(self) -> ProviderMessage:
        content = (
            "".join(self.content)
            if self.content
            else ("" if self.saw_reasoning else None)
        )
        tools = [self.tools[index].build() for index in sorted(self.tools)]
        return ProviderMessage(role="assistant", content=content, tool_calls=tools)


class HfInferenceProviderAdapter:
    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._client = client or httpx.Client()
        self._clock = clock

    def chat_completion(
        self,
        target: ProviderTarget,
        request: ProviderChatRequest,
        *,
        token: str,
        attempt: int = 1,
    ) -> ProviderCallResult:
        if not token:
            raise ValueError("provider token must not be empty")
        if attempt > target.limits.max_attempts:
            raise ValueError("provider attempt exceeds the configured retry budget")
        started = self._clock()
        payload = _request_payload(target, request)
        headers = {"Authorization": f"Bearer {token}"}
        if request.stream:
            return self._stream(target, request, attempt, payload, headers, started)
        return self._complete(target, request, attempt, payload, headers, started)

    def _complete(
        self,
        target: ProviderTarget,
        request: ProviderChatRequest,
        attempt: int,
        payload: dict[str, JsonValue],
        headers: dict[str, str],
        started: float,
    ) -> ProviderCallResult:
        try:
            response = self._client.post(
                _CHAT_COMPLETIONS_URL,
                json=payload,
                headers=headers,
                timeout=target.timeout_seconds,
            )
        except httpx.TimeoutException:
            return self._timeout_result(target, request, attempt, started)
        total_ms = _elapsed_ms(started, self._clock())
        if not response.is_success:
            return _http_failure(
                target,
                request,
                attempt,
                response.status_code,
                response.headers,
                total_ms,
            )
        try:
            raw = _RawCompletion.model_validate_json(response.content)
            choice = raw.choices[0]
            message = _provider_message(choice.message)
        except (ValidationError, ValueError):
            return _malformed_result(
                target, request, attempt, response, total_ms, "invalid_completion"
            )
        evidence = _evidence(
            target,
            request,
            attempt,
            response.headers,
            total_ms,
            unavailable("not_applicable"),
            raw.model,
            raw.usage,
            "no_retry",
        )
        return ProviderCallResult(
            status="succeeded",
            remote_outcome="completed",
            response_id=_optional_string(raw.id),
            finish_reason=_optional_string(choice.finish_reason),
            message=message,
            evidence=evidence,
        )

    def _stream(
        self,
        target: ProviderTarget,
        request: ProviderChatRequest,
        attempt: int,
        payload: dict[str, JsonValue],
        headers: dict[str, str],
        started: float,
    ) -> ProviderCallResult:
        state = _StreamState()
        try:
            with self._client.stream(
                "POST",
                _CHAT_COMPLETIONS_URL,
                json=payload,
                headers=headers,
                timeout=target.timeout_seconds,
            ) as response:
                if not response.is_success:
                    total_ms = _elapsed_ms(started, self._clock())
                    return _http_failure(
                        target,
                        request,
                        attempt,
                        response.status_code,
                        response.headers,
                        total_ms,
                    )
                for line in response.iter_lines():
                    _consume_stream_line(line, state, started, self._clock)
                total_ms = _elapsed_ms(started, self._clock())
                return _stream_result(
                    target, request, attempt, response, state, total_ms
                )
        except httpx.TimeoutException:
            return self._timeout_result(target, request, attempt, started)
        except (ValidationError, ValueError):
            total_ms = _elapsed_ms(started, self._clock())
            return _stream_malformed_result(target, request, attempt, state, total_ms)

    def _timeout_result(
        self,
        target: ProviderTarget,
        request: ProviderChatRequest,
        attempt: int,
        started: float,
    ) -> ProviderCallResult:
        total_ms = _elapsed_ms(started, self._clock())
        ttft = (
            unavailable("not_observed")
            if request.stream
            else unavailable("not_applicable")
        )
        evidence = _evidence(
            target,
            request,
            attempt,
            httpx.Headers(),
            total_ms,
            ttft,
            None,
            None,
            "inspect",
        )
        return ProviderCallResult(
            status="timed_out",
            remote_outcome="ambiguous",
            response_id=unavailable("not_observed"),
            finish_reason=unavailable("not_observed"),
            error_code="timeout",
            evidence=evidence,
        )


def _request_payload(
    target: ProviderTarget, request: ProviderChatRequest
) -> dict[str, JsonValue]:
    payload = dict(target.parameters)
    payload.update(request.parameters)
    payload.update(
        {
            "model": _routed_model(target),
            "messages": [_message_payload(message) for message in request.messages],
            "stream": request.stream,
        }
    )
    if request.tools:
        payload["tools"] = [tool.model_dump(mode="json") for tool in request.tools]
    if request.stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _message_payload(message: ProviderMessage) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    if message.reasoning_content is not None:
        payload["reasoning_content"] = message.reasoning_content
    if message.reasoning is not None:
        payload["reasoning"] = message.reasoning
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function_name,
                    "arguments": call.arguments,
                },
            }
            for call in message.tool_calls
        ]
    return payload


def _routed_model(target: ProviderTarget) -> str:
    route = target.routing
    suffix = (
        route.provider if isinstance(route, ExplicitProviderRoute) else route.policy
    )
    return f"{target.model}:{suffix}"


def routed_provider_model(target: ProviderTarget) -> str:
    return _routed_model(target)


def observe_provider_response(
    target: ProviderTarget,
    request: ProviderChatRequest,
    *,
    attempt: int,
    status_code: int,
    headers: httpx.Headers,
    content: bytes,
    total_ms: float,
    time_to_first_token_ms: float | None = None,
    transport_interrupted: bool = False,
    invalid_content_encoding: bool = False,
) -> ProviderCallResult:
    """Normalize evidence from a transparently forwarded provider response."""
    early = _early_provider_response(
        target,
        request,
        attempt,
        status_code,
        headers,
        total_ms,
        time_to_first_token_ms,
        transport_interrupted,
        invalid_content_encoding,
    )
    if early is not None:
        return early
    try:
        response = httpx.Response(status_code, headers=headers, content=content)
    except httpx.DecodingError:
        return _malformed_encoding_result(target, request, attempt, headers, total_ms)
    if request.stream:
        state = _StreamState()
        elapsed = time_to_first_token_ms or 0.0
        try:
            for line in re.split(r"\r\n?|\n", content.decode("utf-8")):
                _consume_stream_line(line, state, 0.0, lambda: elapsed / 1000)
            state.first_token_ms = time_to_first_token_ms
            return _stream_result(target, request, attempt, response, state, total_ms)
        except (UnicodeDecodeError, ValidationError, ValueError):
            return _stream_malformed_result(target, request, attempt, state, total_ms)
    try:
        raw = _RawCompletion.model_validate_json(content)
        choice = raw.choices[0]
        message = _provider_message(choice.message)
    except (ValidationError, ValueError):
        return _malformed_result(
            target, request, attempt, response, total_ms, "invalid_completion"
        )
    evidence = _evidence(
        target,
        request,
        attempt,
        headers,
        total_ms,
        unavailable("not_applicable"),
        raw.model,
        raw.usage,
        "no_retry",
    )
    return ProviderCallResult(
        status="succeeded",
        remote_outcome="completed",
        response_id=_optional_string(raw.id),
        finish_reason=_optional_string(choice.finish_reason),
        message=message,
        evidence=evidence,
    )


def _early_provider_response(
    target: ProviderTarget,
    request: ProviderChatRequest,
    attempt: int,
    status_code: int,
    headers: httpx.Headers,
    total_ms: float,
    time_to_first_token_ms: float | None,
    transport_interrupted: bool,
    invalid_content_encoding: bool,
) -> ProviderCallResult | None:
    if not 200 <= status_code < 300:
        return _http_failure(target, request, attempt, status_code, headers, total_ms)
    if transport_interrupted:
        return _transport_interrupted_result(
            target,
            request,
            attempt,
            headers,
            total_ms,
            time_to_first_token_ms,
        )
    if invalid_content_encoding:
        return _malformed_encoding_result(target, request, attempt, headers, total_ms)
    return None


def _transport_interrupted_result(
    target: ProviderTarget,
    request: ProviderChatRequest,
    attempt: int,
    headers: httpx.Headers,
    total_ms: float,
    time_to_first_token_ms: float | None,
) -> ProviderCallResult:
    if request.stream and time_to_first_token_ms is not None:
        ttft = observed(time_to_first_token_ms)
    elif request.stream:
        ttft = unavailable("not_observed")
    else:
        ttft = unavailable("not_applicable")
    evidence = _evidence(
        target,
        request,
        attempt,
        headers,
        total_ms,
        ttft,
        None,
        None,
        "inspect",
    )
    return ProviderCallResult(
        status="provider_error",
        remote_outcome="ambiguous",
        response_id=unavailable("not_observed"),
        finish_reason=unavailable("not_observed"),
        error_code="transport_interrupted",
        evidence=evidence,
    )


def _malformed_encoding_result(
    target: ProviderTarget,
    request: ProviderChatRequest,
    attempt: int,
    headers: httpx.Headers,
    total_ms: float,
) -> ProviderCallResult:
    evidence = _evidence(
        target,
        request,
        attempt,
        headers,
        total_ms,
        (
            unavailable("not_observed")
            if request.stream
            else unavailable("not_applicable")
        ),
        None,
        None,
        "inspect",
    )
    return ProviderCallResult(
        status="malformed_response",
        remote_outcome="ambiguous",
        response_id=unavailable("not_observed"),
        finish_reason=unavailable("not_observed"),
        error_code="invalid_content_encoding",
        evidence=evidence,
    )


def _provider_message(value: _RawMessage) -> ProviderMessage:
    return ProviderMessage(
        role="assistant",
        content=value.content,
        reasoning_content=value.reasoning_content,
        reasoning=value.reasoning,
        tool_calls=[
            ProviderToolCall(
                id=tool.id,
                function_name=tool.function.name,
                arguments=tool.function.arguments,
            )
            for tool in value.tool_calls
        ],
    )


def _consume_stream_line(
    line: str,
    state: _StreamState,
    started: float,
    clock: Callable[[], float],
) -> None:
    if not line or line.startswith(":"):
        return
    if not line.startswith("data:"):
        raise ValueError("stream line is not an SSE data event")
    data = line.removeprefix("data:").strip()
    if data == "[DONE]":
        state.saw_done = True
        return
    chunk = _RawStreamChunk.model_validate_json(data)
    state.add(chunk, _elapsed_ms(started, clock()))


def _stream_result(
    target: ProviderTarget,
    request: ProviderChatRequest,
    attempt: int,
    response: httpx.Response,
    state: _StreamState,
    total_ms: float,
) -> ProviderCallResult:
    if not state.saw_done or not state.saw_payload:
        return _stream_malformed_result(target, request, attempt, state, total_ms)
    try:
        message = state.message()
    except ValidationError:
        return _stream_malformed_result(target, request, attempt, state, total_ms)
    ttft = (
        observed(state.first_token_ms)
        if state.first_token_ms is not None
        else unavailable("not_observed")
    )
    evidence = _evidence(
        target,
        request,
        attempt,
        response.headers,
        total_ms,
        ttft,
        state.response_model,
        state.usage,
        "no_retry",
    )
    return ProviderCallResult(
        status="succeeded",
        remote_outcome="completed",
        response_id=_optional_string(state.response_id),
        finish_reason=_optional_string(state.finish_reason),
        message=message,
        evidence=evidence,
    )


def _stream_malformed_result(
    target: ProviderTarget,
    request: ProviderChatRequest,
    attempt: int,
    state: _StreamState,
    total_ms: float,
) -> ProviderCallResult:
    evidence = _evidence(
        target,
        request,
        attempt,
        httpx.Headers(),
        total_ms,
        unavailable("not_observed"),
        state.response_model,
        state.usage,
        "inspect",
    )
    return ProviderCallResult(
        status="malformed_response",
        remote_outcome="ambiguous",
        response_id=_optional_string(state.response_id),
        finish_reason=_optional_string(state.finish_reason),
        error_code="invalid_stream",
        evidence=evidence,
    )


def _malformed_result(
    target: ProviderTarget,
    request: ProviderChatRequest,
    attempt: int,
    response: httpx.Response,
    total_ms: float,
    error_code: str,
) -> ProviderCallResult:
    evidence = _evidence(
        target,
        request,
        attempt,
        response.headers,
        total_ms,
        unavailable("not_applicable"),
        None,
        None,
        "inspect",
    )
    return ProviderCallResult(
        status="malformed_response",
        remote_outcome="completed",
        response_id=unavailable("not_reported"),
        finish_reason=unavailable("not_reported"),
        error_code=error_code,
        evidence=evidence,
    )


def _http_failure(
    target: ProviderTarget,
    request: ProviderChatRequest,
    attempt: int,
    status_code: int,
    headers: httpx.Headers,
    total_ms: float,
) -> ProviderCallResult:
    status, outcome, code = _classify_status(status_code)
    can_retry = attempt < target.limits.max_attempts
    disposition: Literal["no_retry", "retry", "inspect"]
    disposition = "retry" if can_retry and status == "throttled" else "no_retry"
    if status == "provider_error" and can_retry:
        disposition = "inspect"
    evidence = _evidence(
        target,
        request,
        attempt,
        headers,
        total_ms,
        (
            unavailable("not_observed")
            if request.stream
            else unavailable("not_applicable")
        ),
        None,
        None,
        disposition,
    )
    return ProviderCallResult(
        status=status,
        remote_outcome=outcome,
        response_id=unavailable("not_reported"),
        finish_reason=unavailable("not_applicable"),
        error_code=code,
        evidence=evidence,
    )


def _classify_status(
    status_code: int,
) -> tuple[
    Literal["throttled", "rejected", "provider_error"],
    Literal["not_completed", "ambiguous"],
    str,
]:
    if status_code == 429:
        return "throttled", "not_completed", "rate_limited"
    if 400 <= status_code < 500:
        return "rejected", "not_completed", f"http_{status_code}"
    return "provider_error", "ambiguous", f"http_{status_code}"


def _evidence(
    target: ProviderTarget,
    request: ProviderChatRequest,
    attempt: int,
    headers: httpx.Headers,
    total_ms: float,
    ttft: EvidenceValue[float],
    response_model: str | None,
    usage: dict[str, JsonValue] | None,
    retry_disposition: Literal["no_retry", "retry", "inspect"],
) -> ProviderEvidence:
    route = target.routing
    route_kind = route.kind
    route_value = (
        route.provider if isinstance(route, ExplicitProviderRoute) else route.policy
    )
    return ProviderEvidence(
        request=ProviderRequestEvidence(
            request_id=request.request_id,
            provider_request_id=_first_header(headers, _REQUEST_ID_HEADERS),
            streaming=request.stream,
            message_count=len(request.messages),
            tool_count=len(request.tools),
        ),
        model=ProviderModelEvidence(
            requested=target.model,
            routed=_routed_model(target),
            response=_optional_string(response_model),
        ),
        routing=ProviderRoutingEvidence(
            requested_kind=route_kind,
            requested_value=route_value,
            selected_provider=unavailable("not_reported"),
        ),
        quota=_quota_evidence(headers),
        retry=ProviderRetryEvidence(
            attempt=attempt,
            max_attempts=target.limits.max_attempts,
            disposition=retry_disposition,
            retry_after=_optional_header(headers, "retry-after"),
        ),
        usage=_usage_evidence(usage),
        latency=ProviderLatencyEvidence(
            total_ms=observed(total_ms),
            time_to_first_token_ms=ttft,
        ),
        endpoint=ProviderEndpointEvidence(),
    )


def _quota_evidence(headers: httpx.Headers) -> ProviderQuotaEvidence:
    return ProviderQuotaEvidence(
        request_limit=_integer_header(headers, "x-ratelimit-limit-requests"),
        requests_remaining=_integer_header(headers, "x-ratelimit-remaining-requests"),
        token_limit=_integer_header(headers, "x-ratelimit-limit-tokens"),
        tokens_remaining=_integer_header(headers, "x-ratelimit-remaining-tokens"),
        reset=_first_header(
            headers,
            ("x-ratelimit-reset-requests", "x-ratelimit-reset"),
        ),
    )


def _usage_evidence(usage: dict[str, JsonValue] | None) -> ProviderUsageEvidence:
    if usage is None:
        return ProviderUsageEvidence(
            input_tokens=unavailable("not_reported"),
            output_tokens=unavailable("not_reported"),
            total_tokens=unavailable("not_reported"),
        )
    input_tokens = _token_value(usage, "prompt_tokens")
    output_tokens = _token_value(usage, "completion_tokens")
    total_tokens = _token_value(usage, "total_tokens")
    input_value = input_tokens.value
    output_value = output_tokens.value
    total_value = total_tokens.value
    if (
        input_value is not None
        and output_value is not None
        and total_value is not None
        and total_value != input_value + output_value
    ):
        total_tokens = malformed("total_tokens does not equal input plus output")
    return ProviderUsageEvidence(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _token_value(usage: dict[str, JsonValue], key: str) -> EvidenceValue[int]:
    if key not in usage:
        return unavailable("not_reported")
    value = usage[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return malformed(f"{key} must be a non-negative integer")
    return observed(value)


def _integer_header(headers: httpx.Headers, name: str) -> EvidenceValue[int]:
    value = headers.get(name)
    if value is None:
        return unavailable("not_reported")
    try:
        parsed = int(value)
    except ValueError:
        return malformed(f"{name} must be a non-negative integer")
    if parsed < 0:
        return malformed(f"{name} must be a non-negative integer")
    return observed(parsed)


def _first_header(headers: httpx.Headers, names: tuple[str, ...]) -> EvidenceValue[str]:
    for name in names:
        if value := headers.get(name):
            return observed(value)
    return unavailable("not_reported")


def _optional_header(headers: httpx.Headers, name: str) -> EvidenceValue[str]:
    value = headers.get(name)
    return observed(value) if value else unavailable("not_reported")


def _optional_string(value: str | None) -> EvidenceValue[str]:
    return observed(value) if value else unavailable("not_reported")


def _elapsed_ms(started: float, finished: float) -> float:
    return round(max(0.0, finished - started) * 1000, 3)
