from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import cast

import httpx
from pydantic import JsonValue, ValidationError

from harbor_hf.provider_models import (
    ProviderChatRequest,
    ProviderMessage,
    ProviderTarget,
    ProviderTool,
    ProviderToolCall,
)
from harbor_hf.providers import (
    HF_INFERENCE_PROVIDER_BASE_URL,
    observe_provider_response,
    routed_provider_model,
)

_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_MAX_EVIDENCE_RESPONSE_BYTES = 32 * 1024 * 1024
_REQUEST_SCOPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SCOPED_COMPLETIONS = re.compile(
    r"^/scopes/(?P<scope>[A-Za-z0-9][A-Za-z0-9._-]{0,127})/v1/chat/completions/?$"
)
_FORWARDED_RESPONSE_HEADERS = {
    "content-type",
    "retry-after",
    "x-request-id",
    "x-amzn-requestid",
    "request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset",
}


class ProviderProxyError(RuntimeError):
    """Raised when the hosted provider evidence proxy cannot run safely."""


@dataclass(frozen=True)
class _ObservedResponse:
    status_code: int
    headers: httpx.Headers
    content: bytes
    total_ms: float
    semantic_output_ms: float | None


class ProviderEvidenceProxy:
    """Forward OpenAI chat requests while recording content-free evidence."""

    def __init__(
        self,
        target: ProviderTarget,
        *,
        token: str,
        evidence_path: Path,
        client: httpx.Client | None = None,
    ) -> None:
        if not token:
            raise ValueError("provider proxy token must not be empty")
        self.target = target
        self.token = token
        self.evidence_path = evidence_path
        self.client = client or httpx.Client()
        self._owns_client = client is None
        self._lock = threading.Lock()
        self._attempts: dict[tuple[str, str], int] = {}
        self._request_counter = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        if self._server is not None:
            raise ProviderProxyError("provider evidence proxy is already running")
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                proxy._handle(self)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_path.touch(exist_ok=True)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(
            target=server.serve_forever,
            name="harbor-hf-provider-proxy",
            daemon=True,
        )
        self._server = server
        self._thread = thread
        thread.start()
        host, port = server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        server = self._server
        thread = self._thread
        if server is None or thread is None:
            return
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        self._server = None
        self._thread = None
        if self._owns_client:
            self.client.close()

    @staticmethod
    def scoped_base_url(base_url: str, scope: str) -> str:
        if not _REQUEST_SCOPE.fullmatch(scope):
            raise ValueError("provider request scope is invalid")
        return f"{base_url.rstrip('/')}/scopes/{scope}"

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        scope = _request_scope(handler.path)
        if scope is None:
            self._send_json(handler, 404, {"error": "unsupported provider route"})
            return
        try:
            payload = self._read_request(handler)
            request, attempt = self._request(payload, scope=scope)
            forwarded = _forwarded_payload(self.target, payload)
        except (ProviderProxyError, ValidationError, ValueError) as error:
            self._send_json(handler, 400, {"error": str(error)})
            return
        observed = self._forward(handler, forwarded)
        result = observe_provider_response(
            self.target,
            request,
            attempt=attempt,
            status_code=observed.status_code,
            headers=observed.headers,
            content=observed.content,
            total_ms=observed.total_ms,
            time_to_first_token_ms=observed.semantic_output_ms,
        )
        self._record(result.model_dump(mode="json", exclude={"message"}))

    @staticmethod
    def _read_request(handler: BaseHTTPRequestHandler) -> dict[str, JsonValue]:
        try:
            content_length = int(handler.headers.get("Content-Length", ""))
        except ValueError as error:
            raise ProviderProxyError("invalid request size") from error
        if content_length < 0 or content_length > _MAX_REQUEST_BYTES:
            raise ProviderProxyError("invalid request size")
        return _json_object(handler.rfile.read(content_length))

    def _forward(
        self,
        handler: BaseHTTPRequestHandler,
        forwarded: dict[str, JsonValue],
    ) -> _ObservedResponse:
        started = perf_counter()
        semantic_output_ms: float | None = None
        captured = bytearray()
        status_code = 502
        response_headers = httpx.Headers()
        response_started = False
        try:
            with self.client.stream(
                "POST",
                f"{HF_INFERENCE_PROVIDER_BASE_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.token}"},
                json=forwarded,
                timeout=self.target.timeout_seconds,
            ) as response:
                status_code = response.status_code
                response_headers = response.headers
                response_started = True
                captured, semantic_output_ms = _relay_response(
                    handler, response, started
                )
        except httpx.TimeoutException:
            if not response_started:
                self._send_json(handler, 504, {"error": "provider request timed out"})
            status_code = 504
        except httpx.HTTPError:
            if not response_started:
                self._send_json(handler, 502, {"error": "provider transport failed"})
            status_code = 502
        total_ms = round((perf_counter() - started) * 1000, 3)
        return _ObservedResponse(
            status_code=status_code,
            headers=response_headers,
            content=bytes(captured),
            total_ms=total_ms,
            semantic_output_ms=semantic_output_ms,
        )

    def _request(
        self, payload: dict[str, JsonValue], *, scope: str
    ) -> tuple[ProviderChatRequest, int]:
        request_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with self._lock:
            self._request_counter += 1
            request = _provider_request(payload, f"provider-{self._request_counter}")
            attempt_key = (scope, request_digest)
            previous_attempts = self._attempts.get(attempt_key, 0)
            if previous_attempts >= self.target.limits.max_attempts:
                raise ProviderProxyError("provider request attempt budget is exhausted")
            attempt = previous_attempts + 1
            self._attempts[attempt_key] = attempt
        return request, attempt

    def _record(self, value: dict[str, object]) -> None:
        line = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        with self._lock, self.evidence_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    @staticmethod
    def _send_json(
        handler: BaseHTTPRequestHandler, status: int, value: dict[str, str]
    ) -> None:
        if handler.wfile.closed:
            return
        content = json.dumps(value, sort_keys=True).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(content)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(content)
        handler.close_connection = True


def _json_object(content: bytes) -> dict[str, JsonValue]:
    value = json.loads(content)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProviderProxyError("provider request must be a JSON object")
    return cast(dict[str, JsonValue], value)


def _request_scope(path: str) -> str | None:
    matched = _SCOPED_COMPLETIONS.fullmatch(path)
    return matched.group("scope") if matched is not None else None


def _relay_response(
    handler: BaseHTTPRequestHandler,
    response: httpx.Response,
    started: float,
) -> tuple[bytearray, float | None]:
    handler.send_response(response.status_code)
    for name, value in response.headers.items():
        if name.lower() in _FORWARDED_RESPONSE_HEADERS:
            handler.send_header(name, value)
    handler.send_header("Connection", "close")
    handler.end_headers()
    captured = bytearray()
    semantic_output_ms: float | None = None
    probe = _SseSemanticOutputProbe()
    for chunk in response.iter_raw():
        if semantic_output_ms is None and probe.feed(chunk):
            semantic_output_ms = (perf_counter() - started) * 1000
        if len(captured) + len(chunk) <= _MAX_EVIDENCE_RESPONSE_BYTES:
            captured.extend(chunk)
        handler.wfile.write(chunk)
        handler.wfile.flush()
    if semantic_output_ms is None and probe.finish():
        semantic_output_ms = (perf_counter() - started) * 1000
    handler.close_connection = True
    return captured, semantic_output_ms


class _SseSemanticOutputProbe:
    """Observe semantic OpenAI deltas without changing relayed stream bytes."""

    def __init__(self) -> None:
        self._pending = bytearray()
        self._stopped = False

    def feed(self, chunk: bytes) -> bool:
        if self._stopped or not chunk:
            return False
        self._pending.extend(chunk)
        if len(self._pending) > _MAX_EVIDENCE_RESPONSE_BYTES:
            self._pending.clear()
            self._stopped = True
            return False
        while (newline := self._pending.find(b"\n")) >= 0:
            line = bytes(self._pending[:newline]).removesuffix(b"\r")
            del self._pending[: newline + 1]
            if _sse_line_has_semantic_output(line):
                self._stopped = True
                return True
        return False

    def finish(self) -> bool:
        if self._stopped or not self._pending:
            return False
        line = bytes(self._pending).removesuffix(b"\r")
        self._pending.clear()
        return _sse_line_has_semantic_output(line)


def _sse_line_has_semantic_output(line: bytes) -> bool:
    if not line.startswith(b"data:"):
        return False
    data = line.removeprefix(b"data:").strip()
    if not data or data == b"[DONE]":
        return False
    try:
        payload = _json_object(data)
    except (UnicodeDecodeError, json.JSONDecodeError, ProviderProxyError):
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False
    return any(_choice_has_semantic_output(choice) for choice in choices)


def _choice_has_semantic_output(choice: JsonValue) -> bool:
    if not isinstance(choice, dict):
        return False
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return False
    text_values = (
        delta.get("content"),
        delta.get("reasoning_content"),
        delta.get("reasoning"),
    )
    if any(isinstance(value, str) and bool(value) for value in text_values):
        return True
    tool_calls = delta.get("tool_calls")
    return isinstance(tool_calls, list) and bool(tool_calls)


def _provider_request(
    payload: dict[str, JsonValue], request_id: str
) -> ProviderChatRequest:
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ProviderProxyError("provider request messages must be a list")
    messages = [_provider_message(value) for value in raw_messages]
    raw_tools = payload.get("tools", [])
    if not isinstance(raw_tools, list):
        raise ProviderProxyError("provider request tools must be a list")
    tools = [ProviderTool.model_validate(value) for value in raw_tools]
    parameters = {
        key: value
        for key, value in payload.items()
        if key not in {"messages", "model", "stream", "stream_options", "tools"}
    }
    return ProviderChatRequest(
        request_id=request_id,
        messages=messages,
        tools=tools,
        parameters=parameters,
        stream=payload.get("stream") is True,
    )


def _provider_message(value: JsonValue) -> ProviderMessage:
    if not isinstance(value, Mapping):
        raise ProviderProxyError("provider message must be an object")
    payload = dict(value)
    raw_calls = payload.pop("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise ProviderProxyError("provider tool calls must be a list")
    calls: list[ProviderToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            raise ProviderProxyError("provider tool call must be an object")
        function = raw.get("function")
        if not isinstance(function, Mapping):
            raise ProviderProxyError("provider tool call function must be an object")
        calls.append(
            ProviderToolCall(
                id=str(raw.get("id", "")),
                function_name=str(function.get("name", "")),
                arguments=str(function.get("arguments", "")),
            )
        )
    payload["tool_calls"] = calls
    return ProviderMessage.model_validate(payload)


def _forwarded_payload(
    target: ProviderTarget, payload: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    forwarded = dict(target.parameters)
    forwarded.update(payload)
    forwarded["model"] = routed_provider_model(target)
    if forwarded.get("stream") is True:
        forwarded["stream_options"] = {"include_usage": True}
    return forwarded
