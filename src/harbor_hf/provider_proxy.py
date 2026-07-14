from __future__ import annotations

import hashlib
import json
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
    first_byte_ms: float | None


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
        self._attempts: dict[str, int] = {}
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

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(handler, 404, {"error": "unsupported provider route"})
            return
        try:
            payload = self._read_request(handler)
            request, attempt = self._request(payload)
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
            time_to_first_token_ms=observed.first_byte_ms,
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
        first_byte_ms: float | None = None
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
                captured, first_byte_ms = _relay_response(handler, response, started)
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
            first_byte_ms=first_byte_ms,
        )

    def _request(
        self, payload: dict[str, JsonValue]
    ) -> tuple[ProviderChatRequest, int]:
        request_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with self._lock:
            self._request_counter += 1
            request = _provider_request(payload, f"provider-{self._request_counter}")
            previous_attempts = self._attempts.get(request_digest, 0)
            if previous_attempts >= self.target.limits.max_attempts:
                raise ProviderProxyError("provider request attempt budget is exhausted")
            attempt = previous_attempts + 1
            self._attempts[request_digest] = attempt
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
    first_byte_ms: float | None = None
    for chunk in response.iter_raw():
        if first_byte_ms is None and chunk:
            first_byte_ms = (perf_counter() - started) * 1000
        if len(captured) + len(chunk) <= _MAX_EVIDENCE_RESPONSE_BYTES:
            captured.extend(chunk)
        handler.wfile.write(chunk)
        handler.wfile.flush()
    handler.close_connection = True
    return captured, first_byte_ms


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
