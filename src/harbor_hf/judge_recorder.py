from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from harbor_hf.models import TrialEvidencePolicy
from harbor_hf.trial_evidence import TrialEvidenceError

JUDGE_RECORDER_PORT = 8001
_ROUTE = "/scopes/{capability}/v1/chat/completions"
_ALLOWED_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "content-encoding",
        "content-length",
        "content-type",
        "user-agent",
    }
)
_HF_JUDGE_URL = "https://router.huggingface.co/v1/chat/completions"
_OPENAI_JUDGE_URL = "https://api.openai.com/v1/chat/completions"
_ALLOWED_JUDGE_PROVIDERS = {
    _HF_JUDGE_URL: "hf-inference-provider",
    _OPENAI_JUDGE_URL: "openai-api",
}
_ALLOWED_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
_ALLOWED_REQUEST_FIELDS = frozenset(
    {"messages", "model", "reasoning_effort", "response_format", "temperature"}
)
_ALLOWED_RESPONSE_HEADERS = frozenset(
    {
        "content-encoding",
        "content-length",
        "content-type",
        "request-id",
        "retry-after",
        "x-amzn-requestid",
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset",
        "x-ratelimit-reset-requests",
        "x-request-id",
    }
)


class _HeaderCollection(Protocol):
    def items(self) -> Iterable[tuple[object, object]]: ...


class JudgeRecorderError(RuntimeError):
    """Raised when an exact judge exchange cannot be retained safely."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BodyReference(FrozenModel):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    media_type: Literal["application/octet-stream"] = "application/octet-stream"

    @field_validator("path")
    @classmethod
    def path_is_one_safe_component(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or len(path.parts) != 1 or value in {"", ".", ".."}:
            raise ValueError("judge body path must be one relative component")
        return value


class JudgeRecorderSummary(FrozenModel):
    schema_version: Literal["harbor-hf/judge-recorder-summary/v1"] = (
        "harbor-hf/judge-recorder-summary/v1"
    )
    execution_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    exchange_count: int = Field(ge=0)
    rejected_call_count: int = Field(ge=0)
    rejected_error_types: list[str] = Field(default_factory=list, max_length=4096)
    closed_at: datetime

    @model_validator(mode="after")
    def close_time_has_timezone(self) -> JudgeRecorderSummary:
        if self.closed_at.tzinfo is None:
            raise ValueError("judge recorder close time must include a timezone")
        return self


JudgeProvider = Literal["hf-inference-provider", "openai-api"]
JudgeUpstreamUrl = Literal[
    "https://router.huggingface.co/v1/chat/completions",
    "https://api.openai.com/v1/chat/completions",
]
JudgeTransformation = Literal["none", "model_enforced", "parameters_enforced"]


def _enforce_request_parameters(
    *,
    received: bytes,
    payload: dict[str, object],
    model: str,
    reasoning_effort: str | None,
    strip_temperature: bool,
) -> tuple[bytes, JudgeTransformation]:
    model_changed = payload.get("model") != model
    reasoning_changed = (
        reasoning_effort is not None
        and payload.get("reasoning_effort") != reasoning_effort
    )
    temperature_changed = strip_temperature and "temperature" in payload
    if model_changed:
        payload["model"] = model
    if reasoning_changed:
        payload["reasoning_effort"] = reasoning_effort
    if temperature_changed:
        del payload["temperature"]
    if not (model_changed or reasoning_changed or temperature_changed):
        return received, "none"
    forwarded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    transformation: JudgeTransformation = (
        "parameters_enforced"
        if reasoning_changed or temperature_changed
        else "model_enforced"
    )
    return forwarded, transformation


class JudgeExchange(FrozenModel):
    schema_version: Literal["harbor-hf/judge-exchange/v1"] = (
        "harbor-hf/judge-exchange/v1"
    )
    exchange_id: str = Field(pattern=r"^judge-[0-9]{4}$")
    execution_id: str
    trial_id: str
    attempt: int = Field(ge=1)
    provider: JudgeProvider = "hf-inference-provider"
    upstream_url: JudgeUpstreamUrl = _HF_JUDGE_URL
    requested_model: str | None
    forwarded_model: str
    transformation: JudgeTransformation
    request_received_headers: dict[str, str]
    request_forwarded_headers: dict[str, str]
    request_received: BodyReference
    request_forwarded: BodyReference
    response_upstream: BodyReference | None
    response_delivered: BodyReference
    upstream_http_status: int | None = Field(default=None, ge=100, le=599)
    delivered_http_status: int = Field(ge=100, le=599)
    upstream_request_id: str | None = None
    upstream_response_headers: dict[str, str]
    delivered_response_headers: dict[str, str]
    started_at: datetime
    finished_at: datetime
    total_ms: float = Field(ge=0)
    outcome: Literal["success", "upstream_error", "transport_error", "recorder_error"]
    error_type: str | None = Field(default=None, max_length=200)
    error_message: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def provider_is_consistent(self) -> JudgeExchange:
        if _ALLOWED_JUDGE_PROVIDERS[self.upstream_url] != self.provider:
            raise ValueError("judge provider disagrees with upstream URL")
        return self

    @model_validator(mode="after")
    def identity_and_time_are_consistent(self) -> JudgeExchange:
        if int(self.exchange_id.removeprefix("judge-")) != self.attempt:
            raise ValueError("judge exchange ID disagrees with attempt")
        if self.finished_at < self.started_at:
            raise ValueError("judge exchange finished before it started")
        if (self.response_upstream is None) != (self.upstream_http_status is None):
            raise ValueError(
                "judge response evidence disagrees with upstream HTTP status"
            )
        if self.outcome in {"success", "upstream_error"}:
            if self.response_upstream is None or self.error_type is not None:
                raise ValueError("judge response outcome has inconsistent evidence")
        elif self.error_type is None:
            raise ValueError("judge recorder failure has no error evidence")
        if self.outcome == "transport_error" and self.response_upstream is not None:
            raise ValueError("judge transport failure cannot have an upstream response")
        return self

    @field_validator("request_received_headers", "request_forwarded_headers")
    @classmethod
    def request_headers_are_allowlisted(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            name != name.lower() or name not in _ALLOWED_REQUEST_HEADERS
            for name in value
        ):
            raise ValueError("judge request evidence contains a forbidden header")
        return value

    @field_validator("upstream_response_headers")
    @classmethod
    def upstream_headers_are_allowlisted(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            name != name.lower() or name not in _ALLOWED_RESPONSE_HEADERS
            for name in value
        ):
            raise ValueError("judge upstream evidence contains a forbidden header")
        return value

    @field_validator("delivered_response_headers")
    @classmethod
    def delivered_headers_are_allowlisted(cls, value: dict[str, str]) -> dict[str, str]:
        allowed = _ALLOWED_RESPONSE_HEADERS | {"x-harbor-judge-exchange-id"}
        if any(name != name.lower() or name not in allowed for name in value):
            raise ValueError("judge delivered evidence contains a forbidden header")
        return value


@dataclass(frozen=True)
class _Scope:
    execution_id: str
    trial_id: str
    model: str
    destination: Path
    policy: TrialEvidencePolicy
    max_calls: int
    secrets: tuple[str, ...]


class JudgeEvidenceRecorder:
    """A capability-scoped judge gateway that stores exact bodies without secrets."""

    def __init__(
        self,
        *,
        token: str,
        upstream_url: JudgeUpstreamUrl = _HF_JUDGE_URL,
        reasoning_effort: str | None = None,
        strip_temperature: bool = False,
        client: httpx.Client | None = None,
        capability_factory: Callable[[], str] | None = None,
        deadline: float | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not token:
            raise ValueError("judge recorder token must not be empty")
        if upstream_url not in _ALLOWED_JUDGE_PROVIDERS:
            raise ValueError("judge recorder upstream URL is not allowed")
        if reasoning_effort not in {None, *_ALLOWED_REASONING_EFFORTS}:
            raise ValueError("judge recorder reasoning effort is not allowed")
        self._token = token
        self._upstream_url = upstream_url
        self._provider: JudgeProvider = (
            "openai-api"
            if upstream_url == _OPENAI_JUDGE_URL
            else "hf-inference-provider"
        )
        self._reasoning_effort = reasoning_effort
        self._strip_temperature = strip_temperature
        self._client = client or httpx.Client(headers={"Accept-Encoding": "identity"})
        self._owns_client = client is None
        self._capability_factory = capability_factory or (
            lambda: secrets.token_urlsafe(32)
        )
        self._deadline = deadline
        self._monotonic = monotonic
        self._scopes: dict[str, _Scope] = {}
        self._counts: dict[str, int] = {}
        self._rejections: dict[str, int] = {}
        self._rejection_errors: dict[str, list[str]] = {}
        self._active_calls: dict[str, int] = {}
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, *, host: str = "0.0.0.0", port: int = JUDGE_RECORDER_PORT) -> str:
        if self._server is not None:
            raise JudgeRecorderError("judge recorder is already running")
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                if self.path.rstrip("/") == "/healthz":
                    recorder._send(
                        self,
                        200,
                        b'{"status":"ok"}',
                        {"content-type": "application/json"},
                    )
                else:
                    recorder._send_error(self, 404, "unsupported judge route")

            def do_POST(self) -> None:
                recorder._handle(self)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="harbor-hf-judge-recorder",
        )
        self._thread.start()
        bound_host, bound_port = self._server.server_address[:2]
        return f"http://{bound_host}:{bound_port}"

    def close(self) -> None:
        if self._server is not None and self._thread is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
        with self._lock:
            self._scopes.clear()
            self._counts.clear()
            self._rejections.clear()
            self._rejection_errors.clear()
            self._active_calls.clear()
        if self._owns_client:
            self._client.close()

    def register_scope(
        self,
        *,
        execution_id: str,
        trial_id: str,
        model: str,
        destination: Path,
        policy: TrialEvidencePolicy,
        known_secrets: tuple[str, ...] = (),
        max_calls: int | None = None,
    ) -> str:
        if not execution_id or not trial_id or not model:
            raise ValueError(
                "judge scope requires execution, trial, and model identity"
            )
        if destination.exists():
            raise JudgeRecorderError("judge evidence destination already exists")
        effective_max_calls = max_calls or policy.judge_max_calls_per_execution
        if effective_max_calls < 1 or effective_max_calls > 4096:
            raise JudgeRecorderError("judge scope call limit is invalid")
        destination.mkdir(parents=True)
        for _ in range(8):
            capability = self._capability_factory()
            if len(capability) < 22 or not all(
                character.isalnum() or character in "_-" for character in capability
            ):
                raise JudgeRecorderError("judge capability is invalid")
            with self._lock:
                if capability not in self._scopes:
                    self._scopes[capability] = _Scope(
                        execution_id=execution_id,
                        trial_id=trial_id,
                        model=model,
                        destination=destination,
                        policy=policy,
                        max_calls=effective_max_calls,
                        secrets=tuple(
                            dict.fromkeys((self._token, *known_secrets, capability))
                        ),
                    )
                    self._counts[execution_id] = 0
                    self._rejections[execution_id] = 0
                    self._rejection_errors[execution_id] = []
                    self._active_calls[execution_id] = 0
                    return capability
        raise JudgeRecorderError("judge capability generation collided")

    def revoke_scope(self, capability: str) -> None:
        with self._idle:
            scope = self._scopes.pop(capability, None)
            if scope is None:
                return
            deadline = self._monotonic() + scope.policy.judge_timeout_seconds + 5
            while self._active_calls.get(scope.execution_id, 0):
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    raise JudgeRecorderError(
                        "judge scope remained active after deadline"
                    )
                self._idle.wait(timeout=remaining)
            self._active_calls.pop(scope.execution_id, None)
            count = self._counts.pop(scope.execution_id, 0)
            rejected = self._rejections.pop(scope.execution_id, 0)
            rejected_errors = self._rejection_errors.pop(scope.execution_id, [])
        summary = JudgeRecorderSummary(
            execution_id=scope.execution_id,
            trial_id=scope.trial_id,
            model=scope.model,
            exchange_count=count,
            rejected_call_count=rejected,
            rejected_error_types=rejected_errors,
            closed_at=datetime.now(UTC),
        )
        try:
            scope.destination.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                scope.destination / "recorder.json", summary.model_dump(mode="json")
            )
        except OSError as error:
            raise JudgeRecorderError(
                "judge recorder summary could not be written"
            ) from error

    @staticmethod
    def scoped_url(base_url: str, capability: str) -> str:
        return base_url.rstrip("/") + _ROUTE.format(capability=capability)

    @staticmethod
    def capability_digest(capability: str) -> str:
        return "sha256:" + hashlib.sha256(capability.encode()).hexdigest()

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        capability = _capability(handler.path)
        with self._lock:
            scope = self._scopes.get(capability or "")
        if scope is None:
            self._send_error(handler, 404, "unsupported judge route")
            return
        self._begin_call(scope)
        started_at = datetime.now(UTC)
        started = perf_counter()
        exchange_id: str | None = None
        attempt: int | None = None
        received = b""
        forwarded = b""
        requested_model: str | None = None
        transformation: JudgeTransformation = "none"
        response: httpx.Response | None = None
        request_received_headers = _allow_headers(
            handler.headers, _ALLOWED_REQUEST_HEADERS
        )
        try:
            length = int(handler.headers.get("Content-Length", ""))
            if length < 0 or length > scope.policy.judge_max_request_bytes:
                raise JudgeRecorderError("judge request size is invalid")
            received = handler.rfile.read(length)
            _assert_secret_absent(received, scope.secrets)
            payload = json.loads(received)
            if not isinstance(payload, dict):
                raise JudgeRecorderError("judge request must be a JSON object")
            if payload.get("stream") is True:
                raise JudgeRecorderError("streaming judge requests are forbidden")
            if not set(payload).issubset(_ALLOWED_REQUEST_FIELDS):
                raise JudgeRecorderError("judge request contains unsupported fields")
            requested_model = (
                payload.get("model") if isinstance(payload.get("model"), str) else None
            )
            forwarded, transformation = _enforce_request_parameters(
                received=received,
                payload=payload,
                model=scope.model,
                reasoning_effort=self._reasoning_effort,
                strip_temperature=self._strip_temperature,
            )
            _assert_secret_absent(forwarded, scope.secrets)
            exchange_id, attempt = self._allocate(scope)
            response = self._upstream(forwarded, scope)
            decoded = _decoded_for_scan(
                response.headers,
                response.content,
                scope.policy.judge_max_response_bytes,
            )
            _assert_secret_absent(response.content, scope.secrets)
            _assert_secret_absent(decoded, scope.secrets)
            delivered_headers = _allow_headers(
                response.headers, _ALLOWED_RESPONSE_HEADERS
            )
            delivered_headers["x-harbor-judge-exchange-id"] = exchange_id
            outcome: Literal[
                "success", "upstream_error", "transport_error", "recorder_error"
            ] = "success" if 200 <= response.status_code < 300 else "upstream_error"
            self._write_exchange(
                scope=scope,
                exchange_id=exchange_id,
                attempt=attempt,
                requested_model=requested_model,
                transformation=transformation,
                received=received,
                forwarded=forwarded,
                request_received_headers=request_received_headers,
                response=response,
                delivered_body=response.content,
                delivered_status=response.status_code,
                delivered_headers=delivered_headers,
                started_at=started_at,
                total_ms=round((perf_counter() - started) * 1000, 3),
                outcome=outcome,
            )
            self._send(
                handler, response.status_code, response.content, delivered_headers
            )
        except (
            JudgeRecorderError,
            TrialEvidenceError,
            json.JSONDecodeError,
            ValueError,
            httpx.HTTPError,
            OSError,
        ) as error:
            self._handle_failure(
                handler=handler,
                scope=scope,
                exchange_id=exchange_id,
                attempt=attempt,
                requested_model=requested_model,
                transformation=transformation,
                received=received,
                forwarded=forwarded,
                request_received_headers=request_received_headers,
                response=response,
                started_at=started_at,
                started=started,
                error=error,
            )
        finally:
            self._finish_call(scope)

    def _handle_failure(
        self,
        *,
        handler: BaseHTTPRequestHandler,
        scope: _Scope,
        exchange_id: str | None,
        attempt: int | None,
        requested_model: str | None,
        transformation: JudgeTransformation,
        received: bytes,
        forwarded: bytes,
        request_received_headers: dict[str, str],
        response: httpx.Response | None,
        started_at: datetime,
        started: float,
        error: Exception,
    ) -> None:
        if exchange_id is None or attempt is None:
            self._record_rejection(scope, error)
            self._send_error(handler, 502, "judge recorder rejected request")
            return
        error_body = json.dumps(
            {"error": "judge exchange failed"}, sort_keys=True
        ).encode()
        delivered_headers = {
            "content-type": "application/json",
            "x-harbor-judge-exchange-id": exchange_id,
        }
        outcome: Literal["transport_error", "recorder_error"] = (
            "transport_error"
            if isinstance(error, httpx.HTTPError)
            else "recorder_error"
        )
        try:
            self._write_exchange(
                scope=scope,
                exchange_id=exchange_id,
                attempt=attempt,
                requested_model=requested_model,
                transformation=transformation,
                received=received,
                forwarded=forwarded,
                request_received_headers=request_received_headers,
                response=response,
                delivered_body=error_body,
                delivered_status=502,
                delivered_headers=delivered_headers,
                started_at=started_at,
                total_ms=round((perf_counter() - started) * 1000, 3),
                outcome=outcome,
                error_type=type(error).__name__,
                error_message=_safe_error_message(error),
            )
        except (JudgeRecorderError, TrialEvidenceError, ValueError, OSError):
            self._send_error(handler, 502, "judge evidence recording failed")
            return
        self._send(handler, 502, error_body, delivered_headers)

    def _begin_call(self, scope: _Scope) -> None:
        with self._idle:
            self._active_calls[scope.execution_id] = (
                self._active_calls.get(scope.execution_id, 0) + 1
            )

    def _finish_call(self, scope: _Scope) -> None:
        with self._idle:
            active = self._active_calls.get(scope.execution_id, 0)
            self._active_calls[scope.execution_id] = max(0, active - 1)
            self._idle.notify_all()

    def _record_rejection(self, scope: _Scope, error: Exception) -> None:
        with self._lock:
            self._rejections[scope.execution_id] = (
                self._rejections.get(scope.execution_id, 0) + 1
            )
            self._rejection_errors.setdefault(scope.execution_id, []).append(
                type(error).__name__[:200]
            )

    def _allocate(self, scope: _Scope) -> tuple[str, int]:
        with self._lock:
            attempt = self._counts.get(scope.execution_id, 0) + 1
            if attempt > scope.max_calls:
                raise JudgeRecorderError("judge call limit is exhausted")
            self._counts[scope.execution_id] = attempt
        return f"judge-{attempt:04d}", attempt

    def _upstream(self, body: bytes, scope: _Scope) -> httpx.Response:
        started = self._monotonic()
        remaining = float(scope.policy.judge_timeout_seconds)
        if self._deadline is not None:
            remaining = min(remaining, self._deadline - started)
        request_deadline = started + remaining
        if remaining <= 0:
            raise JudgeRecorderError("judge request deadline expired")
        request = self._client.build_request(
            "POST",
            self._upstream_url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept-Encoding": "identity",
            },
            content=body,
            timeout=httpx.Timeout(remaining),
        )
        response = self._client.send(request, stream=True)
        content = _read_bounded_response(
            response,
            scope.policy.judge_max_response_bytes,
            deadline=request_deadline,
            monotonic=self._monotonic,
        )
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=content,
            request=request,
        )

    def _write_exchange(
        self,
        *,
        scope: _Scope,
        exchange_id: str,
        attempt: int,
        requested_model: str | None,
        transformation: JudgeTransformation,
        received: bytes,
        forwarded: bytes,
        request_received_headers: dict[str, str],
        response: httpx.Response | None,
        delivered_body: bytes,
        delivered_status: int,
        delivered_headers: dict[str, str],
        started_at: datetime,
        total_ms: float,
        outcome: Literal[
            "success", "upstream_error", "transport_error", "recorder_error"
        ],
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        parent = scope.destination
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f".{exchange_id}.tmp"
        final = parent / exchange_id
        if temporary.exists() or final.exists():
            raise JudgeRecorderError("judge exchange destination already exists")
        temporary.mkdir()
        try:
            paths = {
                "request-received.bin": received,
                "request-forwarded.bin": forwarded,
                "response-delivered.bin": delivered_body,
            }
            if response is not None:
                paths["response-upstream.bin"] = response.content
            references: dict[str, BodyReference] = {}
            for name, content in paths.items():
                path = temporary / name
                path.write_bytes(content)
                references[name] = _body_reference(path)
            upstream_headers = (
                _allow_headers(response.headers, _ALLOWED_RESPONSE_HEADERS)
                if response is not None
                else {}
            )
            request_forwarded_headers = {
                "accept-encoding": "identity",
                "content-length": str(len(forwarded)),
                "content-type": "application/json",
            }
            exchange = JudgeExchange(
                exchange_id=exchange_id,
                execution_id=scope.execution_id,
                trial_id=scope.trial_id,
                attempt=attempt,
                provider=self._provider,
                upstream_url=self._upstream_url,
                requested_model=requested_model,
                forwarded_model=scope.model,
                transformation=transformation,
                request_received_headers=request_received_headers,
                request_forwarded_headers=request_forwarded_headers,
                request_received=references["request-received.bin"],
                request_forwarded=references["request-forwarded.bin"],
                response_upstream=references.get("response-upstream.bin"),
                response_delivered=references["response-delivered.bin"],
                upstream_http_status=(response.status_code if response else None),
                delivered_http_status=delivered_status,
                upstream_request_id=_request_id(upstream_headers),
                upstream_response_headers=upstream_headers,
                delivered_response_headers=delivered_headers,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                total_ms=total_ms,
                outcome=outcome,
                error_type=error_type,
                error_message=error_message,
            )
            (temporary / "exchange.json").write_text(
                json.dumps(
                    exchange.model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
            for path in temporary.iterdir():
                _assert_secret_absent(path.read_bytes(), scope.secrets)
            os.replace(temporary, final)
        except Exception:
            for path in temporary.glob("*"):
                path.unlink(missing_ok=True)
            temporary.rmdir()
            raise

    @staticmethod
    def _send_error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
        body = json.dumps({"error": message[:400]}, sort_keys=True).encode()
        JudgeEvidenceRecorder._send(
            handler, status, body, {"content-type": "application/json"}
        )

    @staticmethod
    def _send(
        handler: BaseHTTPRequestHandler,
        status: int,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        if handler.wfile.closed:
            return
        handler.send_response(status)
        for name, value in headers.items():
            if name.lower() not in {
                "connection",
                "content-length",
                "transfer-encoding",
            }:
                handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(body)
        handler.close_connection = True


def verify_judge_recorder_summary(path: Path) -> JudgeRecorderSummary:
    try:
        return JudgeRecorderSummary.model_validate_json(path.read_text())
    except (OSError, ValueError) as error:
        raise JudgeRecorderError("judge recorder summary is invalid") from error


def verify_judge_exchange(exchange_dir: Path) -> JudgeExchange:
    metadata_path = exchange_dir / "exchange.json"
    try:
        exchange = JudgeExchange.model_validate_json(metadata_path.read_text())
    except (OSError, ValueError) as error:
        raise JudgeRecorderError("judge exchange metadata is invalid") from error
    if exchange.exchange_id != exchange_dir.name:
        raise JudgeRecorderError("judge exchange directory identity mismatch")
    references = [
        exchange.request_received,
        exchange.request_forwarded,
        exchange.response_delivered,
    ]
    if exchange.response_upstream is not None:
        references.append(exchange.response_upstream)
    expected = {"exchange.json", *(reference.path for reference in references)}
    if _judge_exchange_entry_names(exchange_dir) != expected:
        raise JudgeRecorderError("judge exchange file set is incomplete")
    for reference in references:
        path = exchange_dir / reference.path
        if path.is_symlink() or not path.is_file():
            raise JudgeRecorderError("judge exchange body is missing")
        content = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        if len(content) != reference.size_bytes or digest != reference.sha256:
            raise JudgeRecorderError("judge exchange body digest mismatch")
    return exchange


def write_judge_exchange_schema(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(JudgeExchange.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_bounded_response(
    response: httpx.Response,
    limit: int,
    *,
    deadline: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> bytes:
    try:
        _check_response_deadline(deadline, monotonic)
        if response.is_stream_consumed:
            if len(response.content) > limit:
                raise JudgeRecorderError("judge response exceeds configured byte limit")
            return response.content
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_raw():
            _check_response_deadline(deadline, monotonic)
            total += len(chunk)
            if total > limit:
                raise JudgeRecorderError("judge response exceeds configured byte limit")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        response.close()


def _check_response_deadline(
    deadline: float | None, monotonic: Callable[[], float]
) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise JudgeRecorderError("judge request deadline expired")


def _write_json_atomic(path: Path, value: object) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, default=str).encode() + b"\n"
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, httpx.HTTPError):
        return "upstream judge transport failed"
    return "judge evidence recorder failed"


def _judge_exchange_entry_names(exchange_dir: Path) -> set[str]:
    try:
        entries = list(exchange_dir.iterdir())
    except OSError as error:
        raise JudgeRecorderError("judge exchange directory is unreadable") from error
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise JudgeRecorderError("judge exchange contains an unsupported entry")
    return {path.name for path in entries}


def _capability(path: str) -> str | None:
    parts = path.rstrip("/").split("/")
    if (
        len(parts) != 6
        or parts[1] != "scopes"
        or parts[3:] != ["v1", "chat", "completions"]
    ):
        return None
    capability = parts[2]
    return capability if len(capability) >= 22 else None


def _allow_headers(
    headers: _HeaderCollection, allowed: frozenset[str]
) -> dict[str, str]:
    items = headers.items()
    return {
        str(name).lower(): str(value)
        for name, value in items
        if str(name).lower() in allowed
    }


def _decoded_for_scan(headers: httpx.Headers, content: bytes, limit: int) -> bytes:
    encoding = headers.get("content-encoding", "").lower().strip()
    if not encoding:
        return content
    if encoding == "gzip":
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif encoding == "deflate":
        decompressor = zlib.decompressobj()
    else:
        raise JudgeRecorderError("unsupported judge response content encoding")
    try:
        decoded = decompressor.decompress(content, limit + 1)
        if len(decoded) > limit or decompressor.unconsumed_tail:
            raise JudgeRecorderError(
                "decoded judge response exceeds configured byte limit"
            )
        tail = decompressor.flush(limit - len(decoded) + 1)
    except zlib.error as error:
        raise JudgeRecorderError(
            "judge response content encoding is invalid"
        ) from error
    decoded += tail
    if len(decoded) > limit or not decompressor.eof or decompressor.unused_data:
        raise JudgeRecorderError("decoded judge response exceeds configured byte limit")
    return decoded


def _assert_secret_absent(content: bytes, known_secrets: tuple[str, ...]) -> None:
    for secret in known_secrets:
        if secret and secret.encode() in content:
            raise TrialEvidenceError("known secret detected in judge evidence")


def _body_reference(path: Path) -> BodyReference:
    content = path.read_bytes()
    return BodyReference(
        path=path.name,
        size_bytes=len(content),
        sha256="sha256:" + hashlib.sha256(content).hexdigest(),
    )


def _request_id(headers: dict[str, str]) -> str | None:
    for name in ("x-request-id", "request-id", "x-amzn-requestid"):
        if value := headers.get(name):
            return value
    return None
