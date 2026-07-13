from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from harbor_hf.evidence import is_sensitive_key

ProviderProfileId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")]

EvidenceStatus = Literal[
    "observed",
    "not_observed",
    "not_reported",
    "not_applicable",
    "malformed",
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceValue[T](FrozenModel):
    status: EvidenceStatus
    value: T | None = None
    detail: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def status_matches_value(self) -> EvidenceValue[T]:
        if self.status == "observed" and self.value is None:
            raise ValueError("observed evidence requires a value")
        if self.status != "observed" and self.value is not None:
            raise ValueError("unobserved evidence must not contain a value")
        if self.status == "malformed" and self.detail is None:
            raise ValueError("malformed evidence requires a detail")
        if self.status != "malformed" and self.detail is not None:
            raise ValueError("only malformed evidence may contain a detail")
        return self


def observed[T](value: T) -> EvidenceValue[T]:
    return EvidenceValue[T](status="observed", value=value)


def unavailable[T](
    status: Literal["not_observed", "not_reported", "not_applicable"],
) -> EvidenceValue[T]:
    return EvidenceValue[T](status=status)


def malformed[T](detail: str) -> EvidenceValue[T]:
    return EvidenceValue[T](status="malformed", detail=detail)


class PolicyRoute(FrozenModel):
    kind: Literal["policy"] = "policy"
    policy: Literal["fastest", "cheapest", "preferred"] = "fastest"


class ExplicitProviderRoute(FrozenModel):
    kind: Literal["provider"] = "provider"
    provider: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")


ProviderRoute = Annotated[
    PolicyRoute | ExplicitProviderRoute,
    Field(discriminator="kind"),
]


class ProviderLimits(FrozenModel):
    max_concurrent_requests: int = Field(default=1, ge=1)
    max_attempts: int = Field(default=1, ge=1)
    max_spend_usd: Decimal | None = Field(default=None, gt=0)


class ProviderTarget(FrozenModel):
    id: ProviderProfileId
    kind: Literal["inference-provider"] = "inference-provider"
    service: Literal["hf-inference-providers"] = "hf-inference-providers"
    model: str = Field(min_length=1, pattern=r"^[^\s:]+/[^\s:]+$")
    routing: ProviderRoute = Field(default_factory=PolicyRoute)
    timeout_seconds: float = Field(default=60, gt=0, le=3600)
    token_secret_name: Literal["HF_TOKEN"] = "HF_TOKEN"
    limits: ProviderLimits = Field(default_factory=ProviderLimits)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def parameters_are_safe(self) -> ProviderTarget:
        _validate_parameters(self.parameters, "provider target")
        return self


class ProviderToolFunction(FrozenModel):
    name: str = Field(min_length=1)
    description: str | None = Field(default=None, min_length=1)
    parameters: dict[str, JsonValue]


class ProviderTool(FrozenModel):
    type: Literal["function"] = "function"
    function: ProviderToolFunction


class ProviderToolCall(FrozenModel):
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function_name: str = Field(min_length=1)
    arguments: str


class ProviderMessage(FrozenModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = Field(default=None, min_length=1)
    tool_calls: list[ProviderToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def content_matches_role(self) -> ProviderMessage:
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("only tool messages may set tool_call_id")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may contain tool calls")
        if self.content is None and not self.tool_calls:
            raise ValueError("messages require content or tool calls")
        return self


class ProviderChatRequest(FrozenModel):
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    messages: list[ProviderMessage] = Field(min_length=1)
    tools: list[ProviderTool] = Field(default_factory=list)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    stream: bool = False

    @model_validator(mode="after")
    def parameters_are_safe(self) -> ProviderChatRequest:
        reserved = {"messages", "model", "stream", "stream_options", "tools"}
        overlap = reserved.intersection(self.parameters)
        if overlap:
            raise ValueError(
                "provider request parameters contain reserved keys: "
                + ", ".join(sorted(overlap))
            )
        _validate_parameters(self.parameters, "provider request")
        return self


class ProviderModelEvidence(FrozenModel):
    requested: str
    routed: str
    response: EvidenceValue[str]


class ProviderRoutingEvidence(FrozenModel):
    requested_kind: Literal["policy", "provider"]
    requested_value: str
    selected_provider: EvidenceValue[str]


class ProviderQuotaEvidence(FrozenModel):
    request_limit: EvidenceValue[int]
    requests_remaining: EvidenceValue[int]
    token_limit: EvidenceValue[int]
    tokens_remaining: EvidenceValue[int]
    reset: EvidenceValue[str]


class ProviderRetryEvidence(FrozenModel):
    attempt: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    disposition: Literal["no_retry", "retry", "inspect"]
    retry_after: EvidenceValue[str]

    @model_validator(mode="after")
    def attempt_is_within_budget(self) -> ProviderRetryEvidence:
        if self.attempt > self.max_attempts:
            raise ValueError("provider attempt exceeds the configured retry budget")
        if self.attempt == self.max_attempts and self.disposition == "retry":
            raise ValueError("exhausted provider retry budgets cannot recommend retry")
        return self


class ProviderUsageEvidence(FrozenModel):
    input_tokens: EvidenceValue[int]
    output_tokens: EvidenceValue[int]
    total_tokens: EvidenceValue[int]


class ProviderLatencyEvidence(FrozenModel):
    total_ms: EvidenceValue[float]
    time_to_first_token_ms: EvidenceValue[float]


class ProviderEndpointEvidence(FrozenModel):
    endpoint_name: EvidenceValue[str] = Field(
        default_factory=lambda: unavailable("not_applicable")
    )
    endpoint_status: EvidenceValue[str] = Field(
        default_factory=lambda: unavailable("not_applicable")
    )
    ready_replicas: EvidenceValue[int] = Field(
        default_factory=lambda: unavailable("not_applicable")
    )
    region: EvidenceValue[str] = Field(
        default_factory=lambda: unavailable("not_reported")
    )
    hardware: EvidenceValue[str] = Field(
        default_factory=lambda: unavailable("not_reported")
    )
    engine: EvidenceValue[str] = Field(
        default_factory=lambda: unavailable("not_reported")
    )
    precision: EvidenceValue[str] = Field(
        default_factory=lambda: unavailable("not_reported")
    )


class ProviderRequestEvidence(FrozenModel):
    request_id: str
    provider_request_id: EvidenceValue[str]
    streaming: bool
    message_count: int = Field(ge=1)
    tool_count: int = Field(ge=0)


class ProviderEvidence(FrozenModel):
    schema_version: Literal["harbor-hf/provider-evidence/v1alpha1"] = (
        "harbor-hf/provider-evidence/v1alpha1"
    )
    request: ProviderRequestEvidence
    model: ProviderModelEvidence
    routing: ProviderRoutingEvidence
    quota: ProviderQuotaEvidence
    retry: ProviderRetryEvidence
    usage: ProviderUsageEvidence
    latency: ProviderLatencyEvidence
    endpoint: ProviderEndpointEvidence = Field(default_factory=ProviderEndpointEvidence)


class ProviderCallResult(FrozenModel):
    status: Literal[
        "succeeded",
        "throttled",
        "timed_out",
        "rejected",
        "provider_error",
        "malformed_response",
    ]
    remote_outcome: Literal["completed", "not_completed", "ambiguous"]
    response_id: EvidenceValue[str]
    finish_reason: EvidenceValue[str]
    message: ProviderMessage | None = None
    error_code: str | None = Field(default=None, min_length=1)
    evidence: ProviderEvidence

    @model_validator(mode="after")
    def success_has_message(self) -> ProviderCallResult:
        if self.status == "succeeded" and self.message is None:
            raise ValueError("successful provider calls require a message")
        if self.status != "succeeded" and self.message is not None:
            raise ValueError("failed provider calls must not contain a message")
        if (self.status == "succeeded") == (self.error_code is not None):
            raise ValueError("only failed provider calls require an error code")
        return self


def provider_json_schemas() -> dict[str, dict[str, object]]:
    return {
        "provider_target": ProviderTarget.model_json_schema(),
        "provider_evidence": ProviderEvidence.model_json_schema(),
        "provider_call_result": ProviderCallResult.model_json_schema(),
    }


def _validate_parameters(value: JsonValue, owner: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if is_sensitive_key(key):
                raise ValueError(
                    f"{owner} parameters must not contain secret-like keys"
                )
            _validate_parameters(item, owner)
    elif isinstance(value, list):
        for item in value:
            _validate_parameters(item, owner)
