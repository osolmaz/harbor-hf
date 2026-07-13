from decimal import Decimal

import pytest
from pydantic import ValidationError

from harbor_hf.provider_models import (
    EvidenceValue,
    ExplicitProviderRoute,
    ProviderCallResult,
    ProviderChatRequest,
    ProviderEvidence,
    ProviderLimits,
    ProviderMessage,
    ProviderTarget,
    provider_json_schemas,
)


def test_provider_target_keeps_admission_and_routing_policy() -> None:
    target = ProviderTarget(
        id="provider-target",
        model="openai/gpt-oss-120b",
        routing=ExplicitProviderRoute(provider="groq"),
        limits=ProviderLimits(
            max_concurrent_requests=8,
            max_attempts=3,
            max_spend_usd=Decimal("12.50"),
        ),
    )

    assert target.kind == "inference-provider"
    assert target.service == "hf-inference-providers"
    assert isinstance(target.routing, ExplicitProviderRoute)
    assert target.routing.provider == "groq"
    assert target.limits.max_concurrent_requests == 8
    assert target.limits.max_spend_usd == Decimal("12.50")


@pytest.mark.parametrize("key", ["api_key", "nestedToken", "provider-secret"])
def test_provider_parameters_reject_secret_like_keys(key: str) -> None:
    with pytest.raises(ValidationError, match="secret-like keys"):
        ProviderTarget(
            id="unsafe-provider",
            model="owner/model",
            parameters={"nested": {key: "must-not-be-recorded"}},
        )


def test_provider_request_rejects_transport_owned_parameters() -> None:
    with pytest.raises(ValidationError, match="reserved keys: model"):
        ProviderChatRequest(
            request_id="request-1",
            messages=[ProviderMessage(role="user", content="hello")],
            parameters={"model": "other/model"},
        )


def test_evidence_value_requires_explicit_availability_semantics() -> None:
    with pytest.raises(ValidationError, match="observed evidence requires"):
        EvidenceValue[int](status="observed")
    with pytest.raises(ValidationError, match="unobserved evidence"):
        EvidenceValue[int](status="not_reported", value=1)
    with pytest.raises(ValidationError, match="requires a detail"):
        EvidenceValue[int](status="malformed")


def test_provider_contracts_export_versioned_json_schemas() -> None:
    schemas = provider_json_schemas()

    assert set(schemas) == {
        "provider_target",
        "provider_evidence",
        "provider_call_result",
    }
    assert schemas["provider_target"]["title"] == "ProviderTarget"
    assert schemas["provider_evidence"]["title"] == "ProviderEvidence"
    assert schemas["provider_call_result"]["title"] == "ProviderCallResult"


def test_serialized_evidence_contract_rejects_unknown_fields() -> None:
    schema = ProviderEvidence.model_json_schema()
    result_schema = ProviderCallResult.model_json_schema()

    assert schema["additionalProperties"] is False
    assert result_schema["additionalProperties"] is False
