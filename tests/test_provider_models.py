from decimal import Decimal

import pytest
from pydantic import ValidationError

from harbor_hf.campaigns import (
    ProviderWaveTarget,
    build_campaign_lock,
    build_campaign_plan,
    build_wave_lock,
)
from harbor_hf.control import CampaignSubmittedPayload, new_event
from harbor_hf.models import DeploymentProfile, ExperimentSpec
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
from harbor_hf.reconciler import (
    DeploymentAdmission,
    ReconcileContext,
    plan_reconciliation,
)
from harbor_hf.runs import build_run_lock


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


def test_manifest_and_campaign_lock_provider_admission_separately_from_endpoints(
    remote_spec: ExperimentSpec,
) -> None:
    model = remote_spec.matrix.models[0]
    target = ProviderTarget(
        id="provider-target",
        model=model.repo,
        limits=ProviderLimits(
            max_concurrent_requests=3,
            max_attempts=2,
            max_spend_usd=Decimal("1.25"),
        ),
    )
    spec = ExperimentSpec.model_validate(
        remote_spec.model_copy(
            update={
                "matrix": remote_spec.matrix.model_copy(
                    update={"deployments": [target]}
                )
            }
        ).model_dump(mode="python")
    )

    assert isinstance(remote_spec.matrix.deployments[0], DeploymentProfile)
    assert isinstance(spec.matrix.deployments[0], ProviderTarget)
    with pytest.raises(ValueError, match="require campaign execution"):
        build_run_lock(spec)

    campaign = build_campaign_lock(build_campaign_plan(spec), "provider-campaign")
    run = campaign.runs[0]
    assert run.provider == "hf-inference-providers"
    assert run.max_concurrent_requests == 3
    assert run.spend_cap_microusd == 1_250_000
    submitted = new_event(
        subject_type="campaign",
        subject_id=campaign.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=campaign.plan_digest),
    )

    _projection, blocked = plan_reconciliation(campaign, [submitted])
    assert blocked.actions == []
    assert blocked.blocked[0].reason == "spend-estimate-missing"

    context = ReconcileContext(
        deployments={
            run.deployment_digest: DeploymentAdmission(
                estimated_wave_cost_microusd=500_000
            )
        }
    )
    _projection, admitted = plan_reconciliation(campaign, [submitted], context=context)
    wave = build_wave_lock(campaign, spec, admitted.actions[0])
    assert isinstance(wave.target, ProviderWaveTarget)
    assert wave.target.provider == target
    assert wave.endpoint is None
    assert wave.max_concurrent_shards == 1
    assert wave.spend_cap_microusd == 1_250_000


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
