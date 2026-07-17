from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import cast

import httpx
from huggingface_hub import HfApi, get_token
from pydantic import BaseModel, ConfigDict, Field

from harbor_hf.models import DeploymentProfile, ExperimentSpec
from harbor_hf.profiling import ProfilePlan
from harbor_hf.provider_models import ProviderTarget
from harbor_hf.submission import BucketApi, require_private_bucket

_ENDPOINT_CATALOG = "https://api.endpoints.huggingface.cloud/v2/provider/{namespace}"


class PreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    target_kind: str
    model_revision_verified: bool
    private_bucket_verified: bool
    provider_route_verified: bool
    available_accelerators: int | None = Field(default=None, ge=0)
    required_accelerators: int | None = Field(default=None, ge=1)
    price_per_hour_usd: Decimal | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    spend_cap_usd: Decimal = Field(gt=0)


def preflight_profile_plan(
    plan: ProfilePlan,
    *,
    api: object | None = None,
    client: httpx.Client | None = None,
    token: str | None = None,
) -> PreflightReport:
    resolved_token = token or get_token()
    if not resolved_token:
        raise ValueError("profile preflight requires Hugging Face authentication")
    hub = cast(HfApi, api) if api is not None else HfApi(token=resolved_token)
    info = hub.model_info(plan.model.repo, revision=plan.model.revision)
    if getattr(info, "sha", None) != plan.model.revision:
        raise ValueError("model repository did not resolve to the pinned revision")
    _verify_repository_artifacts(plan, hub)
    require_private_bucket(plan.artifacts.bucket, api=cast(BucketApi, hub))
    cap = Decimal(plan.max_spend_usd)
    if isinstance(plan.deployment, ProviderTarget):
        return _preflight_provider(plan, hub, info, cap)
    return _preflight_endpoint(plan, client, resolved_token, cap)


def _preflight_provider(
    plan: ProfilePlan,
    hub: HfApi,
    model_info: object,
    cap: Decimal,
) -> PreflightReport:
    target = plan.deployment
    assert isinstance(target, ProviderTarget)
    mapping = getattr(model_info, "inference_provider_mapping", None)
    if mapping is None:
        model_info = hub.model_info(
            target.model,
            expand=["inferenceProviderMapping"],
        )
        mapping = getattr(model_info, "inference_provider_mapping", None)
    providers = _live_provider_names(mapping)
    requested = getattr(target.routing, "provider", None)
    if requested is not None and requested not in providers:
        raise ValueError(f"requested Inference Provider is unavailable: {requested}")
    if not providers:
        raise ValueError("model has no live Hugging Face Inference Provider route")
    limits = target.limits
    estimate = (
        Decimal(plan.estimated_profile_cost_usd)
        if plan.estimated_profile_cost_usd is not None
        else None
    )
    target_cap = limits.max_spend_usd
    if estimate is None or target_cap is None:
        raise ValueError(
            "provider profile requires a bounded full-profile cost estimate"
        )
    if estimate > target_cap:
        raise ValueError("full-profile estimate exceeds the provider spend cap")
    if estimate > cap:
        raise ValueError(
            f"profile estimate ${estimate:.2f} exceeds spend cap ${cap:.2f}"
        )
    return PreflightReport(
        profile_id=plan.profile_id,
        target_kind="inference-provider",
        model_revision_verified=True,
        private_bucket_verified=True,
        provider_route_verified=True,
        estimated_cost_usd=estimate,
        spend_cap_usd=cap,
    )


def _preflight_endpoint(
    plan: ProfilePlan,
    client: httpx.Client | None,
    token: str,
    cap: Decimal,
) -> PreflightReport:
    target = plan.deployment
    assert isinstance(target, DeploymentProfile)
    spec = ExperimentSpec.model_validate(plan.experiment)
    namespace = (
        target.endpoint.namespace
        if target.endpoint is not None
        else spec.remote.job.namespace
        if spec.remote is not None
        else None
    )
    if namespace is None:
        raise ValueError("endpoint profile requires a remote namespace")
    owned_client = client is None
    http = client or httpx.Client(timeout=30)
    try:
        response = http.get(
            _ENDPOINT_CATALOG.format(namespace=namespace),
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        compute = _find_compute(response.json(), target)
    finally:
        if owned_client:
            http.close()
    quota = compute.get("quota")
    if not isinstance(quota, dict):
        raise ValueError("endpoint quota is unknown; refusing to guess")
    maximum = quota.get("maxAccelerators")
    used = quota.get("usedAccelerators")
    if not isinstance(maximum, int) or not isinstance(used, int):
        raise ValueError("endpoint quota is unknown; refusing to guess")
    available = max(0, maximum - used)
    maximum_replicas = target.parameters.get("max_replicas", 1)
    if (
        not isinstance(maximum_replicas, int)
        or isinstance(maximum_replicas, bool)
        or maximum_replicas < 1
    ):
        raise ValueError("endpoint max_replicas must be a positive integer")
    required = target.accelerator_count * maximum_replicas
    if available < required:
        raise ValueError(
            f"endpoint quota has {available} accelerators available; "
            f"{required} required"
        )
    price = Decimal(str(compute.get("pricePerHour")))
    duration = plan.profile_timeout_seconds
    estimated = price * Decimal(maximum_replicas) * Decimal(duration) / Decimal(3600)
    if estimated > cap:
        raise ValueError(
            f"profile estimate ${estimated:.2f} exceeds spend cap ${cap:.2f}"
        )
    return PreflightReport(
        profile_id=plan.profile_id,
        target_kind="inference-endpoint",
        model_revision_verified=True,
        private_bucket_verified=True,
        provider_route_verified=True,
        available_accelerators=available,
        required_accelerators=required,
        price_per_hour_usd=price,
        estimated_cost_usd=estimated,
        spend_cap_usd=cap,
    )


def _find_compute(value: object, target: DeploymentProfile) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("endpoint compute catalog is malformed")
    catalog = cast(dict[str, object], value)
    hardware = target.hardware
    requested_vendor, separator, requested_region = target.region.partition("-")
    if not separator:
        raise ValueError("endpoint region must use vendor-region form")
    computes = _region_computes(catalog, requested_vendor, requested_region)
    for compute in computes:
        if _compute_matches(compute, hardware, target.accelerator_count):
            return compute
    raise ValueError(
        "endpoint compute is unavailable: "
        f"{target.region}/{hardware}x{target.accelerator_count}"
    )


def _region_computes(
    catalog: Mapping[str, object], requested_vendor: str, requested_region: str
) -> list[dict[str, object]]:
    for vendor in _dictionary_list(catalog, "vendors"):
        if vendor.get("name") != requested_vendor:
            continue
        for region in _dictionary_list(vendor, "regions"):
            if region.get("name") == requested_region:
                return _dictionary_list(region, "computes")
    return []


def _compute_matches(
    compute: Mapping[str, object], hardware: str, accelerator_count: int
) -> bool:
    instance_type = compute.get("instanceType")
    normalized_hardware = (
        instance_type.removeprefix("nvidia-")
        if isinstance(instance_type, str)
        else None
    )
    return (
        normalized_hardware == hardware
        and compute.get("numAccelerators") == accelerator_count
        and compute.get("status") == "available"
    )


def _verify_repository_artifacts(plan: ProfilePlan, hub: HfApi) -> None:
    target = plan.deployment
    if not isinstance(target, DeploymentProfile):
        return
    repository_prefix = "/repository/"
    referenced = {
        argument.removeprefix(repository_prefix)
        for argument in target.engine.arguments
        if argument.startswith(repository_prefix)
    }
    if not referenced:
        return
    files = set(
        hub.list_repo_files(
            plan.model.repo,
            revision=plan.model.revision,
            repo_type="model",
        )
    )
    missing = referenced - files
    if missing:
        raise ValueError(
            "deployment references missing model artifacts: "
            + ", ".join(sorted(missing))
        )


def _dictionary_list(value: Mapping[str, object], key: str) -> list[dict[str, object]]:
    items = value.get(key, [])
    if not isinstance(items, list):
        raise ValueError(f"endpoint compute catalog has invalid {key}")
    return [cast(dict[str, object], item) for item in items if isinstance(item, dict)]


def _live_provider_names(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {str(name) for name in value}
    if not isinstance(value, list):
        return set()
    providers: set[str] = set()
    for mapping in value:
        provider = getattr(mapping, "provider", None)
        status = getattr(mapping, "status", None)
        status_value = getattr(status, "value", status)
        if isinstance(provider, str) and status_value in {None, "live"}:
            providers.add(provider)
    return providers
