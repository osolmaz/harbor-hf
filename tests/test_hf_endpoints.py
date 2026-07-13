from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from harbor_hf.endpoints import (
    AmbiguousEndpointCreate,
    AmbiguousEndpointDelete,
    AmbiguousEndpointPause,
    DesiredEndpoint,
    EndpointProviderError,
    build_desired_endpoint,
)
from harbor_hf.hf_endpoints import HuggingFaceEndpointAdapter, environment_secret
from harbor_hf.models import DeploymentProfile, ExperimentSpec


@dataclass
class Resource:
    raw: object


@dataclass
class FakeApi:
    resource: Resource
    errors: dict[str, BaseException] = field(default_factory=dict)
    calls: list[tuple[str, str, dict[str, object]]] = field(default_factory=list)

    def _result(self, operation: str, name: str, kwargs: dict[str, object]) -> Resource:
        self.calls.append((operation, name, kwargs))
        error = self.errors.get(operation)
        if error is not None:
            raise error
        return self.resource

    def create_inference_endpoint(self, name: str, **kwargs: object) -> Resource:
        return self._result("create", name, kwargs)

    def get_inference_endpoint(self, name: str, **kwargs: object) -> Resource:
        return self._result("inspect", name, kwargs)

    def pause_inference_endpoint(self, name: str, **kwargs: object) -> Resource:
        return self._result("pause", name, kwargs)

    def delete_inference_endpoint(self, name: str, **kwargs: object) -> None:
        self._result("delete", name, kwargs)


def _desired(remote_spec: ExperimentSpec) -> DesiredEndpoint:
    deployment = remote_spec.matrix.deployments[0]
    assert isinstance(deployment, DeploymentProfile)
    deployment = deployment.model_copy(
        update={
            "parameters": {
                "min_replicas": 0,
                "max_replicas": 2,
                "scale_to_zero_timeout": 15,
                "scaling_metric": "pendingRequests",
                "scaling_threshold": 1.5,
                "health_route": "/ready",
                "port": 8080,
                "account_id": "account-one",
                "domain": "endpoint.example.test",
                "path": "/models/qwen",
                "cache_http_responses": True,
                "tags": ["benchmark"],
            }
        }
    )
    return build_desired_endpoint(
        namespace="osolmaz",
        campaign_id="campaign-one",
        model=remote_spec.matrix.models[0],
        deployment=deployment,
    )


def _raw(desired: DesiredEndpoint) -> dict[str, object]:
    configuration = desired.configuration
    model = configuration.model
    scaling = configuration.compute.scaling
    return {
        "name": desired.identity.name,
        "type": configuration.access_type,
        "accountId": configuration.provider.account_id,
        "model": {
            "repository": model.repository,
            "revision": model.revision,
            "framework": model.framework,
            "task": model.task,
            "image": {
                "custom": {
                    "url": model.image.url,
                    "healthRoute": model.image.health_route,
                    "port": model.image.port,
                }
            },
            "command": model.command,
            "args": model.arguments,
            "env": model.environment,
            "secrets": {name: object() for name in model.secret_names},
        },
        "compute": {
            "accelerator": configuration.compute.accelerator,
            "instanceSize": configuration.compute.instance_size,
            "instanceType": configuration.compute.instance_type,
            "scaling": {
                "minReplica": scaling.min_replicas,
                "maxReplica": scaling.max_replicas,
                "scaleToZeroTimeout": scaling.scale_to_zero_timeout,
                "measure": {scaling.metric: scaling.threshold},
            },
        },
        "provider": {
            "vendor": configuration.provider.vendor,
            "region": configuration.provider.region,
        },
        "route": {
            "domain": configuration.route.domain,
            "path": configuration.route.path,
        },
        "cacheHttpResponses": configuration.cache_http_responses,
        "tags": list(reversed(configuration.tags)),
        "healthRoute": model.image.health_route,
        "status": {
            "state": "paused",
            "readyReplica": 0,
            "targetReplica": 2,
            "url": "https://endpoint.example.test",
        },
    }


def _http_error(status: int) -> HfHubHTTPError:
    request = httpx.Request("POST", "https://api.endpoints.huggingface.test")
    response = httpx.Response(status, request=request)
    return HfHubHTTPError(f"HTTP {status}", response=response)


def test_create_maps_every_effective_field_and_resolves_secrets(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    api = FakeApi(Resource(_raw(desired)))
    adapter = HuggingFaceEndpointAdapter(
        api=api,
        token=False,
        secret_resolver=lambda name: f"value-for-{name}",
    )

    snapshot = adapter.create(desired)

    assert snapshot.configuration == desired.configuration
    operation, name, kwargs = api.calls[0]
    assert (operation, name) == ("create", desired.identity.name)
    assert kwargs == {
        "repository": desired.configuration.model.repository,
        "framework": "custom",
        "accelerator": "gpu",
        "instance_size": "x1",
        "instance_type": "nvidia-rtx-pro-6000",
        "region": "us-east-1",
        "vendor": "aws",
        "account_id": "account-one",
        "min_replica": 0,
        "max_replica": 2,
        "scaling_metric": "pendingRequests",
        "scaling_threshold": 1.5,
        "scale_to_zero_timeout": 15,
        "revision": desired.configuration.model.revision,
        "task": "text-generation",
        "custom_image": {
            "url": desired.configuration.model.image.url,
            "healthRoute": "/ready",
            "port": 8080,
        },
        "container_command": None,
        "container_args": desired.configuration.model.arguments,
        "env": desired.configuration.model.environment,
        "secrets": {"HF_TOKEN": "value-for-HF_TOKEN"},
        "type": "authenticated",
        "domain": "endpoint.example.test",
        "path": "/models/qwen",
        "cache_http_responses": True,
        "tags": desired.configuration.tags,
        "namespace": "osolmaz",
        "token": False,
    }
    assert "value-for-HF_TOKEN" not in snapshot.model_dump_json()


def test_inspect_validates_and_normalizes_sanitized_contract(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    api = FakeApi(Resource(_raw(desired)))

    snapshot = HuggingFaceEndpointAdapter(api=api).inspect(desired.identity)

    assert snapshot is not None
    assert snapshot.status.model_dump() == {
        "state": "paused",
        "ready_replicas": 0,
        "target_replicas": 2,
        "url": "https://endpoint.example.test",
    }
    assert snapshot.configuration.tags == desired.configuration.tags
    assert snapshot.configuration.model.secret_names == ["HF_TOKEN"]
    assert api.calls == [
        (
            "inspect",
            desired.identity.name,
            {"namespace": "osolmaz", "token": None},
        )
    ]


def test_inspect_accepts_provider_default_scaling_measure(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    raw = _raw(desired)
    scaling = cast(
        dict[str, object], cast(dict[str, object], raw["compute"])["scaling"]
    )
    scaling["measure"] = {"hardwareUsage": None}

    snapshot = HuggingFaceEndpointAdapter(api=FakeApi(Resource(raw))).inspect(
        desired.identity
    )

    assert snapshot is not None
    assert snapshot.configuration.compute.scaling.metric is None
    assert snapshot.configuration.compute.scaling.threshold is None


def test_pause_and_delete_use_only_exact_identity(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    api = FakeApi(Resource(_raw(desired)))
    adapter = HuggingFaceEndpointAdapter(api=api, token=False)

    paused = adapter.pause(desired.identity)
    adapter.delete(desired.identity)

    assert paused.status.state == "paused"
    assert api.calls == [
        (
            "pause",
            desired.identity.name,
            {"namespace": "osolmaz", "token": False},
        ),
        (
            "delete",
            desired.identity.name,
            {"namespace": "osolmaz", "token": False},
        ),
    ]


def test_inspect_returns_none_only_for_not_found(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    api = FakeApi(Resource(_raw(desired)), errors={"inspect": _http_error(404)})

    assert HuggingFaceEndpointAdapter(api=api).inspect(desired.identity) is None


@pytest.mark.parametrize("status", [409, 500, 503])
def test_ambiguous_create_http_outcomes_are_adoptable(
    remote_spec: ExperimentSpec, status: int
) -> None:
    desired = _desired(remote_spec)
    api = FakeApi(Resource(_raw(desired)), errors={"create": _http_error(status)})

    with pytest.raises(AmbiguousEndpointCreate, match=f"HTTP {status}"):
        HuggingFaceEndpointAdapter(
            api=api, secret_resolver=lambda name: "secret"
        ).create(desired)


def test_create_transport_failure_is_ambiguous(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    api = FakeApi(
        Resource(_raw(desired)),
        errors={"create": httpx.ReadTimeout("timed out")},
    )

    with pytest.raises(AmbiguousEndpointCreate, match="before a response"):
        HuggingFaceEndpointAdapter(
            api=api, secret_resolver=lambda name: "secret"
        ).create(desired)


def test_definitive_create_failure_is_not_adopted(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    api = FakeApi(Resource(_raw(desired)), errors={"create": _http_error(400)})

    with pytest.raises(EndpointProviderError, match="failed: HTTP 400"):
        HuggingFaceEndpointAdapter(
            api=api, secret_resolver=lambda name: "secret"
        ).create(desired)


def test_pause_transport_failure_is_ambiguous(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    api = FakeApi(
        Resource(_raw(desired)),
        errors={"pause": httpx.ConnectError("disconnected")},
    )

    with pytest.raises(AmbiguousEndpointPause):
        HuggingFaceEndpointAdapter(api=api).pause(desired.identity)


def test_delete_not_found_is_ambiguous_and_verifiable(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    api = FakeApi(Resource(_raw(desired)), errors={"delete": _http_error(404)})

    with pytest.raises(AmbiguousEndpointDelete, match="HTTP 404"):
        HuggingFaceEndpointAdapter(api=api).delete(desired.identity)


def test_inspect_transport_and_server_failures_are_not_absence(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    for error in (httpx.ReadError("broken"), _http_error(503)):
        api = FakeApi(Resource(_raw(desired)), errors={"inspect": error})
        with pytest.raises(EndpointProviderError, match="inspect failed"):
            HuggingFaceEndpointAdapter(api=api).inspect(desired.identity)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("status", "readyReplica"), "zero"),
        (("compute", "scaling", "measure"), {"unknown": 1}),
        (("model", "env"), {"COUNT": 1}),
        (("type",), "protected"),
    ),
)
def test_rejects_malformed_provider_contract(
    remote_spec: ExperimentSpec, path: tuple[str, ...], value: object
) -> None:
    desired = _desired(remote_spec)
    raw = _raw(desired)
    current: dict[str, object] = raw
    for part in path[:-1]:
        nested = current[part]
        assert isinstance(nested, dict)
        current = cast(dict[str, object], nested)
    current[path[-1]] = value
    api = FakeApi(Resource(raw))

    with pytest.raises(EndpointProviderError, match="expected contract"):
        HuggingFaceEndpointAdapter(api=api).inspect(desired.identity)


def test_create_treats_malformed_success_response_as_ambiguous(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    raw = _raw(desired)
    cast(dict[str, object], raw["status"])["readyReplica"] = "zero"

    with pytest.raises(AmbiguousEndpointCreate, match="invalid response"):
        HuggingFaceEndpointAdapter(
            api=FakeApi(Resource(raw)),
            secret_resolver=lambda name: "secret",
        ).create(desired)


def test_missing_secret_fails_before_provider_create(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    api = FakeApi(Resource(_raw(desired)))

    def missing(name: str) -> str:
        raise EndpointProviderError(f"missing {name}")

    with pytest.raises(EndpointProviderError, match="missing HF_TOKEN"):
        HuggingFaceEndpointAdapter(
            api=api,
            secret_resolver=missing,
        ).create(desired)

    assert api.calls == []


def test_environment_secret_reads_only_named_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENDPOINT_SECRET", "resolved-value")

    assert environment_secret("ENDPOINT_SECRET") == "resolved-value"
    with pytest.raises(EndpointProviderError, match="MISSING_SECRET"):
        environment_secret("MISSING_SECRET")
