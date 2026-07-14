from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Literal, Protocol, cast

import httpx
from huggingface_hub.errors import HfHubHTTPError

from harbor_hf.endpoints import (
    AmbiguousEndpointCreate,
    AmbiguousEndpointDelete,
    AmbiguousEndpointPause,
    DesiredEndpoint,
    EndpointCompute,
    EndpointConfiguration,
    EndpointImage,
    EndpointModel,
    EndpointProvider,
    EndpointProviderError,
    EndpointProvisioningError,
    EndpointProvisioningPort,
    EndpointRoute,
    EndpointScaling,
    EndpointSnapshot,
    EndpointStatus,
    ManagedEndpointIdentity,
)


class EndpointResource(Protocol):
    raw: object


class HfEndpointApi(Protocol):
    def create_inference_endpoint(
        self, name: str, **kwargs: object
    ) -> EndpointResource: ...

    def get_inference_endpoint(
        self, name: str, **kwargs: object
    ) -> EndpointResource: ...

    def pause_inference_endpoint(
        self, name: str, **kwargs: object
    ) -> EndpointResource: ...

    def delete_inference_endpoint(self, name: str, **kwargs: object) -> None: ...


class SecretResolver(Protocol):
    def __call__(self, name: str) -> str: ...


def environment_secret(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError as error:
        raise EndpointProviderError(
            f"required endpoint secret is not available: {name}"
        ) from error


class HuggingFaceEndpointAdapter(EndpointProvisioningPort):
    def __init__(
        self,
        *,
        api: HfEndpointApi | None = None,
        token: str | bool | None = None,
        secret_resolver: SecretResolver = environment_secret,
    ) -> None:
        if api is None:
            from huggingface_hub import HfApi

            api = cast(HfEndpointApi, HfApi(token=token))
        self.api = api
        self.token = token
        self.secret_resolver = secret_resolver

    def create(self, desired: DesiredEndpoint) -> EndpointSnapshot:
        identity = desired.identity
        configuration = desired.configuration
        model = configuration.model
        compute = configuration.compute
        scaling = compute.scaling
        provider = configuration.provider
        secrets = {name: self.secret_resolver(name) for name in model.secret_names}
        custom_image: dict[str, object] = {
            "url": model.image.url,
            "healthRoute": model.image.health_route,
        }
        if model.image.port is not None:
            custom_image["port"] = model.image.port

        def request() -> EndpointResource:
            return self.api.create_inference_endpoint(
                identity.name,
                repository=model.repository,
                framework=model.framework,
                accelerator=compute.accelerator,
                instance_size=compute.instance_size,
                instance_type=compute.instance_type,
                region=provider.region,
                vendor=provider.vendor,
                account_id=provider.account_id,
                min_replica=scaling.min_replicas,
                max_replica=scaling.max_replicas,
                scaling_metric=scaling.metric,
                scaling_threshold=scaling.threshold,
                scale_to_zero_timeout=scaling.scale_to_zero_timeout,
                revision=model.revision,
                task=model.task,
                custom_image=custom_image,
                container_command=model.command or None,
                container_args=model.arguments or None,
                env=model.environment or None,
                secrets=secrets or None,
                type=configuration.access_type,
                domain=configuration.route.domain,
                path=configuration.route.path,
                cache_http_responses=configuration.cache_http_responses,
                tags=configuration.tags,
                namespace=identity.namespace,
                token=self.token,
            )

        resource = _provider_call("create", request, ambiguous=AmbiguousEndpointCreate)
        try:
            return _validated_snapshot(resource, identity.namespace)
        except EndpointProviderError as error:
            raise AmbiguousEndpointCreate(
                "Hugging Face endpoint create returned an invalid response"
            ) from error

    def inspect(self, identity: ManagedEndpointIdentity) -> EndpointSnapshot | None:
        try:
            resource = self.api.get_inference_endpoint(
                identity.name,
                namespace=identity.namespace,
                token=self.token,
            )
        except HfHubHTTPError as error:
            if error.response.status_code == 404:
                return None
            raise EndpointProviderError(
                "Hugging Face endpoint inspect failed: "
                f"HTTP {error.response.status_code}"
            ) from error
        except httpx.TransportError as error:
            raise EndpointProviderError(
                "Hugging Face endpoint inspect failed before a response"
            ) from error
        return _validated_snapshot(resource, identity.namespace)

    def pause(self, identity: ManagedEndpointIdentity) -> EndpointSnapshot:
        def request() -> EndpointResource:
            return self.api.pause_inference_endpoint(
                identity.name,
                namespace=identity.namespace,
                token=self.token,
            )

        resource = _provider_call("pause", request, ambiguous=AmbiguousEndpointPause)
        try:
            return _validated_snapshot(resource, identity.namespace)
        except EndpointProviderError as error:
            raise AmbiguousEndpointPause(
                "Hugging Face endpoint pause returned an invalid response"
            ) from error

    def delete(self, identity: ManagedEndpointIdentity) -> None:
        def request() -> None:
            self.api.delete_inference_endpoint(
                identity.name,
                namespace=identity.namespace,
                token=self.token,
            )

        _provider_call("delete", request, ambiguous=AmbiguousEndpointDelete)


def _provider_call[Result](
    operation: str,
    request: Callable[[], Result],
    *,
    ambiguous: type[EndpointProvisioningError],
) -> Result:
    try:
        return request()
    except HfHubHTTPError as error:
        status = error.response.status_code
        if status == 409 or status >= 500 or (operation == "delete" and status == 404):
            raise ambiguous(
                f"Hugging Face endpoint {operation} outcome is ambiguous: HTTP {status}"
            ) from error
        raise EndpointProviderError(
            f"Hugging Face endpoint {operation} failed: HTTP {status}"
        ) from error
    except httpx.TransportError as error:
        raise ambiguous(
            f"Hugging Face endpoint {operation} outcome is ambiguous before a response"
        ) from error


def _validated_snapshot(resource: EndpointResource, namespace: str) -> EndpointSnapshot:
    try:
        return _snapshot_from_raw(resource.raw, namespace)
    except (KeyError, TypeError, ValueError) as error:
        raise EndpointProviderError(
            "Hugging Face endpoint response does not match the expected contract"
        ) from error


def _snapshot_from_raw(raw_value: object, namespace: str) -> EndpointSnapshot:
    raw = _mapping(raw_value, "endpoint")
    model = _mapping(raw.get("model"), "endpoint.model")
    image = _mapping(model.get("image"), "endpoint.model.image")
    custom = _mapping(image.get("custom"), "endpoint.model.image.custom")
    compute = _mapping(raw.get("compute"), "endpoint.compute")
    scaling = _mapping(compute.get("scaling"), "endpoint.compute.scaling")
    provider = _mapping(raw.get("provider"), "endpoint.provider")
    status = _mapping(raw.get("status"), "endpoint.status")
    route = _optional_mapping(raw.get("route"), "endpoint.route")
    metric, threshold = _scaling_measure(scaling.get("measure"))
    health_route = custom.get("healthRoute", raw.get("healthRoute"))
    return EndpointSnapshot(
        namespace=namespace,
        name=_string(raw.get("name"), "endpoint.name"),
        configuration=EndpointConfiguration(
            model=EndpointModel(
                repository=_string(model.get("repository"), "model.repository"),
                revision=_string(model.get("revision"), "model.revision"),
                framework=_string(model.get("framework"), "model.framework"),
                task=_string(model.get("task"), "model.task"),
                image=EndpointImage(
                    url=_string(custom.get("url"), "model.image.custom.url"),
                    health_route=_string(health_route, "model.image.healthRoute"),
                    port=_optional_integer(custom.get("port"), "model.image.port"),
                ),
                command=_string_list(model.get("command", []), "model.command"),
                arguments=_string_list(model.get("args", []), "model.args"),
                environment=_string_mapping(model.get("env", {}), "model.env"),
                secret_names=_secret_names(model.get("secrets", {})),
            ),
            compute=EndpointCompute(
                accelerator=_string(compute.get("accelerator"), "compute.accelerator"),
                instance_size=_string(
                    compute.get("instanceSize"), "compute.instanceSize"
                ),
                instance_type=_string(
                    compute.get("instanceType"), "compute.instanceType"
                ),
                scaling=EndpointScaling(
                    min_replicas=_integer(
                        scaling.get("minReplica"), "scaling.minReplica"
                    ),
                    max_replicas=_integer(
                        scaling.get("maxReplica"), "scaling.maxReplica"
                    ),
                    scale_to_zero_timeout=_optional_integer(
                        scaling.get("scaleToZeroTimeout"),
                        "scaling.scaleToZeroTimeout",
                    ),
                    metric=metric,
                    threshold=threshold,
                ),
            ),
            provider=EndpointProvider(
                vendor=_string(provider.get("vendor"), "provider.vendor"),
                region=_string(provider.get("region"), "provider.region"),
                account_id=_optional_string(raw.get("accountId"), "accountId"),
            ),
            access_type=_access_type(raw.get("type")),
            route=EndpointRoute(
                domain=_optional_string(route.get("domain"), "route.domain"),
                path=_optional_string(route.get("path"), "route.path"),
            ),
            cache_http_responses=_boolean(
                raw.get("cacheHttpResponses", False), "cacheHttpResponses"
            ),
            tags=_string_list(raw.get("tags", []), "endpoint.tags"),
        ),
        status=EndpointStatus(
            state=_string(status.get("state"), "status.state"),
            ready_replicas=_integer(status.get("readyReplica"), "status.readyReplica"),
            target_replicas=_integer(
                status.get("targetReplica", 0), "status.targetReplica"
            ),
            url=_optional_string(status.get("url"), "status.url"),
        ),
    )


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object")
    return cast(Mapping[str, object], value)


def _optional_mapping(value: object, path: str) -> Mapping[str, object]:
    return {} if value is None else _mapping(value, path)


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    return value


def _optional_string(value: object, path: str) -> str | None:
    return None if value is None else _string(value, path)


def _integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path} must be an integer")
    return value


def _optional_integer(value: object, path: str) -> int | None:
    return None if value is None else _integer(value, path)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


def _string_list(value: object, path: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{path} must be an array of strings")
    return cast(list[str], value)


def _access_type(value: object) -> Literal["public", "authenticated", "private"]:
    observed = _string(value, "endpoint.type")
    if observed not in {"public", "authenticated", "private"}:
        raise ValueError("endpoint.type is unsupported")
    return cast(Literal["public", "authenticated", "private"], observed)


def _string_mapping(value: object, path: str) -> dict[str, str]:
    mapping = _mapping(value, path)
    if not all(isinstance(item, str) for item in mapping.values()):
        raise TypeError(f"{path} values must be strings")
    return {key: cast(str, item) for key, item in mapping.items()}


def _secret_names(value: object) -> list[str]:
    return sorted(_mapping(value, "model.secrets"))


def _scaling_measure(
    value: object,
) -> tuple[Literal["pendingRequests", "hardwareUsage"] | None, float | None]:
    if value is None:
        return None, None
    measure = _mapping(value, "scaling.measure")
    if len(measure) != 1:
        raise ValueError("scaling.measure must contain exactly one metric")
    metric, threshold = next(iter(measure.items()))
    if metric not in {"pendingRequests", "hardwareUsage"}:
        raise ValueError("scaling.measure contains an unsupported metric")
    if threshold is None:
        return None, None
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise TypeError("scaling.measure threshold must be numeric")
    return cast(Literal["pendingRequests", "hardwareUsage"], metric), float(threshold)
