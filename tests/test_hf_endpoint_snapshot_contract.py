from __future__ import annotations

import re
from copy import deepcopy
from typing import cast

import pytest

from harbor_hf.endpoints import (
    EndpointCompute,
    EndpointConfiguration,
    EndpointImage,
    EndpointModel,
    EndpointProvider,
    EndpointRoute,
    EndpointScaling,
    EndpointSnapshot,
    EndpointStatus,
)
from harbor_hf.hf_endpoints import _snapshot_from_raw


def _complete_raw_snapshot() -> dict[str, object]:
    return {
        "name": "endpoint-contract",
        "type": "private",
        "accountId": "account-contract",
        "model": {
            "repository": "organization/model-contract",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "framework": "framework-contract",
            "task": "task-contract",
            "image": {
                "custom": {
                    "url": "registry.example/model@sha256:contract",
                    "healthRoute": "/custom-health",
                    "port": 9097,
                }
            },
            "command": ["serve-contract", "--strict"],
            "args": ["--alpha", "argument-contract"],
            "env": {"ALPHA": "first", "OMEGA": "last"},
            "secrets": {"TOKEN_Z": "hidden-z", "TOKEN_A": "hidden-a"},
        },
        "compute": {
            "accelerator": "accelerator-contract",
            "instanceSize": "size-contract",
            "instanceType": "type-contract",
            "scaling": {
                "minReplica": 2,
                "maxReplica": 7,
                "scaleToZeroTimeout": 43,
                "measure": {"hardwareUsage": 72.5},
            },
        },
        "provider": {
            "vendor": "vendor-contract",
            "region": "region-contract",
        },
        "route": {"domain": "endpoint.example.test", "path": "/models/contract"},
        "cacheHttpResponses": True,
        "tags": ["tag-two", "tag-one"],
        "healthRoute": "/root-health-must-not-win",
        "status": {
            "state": "running",
            "readyReplica": 3,
            "targetReplica": 5,
            "url": "https://endpoint.example.test/models/contract",
        },
    }


def _dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def test_complete_provider_snapshot_maps_every_typed_field() -> None:
    observed = _snapshot_from_raw(_complete_raw_snapshot(), "namespace-contract")

    assert observed == EndpointSnapshot(
        namespace="namespace-contract",
        name="endpoint-contract",
        configuration=EndpointConfiguration(
            model=EndpointModel(
                repository="organization/model-contract",
                revision="0123456789abcdef0123456789abcdef01234567",
                framework="framework-contract",
                task="task-contract",
                image=EndpointImage(
                    url="registry.example/model@sha256:contract",
                    health_route="/custom-health",
                    port=9097,
                ),
                command=["serve-contract", "--strict"],
                arguments=["--alpha", "argument-contract"],
                environment={"ALPHA": "first", "OMEGA": "last"},
                secret_names=["TOKEN_A", "TOKEN_Z"],
            ),
            compute=EndpointCompute(
                accelerator="accelerator-contract",
                instance_size="size-contract",
                instance_type="type-contract",
                scaling=EndpointScaling(
                    min_replicas=2,
                    max_replicas=7,
                    scale_to_zero_timeout=43,
                    metric="hardwareUsage",
                    threshold=72.5,
                ),
            ),
            provider=EndpointProvider(
                vendor="vendor-contract",
                region="region-contract",
                account_id="account-contract",
            ),
            access_type="private",
            route=EndpointRoute(
                domain="endpoint.example.test",
                path="/models/contract",
            ),
            cache_http_responses=True,
            tags=["tag-two", "tag-one"],
        ),
        status=EndpointStatus(
            state="running",
            ready_replicas=3,
            target_replicas=5,
            url="https://endpoint.example.test/models/contract",
        ),
    )


def test_sparse_provider_snapshot_preserves_all_contract_defaults() -> None:
    raw = _complete_raw_snapshot()
    model = _dict(raw["model"])
    image = _dict(model["image"])
    custom = _dict(image["custom"])
    scaling = _dict(_dict(raw["compute"])["scaling"])

    raw.pop("accountId")
    raw.pop("route")
    raw.pop("cacheHttpResponses")
    raw.pop("tags")
    custom.pop("healthRoute")
    custom.pop("port")
    model.pop("command")
    model.pop("args")
    model.pop("env")
    model.pop("secrets")
    scaling.pop("scaleToZeroTimeout")
    scaling.pop("measure")
    status = _dict(raw["status"])
    status.pop("targetReplica")
    status["url"] = None
    raw["type"] = "authenticated"
    raw["healthRoute"] = "/root-health-fallback"

    observed = _snapshot_from_raw(raw, "namespace-sparse")

    assert observed.model_dump() == {
        "namespace": "namespace-sparse",
        "name": "endpoint-contract",
        "configuration": {
            "model": {
                "repository": "organization/model-contract",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "framework": "framework-contract",
                "task": "task-contract",
                "image": {
                    "url": "registry.example/model@sha256:contract",
                    "health_route": "/root-health-fallback",
                    "port": None,
                },
                "command": [],
                "arguments": [],
                "environment": {},
                "secret_names": [],
            },
            "compute": {
                "accelerator": "accelerator-contract",
                "instance_size": "size-contract",
                "instance_type": "type-contract",
                "scaling": {
                    "min_replicas": 2,
                    "max_replicas": 7,
                    "scale_to_zero_timeout": None,
                    "metric": None,
                    "threshold": None,
                },
            },
            "provider": {
                "vendor": "vendor-contract",
                "region": "region-contract",
                "account_id": None,
            },
            "access_type": "authenticated",
            "route": {"domain": None, "path": None},
            "cache_http_responses": False,
            "tags": [],
        },
        "status": {
            "state": "running",
            "ready_replicas": 3,
            "target_replicas": 0,
            "url": None,
        },
    }


@pytest.mark.parametrize(
    ("path", "invalid", "message"),
    [
        ((), [], "endpoint must be an object"),
        (("model",), None, "endpoint.model must be an object"),
        (("model", "image"), None, "endpoint.model.image must be an object"),
        (
            ("model", "image", "custom"),
            None,
            "endpoint.model.image.custom must be an object",
        ),
        (("compute",), None, "endpoint.compute must be an object"),
        (("compute", "scaling"), None, "endpoint.compute.scaling must be an object"),
        (("provider",), None, "endpoint.provider must be an object"),
        (("status",), None, "endpoint.status must be an object"),
        (("route",), [], "endpoint.route must be an object"),
        (("name",), None, "endpoint.name must be a string"),
        (
            ("model", "image", "custom", "port"),
            True,
            "model.image.port must be an integer",
        ),
        (
            ("model", "command"),
            ["valid", 3],
            "model.command must be an array of strings",
        ),
        (("model", "env"), {"COUNT": 3}, "model.env values must be strings"),
        (("type",), "unsupported", "endpoint.type is unsupported"),
        (
            ("compute", "scaling", "measure"),
            {"hardwareUsage": True},
            "scaling.measure threshold must be numeric",
        ),
        (("cacheHttpResponses",), 1, "cacheHttpResponses must be a boolean"),
        (("status", "readyReplica"), True, "status.readyReplica must be an integer"),
    ],
)
def test_snapshot_contract_reports_exact_invalid_provider_path(
    path: tuple[str, ...], invalid: object, message: str
) -> None:
    raw: object = deepcopy(_complete_raw_snapshot())
    if path:
        current = raw
        for part in path[:-1]:
            current = _dict(current)[part]
        _dict(current)[path[-1]] = invalid
    else:
        raw = invalid

    with pytest.raises((TypeError, ValueError), match=f"^{re.escape(message)}$"):
        _snapshot_from_raw(raw, "namespace-contract")
