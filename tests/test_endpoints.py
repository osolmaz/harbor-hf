from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import pytest
from pydantic import ValidationError

from harbor_hf.endpoints import (
    AmbiguousEndpointCreate,
    AmbiguousEndpointDelete,
    AmbiguousEndpointPause,
    DesiredEndpoint,
    EndpointConfiguration,
    EndpointConfigurationMismatch,
    EndpointIdentityMismatch,
    EndpointNotFound,
    EndpointNotPaused,
    EndpointProvisioner,
    EndpointSnapshot,
    EndpointStatus,
    EndpointVerificationTimeout,
    ManagedEndpointIdentity,
    build_desired_endpoint,
    deployment_digest,
    effective_configuration_mismatches,
    managed_endpoint_identity,
    require_paused_zero_ready,
    verify_exact_endpoint,
)
from harbor_hf.models import EndpointRef, ExperimentSpec


def _desired(remote_spec: ExperimentSpec) -> DesiredEndpoint:
    deployment = remote_spec.matrix.deployments[0]
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


def _snapshot(
    desired: DesiredEndpoint,
    *,
    state: str = "paused",
    ready: int = 0,
    target: int = 1,
    configuration: EndpointConfiguration | None = None,
) -> EndpointSnapshot:
    return EndpointSnapshot(
        namespace=desired.identity.namespace,
        name=desired.identity.name,
        configuration=configuration or desired.configuration,
        status=EndpointStatus(
            state=state,
            ready_replicas=ready,
            target_replicas=target,
            url="https://endpoint.example.test",
        ),
    )


def _replace_configuration(
    configuration: EndpointConfiguration, path: str, value: object
) -> EndpointConfiguration:
    payload = cast(dict[str, object], configuration.model_dump(mode="python"))
    current: dict[str, object] = payload
    parts = path.split(".")
    for part in parts[:-1]:
        nested = current[part]
        assert isinstance(nested, dict)
        current = cast(dict[str, object], nested)
    current[parts[-1]] = value
    return EndpointConfiguration.model_validate(payload)


@dataclass
class FakePort:
    inspections: list[EndpointSnapshot | None]
    create_result: EndpointSnapshot | BaseException | None = None
    pause_result: EndpointSnapshot | BaseException | None = None
    delete_error: BaseException | None = None
    calls: list[str] = field(default_factory=list)

    def inspect(self, identity: ManagedEndpointIdentity) -> EndpointSnapshot | None:
        self.calls.append(f"inspect:{identity.name}")
        if len(self.inspections) > 1:
            return self.inspections.pop(0)
        return self.inspections[0] if self.inspections else None

    def create(self, desired: DesiredEndpoint) -> EndpointSnapshot:
        self.calls.append(f"create:{desired.identity.name}")
        if isinstance(self.create_result, BaseException):
            raise self.create_result
        assert self.create_result is not None
        return self.create_result

    def pause(self, identity: ManagedEndpointIdentity) -> EndpointSnapshot:
        self.calls.append(f"pause:{identity.name}")
        if isinstance(self.pause_result, BaseException):
            raise self.pause_result
        assert self.pause_result is not None
        return self.pause_result

    def delete(self, identity: ManagedEndpointIdentity) -> None:
        self.calls.append(f"delete:{identity.name}")
        if self.delete_error is not None:
            raise self.delete_error


@dataclass
class FakeTime:
    now: float = 0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _provisioner(port: FakePort) -> EndpointProvisioner:
    clock = FakeTime()
    return EndpointProvisioner(
        port,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def test_builds_complete_deterministic_desired_endpoint(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)

    assert desired.identity.name.startswith("harbor-hf-")
    assert len(desired.identity.name) == 50
    assert desired.identity.tags == sorted(desired.identity.tags)
    assert set(desired.identity.tags) < set(desired.configuration.tags)
    assert desired.configuration.model.repository == remote_spec.matrix.models[0].repo
    assert desired.configuration.compute.instance_type == "nvidia-rtx-pro-6000"
    assert desired.configuration.compute.instance_size == "x1"
    assert desired.configuration.provider.model_dump() == {
        "vendor": "aws",
        "region": "us-east-1",
        "account_id": "account-one",
    }


def test_deployment_digest_ignores_labels_and_prebound_endpoint_name(
    remote_spec: ExperimentSpec,
) -> None:
    model = remote_spec.matrix.models[0]
    deployment = remote_spec.matrix.deployments[0]
    renamed = deployment.model_copy(
        update={
            "id": "renamed",
            "endpoint": EndpointRef(
                namespace="another",
                name="another-endpoint",
                served_model_name="another-model",
            ),
        }
    )

    digest = deployment_digest(model, deployment)
    assert (
        digest
        == "sha256:44eb3bf9ec103630645e99e08b577c8dba431cb1ad83f237a3ab17e949fb0c65"
    )
    assert digest == deployment_digest(model, renamed)
    changed = renamed.model_copy(update={"hardware": "h200"})
    assert deployment_digest(model, deployment) != deployment_digest(model, changed)


def test_identity_changes_by_campaign_namespace_and_deployment(
    remote_spec: ExperimentSpec,
) -> None:
    digest = deployment_digest(
        remote_spec.matrix.models[0], remote_spec.matrix.deployments[0]
    )
    first = managed_endpoint_identity(
        namespace="osolmaz", campaign_id="one", deployment_digest=digest
    )

    assert first == managed_endpoint_identity(
        namespace="osolmaz", campaign_id="one", deployment_digest=digest
    )
    assert (
        first.name
        != managed_endpoint_identity(
            namespace="osolmaz", campaign_id="two", deployment_digest=digest
        ).name
    )
    assert (
        first.name
        != managed_endpoint_identity(
            namespace="another", campaign_id="one", deployment_digest=digest
        ).name
    )


def test_managed_identity_has_stable_golden_value(
    remote_spec: ExperimentSpec,
) -> None:
    digest = deployment_digest(
        remote_spec.matrix.models[0], remote_spec.matrix.deployments[0]
    )

    identity = managed_endpoint_identity(
        namespace="osolmaz",
        campaign_id="campaign-one",
        deployment_digest=digest,
    )

    assert identity.name == "harbor-hf-1b6bb067d5c61cbac7d4970caea72b95d95e57ac"
    assert identity.tags == [
        "harbor-hf-campaign-27f8a68166e2255551573839",
        "harbor-hf-deployment-44eb3bf9ec103630645e99e0",
        "harbor-hf-managed",
    ]


def test_rejects_unknown_endpoint_parameters(remote_spec: ExperimentSpec) -> None:
    deployment = remote_spec.matrix.deployments[0].model_copy(
        update={"parameters": {"unreported_provider_control": True}}
    )

    with pytest.raises(ValidationError, match="unreported_provider_control"):
        build_desired_endpoint(
            namespace="osolmaz",
            campaign_id="campaign-one",
            model=remote_spec.matrix.models[0],
            deployment=deployment,
        )


def test_rejects_noncomposite_provider_region(remote_spec: ExperimentSpec) -> None:
    deployment = remote_spec.matrix.deployments[0].model_copy(update={"region": "aws"})

    with pytest.raises(ValueError, match="vendor-region"):
        build_desired_endpoint(
            namespace="osolmaz",
            campaign_id="campaign-one",
            model=remote_spec.matrix.models[0],
            deployment=deployment,
        )


MISMATCH_CASES: Sequence[tuple[str, object]] = (
    ("model.repository", "other/repository"),
    ("model.revision", "b" * 40),
    ("model.framework", "pytorch"),
    ("model.task", "feature-extraction"),
    ("model.image.url", "registry.example/image@sha256:" + "f" * 64),
    ("model.image.health_route", "/healthz"),
    ("model.image.port", 9090),
    ("model.command", ["serve"]),
    ("model.arguments", ["--different"]),
    ("model.environment", {"DIFFERENT": "1"}),
    ("model.secret_names", ["OTHER_SECRET"]),
    ("compute.accelerator", "cpu"),
    ("compute.instance_size", "x2"),
    ("compute.instance_type", "nvidia-h200"),
    ("compute.scaling.min_replicas", 1),
    ("compute.scaling.max_replicas", 3),
    ("compute.scaling.scale_to_zero_timeout", 30),
    ("compute.scaling.metric", "hardwareUsage"),
    ("compute.scaling.threshold", 2.5),
    ("provider.vendor", "gcp"),
    ("provider.region", "us-central1"),
    ("provider.account_id", "account-two"),
    ("access_type", "private"),
    ("route.domain", "other.example.test"),
    ("route.path", "/other"),
    ("cache_http_responses", False),
    ("tags", ["different"]),
)


@pytest.mark.parametrize(("path", "value"), MISMATCH_CASES)
def test_detects_every_effective_configuration_mismatch(
    remote_spec: ExperimentSpec, path: str, value: object
) -> None:
    desired = _desired(remote_spec)
    observed = _replace_configuration(desired.configuration, path, value)

    mismatches = effective_configuration_mismatches(desired.configuration, observed)

    assert mismatches
    assert all(
        mismatch.path == f"configuration.{path}"
        or mismatch.path.startswith(f"configuration.{path}.")
        for mismatch in mismatches
    )


def test_reports_all_configuration_mismatches_together(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    observed = desired.configuration.model_copy(
        update={
            "access_type": "private",
            "cache_http_responses": False,
        }
    )

    with pytest.raises(EndpointConfigurationMismatch) as caught:
        verify_exact_endpoint(desired, _snapshot(desired, configuration=observed))

    assert [item.path for item in caught.value.mismatches] == [
        "configuration.access_type",
        "configuration.cache_http_responses",
    ]


def test_rejects_missing_managed_identity_tag(remote_spec: ExperimentSpec) -> None:
    desired = _desired(remote_spec)
    observed = desired.configuration.model_copy(update={"tags": ["benchmark"]})

    with pytest.raises(EndpointIdentityMismatch, match="managed identity"):
        verify_exact_endpoint(desired, _snapshot(desired, configuration=observed))


def test_paused_verification_allows_nonzero_target(remote_spec: ExperimentSpec) -> None:
    require_paused_zero_ready(_snapshot(_desired(remote_spec), target=2))


@pytest.mark.parametrize(("state", "ready"), (("running", 0), ("paused", 1)))
def test_paused_verification_rejects_active_state(
    remote_spec: ExperimentSpec, state: str, ready: int
) -> None:
    with pytest.raises(EndpointNotPaused, match="targetReplica=1"):
        require_paused_zero_ready(
            _snapshot(_desired(remote_spec), state=state, ready=ready)
        )


def test_adopts_only_an_exact_paused_endpoint(remote_spec: ExperimentSpec) -> None:
    desired = _desired(remote_spec)
    port = FakePort(inspections=[_snapshot(desired)])

    result = _provisioner(port).create_or_adopt(desired)

    assert result.action == "adopted"
    assert result.snapshot.status.ready_replicas == 0
    assert not any(call.startswith(("create:", "pause:")) for call in port.calls)


def test_rejects_active_endpoint_instead_of_pausing_competing_work(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    port = FakePort(inspections=[_snapshot(desired, state="running", ready=1)])

    with pytest.raises(EndpointNotPaused, match="state='running'"):
        _provisioner(port).create_or_adopt(desired)

    assert not any(call.startswith("pause:") for call in port.calls)


def test_creates_then_pauses_and_verifies_zero_ready(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    running = _snapshot(desired, state="running", ready=1)
    pausing = _snapshot(desired, state="pausing", ready=1)
    paused = _snapshot(desired, target=2)
    port = FakePort(
        inspections=[None, running, pausing, paused],
        create_result=running,
        pause_result=pausing,
    )

    result = _provisioner(port).create_or_adopt(
        desired, timeout_seconds=10, poll_seconds=1
    )

    assert result.action == "created"
    assert result.snapshot == paused
    assert sum(call.startswith("create:") for call in port.calls) == 1
    assert sum(call.startswith("pause:") for call in port.calls) == 1


def test_adopts_after_ambiguous_create_and_pauses(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    running = _snapshot(desired, state="running", ready=1)
    paused = _snapshot(desired)
    port = FakePort(
        inspections=[None, None, running, running, paused],
        create_result=AmbiguousEndpointCreate("timeout"),
        pause_result=AmbiguousEndpointPause("timeout"),
    )

    result = _provisioner(port).create_or_adopt(
        desired, timeout_seconds=10, poll_seconds=1
    )

    assert result.action == "adopted"
    assert result.snapshot == paused


def test_ambiguous_create_times_out_without_duplicate_create(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    port = FakePort(
        inspections=[None],
        create_result=AmbiguousEndpointCreate("timeout"),
    )

    with pytest.raises(EndpointVerificationTimeout, match="could not be adopted"):
        _provisioner(port).create_or_adopt(desired, timeout_seconds=2, poll_seconds=1)

    assert sum(call.startswith("create:") for call in port.calls) == 1


def test_mismatched_create_is_paused_before_rejection(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    mismatched = desired.configuration.model_copy(update={"access_type": "private"})
    running = _snapshot(
        desired,
        state="running",
        ready=1,
        configuration=mismatched,
    )
    paused = _snapshot(desired, configuration=mismatched)
    port = FakePort(
        inspections=[None, paused],
        create_result=running,
        pause_result=paused,
    )

    with pytest.raises(EndpointConfigurationMismatch):
        _provisioner(port).create_or_adopt(desired)

    assert sum(call.startswith("pause:") for call in port.calls) == 1


def test_pause_rejects_disappearing_endpoint(remote_spec: ExperimentSpec) -> None:
    desired = _desired(remote_spec)
    running = _snapshot(desired, state="running", ready=1)
    port = FakePort(
        inspections=[running, None],
        pause_result=running,
    )

    with pytest.raises(EndpointNotFound, match="disappeared"):
        _provisioner(port).pause_and_verify(desired)


def test_pause_times_out_until_zero_ready(remote_spec: ExperimentSpec) -> None:
    desired = _desired(remote_spec)
    running = _snapshot(desired, state="pausing", ready=1)
    port = FakePort(inspections=[running], pause_result=running)

    with pytest.raises(EndpointVerificationTimeout, match="readyReplica=1"):
        _provisioner(port).pause_and_verify(desired, timeout_seconds=2, poll_seconds=1)


def test_delete_is_explicit_exact_and_verified(remote_spec: ExperimentSpec) -> None:
    desired = _desired(remote_spec)
    port = FakePort(inspections=[_snapshot(desired), None])

    assert _provisioner(port).delete(desired)
    assert sum(call.startswith("delete:") for call in port.calls) == 1


def test_ambiguous_delete_adopts_absence(remote_spec: ExperimentSpec) -> None:
    desired = _desired(remote_spec)
    port = FakePort(
        inspections=[_snapshot(desired), None],
        delete_error=AmbiguousEndpointDelete("timeout"),
    )

    assert _provisioner(port).delete(desired)


def test_delete_is_noop_when_exact_identity_is_absent(
    remote_spec: ExperimentSpec,
) -> None:
    port = FakePort(inspections=[None])

    assert not _provisioner(port).delete(_desired(remote_spec))
    assert not any(call.startswith("delete:") for call in port.calls)


def test_delete_refuses_active_endpoint(remote_spec: ExperimentSpec) -> None:
    desired = _desired(remote_spec)
    port = FakePort(inspections=[_snapshot(desired, state="running", ready=1)])

    with pytest.raises(EndpointNotPaused):
        _provisioner(port).delete(desired)

    assert not any(call.startswith("delete:") for call in port.calls)


def test_delete_times_out_while_endpoint_remains(remote_spec: ExperimentSpec) -> None:
    desired = _desired(remote_spec)
    paused = _snapshot(desired)
    port = FakePort(inspections=[paused])

    with pytest.raises(EndpointVerificationTimeout, match="deletion"):
        _provisioner(port).delete(desired, timeout_seconds=2, poll_seconds=1)


def test_managed_identity_model_rejects_forged_name(
    remote_spec: ExperimentSpec,
) -> None:
    identity = _desired(remote_spec).identity
    payload = identity.model_dump(mode="python")
    payload["name"] = "forged"

    with pytest.raises(ValidationError, match="not deterministic"):
        ManagedEndpointIdentity.model_validate(payload)
