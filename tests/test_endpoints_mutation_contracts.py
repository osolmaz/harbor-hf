"""Exact behavioral contracts for endpoint models and provisioning."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from test_endpoints import FakePort, FakeTime, _desired, _snapshot

from harbor_hf.endpoints import (
    EndpointConfigurationMismatch,
    EndpointIdentityMismatch,
    EndpointNotFound,
    EndpointNotPaused,
    EndpointProvisioner,
    EndpointScaling,
    EndpointSnapshot,
    EndpointVerificationTimeout,
    build_desired_endpoint,
    deployment_digest,
    require_paused_zero_ready,
    verify_exact_endpoint,
)
from harbor_hf.models import DeploymentProfile, ExperimentSpec


def _clocked_provisioner(port: FakePort) -> EndpointProvisioner:
    clock = FakeTime()
    return EndpointProvisioner(port, sleep=clock.sleep, monotonic=clock.monotonic)


def test_scaling_validator_enforces_replica_and_metric_pairing() -> None:
    with pytest.raises(ValidationError, match="at least minimum replicas"):
        EndpointScaling(min_replicas=2, max_replicas=1)
    with pytest.raises(ValidationError, match="configured together"):
        EndpointScaling(min_replicas=0, max_replicas=1, metric="pendingRequests")
    with pytest.raises(ValidationError, match="configured together"):
        EndpointScaling(min_replicas=0, max_replicas=1, threshold=1.5)
    assert EndpointScaling(min_replicas=1, max_replicas=1).metric is None
    scaling = EndpointScaling(
        min_replicas=1, max_replicas=1, metric="hardwareUsage", threshold=0.5
    )
    assert scaling.threshold == 0.5


def test_verify_rejects_namespace_and_name_mismatches(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    matching = _snapshot(desired)

    foreign_namespace = EndpointSnapshot(
        namespace="other",
        name=matching.name,
        configuration=matching.configuration,
        status=matching.status,
    )
    with pytest.raises(EndpointIdentityMismatch):
        verify_exact_endpoint(desired, foreign_namespace)

    foreign_name = EndpointSnapshot(
        namespace=matching.namespace,
        name="harbor-hf-" + "0" * 40,
        configuration=matching.configuration,
        status=matching.status,
    )
    with pytest.raises(EndpointIdentityMismatch):
        verify_exact_endpoint(desired, foreign_name)

    verify_exact_endpoint(desired, matching)


def test_not_paused_error_reports_exact_observation(
    remote_spec: ExperimentSpec,
) -> None:
    snapshot = _snapshot(_desired(remote_spec), state="running", ready=2, target=3)

    with pytest.raises(EndpointNotPaused) as caught:
        require_paused_zero_ready(snapshot)

    assert str(caught.value) == (
        "endpoint must report state=paused and readyReplica=0; "
        "observed state='running', readyReplica=2, targetReplica=3"
    )


def test_configuration_mismatch_message_joins_paths(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    observed = desired.configuration.model_copy(
        update={"access_type": "private", "cache_http_responses": False}
    )

    with pytest.raises(EndpointConfigurationMismatch) as caught:
        verify_exact_endpoint(desired, _snapshot(desired, configuration=observed))

    assert str(caught.value) == (
        "endpoint effective configuration mismatch: "
        "configuration.access_type, configuration.cache_http_responses"
    )
    first = caught.value.mismatches[0]
    assert first.expected == '"authenticated"'
    assert first.observed == '"private"'


def test_pause_and_verify_skips_pause_when_already_paused(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    paused = _snapshot(desired)
    port = FakePort(inspections=[paused])

    result = _clocked_provisioner(port).pause_and_verify(desired)

    assert result == paused
    assert not any(call.startswith("pause:") for call in port.calls)


def test_wait_loops_reject_nonpositive_timeouts(remote_spec: ExperimentSpec) -> None:
    desired = _desired(remote_spec)
    pausing = _snapshot(desired, state="pausing", ready=1)
    port = FakePort(inspections=[pausing], pause_result=pausing)

    with pytest.raises(ValueError, match="must be positive"):
        _clocked_provisioner(port).pause_and_verify(
            desired, timeout_seconds=0, poll_seconds=1
        )
    with pytest.raises(ValueError, match="must be positive"):
        _clocked_provisioner(port).pause_and_verify(
            desired, timeout_seconds=10, poll_seconds=0
        )


def test_pause_polls_exactly_until_deadline(remote_spec: ExperimentSpec) -> None:
    desired = _desired(remote_spec)
    pausing = _snapshot(desired, state="pausing", ready=1)
    port = FakePort(inspections=[pausing], pause_result=pausing)

    with pytest.raises(EndpointVerificationTimeout):
        _clocked_provisioner(port).pause_and_verify(
            desired, timeout_seconds=2.5, poll_seconds=1
        )

    assert sum(call.startswith("inspect:") for call in port.calls) == 5


def test_cleanup_of_mismatched_create_rejects_disappearance(
    remote_spec: ExperimentSpec,
) -> None:
    desired = _desired(remote_spec)
    mismatched = desired.configuration.model_copy(update={"access_type": "private"})
    running = _snapshot(desired, state="running", ready=1, configuration=mismatched)
    port = FakePort(
        inspections=[None, None],
        create_result=running,
        pause_result=running,
    )

    with pytest.raises(EndpointNotFound, match="disappeared during cleanup"):
        _clocked_provisioner(port).create_or_adopt(desired)


def test_cleanup_of_mismatched_create_times_out(remote_spec: ExperimentSpec) -> None:
    desired = _desired(remote_spec)
    mismatched = desired.configuration.model_copy(update={"access_type": "private"})
    running = _snapshot(desired, state="running", ready=1, configuration=mismatched)
    port = FakePort(
        inspections=[None, running],
        create_result=running,
        pause_result=running,
    )

    with pytest.raises(EndpointVerificationTimeout, match="cleanup"):
        _clocked_provisioner(port).create_or_adopt(
            desired, timeout_seconds=2, poll_seconds=1
        )


def test_deployment_digest_ignores_model_id(remote_spec: ExperimentSpec) -> None:
    model = remote_spec.matrix.models[0]
    deployment = remote_spec.matrix.deployments[0]
    renamed_model = model.model_copy(update={"id": "renamed-model"})

    assert deployment_digest(model, deployment) == deployment_digest(
        renamed_model, deployment
    )
    changed = model.model_copy(update={"revision": "f" * 40})
    assert deployment_digest(model, deployment) != deployment_digest(
        changed, deployment
    )


def test_build_desired_endpoint_applies_documented_defaults(
    remote_spec: ExperimentSpec,
) -> None:
    deployment = remote_spec.matrix.deployments[0]
    assert isinstance(deployment, DeploymentProfile)
    deployment = deployment.model_copy(update={"parameters": {}})

    desired = build_desired_endpoint(
        namespace="osolmaz",
        campaign_id="campaign-one",
        model=remote_spec.matrix.models[0],
        deployment=deployment,
    )

    configuration = desired.configuration
    assert configuration.model.framework == "custom"
    assert configuration.model.task == "text-generation"
    assert configuration.model.image.health_route == "/health"
    assert configuration.model.image.port is None
    assert configuration.compute.accelerator == "gpu"
    assert configuration.compute.scaling.min_replicas == 1
    assert configuration.compute.scaling.max_replicas == 1
    assert configuration.access_type == "authenticated"
    assert configuration.cache_http_responses is False
    assert configuration.route.domain is None
    assert configuration.route.path is None
    assert configuration.tags == desired.identity.tags
