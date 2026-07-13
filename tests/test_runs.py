from datetime import UTC, datetime

import pytest

from harbor_hf.models import ExperimentSpec
from harbor_hf.runs import build_run_lock

NOW = datetime(2026, 7, 13, 1, 2, 3, tzinfo=UTC)


def test_build_run_lock_resolves_one_cell(remote_spec: ExperimentSpec) -> None:
    lock = build_run_lock(remote_spec, clock=lambda: NOW)

    assert lock.run_id == "20260713T010203Z-b650110073"
    assert lock.spec_digest == (
        "sha256:d4b81a15613bd4c10283ecbd95e69168b8dbf5c927a37418e76e389028aeff7f"
    )
    assert lock.artifact_bucket == "example/benchmark-runs"
    assert lock.artifact_prefix == f"runs/{remote_spec.metadata.name}/{lock.run_id}"
    assert lock.deployment.endpoint is not None
    assert lock.deployment.endpoint.name == "qwen-endpoint"
    assert lock.benchmark_tasks == ["cancel-async-tasks"]
    assert lock.created_at == NOW
    assert lock.attempts == 1
    assert lock.concurrent_trials == 1
    assert lock.timeout_seconds == 60


def test_run_id_override_is_preserved(remote_spec: ExperimentSpec) -> None:
    lock = build_run_lock(remote_spec, run_id="manual-run", clock=lambda: NOW)

    assert lock.run_id == "manual-run"


def test_run_id_accepts_hf_label_limit(remote_spec: ExperimentSpec) -> None:
    run_id = "x" * 100

    assert build_run_lock(remote_spec, run_id=run_id).run_id == run_id


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "nested/path", "/absolute", ".", "x" * 101],
)
def test_run_id_override_must_be_a_safe_path_component(
    remote_spec: ExperimentSpec, run_id: str
) -> None:
    with pytest.raises(ValueError, match="safe path component"):
        build_run_lock(remote_spec, run_id=run_id)


def test_submit_requires_remote_configuration(remote_spec: ExperimentSpec) -> None:
    local = remote_spec.model_copy(update={"remote": None})

    with pytest.raises(ValueError, match="requires a remote configuration"):
        build_run_lock(local)


def test_submit_requires_matrix_selection(remote_spec: ExperimentSpec) -> None:
    deployments = [
        remote_spec.matrix.deployments[0],
        remote_spec.matrix.deployments[0].model_copy(update={"id": "second"}),
    ]
    ambiguous = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(update={"deployments": deployments})
        }
    )

    with pytest.raises(ValueError, match="requires --deployment"):
        build_run_lock(ambiguous)
    assert build_run_lock(ambiguous, deployment_id="second").deployment.id == "second"


def test_submit_rejects_unknown_selection(remote_spec: ExperimentSpec) -> None:
    with pytest.raises(ValueError, match="unknown model profile"):
        build_run_lock(remote_spec, model_id="missing")


def test_agent_version_parameter_is_reserved(remote_spec: ExperimentSpec) -> None:
    agent = remote_spec.matrix.agents[0].model_copy(
        update={"parameters": {"version": "different"}}
    )
    invalid = remote_spec.model_copy(
        update={"matrix": remote_spec.matrix.model_copy(update={"agents": [agent]})}
    )

    with pytest.raises(ValueError, match="parameter 'version' is reserved"):
        build_run_lock(invalid)


def test_harbor_source_agent_must_share_harbor_revision(
    remote_spec: ExperimentSpec,
) -> None:
    agent = remote_spec.matrix.agents[0].model_copy(
        update={
            "revision": "0" * 40,
            "revision_kind": "harbor-source",
            "reported_version": "2.0.0",
        }
    )
    invalid = remote_spec.model_copy(
        update={"matrix": remote_spec.matrix.model_copy(update={"agents": [agent]})}
    )

    with pytest.raises(ValueError, match="must match the Harbor source"):
        build_run_lock(invalid)


def test_controller_and_endpoint_must_share_lease_namespace(
    remote_spec: ExperimentSpec,
) -> None:
    deployment = remote_spec.matrix.deployments[0]
    endpoint = deployment.endpoint
    assert endpoint is not None
    mismatched = deployment.model_copy(
        update={"endpoint": endpoint.model_copy(update={"namespace": "other"})}
    )
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={"deployments": [mismatched]}
            )
        }
    )

    with pytest.raises(ValueError, match="namespace must match"):
        build_run_lock(spec)
