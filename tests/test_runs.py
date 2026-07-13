from datetime import UTC, datetime

import pytest

from harbor_hf.models import ExperimentSpec
from harbor_hf.runs import build_run_lock

NOW = datetime(2026, 7, 13, 1, 2, 3, tzinfo=UTC)


def test_build_run_lock_resolves_one_cell(remote_spec: ExperimentSpec) -> None:
    lock = build_run_lock(remote_spec, clock=lambda: NOW)

    assert lock.run_id == "20260713T010203Z-c0ef9d7c35"
    assert lock.spec_digest == (
        "sha256:0a82012432e46a47359443164dce9b7777526b44a76bdb6161f572a5a530ab3b"
    )
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
