from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from harbor_hf.io import load_experiment
from harbor_hf.models import EndpointRef, ExperimentSpec, RemoteExecutionSpec

EXAMPLE = Path(__file__).parent.parent / "examples" / "shellbench.yaml"


class _WaveClaims:
    def acquire(self, path: str, owner: Mapping[str, str]) -> None:
        del path, owner

    def release(self, path: str, owner: Mapping[str, str]) -> None:
        del path, owner


@pytest.fixture(autouse=True)
def wave_worker_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_ID", "test-wave-job")
    monkeypatch.setattr(
        "harbor_hf.wave_worker.HubClaimStore",
        lambda *_args, **_kwargs: _WaveClaims(),
    )


@pytest.fixture
def remote_spec() -> ExperimentSpec:
    spec = load_experiment(EXAMPLE)
    deployment = spec.matrix.deployments[0].model_copy(
        update={
            "endpoint": EndpointRef(
                namespace="osolmaz",
                name="qwen-endpoint",
                served_model_name="/repository",
            )
        }
    )
    return spec.model_copy(
        update={
            "benchmark": spec.benchmark.model_copy(
                update={
                    "dataset": "harbor/terminal-bench@2.0",
                    "dataset_digest": "sha256:" + "1" * 64,
                    "task_names": ["cancel-async-tasks"],
                    "task_digests": {"cancel-async-tasks": "sha256:" + "2" * 64},
                }
            ),
            "matrix": spec.matrix.model_copy(update={"deployments": [deployment]}),
            "execution": spec.execution.model_copy(
                update={"concurrent_trials": 1, "timeout_seconds": 60}
            ),
            "remote": RemoteExecutionSpec.model_validate(
                {
                    "job": {
                        "namespace": "osolmaz",
                        "image": "ghcr.io/astral-sh/uv@sha256:" + "0" * 64,
                    },
                    "worker": {
                        "repository": "osolmaz/harbor-hf",
                        "revision": "1234567890abcdef1234567890abcdef12345678",
                    },
                    "harbor": {
                        "source": {
                            "repository": "harbor-framework/harbor",
                            "revision": "abcdef1234567890abcdef1234567890abcdef12",
                        }
                    },
                }
            ),
        }
    )


@pytest.fixture
def remote_manifest(tmp_path: Path, remote_spec: ExperimentSpec) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(remote_spec.model_dump(mode="json", exclude_none=True)),
        encoding="utf-8",
    )
    return path
