import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import yaml

from harbor_hf.harbor_adapter.exporter import classify_private_artifact
from harbor_hf.io import load_experiment
from harbor_hf.models import EndpointRef, ExperimentSpec, RemoteExecutionSpec

EXAMPLE = Path(__file__).parent.parent / "examples" / "shellbench.yaml"


def write_fake_compatibility_bundle(command: Sequence[str], log_path: Path) -> None:
    jobs_dir = Path(command[command.index("--jobs-dir") + 1])
    output = Path(command[command.index("--output") + 1])

    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    trials: list[dict[str, object]] = []
    for result_path in sorted(jobs_dir.glob("*/*/result.json")):
        trial_dir = result_path.parent
        lock_path = trial_dir / "lock.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        agent = result["agent_info"]
        model = agent.get("model_info") or {}
        exception = result.get("exception_info")
        step_exceptions = [
            {
                "step_name": step.get("step_name") or str(index),
                "exception_type": step["exception_info"].get(
                    "exception_type", "malformed result"
                ),
            }
            for index, step in enumerate(result.get("step_results") or [], start=1)
            if isinstance(step, dict) and isinstance(step.get("exception_info"), dict)
        ]
        verifier = result.get("verifier_result") or {}
        artifacts = [
            {
                "path": path.relative_to(trial_dir).as_posix(),
                "size": path.stat().st_size,
                "digest": digest(path),
                "kind": classify_private_artifact(
                    path.relative_to(trial_dir).as_posix()
                ),
                "classification": "private",
            }
            for path in sorted(trial_dir.rglob("*"))
            if path.is_file()
        ]
        trials.append(
            {
                "path": trial_dir.relative_to(jobs_dir).as_posix(),
                "trial_id": str(
                    result.get("id", "00000000-0000-0000-0000-000000000001")
                ),
                "trial_name": str(result.get("trial_name", trial_dir.name)),
                "lock_digest": digest(lock_path),
                "result_digest": digest(result_path),
                "task_name": result["task_name"],
                "task_digest": lock["task"]["digest"],
                "agent_name": agent["name"],
                "agent_version": agent["version"],
                "model_provider": model.get("provider"),
                "model_name": model.get("name"),
                "exception_type": (
                    exception.get("exception_type")
                    if isinstance(exception, dict)
                    else None
                ),
                "step_exceptions": step_exceptions,
                "rewards": verifier.get("rewards"),
                "timing": {
                    "trial": {"started_at": None, "finished_at": None},
                    "environment_setup": None,
                    "agent_setup": None,
                    "agent_execution": None,
                    "verifier": None,
                    "steps": [],
                },
                "usage": {
                    "input_tokens": None,
                    "cache_tokens": None,
                    "output_tokens": None,
                    "cost_usd": None,
                },
                "artifacts": artifacts,
            }
        )
    bundle = {
        "schema_version": "harbor-hf/harbor-compatibility/v1alpha2",
        "harbor_revision": command[command.index("--harbor-revision") + 1],
        "harbor_version": "test",
        "request_digest": command[command.index("--request-digest") + 1],
        "jobs": [],
        "trials": trials,
    }
    output.write_text(json.dumps(bundle), encoding="utf-8")
    log_path.write_text("exported\n", encoding="utf-8")


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
