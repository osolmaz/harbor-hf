from pathlib import Path

import pytest

from harbor_hf.io import ManifestError, load_experiment
from harbor_hf.models import BenchmarkSpec, RemoteJobSpec

EXAMPLE = Path(__file__).parent.parent / "examples" / "shellbench.yaml"


def test_load_example() -> None:
    spec = load_experiment(EXAMPLE)

    assert spec.metadata.name == "shellbench-qwen-hardware"
    assert spec.matrix.models[0].revision == "0123456789abcdef0123456789abcdef01234567"
    assert spec.matrix.models[0].weights.format == "safetensors"
    assert spec.matrix.models[0].weights.quantization is not None
    assert spec.matrix.models[0].weights.quantization.scheme == "nvfp4"
    assert spec.matrix.agents[0].revision == "replace-with-commit"
    assert spec.artifacts.bucket == "example/benchmark-runs"
    assert spec.publishing.dataset == "example/shellbench-results"


def test_rejects_non_object_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.yaml"
    manifest.write_text("- not\n- an\n- object\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="must contain a YAML object"):
        load_experiment(manifest)


def test_rejects_unknown_fields(tmp_path: Path) -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    manifest = tmp_path / "unknown.yaml"
    manifest.write_text(f"{source}\nunknown: true\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="Extra inputs are not permitted"):
        load_experiment(manifest)


def test_reports_unreadable_path(tmp_path: Path) -> None:
    manifest = tmp_path / "missing.yaml"

    with pytest.raises(ManifestError, match=f"cannot read {manifest}"):
        load_experiment(manifest)


def test_remote_job_timeout_preserves_watchdog_cleanup_margin() -> None:
    with pytest.raises(ValueError, match="less than or equal to 85800"):
        RemoteJobSpec(namespace="org", timeout_seconds=85801)


@pytest.mark.parametrize("task_names", [[], [""], ["same", "same"]])
def test_benchmark_requires_distinct_nonempty_task_names(
    task_names: list[str],
) -> None:
    with pytest.raises(ValueError):
        BenchmarkSpec(dataset="dataset", task_names=task_names)
