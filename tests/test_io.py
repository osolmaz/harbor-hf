from pathlib import Path

import pytest

from harbor_hf.io import ManifestError, load_experiment
from harbor_hf.models import (
    AgentProfile,
    BenchmarkSpec,
    DeploymentProfile,
    EngineSpec,
    ExperimentSpec,
    RemoteJobSpec,
    SourcePin,
)

EXAMPLE = Path(__file__).parent.parent / "examples" / "shellbench.yaml"


def test_load_example() -> None:
    spec = load_experiment(EXAMPLE)

    assert spec.metadata.name == "shellbench-qwen-hardware"
    assert spec.matrix.models[0].revision == "0123456789abcdef0123456789abcdef01234567"
    assert spec.matrix.models[0].weights.format == "safetensors"
    assert spec.matrix.models[0].weights.quantization is not None
    assert spec.matrix.models[0].weights.quantization.scheme == "nvfp4"
    assert spec.matrix.agents[0].revision == "replace-with-package-version"
    assert spec.matrix.agents[0].revision_kind == "package"
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
        RemoteJobSpec(
            namespace="org",
            image="registry/controller@sha256:" + "0" * 64,
            timeout_seconds=85801,
        )


def test_remote_job_timeout_reserves_controller_lifecycle_headroom(
    remote_spec: ExperimentSpec,
) -> None:
    value = remote_spec.model_dump(mode="json")
    remote = value["remote"]
    assert isinstance(remote, dict)
    job = remote["job"]
    assert isinstance(job, dict)
    job["timeout_seconds"] = 4859

    with pytest.raises(ValueError, match="exceed execution timeout by at least 4800"):
        ExperimentSpec.model_validate(value)


def test_remote_job_image_requires_immutable_digest() -> None:
    with pytest.raises(ValueError, match="String should match pattern"):
        RemoteJobSpec(namespace="org", image="registry/controller:latest")


@pytest.mark.parametrize(
    "repository",
    ["org/repo", "https://github.com/org/repo", "https://github.com/org/repo.git"],
)
def test_source_pin_accepts_supported_github_repositories(repository: str) -> None:
    assert SourcePin(repository=repository, revision="0" * 40).repository == repository


@pytest.mark.parametrize(
    "repository",
    ["", "https://gitlab.com/org/repo", "git@github.com:org/repo.git", "org/repo/x"],
)
def test_source_pin_rejects_unsupported_repositories(repository: str) -> None:
    with pytest.raises(ValueError, match="String should match pattern"):
        SourcePin(repository=repository, revision="0" * 40)


@pytest.mark.parametrize("task_names", [[], [""], ["same", "same"]])
def test_benchmark_requires_distinct_nonempty_task_names(
    task_names: list[str],
) -> None:
    with pytest.raises(ValueError):
        BenchmarkSpec(dataset="dataset", task_names=task_names)


def test_agent_revision_metadata_is_explicit() -> None:
    with pytest.raises(ValueError, match="require reported_version"):
        AgentProfile(
            id="agent",
            name="terminus-2",
            revision="commit",
            revision_kind="harbor-source",
        )
    with pytest.raises(ValueError, match="report their package revision"):
        AgentProfile(
            id="agent",
            name="openclaw",
            revision="1.0.0",
            revision_kind="package",
            reported_version="different",
        )


@pytest.mark.parametrize("key", ["", " version", "version ", "version=x"])
def test_agent_parameter_keys_are_unambiguous_for_harbor(key: str) -> None:
    with pytest.raises(ValueError, match="agent parameter keys must not"):
        AgentProfile(
            id="agent",
            name="openclaw",
            revision="1.0.0",
            revision_kind="package",
            parameters={key: "value"},
        )


@pytest.mark.parametrize(
    "key",
    [
        "HF_TOKEN",
        "api-key",
        "PASSWORD",
        "OPENAI_API_KEY",
        "AWS_ACCESS_KEY",
        "SECRET_KEY",
        "FOO_SECRET_KEY",
        "APIKEY",
        "OPENAI_APIKEY",
    ],
)
def test_engine_environment_rejects_inline_secret_values(key: str) -> None:
    with pytest.raises(ValueError, match="must not contain inline secret values"):
        EngineSpec(
            name="vllm",
            image="image",
            environment={key: "credential"},
            secret_names=["HF_TOKEN"],
        )


def test_engine_environment_allows_non_secret_runtime_controls() -> None:
    engine = EngineSpec(
        name="vllm",
        image="image",
        environment={"VLLM_USE_FLASHINFER_MOE_FP4": "1"},
        secret_names=["HF_TOKEN"],
    )

    assert engine.environment == {"VLLM_USE_FLASHINFER_MOE_FP4": "1"}


def test_serialized_parameters_reject_nested_secret_keys(
    remote_spec: ExperimentSpec,
) -> None:
    agent = remote_spec.matrix.agents[0].model_dump(mode="json")
    agent["parameters"] = {"nested": {"api_key": "credential"}}
    with pytest.raises(ValueError, match="agent parameters must not contain"):
        AgentProfile.model_validate(agent)

    deployment = remote_spec.matrix.deployments[0].model_dump(mode="json")
    deployment["parameters"] = {"credentials": {"value": "credential"}}
    with pytest.raises(ValueError, match="deployment parameters must not contain"):
        DeploymentProfile.model_validate(deployment)

    agent["parameters"] = {"nested": [{"password": "credential"}]}
    with pytest.raises(ValueError, match="agent parameters must not contain"):
        AgentProfile.model_validate(agent)

    agent["parameters"] = {"nested": {"openaiApiKey": "credential"}}
    with pytest.raises(ValueError, match="agent parameters must not contain"):
        AgentProfile.model_validate(agent)

    agent["parameters"] = {"nested": {"openAIKey": "credential"}}
    with pytest.raises(ValueError, match="agent parameters must not contain"):
        AgentProfile.model_validate(agent)
