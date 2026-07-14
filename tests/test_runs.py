import os
import subprocess
from datetime import UTC, datetime

import pytest

from harbor_hf.models import (
    DeploymentProfile,
    ExperimentSpec,
    GitBenchmarkSource,
    GitHubTokenCredentials,
    MatrixRule,
    _validate_remote_input_pins,
    _validate_task_pins,
)
from harbor_hf.provider_models import ProviderTarget
from harbor_hf.runs import RunLock, build_run_lock, harbor_process_environment

NOW = datetime(2026, 7, 13, 1, 2, 3, tzinfo=UTC)


def test_build_run_lock_resolves_one_cell(remote_spec: ExperimentSpec) -> None:
    lock = build_run_lock(remote_spec, clock=lambda: NOW)
    assert isinstance(lock.deployment, DeploymentProfile)

    assert lock.run_id == "20260713T010203Z-af768c3088"
    assert lock.benchmark_dataset == "harbor/terminal-bench@2.0"
    assert lock.benchmark_dataset_digest == "sha256:" + "1" * 64
    assert lock.spec_digest == (
        "sha256:928a50b654af14b1aec17be91e99911a9160ba6139ae346e2f608f66934c66ba"
    )
    assert lock.artifact_bucket == "example/benchmark-runs"
    assert lock.artifact_prefix == f"runs/{remote_spec.metadata.name}/{lock.run_id}"
    assert lock.deployment.endpoint is not None
    assert lock.deployment.endpoint.name == "qwen-endpoint"
    assert lock.benchmark_tasks == ["cancel-async-tasks"]
    assert lock.benchmark_task_digests == {"cancel-async-tasks": "sha256:" + "2" * 64}
    assert lock.created_at == NOW
    assert lock.attempts == 1
    assert lock.concurrent_trials == 1
    assert lock.timeout_seconds == 60
    assert lock.schema_version == "harbor-hf/run-lock/v1alpha1"


def test_build_run_lock_preserves_git_benchmark_source(
    remote_spec: ExperimentSpec,
) -> None:
    source = GitBenchmarkSource(
        repository="ShellBench/public-tasks",
        revision="8" * 40,
        path="tasks/115-tasks",
    )
    benchmark = remote_spec.benchmark.model_copy(
        update={"dataset": "shellbench/public-115", "source": source}
    )
    raw = remote_spec.model_dump(mode="python")
    raw["benchmark"] = benchmark.model_dump(mode="python", exclude={"dataset_digest"})
    spec = ExperimentSpec.model_validate(raw)

    lock = build_run_lock(spec)

    assert lock.benchmark_source == source
    assert lock.benchmark_dataset_digest == spec.benchmark.dataset_digest
    assert lock.schema_version == "harbor-hf/run-lock/v1alpha2"


def test_authenticated_git_source_uses_v1alpha3_lock(
    remote_spec: ExperimentSpec,
) -> None:
    source = GitBenchmarkSource(
        repository="ShellBench/public-tasks",
        revision="8" * 40,
        path="tasks/115-tasks",
        credentials=GitHubTokenCredentials(secret_name="GITHUB_TOKEN"),
    )
    raw = remote_spec.model_dump(mode="python")
    raw["benchmark"].update(
        {
            "dataset": "shellbench/public-115",
            "source": source.model_dump(mode="python"),
        }
    )
    raw["benchmark"].pop("dataset_digest", None)
    spec = ExperimentSpec.model_validate(raw)

    lock = build_run_lock(spec)

    assert lock.benchmark_source == source
    assert lock.schema_version == "harbor-hf/run-lock/v1alpha3"

    legacy = lock.model_dump(mode="json")
    legacy["schema_version"] = "harbor-hf/run-lock/v1alpha2"
    with pytest.raises(ValueError, match="authenticated source fields"):
        RunLock.model_validate(legacy)


def test_authenticated_git_environment_uses_scoped_helper_and_redacted_secret(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = GitBenchmarkSource(
        repository="ShellBench/public-tasks",
        revision="8" * 40,
        path="tasks/115-tasks",
        credentials=GitHubTokenCredentials(secret_name="GITHUB_TOKEN"),
    )
    raw = remote_spec.model_dump(mode="python")
    raw["benchmark"].update(
        {
            "dataset": "shellbench/public-115",
            "source": source.model_dump(mode="python"),
        }
    )
    raw["benchmark"].pop("dataset_digest", None)
    lock = build_run_lock(ExperimentSpec.model_validate(raw))
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")

    environment = harbor_process_environment(
        lock, token="hf-secret", inference_base_url="https://endpoint.example"
    )

    assert environment["GITHUB_TOKEN"] == "github-secret"
    assert environment["GIT_CONFIG_COUNT"] == "2"
    assert environment["GIT_CONFIG_KEY_0"] == "credential.useHttpPath"
    assert environment["GIT_CONFIG_VALUE_0"] == "true"
    assert environment["GIT_CONFIG_KEY_1"] == (
        "credential.https://github.com/ShellBench/public-tasks.git.helper"
    )
    assert environment["GIT_CONFIG_VALUE_1"] == "harbor-hf"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["HARBOR_HF_GIT_CREDENTIAL_ENV"] == "GITHUB_TOKEN"
    assert environment["HARBOR_HF_GIT_REPOSITORY"] == "ShellBench/public-tasks"

    monkeypatch.delenv("GITHUB_TOKEN")
    with pytest.raises(ValueError, match="required secret GITHUB_TOKEN"):
        harbor_process_environment(
            lock, token="hf-secret", inference_base_url="https://endpoint.example"
        )


def test_authenticated_git_environment_is_scoped_by_real_git(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = GitBenchmarkSource(
        repository="ShellBench/public-tasks",
        revision="8" * 40,
        path="tasks/115-tasks",
        credentials=GitHubTokenCredentials(secret_name="GITHUB_TOKEN"),
    )
    raw = remote_spec.model_dump(mode="python")
    raw["benchmark"].update(
        {
            "dataset": "shellbench/public-115",
            "source": source.model_dump(mode="python"),
        }
    )
    raw["benchmark"].pop("dataset_digest", None)
    lock = build_run_lock(ExperimentSpec.model_validate(raw))
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    environment = os.environ.copy()
    environment.update(
        harbor_process_environment(
            lock, token="hf-secret", inference_base_url="https://endpoint.example"
        )
    )
    environment.update(
        {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    )

    allowed = subprocess.run(
        ["git", "credential", "fill"],
        input=("protocol=https\nhost=github.com\npath=ShellBench/public-tasks.git\n\n"),
        text=True,
        capture_output=True,
        env=environment,
        check=True,
    )
    refused = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\npath=other/repo.git\n\n",
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )

    assert "username=x-access-token" in allowed.stdout
    assert "password=github-secret" in allowed.stdout
    assert refused.returncode != 0
    assert "github-secret" not in refused.stdout + refused.stderr


def test_run_lock_preserves_and_renders_hosted_judge(
    remote_spec: ExperimentSpec,
) -> None:
    raw = remote_spec.model_dump(mode="python")
    raw["benchmark"]["judge"] = {
        "api_url": "https://router.huggingface.co/v1/chat/completions",
        "model": "deepseek-ai/DeepSeek-V3.2",
    }
    spec = ExperimentSpec.model_validate(raw)

    lock = build_run_lock(spec, run_id="judge-lock")
    environment = harbor_process_environment(
        lock, token="secret-token", inference_base_url="https://endpoint.example/"
    )

    assert lock.benchmark_judge == spec.benchmark.judge
    assert lock.schema_version == "harbor-hf/run-lock/v1alpha2"
    assert environment == {
        "AGENT_JUDGE_API_KEY": "secret-token",
        "AGENT_JUDGE_API_URL": "https://router.huggingface.co/v1/chat/completions",
        "AGENT_JUDGE_MODEL": "deepseek-ai/DeepSeek-V3.2",
        "HF_TOKEN": "secret-token",
        "OPENAI_API_KEY": "secret-token",
        "OPENAI_BASE_URL": "https://endpoint.example/v1",
    }


def test_run_lock_reader_accepts_legacy_v1alpha1(remote_spec: ExperimentSpec) -> None:
    payload = build_run_lock(remote_spec, run_id="legacy-lock").model_dump(mode="json")
    payload.pop("benchmark_source", None)
    payload.pop("benchmark_judge", None)

    lock = RunLock.model_validate(payload)

    assert lock.schema_version == "harbor-hf/run-lock/v1alpha1"
    assert lock.benchmark_source is None
    assert lock.benchmark_judge is None


def test_run_lock_v1alpha1_rejects_new_fields(remote_spec: ExperimentSpec) -> None:
    payload = build_run_lock(remote_spec, run_id="legacy-lock").model_dump(mode="json")
    payload["benchmark_source"] = {
        "repository": "ShellBench/public-tasks",
        "revision": "8" * 40,
        "path": "tasks/115-tasks",
    }

    with pytest.raises(ValueError, match="v1alpha1 cannot contain"):
        RunLock.model_validate(payload)


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


def test_submit_rejects_cell_excluded_by_matrix_rules(
    remote_spec: ExperimentSpec,
) -> None:
    excluded = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={
                    "exclude": [MatrixRule(models=[remote_spec.matrix.models[0].id])]
                }
            )
        }
    )

    with pytest.raises(ValueError, match="exclude every run cell"):
        build_run_lock(excluded)


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
    assert isinstance(deployment, DeploymentProfile)
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


def test_provider_target_rejects_unsupported_agent_before_submission(
    remote_spec: ExperimentSpec,
) -> None:
    model = remote_spec.matrix.models[0]
    provider = ProviderTarget(id="provider-one", model=model.repo)
    agent = remote_spec.matrix.agents[0].model_copy(update={"name": "terminus"})
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={"deployments": [provider], "agents": [agent]}
            )
        }
    )

    with pytest.raises(ValueError, match="require the OpenClaw Harbor agent"):
        build_run_lock(spec, allow_provider=True)


def test_remote_lock_rejects_mutable_model_revision(
    remote_spec: ExperimentSpec,
) -> None:
    model = remote_spec.matrix.models[0].model_copy(update={"revision": "main"})
    spec = remote_spec.model_copy(
        update={"matrix": remote_spec.matrix.model_copy(update={"models": [model]})}
    )

    with pytest.raises(ValueError) as captured:
        _validate_remote_input_pins(spec)
    assert str(captured.value) == "remote model revisions must be full Git commit IDs"
    with pytest.raises(ValueError, match="model revisions must be full Git commit IDs"):
        build_run_lock(spec)


def test_remote_lock_rejects_mutable_serving_image(remote_spec: ExperimentSpec) -> None:
    deployment = remote_spec.matrix.deployments[0]
    assert isinstance(deployment, DeploymentProfile)
    engine = deployment.engine.model_copy(update={"image": "example/vllm:latest"})
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={
                    "deployments": [deployment.model_copy(update={"engine": engine})]
                }
            )
        }
    )

    with pytest.raises(ValueError) as captured:
        _validate_remote_input_pins(spec)
    assert (
        str(captured.value) == "remote serving images must be pinned by sha256 digest"
    )


def test_remote_lock_rejects_mutable_agent_revision(
    remote_spec: ExperimentSpec,
) -> None:
    agent = remote_spec.matrix.agents[0].model_copy(update={"revision": "latest"})
    spec = remote_spec.model_copy(
        update={"matrix": remote_spec.matrix.model_copy(update={"agents": [agent]})}
    )

    with pytest.raises(ValueError) as captured:
        _validate_remote_input_pins(spec)
    assert str(captured.value) == "remote agent revisions must be immutable"


def test_remote_lock_rejects_unresolved_benchmark(remote_spec: ExperimentSpec) -> None:
    benchmark = remote_spec.benchmark.model_copy(update={"dataset_digest": None})
    spec = remote_spec.model_copy(update={"benchmark": benchmark})
    with pytest.raises(ValueError) as captured:
        _validate_remote_input_pins(spec)
    assert str(captured.value) == (
        "remote benchmark dataset requires an immutable sha256 digest"
    )

    benchmark = remote_spec.benchmark.model_copy(update={"task_digests": {}})
    with pytest.raises(ValueError) as captured:
        _validate_task_pins(benchmark)
    assert str(captured.value) == "remote benchmarks require resolved task digests"


def test_remote_lock_rejects_legacy_dataset_that_cannot_use_digest(
    remote_spec: ExperimentSpec,
) -> None:
    benchmark = remote_spec.benchmark.model_copy(
        update={"dataset": "terminal-bench@2.0"}
    )
    spec = remote_spec.model_copy(update={"benchmark": benchmark})

    with pytest.raises(ValueError) as captured:
        _validate_remote_input_pins(spec)

    assert str(captured.value) == (
        "remote benchmark dataset must use a Harbor package name in org/name form"
    )


def test_content_addressed_dataset_infers_and_preserves_digest(
    remote_spec: ExperimentSpec,
) -> None:
    digest = "sha256:" + "4" * 64
    raw = remote_spec.model_dump(mode="python")
    raw["benchmark"]["dataset"] = f"harbor/terminal-bench@{digest}"
    raw["benchmark"].pop("dataset_digest")

    spec = ExperimentSpec.model_validate(raw)
    lock = build_run_lock(spec)

    assert spec.benchmark.dataset_digest == digest
    assert lock.benchmark_dataset == f"harbor/terminal-bench@{digest}"
    assert lock.benchmark_dataset_digest == digest


def test_content_addressed_dataset_rejects_conflicting_digest(
    remote_spec: ExperimentSpec,
) -> None:
    raw = remote_spec.model_dump(mode="python")
    raw["benchmark"]["dataset"] = "harbor/terminal-bench@sha256:" + "4" * 64
    raw["benchmark"]["dataset_digest"] = "sha256:" + "5" * 64

    with pytest.raises(
        ValueError,
        match="dataset digest must match its content-addressed reference",
    ):
        ExperimentSpec.model_validate(raw)


@pytest.mark.parametrize(
    ("task_names", "task_digests"),
    [
        (
            ["cancel-async-tasks", "missing-*"],
            {"cancel-async-tasks": "sha256:" + "2" * 64},
        ),
        (
            ["cancel-async-tasks"],
            {
                "cancel-async-tasks": "sha256:" + "2" * 64,
                "unexpected-task": "sha256:" + "3" * 64,
            },
        ),
    ],
)
def test_remote_task_pins_reject_each_inexact_selection_direction(
    remote_spec: ExperimentSpec,
    task_names: list[str],
    task_digests: dict[str, str],
) -> None:
    benchmark = remote_spec.benchmark.model_copy(
        update={"task_names": task_names, "task_digests": task_digests}
    )

    with pytest.raises(ValueError) as captured:
        _validate_task_pins(benchmark)
    assert str(captured.value) == (
        "remote task digests must exactly resolve the task selection"
    )
