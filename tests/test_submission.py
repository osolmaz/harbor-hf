from pathlib import Path

import pytest

from harbor_hf.models import ExperimentSpec
from harbor_hf.runs import build_run_lock
from harbor_hf.submission import (
    bucket_uri,
    build_submit_command,
    endpoint_lease_label,
    endpoint_lease_label_for,
    github_archive,
    github_repository,
    locked_source_command,
    submit,
)


class FakeRunner:
    def __init__(self, output: str) -> None:
        self.output = output
        self.command: list[str] | None = None

    def run_text(self, command: list[str]) -> str:
        self.command = command
        return self.output


def test_build_submit_command_contains_only_secret_name(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = build_run_lock(remote_spec, run_id="run-1")

    command = build_submit_command(
        lock, input_dir=tmp_path, bucket="osolmaz/benchmark-runs"
    )

    assert command[:22] == [
        "hf",
        "jobs",
        "run",
        "--detach",
        "--namespace",
        "osolmaz",
        "--flavor",
        "cpu-basic",
        "--timeout",
        "10800s",
        "--secrets",
        "HF_TOKEN",
        "--label",
        "harbor-hf-run=run-1",
        "--label",
        f"harbor-hf-endpoint={endpoint_lease_label(lock)}",
        "--volume",
        f"{tmp_path}:/input:ro",
        "--volume",
        "hf://buckets/osolmaz/benchmark-runs:/output:rw",
        "--",
        "ghcr.io/astral-sh/uv:python3.12-bookworm",
    ]
    assert command[22:25] == ["bash", "-lc", command[24]]
    assert "git clone --filter=blob:none --no-checkout" in command[24]
    assert "uv run --project" in command[24]
    assert "--locked" in command[24]
    assert command[25:] == [
        "locked-source",
        "harbor-hf",
        "worker",
        "/input/manifest.yaml",
        "/input/run.lock.json",
        "--output-root",
        "/output",
    ]
    assert "super-secret" not in " ".join(command)


def test_endpoint_lease_label_is_stable_and_bounded(
    remote_spec: ExperimentSpec,
) -> None:
    label = endpoint_lease_label(build_run_lock(remote_spec))

    assert label == "d026b68a5286b3887f1e9ea13d304aed"
    assert len(label) == 32


def test_endpoint_lease_label_uses_complete_endpoint_identity() -> None:
    assert endpoint_lease_label_for("org", "endpoint") == (
        "aa3808503c913daab53ed1415fe04988"
    )
    assert endpoint_lease_label_for("org", "other") != endpoint_lease_label_for(
        "other", "org"
    )


def test_endpoint_lease_label_requires_endpoint_binding(
    remote_spec: ExperimentSpec,
) -> None:
    lock = build_run_lock(remote_spec)
    lock = lock.model_copy(
        update={"deployment": lock.deployment.model_copy(update={"endpoint": None})}
    )

    with pytest.raises(ValueError, match="^run lock has no endpoint binding$"):
        endpoint_lease_label(lock)


def test_submit_parses_job_id(remote_spec: ExperimentSpec, tmp_path: Path) -> None:
    lock = build_run_lock(remote_spec, run_id="run-1")
    runner = FakeRunner("Job started: 0123456789abcdef01234567\n")

    result = submit(
        lock,
        input_dir=tmp_path,
        bucket="osolmaz/benchmark-runs",
        runner=runner,
    )

    assert result.job_id == "0123456789abcdef01234567"
    assert runner.command is not None


def test_submit_rejects_missing_job_id(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = build_run_lock(remote_spec)

    with pytest.raises(ValueError, match="did not return a job ID"):
        submit(
            lock,
            input_dir=tmp_path,
            bucket="osolmaz/benchmark-runs",
            runner=FakeRunner("submitted"),
        )


def test_source_and_bucket_normalization() -> None:
    assert github_archive("https://github.com/org/repo.git", "abcdef0") == (
        "https://github.com/org/repo/archive/abcdef0.zip"
    )
    assert bucket_uri("hf://buckets/org/name") == "hf://buckets/org/name"
    assert bucket_uri("buckets/org/name") == "hf://buckets/org/name"
    assert bucket_uri("org/name") == "hf://buckets/org/name"
    assert github_archive("org/repo", "abcdef0") == (
        "https://github.com/org/repo/archive/abcdef0.zip"
    )
    assert github_repository("org/repo") == "https://github.com/org/repo"
    assert github_repository("https://github.com/org/repo.git") == (
        "https://github.com/org/repo"
    )
    with pytest.raises(ValueError, match="GitHub"):
        github_archive("https://example.com/repo", "abcdef0")
    with pytest.raises(ValueError, match="GitHub"):
        github_archive("org/repo/extra", "abcdef0")
    with pytest.raises(ValueError, match="GitHub"):
        github_repository("git@github.com:org/repo.git")


def test_locked_source_command_passes_arguments_after_shell_script(
    remote_spec: ExperimentSpec,
) -> None:
    remote = remote_spec.remote
    assert remote is not None

    command = locked_source_command(remote.worker, "harbor-hf", "worker", "a b")

    assert command == [
        "bash",
        "-lc",
        "set -euo pipefail\n"
        "repo_dir=$(mktemp -d)\n"
        "git clone --filter=blob:none --no-checkout "
        'https://github.com/osolmaz/harbor-hf "$repo_dir"\n'
        'git -C "$repo_dir" fetch --depth 1 origin '
        "1234567890abcdef1234567890abcdef12345678\n"
        'git -C "$repo_dir" checkout --detach '
        "1234567890abcdef1234567890abcdef12345678\n"
        'exec uv run --project "$repo_dir" --locked --no-dev "$@"\n',
        "locked-source",
        "harbor-hf",
        "worker",
        "a b",
    ]
