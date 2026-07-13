from pathlib import Path

import pytest

from harbor_hf.models import ExperimentSpec
from harbor_hf.runs import build_run_lock
from harbor_hf.submission import (
    bucket_uri,
    build_submit_command,
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

    assert command[:20] == [
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
        "--volume",
        f"{tmp_path}:/input:ro",
        "--volume",
        "hf://buckets/osolmaz/benchmark-runs:/output:rw",
        "--",
        "ghcr.io/astral-sh/uv:python3.12-bookworm",
    ]
    assert command[20:23] == ["bash", "-lc", command[22]]
    assert "git clone --filter=blob:none --no-checkout" in command[22]
    assert "uv run --project" in command[22]
    assert "--locked" in command[22]
    assert command[23:] == [
        "locked-source",
        "harbor-hf",
        "worker",
        "/input/manifest.yaml",
        "/input/run.lock.json",
        "--output-root",
        "/output",
    ]
    assert "super-secret" not in " ".join(command)


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
        'exec uv run --project "$repo_dir" --locked "$@"\n',
        "locked-source",
        "harbor-hf",
        "worker",
        "a b",
    ]
