from pathlib import Path

import pytest

from harbor_hf.models import ExperimentSpec
from harbor_hf.runs import build_run_lock
from harbor_hf.submission import (
    bucket_uri,
    build_submit_command,
    github_archive,
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

    assert command == [
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
        "ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
        "uvx",
        "--from",
        "https://github.com/osolmaz/harbor-hf/archive/"
        "1234567890abcdef1234567890abcdef12345678.zip",
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
    with pytest.raises(ValueError, match="GitHub"):
        github_archive("https://example.com/repo", "abcdef0")
    with pytest.raises(ValueError, match="GitHub"):
        github_archive("org/repo/extra", "abcdef0")
