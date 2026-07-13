from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from harbor_hf.runs import RunLock

_JOB_ID = re.compile(r"[a-f0-9]{24}")


class TextRunner(Protocol):
    def run_text(self, command: list[str]) -> str: ...


class Submission(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    artifact_prefix: str
    job_id: str | None
    command: list[str]


def github_archive(repository: str, revision: str) -> str:
    normalized = repository.removesuffix(".git").rstrip("/")
    if normalized.startswith("https://github.com/"):
        normalized = normalized.removeprefix("https://github.com/")
    if normalized.count("/") != 1 or ":" in normalized:
        raise ValueError("source repository must be a GitHub owner/name or HTTPS URL")
    return f"https://github.com/{normalized}/archive/{revision}.zip"


def bucket_uri(bucket: str) -> str:
    if bucket.startswith("hf://buckets/"):
        return bucket
    return f"hf://buckets/{bucket.removeprefix('buckets/')}"


def build_submit_command(
    lock: RunLock,
    *,
    input_dir: Path,
    bucket: str,
) -> list[str]:
    job = lock.remote.job
    worker_archive = github_archive(
        lock.remote.worker.repository, lock.remote.worker.revision
    )
    return [
        "hf",
        "jobs",
        "run",
        "--detach",
        "--namespace",
        job.namespace,
        "--flavor",
        job.flavor,
        "--timeout",
        f"{job.timeout_seconds}s",
        "--secrets",
        job.token_secret_name,
        "--label",
        f"harbor-hf-run={lock.run_id}",
        "--volume",
        f"{input_dir}:/input:ro",
        "--volume",
        f"{bucket_uri(bucket)}:/output:rw",
        job.image,
        "uvx",
        "--from",
        worker_archive,
        "harbor-hf",
        "worker",
        "/input/manifest.yaml",
        "/input/run.lock.json",
        "--output-root",
        "/output",
    ]


def submit(
    lock: RunLock,
    *,
    input_dir: Path,
    bucket: str,
    runner: TextRunner,
) -> Submission:
    command = build_submit_command(lock, input_dir=input_dir, bucket=bucket)
    output = runner.run_text(command)
    match = _JOB_ID.search(output)
    if match is None:
        raise ValueError("HF Jobs submission did not return a job ID")
    return Submission(
        run_id=lock.run_id,
        artifact_prefix=lock.artifact_prefix,
        job_id=match.group(),
        command=command,
    )
