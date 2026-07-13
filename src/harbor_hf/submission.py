from __future__ import annotations

import hashlib
import re
import shlex
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from harbor_hf.models import SourcePin
from harbor_hf.runs import RunLock

_JOB_ID = re.compile(r"[a-f0-9]{24}")
_GITHUB_REPOSITORY = re.compile(
    r"^(?:https://github\.com/)?"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)


class TextRunner(Protocol):
    def run_text(self, command: list[str]) -> str: ...


class Submission(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    artifact_prefix: str
    job_id: str | None
    command: list[str]


def github_archive(repository: str, revision: str) -> str:
    return f"{github_repository(repository)}/archive/{revision}.zip"


def github_repository(repository: str) -> str:
    match = _GITHUB_REPOSITORY.fullmatch(repository)
    if match is None:
        raise ValueError("source repository must be a GitHub owner/name or HTTPS URL")
    return f"https://github.com/{match['owner']}/{match['name']}"


def locked_source_command(source: SourcePin, *arguments: str) -> list[str]:
    repository = shlex.quote(github_repository(source.repository))
    revision = shlex.quote(source.revision)
    script = (
        "set -euo pipefail\n"
        "repo_dir=$(mktemp -d)\n"
        f'git clone --filter=blob:none --no-checkout {repository} "$repo_dir"\n'
        f'git -C "$repo_dir" fetch --depth 1 origin {revision}\n'
        f'git -C "$repo_dir" checkout --detach {revision}\n'
        'exec uv run --project "$repo_dir" --locked --no-dev "$@"\n'
    )
    return ["bash", "-lc", script, "locked-source", *arguments]


def bucket_uri(bucket: str) -> str:
    if bucket.startswith("hf://buckets/"):
        return bucket
    return f"hf://buckets/{bucket.removeprefix('buckets/')}"


def endpoint_lease_label(lock: RunLock) -> str:
    endpoint = lock.deployment.endpoint
    if endpoint is None:
        raise ValueError("run lock has no endpoint binding")
    return endpoint_lease_label_for(endpoint.namespace, endpoint.name)


def endpoint_lease_label_for(namespace: str, name: str) -> str:
    identity = f"{namespace}/{name}".encode()
    return hashlib.sha256(identity).hexdigest()[:32]


def build_submit_command(
    lock: RunLock,
    *,
    input_dir: Path,
    bucket: str,
) -> list[str]:
    job = lock.remote.job
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
        "--label",
        f"harbor-hf-endpoint={endpoint_lease_label(lock)}",
        "--volume",
        f"{input_dir}:/input:ro",
        "--volume",
        f"{bucket_uri(bucket)}:/output:rw",
        "--",
        job.image,
        *locked_source_command(
            lock.remote.worker,
            "harbor-hf",
            "worker",
            "/input/manifest.yaml",
            "/input/run.lock.json",
            "--output-root",
            "/output",
        ),
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
