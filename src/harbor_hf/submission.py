from __future__ import annotations

import hashlib
import re
import shlex
from pathlib import Path
from typing import Protocol, cast

from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel, ConfigDict

from harbor_hf.coordination import bucket_id, coordination_repository
from harbor_hf.models import SourcePin
from harbor_hf.runs import RunLock

_JOB_ID = re.compile(r"[a-f0-9]{24}")
_GITHUB_REPOSITORY = re.compile(
    r"^(?:https://github\.com/)?"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_JOB_INPUT_BUCKET_NAME = "jobs-artifacts"
_COORDINATION_INITIALIZATION_PATH = ".harbor-hf-initialized"
_COORDINATION_INITIALIZATION_PAYLOAD = b"harbor-hf coordination repository\n"


class TextRunner(Protocol):
    def run_text(self, command: list[str]) -> str: ...


class BucketApi(Protocol):
    def create_bucket(self, bucket_id: str, **kwargs: object) -> object: ...

    def bucket_info(self, bucket_id: str) -> object: ...

    def create_repo(self, repo_id: str, **kwargs: object) -> object: ...

    def repo_info(self, repo_id: str, **kwargs: object) -> object: ...

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object: ...


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


def ensure_private_coordination_repository(
    namespace: str, *, api: BucketApi | None = None
) -> str:
    if api is None:
        from huggingface_hub import HfApi

        api = cast(BucketApi, HfApi())
    repository = coordination_repository(namespace)
    api.create_repo(
        repository,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    info = api.repo_info(repository, repo_type="dataset")
    if getattr(info, "private", None) is not True:
        raise ValueError(f"coordination repository {repository} must be private")
    if _commit_identity(info) is None:
        _initialize_coordination_repository(repository, api)
    return repository


def _initialize_coordination_repository(repository: str, api: BucketApi) -> None:
    initialization_error: HfHubHTTPError | None = None
    try:
        api.create_commit(
            repository,
            [
                CommitOperationAdd(
                    path_in_repo=_COORDINATION_INITIALIZATION_PATH,
                    path_or_fileobj=_COORDINATION_INITIALIZATION_PAYLOAD,
                )
            ],
            commit_message="chore: initialize coordination repository",
            repo_type="dataset",
            revision="main",
        )
    except HfHubHTTPError as error:
        initialization_error = error
    info = api.repo_info(repository, repo_type="dataset", revision="main")
    if _commit_identity(info) is not None:
        return
    if initialization_error is not None:
        raise initialization_error
    raise ValueError(f"coordination repository {repository} has no commit identity")


def _commit_identity(info: object) -> str | None:
    revision = getattr(info, "sha", None)
    return revision if isinstance(revision, str) and revision else None


def ensure_private_job_input_bucket(namespace: str, *, api: BucketApi) -> str:
    bucket = f"{namespace}/{_JOB_INPUT_BUCKET_NAME}"
    api.create_bucket(bucket, private=True, exist_ok=True)
    if getattr(api.bucket_info(bucket), "private", None) is not True:
        raise ValueError(f"Job input bucket {bucket} must be private")
    return bucket


def require_private_bucket(bucket: str, *, api: BucketApi) -> str:
    normalized = bucket_id(bucket)
    if getattr(api.bucket_info(normalized), "private", None) is not True:
        raise ValueError(f"artifact bucket {normalized} must be private")
    return normalized


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
    bucket_api: BucketApi | None = None,
) -> Submission:
    if bucket_api is None:
        from huggingface_hub import HfApi

        bucket_api = cast(BucketApi, HfApi())
    ensure_private_coordination_repository(lock.remote.job.namespace, api=bucket_api)
    ensure_private_job_input_bucket(lock.remote.job.namespace, api=bucket_api)
    require_private_bucket(bucket, api=bucket_api)
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
