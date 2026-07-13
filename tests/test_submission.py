from pathlib import Path

import pytest

from harbor_hf.models import ExperimentSpec
from harbor_hf.runs import build_run_lock
from harbor_hf.submission import (
    bucket_id,
    bucket_uri,
    build_submit_command,
    endpoint_lease_label,
    endpoint_lease_label_for,
    ensure_private_coordination_repository,
    ensure_private_job_input_bucket,
    github_archive,
    github_repository,
    locked_source_command,
    require_private_bucket,
    submit,
)


class FakeRunner:
    def __init__(self, output: str) -> None:
        self.output = output
        self.command: list[str] | None = None

    def run_text(self, command: list[str]) -> str:
        self.command = command
        return self.output


class FakeBucketApi:
    def __init__(
        self,
        *,
        private: bool = True,
        privacy: dict[str, bool] | None = None,
        repository_private: bool = True,
    ) -> None:
        self.private = private
        self.privacy = privacy or {}
        self.repository_private = repository_private
        self.created: list[tuple[str, dict[str, object]]] = []
        self.inspected: list[str] = []
        self.created_repositories: list[tuple[str, dict[str, object]]] = []
        self.inspected_repositories: list[tuple[str, dict[str, object]]] = []

    def create_bucket(self, bucket_id: str, **kwargs: object) -> object:
        self.created.append((bucket_id, kwargs))
        return object()

    def bucket_info(self, bucket_id: str) -> object:
        self.inspected.append(bucket_id)
        private = self.privacy.get(bucket_id, self.private)
        return type("BucketInfo", (), {"private": private})()

    def create_repo(self, repo_id: str, **kwargs: object) -> object:
        self.created_repositories.append((repo_id, kwargs))
        return object()

    def repo_info(self, repo_id: str, **kwargs: object) -> object:
        self.inspected_repositories.append((repo_id, kwargs))
        return type("RepoInfo", (), {"private": self.repository_private})()


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
        "ghcr.io/astral-sh/uv@sha256:" + "0" * 64,
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
    bucket_api = FakeBucketApi()

    result = submit(
        lock,
        input_dir=tmp_path,
        bucket="osolmaz/benchmark-runs",
        runner=runner,
        bucket_api=bucket_api,
    )

    assert result.job_id == "0123456789abcdef01234567"
    assert runner.command is not None
    assert bucket_api.created == [
        ("osolmaz/jobs-artifacts", {"private": True, "exist_ok": True})
    ]
    assert bucket_api.inspected == [
        "osolmaz/jobs-artifacts",
        "osolmaz/benchmark-runs",
    ]
    assert bucket_api.created_repositories == [
        (
            "osolmaz/harbor-hf-coordination",
            {"repo_type": "dataset", "private": True, "exist_ok": True},
        )
    ]


def test_submit_builds_default_bucket_api(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeBucketApi()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
    runner = FakeRunner("0123456789abcdef01234567")

    result = submit(
        build_run_lock(remote_spec),
        input_dir=tmp_path,
        bucket="osolmaz/benchmark-runs",
        runner=runner,
    )

    assert result.job_id == "0123456789abcdef01234567"
    assert runner.command is not None
    assert f"{tmp_path}:/input:ro" in runner.command
    assert api.inspected == [
        "osolmaz/jobs-artifacts",
        "osolmaz/benchmark-runs",
    ]


def test_submit_rejects_missing_job_id(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = build_run_lock(remote_spec)

    with pytest.raises(
        ValueError, match="^HF Jobs submission did not return a job ID$"
    ):
        submit(
            lock,
            input_dir=tmp_path,
            bucket="osolmaz/benchmark-runs",
            runner=FakeRunner("submitted"),
            bucket_api=FakeBucketApi(),
        )


def test_submit_ensures_private_coordination_repository() -> None:
    api = FakeBucketApi()

    assert ensure_private_coordination_repository("osolmaz", api=api) == (
        "osolmaz/harbor-hf-coordination"
    )
    assert api.created_repositories == [
        (
            "osolmaz/harbor-hf-coordination",
            {"repo_type": "dataset", "private": True, "exist_ok": True},
        )
    ]


def test_submit_builds_default_authenticated_coordination_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeBucketApi()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)

    assert ensure_private_coordination_repository("osolmaz") == (
        "osolmaz/harbor-hf-coordination"
    )
    assert len(api.created_repositories) == 1


def test_submit_ensures_private_job_input_bucket() -> None:
    api = FakeBucketApi()

    assert ensure_private_job_input_bucket("osolmaz", api=api) == (
        "osolmaz/jobs-artifacts"
    )
    assert api.created == [
        ("osolmaz/jobs-artifacts", {"private": True, "exist_ok": True})
    ]


@pytest.mark.parametrize(
    "value",
    [
        "osolmaz/benchmark-runs",
        "buckets/osolmaz/benchmark-runs",
        "hf://buckets/osolmaz/benchmark-runs",
    ],
)
def test_bucket_id_normalizes_supported_references(value: str) -> None:
    assert bucket_id(value) == "osolmaz/benchmark-runs"


def test_require_private_artifact_bucket_returns_normalized_id() -> None:
    api = FakeBucketApi()

    assert (
        require_private_bucket("buckets/osolmaz/benchmark-runs", api=api)
        == "osolmaz/benchmark-runs"
    )
    assert api.inspected == ["osolmaz/benchmark-runs"]


def test_submit_rejects_public_coordination_repository() -> None:
    with pytest.raises(ValueError, match="must be private"):
        ensure_private_coordination_repository(
            "osolmaz", api=FakeBucketApi(repository_private=False)
        )


def test_submit_rejects_public_job_input_bucket() -> None:
    with pytest.raises(ValueError, match="must be private"):
        ensure_private_job_input_bucket("osolmaz", api=FakeBucketApi(private=False))


def test_submit_rejects_public_artifact_bucket() -> None:
    api = FakeBucketApi(
        privacy={
            "osolmaz/benchmark-runs": False,
        }
    )

    with pytest.raises(
        ValueError, match="^artifact bucket osolmaz/benchmark-runs must be private$"
    ):
        require_private_bucket("hf://buckets/osolmaz/benchmark-runs", api=api)


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
