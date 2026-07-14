from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError

from harbor_hf.campaigns import (
    WaveLock,
    build_campaign_lock,
    build_campaign_plan,
    build_wave_lock,
)
from harbor_hf.control import CampaignSubmittedPayload, new_event
from harbor_hf.models import ExperimentSpec, GitBenchmarkSource, GitHubTokenCredentials
from harbor_hf.provider_models import ProviderTarget
from harbor_hf.reconciler import plan_reconciliation
from harbor_hf.runs import build_run_lock
from harbor_hf.submission import (
    bucket_id,
    bucket_uri,
    build_submit_command,
    build_submit_wave_command,
    endpoint_lease_label,
    endpoint_lease_label_for,
    ensure_private_coordination_repository,
    ensure_private_job_input_bucket,
    github_archive,
    github_repository,
    job_secret_names,
    locked_source_command,
    require_private_bucket,
    stage_job_input,
    submit,
    submit_wave,
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
        repository_sha: str | None = "1" * 40,
    ) -> None:
        self.private = private
        self.privacy = privacy or {}
        self.repository_private = repository_private
        self.repository_sha = repository_sha
        self.created: list[tuple[str, dict[str, object]]] = []
        self.inspected: list[str] = []
        self.created_repositories: list[tuple[str, dict[str, object]]] = []
        self.inspected_repositories: list[tuple[str, dict[str, object]]] = []
        self.repository_commits: list[tuple[str, list[object], dict[str, object]]] = []
        self.bucket_batches: list[
            tuple[str, list[tuple[bytes, str]], dict[str, object]]
        ] = []

    def create_bucket(self, bucket_id: str, **kwargs: object) -> object:
        self.created.append((bucket_id, kwargs))
        return object()

    def bucket_info(self, bucket_id: str) -> object:
        self.inspected.append(bucket_id)
        private = self.privacy.get(bucket_id, self.private)
        return type("BucketInfo", (), {"private": private})()

    def batch_bucket_files(
        self,
        bucket_id: str,
        *,
        add: list[tuple[bytes, str]],
        **kwargs: object,
    ) -> object:
        self.bucket_batches.append((bucket_id, add, kwargs))
        return object()

    def create_repo(self, repo_id: str, **kwargs: object) -> object:
        self.created_repositories.append((repo_id, kwargs))
        return object()

    def repo_info(self, repo_id: str, **kwargs: object) -> object:
        self.inspected_repositories.append((repo_id, kwargs))
        return SimpleNamespace(
            private=self.repository_private,
            sha=self.repository_sha,
        )

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object:
        self.repository_commits.append((repo_id, operations, kwargs))
        self.repository_sha = "2" * 40
        return SimpleNamespace(oid=self.repository_sha)


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


def test_build_submit_wave_command_targets_hidden_worker(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _wave_lock(remote_spec)

    command = build_submit_wave_command(
        lock, input_dir=tmp_path, bucket="osolmaz/benchmark-runs"
    )

    job = lock.remote.job
    assert command[:22] == [
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
        f"harbor-hf-wave={lock.wave_id}",
        "--label",
        f"harbor-hf-endpoint={endpoint_lease_label_for('osolmaz', 'qwen-endpoint')}",
        "--volume",
        f"{tmp_path}:/input:ro",
        "--volume",
        "hf://buckets/osolmaz/benchmark-runs:/output:rw",
        "--",
        job.image,
    ]
    assert command[22:25] == ["bash", "-lc", command[24]]
    assert "set -euo pipefail" in command[24]
    assert command[25:] == [
        "locked-source",
        "harbor-hf",
        "wave-worker",
        "/input/manifest.yaml",
        "/input/campaign.lock.json",
        "/input/wave.lock.json",
        "--output-root",
        "/output",
    ]
    assert "test-token" not in " ".join(command)


def test_authenticated_source_adds_named_secret_to_controller_jobs(
    remote_spec: ExperimentSpec, tmp_path: Path
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
    run_lock = build_run_lock(spec, run_id="private-source")
    wave_lock = _wave_lock(spec)

    run_command = build_submit_command(
        run_lock, input_dir=tmp_path, bucket="osolmaz/benchmark-runs"
    )
    wave_command = build_submit_wave_command(
        wave_lock, input_dir=tmp_path, bucket="osolmaz/benchmark-runs"
    )

    assert job_secret_names(run_lock) == ["HF_TOKEN", "GITHUB_TOKEN"]
    assert job_secret_names(wave_lock) == ["HF_TOKEN", "GITHUB_TOKEN"]
    assert run_command.count("--secrets") == 2
    assert wave_command.count("--secrets") == 2
    assert run_command[10:14] == [
        "--secrets",
        "HF_TOKEN",
        "--secrets",
        "GITHUB_TOKEN",
    ]
    assert "github-secret" not in " ".join(run_command + wave_command)


def test_private_source_submission_requires_local_secret_before_staging(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = GitBenchmarkSource(
        repository="ShellBench/public-tasks",
        revision="8" * 40,
        path="tasks/115-tasks",
        credentials=GitHubTokenCredentials(secret_name="GITHUB_TOKEN"),
    )
    raw = remote_spec.model_dump(mode="python")
    raw["benchmark"].update(
        {"dataset": "shellbench/public-115", "source": source.model_dump()}
    )
    raw["benchmark"].pop("dataset_digest", None)
    spec = ExperimentSpec.model_validate(raw)
    run_lock = build_run_lock(spec, run_id="private-source")
    wave_lock = _wave_lock(spec)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    runner = FakeRunner("a" * 24)
    api = FakeBucketApi()

    with pytest.raises(ValueError, match="required secret GITHUB_TOKEN"):
        submit(
            run_lock,
            input_dir=tmp_path,
            bucket="osolmaz/benchmark-runs",
            runner=runner,
            bucket_api=api,
        )
    with pytest.raises(ValueError, match="required secret GITHUB_TOKEN"):
        submit_wave(
            wave_lock,
            input_dir=tmp_path,
            bucket="osolmaz/benchmark-runs",
            runner=runner,
            bucket_api=api,
        )

    assert runner.command is None
    assert api.created == []
    assert api.created_repositories == []


def test_provider_wave_submission_has_no_endpoint_lease_label(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    model = remote_spec.matrix.models[0]
    target = ProviderTarget(id="hf-provider", model=model.repo)
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(update={"deployments": [target]})
        }
    )
    lock = _wave_lock(spec)

    command = build_submit_wave_command(
        lock, input_dir=tmp_path, bucket="osolmaz/benchmark-runs"
    )
    rendered = " ".join(command)

    assert "harbor-hf-provider=" in rendered
    assert "harbor-hf-endpoint=" not in rendered


def test_submit_wave_parses_job_id_and_checks_private_stores(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = _wave_lock(remote_spec)
    runner = FakeRunner("Job started: 0123456789abcdef01234567\n")
    api = FakeBucketApi()
    (tmp_path / "manifest.yaml").write_text("kind: Experiment\n")

    result = submit_wave(
        lock,
        input_dir=tmp_path,
        bucket=lock.artifact_bucket,
        runner=runner,
        bucket_api=api,
    )

    assert result.wave_id == lock.wave_id
    assert result.job_id == "0123456789abcdef01234567"
    assert api.inspected == ["osolmaz/jobs-artifacts", "example/benchmark-runs"]
    assert len(api.bucket_batches) == 1
    input_bucket, additions, kwargs = api.bucket_batches[0]
    assert input_bucket == "osolmaz/jobs-artifacts"
    assert kwargs == {}
    assert additions[0][0] == b"kind: Experiment\n"
    assert additions[0][1].endswith("/manifest.yaml")
    assert runner.command is not None
    assert any(
        value.startswith("hf://buckets/osolmaz/jobs-artifacts/job-inputs/")
        and value.endswith(":/input:ro")
        for value in runner.command
    )


def _wave_lock(spec: ExperimentSpec) -> WaveLock:
    campaign = build_campaign_lock(build_campaign_plan(spec), "campaign-one")
    submitted = new_event(
        subject_type="campaign",
        subject_id=campaign.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=campaign.plan_digest),
    )
    action = plan_reconciliation(campaign, [submitted])[1].actions[0]
    return build_wave_lock(campaign, spec, action)


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
    (tmp_path / "manifest.yaml").write_text("kind: Experiment\n")

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
    (tmp_path / "manifest.yaml").write_text("kind: Experiment\n")

    result = submit(
        build_run_lock(remote_spec),
        input_dir=tmp_path,
        bucket="osolmaz/benchmark-runs",
        runner=runner,
    )

    assert result.job_id == "0123456789abcdef01234567"
    assert runner.command is not None
    assert any(
        value.startswith("hf://buckets/osolmaz/jobs-artifacts/job-inputs/")
        and value.endswith(":/input:ro")
        for value in runner.command
    )
    assert api.inspected == [
        "osolmaz/jobs-artifacts",
        "osolmaz/benchmark-runs",
    ]


def test_submit_rejects_missing_job_id(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = build_run_lock(remote_spec)
    (tmp_path / "manifest.yaml").write_text("kind: Experiment\n")

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


@pytest.mark.parametrize("digest", ["a" * 40, "b" * 64])
def test_submit_does_not_extract_job_id_from_longer_hex_digest(
    remote_spec: ExperimentSpec, tmp_path: Path, digest: str
) -> None:
    lock = build_run_lock(remote_spec)
    (tmp_path / "manifest.yaml").write_text("kind: Experiment\n")

    with pytest.raises(ValueError, match="did not return a job ID"):
        submit(
            lock,
            input_dir=tmp_path,
            bucket="osolmaz/benchmark-runs",
            runner=FakeRunner(f"revision {digest}"),
            bucket_api=FakeBucketApi(),
        )


def test_stage_job_input_is_content_addressed_and_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    api = FakeBucketApi()
    (tmp_path / "nested").mkdir()
    (tmp_path / "manifest.yaml").write_bytes(b"manifest")
    (tmp_path / "nested" / "lock.json").write_bytes(b"lock")

    first = stage_job_input(
        tmp_path,
        bucket="osolmaz/jobs-artifacts",
        identity="wave-one",
        api=api,
    )
    second = stage_job_input(
        tmp_path,
        bucket="osolmaz/jobs-artifacts",
        identity="wave-one",
        api=api,
    )

    assert first == second
    assert first.startswith("hf://buckets/osolmaz/jobs-artifacts/job-inputs/wave-one/")
    assert api.bucket_batches[0] == api.bucket_batches[1]
    assert [path for _content, path in api.bucket_batches[0][1]] == [
        first.removeprefix("hf://buckets/osolmaz/jobs-artifacts/") + "/manifest.yaml",
        first.removeprefix("hf://buckets/osolmaz/jobs-artifacts/")
        + "/nested/lock.json",
    ]

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(
        ValueError, match="^Job input directory must contain at least one file$"
    ):
        stage_job_input(
            empty,
            bucket="osolmaz/jobs-artifacts",
            identity="wave-empty",
            api=api,
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


def test_submit_initializes_empty_coordination_repository() -> None:
    api = FakeBucketApi(repository_sha=None)

    ensure_private_coordination_repository("osolmaz", api=api)

    assert api.repository_sha == "2" * 40
    assert len(api.repository_commits) == 1
    repository, operations, kwargs = api.repository_commits[0]
    assert repository == "osolmaz/harbor-hf-coordination"
    assert kwargs == {
        "commit_message": "chore: initialize coordination repository",
        "repo_type": "dataset",
        "revision": "main",
    }
    assert len(operations) == 1
    operation = operations[0]
    assert isinstance(operation, CommitOperationAdd)
    assert operation.path_in_repo == ".harbor-hf-initialized"
    assert operation.path_or_fileobj == b"harbor-hf coordination repository\n"
    assert api.inspected_repositories[-1] == (
        "osolmaz/harbor-hf-coordination",
        {"repo_type": "dataset", "revision": "main"},
    )


def test_submit_accepts_concurrent_coordination_initialization() -> None:
    class ConcurrentApi(FakeBucketApi):
        def create_commit(
            self, repo_id: str, operations: list[object], **kwargs: object
        ) -> object:
            self.repository_sha = "3" * 40
            request = httpx.Request("POST", "https://huggingface.co/api/datasets")
            raise HfHubHTTPError(
                "already initialized",
                response=httpx.Response(409, request=request),
            )

    api = ConcurrentApi(repository_sha=None)

    ensure_private_coordination_repository("osolmaz", api=api)

    assert api.repository_sha == "3" * 40


def test_submit_rejects_failed_coordination_initialization() -> None:
    class FailingApi(FakeBucketApi):
        def create_commit(
            self, repo_id: str, operations: list[object], **kwargs: object
        ) -> object:
            request = httpx.Request("POST", "https://huggingface.co/api/datasets")
            raise HfHubHTTPError(
                "initialization failed",
                response=httpx.Response(500, request=request),
            )

    with pytest.raises(HfHubHTTPError, match="initialization failed"):
        ensure_private_coordination_repository(
            "osolmaz", api=FailingApi(repository_sha=None)
        )


def test_submit_rejects_initialization_without_commit_identity() -> None:
    class MissingCommitApi(FakeBucketApi):
        def create_commit(
            self, repo_id: str, operations: list[object], **kwargs: object
        ) -> object:
            return SimpleNamespace(oid=None)

    with pytest.raises(ValueError, match="has no commit identity"):
        ensure_private_coordination_repository(
            "osolmaz", api=MissingCommitApi(repository_sha=None)
        )


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
