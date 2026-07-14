"""Exact behavioral contracts for submission staging and coordination."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from test_submission import FakeBucketApi, FakeRunner, _wave_lock

from harbor_hf.models import ExperimentSpec
from harbor_hf.provider_models import ProviderTarget
from harbor_hf.runs import build_run_lock
from harbor_hf.submission import (
    build_submit_wave_command,
    ensure_private_coordination_repository,
    github_repository,
    stage_job_input,
    submit,
)


def test_stage_job_input_produces_golden_content_address(tmp_path: Path) -> None:
    api = FakeBucketApi()
    (tmp_path / "nested").mkdir()
    (tmp_path / "manifest.yaml").write_bytes(b"manifest")
    (tmp_path / "nested" / "lock.json").write_bytes(b"lock")

    uri = stage_job_input(
        tmp_path, bucket="osolmaz/jobs-artifacts", identity="wave-x", api=api
    )

    assert uri == (
        "hf://buckets/osolmaz/jobs-artifacts/job-inputs/wave-x/"
        "c474db8895ff4fc78f149eed68c32242f38dc8c6aa269ecf46069517e2a0b7a7"
    )


def test_stage_job_input_digest_is_sensitive_to_names_and_content(
    tmp_path: Path,
) -> None:
    api = FakeBucketApi()
    uris: list[str] = []
    for name, (file_name, content) in enumerate(
        [("a.txt", b"x"), ("b.txt", b"x"), ("a.txt", b"y")]
    ):
        directory = tmp_path / f"case-{name}"
        directory.mkdir()
        (directory / file_name).write_bytes(content)
        uris.append(
            stage_job_input(
                directory, bucket="osolmaz/jobs-artifacts", identity="i", api=api
            )
        )

    assert len(set(uris)) == 3


def test_submit_stages_input_under_run_identity(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = build_run_lock(remote_spec, run_id="run-42")
    runner = FakeRunner("Job started: 0123456789abcdef01234567\n")
    api = FakeBucketApi()
    (tmp_path / "manifest.yaml").write_text("kind: Experiment\n")

    result = submit(
        lock,
        input_dir=tmp_path,
        bucket="osolmaz/benchmark-runs",
        runner=runner,
        bucket_api=api,
    )

    assert result.run_id == "run-42"
    assert result.artifact_prefix == lock.artifact_prefix
    assert result.command == runner.command
    paths = [path for _content, path in api.bucket_batches[0][1]]
    assert all(path.startswith("job-inputs/run-42/") for path in paths)


def test_wave_submission_provider_label_is_golden(
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

    label_index = command.index("harbor-hf-provider=22425020c0be5b2b7b01777dbb6c25d3")
    assert command[label_index - 1] == "--label"


def test_coordination_repository_requires_strictly_private_flag() -> None:
    class UnknownPrivacyApi(FakeBucketApi):
        def repo_info(self, repo_id: str, **kwargs: object) -> object:
            return SimpleNamespace(private=None, sha="1" * 40)

    with pytest.raises(ValueError) as caught:
        ensure_private_coordination_repository("osolmaz", api=UnknownPrivacyApi())

    assert str(caught.value) == (
        "coordination repository osolmaz/harbor-hf-coordination must be private"
    )


def test_empty_string_commit_sha_triggers_initialization() -> None:
    api = FakeBucketApi(repository_sha="")

    ensure_private_coordination_repository("osolmaz", api=api)

    assert len(api.repository_commits) == 1
    assert api.repository_sha == "2" * 40


def test_github_repository_normalizes_suffix_variants() -> None:
    assert github_repository("org/repo/") == "https://github.com/org/repo"
    assert github_repository("org/repo.git") == "https://github.com/org/repo"
    assert github_repository("https://github.com/org/repo/") == (
        "https://github.com/org/repo"
    )
