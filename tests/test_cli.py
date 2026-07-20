import json
from pathlib import Path

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from harbor_hf.campaigns import build_campaign_lock, build_campaign_plan
from harbor_hf.cli import _write_lock, app
from harbor_hf.control import CampaignSubmittedPayload, new_event
from harbor_hf.models import (
    ExperimentSpec,
    GitBenchmarkSource,
    GitHubTokenCredentials,
)
from harbor_hf.operations import (
    ArtifactVerificationReport,
    CampaignEventResult,
    CampaignPublicationReport,
    CampaignSealReport,
    PublishedRun,
    SealedRun,
    VerifiedRun,
)
from harbor_hf.process import ProcessError
from harbor_hf.runs import build_run_lock

EXAMPLE = Path(__file__).parent.parent / "examples" / "shellbench.yaml"
runner = CliRunner()


def _manifest_with_shared_publishing_dataset(
    spec: ExperimentSpec, tmp_path: Path
) -> Path:
    value = spec.model_dump(mode="json", exclude_none=True)
    publishing = value["publishing"]
    assert isinstance(publishing, dict)
    publishing["index_dataset"] = publishing["dataset"]
    manifest = tmp_path / "shared-publishing-dataset.yaml"
    manifest.write_text(yaml.safe_dump(value), encoding="utf-8")
    return manifest


def test_validate_command() -> None:
    result = runner.invoke(app, ["validate", str(EXAMPLE)])

    assert result.exit_code == 0
    assert "Valid Experiment: shellbench-qwen-hardware" in result.stdout


def test_plan_command() -> None:
    result = runner.invoke(app, ["plan", str(EXAMPLE)])

    assert result.exit_code == 0
    assert '"run_count": 2' in result.stdout


def test_campaign_plan_command_prints_human_summary(remote_manifest: Path) -> None:
    result = runner.invoke(app, ["campaign", "plan", str(remote_manifest)])

    assert result.exit_code == 0
    assert "Campaign plan: shellbench-qwen-hardware" in result.stdout
    assert "Runs: 1" in result.stdout
    assert "Shards: 1" in result.stdout
    assert "Trials: 1" in result.stdout


def test_campaign_plan_command_supports_json(remote_manifest: Path) -> None:
    result = runner.invoke(
        app, ["campaign", "plan", str(remote_manifest), "--format", "json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "harbor-hf/campaign-plan/v1alpha1"
    assert payload["run_count"] == 1


def test_campaign_plan_reports_unresolved_tasks() -> None:
    result = runner.invoke(app, ["campaign", "plan", str(EXAMPLE)])

    assert result.exit_code == 0
    assert "Runs: 2" in result.stdout


def test_campaign_plan_rejects_shared_publishing_dataset_before_planning(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest_with_shared_publishing_dataset(remote_spec, tmp_path)

    def unexpected_planning(_spec: ExperimentSpec) -> object:
        pytest.fail("campaign planning must not start for an invalid manifest")

    monkeypatch.setattr("harbor_hf.cli.build_campaign_plan", unexpected_planning)

    result = runner.invoke(app, ["campaign", "plan", str(manifest)])

    assert result.exit_code == 2
    assert (
        "publishing.index_dataset must differ from publishing.dataset" in result.stderr
    )


def test_campaign_schema_command_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "campaign.schema.json"

    result = runner.invoke(app, ["campaign", "schema", "--output", str(output)])

    assert result.exit_code == 0
    assert set(json.loads(output.read_text(encoding="utf-8"))) == {
        "campaign_plan",
        "campaign_lock",
        "wave_lock",
    }


def test_campaign_submit_dry_run_has_no_remote_mutation(
    remote_manifest: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "campaign",
            "submit",
            str(remote_manifest),
            "--campaign-id",
            "campaign-one",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "artifact_prefix": "campaigns/campaign-one",
        "campaign_id": "campaign-one",
        "plan_digest": payload["plan_digest"],
        "stored": False,
    }


def test_campaign_submit_requires_remote_configuration() -> None:
    result = runner.invoke(app, ["campaign", "submit", str(EXAMPLE), "--dry-run"])

    assert result.exit_code == 1
    assert "requires a remote configuration" in result.stderr


def test_campaign_submit_rejects_missing_index_before_remote_mutation(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    spec = remote_spec.model_copy(
        update={
            "publishing": remote_spec.publishing.model_copy(
                update={"index_dataset": None}
            )
        }
    )
    manifest = tmp_path / "missing-index.yaml"
    manifest.write_text(
        yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True)),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["campaign", "submit", str(manifest), "--dry-run"])

    assert result.exit_code == 1
    assert "requires publishing.index_dataset" in result.stderr


def test_campaign_submit_rejects_shared_publishing_dataset_before_remote_work(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest_with_shared_publishing_dataset(remote_spec, tmp_path)

    def unexpected_remote_work(*_args: object, **_kwargs: object) -> None:
        pytest.fail("remote work must not start for an invalid manifest")

    monkeypatch.setattr(
        "harbor_hf.submission.ensure_private_coordination_repository",
        unexpected_remote_work,
    )
    monkeypatch.setattr("harbor_hf.cli.HubCampaignStore", unexpected_remote_work)

    result = runner.invoke(app, ["campaign", "submit", str(manifest)])

    assert result.exit_code == 2
    assert (
        "publishing.index_dataset must differ from publishing.dataset" in result.stderr
    )


def test_campaign_status_and_dry_reconcile(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = build_campaign_lock(build_campaign_plan(remote_spec), "campaign-one")
    event = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=lock.plan_digest),
    )

    class FakeStore:
        def __init__(self, namespace: str) -> None:
            assert namespace == "org"

        def load_campaign(self, campaign_id: str) -> tuple[object, list[object]]:
            assert campaign_id == "campaign-one"
            return lock, [event]

    monkeypatch.setattr("harbor_hf.cli.HubCampaignStore", FakeStore)

    status = runner.invoke(
        app, ["campaign", "status", "campaign-one", "--namespace", "org"]
    )
    reconcile = runner.invoke(
        app,
        [
            "campaign",
            "reconcile",
            "campaign-one",
            "--namespace",
            "org",
            "--dry-run",
        ],
    )

    assert status.exit_code == 0
    assert json.loads(status.stdout)["status"] == "queued"
    assert reconcile.exit_code == 0
    assert json.loads(reconcile.stdout)["action_count"] == 1


def test_campaign_reconcile_requires_an_explicit_mode() -> None:
    result = runner.invoke(
        app, ["campaign", "reconcile", "campaign", "--namespace", "org"]
    )

    assert result.exit_code == 2
    assert "choose exactly one of --dry-run or --apply" in result.stderr


def test_campaign_cancel_and_retry_print_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_cancel(_store: object, campaign_id: str, **_kwargs: object) -> object:
        calls.append(("cancel", campaign_id))
        return CampaignEventResult(
            campaign_id=campaign_id,
            event_id="evt-" + "1" * 32,
            kind="campaign.cancel-requested",
            recorded=True,
            dry_run=False,
        )

    def fake_retry(_store: object, campaign_id: str, **kwargs: object) -> object:
        calls.append(("retry", str(kwargs["shard_id"])))
        return CampaignEventResult(
            campaign_id=campaign_id,
            event_id="evt-" + "2" * 32,
            kind="campaign.shard-retry-requested",
            recorded=False,
            dry_run=True,
        )

    monkeypatch.setattr("harbor_hf.cli.cancel_campaign", fake_cancel)
    monkeypatch.setattr("harbor_hf.cli.retry_campaign_shard", fake_retry)

    cancel = runner.invoke(
        app, ["campaign", "cancel", "campaign-one", "--namespace", "org"]
    )
    retry = runner.invoke(
        app,
        [
            "campaign",
            "retry",
            "campaign-one",
            "--namespace",
            "org",
            "--shard",
            "shard-one",
            "--dry-run",
        ],
    )

    assert cancel.exit_code == 0
    assert json.loads(cancel.stdout)["recorded"] is True
    assert retry.exit_code == 0
    assert json.loads(retry.stdout)["dry_run"] is True
    assert calls == [("cancel", "campaign-one"), ("retry", "shard-one")]


def test_campaign_seal_prints_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStore:
        def __init__(self, namespace: str) -> None:
            assert namespace == "org"

        def load_snapshot(self, campaign_id: str) -> object:
            assert campaign_id == "campaign-one"
            return object()

    monkeypatch.setattr("harbor_hf.cli.HubCampaignStore", FakeStore)
    monkeypatch.setattr(
        "harbor_hf.cli.seal_partial_campaign_runs",
        lambda *_args, **_kwargs: CampaignSealReport(
            campaign_id="campaign-one",
            artifact_bucket="org/evidence",
            dry_run=True,
            runs=[
                SealedRun(
                    run_id="run-one",
                    source_prefix="campaigns/campaign-one/runs/run-one",
                )
            ],
        ),
    )

    result = runner.invoke(
        app,
        [
            "campaign",
            "seal",
            "campaign-one",
            "--namespace",
            "org",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["runs"][0]["run_id"] == "run-one"


def test_campaign_resume_requires_cleanup_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("harbor_hf.cli.HubCampaignStore", lambda _namespace: object())

    def fake_resume(_store: object, _campaign_id: str, **kwargs: object) -> object:
        calls.append(kwargs)
        if not kwargs["cleanup_verified"]:
            raise ValueError("manual recovery requires verified endpoint cleanup")
        return CampaignEventResult(
            campaign_id="campaign-one",
            event_id="evt-" + "1" * 32,
            kind="campaign.manual-intervention-resolved",
            recorded=True,
            dry_run=False,
        )

    monkeypatch.setattr("harbor_hf.cli.resume_campaign", fake_resume)

    rejected = runner.invoke(
        app, ["campaign", "resume", "campaign-one", "--namespace", "org"]
    )
    accepted = runner.invoke(
        app,
        [
            "campaign",
            "resume",
            "campaign-one",
            "--namespace",
            "org",
            "--cleanup-verified",
        ],
    )

    assert rejected.exit_code == 1
    assert accepted.exit_code == 0
    assert json.loads(accepted.stdout)["recorded"] is True
    assert [call["cleanup_verified"] for call in calls] == [False, True]


def test_artifacts_verify_and_results_publish_print_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        def __init__(self, namespace: str) -> None:
            assert namespace == "org"

        def load_snapshot(self, campaign_id: str) -> object:
            assert campaign_id == "campaign-one"
            return object()

    verified_run = VerifiedRun(
        run_id="run-one",
        publication_id="pub-one",
        source_prefix="campaigns/campaign-one/runs/run-one",
        source_checksum="sha256:" + "1" * 64,
        row_counts={
            "runs": 1,
            "trials": 1,
            "executions": 1,
            "metrics": 1,
            "artifacts": 0,
        },
    )
    monkeypatch.setattr("harbor_hf.cli.HubCampaignStore", FakeStore)
    monkeypatch.setattr(
        "harbor_hf.cli.verify_campaign_artifacts",
        lambda *_args, **_kwargs: ArtifactVerificationReport(
            campaign_id="campaign-one",
            artifact_bucket="org/evidence",
            control_commit="c" * 40,
            runs=[verified_run],
        ),
    )
    monkeypatch.setattr(
        "harbor_hf.cli.publish_campaign_results",
        lambda *_args, **_kwargs: CampaignPublicationReport(
            campaign_id="campaign-one",
            control_commit="c" * 40,
            dry_run=True,
            runs=[
                PublishedRun(
                    run_id="run-one",
                    publication_id="pub-one",
                    result_dataset="org/results",
                    index_dataset="org/index",
                    published=False,
                )
            ],
        ),
    )

    verify = runner.invoke(
        app, ["artifacts", "verify", "campaign-one", "--namespace", "org"]
    )
    publish = runner.invoke(
        app,
        [
            "results",
            "publish",
            "campaign-one",
            "--namespace",
            "org",
            "--dry-run",
        ],
    )

    assert verify.exit_code == 0
    assert json.loads(verify.stdout)["verified"] is True
    assert publish.exit_code == 0
    assert json.loads(publish.stdout)["runs"][0]["published"] is False


def test_results_publish_uses_repository_creating_recovery_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interactions: list[object] = []

    class FakeStore:
        def __init__(self, namespace: str) -> None:
            assert namespace == "org"

    class FakeAutomaticPublisher:
        def __init__(self, **kwargs: object) -> None:
            interactions.append(("repositories", kwargs["repositories"]))

        def publish(self, campaign_id: str) -> CampaignPublicationReport:
            interactions.append(("publish", campaign_id))
            return CampaignPublicationReport(
                campaign_id=campaign_id,
                control_commit="c" * 40,
                dry_run=False,
                runs=[
                    PublishedRun(
                        run_id="run-one",
                        publication_id="pub-one",
                        result_dataset="org/results",
                        index_dataset="org/index",
                        published=True,
                        result_revision="a" * 40,
                        index_revision="b" * 40,
                    )
                ],
            )

    api = object()
    monkeypatch.setattr("harbor_hf.cli.HubCampaignStore", FakeStore)
    monkeypatch.setattr("harbor_hf.cli.HfApi", lambda: api)
    monkeypatch.setattr("harbor_hf.cli.get_token", lambda: "test-token")
    monkeypatch.setattr("harbor_hf.cli.HubClaimStore", lambda *_args: object())
    monkeypatch.setattr("harbor_hf.cli.HubDatasetPublisher", lambda **_kwargs: object())
    monkeypatch.setattr(
        "harbor_hf.cli.AutomaticCampaignPublisher", FakeAutomaticPublisher
    )

    result = runner.invoke(
        app,
        ["results", "publish", "campaign-one", "--namespace", "org"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["runs"][0]["published"] is True
    assert interactions == [
        ("repositories", api),
        ("publish", "campaign-one"),
    ]


def test_automation_install_dry_run_is_secret_safe(remote_manifest: Path) -> None:
    result = runner.invoke(
        app,
        [
            "automation",
            "install",
            str(remote_manifest),
            "--schedule",
            "*/10 * * * *",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["installed"] is False
    assert payload["secret_names"] == ["HF_TOKEN"]
    assert "test-only" not in result.stdout


def test_automation_install_derives_private_source_secret(
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
        {"dataset": "shellbench/public-115", "source": source.model_dump()}
    )
    raw["benchmark"].pop("dataset_digest", None)
    manifest = tmp_path / "private-source.yaml"
    manifest.write_text(
        yaml.safe_dump(ExperimentSpec.model_validate(raw).model_dump(mode="json")),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "automation",
            "install",
            str(manifest),
            "--schedule",
            "*/10 * * * *",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["secret_names"] == ["HF_TOKEN", "GITHUB_TOKEN"]


def test_invalid_manifest_reports_stderr_and_exit_two(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.yaml"
    manifest.write_text("kind: Invalid\n", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(manifest)])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "Field required" in result.stderr


def test_submit_dry_run_prints_sanitized_remote_command(
    remote_manifest: Path,
) -> None:
    result = runner.invoke(app, ["submit", str(remote_manifest), "--dry-run"])

    assert result.exit_code == 0
    assert '"job_id": null' in result.stdout
    assert '"HF_TOKEN"' in result.stdout
    assert '"worker"' in result.stdout


def test_submit_rejects_manifest_without_remote_configuration() -> None:
    result = runner.invoke(app, ["submit", str(EXAMPLE), "--dry-run"])

    assert result.exit_code == 2
    assert "requires a remote configuration" in result.stderr


def test_lock_writer_uses_canonical_json(
    remote_spec: ExperimentSpec, tmp_path: Path
) -> None:
    lock = build_run_lock(remote_spec, run_id="written")
    path = tmp_path / "lock.json"

    _write_lock(path, lock)

    assert path.read_text(encoding="utf-8") == (
        json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )


def test_submit_reports_process_failure_without_traceback(
    remote_manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise ProcessError("submission failed")

    monkeypatch.setattr("harbor_hf.cli.submit_job", fail)

    result = runner.invoke(app, ["submit", str(remote_manifest)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: submission failed\n"


def test_submit_reports_malformed_job_output_without_traceback(
    remote_manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise ValueError("HF Jobs submission did not return a job ID")

    monkeypatch.setattr("harbor_hf.cli.submit_job", fail)

    result = runner.invoke(app, ["submit", str(remote_manifest)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: HF Jobs submission did not return a job ID\n"


def test_submit_reports_hub_failure_without_traceback(
    remote_manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise httpx.ConnectError("Hub unavailable")

    monkeypatch.setattr("harbor_hf.cli.submit_job", fail)

    result = runner.invoke(app, ["submit", str(remote_manifest)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: Hub unavailable\n"


def test_watchdog_command_reports_verified_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "harbor_hf.cli.run_endpoint_watchdog",
        lambda **_kwargs: {"status": {"state": "paused", "readyReplica": 0}},
    )

    result = runner.invoke(
        app,
        [
            "watchdog",
            "--controller-job-id",
            "job",
            "--controller-namespace",
            "org",
            "--endpoint-name",
            "endpoint",
            "--endpoint-namespace",
            "org",
            "--run-id",
            "run-1",
            "--token-secret-name",
            "HF_TOKEN",
            "--timeout-seconds",
            "60",
        ],
    )

    assert result.exit_code == 0
    assert '"state": "paused"' in result.stdout


def test_hidden_wave_worker_command_dispatches_all_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.yaml"
    campaign = tmp_path / "campaign.lock.json"
    wave = tmp_path / "wave.lock.json"
    for path in (manifest, campaign, wave):
        path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    calls: list[tuple[Path, Path, Path, Path]] = []

    def fake_worker(
        manifest_path: Path,
        campaign_path: Path,
        wave_path: Path,
        output_root: Path,
    ) -> Path:
        calls.append((manifest_path, campaign_path, wave_path, output_root))
        return output_root / "campaigns/campaign-one/waves/wave-one"

    monkeypatch.setattr("harbor_hf.cli.run_wave_worker", fake_worker)

    result = runner.invoke(
        app,
        [
            "wave-worker",
            str(manifest),
            str(campaign),
            str(wave),
            "--output-root",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert calls == [(manifest, campaign, wave, output)]
    assert result.stdout.strip().endswith("campaigns/campaign-one/waves/wave-one")
