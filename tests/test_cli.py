import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from harbor_hf.campaigns import build_campaign_lock, build_campaign_plan
from harbor_hf.cli import _write_lock, app
from harbor_hf.control import CampaignSubmittedPayload, new_event
from harbor_hf.models import ExperimentSpec
from harbor_hf.process import ProcessError
from harbor_hf.runs import build_run_lock

EXAMPLE = Path(__file__).parent.parent / "examples" / "shellbench.yaml"
runner = CliRunner()


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


def test_campaign_reconcile_requires_dry_run() -> None:
    result = runner.invoke(
        app, ["campaign", "reconcile", "campaign", "--namespace", "org"]
    )

    assert result.exit_code == 2
    assert "currently requires --dry-run" in result.stderr


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
