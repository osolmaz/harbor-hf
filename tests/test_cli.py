import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from harbor_hf.cli import _write_lock, app
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
