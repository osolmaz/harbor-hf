from pathlib import Path

from typer.testing import CliRunner

from harbor_hf.cli import app

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
