from pathlib import Path

import pytest

from harbor_hf.process import ProcessError, SubprocessRunner, run_streaming


def test_subprocess_runner_reads_text_and_json() -> None:
    runner = SubprocessRunner()

    assert runner.run_text(["python", "-c", "print('hello')"]) == "hello"
    assert runner.run_json(["python", "-c", "print('{\"value\": 3}')"]) == {"value": 3}


def test_subprocess_runner_rejects_failed_or_invalid_commands() -> None:
    runner = SubprocessRunner()

    with pytest.raises(ProcessError, match="exit 4"):
        runner.run_text(["python", "-c", "import sys; sys.exit(4)"])
    with pytest.raises(ProcessError, match="did not return JSON"):
        runner.run_json(["python", "-c", "print('no')"])
    with pytest.raises(ProcessError, match="non-object"):
        runner.run_json(["python", "-c", "print('[]')"])


def test_streaming_runner_captures_combined_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "one" / "two" / "command.log"

    code = run_streaming(
        ["python", "-c", "import os; print(os.environ['EXAMPLE'])"],
        log,
        environment={"EXAMPLE": "streamed-cafe"},
    )

    assert code == 0
    assert log.read_text(encoding="utf-8") == "streamed-cafe\n"
    assert capsys.readouterr().out == "streamed-cafe\n"


def test_streaming_runner_redacts_secret_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "command.log"

    code = run_streaming(
        ["python", "-c", "import os; print(os.environ['API_KEY'])"],
        log,
        environment={"API_KEY": "credential-value"},
    )

    assert code == 0
    assert log.read_text() == "[REDACTED]\n"
    assert capsys.readouterr().out == "[REDACTED]\n"
