"""Exact behavioral contracts for subprocess running and log redaction."""

from pathlib import Path

import pytest

from harbor_hf.process import ProcessError, SubprocessRunner, run_streaming


def test_streaming_runner_returns_nonzero_exit_code(tmp_path: Path) -> None:
    code = run_streaming(
        ["python", "-c", "import sys; sys.exit(7)"],
        tmp_path / "exit.log",
        environment={},
    )

    assert code == 7


def test_run_text_failure_prefers_stderr_detail() -> None:
    with pytest.raises(ProcessError) as caught:
        SubprocessRunner().run_text(
            [
                "python",
                "-c",
                (
                    "import sys; print('out-detail'); "
                    "print('err-detail', file=sys.stderr); sys.exit(3)"
                ),
            ]
        )

    assert str(caught.value) == "command failed with exit 3: err-detail"


def test_run_text_failure_falls_back_to_stdout_detail() -> None:
    with pytest.raises(ProcessError) as caught:
        SubprocessRunner().run_text(
            ["python", "-c", "import sys; print('out-detail'); sys.exit(5)"]
        )

    assert str(caught.value) == "command failed with exit 5: out-detail"


def test_run_text_strips_surrounding_whitespace() -> None:
    assert (
        SubprocessRunner().run_text(["python", "-c", "print('  padded  ')"]) == "padded"
    )


def test_run_json_propagates_timeout() -> None:
    with pytest.raises(ProcessError, match="^command timed out after 0.05 seconds$"):
        SubprocessRunner().run_json(
            ["python", "-c", "import time; time.sleep(60)"],
            timeout_seconds=0.05,
        )


def test_streaming_runner_ignores_empty_secret_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "empty-secret.log"

    code = run_streaming(
        ["python", "-c", "print('hello world')"],
        log,
        environment={"API_KEY": "", "GREETING": "hello"},
    )

    assert code == 0
    assert log.read_text(encoding="utf-8") == "hello world\n"
    assert capsys.readouterr().out == "hello world\n"


def test_streaming_runner_leaves_non_secret_keys_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "non-secret.log"

    code = run_streaming(
        ["python", "-c", "import os; print(os.environ['USERNAME'])"],
        log,
        environment={"USERNAME": "credential-value"},
    )

    assert code == 0
    assert log.read_text(encoding="utf-8") == "credential-value\n"
    assert capsys.readouterr().out == "credential-value\n"
