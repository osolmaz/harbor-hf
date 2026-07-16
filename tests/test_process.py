import os
import signal
import subprocess
import time
from contextlib import suppress
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


def test_subprocess_runner_enforces_command_timeout() -> None:
    with pytest.raises(ProcessError, match="timed out after 0.05 seconds"):
        SubprocessRunner().run_text(
            ["python", "-c", "import time; time.sleep(60)"],
            timeout_seconds=0.05,
        )


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


@pytest.mark.parametrize(
    "secret_key",
    ["AUTHORIZATION", "PASSWORD", "CLIENT_SECRET", "ACCESS_TOKEN", "API_KEY"],
)
def test_streaming_runner_redacts_secret_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], secret_key: str
) -> None:
    log = tmp_path / "command.log"

    code = run_streaming(
        ["python", "-c", f"import os; print(os.environ[{secret_key!r}])"],
        log,
        environment={secret_key: "credential-value"},
    )

    assert code == 0
    assert log.read_text() == "[REDACTED]\n"
    assert capsys.readouterr().out == "[REDACTED]\n"


def test_streaming_runner_redacts_secret_loaded_from_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    credential = tmp_path / "credential"
    credential.write_text("credential-value\nroute-capability\n", encoding="utf-8")
    log = tmp_path / "credential-file.log"

    code = run_streaming(
        ["printf", "%s", "credential-value route-capability"],
        log,
        environment={"HARBOR_HF_REDACTION_SECRET_FILE": str(credential)},
    )

    assert code == 0
    assert log.read_text() == "[REDACTED] [REDACTED]"
    assert capsys.readouterr().out == "[REDACTED] [REDACTED]"


def test_streaming_runner_redacts_secrets_split_across_reads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "split.log"

    code = run_streaming(
        [
            "python",
            "-c",
            (
                "import os,time; os.write(1,b'credential-'); time.sleep(.05); "
                "os.write(1,b'value\\n')"
            ),
        ],
        log,
        environment={"ACCESS_TOKEN": "credential-value"},
    )

    assert code == 0
    assert log.read_text() == "[REDACTED]\n"
    assert capsys.readouterr().out == "[REDACTED]\n"


def test_streaming_runner_redacts_repeated_and_overlapping_secrets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "overlapping.log"

    code = run_streaming(
        ["python", "-c", "print('credential-value credential-value')"],
        log,
        environment={
            "ACCESS_TOKEN": "credential",
            "API_KEY": "credential-value",
        },
    )

    assert code == 0
    assert log.read_text() == "[REDACTED] [REDACTED]\n"
    assert capsys.readouterr().out == "[REDACTED] [REDACTED]\n"


def test_streaming_runner_captures_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "stderr.log"

    code = run_streaming(
        ["python", "-c", "import sys; print('error', file=sys.stderr)"],
        log,
        environment={},
    )

    assert code == 0
    assert log.read_text(encoding="utf-8") == "error\n"
    assert capsys.readouterr().out == "error\n"


def test_streaming_runner_terminates_timed_out_process_group(tmp_path: Path) -> None:
    log = tmp_path / "timeout.log"

    command = [
        "python",
        "-c",
        "import time; print('started', flush=True); time.sleep(60)",
    ]
    with pytest.raises(ProcessError, match="timed out after 0.05 seconds") as captured:
        run_streaming(
            command,
            log,
            environment={},
            timeout_seconds=0.05,
        )

    assert log.read_text(encoding="utf-8") == "started\n"
    assert isinstance(captured.value.__cause__, subprocess.TimeoutExpired)
    assert captured.value.__cause__.cmd == command
    assert captured.value.__cause__.timeout == 0.05


def test_streaming_runner_bounds_pipe_held_by_descendant(tmp_path: Path) -> None:
    log = tmp_path / "descendant.log"
    started = time.monotonic()

    with pytest.raises(ProcessError, match="timed out after 0.05 seconds"):
        run_streaming(
            [
                "python",
                "-c",
                (
                    "import subprocess; child=subprocess.Popen(['sleep', '60']); "
                    "print(child.pid, flush=True)"
                ),
            ],
            log,
            environment={},
            timeout_seconds=0.05,
        )

    assert time.monotonic() - started < 2
    child_pid = int(log.read_text().strip())
    try:
        assert _wait_stopped(child_pid)
    finally:
        with suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)


@pytest.mark.parametrize("invalid_bytes", [r"\xff", r"\xc3"])
def test_streaming_runner_propagates_output_capture_failures(
    tmp_path: Path, invalid_bytes: str
) -> None:
    with pytest.raises(ProcessError) as captured:
        run_streaming(
            [
                "python",
                "-c",
                f"import os; os.write(1, b'{invalid_bytes}')",
            ],
            tmp_path / "invalid.log",
            environment={},
        )

    assert str(captured.value) == "failed while capturing command output"


def _process_state(process_id: int) -> str | None:
    try:
        fields = Path(f"/proc/{process_id}/stat").read_text().split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return fields[2]


def _wait_stopped(process_id: int) -> bool:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if _process_state(process_id) in {None, "Z"}:
            return True
        time.sleep(0.01)
    return False
