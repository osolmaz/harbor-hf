from __future__ import annotations

import codecs
import json
import os
import selectors
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO, Protocol, TextIO, cast


class ProcessError(RuntimeError):
    """Raised when an external command fails."""


class CommandRunner(Protocol):
    def run_json(self, command: Sequence[str]) -> dict[str, object]: ...

    def run_text(self, command: Sequence[str]) -> str: ...


class SubprocessRunner:
    def run_text(self, command: Sequence[str]) -> str:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ProcessError(
                f"command failed with exit {completed.returncode}: {detail}"
            )
        return completed.stdout.strip()

    def run_json(self, command: Sequence[str]) -> dict[str, object]:
        output = self.run_text(command)
        try:
            value = json.loads(output)
        except json.JSONDecodeError as error:
            raise ProcessError("command did not return JSON") from error
        if not isinstance(value, dict):
            raise ProcessError("command returned a non-object JSON value")
        return value


def run_streaming(
    command: Sequence[str],
    log_path: Path,
    *,
    environment: Mapping[str, str],
    timeout_seconds: float | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process_environment = os.environ.copy()
    process_environment.update(environment)
    secret_values = tuple(
        value.encode()
        for key, value in environment.items()
        if value
        and any(
            part in key.lower()
            for part in ("authorization", "password", "secret", "token", "api_key")
        )
    )
    with log_path.open("w", encoding="utf-8") as log:  # pragma: no mutate
        process = subprocess.Popen(
            command,
            env=process_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if process.stdout is None:
            raise ProcessError("cannot capture command output")
        stdout = cast(BinaryIO, process.stdout)
        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        writer = _RedactingWriter(log, secret_values)
        try:
            _drain_output(
                stdout,
                writer,
                deadline=deadline,
                command=command,
                timeout_seconds=timeout_seconds,
            )
            return process.wait(timeout=_remaining(deadline))
        except subprocess.TimeoutExpired as error:
            _stop_process_group(process)
            timeout_label = (
                "the configured deadline"
                if timeout_seconds is None
                else f"{timeout_seconds:g} seconds"
            )
            raise ProcessError(f"command timed out after {timeout_label}") from error
        except Exception as error:
            _stop_process_group(process)
            raise ProcessError("failed while capturing command output") from error
        finally:
            stdout.close()


class _RedactingWriter:
    def __init__(self, log: TextIO, secret_values: tuple[bytes, ...]) -> None:
        self.log = log
        self.secret_values = secret_values
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.overlap = max((len(secret) for secret in secret_values), default=1) - 1
        self.carry = b""

    def write(self, chunk: bytes, *, final: bool = False) -> None:
        self.carry += chunk
        flush_limit = (
            len(self.carry) if final else max(0, len(self.carry) - self.overlap)
        )
        safe_parts: list[bytes] = []
        position = 0
        while position < flush_limit:
            matches = [
                (index, -len(secret), secret)
                for secret in self.secret_values
                if 0 <= (index := self.carry.find(secret, position)) < flush_limit
            ]
            if not matches:
                safe_parts.append(self.carry[position:flush_limit])
                position = flush_limit
                break
            index, _, secret = min(matches)
            safe_parts.extend((self.carry[position:index], b"[REDACTED]"))
            position = index + len(secret)
        safe = b"".join(safe_parts)
        self.carry = self.carry[position:]
        text = self.decoder.decode(safe, final=final)
        print(text, end="", flush=True)
        self.log.write(text)
        self.log.flush()


def _drain_output(
    stdout: BinaryIO,
    writer: _RedactingWriter,
    *,
    deadline: float | None,
    command: Sequence[str],
    timeout_seconds: float | None,
) -> None:
    selector = selectors.DefaultSelector()
    selector.register(stdout, selectors.EVENT_READ)
    try:
        while True:
            remaining = _remaining(deadline)
            if remaining == 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds or 0)
            if not selector.select(timeout=remaining):
                if deadline is None:
                    continue
                raise subprocess.TimeoutExpired(command, timeout_seconds or 0)
            chunk = os.read(stdout.fileno(), 1024 * 1024)
            if chunk:
                writer.write(chunk)
            else:
                writer.write(b"", final=True)
                return
    finally:
        selector.close()


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    _signal_process_group(process.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)
    _signal_process_group(process.pid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1)


def _signal_process_group(process_id: int, signal_number: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process_id, signal_number)
