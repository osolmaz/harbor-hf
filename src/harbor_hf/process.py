from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol


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
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process_environment = os.environ.copy()
    process_environment.update(environment)
    secret_values = tuple(
        value
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
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.stdout is None:
            raise ProcessError("cannot capture command output")
        for line in process.stdout:
            safe_line = line
            for secret in secret_values:
                safe_line = safe_line.replace(secret, "[REDACTED]")
            print(safe_line, end="", flush=True)
            log.write(safe_line)
            log.flush()
        return process.wait()
