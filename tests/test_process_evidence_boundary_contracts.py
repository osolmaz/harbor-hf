from __future__ import annotations

import hashlib
import io
import json
import signal
import stat
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import harbor_hf.evidence as evidence
import harbor_hf.process as process
from harbor_hf.evidence import (
    assert_secret_absent,
    is_sensitive_key,
    redact,
    scrub_secret,
    scrub_secret_paths,
    verify_checksums,
    write_checksums,
)
from harbor_hf.process import ProcessError, _RedactingWriter, run_streaming


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("authorization", True),
        ("Authorization", True),
        ("api-key", True),
        ("serviceApiKey", True),
        ("SERVICE_API_KEY", True),
        ("accessKey", True),
        ("client_ai_key", True),
        ("passwordValue", True),
        ("ssh_private_key", True),
        ("signingSecret", True),
        ("signingSecretKey", True),
        ("refreshToken", True),
        ("pat", True),
        ("github_pat", True),
        ("token_count", False),
        ("secretary", False),
        ("monkey", False),
        ("public_key", False),
        ("credential_count", False),
        (17, False),
    ],
)
def test_sensitive_key_normalization_contract(key: object, expected: bool) -> None:
    assert is_sensitive_key(key) is expected


def test_redact_recurses_only_through_mappings_and_lists() -> None:
    source = {
        "outer": [
            {"accessToken": "one", "safe": "visible"},
            ("tuple", {"password": "not-walked"}),
        ],
        7: {"api_key": "two"},
    }

    assert redact(source) == {
        "outer": [
            {"accessToken": "[REDACTED]", "safe": "visible"},
            ("tuple", {"password": "not-walked"}),
        ],
        "7": {"api_key": "[REDACTED]"},
    }
    assert source["outer"][0] == {"accessToken": "one", "safe": "visible"}


def test_checksums_cover_exact_nested_files_and_exclude_root_markers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    nested = root / "nested"
    nested.mkdir(parents=True)
    payloads = {
        "alpha.txt": b"alpha\n",
        "nested/beta.bin": bytes(range(32)),
        "nested/_SUCCESS": b"nested marker\n",
    }
    for relative, payload in payloads.items():
        (root / relative).write_bytes(payload)
    for marker in ("_SUCCESS", "_FAILED", "_CANCELLED"):
        (root / marker).write_text(marker, encoding="utf-8")
    (root / "checksums.json").write_text("stale", encoding="utf-8")

    result = write_checksums(root)

    assert result == {
        relative: f"sha256:{hashlib.sha256(payload).hexdigest()}"
        for relative, payload in payloads.items()
    }
    assert json.loads((root / "checksums.json").read_text(encoding="utf-8")) == result
    assert verify_checksums(root) == result


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"file.txt": 3},
        {"file.txt": "0" * 64},
        {"file.txt": "sha256:" + "A" * 64},
        {"file.txt": "sha256:" + "0" * 63},
        {"file.txt": "sha512:" + "0" * 64},
    ],
)
def test_verify_checksums_rejects_each_malformed_manifest_shape(
    tmp_path: Path, manifest: object
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "file.txt").write_text("payload", encoding="utf-8")
    (root / "checksums.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="^evidence checksum manifest is malformed$"):
        verify_checksums(root)


def test_verify_checksums_rejects_missing_extra_changed_and_escaping_entries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    payload = root / "payload.txt"
    payload.write_text("original", encoding="utf-8")
    checksums = write_checksums(root)

    (root / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="^evidence checksum manifest does not cover exact contents$"
    ):
        verify_checksums(root)
    (root / "extra.txt").unlink()

    payload.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="^evidence checksum mismatch: payload.txt$"):
        verify_checksums(root)
    payload.write_text("original", encoding="utf-8")

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    checksums["../outside.txt"] = evidence._sha256(outside)
    checksums.pop("payload.txt")
    (root / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="^evidence checksum manifest does not cover exact contents$"
    ):
        verify_checksums(root)


def test_secret_detection_and_scrubbing_cross_every_chunk_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evidence, "_STREAM_CHUNK_SIZE", 4)
    root = tmp_path / "evidence"
    root.mkdir()
    secret = "secret-value"
    payload = b"prefix--secret-value--middle--secret-value--suffix"
    artifact = root / "artifact.bin"
    artifact.write_bytes(payload)
    artifact.chmod(0o640)

    with pytest.raises(
        RuntimeError, match="^secret value found in artifact artifact.bin$"
    ):
        assert_secret_absent(root, secret)

    assert scrub_secret(root, secret) == ["artifact.bin"]
    assert artifact.read_bytes() == (b"prefix--[REDACTED]--middle--[REDACTED]--suffix")
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o640
    assert scrub_secret(root, secret) == []
    assert_secret_absent(root, secret)


def test_secret_path_scrubbing_is_deepest_first_and_counts_each_rename(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    leaf = root / "token-dir" / "nested-token.txt"
    leaf.parent.mkdir(parents=True)
    leaf.write_text("safe", encoding="utf-8")

    assert scrub_secret_paths(root, "token") == 2
    renamed = root / "[REDACTED]-dir" / "nested-[REDACTED].txt"
    assert renamed.read_text(encoding="utf-8") == "safe"
    assert_secret_absent(root, "token")


def test_secret_path_scrubbing_refuses_to_overwrite_existing_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "my-token.txt").write_text("source", encoding="utf-8")
    (root / "my-[REDACTED].txt").write_text("existing", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="^secret path redaction would overwrite an artifact$",
    ):
        scrub_secret_paths(root, "token")
    assert (root / "my-token.txt").read_text(encoding="utf-8") == "source"
    assert (root / "my-[REDACTED].txt").read_text(encoding="utf-8") == "existing"


def test_evidence_operations_reject_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    target = tmp_path / "target"
    target.write_text("payload", encoding="utf-8")
    (root / "link").symlink_to(target)

    operations = [
        lambda: write_checksums(root),
        lambda: assert_secret_absent(root, "secret"),
        lambda: scrub_secret(root, "secret"),
        lambda: scrub_secret_paths(root, "secret"),
    ]
    for operation in operations:
        with pytest.raises(
            RuntimeError, match="^symbolic links are not allowed in run evidence$"
        ):
            operation()


@pytest.mark.parametrize("split", range(len(b"boundary-secret") + 1))
def test_redacting_writer_hides_secret_across_every_write_boundary(
    split: int, capsys: pytest.CaptureFixture[str]
) -> None:
    output = io.StringIO()
    writer = _RedactingWriter(output, (b"boundary-secret",))
    payload = b"before-boundary-secret-after\n"
    boundary = len(b"before-") + split

    writer.write(payload[:boundary])
    writer.write(payload[boundary:])
    writer.write(b"", final=True)

    assert output.getvalue() == "before-[REDACTED]-after\n"
    assert capsys.readouterr().out == "before-[REDACTED]-after\n"


def test_redacting_writer_uses_earliest_then_longest_match(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = io.StringIO()
    writer = _RedactingWriter(output, (b"token", b"token-long", b"later"))

    writer.write(b"token-long / later / token")
    writer.write(b"", final=True)

    assert output.getvalue() == "[REDACTED] / [REDACTED] / [REDACTED]"
    assert capsys.readouterr().out == output.getvalue()


@pytest.mark.parametrize(
    ("environment_key", "value"),
    [
        ("AUTHORIZATION", "auth-value"),
        ("DB_PASSWORD", "password-value"),
        ("CLIENT_SECRET", "secret-value"),
        ("ACCESS_TOKEN", "token-value"),
        ("SERVICE_API_KEY", "api-key-value"),
    ],
)
def test_streaming_redacts_each_sensitive_environment_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    environment_key: str,
    value: str,
) -> None:
    log = tmp_path / f"{environment_key}.log"
    command = [
        "python",
        "-c",
        (
            "import os,sys; "
            "sys.stdout.write('before-' + os.environ[sys.argv[1]] + '-after')"
        ),
        environment_key,
    ]

    assert run_streaming(command, log, environment={environment_key: value}) == 0
    assert log.read_text(encoding="utf-8") == "before-[REDACTED]-after"
    assert capsys.readouterr().out == "before-[REDACTED]-after"


class _FakeProcess:
    def __init__(self, waits: Sequence[object]) -> None:
        self.pid = 4321
        self._waits = list(waits)
        self.timeouts: list[float] = []

    def wait(self, timeout: float | None = None) -> int:
        assert timeout is not None
        self.timeouts.append(timeout)
        outcome = self._waits.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return cast(int, outcome)


def test_stop_process_group_always_sends_term_then_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeProcess([0, 0])
    signals: list[tuple[int, int]] = []

    def record(process_id: int, signal_number: int) -> None:
        signals.append((process_id, signal_number))

    monkeypatch.setattr(process, "_signal_process_group", record)

    process._stop_process_group(cast(Any, fake))

    assert fake.timeouts == [10, 1]
    assert signals == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]


def test_stop_process_group_escalates_after_both_wait_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeProcess(
        [
            subprocess.TimeoutExpired(["command"], 10),
            subprocess.TimeoutExpired(["command"], 1),
        ]
    )
    signals: list[tuple[int, int]] = []

    def record(process_id: int, signal_number: int) -> None:
        signals.append((process_id, signal_number))

    monkeypatch.setattr(process, "_signal_process_group", record)

    process._stop_process_group(cast(Any, fake))

    assert fake.timeouts == [10, 1]
    assert signals == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]


def test_signal_process_group_suppresses_only_missing_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def missing(process_id: int, signal_number: int) -> None:
        calls.append((process_id, signal_number))
        raise ProcessLookupError

    monkeypatch.setattr(process.os, "killpg", missing)
    process._signal_process_group(91, signal.SIGTERM)
    assert calls == [(91, signal.SIGTERM)]

    monkeypatch.setattr(
        process.os,
        "killpg",
        lambda process_id, signal_number: (_ for _ in ()).throw(PermissionError()),
    )
    with pytest.raises(PermissionError):
        process._signal_process_group(91, signal.SIGKILL)


def test_streaming_timeout_terminates_and_reports_exact_deadline(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProcessError, match="^command timed out after 0.05 seconds$"):
        run_streaming(
            [
                "python",
                "-c",
                "import time; print('started', flush=True); time.sleep(60)",
            ],
            tmp_path / "timeout.log",
            environment={},
            timeout_seconds=0.05,
        )
    assert (tmp_path / "timeout.log").read_text(encoding="utf-8") == "started\n"


def test_remaining_clamps_at_zero_and_preserves_no_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process.time, "monotonic", lambda: 10.25)
    assert process._remaining(None) is None
    assert process._remaining(15.5) == 5.25
    assert process._remaining(10.25) == 0.0
    assert process._remaining(1.0) == 0.0


def test_run_streaming_wraps_capture_failure_and_stops_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stopped: list[object] = []
    fake = SimpleNamespace(
        stdout=io.BytesIO(b"output"),
        wait=lambda timeout=None: 0,
    )
    monkeypatch.setattr(process.subprocess, "Popen", lambda *args, **kwargs: fake)
    monkeypatch.setattr(
        process,
        "_drain_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("capture broke")),
    )
    monkeypatch.setattr(process, "_stop_process_group", stopped.append)

    with pytest.raises(
        ProcessError, match="^failed while capturing command output$"
    ) as captured:
        run_streaming(["command"], tmp_path / "capture.log", environment={})

    assert isinstance(captured.value.__cause__, OSError)
    assert stopped == [fake]
