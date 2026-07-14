import json
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from harbor_hf.evidence import (
    _file_contains,
    _scrub_file,
    append_event,
    archive_directory,
    assert_secret_absent,
    redact,
    scrub_secret,
    scrub_secret_paths,
    verify_checksums,
    write_checksums,
    write_json,
)


def test_redact_removes_nested_sensitive_values() -> None:
    value = {
        "token": "one",
        "nested": [
            {
                "password_value": "two",
                "OPENAI_API_KEY": "three",
                "awsAccessKey": "four",
                "openAIKey": "five",
                "GITHUB_PAT": "six",
                "ok": 3,
            }
        ],
        "tokenizer": "qwen",
        "tokenizer_config": {"max_tokens": 8192},
    }

    assert redact(value) == {
        "token": "[REDACTED]",
        "nested": [
            {
                "password_value": "[REDACTED]",
                "OPENAI_API_KEY": "[REDACTED]",
                "awsAccessKey": "[REDACTED]",
                "openAIKey": "[REDACTED]",
                "GITHUB_PAT": "[REDACTED]",
                "ok": 3,
            }
        ],
        "tokenizer": "qwen",
        "tokenizer_config": {"max_tokens": 8192},
    }


def test_checksums_cover_regular_evidence(tmp_path: Path) -> None:
    (tmp_path / "record.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "checksums.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "_SUCCESS").write_text("\n", encoding="utf-8")
    (tmp_path / "_FAILED").write_text("\n", encoding="utf-8")

    checksums = write_checksums(tmp_path)

    assert list(checksums) == ["record.json"]
    assert checksums["record.json"] == (
        "sha256:ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
    )
    assert json.loads((tmp_path / "checksums.json").read_text()) == checksums


def test_checksums_include_nested_terminal_and_checksum_files(tmp_path: Path) -> None:
    nested = tmp_path / "harbor" / "trial"
    nested.mkdir(parents=True)
    for name in ("checksums.json", "_SUCCESS", "_FAILED"):
        (nested / name).write_text(f"{name}\n", encoding="utf-8")

    checksums = write_checksums(tmp_path)

    assert set(checksums) == {
        "harbor/trial/_FAILED",
        "harbor/trial/_SUCCESS",
        "harbor/trial/checksums.json",
    }


def test_verify_checksums_rejects_tampering_and_untracked_files(
    tmp_path: Path,
) -> None:
    record = tmp_path / "record.json"
    record.write_text("{}\n", encoding="utf-8")
    expected = write_checksums(tmp_path)

    assert verify_checksums(tmp_path) == expected
    record.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_checksums(tmp_path)

    record.write_text("{}\n", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not cover exact contents"):
        verify_checksums(tmp_path)


def test_secret_scan_names_the_bad_artifact(tmp_path: Path) -> None:
    (tmp_path / "log.txt").write_text("prefix secret-value suffix")

    with pytest.raises(RuntimeError, match="log.txt"):
        assert_secret_absent(tmp_path, "secret-value")

    assert scrub_secret(tmp_path, "secret-value") == ["log.txt"]
    assert_secret_absent(tmp_path, "secret-value")
    assert "[REDACTED]" in (tmp_path / "log.txt").read_text()


def test_secret_path_components_are_redacted_deepest_first(tmp_path: Path) -> None:
    directory = tmp_path / "secret-value-directory"
    directory.mkdir()
    (directory / "secret-value.log").write_text("safe", encoding="utf-8")
    (tmp_path / "zzz-unrelated.log").write_text("safe", encoding="utf-8")

    assert scrub_secret_paths(tmp_path, "secret-value") == 2

    redacted = tmp_path / "[REDACTED]-directory" / "[REDACTED].log"
    assert redacted.read_text(encoding="utf-8") == "safe"
    assert_secret_absent(tmp_path, "secret-value")


def test_secret_scan_rejects_path_before_content(tmp_path: Path) -> None:
    (tmp_path / "secret-value.log").write_text("safe", encoding="utf-8")

    with pytest.raises(RuntimeError, match="^secret value found in artifact path$"):
        assert_secret_absent(tmp_path, "secret-value")


def test_secret_scan_detects_content_at_first_byte(tmp_path: Path) -> None:
    (tmp_path / "log.txt").write_text("secret-value suffix", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact log.txt"):
        assert_secret_absent(tmp_path, "secret-value")


def test_secret_path_redaction_rejects_collision(tmp_path: Path) -> None:
    (tmp_path / "secret-value.log").write_text("one", encoding="utf-8")
    (tmp_path / "[REDACTED].log").write_text("two", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="^secret path redaction would overwrite an artifact$",
    ):
        scrub_secret_paths(tmp_path, "secret-value")


def test_empty_secret_changes_no_paths(tmp_path: Path) -> None:
    assert scrub_secret_paths(tmp_path, "") == 0


def test_evidence_operations_reject_symlinks(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret-value", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)

    with pytest.raises(RuntimeError, match="^symbolic links are not allowed"):
        scrub_secret(root, "secret-value")
    with pytest.raises(RuntimeError, match="^symbolic links are not allowed"):
        archive_directory(root, tmp_path / "archive.tar.gz")

    assert outside.read_text(encoding="utf-8") == "secret-value"


def test_archive_is_deterministic_and_preserves_relative_tree(tmp_path: Path) -> None:
    source = tmp_path / "harbor-jobs"
    nested = source / "job" / "trial"
    nested.mkdir(parents=True)
    artifact = nested / "session.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    archive_directory(source, first)
    artifact.touch()
    archive_directory(source, second)

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        assert "harbor-jobs/job/trial/session.jsonl" in archive.getnames()


def test_secret_scrubbing_streams_across_chunk_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "log.txt"
    path.write_bytes(b"abcsecret-value-1secret-value-tail")
    monkeypatch.setattr("harbor_hf.evidence._STREAM_CHUNK_SIZE", 4)

    assert scrub_secret(tmp_path, "secret-value") == ["log.txt"]
    assert path.read_bytes() == b"abc[REDACTED]-1[REDACTED]-tail"
    assert_secret_absent(tmp_path, "secret-value")


def test_bounded_stream_helpers_report_changed_and_unchanged_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "log.txt"
    path.write_bytes(b"abcsecret-value-tail")
    monkeypatch.setattr("harbor_hf.evidence._STREAM_CHUNK_SIZE", 4)

    assert _file_contains(path, b"secret-value") is True
    assert _file_contains(path, b"XXXXa") is False
    assert _scrub_file(path, b"secret-value") is True
    assert path.read_bytes() == b"abc[REDACTED]-tail"
    assert _scrub_file(path, b"missing") is False


def test_stream_helpers_use_bounded_reads_and_private_temporary_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "log.txt"
    path.write_bytes(b"abcdef")
    read_sizes: list[int] = []
    prefixes: list[str | None] = []
    original_open: Any = Path.open
    original_mkstemp = tempfile.mkstemp

    class TrackingReader:
        def __init__(self, stream: BinaryIO) -> None:
            self.stream = stream

        def __enter__(self) -> "TrackingReader":
            return self

        def __exit__(self, *_args: object) -> None:
            self.stream.close()

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self.stream.read(size)

    def tracking_open(
        value: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        stream = original_open(value, mode, *args, **kwargs)
        if value == path and mode == "rb":
            return TrackingReader(stream)
        return stream

    def tracking_mkstemp(*, prefix: str | None, dir: Path) -> tuple[int, str]:
        prefixes.append(prefix)
        return original_mkstemp(prefix=prefix, dir=dir)

    monkeypatch.setattr("harbor_hf.evidence._STREAM_CHUNK_SIZE", 2)
    monkeypatch.setattr(Path, "open", tracking_open)
    monkeypatch.setattr("harbor_hf.evidence.tempfile.mkstemp", tracking_mkstemp)

    assert _file_contains(path, b"missing") is False
    assert _scrub_file(path, b"missing") is False
    assert set(read_sizes) == {2}
    assert prefixes == []


def test_json_and_event_writers_create_nested_canonical_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixedDateTime:
        @staticmethod
        def now(_timezone: object) -> datetime:
            return datetime(2026, 7, 13, 1, 2, 3, tzinfo=UTC)

    monkeypatch.setattr("harbor_hf.evidence.datetime", FixedDateTime)
    json_path = tmp_path / "nested" / "record.json"
    event_path = tmp_path / "more" / "events.jsonl"

    write_json(json_path, {"z": 1, "a": datetime(2026, 1, 1, tzinfo=UTC)})
    append_event(event_path, "started", run_id="run-1")

    assert json_path.read_text() == (
        '{\n  "a": "2026-01-01 00:00:00+00:00",\n  "z": 1\n}\n'
    )
    assert event_path.read_text() == (
        '{"at": "2026-07-13T01:02:03+00:00", "event": "started", "run_id": "run-1"}\n'
    )


def test_archive_and_empty_secret_operations(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.txt").write_text("value", encoding="utf-8")
    destination = tmp_path / "archive.tar.gz"

    archive_directory(source, destination)
    assert_secret_absent(source, "")
    assert scrub_secret(source, "") == []
    assert scrub_secret_paths(source, "") == 0
    assert scrub_secret(source, "missing") == []

    with tarfile.open(destination, "r:gz") as archive:
        assert archive.getnames() == ["source", "source/value.txt"]


def test_scrub_missing_secret_does_not_create_temporary_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "large.bin").write_bytes(b"ordinary evidence")

    def fail_mkstemp(**_kwargs: object) -> tuple[int, str]:
        raise AssertionError("temporary copy should not be created")

    monkeypatch.setattr("harbor_hf.evidence.tempfile.mkstemp", fail_mkstemp)

    assert scrub_secret(tmp_path, "missing-secret") == []
