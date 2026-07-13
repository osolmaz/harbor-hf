import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from harbor_hf.evidence import (
    append_event,
    archive_directory,
    assert_secret_absent,
    redact,
    scrub_secret,
    write_checksums,
    write_json,
)


def test_redact_removes_nested_sensitive_values() -> None:
    value = {
        "token": "one",
        "nested": [{"password_value": "two", "ok": 3}],
        "tokenizer": "qwen",
        "tokenizer_config": {"max_tokens": 8192},
    }

    assert redact(value) == {
        "token": "[REDACTED]",
        "nested": [{"password_value": "[REDACTED]", "ok": 3}],
        "tokenizer": "qwen",
        "tokenizer_config": {"max_tokens": 8192},
    }


def test_checksums_cover_regular_evidence(tmp_path: Path) -> None:
    (tmp_path / "record.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "_SUCCESS").write_text("\n", encoding="utf-8")

    checksums = write_checksums(tmp_path)

    assert list(checksums) == ["record.json"]
    assert checksums["record.json"] == (
        "sha256:ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
    )
    assert json.loads((tmp_path / "checksums.json").read_text()) == checksums


def test_secret_scan_names_the_bad_artifact(tmp_path: Path) -> None:
    (tmp_path / "log.txt").write_text("prefix secret-value suffix")

    with pytest.raises(RuntimeError, match="log.txt"):
        assert_secret_absent(tmp_path, "secret-value")

    assert scrub_secret(tmp_path, "secret-value") == ["log.txt"]
    assert_secret_absent(tmp_path, "secret-value")
    assert "[REDACTED]" in (tmp_path / "log.txt").read_text()


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
    assert scrub_secret(source, "missing") == []

    with tarfile.open(destination, "r:gz") as archive:
        assert archive.getnames() == ["source", "source/value.txt"]
