from __future__ import annotations

import json
from pathlib import Path

import pytest

from harbor_hf.worker import (
    WorkerError,
    _failure_details,
    _prepare_evidence_destination,
    _publish_evidence,
    _publish_success,
)


def _events(path: Path) -> list[dict[str, object]]:
    return [
        {key: value for key, value in json.loads(line).items() if key != "at"}
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_evidence_destination_creates_the_complete_reservation_contract(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "new-parent" / "nested-parent" / "run-contract"

    _prepare_evidence_destination(destination)

    assert destination.is_dir()
    assert sorted(path.name for path in destination.iterdir()) == ["_RESERVED"]
    assert (destination / "_RESERVED").read_text(encoding="utf-8") == "\n"


@pytest.mark.parametrize("marker", ["_FAILED", "_SUCCESS"])
def test_evidence_publication_orders_complete_tree_before_terminal_marker(
    tmp_path: Path, marker: str
) -> None:
    source = tmp_path / f"source-{marker}"
    source.mkdir()
    (source / marker).write_text(f"{marker}-contract\n", encoding="utf-8")
    (source / "root.json").write_text('{"root": true}\n', encoding="utf-8")
    nested = source / "nested"
    nested.mkdir()
    (nested / "_FAILED").write_text("nested-failed\n", encoding="utf-8")
    (nested / "_SUCCESS").write_text("nested-success\n", encoding="utf-8")
    (nested / "artifact.txt").write_text("artifact-contract\n", encoding="utf-8")
    destination = tmp_path / f"destination-{marker}"
    _prepare_evidence_destination(destination)

    _publish_evidence(source, destination)

    assert {
        str(path.relative_to(destination)): path.read_text(encoding="utf-8")
        for path in destination.rglob("*")
        if path.is_file()
    } == {
        marker: f"{marker}-contract\n",
        "root.json": '{"root": true}\n',
        "nested/_FAILED": "nested-failed\n",
        "nested/_SUCCESS": "nested-success\n",
        "nested/artifact.txt": "artifact-contract\n",
    }
    assert not (destination / "_RESERVED").exists()


def test_success_publication_failure_records_complete_redacted_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    events = root / "events.jsonl"

    def fail_finalization(path: Path, token: str) -> None:
        assert path == root
        assert token == "contract-token"
        raise RuntimeError("finalization failed with contract-token")

    monkeypatch.setattr("harbor_hf.worker._finalize_evidence", fail_finalization)

    with pytest.raises(WorkerError, match="^evidence finalization failed$"):
        _publish_success(root, events, "contract-token")

    assert _events(events) == [
        {"event": "run_succeeded"},
        {"event": "evidence_finalization_failed", "error": "RuntimeError"},
    ]
    assert json.loads((root / "_FAILED").read_text(encoding="utf-8")) == {
        "error_type": "RuntimeError",
        "message": "finalization failed with [REDACTED]",
    }
    assert not (root / "_SUCCESS").exists()
    assert "contract-token" not in events.read_text(encoding="utf-8")
    assert "contract-token" not in (root / "_FAILED").read_text(encoding="utf-8")


def test_failure_details_preserve_primary_cleanup_and_redaction_contract() -> None:
    primary = ValueError("primary contract-token")
    cleanup = RuntimeError("cleanup contract-token")

    failure, record, event, reported = _failure_details(
        primary, cleanup, "contract-token"
    )

    assert failure is primary
    assert record == {
        "error_type": "ValueError",
        "message": "primary [REDACTED]",
        "cleanup_error": {
            "error_type": "RuntimeError",
            "message": "cleanup [REDACTED]",
        },
    }
    assert event == {
        "error_type": "ValueError",
        "cleanup_error_type": "RuntimeError",
    }
    assert reported == (
        "primary [REDACTED]; endpoint cleanup failed: cleanup [REDACTED]"
    )
