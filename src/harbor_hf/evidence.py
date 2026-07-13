from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "credential",
        "credentials",
        "api_key",
        "access_key",
        "password",
        "password_value",
        "private_key",
        "secret",
        "secrets",
        "token",
    }
)
_SENSITIVE_SUFFIXES = ("_credential", "_password", "_secret", "_token")


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).replace("-", "_").lower()
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def append_event(path: Path, event: str, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": datetime.now(UTC).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def redact(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): ("[REDACTED]" if _is_sensitive_key(key) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def archive_directory(source: Path, destination: Path) -> None:
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(source, arcname=source.name)


def write_checksums(root: Path) -> dict[str, str]:
    excluded = {"checksums.json", "_SUCCESS", "_FAILED"}
    checksums = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    write_json(root / "checksums.json", checksums)
    return checksums


def assert_secret_absent(root: Path, secret: str) -> None:
    if not secret:
        return
    needle = secret.encode()
    for path in root.rglob("*"):
        if path.is_file() and path.read_bytes().find(needle) >= 0:
            relative = path.relative_to(root)
            raise RuntimeError(f"secret value found in artifact {relative}")


def scrub_secret(root: Path, secret: str) -> list[str]:
    if not secret:
        return []
    needle = secret.encode()
    changed: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_bytes()
        if needle not in content:
            continue
        path.write_bytes(content.replace(needle, b"[REDACTED]"))
        changed.append(str(path.relative_to(root)))
    return changed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
