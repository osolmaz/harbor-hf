from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import tempfile
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
_SENSITIVE_SUFFIXES = (
    "_access_key",
    "_ai_key",
    "_api_key",
    "_credential",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)
_STREAM_CHUNK_SIZE = 1024 * 1024


def is_sensitive_key(key: object) -> bool:
    words = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(key))
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words).replace("-", "_").lower()
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
            str(key): ("[REDACTED]" if is_sensitive_key(key) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def archive_directory(source: Path, destination: Path) -> None:
    _evidence_paths(source)
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(source, arcname=source.name)


def write_checksums(root: Path) -> dict[str, str]:
    excluded = {"checksums.json", "_SUCCESS", "_FAILED"}
    checksums = {
        str(path.relative_to(root)): _sha256(path)
        for path in _evidence_paths(root)
        if path.is_file() and path.name not in excluded
    }
    write_json(root / "checksums.json", checksums)
    return checksums


def assert_secret_absent(root: Path, secret: str) -> None:
    if not secret:
        return
    needle = secret.encode()
    for path in _evidence_paths(root):
        if secret in str(path.relative_to(root)):
            raise RuntimeError("secret value found in artifact path")
        if path.is_file() and _file_contains(path, needle):
            relative = path.relative_to(root)
            raise RuntimeError(f"secret value found in artifact {relative}")


def scrub_secret_paths(root: Path, secret: str) -> int:
    if not secret:
        return 0
    changed = 0
    paths = sorted(
        _evidence_paths(root), key=lambda path: len(path.parts), reverse=True
    )
    for path in paths:
        if secret not in path.name:
            continue
        destination = path.with_name(path.name.replace(secret, "[REDACTED]"))
        if destination.exists():
            raise RuntimeError("secret path redaction would overwrite an artifact")
        path.rename(destination)
        changed += 1
    return changed


def scrub_secret(root: Path, secret: str) -> list[str]:
    if not secret:
        return []
    needle = secret.encode()
    changed: list[str] = []
    for path in _evidence_paths(root):
        if not path.is_file():
            continue
        if _scrub_file(path, needle):
            changed.append(str(path.relative_to(root)))
    return changed


def _evidence_paths(root: Path) -> list[Path]:
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("symbolic links are not allowed in run evidence")
    return paths


def _file_contains(path: Path, needle: bytes) -> bool:
    overlap = max(len(needle) - 1, 0)
    carry = b""
    with path.open("rb") as stream:
        while chunk := stream.read(_STREAM_CHUNK_SIZE):
            data = carry + chunk
            if needle in data:
                return True
            carry = data[-overlap:] if overlap else b""
    return False


def _scrub_file(path: Path, needle: bytes) -> bool:
    if not _file_contains(path, needle):
        return False
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".harbor-hf-redact-", dir=path.parent
    )
    temporary = Path(temporary_name)
    changed = False
    replacement = b"[REDACTED]"
    try:
        with path.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
            carry = b""
            while chunk := source.read(_STREAM_CHUNK_SIZE):
                carry += chunk
                while (position := carry.find(needle)) >= 0:
                    destination.write(carry[:position])
                    destination.write(replacement)
                    carry = carry[position + len(needle) :]
                    changed = True
                flush_length = max(0, len(carry) - len(needle) + 1)
                destination.write(carry[:flush_length])
                carry = carry[flush_length:]
            destination.write(carry)
        if changed:
            os.chmod(temporary, path.stat().st_mode)
            os.replace(temporary, path)
        return changed
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
