from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
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
    "_secret_key",
    "_token",
)
_STREAM_CHUNK_SIZE = 1024 * 1024


def is_sensitive_key(key: object) -> bool:
    words = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(key))
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words).replace("-", "_").lower()
    collapsed = normalized.replace("_", "")
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith(_SENSITIVE_SUFFIXES)
        or normalized == "pat"
        or normalized.endswith("_pat")
        or collapsed.endswith("apikey")
        or collapsed.endswith("secretkey")
    )


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
    paths = [source, *_evidence_paths(source)]
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
        ) as archive,
    ):
        for path in paths:
            relative = path.relative_to(source)
            arcname = Path(source.name) / relative
            info = archive.gettarinfo(str(path), arcname.as_posix())
            if info is None:
                raise RuntimeError("unsupported special file in run evidence")
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() else 0o644
            if path.is_file():
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            else:
                archive.addfile(info)


def write_checksums(root: Path) -> dict[str, str]:
    excluded = {
        "checksums.json",
        "_SUCCESS",
        "_FAILED",
        "_CANCELLED",
        "_SELECTED",
    }
    checksums = {
        str(path.relative_to(root)): _sha256(path)
        for path in _evidence_paths(root)
        if path.is_file() and str(path.relative_to(root)) not in excluded
    }
    write_json(root / "checksums.json", checksums)
    return checksums


def verify_checksums(root: Path) -> dict[str, str]:
    """Verify that a finalized evidence tree is complete and unchanged."""
    checksum_path = root / "checksums.json"
    try:
        value = json.loads(checksum_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("evidence has no valid checksum manifest") from error
    if not isinstance(value, dict) or not all(
        isinstance(path, str)
        and isinstance(digest, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        for path, digest in value.items()
    ):
        raise RuntimeError("evidence checksum manifest is malformed")
    expected_paths = {
        str(path.relative_to(root))
        for path in _evidence_paths(root)
        if path.is_file()
        and str(path.relative_to(root))
        not in {
            "checksums.json",
            "_SUCCESS",
            "_FAILED",
            "_CANCELLED",
            "_SELECTED",
        }
    }
    if set(value) != expected_paths:
        raise RuntimeError("evidence checksum manifest does not cover exact contents")
    for relative, expected in value.items():
        candidate = root / relative
        if not candidate.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("evidence checksum path escapes its root")
        if _sha256(candidate) != expected:
            raise RuntimeError(f"evidence checksum mismatch: {relative}")
    return value


SecretValues = str | Iterable[str]


def assert_secret_absent(
    root: Path, secrets: SecretValues, *, allow_symlinks: bool = False
) -> None:
    for secret in _secret_values(secrets):
        needle = secret.encode()
        for path in _evidence_paths(root, allow_symlinks=allow_symlinks):
            if secret in str(path.relative_to(root)):
                raise RuntimeError("secret value found in artifact path")
            if path.is_symlink():
                if secret in os.readlink(path):
                    raise RuntimeError("secret value found in artifact symlink")
                continue
            if path.is_file() and _file_contains(path, needle):
                relative = path.relative_to(root)
                raise RuntimeError(f"secret value found in artifact {relative}")


def scrub_secret_paths(root: Path, secrets: SecretValues) -> int:
    changed = 0
    for secret in _secret_values(secrets):
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


def scrub_secret(
    root: Path, secrets: SecretValues, *, allow_symlinks: bool = False
) -> list[str]:
    changed: list[str] = []
    for secret in _secret_values(secrets):
        needle = secret.encode()
        for path in _evidence_paths(root, allow_symlinks=allow_symlinks):
            if path.is_symlink():
                if secret in os.readlink(path):
                    raise RuntimeError("secret value found in artifact symlink")
                continue
            if not path.is_file():
                continue
            relative = str(path.relative_to(root))
            if _scrub_file(path, needle) and relative not in changed:
                changed.append(relative)
    return sorted(changed)


def _secret_values(secrets: SecretValues) -> tuple[str, ...]:
    values = (secrets,) if isinstance(secrets, str) else tuple(secrets)
    return tuple(dict.fromkeys(value for value in values if value))


def _evidence_paths(root: Path, *, allow_symlinks: bool = False) -> list[Path]:
    paths = sorted(root.rglob("*"))
    if not allow_symlinks and any(path.is_symlink() for path in paths):
        raise RuntimeError("symbolic links are not allowed in run evidence")
    if any(
        not path.is_symlink() and not path.is_dir() and not path.is_file()
        for path in paths
    ):
        raise RuntimeError("special files are not allowed in run evidence")
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
