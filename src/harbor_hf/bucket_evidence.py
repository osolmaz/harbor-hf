from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from collections.abc import Iterable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast

from huggingface_hub import HfApi

from harbor_hf.coordination import (
    ClaimConflict,
    ClaimStore,
    bucket_evidence_claim_path,
    bucket_id,
)

_DOWNLOAD_BATCH_SIZE = 1024


class BucketEvidenceError(RuntimeError):
    """Raised when canonical evidence cannot be read safely from a Bucket."""


class BucketEvidenceApi(Protocol):
    def list_bucket_tree(
        self, bucket_id: str, prefix: str | None = None, **kwargs: object
    ) -> Iterable[object]: ...

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[object, str | Path]],
        **kwargs: object,
    ) -> None: ...


class BucketEvidenceWriterApi(Protocol):
    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[object, str | Path]],
        **kwargs: object,
    ) -> None: ...

    def get_bucket_paths_info(
        self, bucket_id: str, paths: Iterable[str], **kwargs: object
    ) -> Iterable[object]: ...

    def batch_bucket_files(
        self,
        bucket_id: str,
        *,
        add: list[tuple[bytes, str]],
        **kwargs: object,
    ) -> None: ...


class HubBucketEvidenceReader:
    """Read Bucket objects through the public SDK with a bounded local cache."""

    def __init__(
        self,
        cache_root: Path,
        *,
        api: BucketEvidenceApi | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.api = api or cast(BucketEvidenceApi, HfApi())
        self._listings: dict[tuple[str, str], dict[str, object]] = {}
        self._prefetched: set[tuple[str, str]] = set()

    def list_files(self, *, bucket: str, prefix: str) -> list[str]:
        key = (bucket_id(bucket), prefix.rstrip("/"))
        listing = self._listing(*key)
        return sorted(listing)

    def read_bytes(self, *, bucket: str, prefix: str, path: str) -> bytes:
        normalized_bucket = bucket_id(bucket)
        normalized_prefix = prefix.rstrip("/")
        listing = self._listing(normalized_bucket, normalized_prefix)
        remote = listing.get(path)
        if remote is None:
            raise BucketEvidenceError(f"Bucket evidence object is missing: {path}")
        destination = self._cache_path(normalized_bucket, normalized_prefix, path)
        if not destination.exists():
            self._prefetch(normalized_bucket, normalized_prefix, listing)
        try:
            return destination.read_bytes()
        except OSError as error:
            raise BucketEvidenceError(
                f"Bucket evidence object cannot be read: {path}"
            ) from error

    def prefetch_files(self, *, bucket: str, prefix: str, paths: list[str]) -> None:
        """Batch only caller-selected evidence files into the local cache."""
        normalized_bucket = bucket_id(bucket)
        normalized_prefix = prefix.rstrip("/")
        listing = self._listing(normalized_bucket, normalized_prefix)
        selected: dict[str, object] = {}
        for path in paths:
            remote = listing.get(path)
            if remote is None:
                raise BucketEvidenceError(f"Bucket evidence object is missing: {path}")
            selected[path] = remote
        self._download_missing(normalized_bucket, normalized_prefix, selected)

    def refresh(self) -> None:
        """Discard listings after another component has published new objects."""
        self._listings.clear()
        self._prefetched.clear()
        if self.cache_root.exists():
            for path in self.cache_root.iterdir():
                if path.is_file():
                    path.unlink()

    def _listing(self, bucket: str, prefix: str) -> dict[str, object]:
        key = (bucket, prefix)
        cached = self._listings.get(key)
        if cached is not None:
            return cached
        remote_prefix = f"{prefix}/"
        listing: dict[str, object] = {}
        items = self.api.list_bucket_tree(bucket, prefix=prefix, recursive=True)
        for item in items:
            if getattr(item, "type", None) != "file":
                continue
            remote_path = getattr(item, "path", None)
            if not isinstance(remote_path, str) or not remote_path.startswith(
                remote_prefix
            ):
                raise BucketEvidenceError("Bucket listing escaped its evidence prefix")
            relative = remote_path.removeprefix(remote_prefix)
            if not relative or relative in listing:
                raise BucketEvidenceError("Bucket listing contains invalid file paths")
            listing[relative] = item
        self._listings[key] = listing
        return listing

    def _prefetch(self, bucket: str, prefix: str, listing: dict[str, object]) -> None:
        key = (bucket, prefix)
        if key in self._prefetched:
            return
        self._download_missing(bucket, prefix, listing)
        self._prefetched.add(key)

    def _download_missing(
        self, bucket: str, prefix: str, listing: dict[str, object]
    ) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        missing = [
            (path, remote, self._cache_path(bucket, prefix, path))
            for path, remote in sorted(listing.items())
            if not self._cache_path(bucket, prefix, path).exists()
        ]
        for offset in range(0, len(missing), _DOWNLOAD_BATCH_SIZE):
            batch = missing[offset : offset + _DOWNLOAD_BATCH_SIZE]
            staged: list[tuple[Path, Path]] = []
            try:
                downloads: list[tuple[object, str | Path]] = []
                for _path, remote, destination in batch:
                    descriptor, temporary_name = tempfile.mkstemp(
                        prefix=f".{destination.name}-", dir=self.cache_root
                    )
                    os.close(descriptor)
                    temporary = Path(temporary_name)
                    temporary.unlink()
                    staged.append((temporary, destination))
                    downloads.append((remote, temporary))
                self.api.download_bucket_files(
                    bucket,
                    downloads,
                    raise_on_missing_files=True,
                )
                for temporary, destination in staged:
                    temporary.replace(destination)
            finally:
                for temporary, _destination in staged:
                    temporary.unlink(missing_ok=True)

    def _cache_path(self, bucket: str, prefix: str, path: str) -> Path:
        identity = hashlib.sha256(f"{bucket}/{prefix}/{path}".encode()).hexdigest()
        return self.cache_root / identity


class HubBucketEvidenceWriter:
    """Create immutable Bucket objects and adopt byte-identical retries."""

    def __init__(
        self,
        *,
        api: BucketEvidenceWriterApi | None = None,
        claims: ClaimStore | None = None,
    ) -> None:
        self.api = api or cast(BucketEvidenceWriterApi, HfApi())
        self.claims = claims

    def write_immutable(self, *, bucket: str, path: str, content: bytes) -> bool:
        normalized = bucket_id(bucket)
        if self.claims is None:
            return self._write_locked(normalized, path, content)
        claim_path = bucket_evidence_claim_path(normalized, path)
        owner = {
            "bucket": normalized,
            "path": path,
            "writer_id": uuid.uuid4().hex,
            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        }
        try:
            self.claims.acquire(claim_path, owner)
        except ClaimConflict as error:
            raise BucketEvidenceError(
                f"Bucket immutable evidence is being published: {path}"
            ) from error
        try:
            return self._write_locked(normalized, path, content)
        finally:
            with suppress(Exception):
                self.claims.release(claim_path, owner)

    def _write_locked(self, normalized: str, path: str, content: bytes) -> bool:
        observed = list(self.api.get_bucket_paths_info(normalized, [path]))
        if observed:
            if len(observed) != 1 or getattr(observed[0], "path", None) != path:
                raise BucketEvidenceError("Bucket immutable-path lookup is ambiguous")
            with tempfile.TemporaryDirectory(prefix="harbor-hf-bucket-") as name:
                destination = Path(name) / "object"
                self.api.download_bucket_files(
                    normalized,
                    [(observed[0], destination)],
                    raise_on_missing_files=True,
                )
                try:
                    existing = destination.read_bytes()
                except OSError as error:
                    raise BucketEvidenceError(
                        f"Bucket evidence object cannot be read: {path}"
                    ) from error
            if existing != content:
                raise BucketEvidenceError(
                    f"Bucket immutable evidence conflicts: {path}"
                )
            return False
        self.api.batch_bucket_files(normalized, add=[(content, path)])
        return True
