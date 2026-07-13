from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

from huggingface_hub import HfApi

from harbor_hf.coordination import bucket_id


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
        identity = hashlib.sha256(
            f"{normalized_bucket}/{normalized_prefix}/{path}".encode()
        ).hexdigest()
        destination = self.cache_root / identity
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.api.download_bucket_files(
                normalized_bucket,
                [(remote, destination)],
                raise_on_missing_files=True,
            )
        try:
            return destination.read_bytes()
        except OSError as error:
            raise BucketEvidenceError(
                f"Bucket evidence object cannot be read: {path}"
            ) from error

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
