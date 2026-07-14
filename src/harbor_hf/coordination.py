from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.errors import HfHubHTTPError

_REPOSITORY_NAME = "harbor-hf-coordination"
_MAX_COMMIT_ATTEMPTS = 8


class CoordinationError(RuntimeError):
    """Raised when a distributed claim cannot be safely changed."""


class ClaimConflict(CoordinationError):
    """Raised when a distributed claim is already held."""


class CoordinationApi(Protocol):
    def repo_info(self, repo_id: str, **kwargs: object) -> object: ...

    def get_paths_info(
        self, repo_id: str, paths: list[str] | str, **kwargs: object
    ) -> list[object]: ...

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: object) -> str: ...

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object: ...


class ClaimStore(Protocol):
    def acquire(self, path: str, owner: Mapping[str, str]) -> None: ...

    def release(self, path: str, owner: Mapping[str, str]) -> None: ...


def coordination_repository(namespace: str) -> str:
    return f"{namespace}/{_REPOSITORY_NAME}"


def endpoint_claim_path(namespace: str, name: str) -> str:
    identity = hashlib.sha256(f"{namespace}/{name}".encode()).hexdigest()[:32]
    return f"endpoint-leases/{identity}.json"


def bucket_id(bucket: str) -> str:
    return bucket.removeprefix("hf://buckets/").removeprefix("buckets/")


def run_claim_path(artifact_bucket: str, artifact_prefix: str) -> str:
    target = f"{bucket_id(artifact_bucket)}/{artifact_prefix}".encode()
    identity = hashlib.sha256(target).hexdigest()
    return f"run-reservations/{identity}.json"


class HubClaimStore:
    """Serialize claims through optimistic commits to a private Hub repository."""

    def __init__(
        self,
        namespace: str,
        token: str,
        *,
        api: CoordinationApi | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.repository = coordination_repository(namespace)
        self.token = token
        self.api = api or cast(CoordinationApi, HfApi(token=token))
        self.clock = clock

    def acquire(self, path: str, owner: Mapping[str, str]) -> None:
        payload = _owner_payload(owner)
        for _attempt in range(_MAX_COMMIT_ATTEMPTS):
            head = self._head()
            if self._exists(path, head):
                observed = self._read_owner(path, head)
                if not _claim_expired(observed, self.clock()):
                    raise ClaimConflict(f"claim is already held: {path}")
            try:
                self.api.create_commit(
                    self.repository,
                    [CommitOperationAdd(path_in_repo=path, path_or_fileobj=payload)],
                    commit_message=f"chore: acquire {path}",
                    repo_type="dataset",
                    revision="main",
                    parent_commit=head,
                    token=self.token,
                )
                return
            except HfHubHTTPError as error:
                if not _is_parent_conflict(error):
                    raise
        raise CoordinationError("coordination repository remained contended")

    def release(self, path: str, owner: Mapping[str, str]) -> None:
        expected = dict(owner)
        for _attempt in range(_MAX_COMMIT_ATTEMPTS):
            head = self._head()
            observed = self._read_owner(path, head)
            if observed != expected:
                raise CoordinationError("claim ownership cannot be verified")
            try:
                self.api.create_commit(
                    self.repository,
                    [CommitOperationDelete(path_in_repo=path)],
                    commit_message=f"chore: release {path}",
                    repo_type="dataset",
                    revision="main",
                    parent_commit=head,
                    token=self.token,
                )
                return
            except HfHubHTTPError as error:
                if not _is_parent_conflict(error):
                    raise
        raise CoordinationError("coordination repository remained contended")

    def _head(self) -> str:
        info = self.api.repo_info(
            self.repository,
            repo_type="dataset",
            revision="main",
            token=self.token,
        )
        revision = getattr(info, "sha", None)
        if not isinstance(revision, str) or not revision:
            raise CoordinationError("coordination repository has no commit identity")
        return revision

    def _exists(self, path: str, revision: str) -> bool:
        return bool(
            self.api.get_paths_info(
                self.repository,
                path,
                repo_type="dataset",
                revision=revision,
                token=self.token,
            )
        )

    def _read_owner(self, path: str, revision: str) -> dict[str, object]:
        if not self._exists(path, revision):
            raise CoordinationError("claim no longer exists")
        local_path = self.api.hf_hub_download(
            self.repository,
            path,
            repo_type="dataset",
            revision=revision,
            token=self.token,
        )
        try:
            value = json.loads(Path(local_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CoordinationError("claim owner cannot be read") from error
        if not isinstance(value, dict):
            raise CoordinationError("claim owner is invalid")
        return value


def _owner_payload(owner: Mapping[str, str]) -> bytes:
    return (json.dumps(dict(owner), sort_keys=True) + "\n").encode()


def _claim_expired(owner: Mapping[str, object], now: datetime) -> bool:
    value = owner.get("expires_at")
    if not isinstance(value, str):
        return False
    try:
        expires_at = datetime.fromisoformat(value)
    except ValueError:
        return False
    return expires_at.tzinfo is not None and expires_at <= now.astimezone(UTC)


def _is_parent_conflict(error: HfHubHTTPError) -> bool:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) in {409, 412}
