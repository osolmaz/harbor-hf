from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import HfHubHTTPError

from harbor_hf.coordination import (
    ClaimConflict,
    CoordinationError,
    HubClaimStore,
    coordination_repository,
    endpoint_claim_path,
    run_claim_path,
)


class FakeCoordinationApi:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.generation = 1
        self.files: dict[str, bytes] = {}
        self.conflicts = 0
        self.commits: list[dict[str, object]] = []

    @property
    def head(self) -> str:
        return f"{self.generation:040x}"

    def repo_info(self, repo_id: str, **kwargs: object) -> object:
        assert repo_id == "org/harbor-hf-coordination"
        assert kwargs == {
            "repo_type": "dataset",
            "revision": "main",
            "token": "token",
        }
        return SimpleNamespace(sha=self.head)

    def get_paths_info(
        self, repo_id: str, paths: list[str] | str, **kwargs: object
    ) -> list[object]:
        assert repo_id == "org/harbor-hf-coordination"
        assert kwargs == {
            "repo_type": "dataset",
            "revision": self.head,
            "token": "token",
        }
        path = paths if isinstance(paths, str) else paths[0]
        return [SimpleNamespace(path=path)] if path in self.files else []

    def hf_hub_download(self, repo_id: str, filename: str, **kwargs: object) -> str:
        assert repo_id == "org/harbor-hf-coordination"
        assert kwargs == {
            "repo_type": "dataset",
            "revision": self.head,
            "token": "token",
        }
        destination = self.root / filename.replace("/", "-")
        destination.write_bytes(self.files[filename])
        return str(destination)

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object:
        assert repo_id == "org/harbor-hf-coordination"
        if self.conflicts:
            self.conflicts -= 1
            self.generation += 1
            raise _http_error(409)
        assert kwargs["parent_commit"] == self.head
        operation = operations[0]
        if isinstance(operation, CommitOperationAdd):
            assert kwargs == {
                "commit_message": f"chore: acquire {operation.path_in_repo}",
                "repo_type": "dataset",
                "revision": "main",
                "parent_commit": self.head,
                "token": "token",
            }
            payload = operation.path_or_fileobj
            assert isinstance(payload, bytes)
            self.files[operation.path_in_repo] = payload
        elif isinstance(operation, CommitOperationDelete):
            assert kwargs == {
                "commit_message": f"chore: release {operation.path_in_repo}",
                "repo_type": "dataset",
                "revision": "main",
                "parent_commit": self.head,
                "token": "token",
            }
            del self.files[operation.path_in_repo]
        else:
            raise AssertionError(type(operation))
        self.commits.append(kwargs)
        self.generation += 1
        return SimpleNamespace(oid=self.head)


def _http_error(status: int) -> HfHubHTTPError:
    request = httpx.Request("POST", "https://huggingface.co/api/datasets")
    return HfHubHTTPError(
        "commit failed", response=httpx.Response(status, request=request)
    )


def test_claim_paths_are_stable_and_namespaced() -> None:
    assert coordination_repository("org") == "org/harbor-hf-coordination"
    assert endpoint_claim_path("org", "endpoint") == (
        "endpoint-leases/aa3808503c913daab53ed1415fe04988.json"
    )
    path = run_claim_path("org/bucket", "runs/experiment/run-1")
    assert path.startswith("run-reservations/")
    assert path.endswith(".json")
    assert path != run_claim_path("org/other", "runs/experiment/run-1")


def test_run_claim_path_canonicalizes_bucket_references() -> None:
    expected = run_claim_path("org/bucket", "runs/experiment/run-1")

    assert run_claim_path("buckets/org/bucket", "runs/experiment/run-1") == expected
    assert (
        run_claim_path("hf://buckets/org/bucket", "runs/experiment/run-1") == expected
    )


def test_claim_acquisition_retries_parent_conflict_and_rejects_duplicate(
    tmp_path: Path,
) -> None:
    api = FakeCoordinationApi(tmp_path)
    api.conflicts = 1
    store = HubClaimStore("org", "token", api=api)
    owner = {"controller_job_id": "controller"}

    store.acquire("endpoint-leases/claim.json", owner)

    assert store.repository == "org/harbor-hf-coordination"
    assert store.token == "token"
    assert store.api is api
    assert api.files["endpoint-leases/claim.json"] == (
        b'{"controller_job_id": "controller"}\n'
    )
    assert api.commits[0]["parent_commit"] == f"{2:040x}"
    with pytest.raises(
        ClaimConflict,
        match="^claim is already held: endpoint-leases/claim.json$",
    ):
        store.acquire("endpoint-leases/claim.json", {"controller_job_id": "other"})


def test_claim_release_retries_and_verifies_owner(tmp_path: Path) -> None:
    api = FakeCoordinationApi(tmp_path)
    store = HubClaimStore("org", "token", api=api)
    owner = {"controller_job_id": "controller", "watchdog_job_id": "watchdog"}
    store.acquire("endpoint-leases/claim.json", owner)
    api.conflicts = 1

    store.release("endpoint-leases/claim.json", owner)

    assert api.files == {}
    assert isinstance(api.commits[-1], dict)


def test_claim_release_rejects_missing_or_changed_owner(tmp_path: Path) -> None:
    api = FakeCoordinationApi(tmp_path)
    store = HubClaimStore("org", "token", api=api)
    path = "endpoint-leases/claim.json"

    with pytest.raises(CoordinationError, match="^claim no longer exists$"):
        store.release(path, {"controller_job_id": "controller"})

    store.acquire(path, {"controller_job_id": "controller"})
    with pytest.raises(CoordinationError, match="^claim ownership cannot be verified$"):
        store.release(path, {"controller_job_id": "other"})
    assert path in api.files


def test_non_conflict_commit_errors_are_not_retried(tmp_path: Path) -> None:
    class FailingApi(FakeCoordinationApi):
        def create_commit(
            self, repo_id: str, operations: list[object], **kwargs: object
        ) -> object:
            raise _http_error(500)

    store = HubClaimStore("org", "token", api=FailingApi(tmp_path))

    with pytest.raises(HfHubHTTPError):
        store.acquire("endpoint-leases/claim.json", {"owner": "one"})


def test_non_conflict_release_errors_are_not_retried(tmp_path: Path) -> None:
    class FailingApi(FakeCoordinationApi):
        fail = False

        def create_commit(
            self, repo_id: str, operations: list[object], **kwargs: object
        ) -> object:
            if self.fail:
                raise _http_error(500)
            return super().create_commit(repo_id, operations, **kwargs)

    api = FailingApi(tmp_path)
    store = HubClaimStore("org", "token", api=api)
    owner = {"owner": "one"}
    store.acquire("endpoint-leases/claim.json", owner)
    api.fail = True

    with pytest.raises(HfHubHTTPError):
        store.release("endpoint-leases/claim.json", owner)


def test_claim_store_rejects_missing_head_and_malformed_owner(tmp_path: Path) -> None:
    class MissingHeadApi(FakeCoordinationApi):
        def repo_info(self, repo_id: str, **kwargs: object) -> object:
            return SimpleNamespace(sha=None)

    api = MissingHeadApi(tmp_path)
    store = HubClaimStore("org", "token", api=api)
    with pytest.raises(
        CoordinationError,
        match="^coordination repository has no commit identity$",
    ):
        store.acquire("endpoint-leases/claim.json", {"owner": "one"})

    api = FakeCoordinationApi(tmp_path)
    store = HubClaimStore("org", "token", api=api)
    api.files["endpoint-leases/claim.json"] = b"not-json"
    with pytest.raises(CoordinationError, match="^claim owner cannot be read$"):
        store.release("endpoint-leases/claim.json", {"owner": "one"})

    api.files["endpoint-leases/claim.json"] = b"[]"
    with pytest.raises(CoordinationError, match="^claim owner is invalid$"):
        store.release("endpoint-leases/claim.json", {"owner": "one"})


def test_claim_store_bounds_persistent_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeCoordinationApi(tmp_path)
    api.conflicts = 3
    monkeypatch.setattr("harbor_hf.coordination._MAX_COMMIT_ATTEMPTS", 2)

    with pytest.raises(
        CoordinationError,
        match="^coordination repository remained contended$",
    ):
        HubClaimStore("org", "token", api=api).acquire(
            "endpoint-leases/claim.json", {"owner": "one"}
        )

    api = FakeCoordinationApi(tmp_path)
    store = HubClaimStore("org", "token", api=api)
    owner = {"owner": "one"}
    store.acquire("endpoint-leases/claim.json", owner)
    api.conflicts = 3
    with pytest.raises(
        CoordinationError,
        match="^coordination repository remained contended$",
    ):
        store.release("endpoint-leases/claim.json", owner)


def test_claim_store_builds_authenticated_default_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[object] = []

    class FakeApi:
        def __init__(self, *, token: str) -> None:
            assert token == "token"
            instances.append(self)

    monkeypatch.setattr("harbor_hf.coordination.HfApi", FakeApi)

    store = HubClaimStore("org", "token")

    assert store.repository == "org/harbor-hf-coordination"
    assert store.token == "token"
    assert store.api is instances[0]
