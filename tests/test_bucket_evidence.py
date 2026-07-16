from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest

from harbor_hf.bucket_evidence import (
    BucketEvidenceError,
    HubBucketEvidenceReader,
    HubBucketEvidenceWriter,
)


class FakeBucketApi:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.list_calls: list[tuple[str, str | None, bool]] = []
        self.downloads = 0
        self.download_batches: list[list[str]] = []

    def list_bucket_tree(
        self, bucket_id: str, prefix: str | None = None, **kwargs: object
    ) -> list[object]:
        self.list_calls.append((bucket_id, prefix, kwargs["recursive"] is True))
        return [
            SimpleNamespace(type="file", path=path) for path in reversed(self.files)
        ]

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[object, str | Path]],
        **kwargs: object,
    ) -> None:
        assert bucket_id == "org/evidence"
        assert kwargs == {"raise_on_missing_files": True}
        self.downloads += 1
        batch: list[str] = []
        for remote, destination in files:
            path = cast(SimpleNamespace, remote).path
            assert isinstance(path, str)
            batch.append(path)
            Path(destination).write_bytes(self.files[path])
        self.download_batches.append(batch)


class FakeBucketWriterApi:
    def __init__(self, observed: list[object] | None = None) -> None:
        self.observed = observed or []
        self.existing_content = b"existing"
        self.download_calls: list[tuple[str, str, bool, str, bool]] = []
        self.batch_calls: list[
            tuple[str, list[tuple[bytes, str]], dict[str, object]]
        ] = []
        self.write_download = True

    def get_bucket_paths_info(
        self, bucket_id: str, paths: Iterable[str], **kwargs: object
    ) -> list[object]:
        assert bucket_id == "org/evidence"
        assert list(paths) == ["campaigns/campaign-one/_SUCCESS"]
        assert kwargs == {}
        return self.observed

    def download_bucket_files(
        self,
        bucket_id: str,
        files: list[tuple[object, str | Path]],
        **kwargs: object,
    ) -> None:
        remote, destination = files[0]
        path = cast(SimpleNamespace, remote).path
        self.download_calls.append(
            (
                bucket_id,
                path,
                kwargs["raise_on_missing_files"] is True,
                Path(destination).name,
                Path(destination).parent.name.startswith("harbor-hf-bucket-"),
            )
        )
        if self.write_download:
            Path(destination).write_bytes(self.existing_content)

    def batch_bucket_files(
        self,
        bucket_id: str,
        *,
        add: list[tuple[bytes, str]],
        **kwargs: object,
    ) -> None:
        self.batch_calls.append((bucket_id, add, kwargs))


class FakeClaims:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, dict[str, str]]] = []
        self.released: list[tuple[str, dict[str, str]]] = []

    def acquire(self, path: str, owner: Mapping[str, str]) -> None:
        self.acquired.append((path, dict(owner)))

    def release(self, path: str, owner: Mapping[str, str]) -> None:
        self.released.append((path, dict(owner)))


def test_lists_and_caches_bucket_evidence(tmp_path: Path) -> None:
    prefix = "campaigns/campaign-one/runs/run-one"
    api = FakeBucketApi(
        {
            f"{prefix}/run-summary.json": b"summary",
            f"{prefix}/_SUCCESS": b"",
        }
    )
    reader = HubBucketEvidenceReader(tmp_path, api=api)

    assert reader.list_files(bucket="hf://buckets/org/evidence", prefix=prefix) == [
        "_SUCCESS",
        "run-summary.json",
    ]
    assert (
        reader.read_bytes(
            bucket="hf://buckets/org/evidence",
            prefix=prefix,
            path="run-summary.json",
        )
        == b"summary"
    )
    assert (
        reader.read_bytes(
            bucket="hf://buckets/org/evidence",
            prefix=prefix,
            path="run-summary.json",
        )
        == b"summary"
    )
    assert api.list_calls == [("org/evidence", prefix, True)]
    assert api.downloads == 1
    assert api.download_batches == [
        [f"{prefix}/_SUCCESS", f"{prefix}/run-summary.json"]
    ]


def test_interrupted_download_never_becomes_a_cached_evidence_object(
    tmp_path: Path,
) -> None:
    prefix = "campaigns/campaign-one/runs/run-one"

    class InterruptedApi(FakeBucketApi):
        fail = True

        def download_bucket_files(
            self,
            bucket_id: str,
            files: list[tuple[object, str | Path]],
            **kwargs: object,
        ) -> None:
            if self.fail:
                self.fail = False
                Path(files[0][1]).write_bytes(b"truncated")
                raise OSError("download interrupted")
            super().download_bucket_files(bucket_id, files, **kwargs)

    api = InterruptedApi(
        {
            f"{prefix}/record.json": b"complete",
            f"{prefix}/second.json": b"also complete",
        }
    )
    reader = HubBucketEvidenceReader(tmp_path, api=api)

    with pytest.raises(OSError, match="download interrupted"):
        reader.read_bytes(bucket="org/evidence", prefix=prefix, path="record.json")

    assert (
        reader.read_bytes(bucket="org/evidence", prefix=prefix, path="record.json")
        == b"complete"
    )
    assert not any(path.read_bytes() == b"truncated" for path in tmp_path.iterdir())
    assert api.download_batches == [[f"{prefix}/record.json", f"{prefix}/second.json"]]


def test_refresh_discards_cached_evidence_bytes(tmp_path: Path) -> None:
    prefix = "campaigns/campaign-one/runs/run-one"
    path = f"{prefix}/record.json"
    api = FakeBucketApi({path: b"first"})
    reader = HubBucketEvidenceReader(tmp_path, api=api)
    assert (
        reader.read_bytes(bucket="org/evidence", prefix=prefix, path="record.json")
        == b"first"
    )

    api.files[path] = b"second"
    reader.refresh()

    assert (
        reader.read_bytes(bucket="org/evidence", prefix=prefix, path="record.json")
        == b"second"
    )


def test_rejects_missing_and_escaped_bucket_objects(tmp_path: Path) -> None:
    prefix = "campaigns/campaign-one/runs/run-one"
    reader = HubBucketEvidenceReader(
        tmp_path,
        api=FakeBucketApi({"campaigns/other/run-summary.json": b"summary"}),
    )

    with pytest.raises(BucketEvidenceError, match="escaped"):
        reader.list_files(bucket="org/evidence", prefix=prefix)

    missing = HubBucketEvidenceReader(
        tmp_path / "missing",
        api=FakeBucketApi({f"{prefix}/_SUCCESS": b""}),
    )
    with pytest.raises(BucketEvidenceError, match="missing"):
        missing.read_bytes(
            bucket="org/evidence", prefix=prefix, path="run-summary.json"
        )


def test_writer_creates_absent_immutable_object_with_exact_batch() -> None:
    api = FakeBucketWriterApi()
    writer = HubBucketEvidenceWriter(api=api)

    created = writer.write_immutable(
        bucket="hf://buckets/org/evidence",
        path="campaigns/campaign-one/_SUCCESS",
        content=b"new evidence",
    )

    assert created is True
    assert api.download_calls == []
    assert api.batch_calls == [
        (
            "org/evidence",
            [(b"new evidence", "campaigns/campaign-one/_SUCCESS")],
            {},
        )
    ]


def test_writer_serializes_check_and_write_with_distributed_claim() -> None:
    api = FakeBucketWriterApi()
    claims = FakeClaims()
    writer = HubBucketEvidenceWriter(api=api, claims=claims)

    assert writer.write_immutable(
        bucket="org/evidence",
        path="campaigns/campaign-one/_SUCCESS",
        content=b"new evidence",
    )

    assert len(claims.acquired) == len(claims.released) == 1
    assert claims.acquired == claims.released
    assert claims.acquired[0][0].startswith("bucket-evidence-leases/")


def test_writer_does_not_fail_completed_write_when_claim_release_fails() -> None:
    class FailingReleaseClaims(FakeClaims):
        def release(self, path: str, owner: Mapping[str, str]) -> None:
            super().release(path, owner)
            raise httpx.ConnectError("claim release timed out")

    api = FakeBucketWriterApi()
    claims = FailingReleaseClaims()
    writer = HubBucketEvidenceWriter(api=api, claims=claims)

    created = writer.write_immutable(
        bucket="org/evidence",
        path="campaigns/campaign-one/_SUCCESS",
        content=b"new evidence",
    )

    assert created is True
    assert len(claims.acquired) == len(claims.released) == 1


def test_writer_adopts_only_byte_identical_immutable_object() -> None:
    remote = SimpleNamespace(path="campaigns/campaign-one/_SUCCESS")
    api = FakeBucketWriterApi([remote])
    writer = HubBucketEvidenceWriter(api=api)

    created = writer.write_immutable(
        bucket="org/evidence",
        path="campaigns/campaign-one/_SUCCESS",
        content=b"existing",
    )

    assert created is False
    assert api.download_calls == [
        (
            "org/evidence",
            "campaigns/campaign-one/_SUCCESS",
            True,
            "object",
            True,
        )
    ]
    assert api.batch_calls == []


@pytest.mark.parametrize(
    ("observed", "message"),
    [
        (
            [SimpleNamespace(path="campaigns/campaign-one/other")],
            "Bucket immutable-path lookup is ambiguous",
        ),
        (
            [SimpleNamespace(identity="missing-path")],
            "Bucket immutable-path lookup is ambiguous",
        ),
        (
            [
                SimpleNamespace(path="campaigns/campaign-one/_SUCCESS"),
                SimpleNamespace(path="campaigns/campaign-one/_SUCCESS"),
            ],
            "Bucket immutable-path lookup is ambiguous",
        ),
    ],
)
def test_writer_rejects_ambiguous_immutable_lookup(
    observed: list[object], message: str
) -> None:
    api = FakeBucketWriterApi(observed)

    with pytest.raises(BucketEvidenceError) as captured:
        HubBucketEvidenceWriter(api=api).write_immutable(
            bucket="org/evidence",
            path="campaigns/campaign-one/_SUCCESS",
            content=b"existing",
        )

    assert str(captured.value) == message
    assert api.download_calls == []
    assert api.batch_calls == []


def test_writer_rejects_conflicting_or_unreadable_existing_object() -> None:
    remote = SimpleNamespace(path="campaigns/campaign-one/_SUCCESS")
    conflicting = FakeBucketWriterApi([remote])

    with pytest.raises(BucketEvidenceError) as conflict:
        HubBucketEvidenceWriter(api=conflicting).write_immutable(
            bucket="org/evidence",
            path="campaigns/campaign-one/_SUCCESS",
            content=b"different",
        )

    assert str(conflict.value) == (
        "Bucket immutable evidence conflicts: campaigns/campaign-one/_SUCCESS"
    )
    unreadable = FakeBucketWriterApi([remote])
    unreadable.write_download = False
    with pytest.raises(BucketEvidenceError) as read_error:
        HubBucketEvidenceWriter(api=unreadable).write_immutable(
            bucket="org/evidence",
            path="campaigns/campaign-one/_SUCCESS",
            content=b"existing",
        )

    assert str(read_error.value) == (
        "Bucket evidence object cannot be read: campaigns/campaign-one/_SUCCESS"
    )
