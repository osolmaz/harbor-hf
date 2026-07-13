from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from harbor_hf.bucket_evidence import BucketEvidenceError, HubBucketEvidenceReader


class FakeBucketApi:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.list_calls: list[tuple[str, str | None, bool]] = []
        self.downloads = 0

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
        for remote, destination in files:
            path = cast(SimpleNamespace, remote).path
            assert isinstance(path, str)
            Path(destination).write_bytes(self.files[path])


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
