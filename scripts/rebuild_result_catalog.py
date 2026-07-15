from __future__ import annotations

import argparse
from pathlib import Path

from harbor_hf.presentation.config import PresentationConfig
from harbor_hf.presentation.repository import ResultRepository
from harbor_hf.results import build_catalog_lookup_file, build_catalog_window_file

_WINDOW_SIZES = tuple(2**power for power in range(12))


def rebuild(dataset: str, revision: str, output: Path) -> int:
    repository = ResultRepository(
        PresentationConfig(
            index_dataset=dataset,
            index_revision=revision,
            max_publications=max(_WINDOW_SIZES),
        )
    )
    rows = repository.rebuild_catalog()
    output.mkdir(parents=True, exist_ok=True)
    for size in _WINDOW_SIZES:
        artifact = build_catalog_window_file(rows, size)
        destination = output / artifact.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(artifact.content)
    for row in rows:
        artifact = build_catalog_lookup_file(row)
        destination = output / artifact.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(artifact.content)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild bounded result catalog snapshots"
    )
    parser.add_argument("dataset")
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    count = rebuild(args.dataset, args.revision, args.output)
    print(f"staged {count} catalog rows in {args.output}")


if __name__ == "__main__":
    main()
