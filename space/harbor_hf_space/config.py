from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class SpaceConfig:
    """Non-secret configuration for the public, read-only Space."""

    index_dataset: str
    index_revision: str = "main"
    max_publications: int = 250
    title: str = "Harbor results"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SpaceConfig:
        values = os.environ if environ is None else environ
        dataset = values.get("HARBOR_HF_INDEX_DATASET", "").strip()
        revision = values.get("HARBOR_HF_INDEX_REVISION", "main").strip()
        title = values.get("HARBOR_HF_SPACE_TITLE", "Harbor results").strip()
        maximum = values.get("HARBOR_HF_MAX_PUBLICATIONS", "250").strip()
        if not _DATASET_ID.fullmatch(dataset):
            raise ValueError(
                "HARBOR_HF_INDEX_DATASET must be a namespace/name Dataset ID"
            )
        if not revision or any(character.isspace() for character in revision):
            raise ValueError("HARBOR_HF_INDEX_REVISION must be a non-empty revision")
        if not title:
            raise ValueError("HARBOR_HF_SPACE_TITLE must not be empty")
        try:
            max_publications = int(maximum)
        except ValueError as error:
            raise ValueError("HARBOR_HF_MAX_PUBLICATIONS must be an integer") from error
        if not 1 <= max_publications <= 2_000:
            raise ValueError("HARBOR_HF_MAX_PUBLICATIONS must be between 1 and 2000")
        return cls(
            index_dataset=dataset,
            index_revision=revision,
            max_publications=max_publications,
            title=title,
        )
