from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class PresentationConfig:
    """Non-secret configuration for the public, read-only results service."""

    index_dataset: str
    index_revision: str = "main"
    max_publications: int = 256
    title: str = "Harbor Results"
    refresh_seconds: int = 60

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PresentationConfig:
        values = os.environ if environ is None else environ
        dataset = values.get("HARBOR_HF_INDEX_DATASET", "").strip()
        revision = values.get("HARBOR_HF_INDEX_REVISION", "main").strip()
        title = values.get("HARBOR_HF_SPACE_TITLE", "Harbor Results").strip()
        maximum = values.get("HARBOR_HF_MAX_PUBLICATIONS", "256").strip()
        refresh = values.get("HARBOR_HF_REFRESH_SECONDS", "60").strip()
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
        if not 1 <= max_publications <= 2048:
            raise ValueError("HARBOR_HF_MAX_PUBLICATIONS must be between 1 and 2048")
        try:
            refresh_seconds = int(refresh)
        except ValueError as error:
            raise ValueError("HARBOR_HF_REFRESH_SECONDS must be an integer") from error
        if not 5 <= refresh_seconds <= 3600:
            raise ValueError("HARBOR_HF_REFRESH_SECONDS must be between 5 and 3600")
        return cls(dataset, revision, max_publications, title, refresh_seconds)
