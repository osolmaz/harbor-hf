from __future__ import annotations

import json
from pathlib import Path

from harbor_hf.publication_schemas import publication_schema_documents


def test_checked_in_publication_schemas_match_models() -> None:
    root = Path(__file__).parent.parent / "schemas"

    for filename, expected in publication_schema_documents().items():
        assert json.loads((root / filename).read_text(encoding="utf-8")) == expected
