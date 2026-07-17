from __future__ import annotations

import json
from pathlib import Path

from harbor_hf.publication_schemas import publication_schema_documents


def main() -> None:
    root = Path("schemas")
    for filename, schema in publication_schema_documents().items():
        (root / filename).write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
