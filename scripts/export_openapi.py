from __future__ import annotations

import json
from pathlib import Path

from harbor_hf.presentation.api import create_app


def main() -> None:
    destination = Path("docs/api-v1.openapi.json")
    destination.write_text(
        json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
