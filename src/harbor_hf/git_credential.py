from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TextIO

_CREDENTIAL_FILE_ENV = "HARBOR_HF_GIT_CREDENTIAL_FILE"
_REPOSITORY_ENV = "HARBOR_HF_GIT_REPOSITORY"


def main() -> None:
    operation = sys.argv[1] if len(sys.argv) > 1 else ""
    request = _credential_request(sys.stdin)
    repository = os.environ.get(_REPOSITORY_ENV, "")
    path = request.get("path", "").removeprefix("/")
    expected_paths = {repository, f"{repository}.git"}
    is_lfs_path = path in {
        f"{repository}/info/lfs",
        f"{repository}.git/info/lfs",
    }
    if (
        operation != "get"
        or request.get("protocol") != "https"
        or request.get("host") != "github.com"
        or (path not in expected_paths and not is_lfs_path)
    ):
        return
    credential_file = os.environ.get(_CREDENTIAL_FILE_ENV, "")
    secret = (
        Path(credential_file).read_text(encoding="utf-8") if credential_file else ""
    )
    if not secret:
        raise SystemExit("Git credential secret is not available")
    print("username=x-access-token")
    print(f"password={secret}")


def _credential_request(stream: TextIO) -> dict[str, str]:
    request: dict[str, str] = {}
    for line in stream:
        stripped = line.rstrip("\n")
        if not stripped:
            break
        key, separator, value = stripped.partition("=")
        if separator:
            request[key] = value
    return request


if __name__ == "__main__":
    main()
