from __future__ import annotations

import os
import sys
from typing import TextIO

_CREDENTIAL_ENV = "HARBOR_HF_GIT_CREDENTIAL_ENV"
_REPOSITORY_ENV = "HARBOR_HF_GIT_REPOSITORY"


def main() -> None:
    operation = sys.argv[1] if len(sys.argv) > 1 else ""
    request = _credential_request(sys.stdin)
    repository = os.environ.get(_REPOSITORY_ENV, "")
    expected_paths = {repository, f"{repository}.git"}
    if (
        operation != "get"
        or request.get("protocol") != "https"
        or request.get("host") != "github.com"
        or request.get("path", "").removeprefix("/") not in expected_paths
    ):
        return
    secret_name = os.environ.get(_CREDENTIAL_ENV, "")
    secret = os.environ.get(secret_name, "") if secret_name else ""
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
