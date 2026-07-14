import io
import sys
from pathlib import Path

import pytest

from harbor_hf.git_credential import main


@pytest.mark.parametrize(
    "path",
    [
        "ShellBench/public-tasks",
        "ShellBench/public-tasks.git",
        "ShellBench/public-tasks/info/lfs",
        "ShellBench/public-tasks.git/info/lfs",
    ],
)
def test_credential_helper_returns_scoped_github_token(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "argv", ["git-credential-harbor-hf", "get"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(f"protocol=https\nhost=github.com\npath={path}\n\n"),
    )
    credential_file = tmp_path / "credential"
    credential_file.write_text("github-secret", encoding="utf-8")
    monkeypatch.setenv("HARBOR_HF_GIT_CREDENTIAL_FILE", str(credential_file))
    monkeypatch.setenv("HARBOR_HF_GIT_REPOSITORY", "ShellBench/public-tasks")

    main()

    assert capsys.readouterr().out == (
        "username=x-access-token\npassword=github-secret\n"
    )


@pytest.mark.parametrize(
    "credential_input",
    [
        "protocol=http\nhost=github.com\npath=ShellBench/public-tasks\n\n",
        "protocol=https\nhost=example.com\npath=ShellBench/public-tasks\n\n",
        "protocol=https\nhost=github.com\npath=other/repo\n\n",
        "protocol=https\nhost=github.com\npath=ShellBench/public-tasks.git/info/lfs/locks\n\n",
    ],
)
def test_credential_helper_refuses_other_targets(
    credential_input: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "argv", ["git-credential-harbor-hf", "get"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(credential_input))
    credential_file = tmp_path / "credential"
    credential_file.write_text("github-secret", encoding="utf-8")
    monkeypatch.setenv("HARBOR_HF_GIT_CREDENTIAL_FILE", str(credential_file))
    monkeypatch.setenv("HARBOR_HF_GIT_REPOSITORY", "ShellBench/public-tasks")

    main()

    assert capsys.readouterr().out == ""
