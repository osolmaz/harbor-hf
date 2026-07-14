import io
import sys

import pytest

from harbor_hf.git_credential import main


def test_credential_helper_returns_scoped_github_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["git-credential-harbor-hf", "get"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            "protocol=https\nhost=github.com\npath=ShellBench/public-tasks.git\n\n"
        ),
    )
    monkeypatch.setenv("HARBOR_HF_GIT_CREDENTIAL_ENV", "GITHUB_TOKEN")
    monkeypatch.setenv("HARBOR_HF_GIT_REPOSITORY", "ShellBench/public-tasks")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")

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
    ],
)
def test_credential_helper_refuses_other_targets(
    credential_input: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["git-credential-harbor-hf", "get"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(credential_input))
    monkeypatch.setenv("HARBOR_HF_GIT_CREDENTIAL_ENV", "GITHUB_TOKEN")
    monkeypatch.setenv("HARBOR_HF_GIT_REPOSITORY", "ShellBench/public-tasks")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")

    main()

    assert capsys.readouterr().out == ""
