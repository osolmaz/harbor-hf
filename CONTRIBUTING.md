# Contributing

## Development

Install the locked development environment:

```bash
uv sync --all-groups
```

Before submitting a change, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest --cov=src/harbor_hf --cov-fail-under=85
uv run pytest tests/test_presentation.py --cov=space --cov-fail-under=85
uv run slophammer-py dry .
uv run python scripts/check_mutation.py --min-kill-rate 90
uv run pip-audit
uv run slophammer-py check .
```

Tests must mock Hugging Face and Harbor network boundaries unless they are
explicitly marked remote integration tests. Never place tokens, endpoint URLs,
or captured secrets in fixtures.

Use Conventional Commits for commit messages and pull request titles.
