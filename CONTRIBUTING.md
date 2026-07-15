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
(cd apps/results-web && npm ci && npm run build)
docker build -f deploy/space/Dockerfile .
uv run slophammer-py dry .
uv run pip-audit
uv run slophammer-py check . --baseline
```

The slower mutation suite is available as an explicit local command and a
manually dispatched GitHub Actions workflow. It is not part of the pull-request
critical path. The checked-in Slophammer baseline records that deliberate
exception and still rejects every new finding:

```bash
uv run python scripts/check_mutation.py --min-kill-rate 90
```

Tests must mock Hugging Face and Harbor network boundaries unless they are
explicitly marked remote integration tests. Never place tokens, endpoint URLs,
or captured secrets in fixtures.

Use Conventional Commits for commit messages and pull request titles.
