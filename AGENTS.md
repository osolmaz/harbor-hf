# Repository Instructions

- Use Python 3.12+, uv, Pydantic, Typer, Ruff, ty, and pytest.
- Run `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run ty check`, and `uv run pytest --cov=src/harbor_hf --cov-fail-under=85`
  before finishing code changes.
- Run `uv run slophammer-py check .` after changing project structure or CI.
- Run `uv run slophammer-py dry .` and
  `uv run python scripts/check_mutation.py --min-kill-rate 90` before finishing
  behavior changes.
- Keep domain planning separate from Hugging Face, Harbor, filesystem, clock,
  and process-state adapters.
- Use only public Harbor APIs. Do not monkeypatch Harbor internals.
- Do not load models or run inference locally. Remote integration tests must be
  explicit and leave every Inference Endpoint paused.
- Never write secret values to manifests, logs, tests, locks, or artifacts.
- Add tests for every behavior change and preserve at least 85% coverage.
- Avoid `Any`; validate untrusted provider data at the adapter boundary.
- Use Conventional Commits.
- Apply the standards in
  `https://github.com/dutifuldev/slophammer/blob/main/docs/AGENT_ENTRYPOINT.md`.
