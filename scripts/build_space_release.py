from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_release(root: Path, output: Path) -> None:
    root = root.resolve()
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("release output must be outside the repository")
    web_dist = root / "apps/results-web/dist"
    if not (web_dist / "index.html").is_file():
        raise FileNotFoundError("build apps/results-web before staging the Space")
    temporary = output.with_name(f".{output.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock", "LICENSE"):
        shutil.copy2(root / name, temporary / name)
    shutil.copy2(root / "deploy/space/Dockerfile", temporary / "Dockerfile")
    shutil.copy2(root / "deploy/space/README.md", temporary / "README.md")
    shutil.copytree(
        root / "src",
        temporary / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(web_dist, temporary / "apps/results-web/dist")
    shutil.rmtree(output, ignore_errors=True)
    temporary.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage a Harbor Results Space")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_release(Path(__file__).resolve().parents[1], args.output)


if __name__ == "__main__":
    main()
