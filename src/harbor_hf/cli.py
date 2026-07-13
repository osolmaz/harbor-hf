from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from harbor_hf.io import ManifestError, load_experiment
from harbor_hf.models import ExperimentSpec
from harbor_hf.planner import build_plan

app = typer.Typer(
    no_args_is_help=True,
    help="Plan and run Harbor benchmarks on Hugging Face infrastructure.",
)


def _load_or_exit(path: Path) -> ExperimentSpec:
    try:
        return load_experiment(path)
    except ManifestError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error


@app.command()
def validate(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate an experiment manifest."""
    spec = _load_or_exit(manifest)
    plan = build_plan(spec)
    typer.echo(f"Valid {spec.kind}: {spec.metadata.name} ({plan.spec_digest})")


@app.command()
def plan(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Resolve an experiment matrix without creating remote resources."""
    experiment_plan = build_plan(_load_or_exit(manifest))
    typer.echo(json.dumps(experiment_plan.model_dump(mode="json"), indent=2))
