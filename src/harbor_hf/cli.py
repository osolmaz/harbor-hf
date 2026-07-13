from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from harbor_hf.io import ManifestError, load_experiment
from harbor_hf.models import ExperimentSpec
from harbor_hf.planner import build_plan
from harbor_hf.process import SubprocessRunner
from harbor_hf.runs import RunLock, build_run_lock
from harbor_hf.submission import Submission, build_submit_command
from harbor_hf.submission import submit as submit_job
from harbor_hf.worker import WorkerError, run_worker

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


@app.command()
def submit(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    model: Annotated[str | None, typer.Option("--model")] = None,
    deployment: Annotated[str | None, typer.Option("--deployment")] = None,
    agent: Annotated[str | None, typer.Option("--agent")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Submit one resolved matrix cell to a remote Hugging Face Job."""
    spec = _load_or_exit(manifest)
    try:
        lock = build_run_lock(
            spec,
            model_id=model,
            deployment_id=deployment,
            agent_id=agent,
            run_id=run_id,
        )
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error

    with tempfile.TemporaryDirectory(prefix="harbor-hf-") as staging_name:
        staging = Path(staging_name)
        shutil.copyfile(manifest, staging / "manifest.yaml")
        _write_lock(staging / "run.lock.json", lock)
        if dry_run:
            command = build_submit_command(
                lock, input_dir=staging, bucket=spec.artifacts.bucket
            )
            result = Submission(
                run_id=lock.run_id,
                artifact_prefix=lock.artifact_prefix,
                job_id=None,
                command=command,
            )
        else:
            result = submit_job(
                lock,
                input_dir=staging,
                bucket=spec.artifacts.bucket,
                runner=SubprocessRunner(),
            )
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@app.command(hidden=True)
def worker(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    lock: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Option("--output-root", file_okay=False)],
) -> None:
    """Run one benchmark cell from inside a Hugging Face Job."""
    try:
        destination = run_worker(manifest, lock, output_root)
    except (OSError, ValueError, WorkerError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(str(destination))


def _write_lock(path: Path, lock: RunLock) -> None:
    path.write_text(
        json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",  # pragma: no mutate
    )
