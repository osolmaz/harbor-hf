from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import typer
from httpx import HTTPError

from harbor_hf.campaigns import (
    build_campaign_lock,
    build_campaign_plan,
    campaign_json_schemas,
    new_campaign_id,
)
from harbor_hf.control import (
    CampaignSubmittedPayload,
    ControlError,
    HubCampaignStore,
    new_event,
    project_campaign,
)
from harbor_hf.coordination import CoordinationError
from harbor_hf.io import ManifestError, load_experiment
from harbor_hf.models import ExperimentSpec
from harbor_hf.planner import build_plan
from harbor_hf.process import ProcessError, SubprocessRunner
from harbor_hf.reconciler import plan_reconciliation
from harbor_hf.runs import RunLock, build_run_lock
from harbor_hf.submission import Submission, build_submit_command
from harbor_hf.submission import submit as submit_job
from harbor_hf.worker import WorkerError, run_endpoint_watchdog, run_worker

app = typer.Typer(
    no_args_is_help=True,
    help="Plan and run Harbor benchmarks on Hugging Face infrastructure.",
)
campaign_app = typer.Typer(no_args_is_help=True, help="Plan and run campaigns.")
app.add_typer(campaign_app, name="campaign")


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


@campaign_app.command("plan")
def campaign_plan(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_format: Annotated[
        Literal["json", "text"], typer.Option("--format")
    ] = "text",
) -> None:
    """Resolve an immutable campaign without creating remote resources."""
    try:
        resolved = build_campaign_plan(_load_or_exit(manifest))
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error
    if output_format == "json":
        typer.echo(
            json.dumps(resolved.model_dump(mode="json"), indent=2, sort_keys=True)
        )
        return
    typer.echo(f"Campaign plan: {resolved.experiment}")
    typer.echo(f"Plan digest: {resolved.plan_digest}")
    typer.echo(f"Runs: {resolved.run_count}")
    typer.echo(f"Shards: {resolved.shard_count}")
    typer.echo(f"Trials: {resolved.trial_count}")


@campaign_app.command("schema")
def campaign_schema(
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
) -> None:
    """Export the campaign plan and lock JSON Schemas."""
    rendered = json.dumps(campaign_json_schemas(), indent=2, sort_keys=True) + "\n"
    if output is None:
        typer.echo(rendered, nl=False)
        return
    output.write_text(rendered, encoding="utf-8")


@campaign_app.command("submit")
def campaign_submit(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    campaign_id: Annotated[str | None, typer.Option("--campaign-id")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Persist an immutable campaign request on Hugging Face."""
    spec = _load_or_exit(manifest)
    try:
        if spec.remote is None:
            raise ValueError("campaign submission requires a remote configuration")
        resolved = build_campaign_plan(spec)
        resolved_id = campaign_id or new_campaign_id(resolved)
        lock = build_campaign_lock(resolved, resolved_id)
        submitted = new_event(
            subject_type="campaign",
            subject_id=resolved_id,
            kind="campaign.submitted",
            producer="cli",
            payload=CampaignSubmittedPayload(plan_digest=resolved.plan_digest),
        )
        if not dry_run:
            from harbor_hf.submission import ensure_private_coordination_repository

            ensure_private_coordination_repository(spec.remote.job.namespace)
            HubCampaignStore(spec.remote.job.namespace).create_campaign(
                lock, manifest.read_bytes(), submitted
            )
    except (HTTPError, OSError, ValueError, ControlError, CoordinationError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        json.dumps(
            {
                "campaign_id": lock.campaign_id,
                "plan_digest": lock.plan_digest,
                "artifact_prefix": lock.artifact_prefix,
                "stored": not dry_run,
            },
            indent=2,
            sort_keys=True,
        )
    )


@campaign_app.command("status")
def campaign_status(
    campaign_id: Annotated[str, typer.Argument()],
    namespace: Annotated[str, typer.Option("--namespace")],
) -> None:
    """Read the durable projection of one campaign."""
    try:
        lock, events = HubCampaignStore(namespace).load_campaign(campaign_id)
        projection = project_campaign(lock, events)
    except (HTTPError, OSError, ValueError, ControlError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(projection.model_dump(mode="json"), indent=2, sort_keys=True))


@campaign_app.command("reconcile")
def campaign_reconcile(
    campaign_id: Annotated[str, typer.Argument()],
    namespace: Annotated[str, typer.Option("--namespace")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Plan the next idempotent campaign actions."""
    if not dry_run:
        typer.echo(
            "Error: campaign reconciliation currently requires --dry-run", err=True
        )
        raise typer.Exit(code=2)
    try:
        lock, events = HubCampaignStore(namespace).load_campaign(campaign_id)
        _projection, reconciliation = plan_reconciliation(lock, events)
    except (HTTPError, OSError, ValueError, ControlError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        json.dumps(reconciliation.model_dump(mode="json"), indent=2, sort_keys=True)
    )


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
            try:
                result = submit_job(
                    lock,
                    input_dir=staging,
                    bucket=spec.artifacts.bucket,
                    runner=SubprocessRunner(),
                )
            except (HTTPError, ProcessError, ValueError) as error:
                typer.echo(f"Error: {error}", err=True)
                raise typer.Exit(code=1) from error
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


@app.command(hidden=True)
def watchdog(
    controller_job_id: Annotated[str, typer.Option("--controller-job-id")],
    controller_namespace: Annotated[str, typer.Option("--controller-namespace")],
    endpoint_name: Annotated[str, typer.Option("--endpoint-name")],
    endpoint_namespace: Annotated[str, typer.Option("--endpoint-namespace")],
    run_id: Annotated[str, typer.Option("--run-id")],
    token_secret_name: Annotated[str, typer.Option("--token-secret-name")],
    timeout_seconds: Annotated[int, typer.Option("--timeout-seconds", min=1)],
) -> None:
    """Pause an endpoint after its controller Job exits or times out."""
    try:
        snapshot = run_endpoint_watchdog(
            controller_job_id=controller_job_id,
            controller_namespace=controller_namespace,
            endpoint_name=endpoint_name,
            endpoint_namespace=endpoint_namespace,
            run_id=run_id,
            token_secret_name=token_secret_name,
            timeout_seconds=timeout_seconds,
        )
    except (OSError, ValueError, WorkerError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(snapshot, indent=2, sort_keys=True))


def _write_lock(path: Path, lock: RunLock) -> None:
    path.write_text(
        json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",  # pragma: no mutate
    )
