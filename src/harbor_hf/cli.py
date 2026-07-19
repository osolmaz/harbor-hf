from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Never, cast
from uuid import uuid4

import typer
from httpx import HTTPError
from huggingface_hub import HfApi, get_token

from harbor_hf.automation import (
    AutomationError,
    AutomationRequest,
    automation_plan,
    install_automation,
)
from harbor_hf.bucket_evidence import (
    BucketEvidenceError,
    HubBucketEvidenceReader,
    HubBucketEvidenceWriter,
)
from harbor_hf.campaign_apply import (
    CampaignApplyError,
    hugging_face_campaign_reconciler,
)
from harbor_hf.campaign_finalizer import CampaignFinalizationError
from harbor_hf.campaign_observer import CampaignObservationError
from harbor_hf.campaigns import (
    build_campaign_lock,
    build_campaign_plan,
    campaign_json_schemas,
    new_campaign_id,
)
from harbor_hf.catalog_cutover import (
    CatalogCutoverError,
    CatalogCutoverPlan,
    CutoverDatasetApi,
    HubCatalogCutover,
)
from harbor_hf.control import (
    CampaignSubmittedPayload,
    ControlError,
    HubCampaignStore,
    new_event,
)
from harbor_hf.coordination import CoordinationError, HubClaimStore
from harbor_hf.io import ManifestError, load_experiment
from harbor_hf.models import ExperimentSpec
from harbor_hf.operations import (
    AutomaticCampaignPublisher,
    DatasetRepositoryApi,
    cancel_campaign,
    publish_campaign_results,
    retry_campaign_shard,
    seal_partial_campaign_runs,
    verify_campaign_artifacts,
)
from harbor_hf.planner import build_plan
from harbor_hf.process import ProcessError, SubprocessRunner
from harbor_hf.profile_preflight import preflight_profile_plan
from harbor_hf.profile_submission import (
    ProfileSubmission,
    build_profile_submit_command,
    submit_profile,
)
from harbor_hf.profile_worker import ProfileWorkerError, run_profile_worker
from harbor_hf.profile_worker_transport import ProfileTransportError
from harbor_hf.profiling import (
    ProfilePlan,
    build_profile_plan,
    load_serving_profile,
    select_profile,
)
from harbor_hf.reconciler import AdmissionLimits, ReconcileContext, plan_reconciliation
from harbor_hf.recovery import project_recovery
from harbor_hf.result_publisher import (
    DatasetApi,
    DatasetPublicationError,
    HubDatasetPublisher,
)
from harbor_hf.results import CatalogDecision, ResultPublicationError
from harbor_hf.runs import RunLock, build_run_lock
from harbor_hf.submission import Submission, build_submit_command
from harbor_hf.submission import submit as submit_job
from harbor_hf.wave_worker import run_wave_worker
from harbor_hf.worker import WorkerError, run_endpoint_watchdog, run_worker

app = typer.Typer(
    no_args_is_help=True,
    help="Plan and run Harbor benchmarks on Hugging Face infrastructure.",
)
campaign_app = typer.Typer(no_args_is_help=True, help="Plan and run campaigns.")
artifacts_app = typer.Typer(no_args_is_help=True, help="Inspect campaign evidence.")
results_app = typer.Typer(no_args_is_help=True, help="Publish campaign results.")
automation_app = typer.Typer(
    no_args_is_help=True, help="Install campaign reconciliation automation."
)
profile_app = typer.Typer(no_args_is_help=True, help="Profile serving deployments.")
app.add_typer(campaign_app, name="campaign")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(results_app, name="results")
app.add_typer(automation_app, name="automation")
app.add_typer(profile_app, name="profile")

_OPERATION_ERRORS = (
    HTTPError,
    OSError,
    ValueError,
    AutomationError,
    CampaignApplyError,
    CampaignFinalizationError,
    CampaignObservationError,
    BucketEvidenceError,
    ControlError,
    CoordinationError,
    DatasetPublicationError,
    ResultPublicationError,
    CatalogCutoverError,
    ProfileWorkerError,
    ProfileTransportError,
)


def _echo_json(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def _exit_operation(error: Exception) -> Never:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1) from error


def _load_or_exit(path: Path) -> ExperimentSpec:
    try:
        return load_experiment(path)
    except ManifestError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error


def _load_profile_plan(path: Path) -> ProfilePlan:
    try:
        return ProfilePlan.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error


@profile_app.command("plan")
def profile_plan(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    profile_id: Annotated[str, typer.Option("--profile-id")],
    max_spend_usd: Annotated[str, typer.Option("--max-spend-usd")],
    estimated_profile_cost_usd: Annotated[
        str | None, typer.Option("--estimated-profile-cost-usd")
    ] = None,
    timeout_seconds: Annotated[int, typer.Option("--timeout-seconds", min=1)] = 3600,
    concurrency: Annotated[
        list[int] | None, typer.Option("--concurrency", min=1)
    ] = None,
) -> None:
    """Resolve one deterministic serving profile without remote work."""
    try:
        resolved = build_profile_plan(
            _load_or_exit(manifest),
            profile_id=profile_id,
            candidate_concurrency=concurrency or [1, 2, 4, 8, 16, 32, 64],
            max_spend_usd=max_spend_usd,
            profile_timeout_seconds=timeout_seconds,
            estimated_profile_cost_usd=estimated_profile_cost_usd,
        )
        output.write_text(
            json.dumps(resolved.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        _exit_operation(error)
    _echo_json(
        {
            "profile_id": resolved.profile_id,
            "plan_sha256": resolved.plan_sha256,
            "output": str(output),
            "remote_work": False,
        }
    )


@profile_app.command("preflight")
def profile_preflight(
    plan: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Verify quota, price, routing, revision, and private storage."""
    try:
        report = preflight_profile_plan(_load_profile_plan(plan))
    except (HTTPError, OSError, ValueError) as error:
        _exit_operation(error)
    _echo_json(report.model_dump(mode="json"))


@profile_app.command("run")
def profile_run(
    plan: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Submit a remote-only serving profile Job."""
    resolved = _load_profile_plan(plan)
    try:
        if dry_run:
            command = build_profile_submit_command(
                resolved,
                input_dir="hf://buckets/NAMESPACE/jobs-artifacts/DRY-RUN",
                bucket=resolved.artifacts.bucket,
            )
            result = ProfileSubmission(
                profile_id=resolved.profile_id,
                artifact_prefix=resolved.artifacts.prefix,
                job_id=None,
                command=command,
            )
        else:
            preflight_profile_plan(resolved)
            result = submit_profile(resolved)
    except (HTTPError, OSError, ProcessError, ValueError) as error:
        _exit_operation(error)
    _echo_json(result.model_dump(mode="json"))


@profile_app.command("select")
def profile_select(
    profile: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
) -> None:
    """Validate immutable point evidence and select the winning concurrency."""
    try:
        selected = select_profile(load_serving_profile(profile))
        output.write_text(
            json.dumps(selected.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        _exit_operation(error)
    assert selected.selection is not None
    _echo_json(selected.selection.model_dump(mode="json"))


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
        if spec.publishing.index_dataset is None:
            raise ValueError("campaign submission requires publishing.index_dataset")
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
        projection = project_recovery(lock, events)
    except (HTTPError, OSError, ValueError, ControlError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    payload = projection.model_dump(mode="json")
    payload["status"] = projection.status
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@campaign_app.command("reconcile")
def campaign_reconcile(
    campaign_id: Annotated[str, typer.Argument()],
    namespace: Annotated[str, typer.Option("--namespace")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    apply: Annotated[bool, typer.Option("--apply")] = False,
) -> None:
    """Plan the next idempotent campaign actions."""
    if dry_run == apply:
        typer.echo("Error: choose exactly one of --dry-run or --apply", err=True)
        raise typer.Exit(code=2)
    try:
        if apply:
            with hugging_face_campaign_reconciler(namespace) as reconciler:
                result = reconciler.apply_campaign(campaign_id)
        else:
            lock, events = HubCampaignStore(namespace).load_campaign(campaign_id)
            _projection, result = plan_reconciliation(lock, events)
    except _OPERATION_ERRORS as error:
        _exit_operation(error)
    _echo_json(result.model_dump(mode="json"))


@campaign_app.command("reconcile-all")
def campaign_reconcile_all(
    namespace: Annotated[str, typer.Option("--namespace")],
    apply: Annotated[bool, typer.Option("--apply")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    campaign_ids: Annotated[list[str] | None, typer.Option("--campaign-id")] = None,
    provider_active_waves: Annotated[
        int | None, typer.Option("--provider-active-waves", min=1)
    ] = None,
) -> None:
    """Reconcile every campaign in the namespace once."""
    if dry_run == apply:
        typer.echo("Error: choose exactly one of --dry-run or --apply", err=True)
        raise typer.Exit(code=2)
    context = ReconcileContext(
        limits=AdmissionLimits(
            provider_active_waves=(
                provider_active_waves
                if provider_active_waves is not None
                else AdmissionLimits().provider_active_waves
            )
        )
    )
    try:
        if apply:
            with hugging_face_campaign_reconciler(namespace) as reconciler:
                results = reconciler.apply_all(
                    context=context,
                    campaign_ids=campaign_ids,
                )
        else:
            store = HubCampaignStore(namespace)
            results = [
                plan_reconciliation(*store.load_campaign(campaign_id), context=context)[
                    1
                ]
                for campaign_id in (
                    list(dict.fromkeys(campaign_ids))
                    if campaign_ids is not None
                    else store.list_campaigns()
                )
            ]
    except _OPERATION_ERRORS as error:
        _exit_operation(error)
    _echo_json([result.model_dump(mode="json") for result in results])


@campaign_app.command("cancel")
def campaign_cancel(
    campaign_id: Annotated[str, typer.Argument()],
    namespace: Annotated[str, typer.Option("--namespace")],
    reason: Annotated[str, typer.Option("--reason")] = "operator request",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    output_format: Annotated[Literal["json"], typer.Option("--format")] = "json",
) -> None:
    """Record one durable campaign cancellation request."""
    del output_format
    try:
        result = cancel_campaign(
            HubCampaignStore(namespace),
            campaign_id,
            reason=reason,
            dry_run=dry_run,
        )
    except _OPERATION_ERRORS as error:
        _exit_operation(error)
    _echo_json(result.model_dump(mode="json"))


@campaign_app.command("retry")
def campaign_retry(
    campaign_id: Annotated[str, typer.Argument()],
    namespace: Annotated[str, typer.Option("--namespace")],
    shard_id: Annotated[str, typer.Option("--shard")],
    reason: Annotated[str, typer.Option("--reason")] = "operator retry request",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    output_format: Annotated[Literal["json"], typer.Option("--format")] = "json",
) -> None:
    """Request an immediate retry for retryable trials in one shard."""
    del output_format
    try:
        result = retry_campaign_shard(
            HubCampaignStore(namespace),
            campaign_id,
            shard_id=shard_id,
            reason=reason,
            dry_run=dry_run,
        )
    except _OPERATION_ERRORS as error:
        _exit_operation(error)
    _echo_json(result.model_dump(mode="json"))


@campaign_app.command("seal")
def campaign_seal(
    campaign_id: Annotated[str, typer.Argument()],
    namespace: Annotated[str, typer.Option("--namespace")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    output_format: Annotated[Literal["json"], typer.Option("--format")] = "json",
) -> None:
    """Seal failed retries in a drained partial campaign as zero-score outcomes."""
    del output_format
    try:
        snapshot = HubCampaignStore(namespace).load_snapshot(campaign_id)
        with tempfile.TemporaryDirectory(prefix="harbor-hf-evidence-") as cache:
            result = seal_partial_campaign_runs(
                snapshot,
                namespace=namespace,
                reader=HubBucketEvidenceReader(Path(cache)),
                writer=None if dry_run else HubBucketEvidenceWriter(),
                dry_run=dry_run,
            )
    except _OPERATION_ERRORS as error:
        _exit_operation(error)
    _echo_json(result.model_dump(mode="json"))


@artifacts_app.command("verify")
def artifacts_verify(
    campaign_id: Annotated[str, typer.Argument()],
    namespace: Annotated[str, typer.Option("--namespace")],
    output_format: Annotated[Literal["json"], typer.Option("--format")] = "json",
) -> None:
    """Verify publishable run evidence and every declared checksum."""
    del output_format
    try:
        store = HubCampaignStore(namespace)
        snapshot = store.load_snapshot(campaign_id)
        with tempfile.TemporaryDirectory(prefix="harbor-hf-evidence-") as cache:
            reader = HubBucketEvidenceReader(Path(cache))
            result = verify_campaign_artifacts(
                snapshot, namespace=namespace, reader=reader
            )
    except _OPERATION_ERRORS as error:
        _exit_operation(error)
    _echo_json(result.model_dump(mode="json"))


@results_app.command("publish")
def results_publish(
    campaign_id: Annotated[str, typer.Argument()],
    namespace: Annotated[str, typer.Option("--namespace")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    output_format: Annotated[Literal["json"], typer.Option("--format")] = "json",
) -> None:
    """Verify and publish normalized campaign result tables."""
    del output_format
    try:
        store = HubCampaignStore(namespace)
        api = HfApi()
        with tempfile.TemporaryDirectory(prefix="harbor-hf-evidence-") as cache:
            reader = HubBucketEvidenceReader(Path(cache))
            if dry_run:
                result = publish_campaign_results(
                    store.load_snapshot(campaign_id),
                    namespace=namespace,
                    reader=reader,
                    publisher=None,
                    dry_run=True,
                )
            else:
                token = get_token()
                if token is None:
                    raise ValueError("result publication requires HF authentication")
                result = AutomaticCampaignPublisher(
                    namespace=namespace,
                    store=store,
                    reader=reader,
                    publisher=HubDatasetPublisher(
                        publisher_id=f"cli-{campaign_id}",
                        leases=HubClaimStore(namespace, token),
                        api=cast(DatasetApi, api),
                    ),
                    repositories=cast(DatasetRepositoryApi, api),
                ).publish(campaign_id)
    except _OPERATION_ERRORS as error:
        _exit_operation(error)
    _echo_json(result.model_dump(mode="json"))


@results_app.command("catalog")
def results_catalog(
    publication_id: Annotated[str, typer.Argument()],
    action: Annotated[Literal["promote", "withdraw"], typer.Option("--action")],
    reason: Annotated[str, typer.Option("--reason")],
    actor: Annotated[str, typer.Option("--actor")],
    index_dataset: Annotated[str, typer.Option("--index-dataset")],
    namespace: Annotated[str, typer.Option("--namespace")],
    output_format: Annotated[Literal["json"], typer.Option("--format")] = "json",
) -> None:
    """Record an append-only primary catalog decision."""
    del output_format
    try:
        token = get_token()
        if token is None:
            raise ValueError("catalog decisions require HF authentication")
        decision = CatalogDecision(
            decision_id=f"decision-{uuid4().hex}",
            publication_id=publication_id,
            action=action,
            actor=actor,
            reason=reason,
            created_at=datetime.now(UTC),
        )
        result = HubDatasetPublisher(
            publisher_id=f"cli-{decision.decision_id}",
            leases=HubClaimStore(namespace, token),
            api=cast(DatasetApi, HfApi()),
        ).decide_catalog(decision, index_dataset=index_dataset)
    except _OPERATION_ERRORS as error:
        _exit_operation(error)
    _echo_json(result.model_dump(mode="json"))


@results_app.command("cutover-catalog")
def results_cutover_catalog(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    namespace: Annotated[str, typer.Option("--namespace")],
    output_format: Annotated[Literal["json"], typer.Option("--format")] = "json",
) -> None:
    """Apply an explicit, parent-checked V1 catalog cutover."""
    del output_format
    try:
        plan = CatalogCutoverPlan.model_validate_json(
            manifest.read_text(encoding="utf-8")
        )
        token = get_token()
        if token is None:
            raise ValueError("catalog cutover requires HF authentication")
        result = HubCatalogCutover(
            publisher_id=f"cli-{plan.cutover_id}",
            leases=HubClaimStore(namespace, token),
            api=cast(CutoverDatasetApi, HfApi()),
        ).apply(plan)
    except _OPERATION_ERRORS as error:
        _exit_operation(error)
    _echo_json(result.model_dump(mode="json"))


@automation_app.command("install")
def automation_install(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    schedule: Annotated[str, typer.Option("--schedule")],
    namespace: Annotated[str | None, typer.Option("--namespace")] = None,
    provider_active_waves: Annotated[
        int | None, typer.Option("--provider-active-waves", min=1)
    ] = None,
    campaign_ids: Annotated[list[str] | None, typer.Option("--campaign-id")] = None,
    suspended: Annotated[bool, typer.Option("--suspended")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    output_format: Annotated[Literal["json"], typer.Option("--format")] = "json",
) -> None:
    """Install or adopt the managed schedule and control webhook."""
    del output_format
    spec = _load_or_exit(manifest)
    try:
        if spec.remote is None:
            raise ValueError("automation installation requires remote configuration")
        request = AutomationRequest(
            namespace=namespace or spec.remote.job.namespace,
            schedule=schedule,
            remote=spec.remote,
            secret_names=(
                [spec.benchmark.source.credentials.secret_name]
                if spec.benchmark.source is not None
                and spec.benchmark.source.credentials is not None
                else []
            ),
            provider_active_waves=provider_active_waves,
            campaign_ids=campaign_ids or [],
            suspended=suspended,
        )
        if dry_run:
            payload = {
                **automation_plan(request).model_dump(mode="json"),
                "installed": False,
                "dry_run": True,
            }
        else:
            token = get_token()
            if token is None:
                raise ValueError("automation installation requires HF authentication")
            installation = install_automation(request, token=token)
            payload = {
                **installation.model_dump(mode="json"),
                "installed": True,
                "dry_run": False,
            }
    except _OPERATION_ERRORS as error:
        _exit_operation(error)
    _echo_json(payload)


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


@app.command("wave-worker", hidden=True)
def wave_worker(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    campaign_lock: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    wave_lock: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Option("--output-root", file_okay=False)],
) -> None:
    """Run one bounded deployment wave from inside a Hugging Face Job."""
    try:
        destination = run_wave_worker(
            manifest,
            campaign_lock,
            wave_lock,
            output_root,
        )
    except (OSError, ValueError, WorkerError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(str(destination))


@app.command("profile-worker", hidden=True)
def profile_worker(
    plan: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Option("--output-root", file_okay=False)],
) -> None:
    """Run one serving profile from inside a Hugging Face Job."""
    try:
        destination = run_profile_worker(plan, output_root)
    except (OSError, ValueError, ProfileWorkerError, ProfileTransportError) as error:
        _exit_operation(error)
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
