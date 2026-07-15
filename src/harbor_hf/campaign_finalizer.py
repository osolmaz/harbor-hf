from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Literal, Protocol, cast

from pydantic import TypeAdapter

from harbor_hf.campaigns import CampaignLock, CampaignRunLock, CampaignTrialLock
from harbor_hf.harbor_native_bundle import (
    HARBOR_NATIVE_BUNDLE_PATH,
    BundleObject,
    HarborNativeBundle,
    load_harbor_native_bundle,
)
from harbor_hf.models import DeploymentProfile, ExperimentSpec
from harbor_hf.provider_models import ProviderTarget
from harbor_hf.publication_envelope import (
    PUBLICATION_ENVELOPE_PATH,
    HarborBundleReference,
    ObjectReference,
    PhysicalExecutionReference,
    ProfileDigests,
    PublicationEnvelope,
    RuntimeIdentity,
    canonical_json_bytes,
    object_reference,
    profile_digest,
)
from harbor_hf.recovery import RecoveryProjection, TerminalDecision
from harbor_hf.results import (
    ArtifactEvidence,
    ArtifactKind,
    EvidenceReader,
    ExecutionEvidence,
    MetricEvidence,
    ResultEvidence,
    RunEvidence,
    RuntimeKind,
    TrialEvidence,
)
from harbor_hf.runs import RunLock
from harbor_hf.wave_worker import ExecutionLock

_JSON_OBJECT = TypeAdapter(dict[str, object])
_TERMINAL_MARKERS = frozenset({"_SUCCESS", "_FAILED", "_CANCELLED"})


class CampaignFinalizationError(RuntimeError):
    """Raised when terminal campaign evidence cannot be finalized safely."""


class ImmutableEvidenceWriter(Protocol):
    def write_immutable(self, *, bucket: str, path: str, content: bytes) -> bool: ...


class CampaignFinalizer(Protocol):
    def finalize(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        projection: RecoveryProjection,
        decision: TerminalDecision,
    ) -> None: ...


class BucketCampaignFinalizer:
    """Build terminal run and campaign records from canonical Bucket evidence."""

    def __init__(
        self,
        reader: EvidenceReader,
        writer: ImmutableEvidenceWriter,
    ) -> None:
        self.reader = reader
        self.writer = writer

    def finalize(
        self,
        lock: CampaignLock,
        spec: ExperimentSpec,
        projection: RecoveryProjection,
        decision: TerminalDecision,
    ) -> None:
        paths = self.reader.list_files(
            bucket=spec.artifacts.bucket,
            prefix=lock.artifact_prefix,
        )
        run_checksums: dict[str, str] = {}
        for run in lock.runs:
            if projection.runs[run.run_id].status != "complete":
                continue
            checksum = self._finalize_run(lock, spec, run, paths)
            run_checksums[run.run_id] = checksum
        if decision.status == "completed" and len(run_checksums) != len(lock.runs):
            raise CampaignFinalizationError(
                "completed campaign does not have complete run evidence"
            )
        summary = _json_bytes(
            {
                "schema_version": "harbor-hf/campaign-summary/v1alpha1",
                "campaign_id": lock.campaign_id,
                "status": decision.status,
                "reason": decision.reason,
                "counts": decision.counts.model_dump(mode="json"),
                "run_checksums": dict(sorted(run_checksums.items())),
            }
        )
        self.writer.write_immutable(
            bucket=spec.artifacts.bucket,
            path=decision.summary_path,
            content=summary,
        )
        self.writer.write_immutable(
            bucket=spec.artifacts.bucket,
            path=decision.marker_path,
            content=b"\n",
        )

    def _finalize_run(
        self,
        campaign: CampaignLock,
        spec: ExperimentSpec,
        run: CampaignRunLock,
        campaign_paths: list[str],
    ) -> str:
        prefix = f"runs/{run.run_id}"
        run_paths = _under(campaign_paths, prefix)
        run_lock_bytes = self._read(spec, campaign, f"{prefix}/run.lock.json")
        configuration = RunLock.model_validate_json(run_lock_bytes)
        _validate_run_identity(campaign, run, configuration)
        trials: list[TrialEvidence] = []
        executions: list[ExecutionEvidence] = []
        metrics: list[MetricEvidence] = []
        physical_executions: list[PhysicalExecutionReference] = []
        verification_trials: list[object] = []
        completion_times: list[datetime] = []
        runtime_kind = (
            "provider"
            if isinstance(configuration.deployment, ProviderTarget)
            else "endpoint"
        )
        for shard in run.shards:
            for trial in shard.trials:
                records = self._trial_records(
                    spec,
                    campaign,
                    prefix,
                    run_paths,
                    trial,
                    runtime_kind,
                )
                trials.append(records.trial)
                executions.extend(records.executions)
                physical_executions.extend(records.physical_executions)
                metrics.extend(records.metrics)
                verification_trials.extend(records.verification_trials)
                completion_times.append(records.completed_at)
        verification = _json_bytes(
            {
                "trial_count": len(verification_trials),
                "trials": verification_trials,
            }
        )
        verification_path = f"{prefix}/verification.json"
        self.writer.write_immutable(
            bucket=spec.artifacts.bucket,
            path=f"{campaign.artifact_prefix}/{verification_path}",
            content=verification,
        )
        completed_at = max(completion_times)
        artifacts = [
            _artifact(
                run.run_id,
                "run_lock",
                "run.lock.json",
                run_lock_bytes,
            ),
            _artifact(
                run.run_id,
                "verification",
                "verification.json",
                verification,
            ),
        ]
        run_evidence = _run_evidence(campaign, configuration, completed_at)
        evidence = ResultEvidence(
            sanitized=True,
            run=run_evidence,
            trials=trials,
            executions=executions,
            metrics=metrics,
            artifacts=artifacts,
        )
        summary = _json_bytes(evidence.model_dump(mode="json"))
        summary_path = f"{prefix}/run-summary.json"
        self.writer.write_immutable(
            bucket=spec.artifacts.bucket,
            path=f"{campaign.artifact_prefix}/{summary_path}",
            content=summary,
        )
        envelope = PublicationEnvelope(
            run_id=run.run_id,
            campaign_id=campaign.campaign_id,
            created_at=configuration.created_at,
            completed_at=completed_at,
            evidence_bucket=spec.artifacts.bucket,
            evidence_prefix=f"{campaign.artifact_prefix}/{prefix}",
            run_lock=object_reference("run.lock.json", run_lock_bytes),
            profiles=ProfileDigests(
                experiment=configuration.spec_digest,
                model=profile_digest(configuration.model),
                deployment=profile_digest(configuration.deployment),
                agent=profile_digest(configuration.agent),
            ),
            runtime=RuntimeIdentity(
                kind=runtime_kind,
                provider=run_evidence.provider,
                region=run_evidence.region,
                hardware=run_evidence.hardware,
                accelerator_count=run_evidence.accelerator_count,
            ),
            cleanup_outcome=(
                "not_applicable" if runtime_kind == "provider" else "verified"
            ),
            executions=physical_executions,
        )
        envelope_bytes = canonical_json_bytes(envelope.model_dump(mode="json"))
        self.writer.write_immutable(
            bucket=spec.artifacts.bucket,
            path=(f"{campaign.artifact_prefix}/{prefix}/{PUBLICATION_ENVELOPE_PATH}"),
            content=envelope_bytes,
        )
        additions = {
            "verification.json": verification,
            "run-summary.json": summary,
            PUBLICATION_ENVELOPE_PATH: envelope_bytes,
        }
        checksums = self._aggregate_checksums(
            spec,
            campaign,
            prefix,
            run_paths,
            additions,
        )
        checksum_bytes = _json_bytes(checksums)
        self.writer.write_immutable(
            bucket=spec.artifacts.bucket,
            path=f"{campaign.artifact_prefix}/{prefix}/checksums.json",
            content=checksum_bytes,
        )
        self.writer.write_immutable(
            bucket=spec.artifacts.bucket,
            path=f"{campaign.artifact_prefix}/{prefix}/_SUCCESS",
            content=b"\n",
        )
        return _sha256(checksum_bytes)

    def _trial_records(
        self,
        spec: ExperimentSpec,
        campaign: CampaignLock,
        run_prefix: str,
        run_paths: list[str],
        trial: CampaignTrialLock,
        runtime_kind: RuntimeKind,
    ) -> _TrialRecords:
        prefix = f"{run_prefix}/trials/{trial.trial_id}"
        relative_prefix = prefix.removeprefix(f"{run_prefix}/")
        if f"{relative_prefix}/_SUCCESS" not in run_paths:
            raise CampaignFinalizationError(
                f"complete trial has no success marker: {trial.trial_id}"
            )
        summary = _JSON_OBJECT.validate_json(
            self._read(spec, campaign, f"{prefix}/trial-summary.json")
        )
        selected_id = summary.get("execution_id")
        if not isinstance(selected_id, str):
            raise CampaignFinalizationError("trial summary has no selected execution")
        execution_paths = sorted(
            path
            for path in run_paths
            if path.startswith(f"{relative_prefix}/executions/")
            and path.endswith("/execution.lock.json")
        )
        records: list[ExecutionEvidence] = []
        physical_executions: list[PhysicalExecutionReference] = []
        selected: tuple[dict[str, object], datetime] | None = None
        for relative in execution_paths:
            execution_prefix = str(PurePosixPath(relative).parent)
            absolute_prefix = f"{run_prefix}/{execution_prefix}"
            record = self._execution_record(
                spec,
                campaign,
                run_paths,
                execution_prefix,
                absolute_prefix,
                runtime_kind,
            )
            records.append(record.evidence)
            physical_executions.append(record.physical_execution)
            if record.evidence.execution_id != selected_id:
                continue
            if record.evidence.status != "succeeded":
                raise CampaignFinalizationError(
                    "trial selected execution is not successful"
                )
            selected = (
                _JSON_OBJECT.validate_json(
                    self._read(spec, campaign, f"{absolute_prefix}/verification.json")
                ),
                record.evidence.completed_at,
            )
        if selected is None:
            raise CampaignFinalizationError("trial selected execution is missing")
        verifier = _selected_verifier(selected[0])
        return _TrialRecords(
            trial=TrialEvidence(
                trial_id=trial.trial_id,
                task_name=trial.task_name,
                task_digest=trial.task_digest,
                logical_attempt=trial.logical_attempt,
                selected_execution_id=selected_id,
            ),
            executions=records,
            physical_executions=physical_executions,
            metrics=_reward_metrics(trial.trial_id, verifier),
            verification_trials=[dict(verifier)],
            completed_at=selected[1],
        )

    def _execution_record(
        self,
        spec: ExperimentSpec,
        campaign: CampaignLock,
        run_paths: list[str],
        execution_prefix: str,
        absolute_prefix: str,
        runtime_kind: RuntimeKind,
    ) -> _ExecutionRecord:
        execution = ExecutionLock.model_validate_json(
            self._read(spec, campaign, f"{absolute_prefix}/execution.lock.json")
        )
        marker = _marker(run_paths, execution_prefix)
        raw_events = _json_lines(
            self._read(spec, campaign, f"{absolute_prefix}/events.jsonl")
        )
        started = _event_time(raw_events, "execution_started")
        finished = _event_time(
            raw_events,
            "execution_succeeded",
            "execution_failed",
            "execution_cancelled",
        )
        status = cast(
            Literal["succeeded", "failed_infrastructure", "cancelled"],
            {
                "_SUCCESS": "succeeded",
                "_FAILED": "failed_infrastructure",
                "_CANCELLED": "cancelled",
            }[marker],
        )
        evidence = ExecutionEvidence(
            execution_id=execution.execution_id,
            trial_id=execution.trial_id,
            physical_attempt=execution.physical_attempt,
            runtime_kind=runtime_kind,
            status=status,
            started_at=started,
            completed_at=finished,
            retry_reason=(
                "infrastructure_retry" if execution.physical_attempt > 1 else None
            ),
            remote_job_id=execution.remote_job_id,
        )
        bundle = self._execution_bundle_reference(
            spec,
            campaign,
            run_paths,
            execution_prefix,
            absolute_prefix,
        )
        if evidence.status == "succeeded" and bundle is None:
            raise CampaignFinalizationError(
                "successful execution has no verified Harbor native bundle"
            )
        return _ExecutionRecord(
            evidence=evidence,
            physical_execution=PhysicalExecutionReference(
                execution_id=evidence.execution_id,
                trial_id=evidence.trial_id,
                physical_attempt=evidence.physical_attempt,
                status=evidence.status,
                started_at=evidence.started_at,
                completed_at=evidence.completed_at,
                retry_reason=evidence.retry_reason,
                remote_job_id=evidence.remote_job_id,
                bundle_status="verified" if bundle is not None else "not_available",
                harbor_bundle=bundle,
            ),
        )

    def _execution_bundle_reference(
        self,
        spec: ExperimentSpec,
        campaign: CampaignLock,
        run_paths: list[str],
        execution_prefix: str,
        absolute_prefix: str,
    ) -> HarborBundleReference | None:
        relative_manifest = f"{execution_prefix}/{HARBOR_NATIVE_BUNDLE_PATH}"
        if relative_manifest not in run_paths:
            return None
        manifest_bytes = self._read(
            spec,
            campaign,
            f"{absolute_prefix}/{HARBOR_NATIVE_BUNDLE_PATH}",
        )
        try:
            manifest = load_harbor_native_bundle(manifest_bytes)
        except Exception as error:
            raise CampaignFinalizationError(
                "execution Harbor native bundle is invalid"
            ) from error
        _validate_native_bundle_paths(manifest)
        self._verify_bundle_documents(
            spec, campaign, run_paths, execution_prefix, absolute_prefix, manifest
        )
        archive = self._verified_bundle_object(
            spec,
            campaign,
            run_paths,
            execution_prefix,
            absolute_prefix,
            manifest.archive,
            "archive",
        )
        self._verified_bundle_object(
            spec,
            campaign,
            run_paths,
            execution_prefix,
            absolute_prefix,
            manifest.compatibility,
            "compatibility export",
        )
        return HarborBundleReference(
            manifest=object_reference(relative_manifest, manifest_bytes),
            archive=archive,
            harbor_revision=manifest.harbor_revision,
            harbor_version=manifest.harbor_version,
            compatibility_schema=manifest.compatibility_schema,
            request_digest=manifest.request_digest,
            document_count=len(manifest.documents),
        )

    def _verified_bundle_object(
        self,
        spec: ExperimentSpec,
        campaign: CampaignLock,
        run_paths: list[str],
        execution_prefix: str,
        absolute_prefix: str,
        reference: BundleObject,
        label: str,
    ) -> ObjectReference:
        relative = f"{execution_prefix}/{reference.path}"
        if relative not in run_paths:
            raise CampaignFinalizationError(f"execution Harbor {label} is missing")
        content = self._read(spec, campaign, f"{absolute_prefix}/{reference.path}")
        observed = object_reference(relative, content)
        if (
            observed.digest != reference.digest
            or observed.size_bytes != reference.size_bytes
        ):
            raise CampaignFinalizationError(
                f"execution Harbor {label} conflicts with its bundle manifest"
            )
        return observed

    def _verify_bundle_documents(
        self,
        spec: ExperimentSpec,
        campaign: CampaignLock,
        run_paths: list[str],
        execution_prefix: str,
        absolute_prefix: str,
        manifest: HarborNativeBundle,
    ) -> None:
        for document in manifest.documents:
            relative = f"{execution_prefix}/{document.path}"
            if relative not in run_paths:
                raise CampaignFinalizationError(
                    "execution Harbor native document is missing"
                )
            content = self._read(spec, campaign, f"{absolute_prefix}/{document.path}")
            if _sha256(content) != document.digest:
                raise CampaignFinalizationError(
                    "execution Harbor native document conflicts with its bundle"
                )

    def _aggregate_checksums(
        self,
        spec: ExperimentSpec,
        campaign: CampaignLock,
        run_prefix: str,
        run_paths: list[str],
        additions: dict[str, bytes],
    ) -> dict[str, str]:
        covered = self._child_checksums(spec, campaign, run_prefix, run_paths)
        expected = set(run_paths) | set(additions)
        expected.discard("checksums.json")
        expected.difference_update(_TERMINAL_MARKERS)
        for path in sorted(expected - covered.keys()):
            content = additions.get(path)
            if content is None:
                content = self._read(spec, campaign, f"{run_prefix}/{path}")
            covered[path] = _sha256(content)
        if set(covered) != expected:
            raise CampaignFinalizationError("run checksums do not cover exact evidence")
        for path, digest in sorted(covered.items()):
            content = additions.get(path)
            if content is None:
                content = self._read(spec, campaign, f"{run_prefix}/{path}")
            if _sha256(content) != digest:
                raise CampaignFinalizationError(
                    f"child checksum does not match evidence: {path}"
                )
        return dict(sorted(covered.items()))

    def _child_checksums(
        self,
        spec: ExperimentSpec,
        campaign: CampaignLock,
        run_prefix: str,
        run_paths: list[str],
    ) -> dict[str, str]:
        covered: dict[str, str] = {}
        for path in sorted(run_paths):
            if not path.endswith("/checksums.json"):
                continue
            manifest = json.loads(self._read(spec, campaign, f"{run_prefix}/{path}"))
            if not isinstance(manifest, dict):
                raise CampaignFinalizationError("child checksum manifest is invalid")
            parent = str(PurePosixPath(path).parent)
            for relative, digest in manifest.items():
                if not isinstance(relative, str) or not isinstance(digest, str):
                    raise CampaignFinalizationError(
                        "child checksum manifest is invalid"
                    )
                key = str(PurePosixPath(parent, relative))
                previous = covered.setdefault(key, digest)
                if previous != digest:
                    raise CampaignFinalizationError("child checksums conflict")
        return covered

    def _read(self, spec: ExperimentSpec, campaign: CampaignLock, path: str) -> bytes:
        return self.reader.read_bytes(
            bucket=spec.artifacts.bucket,
            prefix=campaign.artifact_prefix,
            path=path,
        )


@dataclass(frozen=True)
class _ExecutionRecord:
    evidence: ExecutionEvidence
    physical_execution: PhysicalExecutionReference


@dataclass(frozen=True)
class _TrialRecords:
    trial: TrialEvidence
    executions: list[ExecutionEvidence]
    physical_executions: list[PhysicalExecutionReference]
    metrics: list[MetricEvidence]
    verification_trials: list[object]
    completed_at: datetime


def _validate_native_bundle_paths(manifest: HarborNativeBundle) -> None:
    if (
        manifest.archive.path != "artifacts.tar.gz"
        or manifest.compatibility.path != "harbor-compatibility.json"
    ):
        raise CampaignFinalizationError(
            "execution Harbor native bundle uses noncanonical paths"
        )


def _run_evidence(
    campaign: CampaignLock,
    lock: RunLock,
    completed_at: datetime,
) -> RunEvidence:
    deployment = lock.deployment
    if isinstance(deployment, DeploymentProfile):
        provider = deployment.provider
        region = deployment.region
        hardware = deployment.hardware
        accelerators = deployment.accelerator_count
    else:
        provider = deployment.service
        region = "not_reported"
        hardware = "not_reported"
        accelerators = 0
    agent_revision = (
        lock.agent.revision
        if lock.agent.revision_kind == "package"
        else cast(str, lock.agent.reported_version)
    )
    return RunEvidence(
        run_id=lock.run_id,
        campaign_id=campaign.campaign_id,
        experiment=lock.experiment,
        benchmark=lock.benchmark_dataset,
        benchmark_revision=lock.benchmark_dataset_digest,
        created_at=lock.created_at,
        completed_at=completed_at,
        model_id=lock.model.id,
        model_repo=lock.model.repo,
        model_revision=(
            lock.model.revision
            if isinstance(deployment, DeploymentProfile)
            else "not_observed"
        ),
        deployment_id=deployment.id,
        provider=provider,
        region=region,
        hardware=hardware,
        accelerator_count=accelerators,
        agent_id=lock.agent.id,
        agent_name=lock.agent.name,
        agent_revision=agent_revision,
    )


def _validate_run_identity(
    campaign: CampaignLock,
    expected: CampaignRunLock,
    observed: RunLock,
) -> None:
    identity = (
        observed.run_id,
        observed.model.id,
        observed.deployment.id,
        observed.agent.id,
    )
    locked = (
        expected.run_id,
        expected.model,
        expected.deployment,
        expected.agent,
    )
    if identity != locked or observed.created_at != campaign.created_at:
        raise CampaignFinalizationError("run evidence does not match campaign lock")


def _under(paths: list[str], prefix: str) -> list[str]:
    root = f"{prefix}/"
    return sorted(path.removeprefix(root) for path in paths if path.startswith(root))


def _marker(paths: list[str], prefix: str) -> str:
    markers = {
        PurePosixPath(path).name
        for path in paths
        if str(PurePosixPath(path).parent) == prefix
        and PurePosixPath(path).name in _TERMINAL_MARKERS
    }
    if len(markers) != 1:
        raise CampaignFinalizationError("execution has no exclusive terminal marker")
    return markers.pop()


def _json_lines(value: bytes) -> list[dict[str, object]]:
    try:
        return [
            _JSON_OBJECT.validate_json(line)
            for line in value.splitlines()
            if line.strip()
        ]
    except Exception as error:
        raise CampaignFinalizationError("execution event log is invalid") from error


def _event_time(records: list[dict[str, object]], *names: str) -> datetime:
    for record in records:
        if record.get("event") not in names:
            continue
        value = record.get("at")
        if not isinstance(value, str):
            break
        observed = datetime.fromisoformat(value)
        if observed.tzinfo is None:
            break
        return observed.astimezone(UTC)
    raise CampaignFinalizationError("execution event log omits a required timestamp")


def _artifact(
    run_id: str,
    kind: ArtifactKind,
    path: str,
    content: bytes,
) -> ArtifactEvidence:
    return ArtifactEvidence(
        owner_type="run",
        owner_id=run_id,
        kind=kind,
        path=path,
        sha256=_sha256(content),
        media_type="application/json",
        size_bytes=len(content),
    )


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _reward_metrics(
    trial_id: str, verifier: Mapping[object, object]
) -> list[MetricEvidence]:
    rewards = verifier.get("rewards")
    if not isinstance(rewards, Mapping):
        raise CampaignFinalizationError("trial verification has no rewards")
    metrics: list[MetricEvidence] = []
    for name, value in sorted(rewards.items(), key=lambda item: str(item[0])):
        if (
            not isinstance(name, str)
            or not isinstance(value, int | float)
            or isinstance(value, bool)
        ):
            raise CampaignFinalizationError("trial reward evidence is invalid")
        metrics.append(
            MetricEvidence(
                owner_type="trial",
                owner_id=trial_id,
                name=name,
                value=float(value),
                unit="score",
            )
        )
    return metrics


def _selected_verifier(value: Mapping[str, object]) -> Mapping[object, object]:
    trials = value.get("trials")
    if not isinstance(trials, list) or len(trials) != 1:
        raise CampaignFinalizationError("trial verification evidence is invalid")
    verifier = trials[0]
    if not isinstance(verifier, Mapping):
        raise CampaignFinalizationError("trial verification record is invalid")
    return cast(Mapping[object, object], verifier)
