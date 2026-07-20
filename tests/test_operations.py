from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
import yaml
from huggingface_hub import CommitOperationAdd

from harbor_hf.campaigns import build_campaign_lock, build_campaign_plan
from harbor_hf.control import (
    CampaignEvent,
    CampaignSnapshot,
    CampaignSubmittedPayload,
    ExecutionOutcomePayload,
    ExecutionStartedPayload,
    LifecyclePayload,
    ManualInterventionResolutionPayload,
    WaveLifecyclePayload,
    new_event,
)
from harbor_hf.io import ManifestError
from harbor_hf.models import ExperimentSpec
from harbor_hf.operations import (
    AutomaticCampaignPublisher,
    cancel_campaign,
    publish_campaign_results,
    resume_campaign,
    retry_campaign_shard,
    verify_campaign_artifacts,
)
from harbor_hf.reconciler import plan_reconciliation
from harbor_hf.recovery import (
    durable_manual_intervention_resolution_event,
    project_recovery,
)
from harbor_hf.result_publisher import PublicationResult
from harbor_hf.results import ResultPublication, ResultPublicationError


class MemoryStore:
    def __init__(self, snapshot: CampaignSnapshot) -> None:
        self.snapshot = snapshot

    def load_snapshot(self, campaign_id: str) -> CampaignSnapshot:
        assert campaign_id == self.snapshot.lock.campaign_id
        return self.snapshot

    def ensure_event(self, campaign_id: str, event: CampaignEvent) -> bool:
        assert campaign_id == self.snapshot.lock.campaign_id
        if event in self.snapshot.events:
            return False
        self.snapshot.events.append(event)
        return True


class MemoryEvidence:
    def __init__(
        self,
        prefix: str,
        files: dict[str, bytes],
        *,
        interactions: list[object] | None = None,
    ) -> None:
        self.prefix = prefix
        self.files = files
        self.interactions = interactions
        self.refresh_calls = 0

    def refresh(self) -> None:
        self.refresh_calls += 1
        if self.interactions is not None:
            self.interactions.append("refresh")

    def list_files(self, *, bucket: str, prefix: str) -> list[str]:
        assert bucket == "example/benchmark-runs"
        assert prefix == self.prefix
        return list(reversed(self.files))

    def read_bytes(self, *, bucket: str, prefix: str, path: str) -> bytes:
        assert bucket == "example/benchmark-runs"
        assert prefix == self.prefix
        return self.files[path]


class FakePublisher:
    def __init__(self, *, interactions: list[object] | None = None) -> None:
        self.publications: list[ResultPublication] = []
        self.interactions = interactions

    def publish(
        self,
        publication: ResultPublication,
        *,
        result_dataset: str,
        index_dataset: str,
    ) -> PublicationResult:
        assert result_dataset == "example/shellbench-results"
        assert index_dataset == "example/benchmark-run-index"
        if self.interactions is not None:
            self.interactions.append(
                ("publish", result_dataset, index_dataset, publication)
            )
        self.publications.append(publication)
        return PublicationResult(
            publication_id=publication.tables.publication_id,
            result_dataset=result_dataset,
            result_revision="a" * 40,
            index_dataset=index_dataset,
            index_revision="b" * 40,
        )


class MemoryRepositories:
    def __init__(
        self,
        interactions: list[object],
        *,
        existing: dict[str, bool] | None = None,
    ) -> None:
        self.interactions = interactions
        self.private = dict(existing or {})
        self.sha = {repository: "1" * 40 for repository in self.private}
        self.commits: list[tuple[str, list[object], dict[str, object]]] = []

    def create_repo(self, repo_id: str, **kwargs: object) -> object:
        self.interactions.append(("create_repo", repo_id, kwargs))
        requested_private = kwargs.get("private")
        assert isinstance(requested_private, bool)
        self.private.setdefault(repo_id, requested_private)
        return object()

    def repo_info(self, repo_id: str, **kwargs: object) -> object:
        self.interactions.append(("repo_info", repo_id, kwargs))
        return SimpleNamespace(
            private=self.private[repo_id],
            sha=self.sha.get(repo_id),
        )

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object:
        self.interactions.append(("create_commit", repo_id, kwargs))
        self.commits.append((repo_id, operations, kwargs))
        self.sha[repo_id] = "2" * 40
        return SimpleNamespace(oid=self.sha[repo_id])


def _snapshot(spec: ExperimentSpec) -> CampaignSnapshot:
    lock = build_campaign_lock(build_campaign_plan(spec), "campaign-one")
    submitted = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=lock.plan_digest),
        clock=lambda: lock.created_at - timedelta(seconds=3),
        identifier=lambda: "1" * 32,
    )
    request = yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True)).encode()
    return CampaignSnapshot(
        lock=lock,
        events=[submitted],
        request=request,
        control_commit="c" * 40,
    )


def _retry_snapshot(spec: ExperimentSpec) -> CampaignSnapshot:
    snapshot = _snapshot(spec)
    shard = snapshot.lock.runs[0].shards[0]
    trial = shard.trials[0]
    started = new_event(
        subject_type="execution",
        subject_id="execution-one",
        kind="execution.started",
        producer="wave-controller",
        payload=ExecutionStartedPayload(
            trial_id=trial.trial_id,
            shard_id=shard.shard_id,
            physical_attempt=1,
        ),
        clock=lambda: snapshot.lock.created_at - timedelta(seconds=2),
        identifier=lambda: "2" * 32,
    )
    failed = new_event(
        subject_type="execution",
        subject_id="execution-one",
        kind="execution.failed",
        producer="wave-controller",
        payload=ExecutionOutcomePayload(
            trial_id=trial.trial_id,
            physical_attempt=1,
            category="transient",
        ),
        clock=lambda: snapshot.lock.created_at - timedelta(seconds=1),
        identifier=lambda: "3" * 32,
    )
    snapshot.events.extend([started, failed])
    return snapshot


def _evidence(snapshot: CampaignSnapshot) -> MemoryEvidence:
    run = snapshot.lock.runs[0]
    trial = run.shards[0].trials[0]
    created = snapshot.lock.created_at
    summary = {
        "schema_version": "harbor-hf/result-evidence/v1",
        "sanitized": True,
        "run": {
            "run_id": run.run_id,
            "campaign_id": snapshot.lock.campaign_id,
            "experiment": "experiment",
            "evaluation_id": snapshot.lock.evaluation_id,
            "publication_role": snapshot.lock.publication_role,
            "component_kind": snapshot.lock.component_kind,
            "benchmark": "shellbench",
            "benchmark_revision": "sha256:" + "1" * 64,
            "result_kind": "ordinary",
            "outcome": "complete",
            "quality": "clean",
            "created_at": created.isoformat(),
            "completed_at": (created + timedelta(minutes=1)).isoformat(),
            "model_id": "model-one",
            "model_repo": "org/model",
            "model_revision": "a" * 40,
            "deployment_id": "deployment-one",
            "provider": "huggingface",
            "region": "aws-us-east-1",
            "hardware": "cpu-basic",
            "accelerator_count": 0,
            "agent_id": "agent-one",
            "agent_name": "agent",
            "agent_revision": "1.0.0",
        },
        "trials": [
            {
                "trial_id": trial.trial_id,
                "task_name": trial.task_name,
                "task_digest": trial.task_digest,
                "logical_attempt": trial.logical_attempt,
                "selected_execution_id": "execution-one",
                "outcome": "scored",
            }
        ],
        "executions": [
            {
                "execution_id": "execution-one",
                "trial_id": trial.trial_id,
                "physical_attempt": 1,
                "runtime_kind": "endpoint",
                "status": "succeeded",
                "failure_category": None,
                "started_at": created.isoformat(),
                "completed_at": (created + timedelta(minutes=1)).isoformat(),
                "retry_reason": None,
                "remote_job_id": "job-one",
            }
        ],
        "metrics": [
            {
                "owner_type": "trial",
                "owner_id": trial.trial_id,
                "name": "reward",
                "value": 0.0,
                "unit": "score",
                "aggregation": None,
            }
        ],
        "artifacts": [],
    }
    files = {
        "run.lock.json": json.dumps(
            {
                "run_id": run.run_id,
                "evaluation_id": snapshot.lock.evaluation_id,
                "publication_role": snapshot.lock.publication_role,
                "component_kind": snapshot.lock.component_kind,
                "attempts": 1,
                "model": {
                    "id": "model-one",
                    "repo": "org/model",
                    "revision": "a" * 40,
                    "weights": {"format": "safetensors"},
                },
                "deployment": {
                    "id": "deployment-one",
                    "provider": "hf-inference-endpoints",
                    "hardware": "cpu-basic",
                    "accelerator_count": 1,
                    "region": "aws-us-east-1",
                    "engine": {"name": "test", "image": "test:latest"},
                },
                "agent": {
                    "id": "agent-one",
                    "name": "agent",
                    "revision": "1.0.0",
                    "revision_kind": "package",
                },
                "benchmark_task_digests": {
                    trial.task_name: trial.task_digest,
                },
            }
        ).encode(),
        "run-summary.json": json.dumps(summary).encode(),
    }
    execution_prefix = f"trials/{trial.trial_id}/executions/execution-one"
    manifest_path = f"{execution_prefix}/harbor-native-bundle.json"
    archive_path = f"{execution_prefix}/artifacts.tar.gz"
    files[manifest_path] = b"native bundle manifest"
    files[archive_path] = b"native bundle archive"
    prefix = f"{snapshot.lock.artifact_prefix}/runs/{run.run_id}"
    run_lock = files["run.lock.json"]
    files["publication-envelope.v1.json"] = json.dumps(
        {
            "schema_version": "harbor-hf/publication-envelope/v1",
            "run_id": run.run_id,
            "campaign_id": snapshot.lock.campaign_id,
            "created_at": created.isoformat(),
            "completed_at": (created + timedelta(minutes=1)).isoformat(),
            "evidence_bucket": "example/benchmark-runs",
            "evidence_prefix": prefix,
            "run_lock": {
                "path": "run.lock.json",
                "digest": f"sha256:{hashlib.sha256(run_lock).hexdigest()}",
                "size_bytes": len(run_lock),
            },
            "profiles": {
                "experiment": "sha256:" + "1" * 64,
                "model": "sha256:" + "2" * 64,
                "deployment": "sha256:" + "3" * 64,
                "agent": "sha256:" + "4" * 64,
            },
            "runtime": {
                "kind": "endpoint",
                "provider": "huggingface",
                "region": "aws-us-east-1",
                "hardware": "cpu-basic",
                "accelerator_count": 0,
            },
            "sanitizer_version": "harbor-hf/public-results/v1",
            "projection_version": "harbor-hf/results-projection/v1",
            "cleanup_outcome": "verified",
            "executions": [
                {
                    "execution_id": "execution-one",
                    "trial_id": trial.trial_id,
                    "physical_attempt": 1,
                    "status": "succeeded",
                    "failure_category": None,
                    "started_at": created.isoformat(),
                    "completed_at": (created + timedelta(minutes=1)).isoformat(),
                    "retry_reason": None,
                    "remote_job_id": "job-one",
                    "bundle_status": "verified",
                    "harbor_bundle": {
                        "manifest": {
                            "path": manifest_path,
                            "digest": "sha256:"
                            + hashlib.sha256(files[manifest_path]).hexdigest(),
                            "size_bytes": len(files[manifest_path]),
                        },
                        "archive": {
                            "path": archive_path,
                            "digest": "sha256:"
                            + hashlib.sha256(files[archive_path]).hexdigest(),
                            "size_bytes": len(files[archive_path]),
                        },
                        "harbor_revision": "a" * 40,
                        "harbor_version": "0.1.0",
                        "compatibility_schema": (
                            "harbor-hf/harbor-compatibility/v1alpha3"
                        ),
                        "request_digest": "sha256:" + "5" * 64,
                        "document_count": 2,
                    },
                }
            ],
        }
    ).encode()
    checksums = {
        path: f"sha256:{hashlib.sha256(content).hexdigest()}"
        for path, content in files.items()
    }
    files["checksums.json"] = json.dumps(checksums).encode()
    files["_SUCCESS"] = b""
    return MemoryEvidence(prefix, files)


def test_cancel_is_durable_idempotent_and_supports_dry_run(
    remote_spec: ExperimentSpec,
) -> None:
    store = MemoryStore(_snapshot(remote_spec))

    first = cancel_campaign(store, "campaign-one", reason="stop", dry_run=False)
    repeated = cancel_campaign(store, "campaign-one", reason="different", dry_run=False)
    dry_store = MemoryStore(_snapshot(remote_spec))
    dry = cancel_campaign(dry_store, "campaign-one", reason="stop", dry_run=True)

    assert first.recorded
    assert not repeated.recorded
    assert repeated.event_id == first.event_id
    assert not dry.recorded
    assert len(dry_store.snapshot.events) == 1


def test_retry_makes_backoff_ready_and_is_idempotent(
    remote_spec: ExperimentSpec,
) -> None:
    store = MemoryStore(_retry_snapshot(remote_spec))
    shard_id = store.snapshot.lock.runs[0].shards[0].shard_id

    first = retry_campaign_shard(
        store,
        "campaign-one",
        shard_id=shard_id,
        reason="retry now",
        dry_run=False,
    )
    repeated = retry_campaign_shard(
        store,
        "campaign-one",
        shard_id=shard_id,
        reason="retry again",
        dry_run=False,
    )
    _projection, plan = plan_reconciliation(store.snapshot.lock, store.snapshot.events)

    assert first.recorded
    assert not repeated.recorded
    assert repeated.event_id == first.event_id
    assert [action.kind for action in plan.actions] == ["retry-shard"]


def test_retry_rejects_nonretryable_shard(remote_spec: ExperimentSpec) -> None:
    store = MemoryStore(_snapshot(remote_spec))
    shard_id = store.snapshot.lock.runs[0].shards[0].shard_id

    with pytest.raises(ValueError, match="no retryable"):
        retry_campaign_shard(
            store,
            "campaign-one",
            shard_id=shard_id,
            reason="retry",
            dry_run=False,
        )


def test_resume_requires_verified_cleanup_and_records_resolution(
    remote_spec: ExperimentSpec,
) -> None:
    snapshot = _snapshot(remote_spec)
    shard = snapshot.lock.runs[0].shards[0]
    snapshot.events.extend(
        [
            new_event(
                subject_type="wave",
                subject_id="wave-one",
                kind="wave.cleanup-failed",
                producer="watchdog",
                payload=WaveLifecyclePayload(
                    deployment_digest=snapshot.lock.runs[0].deployment_digest,
                    provider="hf-inference-endpoints",
                    shard_ids=[shard.shard_id],
                ),
                clock=lambda: snapshot.lock.created_at - timedelta(seconds=1),
                identifier=lambda: "3" * 32,
            ),
            new_event(
                subject_type="campaign",
                subject_id=snapshot.lock.campaign_id,
                kind="campaign.manual-intervention-required",
                producer="reconciler",
                payload=LifecyclePayload(
                    parent_id="wave-one", message="cleanup failed"
                ),
                clock=lambda: snapshot.lock.created_at,
                identifier=lambda: "4" * 32,
            ),
        ]
    )
    store = MemoryStore(snapshot)

    with pytest.raises(ValueError, match="requires verified endpoint cleanup"):
        resume_campaign(
            store,
            "campaign-one",
            reason="not checked",
            cleanup_verified=False,
            dry_run=False,
        )
    result = resume_campaign(
        store,
        "campaign-one",
        reason="verified paused",
        cleanup_verified=True,
        dry_run=False,
    )
    repeated = resume_campaign(
        store,
        "campaign-one",
        reason="already verified",
        cleanup_verified=True,
        dry_run=False,
    )

    assert result.recorded
    assert not repeated.recorded
    assert repeated.event_id == result.event_id
    assert result.kind == "campaign.manual-intervention-resolved"
    projection = project_recovery(snapshot.lock, snapshot.events)
    assert projection.status == "active"
    assert projection.waves["wave-one"].status == "closed"


def test_resume_acknowledges_every_failed_cleanup_wave(
    remote_spec: ExperimentSpec,
) -> None:
    snapshot = _snapshot(remote_spec)
    shard = snapshot.lock.runs[0].shards[0]
    wave_payload = WaveLifecyclePayload(
        deployment_digest=snapshot.lock.runs[0].deployment_digest,
        provider="hf-inference-endpoints",
        shard_ids=[shard.shard_id],
    )
    for index, wave_id in enumerate(["wave-one", "wave-two"], start=1):
        snapshot.events.extend(
            [
                new_event(
                    subject_type="wave",
                    subject_id=wave_id,
                    kind="wave.cleanup-failed",
                    producer="watchdog",
                    payload=wave_payload,
                    clock=lambda index=index: (
                        snapshot.lock.created_at + timedelta(seconds=index * 2)
                    ),
                    identifier=lambda index=index: f"{index * 2:032x}",
                ),
                new_event(
                    subject_type="campaign",
                    subject_id=snapshot.lock.campaign_id,
                    kind="campaign.manual-intervention-required",
                    producer="reconciler",
                    payload=LifecyclePayload(parent_id=wave_id),
                    clock=lambda index=index: (
                        snapshot.lock.created_at + timedelta(seconds=index * 2 + 1)
                    ),
                    identifier=lambda index=index: f"{index * 2 + 1:032x}",
                ),
            ]
        )
    store = MemoryStore(snapshot)

    result = resume_campaign(
        store,
        "campaign-one",
        reason="all endpoints verified paused",
        cleanup_verified=True,
        dry_run=False,
    )

    resolution = next(
        event for event in snapshot.events if event.event_id == result.event_id
    )
    assert isinstance(resolution.payload, ManualInterventionResolutionPayload)
    assert resolution.payload.wave_ids == ["wave-one", "wave-two"]
    projection = project_recovery(snapshot.lock, snapshot.events)
    assert projection.status == "active"
    assert {
        projection.waves[wave_id].status for wave_id in resolution.payload.wave_ids
    } == {"closed"}


def test_unpaired_cleanup_failure_keeps_campaign_in_manual_intervention(
    remote_spec: ExperimentSpec,
) -> None:
    snapshot = _snapshot(remote_spec)
    shard = snapshot.lock.runs[0].shards[0]
    snapshot.events.append(
        new_event(
            subject_type="wave",
            subject_id="wave-unpaired",
            kind="wave.cleanup-failed",
            producer="watchdog",
            payload=WaveLifecyclePayload(
                deployment_digest=snapshot.lock.runs[0].deployment_digest,
                provider="hf-inference-endpoints",
                shard_ids=[shard.shard_id],
            ),
            clock=lambda: snapshot.lock.created_at,
            identifier=lambda: "5" * 32,
        )
    )

    assert (
        project_recovery(snapshot.lock, snapshot.events).status == "manual_intervention"
    )
    with pytest.raises(ValueError, match="requirement has not been recorded"):
        resume_campaign(
            MemoryStore(snapshot),
            "campaign-one",
            reason="verified",
            cleanup_verified=True,
            dry_run=False,
        )


def test_resume_accepts_cleanup_wave_already_closed(
    remote_spec: ExperimentSpec,
) -> None:
    snapshot = _snapshot(remote_spec)
    shard = snapshot.lock.runs[0].shards[0]
    wave_payload = WaveLifecyclePayload(
        deployment_digest=snapshot.lock.runs[0].deployment_digest,
        provider="hf-inference-endpoints",
        shard_ids=[shard.shard_id],
    )
    snapshot.events.extend(
        [
            new_event(
                subject_type="wave",
                subject_id="wave-one",
                kind="wave.cleanup-failed",
                producer="watchdog",
                payload=wave_payload,
                clock=lambda: snapshot.lock.created_at - timedelta(seconds=2),
                identifier=lambda: "2" * 32,
            ),
            new_event(
                subject_type="campaign",
                subject_id=snapshot.lock.campaign_id,
                kind="campaign.manual-intervention-required",
                producer="reconciler",
                payload=LifecyclePayload(parent_id="wave-one"),
                clock=lambda: snapshot.lock.created_at - timedelta(seconds=1),
                identifier=lambda: "3" * 32,
            ),
            new_event(
                subject_type="wave",
                subject_id="wave-one",
                kind="wave.closed",
                producer="watchdog",
                payload=wave_payload,
                clock=lambda: snapshot.lock.created_at,
                identifier=lambda: "4" * 32,
            ),
        ]
    )
    store = MemoryStore(snapshot)

    result = resume_campaign(
        store,
        "campaign-one",
        reason="verified paused",
        cleanup_verified=True,
        dry_run=False,
    )

    assert result.recorded
    assert project_recovery(snapshot.lock, snapshot.events).waves[
        "wave-one"
    ].status == ("closed")


def test_resume_validates_wave_and_orders_after_existing_events(
    remote_spec: ExperimentSpec,
) -> None:
    snapshot = _snapshot(remote_spec)
    required = new_event(
        subject_type="campaign",
        subject_id=snapshot.lock.campaign_id,
        kind="campaign.manual-intervention-required",
        producer="reconciler",
        payload=LifecyclePayload(parent_id="missing-wave"),
        clock=lambda: snapshot.lock.created_at,
        identifier=lambda: "2" * 32,
    )

    with pytest.raises(ValueError, match="does not reference recoverable cleanup"):
        durable_manual_intervention_resolution_event(
            snapshot.lock,
            [*snapshot.events, required],
            "verified",
            cleanup_verified=True,
            clock=lambda: snapshot.lock.created_at - timedelta(days=1),
        )

    shard = snapshot.lock.runs[0].shards[0]
    cleanup_failed = new_event(
        subject_type="wave",
        subject_id="wave-one",
        kind="wave.cleanup-failed",
        producer="watchdog",
        payload=WaveLifecyclePayload(
            deployment_digest=snapshot.lock.runs[0].deployment_digest,
            provider="hf-inference-endpoints",
            shard_ids=[shard.shard_id],
        ),
        clock=lambda: snapshot.lock.created_at,
        identifier=lambda: "3" * 32,
    )
    required = required.model_copy(
        update={
            "payload": LifecyclePayload(parent_id="wave-one"),
            "observed_at": snapshot.lock.created_at + timedelta(seconds=1),
        }
    )
    event, created = durable_manual_intervention_resolution_event(
        snapshot.lock,
        [*snapshot.events, cleanup_failed, required],
        "verified",
        cleanup_verified=True,
        clock=lambda: snapshot.lock.created_at - timedelta(days=1),
    )

    assert created
    assert event.observed_at == required.observed_at + timedelta(microseconds=1)


def test_verifies_and_publishes_campaign_evidence(
    remote_spec: ExperimentSpec,
) -> None:
    snapshot = _snapshot(remote_spec)
    evidence = _evidence(snapshot)

    verified = verify_campaign_artifacts(snapshot, namespace="osolmaz", reader=evidence)
    dry_run = publish_campaign_results(
        snapshot,
        namespace="osolmaz",
        reader=evidence,
        publisher=None,
        dry_run=True,
    )
    publisher = FakePublisher()
    published = publish_campaign_results(
        snapshot,
        namespace="osolmaz",
        reader=evidence,
        publisher=publisher,
        dry_run=False,
    )

    assert verified.verified
    assert verified.runs[0].row_counts == {
        "runs": 1,
        "trials": 1,
        "executions": 1,
        "metrics": 1,
        "artifacts": 0,
    }
    assert not dry_run.runs[0].published
    assert published.runs[0].published
    assert published.runs[0].result_revision == "a" * 40
    assert len(publisher.publications) == 1


def test_verification_rejects_tampered_bucket_evidence(
    remote_spec: ExperimentSpec,
) -> None:
    snapshot = _snapshot(remote_spec)
    evidence = _evidence(snapshot)
    evidence.files["run.lock.json"] = b"tampered"

    with pytest.raises(ResultPublicationError, match="checksum mismatch"):
        verify_campaign_artifacts(snapshot, namespace="osolmaz", reader=evidence)


def test_automatic_publisher_initializes_new_empty_public_repositories(
    remote_spec: ExperimentSpec,
) -> None:
    interactions: list[object] = []
    snapshot = _snapshot(remote_spec)
    source = _evidence(snapshot)
    reader = MemoryEvidence(
        source.prefix,
        source.files,
        interactions=interactions,
    )
    publisher = FakePublisher(interactions=interactions)
    repositories = MemoryRepositories(interactions)

    report = AutomaticCampaignPublisher(
        namespace="osolmaz",
        store=MemoryStore(snapshot),
        reader=reader,
        publisher=publisher,
        repositories=repositories,
    ).publish(snapshot.lock.campaign_id)

    assert report.campaign_id == snapshot.lock.campaign_id
    assert report.control_commit == snapshot.control_commit
    assert report.dry_run is False
    assert len(report.runs) == 1
    assert report.runs[0].model_dump(mode="json") == {
        "run_id": snapshot.lock.runs[0].run_id,
        "publication_id": publisher.publications[0].tables.publication_id,
        "result_dataset": "example/shellbench-results",
        "index_dataset": "example/benchmark-run-index",
        "published": True,
        "result_revision": "a" * 40,
        "index_revision": "b" * 40,
    }
    commit_kwargs = {
        "commit_message": "chore: initialize publication Dataset",
        "repo_type": "dataset",
        "revision": "main",
    }
    assert interactions[:9] == [
        (
            "create_repo",
            "example/shellbench-results",
            {"repo_type": "dataset", "private": False, "exist_ok": True},
        ),
        (
            "create_repo",
            "example/benchmark-run-index",
            {"repo_type": "dataset", "private": False, "exist_ok": True},
        ),
        (
            "repo_info",
            "example/shellbench-results",
            {"repo_type": "dataset"},
        ),
        (
            "repo_info",
            "example/benchmark-run-index",
            {"repo_type": "dataset"},
        ),
        ("create_commit", "example/shellbench-results", commit_kwargs),
        (
            "repo_info",
            "example/shellbench-results",
            {"repo_type": "dataset", "revision": "main"},
        ),
        ("create_commit", "example/benchmark-run-index", commit_kwargs),
        (
            "repo_info",
            "example/benchmark-run-index",
            {"repo_type": "dataset", "revision": "main"},
        ),
        "refresh",
    ]
    assert interactions[9] == (
        "publish",
        "example/shellbench-results",
        "example/benchmark-run-index",
        publisher.publications[0],
    )
    assert [
        repository for repository, _operations, _kwargs in repositories.commits
    ] == [
        "example/shellbench-results",
        "example/benchmark-run-index",
    ]
    for _repository, operations, kwargs in repositories.commits:
        assert kwargs == commit_kwargs
        assert "parent_commit" not in kwargs
        assert len(operations) == 1
        operation = operations[0]
        assert isinstance(operation, CommitOperationAdd)
        assert operation.path_in_repo == ".harbor-hf-initialized"
        assert operation.path_or_fileobj == b"harbor-hf publication Dataset\n"
    assert reader.refresh_calls == 1


def test_automatic_publisher_adopts_initialized_public_repositories(
    remote_spec: ExperimentSpec,
) -> None:
    interactions: list[object] = []
    snapshot = _snapshot(remote_spec)
    source = _evidence(snapshot)
    reader = MemoryEvidence(
        source.prefix,
        source.files,
        interactions=interactions,
    )
    publisher = FakePublisher(interactions=interactions)
    repositories = MemoryRepositories(
        interactions,
        existing={
            "example/shellbench-results": False,
            "example/benchmark-run-index": False,
        },
    )

    report = AutomaticCampaignPublisher(
        namespace="osolmaz",
        store=MemoryStore(snapshot),
        reader=reader,
        publisher=publisher,
        repositories=repositories,
    ).publish(snapshot.lock.campaign_id)

    assert report.runs[0].published
    assert repositories.private == {
        "example/shellbench-results": False,
        "example/benchmark-run-index": False,
    }
    assert reader.refresh_calls == 1
    assert len(publisher.publications) == 1
    assert repositories.commits == []


@pytest.mark.parametrize(
    "private_repository",
    ["example/shellbench-results", "example/benchmark-run-index"],
)
def test_automatic_publisher_rejects_existing_private_repository_before_evidence(
    remote_spec: ExperimentSpec,
    private_repository: str,
) -> None:
    interactions: list[object] = []
    snapshot = _snapshot(remote_spec)
    source = _evidence(snapshot)
    reader = MemoryEvidence(
        source.prefix,
        source.files,
        interactions=interactions,
    )
    publisher = FakePublisher(interactions=interactions)
    repositories = MemoryRepositories(
        interactions,
        existing={
            "example/shellbench-results": (
                private_repository == "example/shellbench-results"
            ),
            "example/benchmark-run-index": (
                private_repository == "example/benchmark-run-index"
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match=f"^Dataset repository {private_repository} must be public$",
    ):
        AutomaticCampaignPublisher(
            namespace="osolmaz",
            store=MemoryStore(snapshot),
            reader=reader,
            publisher=publisher,
            repositories=repositories,
        ).publish(snapshot.lock.campaign_id)

    assert interactions == [
        (
            "create_repo",
            "example/shellbench-results",
            {"repo_type": "dataset", "private": False, "exist_ok": True},
        ),
        (
            "create_repo",
            "example/benchmark-run-index",
            {"repo_type": "dataset", "private": False, "exist_ok": True},
        ),
        (
            "repo_info",
            "example/shellbench-results",
            {"repo_type": "dataset"},
        ),
        (
            "repo_info",
            "example/benchmark-run-index",
            {"repo_type": "dataset"},
        ),
    ]
    assert reader.refresh_calls == 0
    assert publisher.publications == []


def test_automatic_publisher_rejects_missing_index_without_side_effects(
    remote_spec: ExperimentSpec,
) -> None:
    interactions: list[object] = []
    spec = remote_spec.model_copy(
        update={
            "publishing": remote_spec.publishing.model_copy(
                update={"index_dataset": None}
            )
        }
    )
    snapshot = _snapshot(spec)
    source = _evidence(snapshot)
    reader = MemoryEvidence(
        source.prefix,
        source.files,
        interactions=interactions,
    )

    with pytest.raises(ValueError) as captured:
        AutomaticCampaignPublisher(
            namespace="osolmaz",
            store=MemoryStore(snapshot),
            reader=reader,
            publisher=FakePublisher(interactions=interactions),
            repositories=MemoryRepositories(interactions),
        ).publish(snapshot.lock.campaign_id)

    assert str(captured.value) == "campaign result publication requires index_dataset"
    assert interactions == []
    assert reader.refresh_calls == 0


def test_automatic_publisher_reports_campaign_identity_for_invalid_request(
    remote_spec: ExperimentSpec,
) -> None:
    interactions: list[object] = []
    snapshot = _snapshot(remote_spec)
    snapshot = CampaignSnapshot(
        lock=snapshot.lock,
        events=snapshot.events,
        request=b"not yaml: [",
        control_commit=snapshot.control_commit,
    )
    source = _evidence(snapshot)
    reader = MemoryEvidence(
        source.prefix,
        source.files,
        interactions=interactions,
    )

    with pytest.raises(ManifestError) as captured:
        AutomaticCampaignPublisher(
            namespace="osolmaz",
            store=MemoryStore(snapshot),
            reader=reader,
            publisher=FakePublisher(interactions=interactions),
            repositories=MemoryRepositories(interactions),
        ).publish(snapshot.lock.campaign_id)

    assert str(captured.value).startswith(
        "cannot read campaign campaign-one request: while parsing a flow node"
    )
    assert interactions == []
    assert reader.refresh_calls == 0
