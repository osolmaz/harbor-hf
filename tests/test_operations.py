from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import pytest
import yaml

from harbor_hf.campaigns import build_campaign_lock, build_campaign_plan
from harbor_hf.control import (
    CampaignEvent,
    CampaignSnapshot,
    CampaignSubmittedPayload,
    ExecutionOutcomePayload,
    ExecutionStartedPayload,
    new_event,
)
from harbor_hf.models import ExperimentSpec
from harbor_hf.operations import (
    cancel_campaign,
    publish_campaign_results,
    retry_campaign_shard,
    verify_campaign_artifacts,
)
from harbor_hf.reconciler import plan_reconciliation
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
    def __init__(self, prefix: str, files: dict[str, bytes]) -> None:
        self.prefix = prefix
        self.files = files

    def list_files(self, *, bucket: str, prefix: str) -> list[str]:
        assert bucket == "example/benchmark-runs"
        assert prefix == self.prefix
        return list(reversed(self.files))

    def read_bytes(self, *, bucket: str, prefix: str, path: str) -> bytes:
        assert bucket == "example/benchmark-runs"
        assert prefix == self.prefix
        return self.files[path]


class FakePublisher:
    def __init__(self) -> None:
        self.publications: list[ResultPublication] = []

    def publish(
        self,
        publication: ResultPublication,
        *,
        result_dataset: str,
        index_dataset: str,
    ) -> PublicationResult:
        assert result_dataset == "example/shellbench-results"
        assert index_dataset == "example/benchmark-run-index"
        self.publications.append(publication)
        return PublicationResult(
            publication_id=publication.tables.publication_id,
            result_dataset=result_dataset,
            result_revision="a" * 40,
            index_dataset=index_dataset,
            index_revision="b" * 40,
        )


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
            "benchmark": "shellbench",
            "benchmark_revision": "sha256:" + "1" * 64,
            "result_kind": "ordinary",
            "outcome": "complete",
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
                "outcome": "complete",
            }
        ],
        "executions": [
            {
                "execution_id": "execution-one",
                "trial_id": trial.trial_id,
                "physical_attempt": 1,
                "runtime_kind": "endpoint",
                "status": "succeeded",
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
        "run.lock.json": json.dumps({"run_id": run.run_id}).encode(),
        "run-summary.json": json.dumps(summary).encode(),
    }
    checksums = {
        path: f"sha256:{hashlib.sha256(content).hexdigest()}"
        for path, content in files.items()
    }
    files["checksums.json"] = json.dumps(checksums).encode()
    files["_SUCCESS"] = b""
    prefix = f"{snapshot.lock.artifact_prefix}/runs/{run.run_id}"
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
