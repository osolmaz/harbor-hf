from __future__ import annotations

from typing import Protocol

from huggingface_hub import CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel, ConfigDict

from harbor_hf.campaign_finalizer import (
    BucketCampaignFinalizer,
    ImmutableEvidenceWriter,
)
from harbor_hf.control import CampaignEvent, CampaignSnapshot
from harbor_hf.io import load_experiment_bytes
from harbor_hf.recovery import (
    durable_cancellation_event,
    durable_shard_retry_event,
    project_recovery,
    seal_partial_projection,
)
from harbor_hf.result_publisher import PublicationResult
from harbor_hf.results import (
    EvidenceReader,
    EvidenceSource,
    ResultPublication,
    TableName,
    build_result_publication,
    build_result_tables,
)

_DATASET_INITIALIZATION_PATH = ".harbor-hf-initialized"
_DATASET_INITIALIZATION_PAYLOAD = b"harbor-hf publication Dataset\n"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CampaignEventResult(FrozenModel):
    campaign_id: str
    event_id: str
    kind: str
    recorded: bool
    dry_run: bool


class VerifiedRun(FrozenModel):
    run_id: str
    publication_id: str
    source_prefix: str
    source_checksum: str
    row_counts: dict[TableName, int]


class ArtifactVerificationReport(FrozenModel):
    campaign_id: str
    artifact_bucket: str
    control_commit: str
    verified: bool = True
    runs: list[VerifiedRun]


class PublishedRun(FrozenModel):
    run_id: str
    publication_id: str
    result_dataset: str
    index_dataset: str
    published: bool
    result_revision: str | None = None
    index_revision: str | None = None


class CampaignPublicationReport(FrozenModel):
    campaign_id: str
    control_commit: str
    dry_run: bool
    runs: list[PublishedRun]


class SealedRun(FrozenModel):
    run_id: str
    source_prefix: str
    source_checksum: str | None = None


class CampaignSealReport(FrozenModel):
    campaign_id: str
    artifact_bucket: str
    dry_run: bool
    runs: list[SealedRun]


class CampaignEventStore(Protocol):
    def load_snapshot(self, campaign_id: str) -> CampaignSnapshot: ...

    def ensure_event(self, campaign_id: str, event: CampaignEvent) -> bool: ...


class ResultPublisher(Protocol):
    def publish(
        self,
        publication: ResultPublication,
        *,
        result_dataset: str,
        index_dataset: str,
    ) -> PublicationResult: ...


class RefreshingEvidenceReader(EvidenceReader, Protocol):
    def refresh(self) -> None: ...


class DatasetRepositoryApi(Protocol):
    def create_repo(self, repo_id: str, **kwargs: object) -> object: ...

    def repo_info(self, repo_id: str, **kwargs: object) -> object: ...

    def create_commit(
        self, repo_id: str, operations: list[object], **kwargs: object
    ) -> object: ...


class AutomaticCampaignPublisher:
    """Publish every complete run after terminal evidence is finalized."""

    def __init__(
        self,
        *,
        namespace: str,
        store: CampaignEventStore,
        reader: RefreshingEvidenceReader,
        publisher: ResultPublisher,
        repositories: DatasetRepositoryApi,
    ) -> None:
        self.namespace = namespace
        self.store = store
        self.reader = reader
        self.publisher = publisher
        self.repositories = repositories

    def publish(self, campaign_id: str) -> CampaignPublicationReport:
        snapshot = self.store.load_snapshot(campaign_id)
        spec = load_experiment_bytes(
            snapshot.request,
            source=f"campaign {campaign_id} request",
        )
        if spec.publishing.index_dataset is None:
            raise ValueError("campaign result publication requires index_dataset")
        repositories = (spec.publishing.dataset, spec.publishing.index_dataset)
        for repository in repositories:
            self.repositories.create_repo(
                repository,
                repo_type="dataset",
                private=False,
                exist_ok=True,
            )
        repository_info = [
            self.repositories.repo_info(repository, repo_type="dataset")
            for repository in repositories
        ]
        for repository, info in zip(repositories, repository_info, strict=True):
            if getattr(info, "private", None) is not False:
                raise ValueError(f"Dataset repository {repository} must be public")
        for repository, info in zip(repositories, repository_info, strict=True):
            if _commit_identity(info) is None:
                _initialize_public_dataset_repository(repository, self.repositories)
        self.reader.refresh()
        return publish_campaign_results(
            snapshot,
            namespace=self.namespace,
            reader=self.reader,
            publisher=self.publisher,
            dry_run=False,
        )


def _initialize_public_dataset_repository(
    repository: str, api: DatasetRepositoryApi
) -> None:
    initialization_error: HfHubHTTPError | None = None
    try:
        api.create_commit(
            repository,
            [
                CommitOperationAdd(
                    path_in_repo=_DATASET_INITIALIZATION_PATH,
                    path_or_fileobj=_DATASET_INITIALIZATION_PAYLOAD,
                )
            ],
            commit_message="chore: initialize publication Dataset",
            repo_type="dataset",
            revision="main",
        )
    except HfHubHTTPError as error:
        initialization_error = error
    info = api.repo_info(repository, repo_type="dataset", revision="main")
    if _commit_identity(info) is not None:
        return
    if initialization_error is not None:
        raise initialization_error
    raise ValueError(f"Dataset repository {repository} has no commit identity")


def _commit_identity(info: object) -> str | None:
    revision = getattr(info, "sha", None)
    return revision if isinstance(revision, str) and revision else None


def cancel_campaign(
    store: CampaignEventStore,
    campaign_id: str,
    *,
    reason: str,
    dry_run: bool,
) -> CampaignEventResult:
    snapshot = store.load_snapshot(campaign_id)
    event, created = durable_cancellation_event(snapshot.lock, snapshot.events, reason)
    recorded = (
        False if dry_run or not created else store.ensure_event(campaign_id, event)
    )
    return CampaignEventResult(
        campaign_id=campaign_id,
        event_id=event.event_id,
        kind=event.kind,
        recorded=recorded,
        dry_run=dry_run,
    )


def retry_campaign_shard(
    store: CampaignEventStore,
    campaign_id: str,
    *,
    shard_id: str,
    reason: str,
    dry_run: bool,
) -> CampaignEventResult:
    snapshot = store.load_snapshot(campaign_id)
    event, created = durable_shard_retry_event(
        snapshot.lock, snapshot.events, shard_id, reason
    )
    recorded = (
        False if dry_run or not created else store.ensure_event(campaign_id, event)
    )
    return CampaignEventResult(
        campaign_id=campaign_id,
        event_id=event.event_id,
        kind=event.kind,
        recorded=recorded,
        dry_run=dry_run,
    )


def seal_partial_campaign_runs(
    snapshot: CampaignSnapshot,
    *,
    namespace: str,
    reader: EvidenceReader,
    writer: ImmutableEvidenceWriter | None,
    dry_run: bool,
) -> CampaignSealReport:
    spec = load_experiment_bytes(
        snapshot.request,
        source=f"campaign {snapshot.lock.campaign_id} request",
    )
    if spec.remote is None or spec.remote.job.namespace != namespace:
        raise ValueError("campaign request does not match the control namespace")
    projection = seal_partial_projection(
        project_recovery(snapshot.lock, snapshot.events)
    )
    checksums: dict[str, str] = {}
    if not dry_run:
        if writer is None:
            raise ValueError("evidence writer is required outside dry-run")
        checksums = BucketCampaignFinalizer(reader, writer).seal_runs(
            snapshot.lock,
            spec,
            projection,
        )
    return CampaignSealReport(
        campaign_id=snapshot.lock.campaign_id,
        artifact_bucket=spec.artifacts.bucket,
        dry_run=dry_run,
        runs=[
            SealedRun(
                run_id=run.run_id,
                source_prefix=f"{snapshot.lock.artifact_prefix}/runs/{run.run_id}",
                source_checksum=checksums.get(run.run_id),
            )
            for run in snapshot.lock.runs
        ],
    )


def verify_campaign_artifacts(
    snapshot: CampaignSnapshot,
    *,
    namespace: str,
    reader: EvidenceReader,
) -> ArtifactVerificationReport:
    report, _publications, _destinations = _prepare_publications(
        snapshot, namespace=namespace, reader=reader
    )
    return report


def publish_campaign_results(
    snapshot: CampaignSnapshot,
    *,
    namespace: str,
    reader: EvidenceReader,
    publisher: ResultPublisher | None,
    dry_run: bool,
) -> CampaignPublicationReport:
    verification, publications, destinations = _prepare_publications(
        snapshot, namespace=namespace, reader=reader
    )
    result_dataset, index_dataset = destinations
    published: list[PublishedRun] = []
    for verified, publication in zip(verification.runs, publications, strict=True):
        receipt = None
        if not dry_run:
            if publisher is None:
                raise ValueError("result publisher is required outside dry-run")
            receipt = publisher.publish(
                publication,
                result_dataset=result_dataset,
                index_dataset=index_dataset,
            )
        published.append(
            PublishedRun(
                run_id=verified.run_id,
                publication_id=verified.publication_id,
                result_dataset=result_dataset,
                index_dataset=index_dataset,
                published=receipt is not None,
                result_revision=(receipt.result_revision if receipt else None),
                index_revision=(receipt.index_revision if receipt else None),
            )
        )
    return CampaignPublicationReport(
        campaign_id=snapshot.lock.campaign_id,
        control_commit=snapshot.control_commit,
        dry_run=dry_run,
        runs=published,
    )


def _prepare_publications(
    snapshot: CampaignSnapshot,
    *,
    namespace: str,
    reader: EvidenceReader,
) -> tuple[
    ArtifactVerificationReport,
    list[ResultPublication],
    tuple[str, str],
]:
    spec = load_experiment_bytes(
        snapshot.request,
        source=f"campaign {snapshot.lock.campaign_id} request",
    )
    if spec.remote is None or spec.remote.job.namespace != namespace:
        raise ValueError("campaign request does not match the control namespace")
    index_dataset = spec.publishing.index_dataset
    if index_dataset is None:
        raise ValueError("campaign result publication requires index_dataset")
    verified: list[VerifiedRun] = []
    publications: list[ResultPublication] = []
    for run in snapshot.lock.runs:
        source = EvidenceSource(
            bucket=spec.artifacts.bucket,
            prefix=f"{snapshot.lock.artifact_prefix}/runs/{run.run_id}",
        )
        tables = build_result_tables(
            reader,
            source,
            control_commit=snapshot.control_commit,
        )
        observed = tables.runs[0]
        if (
            observed.run_id != run.run_id
            or observed.campaign_id != snapshot.lock.campaign_id
        ):
            raise ValueError("run evidence does not match the campaign lock")
        verified.append(
            VerifiedRun(
                run_id=run.run_id,
                publication_id=tables.publication_id,
                source_prefix=source.prefix,
                source_checksum=observed.source_checksum,
                row_counts={
                    "runs": len(tables.runs),
                    "trials": len(tables.trials),
                    "executions": len(tables.executions),
                    "metrics": len(tables.metrics),
                    "artifacts": len(tables.artifacts),
                },
            )
        )
        publications.append(build_result_publication(tables))
    report = ArtifactVerificationReport(
        campaign_id=snapshot.lock.campaign_id,
        artifact_bucket=spec.artifacts.bucket,
        control_commit=snapshot.control_commit,
        runs=verified,
    )
    return report, publications, (spec.publishing.dataset, index_dataset)
