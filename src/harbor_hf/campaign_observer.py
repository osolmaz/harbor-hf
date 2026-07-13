from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Protocol, cast

from pydantic import TypeAdapter, ValidationError

from harbor_hf.campaigns import CampaignLock, WaveLock
from harbor_hf.control import (
    CampaignEvent,
    EventKind,
    EventPayload,
    ExecutionOutcomePayload,
    ExecutionStartedPayload,
    RetryCategory,
    SubjectType,
    WaveLifecyclePayload,
    new_event,
)
from harbor_hf.models import ExperimentSpec
from harbor_hf.results import EvidenceReader
from harbor_hf.wave_worker import ExecutionLock

_JSON_OBJECT = TypeAdapter(dict[str, object])
_TERMINAL_MARKERS = frozenset({"_SUCCESS", "_FAILED", "_CANCELLED"})


class CampaignObservationError(RuntimeError):
    """Raised when terminal Bucket evidence cannot be projected safely."""


class CampaignObserver(Protocol):
    def observe(
        self, lock: CampaignLock, spec: ExperimentSpec
    ) -> list[CampaignEvent]: ...


class BucketCampaignObserver:
    """Derive compact, deterministic control events from terminal Bucket units."""

    def __init__(self, reader: EvidenceReader) -> None:
        self.reader = reader

    def observe(self, lock: CampaignLock, spec: ExperimentSpec) -> list[CampaignEvent]:
        paths = self.reader.list_files(
            bucket=spec.artifacts.bucket,
            prefix=lock.artifact_prefix,
        )
        events: list[CampaignEvent] = []
        for path in _wave_lock_paths(paths):
            wave_prefix = str(PurePosixPath(path).parent)
            marker = _terminal_marker(paths, wave_prefix)
            if marker is None:
                continue
            wave = WaveLock.model_validate_json(self._read(spec, lock, path))
            if wave.campaign_id != lock.campaign_id:
                raise CampaignObservationError(
                    "wave evidence belongs to another campaign"
                )
            self._verify_critical_unit(
                spec,
                lock,
                paths,
                wave_prefix,
                {"wave.lock.json", "events.jsonl", "wave-summary.json"},
            )
            raw_wave_events = _json_lines(
                self._read(spec, lock, f"{wave_prefix}/events.jsonl")
            )
            events.extend(_wave_events(lock, wave, marker, raw_wave_events))
            events.extend(self._execution_events(spec, lock, paths, wave))
        return sorted(events, key=lambda event: (event.observed_at, event.event_id))

    def _execution_events(
        self,
        spec: ExperimentSpec,
        campaign: CampaignLock,
        paths: list[str],
        wave: WaveLock,
    ) -> list[CampaignEvent]:
        events: list[CampaignEvent] = []
        for path in _execution_lock_paths(paths):
            execution = ExecutionLock.model_validate_json(
                self._read(spec, campaign, path)
            )
            if execution.wave_id != wave.wave_id:
                continue
            prefix = str(PurePosixPath(path).parent)
            marker = _terminal_marker(paths, prefix)
            if marker is None:
                continue
            critical = {"execution.lock.json", "events.jsonl"}
            if marker == "_SUCCESS":
                critical.add("verification.json")
            self._verify_critical_unit(spec, campaign, paths, prefix, critical)
            raw_events = _json_lines(
                self._read(spec, campaign, f"{prefix}/events.jsonl")
            )
            events.extend(
                _execution_control_events(
                    campaign,
                    execution,
                    marker,
                    raw_events,
                    self._failure_message(spec, campaign, prefix, marker),
                )
            )
        return events

    def _failure_message(
        self,
        spec: ExperimentSpec,
        campaign: CampaignLock,
        prefix: str,
        marker: str,
    ) -> str | None:
        if marker != "_FAILED":
            return None
        value = _JSON_OBJECT.validate_json(
            self._read(spec, campaign, f"{prefix}/_FAILED")
        )
        message = value.get("message")
        return message if isinstance(message, str) else None

    def _verify_critical_unit(
        self,
        spec: ExperimentSpec,
        campaign: CampaignLock,
        paths: list[str],
        prefix: str,
        critical: set[str],
    ) -> None:
        manifest_path = f"{prefix}/checksums.json"
        if manifest_path not in paths:
            raise CampaignObservationError("terminal evidence has no checksum manifest")
        try:
            manifest = cast(
                dict[str, str],
                json.loads(self._read(spec, campaign, manifest_path)),
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CampaignObservationError(
                "terminal checksum manifest is invalid"
            ) from error
        if not all(
            isinstance(path, str)
            and isinstance(digest, str)
            and digest.startswith("sha256:")
            for path, digest in manifest.items()
        ):
            raise CampaignObservationError("terminal checksum manifest is invalid")
        if not critical.issubset(manifest):
            raise CampaignObservationError("terminal checksum manifest is incomplete")
        for relative in sorted(critical):
            content = self._read(spec, campaign, f"{prefix}/{relative}")
            observed = f"sha256:{hashlib.sha256(content).hexdigest()}"
            if manifest[relative] != observed:
                raise CampaignObservationError(
                    f"terminal evidence checksum mismatch: {prefix}/{relative}"
                )

    def _read(self, spec: ExperimentSpec, campaign: CampaignLock, path: str) -> bytes:
        return self.reader.read_bytes(
            bucket=spec.artifacts.bucket,
            prefix=campaign.artifact_prefix,
            path=path,
        )


def _wave_lock_paths(paths: list[str]) -> list[str]:
    return sorted(
        path
        for path in paths
        if len(PurePosixPath(path).parts) == 3
        and PurePosixPath(path).parts[0] == "waves"
        and PurePosixPath(path).name == "wave.lock.json"
    )


def _execution_lock_paths(paths: list[str]) -> list[str]:
    return sorted(
        path
        for path in paths
        if PurePosixPath(path).name == "execution.lock.json"
        and "executions" in PurePosixPath(path).parts
    )


def _terminal_marker(paths: list[str], prefix: str) -> str | None:
    markers = {
        PurePosixPath(path).name
        for path in paths
        if str(PurePosixPath(path).parent) == prefix
        and PurePosixPath(path).name in _TERMINAL_MARKERS
    }
    if not markers:
        return None
    if len(markers) != 1:
        raise CampaignObservationError("terminal evidence has conflicting markers")
    return markers.pop()


def _json_lines(value: bytes) -> list[dict[str, object]]:
    try:
        records = [
            _JSON_OBJECT.validate_json(line)
            for line in value.splitlines()
            if line.strip()
        ]
    except ValidationError as error:
        raise CampaignObservationError("lifecycle event log is invalid") from error
    if not records:
        raise CampaignObservationError("lifecycle event log is empty")
    return records


def _event_time(records: list[dict[str, object]], *names: str) -> datetime:
    for record in records:
        if record.get("event") not in names:
            continue
        value = record.get("at")
        if not isinstance(value, str):
            break
        try:
            observed = datetime.fromisoformat(value)
        except ValueError as error:
            raise CampaignObservationError(
                "lifecycle event timestamp is invalid"
            ) from error
        if observed.tzinfo is None:
            raise CampaignObservationError("lifecycle event timestamp has no timezone")
        return observed.astimezone(UTC)
    raise CampaignObservationError(
        "lifecycle event log omits required events: " + ", ".join(names)
    )


def _wave_events(
    campaign: CampaignLock,
    wave: WaveLock,
    marker: str,
    records: list[dict[str, object]],
) -> list[CampaignEvent]:
    provider = next(
        run.provider
        for run in campaign.runs
        if run.deployment_digest == wave.deployment_digest
    ) or "hf-inference-endpoints"
    payload = WaveLifecyclePayload(
        deployment_digest=wave.deployment_digest,
        provider=provider,
        shard_ids=wave.shard_ids,
        estimated_cost_microusd=0,
    )
    active = _event_time(records, "wave_started")
    finished = _event_time(records, "wave_succeeded", "wave_failed")
    events = [
        _event(
            campaign,
            subject_type="wave",
            subject_id=wave.wave_id,
            kind="wave.active",
            payload=payload,
            observed_at=active,
            identity=f"{wave.wave_id}:active",
        )
    ]
    cleanup_failed = any(
        record.get("event") == "endpoint_cleanup_failed" for record in records
    )
    if cleanup_failed:
        events.append(
            _event(
                campaign,
                subject_type="wave",
                subject_id=wave.wave_id,
                kind="wave.cleanup-failed",
                payload=payload,
                observed_at=finished,
                identity=f"{wave.wave_id}:cleanup-failed",
            )
        )
        return events
    cleaning = _optional_event_time(records, "endpoint_pause_requested") or finished
    events.extend(
        [
            _event(
                campaign,
                subject_type="wave",
                subject_id=wave.wave_id,
                kind="wave.cleaning",
                payload=payload,
                observed_at=cleaning,
                identity=f"{wave.wave_id}:cleaning",
            ),
            _event(
                campaign,
                subject_type="wave",
                subject_id=wave.wave_id,
                kind="wave.closed",
                payload=payload,
                observed_at=max(cleaning, finished),
                identity=f"{wave.wave_id}:closed:{marker}",
            ),
        ]
    )
    return events


def _execution_control_events(
    campaign: CampaignLock,
    execution: ExecutionLock,
    marker: str,
    records: list[dict[str, object]],
    message: str | None,
) -> list[CampaignEvent]:
    started = _event_time(records, "execution_started")
    finished = _event_time(records, "execution_succeeded", "execution_failed")
    start = _event(
        campaign,
        subject_type="execution",
        subject_id=execution.execution_id,
        kind="execution.started",
        payload=ExecutionStartedPayload(
            trial_id=execution.trial_id,
            shard_id=execution.shard_id,
            physical_attempt=execution.physical_attempt,
            wave_id=execution.wave_id,
        ),
        observed_at=started,
        identity=f"{execution.execution_id}:started",
    )
    if marker == "_SUCCESS":
        kind = "execution.completed"
        category: RetryCategory | None = None
    elif marker == "_CANCELLED":
        kind = "execution.cancelled"
        category = None
    else:
        kind = "execution.failed"
        category = "transient"
    outcome = _event(
        campaign,
        subject_type="execution",
        subject_id=execution.execution_id,
        kind=kind,
        payload=ExecutionOutcomePayload(
            trial_id=execution.trial_id,
            physical_attempt=execution.physical_attempt,
            category=category,
            message=message,
        ),
        observed_at=finished,
        identity=f"{execution.execution_id}:{kind}",
    )
    return [start, outcome]


def _optional_event_time(
    records: list[dict[str, object]], name: str
) -> datetime | None:
    try:
        return _event_time(records, name)
    except CampaignObservationError:
        return None


def _event(
    campaign: CampaignLock,
    *,
    subject_type: SubjectType,
    subject_id: str,
    kind: EventKind,
    payload: EventPayload,
    observed_at: datetime,
    identity: str,
) -> CampaignEvent:
    identifier = hashlib.sha256(
        f"{campaign.campaign_id}:{identity}".encode()
    ).hexdigest()[:32]
    return new_event(
        subject_type=subject_type,
        subject_id=subject_id,
        kind=kind,
        producer="wave-controller",
        payload=payload,
        clock=lambda: observed_at,
        identifier=lambda: identifier,
    )
