from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from harbor_hf.campaigns import CampaignLock
from harbor_hf.control import ActionKind, CampaignEvent
from harbor_hf.recovery import (
    RecoveryProjection,
    TerminalDecision,
    WaveProjection,
    project_recovery,
    retry_is_ready,
)

_UNBOUNDED = 2**31 - 1
_BILLABLE_ACTIONS = {"submit-wave", "retry-shard"}
_PRIORITY: dict[ActionKind, int] = {
    "cancel-execution": 0,
    "cancel-wave": 1,
    "drain-wave": 2,
    "cleanup-wave": 3,
    "manual-intervention": 4,
    "publish-summary": 5,
    "publish-results": 6,
    "retry-shard": 7,
    "submit-wave": 8,
}


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DeploymentAdmission(FrozenModel):
    provider: str = Field(default="hf-inference-endpoints", min_length=1)
    estimated_wave_cost_microusd: int | None = Field(default=None, ge=0)


class AdmissionLimits(FrozenModel):
    action_limit: int = Field(default=64, ge=1)
    global_active_waves: int = Field(default=_UNBOUNDED, ge=1)
    deployment_active_waves: int = Field(default=_UNBOUNDED, ge=1)
    provider_active_waves: int = Field(default=_UNBOUNDED, ge=1)
    campaign_active_waves: int = Field(default=_UNBOUNDED, ge=1)


class AdmissionUsage(FrozenModel):
    global_active_waves: int = Field(default=0, ge=0)
    deployment_active_waves: dict[str, int] = Field(default_factory=dict)
    provider_active_waves: dict[str, int] = Field(default_factory=dict)
    campaign_active_waves: dict[str, int] = Field(default_factory=dict)
    campaign_spend_microusd: dict[str, int] = Field(default_factory=dict)


class ReconcileContext(FrozenModel):
    limits: AdmissionLimits = AdmissionLimits()
    usage: AdmissionUsage = AdmissionUsage()
    deployments: dict[str, DeploymentAdmission] = Field(default_factory=dict)


class ReconcileAction(FrozenModel):
    action_id: str
    action_key: str
    kind: ActionKind
    campaign_id: str
    deployment_digest: str = ""
    provider: str = ""
    wave_id: str | None = None
    shard_ids: list[str] = Field(default_factory=list)
    trial_ids: list[str] = Field(default_factory=list)
    target_ids: list[str] = Field(default_factory=list)
    estimated_cost_microusd: int | None = None


class BlockedAction(FrozenModel):
    kind: Literal["submit-wave", "retry-shard"]
    deployment_digest: str
    shard_ids: list[str]
    reason: Literal[
        "global-budget",
        "deployment-budget",
        "provider-budget",
        "campaign-budget",
        "spend-cap",
        "spend-estimate-missing",
        "backoff",
    ]


class ReconcilePlan(FrozenModel):
    campaign_id: str
    status: str
    action_count: int
    actions: list[ReconcileAction]
    blocked: list[BlockedAction] = Field(default_factory=list)
    terminal_decision: TerminalDecision | None = None


class _Candidate(FrozenModel):
    kind: ActionKind
    deployment_digest: str = ""
    provider: str = ""
    wave_id: str | None = None
    shard_ids: list[str] = Field(default_factory=list)
    trial_ids: list[str] = Field(default_factory=list)
    target_ids: list[str] = Field(default_factory=list)
    estimated_cost_microusd: int | None = None


class _MutableUsage:
    def __init__(
        self, context: ReconcileContext, projection: RecoveryProjection
    ) -> None:
        usage = context.usage
        self.global_waves = usage.global_active_waves + sum(
            wave.status != "closed" for wave in projection.waves.values()
        )
        self.deployments = dict(usage.deployment_active_waves)
        self.providers = dict(usage.provider_active_waves)
        self.campaigns = dict(usage.campaign_active_waves)
        self.spend = dict(usage.campaign_spend_microusd)
        for wave in projection.waves.values():
            if wave.status == "closed":
                continue
            self.deployments[wave.deployment_digest] = (
                self.deployments.get(wave.deployment_digest, 0) + 1
            )
            self.providers[wave.provider] = self.providers.get(wave.provider, 0) + 1
            campaign_id = projection.campaign.campaign_id
            self.campaigns[campaign_id] = self.campaigns.get(campaign_id, 0) + 1
        campaign_id = projection.campaign.campaign_id
        self.spend[campaign_id] = (
            self.spend.get(campaign_id, 0) + projection.spend_microusd
        )

    def admit(self, candidate: _Candidate, campaign_id: str) -> None:
        self.global_waves += 1
        self.deployments[candidate.deployment_digest] = (
            self.deployments.get(candidate.deployment_digest, 0) + 1
        )
        self.providers[candidate.provider] = (
            self.providers.get(candidate.provider, 0) + 1
        )
        self.campaigns[campaign_id] = self.campaigns.get(campaign_id, 0) + 1
        self.spend[campaign_id] = self.spend.get(campaign_id, 0) + (
            candidate.estimated_cost_microusd or 0
        )


def plan_reconciliation(
    lock: CampaignLock,
    events: list[CampaignEvent],
    *,
    context: ReconcileContext | None = None,
    now: datetime | None = None,
) -> tuple[RecoveryProjection, ReconcilePlan]:
    projection = project_recovery(lock, events)
    observed_context = context or ReconcileContext()
    requested_now = (now or datetime.now(UTC)).astimezone(UTC)
    observed_now = max(requested_now, projection.campaign.last_observed_at)
    if projection.campaign.status in {"completed", "partial", "failed", "cancelled"}:
        return projection, _plan(lock, projection, [], [], projection.terminal_decision)

    candidates, preblocked = _candidates(
        lock, projection, observed_context, observed_now
    )
    actions, blocked = _admit(
        lock,
        projection,
        candidates,
        observed_context,
        preblocked,
    )
    return projection, _plan(
        lock, projection, actions, blocked, projection.terminal_decision
    )


def _plan(
    lock: CampaignLock,
    projection: RecoveryProjection,
    actions: list[ReconcileAction],
    blocked: list[BlockedAction],
    terminal: TerminalDecision | None,
) -> ReconcilePlan:
    return ReconcilePlan(
        campaign_id=lock.campaign_id,
        status=projection.campaign.status,
        action_count=len(actions),
        actions=actions,
        blocked=blocked,
        terminal_decision=terminal,
    )


def _candidates(
    lock: CampaignLock,
    projection: RecoveryProjection,
    context: ReconcileContext,
    now: datetime,
) -> tuple[list[_Candidate], list[BlockedAction]]:
    candidates = _cleanup_candidates(lock, projection, now)
    blocked: list[BlockedAction] = []
    cancelling = projection.campaign.status in {
        "cancel_requested",
        "draining",
        "manual_intervention",
    }
    if not cancelling:
        retry, retry_blocked = _retry_candidates(lock, projection, context, now)
        candidates.extend(retry)
        blocked.extend(retry_blocked)
        candidates.extend(_new_wave_candidates(lock, projection, context))
    if projection.terminal_decision is not None:
        candidates.append(
            _Candidate(
                kind="publish-summary",
                target_ids=[projection.terminal_decision.summary_path],
            )
        )
    return sorted(candidates, key=_candidate_key), blocked


def _cleanup_candidates(
    lock: CampaignLock, projection: RecoveryProjection, now: datetime
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    cancelling = projection.campaign.status in {
        "cancel_requested",
        "draining",
        "manual_intervention",
    }
    force_cancel = _cancellation_grace_elapsed(lock, projection, now)
    if cancelling and force_cancel:
        candidates.extend(_execution_cancellations(projection))
    known_wave_ids: set[str] = set()
    for wave in projection.waves.values():
        known_wave_ids.add(wave.wave_id)
        candidates.extend(_wave_cleanup(wave, projection, cancelling, force_cancel))
    if cancelling:
        candidates.extend(
            _unobserved_wave_cancellations(
                lock, projection, known_wave_ids, force_cancel
            )
        )
    return candidates


def _cancellation_grace_elapsed(
    lock: CampaignLock, projection: RecoveryProjection, now: datetime
) -> bool:
    requested_at = projection.cancel_requested_at
    if requested_at is None:
        return False
    deadline = requested_at + timedelta(
        seconds=lock.recovery_policy.cancellation_grace_seconds
    )
    return now >= deadline


def _execution_cancellations(projection: RecoveryProjection) -> list[_Candidate]:
    return [
        _Candidate(
            kind="cancel-execution",
            wave_id=execution.wave_id,
            shard_ids=[execution.shard_id],
            trial_ids=[execution.trial_id],
            target_ids=[execution.execution_id],
        )
        for execution in projection.executions.values()
        if execution.status == "active"
    ]


def _wave_cleanup(
    observed: WaveProjection,
    projection: RecoveryProjection,
    cancelling: bool,
    force_cancel: bool,
) -> list[_Candidate]:
    def candidate(kind: ActionKind) -> _Candidate:
        return _Candidate(
            kind=kind,
            deployment_digest=observed.deployment_digest,
            provider=observed.provider,
            wave_id=observed.wave_id,
            shard_ids=observed.shard_ids,
            target_ids=[observed.wave_id],
        )

    if observed.status == "closed":
        return []
    if observed.status == "cleanup_failed":
        return [candidate("manual-intervention")]
    if cancelling:
        return _cancellation_wave_actions(observed, projection, force_cancel, candidate)
    terminal = all(
        projection.shards[shard_id].status
        in {"complete", "invalid", "failed_infrastructure", "cancelled"}
        for shard_id in observed.shard_ids
    )
    if terminal and observed.status not in {"draining", "cleaning"}:
        return [candidate("drain-wave")]
    if observed.status in {"draining", "cleaning"}:
        return [candidate("cleanup-wave")]
    return []


def _cancellation_wave_actions(
    observed: WaveProjection,
    projection: RecoveryProjection,
    force_cancel: bool,
    candidate: Callable[[ActionKind], _Candidate],
) -> list[_Candidate]:
    if force_cancel:
        return [candidate("cancel-wave"), candidate("cleanup-wave")]
    active_execution = any(
        execution.wave_id == observed.wave_id and execution.status == "active"
        for execution in projection.executions.values()
    )
    if active_execution:
        return [candidate("drain-wave")]
    if observed.status in {"draining", "cleaning"}:
        return [candidate("cleanup-wave")]
    return [candidate("drain-wave")]


def _unobserved_wave_cancellations(
    lock: CampaignLock,
    projection: RecoveryProjection,
    known_wave_ids: set[str],
    force_cancel: bool,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for action in projection.campaign.actions.values():
        if action.action_kind != "submit-wave" or action.status == "failed":
            continue
        wave_id = f"wave-{action.action_key}"
        if wave_id in known_wave_ids:
            continue
        deployment = _deployment_for_shards(lock, action.target_ids)
        kinds: tuple[ActionKind, ...] = (
            ("cancel-wave", "cleanup-wave") if force_cancel else ("drain-wave",)
        )
        candidates.extend(
            [
                _Candidate(
                    kind=kind,
                    deployment_digest=deployment,
                    wave_id=wave_id,
                    shard_ids=action.target_ids,
                    target_ids=[action.action_id],
                )
                for kind in kinds
            ]
        )
    return candidates


def _retry_candidates(
    lock: CampaignLock,
    projection: RecoveryProjection,
    context: ReconcileContext,
    now: datetime,
) -> tuple[list[_Candidate], list[BlockedAction]]:
    ready: dict[str, list[str]] = defaultdict(list)
    waiting: dict[str, list[str]] = defaultdict(list)
    for trial in projection.trials.values():
        if trial.status != "retry_wait":
            continue
        target = ready if retry_is_ready(trial, now) else waiting
        target[trial.shard_id].append(trial.trial_id)
    candidates = []
    blocked = []
    for shard_id, trial_ids in sorted(ready.items()):
        deployment = _deployment_for_shards(lock, [shard_id])
        admission = _deployment(context, deployment)
        candidates.append(
            _Candidate(
                kind="retry-shard",
                deployment_digest=deployment,
                provider=admission.provider,
                shard_ids=[shard_id],
                trial_ids=sorted(trial_ids),
                target_ids=sorted(trial_ids),
                estimated_cost_microusd=admission.estimated_wave_cost_microusd,
            )
        )
    for shard_id in sorted(waiting):
        deployment = _deployment_for_shards(lock, [shard_id])
        blocked.append(
            BlockedAction(
                kind="retry-shard",
                deployment_digest=deployment,
                shard_ids=[shard_id],
                reason="backoff",
            )
        )
    return candidates, blocked


def _new_wave_candidates(
    lock: CampaignLock,
    projection: RecoveryProjection,
    context: ReconcileContext,
) -> list[_Candidate]:
    assigned = _assigned_shards(projection)
    groups: dict[str, list[str]] = defaultdict(list)
    for run in sorted(lock.runs, key=lambda value: value.run_id):
        for shard in sorted(run.shards, key=lambda value: value.shard_id):
            status = projection.shards[shard.shard_id].status
            if shard.shard_id not in assigned and status in {"planned", "queued"}:
                groups[run.deployment_digest].append(shard.shard_id)
    candidates = []
    for deployment_digest in sorted(groups):
        admission = _deployment(context, deployment_digest)
        shard_ids = groups[deployment_digest]
        for offset in range(0, len(shard_ids), lock.max_shards_per_wave):
            chunk = shard_ids[offset : offset + lock.max_shards_per_wave]
            candidates.append(
                _Candidate(
                    kind="submit-wave",
                    deployment_digest=deployment_digest,
                    provider=admission.provider,
                    shard_ids=chunk,
                    target_ids=chunk,
                    estimated_cost_microusd=admission.estimated_wave_cost_microusd,
                )
            )
    return candidates


def _assigned_shards(projection: RecoveryProjection) -> set[str]:
    assigned = {
        shard_id for wave in projection.waves.values() for shard_id in wave.shard_ids
    }
    assigned.update(execution.shard_id for execution in projection.executions.values())
    for action in projection.campaign.actions.values():
        if (
            action.action_kind in {"submit-wave", "retry-shard"}
            and action.status != "failed"
        ):
            assigned.update(action.target_ids)
    return assigned


def _admit(
    lock: CampaignLock,
    projection: RecoveryProjection,
    candidates: list[_Candidate],
    context: ReconcileContext,
    preblocked: list[BlockedAction],
) -> tuple[list[ReconcileAction], list[BlockedAction]]:
    usage = _MutableUsage(context, projection)
    actions: list[ReconcileAction] = []
    blocked = list(preblocked)
    for candidate in candidates:
        action = _materialize(lock, projection, candidate)
        if action is None:
            continue
        if candidate.kind in _BILLABLE_ACTIONS:
            reason = _budget_reason(lock, candidate, context, usage)
            if reason is not None:
                billable_kind = cast(
                    Literal["submit-wave", "retry-shard"], candidate.kind
                )
                if billable_kind not in {"submit-wave", "retry-shard"}:
                    raise AssertionError("only billable actions use admission control")
                blocked.append(
                    BlockedAction(
                        kind=billable_kind,
                        deployment_digest=candidate.deployment_digest,
                        shard_ids=candidate.shard_ids,
                        reason=reason,
                    )
                )
                continue
            usage.admit(candidate, lock.campaign_id)
        if len(actions) >= context.limits.action_limit:
            break
        actions.append(action)
    return actions, blocked


def _budget_reason(
    lock: CampaignLock,
    candidate: _Candidate,
    context: ReconcileContext,
    usage: _MutableUsage,
) -> (
    Literal[
        "global-budget",
        "deployment-budget",
        "provider-budget",
        "campaign-budget",
        "spend-cap",
        "spend-estimate-missing",
    ]
    | None
):
    limits = context.limits
    if usage.global_waves >= limits.global_active_waves:
        return "global-budget"
    if (
        usage.deployments.get(candidate.deployment_digest, 0)
        >= limits.deployment_active_waves
    ):
        return "deployment-budget"
    if usage.providers.get(candidate.provider, 0) >= limits.provider_active_waves:
        return "provider-budget"
    campaign_limit = min(
        limits.campaign_active_waves, lock.recovery_policy.max_active_waves
    )
    if usage.campaigns.get(lock.campaign_id, 0) >= campaign_limit:
        return "campaign-budget"
    cap = lock.recovery_policy.spend_cap_microusd
    if cap is not None and candidate.estimated_cost_microusd is None:
        return "spend-estimate-missing"
    projected = usage.spend.get(lock.campaign_id, 0) + (
        candidate.estimated_cost_microusd or 0
    )
    if cap is not None and projected > cap:
        return "spend-cap"
    return None


def _materialize(
    lock: CampaignLock,
    projection: RecoveryProjection,
    candidate: _Candidate,
) -> ReconcileAction | None:
    retries = sum(
        action.action_kind == candidate.kind
        and action.target_ids == candidate.target_ids
        and action.status == "failed"
        for action in projection.campaign.actions.values()
    )
    value = candidate.model_dump(mode="json")
    value.update({"campaign_id": lock.campaign_id, "action_retry": retries})
    action_key = _short_digest(value)
    if any(
        action.action_key == action_key and action.status != "failed"
        for action in projection.campaign.actions.values()
    ):
        return None
    values = candidate.model_dump(mode="python")
    values["wave_id"] = candidate.wave_id or (
        f"wave-{action_key}" if candidate.kind == "submit-wave" else None
    )
    return ReconcileAction(
        action_id=f"act-{action_key}",
        action_key=action_key,
        campaign_id=lock.campaign_id,
        **values,
    )


def _deployment(
    context: ReconcileContext, deployment_digest: str
) -> DeploymentAdmission:
    return context.deployments.get(deployment_digest, DeploymentAdmission())


def _deployment_for_shards(lock: CampaignLock, shard_ids: list[str]) -> str:
    wanted = set(shard_ids)
    matches = {
        run.deployment_digest
        for run in lock.runs
        if any(shard.shard_id in wanted for shard in run.shards)
    }
    if len(matches) != 1:
        raise ValueError("action shards must belong to one deployment")
    return matches.pop()


def _candidate_key(candidate: _Candidate) -> tuple[int, str, str, str]:
    return (
        _PRIORITY[candidate.kind],
        candidate.deployment_digest,
        candidate.wave_id or "",
        ",".join(candidate.target_ids),
    )


def _short_digest(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest()[:24]
