from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from harbor_hf.campaigns import CampaignLock, CampaignRunLock
from harbor_hf.control import ActionKind, ActionProjection, CampaignEvent
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
    "exhaust-trials": 4,
    "manual-intervention": 5,
    "publish-summary": 6,
    "publish-results": 7,
    "retry-shard": 8,
    "submit-wave": 9,
}


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DeploymentAdmission(FrozenModel):
    provider: str | None = Field(default=None, min_length=1)
    estimated_wave_cost_microusd: int | None = Field(default=None, ge=0)


class AdmissionLimits(FrozenModel):
    action_limit: int = Field(default=64, ge=1)
    global_active_waves: int = Field(default=_UNBOUNDED, ge=1)
    deployment_active_waves: int = Field(default=1, ge=1)
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
    spend_cap_microusd: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )


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
    spend_cap_microusd: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )


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
            self.spend.get(campaign_id, 0)
            + projection.spend_microusd
            + sum(wave.estimated_cost_microusd for wave in projection.waves.values())
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
        if (
            action.action_kind not in {"submit-wave", "retry-shard"}
            or action.status == "failed"
        ):
            continue
        wave_id = f"wave-{action.action_key}"
        if wave_id in known_wave_ids:
            continue
        shard_ids = _action_shard_ids(lock, action.action_kind, action.target_ids)
        deployment = _deployment_for_shards(lock, shard_ids)
        kinds: tuple[ActionKind, ...] = (
            ("cancel-wave", "cleanup-wave") if force_cancel else ("drain-wave",)
        )
        candidates.extend(
            [
                _Candidate(
                    kind=kind,
                    deployment_digest=deployment,
                    wave_id=wave_id,
                    shard_ids=shard_ids,
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
    reserved_trial_ids = {
        trial_id
        for action in projection.campaign.actions.values()
        if action.action_kind == "retry-shard"
        and not _action_is_spent(projection, action)
        for trial_id in action.target_ids
    }
    ready: dict[str, list[str]] = defaultdict(list)
    waiting: dict[str, list[str]] = defaultdict(list)
    for trial in projection.trials.values():
        if trial.status != "retry_wait":
            continue
        if trial.trial_id in reserved_trial_ids:
            continue
        target = ready if retry_is_ready(trial, now) else waiting
        target[trial.shard_id].append(trial.trial_id)
    groups: dict[str, list[str]] = defaultdict(list)
    for shard_id in sorted(ready):
        deployment = _deployment_for_shards(lock, [shard_id])
        groups[deployment].append(shard_id)
    candidates = []
    blocked = []
    for deployment in sorted(groups):
        admission = _deployment(lock, context, deployment)
        shard_ids = groups[deployment]
        for offset in range(0, len(shard_ids), lock.max_shards_per_wave):
            chunk = shard_ids[offset : offset + lock.max_shards_per_wave]
            trial_ids = sorted(
                trial_id for shard_id in chunk for trial_id in ready[shard_id]
            )
            candidates.append(
                _Candidate(
                    kind="retry-shard",
                    deployment_digest=deployment,
                    provider=_admission_provider(admission),
                    shard_ids=chunk,
                    trial_ids=trial_ids,
                    target_ids=trial_ids,
                    estimated_cost_microusd=_estimated_retry_cost(
                        lock,
                        deployment,
                        admission.estimated_wave_cost_microusd,
                        len(trial_ids),
                    ),
                    spend_cap_microusd=_run_admission(
                        lock, deployment
                    ).spend_cap_microusd,
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


def _estimated_retry_cost(
    lock: CampaignLock,
    deployment_digest: str,
    estimated_wave_cost_microusd: int | None,
    retry_trial_count: int,
) -> int | None:
    if estimated_wave_cost_microusd is None:
        return None
    shard_sizes = sorted(
        (
            (shard.shard_id, len(shard.trials))
            for run in lock.runs
            if run.deployment_digest == deployment_digest
            for shard in run.shards
        ),
        key=lambda item: item[0],
    )
    capacities = [
        sum(
            size
            for _shard_id, size in shard_sizes[
                offset : offset + lock.max_shards_per_wave
            ]
        )
        for offset in range(0, len(shard_sizes), lock.max_shards_per_wave)
    ]
    capacity = max(capacities, default=0)
    if capacity <= 0 or retry_trial_count <= 0:
        raise ValueError("retry cost requires a non-empty deployment wave")
    return max(
        1,
        (estimated_wave_cost_microusd * retry_trial_count + capacity - 1) // capacity,
    )


def _new_wave_candidates(
    lock: CampaignLock,
    projection: RecoveryProjection,
    context: ReconcileContext,
) -> list[_Candidate]:
    assigned = _assigned_shards(lock, projection)
    groups: dict[str, list[str]] = defaultdict(list)
    for run in sorted(lock.runs, key=lambda value: value.run_id):
        for shard in sorted(run.shards, key=lambda value: value.shard_id):
            status = projection.shards[shard.shard_id].status
            if shard.shard_id not in assigned and status in {"planned", "queued"}:
                groups[run.deployment_digest].append(shard.shard_id)
    candidates = []
    for deployment_digest in sorted(groups):
        admission = _deployment(lock, context, deployment_digest)
        shard_ids = groups[deployment_digest]
        for offset in range(0, len(shard_ids), lock.max_shards_per_wave):
            chunk = shard_ids[offset : offset + lock.max_shards_per_wave]
            candidates.append(
                _Candidate(
                    kind="submit-wave",
                    deployment_digest=deployment_digest,
                    provider=_admission_provider(admission),
                    shard_ids=chunk,
                    target_ids=chunk,
                    estimated_cost_microusd=admission.estimated_wave_cost_microusd,
                    spend_cap_microusd=_run_admission(
                        lock, deployment_digest
                    ).spend_cap_microusd,
                )
            )
    return candidates


def _assigned_shards(lock: CampaignLock, projection: RecoveryProjection) -> set[str]:
    assigned: set[str] = set()
    for wave in projection.waves.values():
        if wave.status != "closed":
            assigned.update(wave.shard_ids)
            continue
        # A closed wave keeps a shard assigned only while it left terminal or
        # retryable execution evidence; untouched shards return to the pool.
        assigned.update(
            shard_id
            for shard_id in wave.shard_ids
            if _shard_has_execution_evidence(projection, shard_id)
        )
    assigned.update(execution.shard_id for execution in projection.executions.values())
    for action in projection.campaign.actions.values():
        if action.action_kind in {
            "submit-wave",
            "retry-shard",
        } and not _action_is_spent(projection, action):
            assigned.update(
                _action_shard_ids(lock, action.action_kind, action.target_ids)
            )
    return assigned


def _shard_has_execution_evidence(
    projection: RecoveryProjection, shard_id: str
) -> bool:
    return any(
        projection.trials[trial_id].executions
        or projection.trials[trial_id].status != "planned"
        for trial_id in projection.shards[shard_id].trial_ids
    )


def _action_is_spent(projection: RecoveryProjection, action: ActionProjection) -> bool:
    if action.status == "failed":
        return True
    wave = projection.waves.get(f"wave-{action.action_key}")
    return wave is not None and wave.status == "closed"


def _action_shard_ids(
    lock: CampaignLock,
    kind: ActionKind,
    target_ids: list[str],
) -> list[str]:
    if kind == "submit-wave":
        return target_ids
    if kind != "retry-shard":
        return []
    requested = set(target_ids)
    return sorted(
        shard.shard_id
        for run in lock.runs
        for shard in run.shards
        if requested.intersection(trial.trial_id for trial in shard.trials)
    )


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
                action, denied = _denied_billable_action(
                    lock,
                    projection,
                    candidate,
                    reason,
                    exhaust_retry=_durable_spend_cap_blocks_retry(
                        lock, projection, candidate
                    ),
                )
                blocked.extend(denied)
                if action is None:
                    continue
            else:
                usage.admit(candidate, lock.campaign_id)
        if len(actions) >= context.limits.action_limit:
            break
        actions.append(action)
    return actions, blocked


def _denied_billable_action(
    lock: CampaignLock,
    projection: RecoveryProjection,
    candidate: _Candidate,
    reason: Literal[
        "global-budget",
        "deployment-budget",
        "provider-budget",
        "campaign-budget",
        "spend-cap",
        "spend-estimate-missing",
    ],
    *,
    exhaust_retry: bool,
) -> tuple[ReconcileAction | None, list[BlockedAction]]:
    kind = cast(Literal["submit-wave", "retry-shard"], candidate.kind)
    if kind == "retry-shard" and reason == "spend-cap" and exhaust_retry:
        exhaustion = _Candidate(
            kind="exhaust-trials",
            deployment_digest=candidate.deployment_digest,
            shard_ids=candidate.shard_ids,
            trial_ids=candidate.trial_ids,
            target_ids=candidate.trial_ids,
        )
        return _materialize(lock, projection, exhaustion), []
    return None, [
        BlockedAction(
            kind=kind,
            deployment_digest=candidate.deployment_digest,
            shard_ids=candidate.shard_ids,
            reason=reason,
        )
    ]


def _durable_spend_cap_blocks_retry(
    lock: CampaignLock,
    projection: RecoveryProjection,
    candidate: _Candidate,
) -> bool:
    caps = [
        cap
        for cap in (
            lock.recovery_policy.spend_cap_microusd,
            candidate.spend_cap_microusd,
        )
        if cap is not None
    ]
    locked_estimate = _run_admission(
        lock, candidate.deployment_digest
    ).estimated_wave_cost_microusd
    if not caps or locked_estimate is None:
        return False
    durable_spend = projection.spend_microusd + sum(
        wave.estimated_cost_microusd for wave in projection.waves.values()
    )
    return durable_spend + locked_estimate > min(caps)


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
    caps = [
        cap
        for cap in (
            lock.recovery_policy.spend_cap_microusd,
            candidate.spend_cap_microusd,
        )
        if cap is not None
    ]
    cap = min(caps) if caps else None
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
        and _action_is_spent(projection, action)
        for action in projection.campaign.actions.values()
    )
    value = candidate.model_dump(mode="json")
    value.update({"campaign_id": lock.campaign_id, "action_retry": retries})
    action_key = _short_digest(value)
    if any(
        action.action_key == action_key and not _action_is_spent(projection, action)
        for action in projection.campaign.actions.values()
    ):
        return None
    values = candidate.model_dump(mode="python")
    values["wave_id"] = candidate.wave_id or (
        f"wave-{action_key}"
        if candidate.kind in {"submit-wave", "retry-shard"}
        else None
    )
    return ReconcileAction(
        action_id=f"act-{action_key}",
        action_key=action_key,
        campaign_id=lock.campaign_id,
        **values,
    )


def _deployment(
    lock: CampaignLock,
    context: ReconcileContext,
    deployment_digest: str,
) -> DeploymentAdmission:
    locked = _run_admission(lock, deployment_digest)
    observed = context.deployments.get(deployment_digest)
    if observed is None:
        return DeploymentAdmission(
            provider=locked.provider or "hf-inference-endpoints",
            estimated_wave_cost_microusd=locked.estimated_wave_cost_microusd,
        )
    return observed.model_copy(
        update={
            "provider": observed.provider
            or locked.provider
            or "hf-inference-endpoints",
            "estimated_wave_cost_microusd": (
                observed.estimated_wave_cost_microusd
                if observed.estimated_wave_cost_microusd is not None
                else locked.estimated_wave_cost_microusd
            ),
        }
    )


def _admission_provider(admission: DeploymentAdmission) -> str:
    if admission.provider is None:
        raise ValueError("deployment admission has no provider")
    return admission.provider


def _run_admission(lock: CampaignLock, deployment_digest: str) -> CampaignRunLock:
    matches = [run for run in lock.runs if run.deployment_digest == deployment_digest]
    if not matches:
        raise ValueError("unknown deployment admission target")
    first = matches[0]
    identity = (
        first.provider,
        first.max_concurrent_requests,
        first.spend_cap_microusd,
        first.estimated_wave_cost_microusd,
    )
    if any(
        (
            run.provider,
            run.max_concurrent_requests,
            run.spend_cap_microusd,
            run.estimated_wave_cost_microusd,
        )
        != identity
        for run in matches[1:]
    ):
        raise ValueError("deployment admission fields are inconsistent")
    return first


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
