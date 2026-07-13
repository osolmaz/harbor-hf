from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict

from harbor_hf.campaigns import CampaignLock
from harbor_hf.control import CampaignEvent, CampaignProjection, project_campaign


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReconcileAction(FrozenModel):
    action_id: str
    action_key: str
    kind: Literal["submit-wave"]
    campaign_id: str
    deployment_digest: str
    shard_ids: list[str]


class ReconcilePlan(FrozenModel):
    campaign_id: str
    status: str
    action_count: int
    actions: list[ReconcileAction]


def plan_reconciliation(
    lock: CampaignLock, events: list[CampaignEvent]
) -> tuple[CampaignProjection, ReconcilePlan]:
    projection = project_campaign(lock, events)
    if projection.status in {
        "cancel_requested",
        "completed",
        "partial",
        "failed",
        "cancelled",
    }:
        return projection, ReconcilePlan(
            campaign_id=lock.campaign_id,
            status=projection.status,
            action_count=0,
            actions=[],
        )

    existing_keys = {action.action_key for action in projection.actions.values()}
    groups: dict[str, list[str]] = defaultdict(list)
    for run in sorted(lock.runs, key=lambda value: value.run_id):
        for shard in sorted(run.shards, key=lambda value: value.shard_id):
            groups[run.deployment_digest].append(shard.shard_id)

    actions = []
    for deployment_digest in sorted(groups):
        shard_ids = groups[deployment_digest]
        for offset in range(0, len(shard_ids), lock.max_shards_per_wave):
            chunk = shard_ids[offset : offset + lock.max_shards_per_wave]
            action_key = _short_digest(
                {
                    "kind": "submit-wave",
                    "campaign_id": lock.campaign_id,
                    "deployment_digest": deployment_digest,
                    "shard_ids": chunk,
                }
            )
            if action_key in existing_keys:
                continue
            actions.append(
                ReconcileAction(
                    action_id=f"act-{action_key}",
                    action_key=action_key,
                    kind="submit-wave",
                    campaign_id=lock.campaign_id,
                    deployment_digest=deployment_digest,
                    shard_ids=chunk,
                )
            )
    return projection, ReconcilePlan(
        campaign_id=lock.campaign_id,
        status=projection.status,
        action_count=len(actions),
        actions=actions,
    )


def _short_digest(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest()[:24]
