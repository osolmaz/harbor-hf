from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from harbor_hf.campaigns import (
    CampaignLock,
    CampaignRecoveryPolicy,
    build_campaign_lock,
    build_campaign_plan,
    build_wave_lock,
    deterministic_wave_id,
)
from harbor_hf.control import (
    ActionReservedPayload,
    CampaignEvent,
    CampaignSubmittedPayload,
    new_event,
)
from harbor_hf.models import ExperimentSpec
from harbor_hf.reconciler import (
    AdmissionLimits,
    AdmissionUsage,
    DeploymentAdmission,
    ReconcileAction,
    ReconcileContext,
    _action_shard_ids,
    _admission_provider,
    _assigned_shards,
    _deployment_for_shards,
    _run_admission,
    _short_digest,
    plan_reconciliation,
)

NOW = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)


def _hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _spec(remote_spec: ExperimentSpec) -> ExperimentSpec:
    tasks = {f"task-{index}": f"sha256:{index:064x}" for index in range(1, 5)}
    return remote_spec.model_copy(
        update={
            "benchmark": remote_spec.benchmark.model_copy(
                update={"task_names": ["task-*"], "task_digests": tasks}
            ),
            "execution": remote_spec.execution.model_copy(
                update={
                    "attempts": 2,
                    "max_trials_per_shard": 2,
                    "max_shards_per_wave": 2,
                    "concurrent_trials": 2,
                }
            ),
        }
    )


def _submitted(lock: CampaignLock) -> CampaignEvent:
    return new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="campaign.submitted",
        producer="cli",
        payload=CampaignSubmittedPayload(plan_digest=lock.plan_digest),
        clock=lambda: NOW,
        identifier=lambda: "1" * 32,
    )


def _campaign(remote_spec: ExperimentSpec) -> tuple[ExperimentSpec, CampaignLock]:
    spec = _spec(remote_spec)
    lock = build_campaign_lock(
        build_campaign_plan(spec), "campaign-mutation", clock=lambda: NOW
    )
    return spec, lock


def test_campaign_plan_lock_reconcile_and_wave_have_one_canonical_output(
    remote_spec: ExperimentSpec,
) -> None:
    spec = _spec(remote_spec)
    plan = build_campaign_plan(spec)
    lock = build_campaign_lock(plan, "campaign-mutation", clock=lambda: NOW)
    projection, reconciliation = plan_reconciliation(lock, [_submitted(lock)], now=NOW)
    waves = [build_wave_lock(lock, spec, action) for action in reconciliation.actions]

    corpus = {
        "plan": plan.model_dump(mode="json"),
        "lock": lock.model_dump(mode="json"),
        "projection": projection.model_dump(mode="json"),
        "reconciliation": reconciliation.model_dump(mode="json"),
        "waves": [wave.model_dump(mode="json") for wave in waves],
    }

    assert _hash(corpus) == (
        "8c4c1c2ca91762a82167593006dabc3bdc5d39bf47c2dd06529a217c4e2e9d4c"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda action, lock: action.model_copy(update={"kind": "publish-results"}),
            "wave action does not target the campaign",
        ),
        (
            lambda action, lock: action.model_copy(update={"campaign_id": "wrong"}),
            "wave action does not target the campaign",
        ),
        (
            lambda action, lock: action.model_copy(update={"shard_ids": []}),
            "wave action must contain at least one shard",
        ),
        (
            lambda action, lock: action.model_copy(
                update={"shard_ids": [action.shard_ids[0], action.shard_ids[0]]}
            ),
            "wave action shard IDs must be unique",
        ),
        (
            lambda action, lock: action.model_copy(
                update={
                    "shard_ids": [
                        f"unknown-{index}"
                        for index in range(lock.max_shards_per_wave + 1)
                    ]
                }
            ),
            "wave action exceeds the campaign shard bound",
        ),
        (
            lambda action, lock: action.model_copy(update={"action_key": "x" * 24}),
            "wave action identity does not match its immutable contents",
        ),
        (
            lambda action, lock: action.model_copy(update={"action_id": "act-wrong"}),
            "wave action identity does not match its immutable contents",
        ),
        (
            lambda action, lock: action.model_copy(
                update={"shard_ids": ["unknown-shard"]}
            ),
            "wave action references an unknown campaign shard",
        ),
        (
            lambda action, lock: action.model_copy(
                update={"deployment_digest": "sha256:" + "f" * 64}
            ),
            "wave action mixes incompatible deployment digests",
        ),
    ],
)
def test_wave_action_rejection_matrix_has_exact_contract_errors(
    remote_spec: ExperimentSpec,
    mutate: Callable[[ReconcileAction, CampaignLock], ReconcileAction],
    message: str,
) -> None:
    spec, lock = _campaign(remote_spec)
    action = plan_reconciliation(lock, [_submitted(lock)], now=NOW)[1].actions[0]

    with pytest.raises(ValueError) as captured:
        build_wave_lock(lock, spec, mutate(action, lock))

    assert str(captured.value) == message


def test_admission_matrix_records_complete_actions_and_blocked_reasons(
    remote_spec: ExperimentSpec,
) -> None:
    spec = _spec(remote_spec)
    policy = CampaignRecoveryPolicy(
        max_active_waves=3,
        max_physical_executions_per_trial=3,
        retry_base_seconds=10,
        retry_max_seconds=60,
        cancellation_grace_seconds=30,
        spend_cap_microusd=150,
    )
    lock = build_campaign_lock(
        build_campaign_plan(spec, recovery_policy=policy),
        "campaign-admission",
        clock=lambda: NOW,
    )
    digest = lock.runs[0].deployment_digest
    submitted = _submitted(lock)
    contexts = [
        ReconcileContext(
            limits=AdmissionLimits(
                action_limit=8,
                global_active_waves=8,
                deployment_active_waves=8,
                provider_active_waves=8,
                campaign_active_waves=8,
            ),
            deployments={
                digest: DeploymentAdmission(
                    provider="provider-one", estimated_wave_cost_microusd=100
                )
            },
        ),
        ReconcileContext(
            limits=AdmissionLimits(global_active_waves=1),
            usage=AdmissionUsage(global_active_waves=1),
        ),
        ReconcileContext(
            limits=AdmissionLimits(deployment_active_waves=1),
            usage=AdmissionUsage(deployment_active_waves={digest: 1}),
        ),
        ReconcileContext(
            limits=AdmissionLimits(provider_active_waves=1),
            usage=AdmissionUsage(provider_active_waves={"hf-inference-endpoints": 1}),
        ),
        ReconcileContext(
            limits=AdmissionLimits(campaign_active_waves=1),
            usage=AdmissionUsage(campaign_active_waves={lock.campaign_id: 1}),
        ),
        ReconcileContext(),
        ReconcileContext(
            usage=AdmissionUsage(campaign_spend_microusd={lock.campaign_id: 75}),
            deployments={digest: DeploymentAdmission(estimated_wave_cost_microusd=100)},
        ),
    ]

    plans = [
        plan_reconciliation(lock, [submitted], context=context, now=NOW)[1].model_dump(
            mode="json"
        )
        for context in contexts
    ]

    assert _hash(plans) == (
        "c3790fd94451bab793d783d8c3b9f9df62a912353424f373e13a3dc5ef6a8380"
    )
    assert [[item["reason"] for item in plan["blocked"]] for plan in plans] == [
        ["spend-cap"],
        ["global-budget", "global-budget"],
        ["deployment-budget", "deployment-budget"],
        ["provider-budget", "provider-budget"],
        ["campaign-budget", "campaign-budget"],
        ["spend-estimate-missing", "spend-estimate-missing"],
        ["spend-cap", "spend-cap"],
    ]


def test_campaign_input_rejection_matrix_has_exact_contract_errors(
    remote_spec: ExperimentSpec,
) -> None:
    spec = _spec(remote_spec)
    missing = spec.model_copy(
        update={"benchmark": spec.benchmark.model_copy(update={"task_digests": {}})}
    )
    unresolved_selection = spec.model_copy(
        update={
            "benchmark": spec.benchmark.model_copy(update={"task_names": ["absent"]})
        }
    )
    unresolved_digest = spec.model_copy(
        update={
            "benchmark": spec.benchmark.model_copy(
                update={
                    "task_names": ["task-1"],
                    "task_digests": {
                        "task-1": "sha256:" + "1" * 64,
                        "task-2": "sha256:" + "2" * 64,
                    },
                }
            )
        }
    )
    for candidate, message in [
        (missing, "campaign planning requires resolved task digests"),
        (
            unresolved_selection,
            "campaign task digests must exactly resolve task selections",
        ),
        (
            unresolved_digest,
            "campaign task digests must exactly resolve task selections",
        ),
    ]:
        with pytest.raises(ValueError) as captured:
            build_campaign_plan(candidate)
        assert str(captured.value) == message

    plan = build_campaign_plan(spec)
    unsafe_ids = ["", ".campaign", "campaign/path", "campaign space", "x" * 101]
    for campaign_id in unsafe_ids:
        with pytest.raises(ValueError) as captured:
            build_campaign_lock(plan, campaign_id)
        assert str(captured.value) == (
            "campaign ID must be one safe path component containing only letters, "
            "digits, dots, underscores, or hyphens, with at most 100 characters"
        )

    for action_key in ["", "0" * 23, "0" * 25, "g" * 24, "A" * 24]:
        with pytest.raises(ValueError) as captured:
            deterministic_wave_id(action_key)
        assert str(captured.value) == (
            "wave action key must be a 24-character hexadecimal digest"
        )


def test_wave_lock_rejects_tampered_campaign_with_exact_error(
    remote_spec: ExperimentSpec,
) -> None:
    spec, lock = _campaign(remote_spec)
    action = plan_reconciliation(lock, [_submitted(lock)], now=NOW)[1].actions[0]
    tampered = lock.model_copy(update={"plan_digest": "sha256:" + "f" * 64})

    with pytest.raises(ValueError) as captured:
        build_wave_lock(tampered, spec, action)

    assert str(captured.value) == "campaign lock does not match the resolved manifest"


def test_reserved_submit_action_removes_all_assigned_shards_from_candidates(
    remote_spec: ExperimentSpec,
) -> None:
    _specification, lock = _campaign(remote_spec)
    submitted = _submitted(lock)
    initial = plan_reconciliation(lock, [submitted], now=NOW)[1]
    reserved_action = initial.actions[0]
    reserved = new_event(
        subject_type="campaign",
        subject_id=lock.campaign_id,
        kind="action.reserved",
        producer="reconciler",
        payload=ActionReservedPayload(
            action_id=reserved_action.action_id,
            action_key=reserved_action.action_key,
            action_kind=reserved_action.kind,
            target_ids=reserved_action.shard_ids,
        ),
        clock=lambda: NOW,
        identifier=lambda: "2" * 32,
    )

    projection, subsequent = plan_reconciliation(lock, [submitted, reserved], now=NOW)
    corpus = {
        "reserved": projection.campaign.actions[reserved_action.action_id].model_dump(
            mode="json"
        ),
        "initial": initial.model_dump(mode="json"),
        "subsequent": subsequent.model_dump(mode="json"),
    }

    assert _hash(corpus) == (
        "a22e2bfd592f15502ecb2403c15266bc83d987c27d8509ecc10d7cfb4ef1596c"
    )
    assert _assigned_shards(lock, projection) == set(reserved_action.shard_ids)
    assert {
        shard_id for action in subsequent.actions for shard_id in action.shard_ids
    }.isdisjoint(reserved_action.shard_ids)


def test_reconciler_identity_helpers_preserve_exact_campaign_relationships(
    remote_spec: ExperimentSpec,
) -> None:
    _specification, lock = _campaign(remote_spec)
    action = plan_reconciliation(lock, [_submitted(lock)], now=NOW)[1].actions[0]
    run = lock.runs[0]
    shard = run.shards[0]
    trial_ids = [trial.trial_id for trial in shard.trials]

    assert _action_shard_ids(lock, "submit-wave", action.shard_ids) == action.shard_ids
    assert _action_shard_ids(lock, "retry-shard", trial_ids) == [shard.shard_id]
    assert _action_shard_ids(lock, "cleanup-wave", trial_ids) == []
    assert _deployment_for_shards(lock, [shard.shard_id]) == run.deployment_digest
    assert _run_admission(lock, run.deployment_digest) == run
    assert _admission_provider(DeploymentAdmission(provider="provider-contract")) == (
        "provider-contract"
    )
    assert _short_digest({"b": 2, "a": 1}) == "43258cff783fe7036d8a4303"
    assert _short_digest({"é": "雪"}) == "7e406c0769035fe3024bc9f1"


def test_reconciler_identity_helpers_reject_unknown_targets_with_exact_errors(
    remote_spec: ExperimentSpec,
) -> None:
    _specification, lock = _campaign(remote_spec)

    with pytest.raises(ValueError) as admission_error:
        _run_admission(lock, "sha256:" + "f" * 64)
    assert str(admission_error.value) == "unknown deployment admission target"

    with pytest.raises(ValueError) as shard_error:
        _deployment_for_shards(lock, ["unknown-shard"])
    assert str(shard_error.value) == "action shards must belong to one deployment"

    with pytest.raises(ValueError) as provider_error:
        _admission_provider(DeploymentAdmission())
    assert str(provider_error.value) == "deployment admission has no provider"

    run = lock.runs[0]
    inconsistent = lock.model_copy(
        update={
            "runs": [
                run,
                run.model_copy(update={"max_concurrent_requests": 99}),
            ]
        }
    )
    with pytest.raises(ValueError) as inconsistency_error:
        _run_admission(inconsistent, run.deployment_digest)
    assert str(inconsistency_error.value) == (
        "deployment admission fields are inconsistent"
    )
