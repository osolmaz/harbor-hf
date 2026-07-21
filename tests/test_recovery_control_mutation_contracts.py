from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from harbor_hf.campaigns import CampaignLock, build_campaign_lock, build_campaign_plan
from harbor_hf.control import (
    ActionOutcomePayload,
    ActionReservedPayload,
    CampaignEvent,
    CampaignSubmittedPayload,
    CancellationPayload,
    ControlError,
    EventKind,
    EventPayload,
    ExecutionOutcomePayload,
    ExecutionStartedPayload,
    LifecyclePayload,
    RetryCategory,
    SubjectType,
    TerminalPayload,
    WaveLifecyclePayload,
    new_event,
    ordered_events,
    project_campaign,
)
from harbor_hf.models import ExperimentSpec
from harbor_hf.reconciler import plan_reconciliation
from harbor_hf.recovery import project_recovery, retry_delay_seconds

NOW = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)


def _hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _lock(remote_spec: ExperimentSpec, *, tasks: int = 1) -> CampaignLock:
    task_digests = {
        f"task-{index}": f"sha256:{index:064x}" for index in range(1, tasks + 1)
    }
    spec = remote_spec.model_copy(
        update={
            "benchmark": remote_spec.benchmark.model_copy(
                update={"task_names": ["task-*"], "task_digests": task_digests}
            ),
            "execution": remote_spec.execution.model_copy(
                update={"max_trials_per_shard": max(tasks, 1)}
            ),
        }
    )
    return build_campaign_lock(
        build_campaign_plan(spec), "campaign-recovery-mutation", clock=lambda: NOW
    )


def _event(
    lock: CampaignLock,
    sequence: int,
    subject_type: SubjectType,
    subject_id: str,
    kind: EventKind,
    payload: EventPayload,
) -> CampaignEvent:
    return new_event(
        subject_type=subject_type,
        subject_id=subject_id,
        kind=kind,
        producer="reconciler",
        payload=payload,
        clock=lambda: NOW + timedelta(seconds=sequence),
        identifier=lambda: f"{sequence:032x}",
    )


def _submitted(lock: CampaignLock) -> CampaignEvent:
    return _event(
        lock,
        1,
        "campaign",
        lock.campaign_id,
        "campaign.submitted",
        CampaignSubmittedPayload(plan_digest=lock.plan_digest),
    )


def _trial_event(
    lock: CampaignLock, sequence: int, trial_index: int, kind: EventKind
) -> CampaignEvent:
    shard = lock.runs[0].shards[0]
    return _event(
        lock,
        sequence,
        "trial",
        shard.trials[trial_index].trial_id,
        kind,
        LifecyclePayload(parent_id=shard.shard_id),
    )


def _execution_started(
    lock: CampaignLock,
    sequence: int,
    *,
    execution_id: str = "execution-one",
    attempt: int = 1,
) -> CampaignEvent:
    shard = lock.runs[0].shards[0]
    trial = shard.trials[0]
    return _event(
        lock,
        sequence,
        "execution",
        execution_id,
        "execution.started",
        ExecutionStartedPayload(
            trial_id=trial.trial_id,
            shard_id=shard.shard_id,
            physical_attempt=attempt,
            wave_id="wave-one",
        ),
    )


def _execution_outcome(
    lock: CampaignLock,
    sequence: int,
    *,
    execution_id: str = "execution-one",
    kind: EventKind = "execution.completed",
    attempt: int = 1,
    category: str | None = None,
) -> CampaignEvent:
    trial = lock.runs[0].shards[0].trials[0]
    return _event(
        lock,
        sequence,
        "execution",
        execution_id,
        kind,
        ExecutionOutcomePayload(
            trial_id=trial.trial_id,
            physical_attempt=attempt,
            category=cast(RetryCategory | None, category),
        ),
    )


def _wave_event(
    lock: CampaignLock,
    sequence: int,
    kind: EventKind,
    *,
    provider: str = "provider-one",
) -> CampaignEvent:
    run = lock.runs[0]
    return _event(
        lock,
        sequence,
        "wave",
        "wave-one",
        kind,
        WaveLifecyclePayload(
            deployment_digest=run.deployment_digest,
            provider=provider,
            shard_ids=[run.shards[0].shard_id],
            estimated_cost_microusd=123,
        ),
    )


def test_recovery_terminal_decision_matrix_has_complete_canonical_structures(
    remote_spec: ExperimentSpec,
) -> None:
    two = _lock(remote_spec, tasks=2)
    cases = [
        (_lock(remote_spec), []),
        (_lock(remote_spec), ["trial.complete"]),
        (_lock(remote_spec), ["trial.invalid"]),
        (
            _lock(remote_spec),
            ["campaign.cancel-requested", "trial.cancelled"],
        ),
        (two, ["trial.complete", "trial.invalid"]),
    ]
    corpus: list[object] = []
    for lock, kinds in cases:
        events = [_submitted(lock)]
        trial_index = 0
        for sequence, kind in enumerate(kinds, 2):
            if kind == "campaign.cancel-requested":
                events.append(
                    _event(
                        lock,
                        sequence,
                        "campaign",
                        lock.campaign_id,
                        "campaign.cancel-requested",
                        CancellationPayload(reason="operator"),
                    )
                )
            else:
                events.append(
                    _trial_event(lock, sequence, trial_index, cast(EventKind, kind))
                )
                trial_index += 1
        projection, plan = plan_reconciliation(lock, events, now=NOW)
        corpus.append(
            {
                "projection": projection.model_dump(mode="json"),
                "plan": plan.model_dump(mode="json"),
            }
        )

    assert _hash(corpus) == (
        "b12ae01241b8504b76313365646a84fe2a68bc015e80f7c1323dc2621fad2346"
    )


def test_retry_delay_matrix_pins_backoff_jitter_and_bounds(
    remote_spec: ExperimentSpec,
) -> None:
    lock = _lock(remote_spec)
    values = [
        retry_delay_seconds(
            lock,
            cast(RetryCategory, category),
            attempt,
            f"execution-{category}-{attempt}",
            retry_after,
        )
        for category in ["lost", "transient", "quota", "rate-limit", "ambiguous"]
        for attempt in [1, 2, 3, 31]
        for retry_after in [None, 1, 59, 999]
    ]

    assert values == [
        35,
        35,
        59,
        999,
        51,
        51,
        59,
        999,
        115,
        115,
        115,
        999,
        1800,
        1800,
        1800,
        1800,
        28,
        28,
        59,
        999,
        50,
        50,
        59,
        999,
        110,
        110,
        110,
        999,
        1800,
        1800,
        1800,
        1800,
        51,
        51,
        59,
        999,
        140,
        140,
        140,
        999,
        280,
        280,
        280,
        999,
        1800,
        1800,
        1800,
        1800,
        75,
        75,
        75,
        999,
        146,
        146,
        146,
        999,
        213,
        213,
        213,
        999,
        1800,
        1800,
        1800,
        1800,
        25,
        25,
        59,
        999,
        57,
        57,
        59,
        999,
        105,
        105,
        105,
        999,
        1800,
        1800,
        1800,
        1800,
    ]


def _identity_history(lock: CampaignLock, case: str) -> list[CampaignEvent]:
    run = lock.runs[0]
    trial = run.shards[0].trials[0]
    events = [_submitted(lock)]
    if case == "unknown-run":
        events.append(
            _event(lock, 2, "run", "missing", "run.active", LifecyclePayload())
        )
    elif case == "unknown-shard":
        events.append(
            _event(lock, 2, "shard", "missing", "shard.active", LifecyclePayload())
        )
    elif case == "unknown-trial":
        events.append(
            _event(lock, 2, "trial", "missing", "trial.complete", LifecyclePayload())
        )
    elif case == "bad-start":
        events.append(
            _execution_started(lock, 2).model_copy(
                update={
                    "payload": ExecutionStartedPayload(
                        trial_id=trial.trial_id,
                        shard_id="missing",
                        physical_attempt=1,
                        wave_id="wave-one",
                    )
                }
            )
        )
    elif case == "duplicate-start":
        events.extend([_execution_started(lock, 2), _execution_started(lock, 3)])
    elif case == "duplicate-attempt":
        events.extend(
            [
                _execution_started(lock, 2),
                _execution_started(lock, 3, execution_id="execution-two"),
            ]
        )
    return events


def _outcome_wave_history(lock: CampaignLock, case: str) -> list[CampaignEvent]:
    run = lock.runs[0]
    events = [_submitted(lock)]
    if case == "outcome-no-start":
        events.append(_execution_outcome(lock, 2))
    elif case == "multiple-outcomes":
        events.extend(
            [
                _execution_started(lock, 2),
                _execution_outcome(lock, 3),
                _execution_outcome(lock, 4),
            ]
        )
    elif case == "outcome-mismatch":
        events.extend(
            [
                _execution_started(lock, 2),
                _execution_outcome(lock, 3, attempt=2),
            ]
        )
    elif case == "completed-category":
        events.extend(
            [
                _execution_started(lock, 2),
                _execution_outcome(lock, 3, category="transient"),
            ]
        )
    elif case == "failed-no-category":
        events.extend(
            [
                _execution_started(lock, 2),
                _execution_outcome(lock, 3, kind="execution.failed"),
            ]
        )
    elif case == "wave-unknown":
        events.append(
            _wave_event(lock, 2, "wave.acquiring").model_copy(
                update={
                    "payload": WaveLifecyclePayload(
                        deployment_digest=run.deployment_digest,
                        provider="provider-one",
                        shard_ids=["missing"],
                        estimated_cost_microusd=123,
                    )
                }
            )
        )
    elif case == "wave-identity":
        events.extend(
            [
                _wave_event(lock, 2, "wave.acquiring"),
                _wave_event(lock, 3, "wave.provisioning", provider="provider-two"),
            ]
        )
    else:
        events.extend(
            [
                _wave_event(lock, 2, "wave.acquiring"),
                _wave_event(lock, 3, "wave.active"),
            ]
        )
    return events


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown-run", "event references unknown run: missing"),
        ("unknown-shard", "event references unknown shard: missing"),
        ("unknown-trial", "event references unknown trial: missing"),
        ("bad-start", "execution references an unknown trial or shard"),
        ("duplicate-start", "execution started more than once: execution-one"),
        ("duplicate-attempt", "trial has duplicate physical execution numbers"),
        ("outcome-no-start", "execution outcome has no start: execution-one"),
        ("multiple-outcomes", "execution has multiple outcomes: execution-one"),
        ("outcome-mismatch", "execution outcome identity does not match its start"),
        ("completed-category", "completed execution cannot have a failure category"),
        ("failed-no-category", "failed execution requires a failure category"),
        ("wave-unknown", "wave references unknown shards: missing"),
        ("wave-identity", "wave lifecycle identity changed"),
        ("wave-transition", "invalid wave transition: acquiring -> active"),
    ],
)
def test_recovery_history_rejection_matrix_has_exact_errors(
    remote_spec: ExperimentSpec, case: str, message: str
) -> None:
    lock = _lock(remote_spec)
    identity_cases = {
        "unknown-run",
        "unknown-shard",
        "unknown-trial",
        "bad-start",
        "duplicate-start",
        "duplicate-attempt",
    }
    events = (
        _identity_history(lock, case)
        if case in identity_cases
        else _outcome_wave_history(lock, case)
    )

    with pytest.raises(ValueError) as captured:
        project_recovery(lock, events)

    assert str(captured.value) == message


def test_control_action_projection_and_rejections_use_complete_values(
    remote_spec: ExperimentSpec,
) -> None:
    lock = _lock(remote_spec)
    submitted = _submitted(lock)
    events = [submitted]
    for index, outcome in enumerate(
        ["action.succeeded", "action.failed", "action.ambiguous"], 2
    ):
        action_id = f"action-{index}"
        events.extend(
            [
                _event(
                    lock,
                    index,
                    "campaign",
                    lock.campaign_id,
                    "action.reserved",
                    ActionReservedPayload(
                        action_id=action_id,
                        action_key=f"key-{index}",
                        action_kind="submit-wave",
                        target_ids=[f"target-{index}"],
                    ),
                ),
                _event(
                    lock,
                    index + 10,
                    "campaign",
                    lock.campaign_id,
                    cast(EventKind, outcome),
                    ActionOutcomePayload(
                        action_id=action_id,
                        message=f"message-{index}",
                        remote_id=f"remote-{index}",
                    ),
                ),
            ]
        )

    projection = project_campaign(lock, list(reversed(events)))

    assert _hash(projection.model_dump(mode="json")) == (
        "2e1224de01c73898fddfcc038ce848d5e1ad68dfbacab8fda2d8385422b81161"
    )

    conflicting = submitted.model_copy(update={"subject_id": "wrong"})
    with pytest.raises(ControlError) as captured:
        project_campaign(lock, [conflicting])
    assert str(captured.value) == ("campaign submission event does not match its lock")


def test_run_and_shard_event_projection_matrix_is_canonical(
    remote_spec: ExperimentSpec,
) -> None:
    corpus: list[object] = []
    for index, kind in enumerate(
        [
            "run.queued",
            "run.active",
            "run.verifying",
            "run.publishing",
            "shard.queued",
            "shard.active",
            "shard.verifying",
            "shard.publishing",
        ],
        2,
    ):
        lock = _lock(remote_spec)
        subject_type: SubjectType = "run" if kind.startswith("run.") else "shard"
        subject_id = (
            lock.runs[0].run_id
            if subject_type == "run"
            else lock.runs[0].shards[0].shard_id
        )
        event = _event(
            lock,
            index,
            subject_type,
            subject_id,
            cast(EventKind, kind),
            LifecyclePayload(),
        )
        corpus.append(
            project_recovery(lock, [_submitted(lock), event]).model_dump(mode="json")
        )

    assert _hash(corpus) == (
        "c90725d45ac30e7b171f33df3daaaa9d65732fa5c7be15d30d0a5aec2c297467"
    )


@pytest.mark.parametrize(
    ("subject", "kind", "message"),
    [
        ("run", "run.complete", "run completed with non-complete children"),
        ("shard", "shard.complete", "shard completed with non-complete children"),
        (
            "run",
            "run.failed-infrastructure",
            "run became terminal before its children",
        ),
        (
            "shard",
            "shard.failed-infrastructure",
            "shard became terminal before its children",
        ),
    ],
)
def test_observed_terminal_parent_rejection_matrix_has_exact_errors(
    remote_spec: ExperimentSpec, subject: str, kind: EventKind, message: str
) -> None:
    lock = _lock(remote_spec)
    subject_id = (
        lock.runs[0].run_id if subject == "run" else lock.runs[0].shards[0].shard_id
    )
    sequence = 3 if kind.endswith(".complete") else 2
    event = _event(
        lock,
        sequence,
        cast(SubjectType, subject),
        subject_id,
        kind,
        LifecyclePayload(),
    )

    events = [_submitted(lock)]
    if kind.endswith(".complete"):
        events.append(_trial_event(lock, 2, 0, "trial.invalid"))
    events.append(event)

    with pytest.raises(ValueError) as captured:
        project_recovery(lock, events)

    assert str(captured.value) == message


def test_control_terminal_and_subject_boundary_matrix_is_complete(
    remote_spec: ExperimentSpec,
) -> None:
    lock = _lock(remote_spec)
    submitted = _submitted(lock)
    terminal_projections = []
    for index, kind in enumerate(
        [
            "campaign.completed",
            "campaign.partial",
            "campaign.failed",
            "campaign.cancelled",
        ],
        2,
    ):
        terminal = _event(
            lock,
            index,
            "campaign",
            lock.campaign_id,
            cast(EventKind, kind),
            TerminalPayload(message="terminal"),
        )
        terminal_projections.append(
            project_campaign(lock, [submitted, terminal]).model_dump(mode="json")
        )
        late = _event(
            lock,
            index + 10,
            "campaign",
            lock.campaign_id,
            "campaign.draining",
            LifecyclePayload(),
        )
        with pytest.raises(ControlError) as captured:
            project_campaign(lock, [submitted, terminal, late])
        assert str(captured.value) == "campaign has events after a terminal transition"

    assert _hash(terminal_projections) == (
        "3b6e98d6963cfafed6c21a79ce534a191db95a8f2e6a5ce2393c79a748458915"
    )

    wrong_type = _event(
        lock,
        2,
        "campaign",
        lock.campaign_id,
        "campaign.draining",
        LifecyclePayload(),
    ).model_copy(update={"subject_type": "run"})
    wrong_id = wrong_type.model_copy(
        update={"subject_type": "campaign", "subject_id": "wrong"}
    )
    assert project_campaign(lock, [submitted, wrong_type]).status == "queued"
    duplicate = submitted.model_copy(
        update={"event_id": "evt-" + "2" * 32, "observed_at": NOW + timedelta(1)}
    )
    for event, message in [
        (wrong_id, "campaign event has the wrong subject"),
        (duplicate, "campaign has multiple submission events"),
    ]:
        with pytest.raises(ControlError) as captured:
            project_campaign(lock, [submitted, event])
        assert str(captured.value) == message


def test_action_projection_rejection_matrix_has_exact_errors(
    remote_spec: ExperimentSpec,
) -> None:
    lock = _lock(remote_spec)
    submitted = _submitted(lock)
    reserved = _event(
        lock,
        2,
        "campaign",
        lock.campaign_id,
        "action.reserved",
        ActionReservedPayload(
            action_id="action-one",
            action_key="key-one",
            action_kind="submit-wave",
            target_ids=["shard-one"],
        ),
    )
    outcome = _event(
        lock,
        3,
        "campaign",
        lock.campaign_id,
        "action.succeeded",
        ActionOutcomePayload(action_id="action-one", remote_id="remote-one"),
    )
    cases = [
        (
            [
                submitted,
                reserved,
                reserved.model_copy(update={"event_id": "evt-duplicate"}),
            ],
            "action was reserved more than once",
        ),
        (
            [submitted, outcome],
            "action outcome has no reservation",
        ),
        (
            [
                submitted,
                reserved,
                outcome,
                outcome.model_copy(
                    update={
                        "event_id": "evt-second-outcome",
                        "kind": "action.failed",
                    }
                ),
            ],
            "action has multiple outcomes",
        ),
    ]

    for events, message in cases:
        with pytest.raises(ControlError) as captured:
            project_campaign(lock, events)
        assert str(captured.value) == message


def test_event_ordering_deduplication_and_conflicts_are_complete(
    remote_spec: ExperimentSpec,
) -> None:
    lock = _lock(remote_spec)
    first = _submitted(lock).model_copy(
        update={"event_id": "evt-b", "observed_at": NOW}
    )
    second = _event(
        lock,
        2,
        "campaign",
        lock.campaign_id,
        "campaign.draining",
        LifecyclePayload(),
    ).model_copy(update={"event_id": "evt-a", "observed_at": NOW})

    ordered = ordered_events([first, second, first, second])
    assert [event.event_id for event in ordered] == ["evt-a", "evt-b"]
    assert ordered == [second, first]

    conflicting = first.model_copy(update={"producer": "different-producer"})
    with pytest.raises(ControlError) as captured:
        ordered_events([first, conflicting])
    assert str(captured.value) == "event ID has conflicting records"


def test_non_campaign_events_do_not_hide_later_campaign_transitions(
    remote_spec: ExperimentSpec,
) -> None:
    lock = _lock(remote_spec)
    ignored = _event(
        lock,
        2,
        "run",
        lock.runs[0].run_id,
        "run.active",
        LifecyclePayload(),
    )
    draining = _event(
        lock,
        3,
        "campaign",
        lock.campaign_id,
        "campaign.draining",
        LifecyclePayload(),
    )
    projection = project_campaign(lock, [_submitted(lock), ignored, draining])

    assert projection.status == "draining"
    assert projection.event_count == 3
    assert projection.last_observed_at == NOW + timedelta(seconds=3)


def test_repeated_cancellation_preserves_draining_and_manual_states(
    remote_spec: ExperimentSpec,
) -> None:
    lock = _lock(remote_spec)
    submitted = _submitted(lock)
    cancel = _event(
        lock,
        4,
        "campaign",
        lock.campaign_id,
        "campaign.cancel-requested",
        CancellationPayload(reason="operator"),
    )
    states = []
    for kind in [None, "campaign.draining", "campaign.manual-intervention-required"]:
        events = [submitted]
        if kind is not None:
            events.append(
                _event(
                    lock,
                    2,
                    "campaign",
                    lock.campaign_id,
                    cast(EventKind, kind),
                    LifecyclePayload(),
                )
            )
        events.extend(
            [cancel, cancel.model_copy(update={"event_id": "evt-cancel-two"})]
        )
        states.append(project_campaign(lock, events).model_dump(mode="json"))

    assert _hash(states) == (
        "58063632e2ddbe6852ba62689c0358ca5c275a880936f5530a128676acf78d2b"
    )
