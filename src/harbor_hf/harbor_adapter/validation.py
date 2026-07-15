from __future__ import annotations

import math
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path

from pydantic import ValidationError

from harbor_hf.harbor_adapter.errors import (
    HarborTrialFailure,
    HarborVerificationFailure,
)
from harbor_hf.harbor_adapter.models import (
    HarborCompatibilityBundle,
    HarborExecutionRequest,
    HarborStepException,
    HarborVerificationResult,
    HarborVerifiedTrial,
    canonical_json_bytes,
    sha256_digest,
)


def load_compatibility_bundle(
    path: Path, request: HarborExecutionRequest
) -> HarborCompatibilityBundle:
    try:
        bundle = HarborCompatibilityBundle.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise HarborVerificationFailure(
            "Harbor compatibility exporter produced no valid bundle"
        ) from error
    if bundle.harbor_revision != request.harbor_revision:
        raise HarborVerificationFailure(
            "Harbor compatibility bundle revision does not match the request"
        )
    request_digest = sha256_digest(
        canonical_json_bytes(request.model_dump(mode="json"))
    )
    if bundle.request_digest != request_digest:
        raise HarborVerificationFailure(
            "Harbor compatibility bundle request digest does not match"
        )
    return bundle


def validate_compatibility_bundle(
    bundle: HarborCompatibilityBundle, request: HarborExecutionRequest
) -> HarborVerificationResult:
    policy = request.verification
    _validate_trial_count(len(bundle.trials), policy.expected_trials)
    resolved_names = [
        _resolved_task_name(trial.task_name, trial.task_digest, request)
        for trial in bundle.trials
    ]
    observed = Counter(resolved_names)
    _validate_task_counts(observed, request)
    verified: list[HarborVerifiedTrial] = []
    for trial, task_name in zip(bundle.trials, resolved_names, strict=True):
        failure = _trial_failure(
            trial.exception_type, trial.exception_message, trial.step_exceptions
        )
        if failure is not None:
            location, exception_type, exception_message = failure
            raise HarborTrialFailure(
                f"Harbor trial {trial.task_name}{location} failed with "
                f"{exception_type}",
                exception_type,
                exception_message,
            )
        if (
            trial.agent_name != policy.expected_agent_name
            or trial.agent_version != policy.expected_agent_version
        ):
            raise HarborVerificationFailure(
                f"Harbor trial {trial.task_name} agent identity does not match the lock"
            )
        if (
            trial.model_provider != policy.expected_model_provider
            or trial.model_name != policy.expected_model_name
        ):
            raise HarborVerificationFailure(
                f"Harbor trial {trial.task_name} model identity does not match the lock"
            )
        rewards = trial.rewards
        if not rewards:
            raise HarborVerificationFailure(
                f"Harbor trial {trial.task_name} has no verifier rewards"
            )
        if not all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
            for value in rewards.values()
        ):
            raise HarborVerificationFailure(
                f"Harbor trial {trial.task_name} rewards must be finite numbers"
            )
        verified.append(HarborVerifiedTrial(task_name=task_name, rewards=dict(rewards)))
    return HarborVerificationResult(trial_count=len(verified), trials=verified)


def _resolved_task_name(
    observed_name: str, observed_digest: str, request: HarborExecutionRequest
) -> str:
    expected = request.verification.expected_task_digests or {}
    if expected.get(observed_name) == observed_digest:
        return observed_name
    matches = [name for name, digest in expected.items() if digest == observed_digest]
    if len(matches) != 1:
        raise HarborVerificationFailure(
            f"Harbor trial {observed_name} task digest does not match the lock"
        )
    return matches[0]


def _validate_trial_count(observed: int, expected: int | None) -> None:
    if expected is None and observed == 0:
        raise HarborVerificationFailure("Harbor produced no trials")
    if expected is not None and observed != expected:
        raise HarborVerificationFailure(
            f"expected exactly {expected} Harbor trials, found {observed}"
        )


def _validate_task_counts(
    observed: Counter[str], request: HarborExecutionRequest
) -> None:
    policy = request.verification
    expected = policy.expected_task_counts or {}
    valid = all(observed[task] == count for task, count in expected.items())
    if policy.expected_attempts_per_task is not None:
        valid = valid and all(
            count == policy.expected_attempts_per_task for count in observed.values()
        )
    if policy.expected_task_names is not None:
        valid = valid and all(
            any(fnmatch(task, requested) for requested in policy.expected_task_names)
            for task in observed
        )
    if not valid:
        raise HarborVerificationFailure(
            "Harbor trial task counts do not match the requested attempts"
        )


def _trial_failure(
    exception_type: str | None,
    exception_message: str | None,
    step_exceptions: list[HarborStepException],
) -> tuple[str, str, str | None] | None:
    if exception_type is not None:
        return "", exception_type, exception_message
    if step_exceptions:
        step = step_exceptions[0]
        name = step.step_name
        kind = step.exception_type
        return f" step {name}", kind, step.exception_message
    return None
