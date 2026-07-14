from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from fnmatch import fnmatch
from pathlib import Path

from harbor_hf.harbor_adapter.errors import HarborTrialFailure, WorkerError


def validate_harbor_result(
    jobs_dir: Path,
    expected_trials: int | None = 1,
    *,
    expected_task_counts: Mapping[str, int] | None = None,
    expected_attempts_per_task: int | None = None,
    expected_task_names: Sequence[str] | None = None,
    expected_task_digests: Mapping[str, str] | None = None,
    expected_agent_name: str | None = None,
    expected_agent_version: str | None = None,
    expected_model_provider: str | None = None,
    expected_model_name: str | None = None,
) -> dict[str, object]:
    """Read historical pre-adapter evidence.

    New executions use the typed compatibility exporter instead.
    """
    trials: list[dict[str, object]] = []
    for path in sorted(jobs_dir.glob("*/*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("task_name"), str)
            or not value["task_name"].strip()
        ):
            raise WorkerError("Harbor produced a malformed trial result")
        _validate_trial_task_digest(path, value["task_name"], expected_task_digests)
        trials.append(value)
    validate_trial_count(trials, expected_trials)
    validate_task_counts(
        trials,
        expected_task_counts,
        expected_attempts_per_task,
        expected_task_names,
    )
    verified: list[dict[str, object]] = []
    for trial in trials:
        task_name = str(trial["task_name"])
        failure = _trial_failure(trial)
        if failure is not None:
            location, exception_type = failure
            rendered = str(exception_type or "an exception")
            raise HarborTrialFailure(
                f"Harbor trial {task_name}{location} failed with {rendered}",
                rendered,
            )
        _validate_agent_identity(
            trial,
            task_name,
            expected_agent_name,
            expected_agent_version,
            expected_model_provider,
            expected_model_name,
        )
        verifier = trial.get("verifier_result")
        rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
        if not isinstance(rewards, Mapping) or not rewards:
            raise WorkerError(f"Harbor trial {task_name} has no verifier rewards")
        if not all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
            for value in rewards.values()
        ):
            raise WorkerError(
                f"Harbor trial {task_name} rewards must be finite numbers"
            )
        verified.append({"task_name": task_name, "rewards": dict(rewards)})
    return {"trial_count": len(verified), "trials": verified}


def _validate_trial_task_digest(
    result_path: Path,
    task_name: str,
    expected: Mapping[str, str] | None,
) -> None:
    if expected is None:
        return
    expected_digest = expected.get(task_name)
    if expected_digest is None:
        raise WorkerError(f"Harbor trial {task_name} is not in the resolved task set")
    lock_path = result_path.with_name("lock.json")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerError(f"Harbor trial {task_name} has no valid task lock") from error
    task = lock.get("task") if isinstance(lock, Mapping) else None
    if not isinstance(task, Mapping) or task.get("digest") != expected_digest:
        raise WorkerError(
            f"Harbor trial {task_name} task digest does not match the lock"
        )


def validate_task_counts(
    trials: list[dict[str, object]],
    expected: Mapping[str, int] | None,
    attempts_per_observed_task: int | None = None,
    expected_task_names: Sequence[str] | None = None,
) -> None:
    observed = Counter(str(trial["task_name"]) for trial in trials)
    valid = all(observed[task] == count for task, count in (expected or {}).items())
    if attempts_per_observed_task is not None:
        valid = valid and all(
            count == attempts_per_observed_task for count in observed.values()
        )
    if expected_task_names is not None:
        valid = valid and all(
            any(fnmatch(task, requested) for requested in expected_task_names)
            for task in observed
        )
    if not valid:
        raise WorkerError(
            "Harbor trial task counts do not match the requested attempts"
        )


def _trial_failure(trial: Mapping[str, object]) -> tuple[str, object] | None:
    exception = trial.get("exception_info")
    if exception is not None:
        exception_type = (
            exception.get("exception_type")
            if isinstance(exception, Mapping)
            else type(exception).__name__
        )
        return "", exception_type
    steps = trial.get("step_results")
    if steps is None:
        return None
    if not isinstance(steps, list):
        return " step results", "malformed result"
    for ordinal, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            return f" step {ordinal}", "malformed result"
        step_exception = step.get("exception_info")
        if step_exception is None:
            continue
        exception_type = (
            step_exception.get("exception_type")
            if isinstance(step_exception, Mapping)
            else type(step_exception).__name__
        )
        step_name = step.get("step_name") or ordinal
        return f" step {step_name}", exception_type
    return None


def _validate_agent_identity(
    trial: Mapping[str, object],
    task_name: str,
    expected_name: str | None,
    expected_version: str | None,
    expected_model_provider: str | None,
    expected_model_name: str | None,
) -> None:
    if all(
        value is None
        for value in (
            expected_name,
            expected_version,
            expected_model_provider,
            expected_model_name,
        )
    ):
        return
    agent = trial.get("agent_info")
    if not isinstance(agent, Mapping):
        raise WorkerError(f"Harbor trial {task_name} has no agent identity")
    if agent.get("name") != expected_name or agent.get("version") != expected_version:
        raise WorkerError(
            f"Harbor trial {task_name} agent identity does not match the lock"
        )
    if expected_model_provider is not None or expected_model_name is not None:
        model = agent.get("model_info")
        if not isinstance(model, Mapping):
            raise WorkerError(f"Harbor trial {task_name} has no model identity")
        if (
            model.get("provider") != expected_model_provider
            or model.get("name") != expected_model_name
        ):
            raise WorkerError(
                f"Harbor trial {task_name} model identity does not match the lock"
            )


def validate_trial_count(
    trials: list[dict[str, object]], expected_trials: int | None
) -> None:
    if expected_trials is None and not trials:
        raise WorkerError("Harbor produced no trials")
    if expected_trials is not None and len(trials) != expected_trials:
        raise WorkerError(
            f"expected exactly {expected_trials} Harbor trials, found {len(trials)}"
        )
