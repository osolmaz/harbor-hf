from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

ArtifactKind = Literal[
    "agent_log",
    "configuration",
    "execution_log",
    "lock",
    "other",
    "result",
    "runtime",
    "session",
    "trajectory",
    "verifier",
]


class TimingValue(Protocol):
    started_at: datetime | None
    finished_at: datetime | None


UsageTotals = tuple[int | None, int | None, int | None, float | None]


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _relative(path: Path, root: Path) -> str:
    value = path.relative_to(root).as_posix()
    if not value or value.startswith("/") or ".." in Path(value).parts:
        raise ValueError("Harbor artifact path is not safely relative")
    return value


def classify_private_artifact(path: str) -> ArtifactKind:
    relative = Path(path)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != path
    ):
        raise ValueError("Harbor artifact path is not safely relative")
    parts = tuple(part.lower() for part in relative.parts)
    name = parts[-1]
    if "trajectory" in name:
        return "trajectory"
    component_kinds: tuple[tuple[str, ArtifactKind], ...] = (
        ("openclaw-sessions", "session"),
        ("trajectories", "trajectory"),
        ("verifier", "verifier"),
    )
    for component, kind in component_kinds:
        if component in parts:
            return kind
    exact_kinds: dict[str, ArtifactKind] = {
        "openclaw.session.jsonl": "session",
        "reward.txt": "verifier",
        "ctrf.json": "verifier",
        "result.json": "result",
        "verification.json": "result",
        "failure.json": "result",
        "manifest.yaml": "configuration",
        "harbor-job.json": "configuration",
        "harbor-request.json": "configuration",
    }
    if name in exact_kinds:
        return exact_kinds[name]
    return _classify_named_artifact(parts, name)


def _classify_named_artifact(parts: tuple[str, ...], name: str) -> ArtifactKind:
    if name.endswith(".lock.json") or name == "lock.json":
        return "lock"
    if "agent" in parts and (
        name.endswith(".log") or name.endswith(".txt") or name == "exception.txt"
    ):
        return "agent_log"
    if name.endswith(".log") or name in {"job.log", "trial.log"}:
        return "execution_log"
    if name.endswith(".json") and ("config" in name or "openclaw.upload" in name):
        return "configuration"
    if "runtime" in name or "endpoint" in name:
        return "runtime"
    return "other"


def _artifacts(trial_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(trial_dir.rglob("*")):
        relative = path.relative_to(trial_dir)
        if relative.parts[:2] == ("artifacts", "workspace"):
            continue
        if path.is_symlink():
            raise ValueError("Harbor trial artifacts must not contain symlinks")
        if path.is_file():
            entries.append(
                {
                    "path": _relative(path, trial_dir),
                    "size": path.stat().st_size,
                    "digest": _digest(path),
                    "kind": classify_private_artifact(_relative(path, trial_dir)),
                    "classification": "private",
                }
            )
    return entries


def _bundle_directory(jobs_dir: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Harbor compatibility path is malformed")
    candidate = jobs_dir / relative
    if not candidate.resolve().is_relative_to(jobs_dir.resolve()):
        raise ValueError("Harbor compatibility path escapes the jobs directory")
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("Harbor compatibility path is not a directory")
    return candidate


def refresh_bundle_artifacts(jobs_dir: Path, output: Path) -> None:
    """Refresh retained-file metadata after evidence redaction."""
    bundle = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict):
        raise ValueError("Harbor compatibility bundle is not an object")
    jobs = bundle.get("jobs")
    trials = bundle.get("trials")
    if not isinstance(jobs, list) or not isinstance(trials, list):
        raise ValueError("Harbor compatibility bundle has no artifact inventories")
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Harbor compatibility job is malformed")
        job_dir = _bundle_directory(jobs_dir, job.get("path"))
        job["lock_digest"] = _digest(job_dir / "lock.json")
        job["result_digest"] = _digest(job_dir / "result.json")
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("Harbor compatibility trial is malformed")
        trial_dir = _bundle_directory(jobs_dir, trial.get("path"))
        trial["lock_digest"] = _digest(trial_dir / "lock.json")
        trial["result_digest"] = _digest(trial_dir / "result.json")
        trial["artifacts"] = _artifacts(trial_dir)
    output.write_text(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def refresh_retained_bundle(root: Path, *, strict: bool) -> str | None:
    """Refresh published metadata, or report why failed-run metadata was retained."""
    compatibility = root / "harbor-compatibility.json"
    if not compatibility.is_file():
        return None
    try:
        refresh_bundle_artifacts(root / "harbor-jobs", compatibility)
    except (OSError, ValueError) as error:
        if strict:
            raise
        return type(error).__name__
    return None


def _timing(value: TimingValue) -> dict[str, str | None]:
    return {
        "started_at": value.started_at.isoformat() if value.started_at else None,
        "finished_at": value.finished_at.isoformat() if value.finished_at else None,
    }


def _optional_timing(value: TimingValue | None) -> dict[str, str | None] | None:
    return _timing(value) if value is not None else None


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _nonnegative_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def _openclaw_session_paths(trial_dir: Path) -> list[Path]:
    sessions_dir = trial_dir / "agent" / "openclaw-sessions"
    paths = [
        path
        for path in sorted(sessions_dir.glob("*.jsonl"))
        if "trajectory" not in path.name.lower() and not path.is_symlink()
    ]
    if paths:
        return paths
    legacy = trial_dir / "agent" / "openclaw.session.jsonl"
    return [legacy] if legacy.is_file() and not legacy.is_symlink() else []


def _openclaw_usage_record(value: object) -> tuple[int, int, int, float | None] | None:
    if not isinstance(value, dict):
        return None
    message = value.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    uncached = _nonnegative_int(usage.get("input"))
    cache_read = _nonnegative_int(usage.get("cacheRead"))
    cache_write = _nonnegative_int(usage.get("cacheWrite"))
    output = _nonnegative_int(usage.get("output"))
    if None in (uncached, cache_read, cache_write, output):
        return None
    assert uncached is not None
    assert cache_read is not None
    assert cache_write is not None
    assert output is not None
    cached = cache_read + cache_write
    cost = usage.get("cost")
    total_cost = (
        _nonnegative_number(cost.get("total")) if isinstance(cost, dict) else None
    )
    return (uncached + cached, cached, output, total_cost)


def _openclaw_session_usage(trial_dir: Path) -> UsageTotals:
    input_tokens = 0
    cache_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    records = 0
    cost_records = 0
    for path in _openclaw_session_paths(trial_dir):
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        value = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    usage = _openclaw_usage_record(value)
                    if usage is None:
                        continue
                    input_tokens += usage[0]
                    cache_tokens += usage[1]
                    output_tokens += usage[2]
                    records += 1
                    total_cost = usage[3]
                    if total_cost is not None:
                        cost_usd += total_cost
                        cost_records += 1
        except (OSError, UnicodeError):
            continue
    if records == 0:
        return (None, None, None, None)
    return (
        input_tokens,
        cache_tokens,
        output_tokens,
        cost_usd if cost_records == records else None,
    )


def _usage_with_openclaw_fallback(native: UsageTotals, trial_dir: Path) -> UsageTotals:
    if all(value is not None for value in native):
        return native
    fallback = _openclaw_session_usage(trial_dir)
    return (
        native[0] if native[0] is not None else fallback[0],
        native[1] if native[1] is not None else fallback[1],
        native[2] if native[2] is not None else fallback[2],
        native[3] if native[3] is not None else fallback[3],
    )


def export_bundle(
    jobs_dir: Path,
    output: Path,
    harbor_revision: str,
    request_digest: str,
) -> None:
    # These modules exist only in the separately pinned Harbor environment.
    job_lock_module = importlib.import_module("harbor.models.job.lock")
    job_result_module = importlib.import_module("harbor.models.job.result")
    trial_result_module = importlib.import_module("harbor.models.trial.result")
    job_lock_type = job_lock_module.JobLock
    trial_lock_type = job_lock_module.TrialLock
    job_result_type = job_result_module.JobResult
    trial_result_type = trial_result_module.TrialResult

    jobs: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for job_result_path in sorted(jobs_dir.glob("*/result.json")):
        job_dir = job_result_path.parent
        job_lock_path = job_dir / "lock.json"
        job_result = job_result_type.model_validate_json(job_result_path.read_text())
        job_lock_type.model_validate_json(job_lock_path.read_text())
        jobs.append(
            {
                "path": _relative(job_dir, jobs_dir),
                "lock_digest": _digest(job_lock_path),
                "result_digest": _digest(job_result_path),
                "total_trials": job_result.n_total_trials,
                "completed_trials": job_result.stats.n_completed_trials,
                "errored_trials": job_result.stats.n_errored_trials,
            }
        )
    for result_path in sorted(jobs_dir.glob("*/*/result.json")):
        trial_dir = result_path.parent
        lock_path = trial_dir / "lock.json"
        result = trial_result_type.model_validate_json(result_path.read_text())
        lock = trial_lock_type.model_validate_json(lock_path.read_text())
        model = result.agent_info.model_info
        usage = _usage_with_openclaw_fallback(
            result.compute_token_cost_totals(), trial_dir
        )
        step_exceptions = [
            {
                "step_name": step.step_name,
                "exception_type": step.exception_info.exception_type,
                "exception_message": step.exception_info.exception_message,
            }
            for step in (result.step_results or [])
            if step.exception_info is not None
        ]
        rewards = (
            dict(result.verifier_result.rewards)
            if result.verifier_result is not None
            and result.verifier_result.rewards is not None
            else None
        )
        trials.append(
            {
                "path": _relative(trial_dir, jobs_dir),
                "trial_id": str(result.id),
                "trial_name": result.trial_name,
                "lock_digest": _digest(lock_path),
                "result_digest": _digest(result_path),
                "task_name": result.task_name,
                "task_digest": lock.task.digest,
                "agent_name": result.agent_info.name,
                "agent_version": result.agent_info.version,
                "model_provider": model.provider if model else None,
                "model_name": model.name if model else None,
                "exception_type": (
                    result.exception_info.exception_type
                    if result.exception_info is not None
                    else None
                ),
                "exception_message": (
                    result.exception_info.exception_message
                    if result.exception_info is not None
                    else None
                ),
                "step_exceptions": step_exceptions,
                "rewards": rewards,
                "timing": {
                    "trial": _timing(result),
                    "environment_setup": _optional_timing(result.environment_setup),
                    "agent_setup": _optional_timing(result.agent_setup),
                    "agent_execution": _optional_timing(result.agent_execution),
                    "verifier": _optional_timing(result.verifier),
                    "steps": [
                        {
                            "step_name": step.step_name,
                            "agent_execution": _optional_timing(step.agent_execution),
                            "verifier": _optional_timing(step.verifier),
                        }
                        for step in (result.step_results or [])
                    ],
                },
                "usage": {
                    "input_tokens": usage[0],
                    "cache_tokens": usage[1],
                    "output_tokens": usage[2],
                    "cost_usd": usage[3],
                },
                "artifacts": _artifacts(trial_dir),
            }
        )
    bundle = {
        "schema_version": "harbor-hf/harbor-compatibility/v1alpha3",
        "harbor_revision": harbor_revision,
        "harbor_version": importlib.metadata.version("harbor"),
        "request_digest": request_digest,
        "jobs": jobs,
        "trials": trials,
    }
    output.write_text(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harbor-revision", required=True)
    parser.add_argument("--request-digest", required=True)
    args = parser.parse_args()
    try:
        export_bundle(
            args.jobs_dir,
            args.output,
            args.harbor_revision,
            args.request_digest,
        )
    except Exception as error:
        print(
            f"Harbor compatibility export failed: {type(error).__name__}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
