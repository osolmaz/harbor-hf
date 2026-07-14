from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


class TimingValue(Protocol):
    started_at: datetime | None
    finished_at: datetime | None


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


def _artifacts(trial_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(trial_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("Harbor trial artifacts must not contain symlinks")
        if path.is_file():
            entries.append(
                {
                    "path": _relative(path, trial_dir),
                    "size": path.stat().st_size,
                    "digest": _digest(path),
                }
            )
    return entries


def _timing(value: TimingValue) -> dict[str, str | None]:
    return {
        "started_at": value.started_at.isoformat() if value.started_at else None,
        "finished_at": value.finished_at.isoformat() if value.finished_at else None,
    }


def export_bundle(
    jobs_dir: Path,
    output: Path,
    harbor_revision: str,
    request_digest: str,
) -> None:
    from harbor.models.job.lock import JobLock, TrialLock
    from harbor.models.job.result import JobResult
    from harbor.models.trial.result import TrialResult

    jobs: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for job_result_path in sorted(jobs_dir.glob("*/result.json")):
        job_dir = job_result_path.parent
        job_lock_path = job_dir / "lock.json"
        job_result = JobResult.model_validate_json(job_result_path.read_text())
        JobLock.model_validate_json(job_lock_path.read_text())
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
        result = TrialResult.model_validate_json(result_path.read_text())
        lock = TrialLock.model_validate_json(lock_path.read_text())
        model = result.agent_info.model_info
        usage = result.compute_token_cost_totals()
        step_exceptions = [
            {
                "step_name": step.step_name,
                "exception_type": step.exception_info.exception_type,
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
                "step_exceptions": step_exceptions,
                "rewards": rewards,
                "timing": _timing(result),
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
        "schema_version": "harbor-hf/harbor-compatibility/v1alpha1",
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
    export_bundle(
        args.jobs_dir,
        args.output,
        args.harbor_revision,
        args.request_digest,
    )


if __name__ == "__main__":
    main()
