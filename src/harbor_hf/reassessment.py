from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import shutil
import tarfile
import tempfile
import time
import tomllib
from collections.abc import Iterable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
from huggingface_hub import Sandbox
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from harbor_hf.judge_recorder import (
    JUDGE_RECORDER_PORT,
    JudgeEvidenceRecorder,
    verify_judge_exchange,
    verify_judge_recorder_summary,
)
from harbor_hf.models import TrialEvidencePolicy

_OPENAI_CHAT_COMPLETIONS = "https://api.openai.com/v1/chat/completions"
_SAFE_ID = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)


class ReassessmentError(RuntimeError):
    """Raised when a frozen-workspace reassessment cannot be retained safely."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceEvaluation(FrozenModel):
    campaign_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    publication_id: str = Field(min_length=1)
    source_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    result_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    index_revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class ReassessmentJudge(FrozenModel):
    provider: Literal["openai-api"] = "openai-api"
    api_url: Literal["https://api.openai.com/v1/chat/completions"] = (
        _OPENAI_CHAT_COMPLETIONS
    )
    model: Literal["gpt-5.6-luna"] = "gpt-5.6-luna"
    reasoning_effort: Literal["xhigh"] = "xhigh"
    strip_temperature: Literal[True] = True
    api_key_secret_name: Literal["OPENAI_API_KEY"] = "OPENAI_API_KEY"


class ReassessmentTrial(FrozenModel):
    trial_id: str = Field(pattern=r"^trial-[0-9a-f]{24}$")
    task_name: str = Field(min_length=1)
    task_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    logical_attempt: int = Field(ge=1)
    source_execution_id: str = Field(pattern=r"^exec-[0-9a-f]{32}$")
    source_trial_path: str = Field(min_length=1)
    source_outcome: Literal["scored", "agent_failed"]
    source_reward: float = Field(ge=0, le=1)
    action: Literal["rejudge", "fixed_zero"]

    @field_validator("task_name")
    @classmethod
    def task_name_is_safe(cls, value: str) -> str:
        if value != value.strip() or "/" in value or "\\" in value:
            raise ValueError("reassessment task name is unsafe")
        return value

    @field_validator("source_trial_path")
    @classmethod
    def source_path_is_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise ValueError("reassessment source trial path is unsafe")
        return value

    @model_validator(mode="after")
    def action_matches_source_outcome(self) -> ReassessmentTrial:
        expected = "fixed_zero" if self.source_outcome == "agent_failed" else "rejudge"
        if self.action != expected:
            raise ValueError("reassessment action disagrees with source outcome")
        if self.action == "fixed_zero" and self.source_reward != 0:
            raise ValueError(
                "fixed-zero reassessment trial has a nonzero source reward"
            )
        return self


class ReassessmentPlan(FrozenModel):
    schema_version: Literal["harbor-hf/reassessment-plan/v1"] = (
        "harbor-hf/reassessment-plan/v1"
    )
    reassessment_id: str = Field(min_length=1)
    created_at: datetime
    source: SourceEvaluation
    judge: ReassessmentJudge
    verifier_judge_timeout_seconds: Literal[900] = 900
    harbor_hf_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    benchmark_repository: Literal["ShellBench/public-tasks"] = "ShellBench/public-tasks"
    benchmark_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_image: str = Field(min_length=1)
    output_prefix: str = Field(min_length=1)
    judge_policy: dict[str, int | str]
    trials: list[ReassessmentTrial] = Field(min_length=1)
    plan_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("reassessment_id")
    @classmethod
    def reassessment_id_is_safe(cls, value: str) -> str:
        if any(character not in _SAFE_ID for character in value):
            raise ValueError("reassessment ID is unsafe")
        return value

    @field_validator("output_prefix")
    @classmethod
    def output_prefix_is_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise ValueError("reassessment output prefix is unsafe")
        return value

    @model_validator(mode="after")
    def identities_are_unique_and_digest_matches(self) -> ReassessmentPlan:
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("reassessment trial IDs must be unique")
        expected = reassessment_plan_digest(
            self.model_dump(mode="json", exclude={"plan_digest"})
        )
        if self.plan_digest != expected:
            raise ValueError("reassessment plan digest mismatch")
        return self


def reassessment_plan_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def load_reassessment_plan(path: Path) -> ReassessmentPlan:
    try:
        return ReassessmentPlan.model_validate_json(path.read_text())
    except (OSError, ValueError) as error:
        raise ReassessmentError("reassessment plan is invalid") from error


def _policy(plan: ReassessmentPlan) -> TrialEvidencePolicy:
    try:
        return TrialEvidencePolicy.model_validate(plan.judge_policy)
    except ValueError as error:
        raise ReassessmentError("reassessment judge policy is invalid") from error


def _json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _checksums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"checksums.json", "_SUCCESS"}
        and not path.name.startswith(".")
    }


def _assert_secrets_absent(root: Path, known_secrets: Iterable[str]) -> None:
    needles = tuple(value.encode() for value in known_secrets if value)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_bytes()
        if any(needle in content for needle in needles):
            raise ReassessmentError("reassessment output contains a known secret")


def _source_trial_root(evidence_root: Path, trial: ReassessmentTrial) -> Path:
    root = evidence_root / trial.source_trial_path
    if (
        not root.is_dir()
        or root.resolve().is_relative_to(evidence_root.resolve()) is False
    ):
        raise ReassessmentError("source trial root is missing or unsafe")
    execution_lock = json.loads(
        (
            root / "executions" / trial.source_execution_id / "execution.lock.json"
        ).read_text()
    )
    if (
        execution_lock.get("execution_id") != trial.source_execution_id
        or execution_lock.get("trial_id") != trial.trial_id
        or execution_lock.get("task_name") != trial.task_name
        or execution_lock.get("task_digest") != trial.task_digest
        or execution_lock.get("logical_attempt") != trial.logical_attempt
    ):
        raise ReassessmentError("source trial identity disagrees with plan")
    return root


def _source_native_trial(source_trial_root: Path, trial: ReassessmentTrial) -> Path:
    execution = source_trial_root / "executions" / trial.source_execution_id
    matches = list(execution.glob("harbor-jobs/*/*"))
    matches = [path for path in matches if path.is_dir()]
    if len(matches) != 1:
        raise ReassessmentError("source execution has no unique Harbor trial")
    return matches[0]


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def _prepare_verifier_tests(
    task_root: Path, timeout_seconds: int, benchmark_revision: str
) -> tuple[Path, dict[str, object]]:
    source = task_root / "tests"
    destination = Path(tempfile.mkdtemp(prefix="harbor-hf-reassessment-tests-"))
    copied = destination / "tests"
    shutil.copytree(source, copied, symlinks=True)
    source_digest = _tree_digest(copied)
    replacements = 0
    pattern = re.compile(rb"(urlopen\([^\n]*?timeout\s*=\s*)(60|90|120)(\b)")
    for path in copied.rglob("*.py"):
        content = path.read_bytes()
        transformed, count = pattern.subn(
            lambda match: (
                match.group(1) + str(timeout_seconds).encode() + match.group(3)
            ),
            content,
        )
        if count:
            path.write_bytes(transformed)
            replacements += count
    return copied, {
        "schema_version": "harbor-hf/verifier-source/v1",
        "benchmark_revision": benchmark_revision,
        "source_tree_digest": source_digest,
        "effective_tree_digest": _tree_digest(copied),
        "judge_timeout_seconds": timeout_seconds,
        "timeout_replacement_count": replacements,
    }


def _make_tar_gz(source: Path) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as handle:
        target = Path(handle.name)
    with tarfile.open(target, "w:gz", dereference=False) as archive:
        for path in sorted(source.rglob("*")):
            archive.add(
                path, arcname=path.relative_to(source).as_posix(), recursive=False
            )
    return target


def _sandbox_run(
    sandbox: Sandbox,
    command: str,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 900,
) -> tuple[int, str, str]:
    result = sandbox.run(
        ["/bin/bash", "-lc", command],
        shell=False,
        cwd="/",
        env=env,
        timeout=timeout,
        check=False,
    )
    return (
        int(getattr(result, "exit_code", -1)),
        str(getattr(result, "stdout", "") or ""),
        str(getattr(result, "stderr", "") or ""),
    )


def _upload_tree(sandbox: Sandbox, source: Path, destination: str) -> None:
    archive = _make_tar_gz(source)
    remote = f"/tmp/reassessment-{secrets.token_hex(8)}.tar.gz"
    try:
        sandbox.files.write(remote, archive.read_bytes())
        command = (
            f"mkdir -p {destination} && tar -xzf {remote} -C {destination} "
            f"&& rm -f {remote}"
        )
        code, stdout, stderr = _sandbox_run(sandbox, command, timeout=300)
        if code != 0:
            raise ReassessmentError(
                "sandbox upload extraction failed: " + (stderr or stdout)[-500:]
            )
    finally:
        archive.unlink(missing_ok=True)


def _download_tree(sandbox: Sandbox, source: str, destination: Path) -> None:
    remote = f"/tmp/reassessment-{secrets.token_hex(8)}.tar.gz"
    code, stdout, stderr = _sandbox_run(
        sandbox,
        f"test -d {source} && tar -czf {remote} -C {source} .",
        timeout=300,
    )
    if code != 0:
        raise ReassessmentError(
            "sandbox verifier evidence download failed: " + (stderr or stdout)[-500:]
        )
    data = sandbox.files.read(remote)
    _sandbox_run(sandbox, f"rm -f {remote}", timeout=30)
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as handle:
        handle.write(data)
        handle.flush()
        with tarfile.open(handle.name, "r:gz") as archive:
            archive.extractall(destination, filter="data")


def _payload_reward(payload: dict[str, object]) -> float | None:
    if payload.get("status") == "infra_error":
        raise ReassessmentError("reassessment verifier reported infrastructure error")
    for key in (
        "normal_plus_safety_score",
        "combined_score",
        "reward",
        "normal_plus_safety_pass",
        "combined_pass",
    ):
        value = payload.get(key)
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, int | float) and 0 <= float(value) <= 1:
            return float(value)
    return None


def _reward(verifier_dir: Path) -> float:
    result_path = verifier_dir / "agent_judge_results.json"
    reward_path = verifier_dir / "reward.txt"
    if result_path.is_file():
        value = _payload_reward(json.loads(result_path.read_text()))
        if value is not None:
            return value
    if reward_path.is_file():
        value = float(reward_path.read_text().strip())
        if 0 <= value <= 1:
            return value
    raise ReassessmentError("reassessment verifier emitted no bounded reward")


def _recover_unambiguous_selection(trial_root: Path, exchange_ids: list[str]) -> None:
    selection = trial_root / "verifier" / "judge-selection.json"
    calls = trial_root / "verifier" / "judge-calls.json"
    if selection.exists() or calls.exists() or len(exchange_ids) != 1:
        return
    exchange_id = exchange_ids[0]
    _json_atomic(
        calls,
        {
            "schema_version": "harbor-hf/judge-calls/v1",
            "exchange_ids": [exchange_id],
        },
    )
    _json_atomic(
        selection,
        {
            "schema_version": "harbor-hf/judge-selection/v1",
            "exchange_id": exchange_id,
        },
    )
    _json_atomic(
        trial_root / "judge-selection-recovery.json",
        {
            "schema_version": "harbor-hf/judge-selection-recovery/v1",
            "exchange_id": exchange_id,
            "basis": "only_successful_recorded_exchange",
        },
    )


def _validate_selection_evidence(trial_root: Path, exchange_ids: list[str]) -> None:
    selection = trial_root / "verifier" / "judge-selection.json"
    calls = trial_root / "verifier" / "judge-calls.json"
    if exchange_ids:
        if not selection.is_file() or not calls.is_file():
            raise ReassessmentError("reassessment judge selection evidence is missing")
        selected = json.loads(selection.read_text()).get("exchange_id")
        declared = json.loads(calls.read_text()).get("exchange_ids")
        if selected not in exchange_ids or declared != exchange_ids:
            raise ReassessmentError("reassessment judge selection is inconsistent")
    elif selection.exists() or calls.exists():
        raise ReassessmentError("zero-call reassessment has judge selection evidence")


def _validate_judge_evidence(
    trial_root: Path,
    *,
    expected_model: str,
    expected_reasoning_effort: str,
) -> int:
    judge_root = trial_root / "judge-records"
    summary = verify_judge_recorder_summary(judge_root / "recorder.json")
    exchanges = sorted(path for path in judge_root.glob("judge-*") if path.is_dir())
    if len(exchanges) != summary.exchange_count or summary.rejected_call_count != 0:
        raise ReassessmentError("reassessment judge recorder summary is inconsistent")
    for path in exchanges:
        exchange = verify_judge_exchange(path)
        if (
            exchange.provider != "openai-api"
            or exchange.forwarded_model != expected_model
            or exchange.outcome != "success"
            or exchange.transformation != "parameters_enforced"
        ):
            raise ReassessmentError("reassessment judge exchange identity is invalid")
        forwarded = json.loads((path / "request-forwarded.bin").read_bytes())
        if (
            forwarded.get("model") != expected_model
            or forwarded.get("reasoning_effort") != expected_reasoning_effort
            or "temperature" in forwarded
        ):
            raise ReassessmentError("reassessment judge parameters are invalid")
    exchange_ids = [path.name for path in exchanges]
    _recover_unambiguous_selection(trial_root, exchange_ids)
    _validate_selection_evidence(trial_root, exchange_ids)
    return len(exchanges)


def _task_config(tasks_root: Path, trial: ReassessmentTrial) -> tuple[Path, str, str]:
    task_root = tasks_root / trial.task_name
    if not task_root.is_dir() or not task_root.resolve().is_relative_to(
        tasks_root.resolve()
    ):
        raise ReassessmentError("reassessment task source is missing or unsafe")
    config = tomllib.loads((task_root / "task.toml").read_text())
    try:
        image = str(config["environment"]["docker_image"])
        command = str(config["verifier"].get("command", "bash tests/test.sh"))
    except (KeyError, TypeError) as error:
        raise ReassessmentError("reassessment task configuration is invalid") from error
    return task_root, image, command


def _trial_output_root(output_root: Path, trial: ReassessmentTrial) -> Path:
    return output_root / "trials" / trial.trial_id


def _skip_or_require_absent(final: Path, label: str) -> bool:
    if (final / "_SUCCESS").is_file():
        return True
    if final.exists() and {path.name for path in final.iterdir()} != {"attempts"}:
        raise ReassessmentError(f"unfinished {label} output already exists")
    return False


def _publish_success(staging: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    if not final.exists():
        os.replace(staging, final)
        return
    if {path.name for path in final.iterdir()} != {"attempts"}:
        raise ReassessmentError("reassessment success destination is contaminated")
    for path in staging.iterdir():
        os.replace(path, final / path.name)
    staging.rmdir()


def _retain_failed_attempt(
    *,
    staging: Path,
    final: Path,
    execution_id: str,
    error: Exception,
    known_secrets: tuple[str, ...],
) -> None:
    if not staging.is_dir():
        return
    _json_atomic(
        staging / "failure.json",
        {
            "schema_version": "harbor-hf/reassessment-failure/v1",
            "execution_id": execution_id,
            "error_type": type(error).__name__,
            "message": "reassessment execution failed",
            "failed_at": datetime.now(UTC).isoformat(),
        },
    )
    _assert_secrets_absent(staging, known_secrets)
    _json_atomic(staging / "checksums.json", _checksums(staging))
    (staging / "_FAILED").write_text("")
    attempt = final / "attempts" / execution_id
    attempt.parent.mkdir(parents=True, exist_ok=True)
    if attempt.exists():
        raise ReassessmentError("reassessment failed-attempt identity collided")
    os.replace(staging, attempt)


def _require_command_success(code: int, stdout: str, stderr: str, label: str) -> None:
    if code != 0:
        raise ReassessmentError(label + ": " + (stderr or stdout)[-500:])


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ReassessmentError(label)


def _remove_optional_tree(path: Path | None) -> None:
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def _write_fixed_zero(
    output_root: Path,
    trial: ReassessmentTrial,
    source_trial_root: Path,
    plan: ReassessmentPlan,
) -> None:
    final = _trial_output_root(output_root, trial)
    if _skip_or_require_absent(final, "fixed-zero trial"):
        return
    staging = final.with_name(f".{trial.trial_id}.{secrets.token_hex(8)}.tmp")
    staging.mkdir(parents=True)
    _json_atomic(
        staging / "result.json",
        {
            "schema_version": "harbor-hf/reassessment-result/v1",
            "reassessment_id": plan.reassessment_id,
            "trial_id": trial.trial_id,
            "task_name": trial.task_name,
            "logical_attempt": trial.logical_attempt,
            "source_execution_id": trial.source_execution_id,
            "source_execution_checksum": _sha256(
                source_trial_root
                / "executions"
                / trial.source_execution_id
                / "checksums.json"
            ),
            "action": "fixed_zero",
            "reward": 0.0,
            "strict_pass": False,
            "judge_exchange_count": 0,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    _json_atomic(staging / "checksums.json", _checksums(staging))
    (staging / "_SUCCESS").write_text("")
    _publish_success(staging, final)


def _execute_rejudge(
    *,
    output_root: Path,
    evidence_root: Path,
    tasks_root: Path,
    trial: ReassessmentTrial,
    plan: ReassessmentPlan,
    recorder: JudgeEvidenceRecorder,
    recorder_base_url: str,
    hf_token: str,
    openai_token: str,
    policy: TrialEvidencePolicy,
) -> None:
    final = _trial_output_root(output_root, trial)
    if _skip_or_require_absent(final, "reassessment trial"):
        return
    source_root = _source_trial_root(evidence_root, trial)
    native = _source_native_trial(source_root, trial)
    task_root, image, command = _task_config(tasks_root, trial)
    if image != plan.runtime_image:
        raise ReassessmentError("reassessment runtime image disagrees with plan")
    restored = Path(tempfile.mkdtemp(prefix="harbor-hf-reassessment-source-"))
    staging = final.with_name(f".{trial.trial_id}.{secrets.token_hex(8)}.tmp")
    execution_id = "rejudge-" + secrets.token_hex(16)
    capability: str | None = None
    capability_secret = ""
    sandbox: Sandbox | None = None
    prepared_tests: Path | None = None
    started = time.monotonic()
    failure: Exception | None = None
    try:
        evidence_manifest = json.loads(
            (native / "evidence" / "manifest.json").read_text()
        )
        workspace = evidence_manifest["workspace"]
        from harbor_hf.trial_evidence import WorkspaceEvidence, restore_workspace

        restore_workspace(native, WorkspaceEvidence.model_validate(workspace), restored)
        app = restored / "app"
        _require_directory(app, "reassessment source workspace did not restore")
        staging.mkdir(parents=True)
        judge_root = staging / "judge-records"
        capability = recorder.register_scope(
            execution_id=execution_id,
            trial_id=trial.trial_id,
            model=plan.judge.model,
            destination=judge_root,
            policy=policy,
            known_secrets=(hf_token,),
        )
        capability_secret = capability
        judge_url = recorder.scoped_url(recorder_base_url, capability)
        sandbox = Sandbox.create(
            image=image,
            flavor="cpu-basic",
            idle_timeout=1800,
            forward_hf_token=False,
            start_timeout=300,
        )
        reset = (
            "find /app -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; "
            "find /tests -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + "
            "2>/dev/null || true; mkdir -p /app /tests /logs/verifier"
        )
        code, stdout, stderr = _sandbox_run(sandbox, reset, timeout=120)
        _require_command_success(
            code, stdout, stderr, "reassessment sandbox reset failed"
        )
        _upload_tree(sandbox, app, "/app")
        tests, verifier_source = _prepare_verifier_tests(
            task_root,
            plan.verifier_judge_timeout_seconds,
            plan.benchmark_revision,
        )
        prepared_tests = tests.parent
        _upload_tree(sandbox, tests, "/tests")
        _json_atomic(staging / "verifier-source.json", verifier_source)
        verifier_env = {
            "AGENT_JUDGE_API_URL": judge_url,
            "AGENT_JUDGE_MODEL": plan.judge.model,
            "AGENT_JUDGE_API_KEY": hf_token,
        }
        shell = (
            "set +e; "
            + command.replace("tests/", "/tests/")
            + " > /logs/verifier/test-stdout.txt 2>&1; "
            "code=$?; printf '%s\\n' \"$code\" > /logs/verifier/exit-code.txt; exit 0"
        )
        code, stdout, stderr = _sandbox_run(
            sandbox,
            shell,
            env=verifier_env,
            timeout=policy.judge_timeout_seconds * policy.judge_max_calls_per_execution
            + 300,
        )
        _require_command_success(
            code, stdout, stderr, "reassessment verifier wrapper failed"
        )
        _download_tree(sandbox, "/logs/verifier", staging / "verifier")
        recorder.revoke_scope(capability)
        capability = None
        reward = _reward(staging / "verifier")
        exchange_count = _validate_judge_evidence(
            staging,
            expected_model=plan.judge.model,
            expected_reasoning_effort=plan.judge.reasoning_effort,
        )
        _json_atomic(
            staging / "result.json",
            {
                "schema_version": "harbor-hf/reassessment-result/v1",
                "reassessment_id": plan.reassessment_id,
                "trial_id": trial.trial_id,
                "task_name": trial.task_name,
                "logical_attempt": trial.logical_attempt,
                "source_execution_id": trial.source_execution_id,
                "source_trial_checksum": _sha256(source_root / "checksums.json"),
                "action": "rejudge",
                "model": plan.judge.model,
                "reasoning_effort": plan.judge.reasoning_effort,
                "reward": reward,
                "strict_pass": reward == 1.0,
                "judge_exchange_count": exchange_count,
                "duration_seconds": round(time.monotonic() - started, 3),
                "completed_at": datetime.now(UTC).isoformat(),
            },
        )
        _assert_secrets_absent(
            staging,
            (hf_token, openai_token, capability_secret),
        )
        _json_atomic(staging / "checksums.json", _checksums(staging))
        (staging / "_SUCCESS").write_text("")
        _publish_success(staging, final)
    except Exception as error:
        failure = error
        raise
    finally:
        if capability is not None:
            recorder.revoke_scope(capability)
        if sandbox is not None:
            with suppress(Exception):
                sandbox.kill()
        shutil.rmtree(restored, ignore_errors=True)
        _remove_optional_tree(prepared_tests)
        if failure is not None:
            _retain_failed_attempt(
                staging=staging,
                final=final,
                execution_id=execution_id,
                error=failure,
                known_secrets=(hf_token, openai_token, capability_secret),
            )
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


async def _run_trials(
    *,
    plan: ReassessmentPlan,
    evidence_root: Path,
    tasks_root: Path,
    output_root: Path,
    recorder: JudgeEvidenceRecorder,
    recorder_base_url: str,
    hf_token: str,
    openai_token: str,
    max_parallel: int,
) -> None:
    policy = _policy(plan)
    semaphore = asyncio.Semaphore(max_parallel)

    async def one(trial: ReassessmentTrial) -> None:
        async with semaphore:
            source_root = _source_trial_root(evidence_root, trial)
            if trial.action == "fixed_zero":
                await asyncio.to_thread(
                    _write_fixed_zero, output_root, trial, source_root, plan
                )
                return
            await asyncio.to_thread(
                _execute_rejudge,
                output_root=output_root,
                evidence_root=evidence_root,
                tasks_root=tasks_root,
                trial=trial,
                plan=plan,
                recorder=recorder,
                recorder_base_url=recorder_base_url,
                hf_token=hf_token,
                openai_token=openai_token,
                policy=policy,
            )

    results = await asyncio.gather(
        *(one(trial) for trial in plan.trials), return_exceptions=True
    )
    failures = [error for error in results if isinstance(error, BaseException)]
    if failures:
        raise ReassessmentError(
            f"{len(failures)} reassessment trial(s) failed; first: {failures[0]}"
        )


def _wait_for_ingress(base_url: str, hf_token: str) -> None:
    deadline = time.monotonic() + 120
    while True:
        try:
            response = httpx.get(
                f"{base_url}/healthz",
                headers={"Authorization": f"Bearer {hf_token}"},
                timeout=5,
                follow_redirects=False,
            )
            if response.status_code == 200 and response.json() == {"status": "ok"}:
                return
            if response.status_code in {401, 403}:
                raise ReassessmentError("reassessment ingress rejected HF token")
        except httpx.TransportError:
            pass
        if time.monotonic() >= deadline:
            raise ReassessmentError("reassessment ingress did not become ready")
        time.sleep(1)


def run_reassessment(
    *,
    plan_path: Path,
    evidence_root: Path,
    tasks_root: Path,
    output_mount: Path,
    max_parallel: int,
) -> None:
    plan = load_reassessment_plan(plan_path)
    if not 1 <= max_parallel <= 16:
        raise ReassessmentError("reassessment parallelism must be between 1 and 16")
    hf_token = os.environ.get("HF_TOKEN", "")
    openai_token = os.environ.get(plan.judge.api_key_secret_name, "")
    job_id = os.environ.get("JOB_ID", "")
    if not hf_token or not openai_token or not job_id:
        raise ReassessmentError(
            "reassessment worker secrets or HF Job identity are missing"
        )
    output_root = output_mount / plan.output_prefix
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "reassessment.lock.json"
    plan_payload = json.loads(plan_path.read_text())
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != plan_payload:
            raise ReassessmentError("reassessment output lock disagrees with plan")
    else:
        _json_atomic(lock_path, plan_payload)
    recorder = JudgeEvidenceRecorder(
        token=openai_token,
        upstream_url=plan.judge.api_url,
        reasoning_effort=plan.judge.reasoning_effort,
        strip_temperature=plan.judge.strip_temperature,
    )
    base_url = f"https://{job_id}--{JUDGE_RECORDER_PORT}.hf.jobs"
    try:
        recorder.start(port=JUDGE_RECORDER_PORT)
        _wait_for_ingress(base_url, hf_token)
        asyncio.run(
            _run_trials(
                plan=plan,
                evidence_root=evidence_root,
                tasks_root=tasks_root,
                output_root=output_root,
                recorder=recorder,
                recorder_base_url=base_url,
                hf_token=hf_token,
                openai_token=openai_token,
                max_parallel=max_parallel,
            )
        )
    finally:
        recorder.close()
    completed = sum(
        (output_root / "trials" / trial.trial_id / "_SUCCESS").is_file()
        for trial in plan.trials
    )
    _json_atomic(
        output_root / "summary.json",
        {
            "schema_version": "harbor-hf/reassessment-summary/v1",
            "reassessment_id": plan.reassessment_id,
            "planned_trials": len(plan.trials),
            "completed_trials": completed,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    if completed != len(plan.trials):
        raise ReassessmentError("reassessment did not complete every planned trial")
    (output_root / "_SUCCESS").write_text("")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rejudge immutable Harbor workspaces without rerunning the agent."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--output-mount", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=4)
    arguments = parser.parse_args()
    run_reassessment(
        plan_path=arguments.plan,
        evidence_root=arguments.evidence_root,
        tasks_root=arguments.tasks_root,
        output_mount=arguments.output_mount,
        max_parallel=arguments.max_parallel,
    )


if __name__ == "__main__":
    main()
