from __future__ import annotations

import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import JsonValue

from harbor_hf.evidence import write_json
from harbor_hf.models import DeploymentProfile, ExperimentSpec
from harbor_hf.process import SubprocessRunner
from harbor_hf.profiling import (
    ProfilePlan,
    ProfilePoint,
    build_profile_plan,
    canonical_digest,
    new_unselected_profile,
    select_profile,
)
from harbor_hf.providers import routed_provider_model
from harbor_hf.runs import build_run_lock
from harbor_hf.worker import (
    EndpointManager,
    launch_cleanup_watchdog,
    require_paused_endpoint,
    resume_and_probe_endpoint,
    validate_endpoint_model,
)

_CONTROL_PARAMETERS: set[str] = set()


class ProfileWorkerError(RuntimeError):
    """Raised when remote serving-profile evidence cannot be produced safely."""


@dataclass(frozen=True)
class _Observation:
    success: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    saw_content: bool
    saw_reasoning: bool
    saw_tool_call: bool
    error: str | None = None


def run_profile_worker(plan_path: Path, output_root: Path) -> Path:
    plan = ProfilePlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    spec = ExperimentSpec.model_validate(plan.experiment)
    rebuilt = build_profile_plan(
        spec,
        profile_id=plan.profile_id,
        candidate_concurrency=plan.candidate_concurrency,
        max_spend_usd=plan.max_spend_usd,
        profile_timeout_seconds=plan.profile_timeout_seconds,
        sample_task_count=plan.workload.sample_task_count,
        objective=plan.objective,
    )
    if rebuilt != plan:
        raise ProfileWorkerError("profile plan does not match its embedded experiment")
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise ProfileWorkerError("profile worker requires HF_TOKEN")
    destination = output_root / plan.artifacts.prefix
    if destination.exists():
        raise ProfileWorkerError("profile artifact prefix already exists")
    destination.mkdir(parents=True)
    write_json(destination / "plan.json", plan.model_dump(mode="json"))
    profile = new_unselected_profile(plan)
    run_lock = build_run_lock(
        spec,
        model_id=plan.cell.model,
        deployment_id=plan.cell.deployment,
        agent_id=plan.cell.agent,
        run_id=f"profile-{plan.profile_id}",
        allow_provider=True,
    )
    endpoint_manager: EndpointManager | None = None
    deadline = time.monotonic() + plan.profile_timeout_seconds
    try:
        if isinstance(plan.deployment, DeploymentProfile):
            endpoint = plan.deployment.endpoint
            if endpoint is None:
                raise ProfileWorkerError("profile endpoint binding is missing")
            endpoint_manager = EndpointManager(
                endpoint.namespace, endpoint.name, SubprocessRunner()
            )
            baseline = endpoint_manager.describe()
            validate_endpoint_model(run_lock, baseline)
            require_paused_endpoint(baseline)
            launch_cleanup_watchdog(run_lock, endpoint, token)
            base_url = resume_and_probe_endpoint(
                destination,
                destination / "events.jsonl",
                run_lock,
                endpoint_manager,
                token,
                readiness_timeout_seconds=min(3600, _remaining(deadline)),
            )
            model_name = endpoint.served_model_name
        else:
            base_url = "https://router.huggingface.co"
            model_name = routed_provider_model(plan.deployment)
        _verify_smoke(plan, base_url, model_name, token, deadline)
        points = _run_ladder(plan, base_url, model_name, token, destination, deadline)
        profile = profile.model_copy(update={"points": points})
    finally:
        if endpoint_manager is not None:
            endpoint_manager.pause_and_verify()
    selected = select_profile(profile)
    write_json(destination / "profile.json", selected.model_dump(mode="json"))
    checksum = canonical_digest(selected)
    write_json(destination / "checksums.json", {"profile.json": checksum})
    (destination / "_SELECTED").write_text(checksum + "\n", encoding="utf-8")
    return destination


def _verify_smoke(
    plan: ProfilePlan,
    base_url: str,
    model_name: str,
    token: str,
    deadline: float,
) -> None:
    chat = _request(
        plan,
        base_url,
        model_name,
        token,
        "Reply with exactly OK.",
        tools=False,
        timeout=_remaining(deadline),
    )
    if not chat.success or not (chat.saw_content or chat.saw_reasoning):
        raise ProfileWorkerError(f"chat smoke failed: {chat.error or 'empty output'}")
    if plan.reasoning_required and not chat.saw_reasoning:
        raise ProfileWorkerError("reasoning smoke produced no reasoning channel")
    tool = _request(
        plan,
        base_url,
        model_name,
        token,
        "Use the lookup tool to look up the value for key alpha.",
        tools=True,
        timeout=_remaining(deadline),
    )
    if not tool.success or not tool.saw_tool_call:
        raise ProfileWorkerError(f"tool smoke failed: {tool.error or 'no tool call'}")


def _run_ladder(
    plan: ProfilePlan,
    base_url: str,
    model_name: str,
    token: str,
    destination: Path,
    deadline: float,
) -> list[ProfilePoint]:
    points: list[ProfilePoint] = []
    previous_rates: list[float] = []
    for concurrency in plan.candidate_concurrency:
        if time.monotonic() >= deadline:
            points.append(_skipped_point(concurrency, "profile time budget exhausted"))
            break
        planned = max(
            plan.workload.minimum_observations_per_point,
            2 * concurrency,
        )
        observations = _run_point(
            plan,
            base_url,
            model_name,
            token,
            concurrency,
            planned,
            deadline,
        )
        point = _summarize_point(concurrency, observations, repetition=1)
        points.append(point)
        _write_point(destination, point, observations)
        if (
            point.status != "completed"
            or point.error_rate > plan.objective.maximum_error_rate
        ):
            break
        rate = point.aggregate_output_tokens_per_second or 0
        previous_rates.append(rate)
        if (
            len(previous_rates) >= 3
            and previous_rates[-1] <= previous_rates[-2] <= previous_rates[-3]
        ):
            break
    boundary = [point.concurrency for point in points if point.status == "completed"][
        -2:
    ]
    for concurrency in boundary:
        for repetition in range(2, plan.workload.boundary_repetitions + 1):
            if time.monotonic() >= deadline:
                break
            planned = max(
                plan.workload.minimum_observations_per_point,
                2 * concurrency,
            )
            observations = _run_point(
                plan,
                base_url,
                model_name,
                token,
                concurrency,
                planned,
                deadline,
            )
            point = _summarize_point(concurrency, observations, repetition=repetition)
            points.append(point)
            _write_point(destination, point, observations)
    return points


def _write_point(
    destination: Path,
    point: ProfilePoint,
    observations: list[_Observation],
) -> None:
    point_dir = destination / "points" / str(point.concurrency)
    point_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        point_dir / f"{point.repetition}.json",
        {
            "point": point.model_dump(mode="json", exclude_none=True),
            "observations": [observation.__dict__ for observation in observations],
        },
    )


def _run_point(
    plan: ProfilePlan,
    base_url: str,
    model_name: str,
    token: str,
    concurrency: int,
    planned: int,
    deadline: float,
) -> list[_Observation]:
    observations: list[_Observation] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _request,
                plan,
                base_url,
                model_name,
                token,
                "Reply with exactly OK.",
                tools=False,
                timeout=_remaining(deadline),
            )
            for _ in range(planned)
        ]
        for future in as_completed(futures):
            try:
                observations.append(future.result())
            except Exception as error:
                observations.append(
                    _Observation(
                        False,
                        0,
                        0,
                        0,
                        None,
                        False,
                        False,
                        False,
                        type(error).__name__,
                    )
                )
    return observations


def _request(
    plan: ProfilePlan,
    base_url: str,
    model_name: str,
    token: str,
    prompt: str,
    *,
    tools: bool,
    timeout: int,
) -> _Observation:
    parameters = {
        key: value
        for key, value in plan.deployment.parameters.items()
        if key not in _CONTROL_PARAMETERS
    }
    payload: dict[str, JsonValue] = {
        **parameters,
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": plan.identity.max_output_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value by key.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            }
        ]
        payload["tool_choice"] = "required"
    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=max(1, timeout),
        )
        latency = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        usage = body.get("usage") or {}
        content = message.get("content")
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        tool_calls = message.get("tool_calls") or []
        return _Observation(
            True,
            latency,
            _integer(usage.get("prompt_tokens")),
            _integer(usage.get("completion_tokens")),
            choice.get("finish_reason"),
            isinstance(content, str) and bool(content.strip()),
            isinstance(reasoning, str) and bool(reasoning.strip()),
            isinstance(tool_calls, list) and bool(tool_calls),
        )
    except Exception as error:
        return _Observation(
            False,
            (time.perf_counter() - started) * 1000,
            0,
            0,
            None,
            False,
            False,
            False,
            type(error).__name__,
        )


def _summarize_point(
    concurrency: int,
    observations: list[_Observation],
    *,
    repetition: int,
) -> ProfilePoint:
    successful = [observation for observation in observations if observation.success]
    failed = len(observations) - len(successful)
    if not successful:
        return _failed_point(
            concurrency,
            len(observations),
            "all requests failed",
            repetition=repetition,
        )
    elapsed_seconds = max(observation.latency_ms for observation in observations) / 1000
    latencies = [observation.latency_ms for observation in successful]
    inputs = [observation.input_tokens for observation in successful]
    outputs = [observation.output_tokens for observation in successful]
    tpot = [
        observation.latency_ms / max(1, observation.output_tokens)
        for observation in successful
    ]
    error_rate = failed / len(observations)
    payload: dict[str, object] = {
        "concurrency": concurrency,
        "repetition": repetition,
        "status": "completed",
        "planned_count": len(observations),
        "completed_count": len(successful),
        "failed_count": failed,
        "error_rate": error_rate,
        "goodput_rate": 1 - error_rate,
        "aggregate_input_tokens_per_second": sum(inputs) / max(elapsed_seconds, 0.001),
        "aggregate_output_tokens_per_second": sum(outputs)
        / max(elapsed_seconds, 0.001),
        "tasks_per_hour": len(successful) * 3600 / max(elapsed_seconds, 0.001),
        "session_output_tokens_per_second": statistics.median(
            observation.output_tokens / max(observation.latency_ms / 1000, 0.001)
            for observation in successful
        ),
        "ttft_ms_p50": _percentile(latencies, 50),
        "ttft_ms_p95": _percentile(latencies, 95),
        "ttft_ms_p99": _percentile(latencies, 99),
        "tpot_ms_p50": _percentile(tpot, 50),
        "tpot_ms_p95": _percentile(tpot, 95),
        "tpot_ms_p99": _percentile(tpot, 99),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "latency_ms_p99": _percentile(latencies, 99),
        "prompt_tokens_p50": _percentile(inputs, 50),
        "prompt_tokens_p95": _percentile(inputs, 95),
        "prompt_tokens_max": max(inputs),
        "output_tokens_p50": _percentile(outputs, 50),
        "output_tokens_p95": _percentile(outputs, 95),
        "output_tokens_max": max(outputs),
        "artifact_prefix": f"points/{concurrency}/{repetition}.json",
    }
    payload["point_sha256"] = canonical_digest(payload)
    return ProfilePoint.model_validate(payload)


def _failed_point(
    concurrency: int, planned: int, reason: str, *, repetition: int = 1
) -> ProfilePoint:
    payload = {
        "concurrency": concurrency,
        "repetition": repetition,
        "status": "failed",
        "planned_count": planned,
        "completed_count": 0,
        "failed_count": planned,
        "error_rate": 1.0,
        "goodput_rate": 0.0,
        "artifact_prefix": f"points/{concurrency}/{repetition}.json",
        "failure_reason": reason,
    }
    return ProfilePoint(point_sha256=canonical_digest(payload), **payload)


def _skipped_point(concurrency: int, reason: str) -> ProfilePoint:
    payload = {
        "concurrency": concurrency,
        "repetition": 1,
        "status": "skipped",
        "planned_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "error_rate": 0.0,
        "goodput_rate": 0.0,
        "artifact_prefix": f"points/{concurrency}/1.json",
        "failure_reason": reason,
    }
    return ProfilePoint(point_sha256=canonical_digest(payload), **payload)


def _remaining(deadline: float) -> int:
    remaining = math.ceil(deadline - time.monotonic())
    if remaining < 1:
        raise ProfileWorkerError("profile time budget exhausted")
    return remaining


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _percentile(values: list[int] | list[float], percentile: int) -> float:
    ordered = sorted(float(value) for value in values)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return ordered[max(0, index)]
