from __future__ import annotations

import math
import os
import statistics
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from pydantic import JsonValue

from harbor_hf.endpoints import DesiredEndpoint, EndpointProvisioner
from harbor_hf.evidence import (
    assert_secret_absent,
    scrub_secret,
    scrub_secret_paths,
    write_checksums,
    write_json,
)
from harbor_hf.harbor_adapter import FilesystemHarborExecutionAdapter
from harbor_hf.harbor_adapter.errors import HarborTrialFailure
from harbor_hf.harbor_adapter.models import HarborCompatibilityTrial
from harbor_hf.harbor_adapter.validation import load_compatibility_bundle
from harbor_hf.hf_endpoints import HuggingFaceEndpointAdapter
from harbor_hf.models import ExperimentSpec
from harbor_hf.process import SubprocessRunner, run_streaming
from harbor_hf.profile_worker_transport import ProfileTransport
from harbor_hf.profiling import (
    ProfilePlan,
    ProfilePoint,
    bind_profile_target,
    build_profile_plan,
    canonical_digest,
    new_unselected_profile,
    select_profile,
)
from harbor_hf.provider_models import ProviderTarget
from harbor_hf.runs import (
    RunLock,
    build_run_lock,
    harbor_process_environment,
    run_secret_values,
)
from harbor_hf.worker import (
    EndpointManager,
    launch_cleanup_watchdog,
    prepare_locked_source,
    require_executable,
    require_paused_endpoint,
    resume_and_probe_endpoint,
    validate_endpoint_model,
)


class ProfileWorkerError(RuntimeError):
    """Raised when remote serving-profile evidence cannot be produced safely."""


@dataclass(frozen=True)
class _SmokeObservation:
    success: bool
    input_tokens: int
    output_tokens: int
    saw_content: bool
    saw_reasoning: bool
    saw_tool_call: bool
    error: str | None = None


@dataclass(frozen=True)
class _TaskObservation:
    success: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    task_name: str
    error: str | None = None


@dataclass(frozen=True)
class _PointResult:
    observations: list[_TaskObservation]
    elapsed_ms: float


def run_profile_worker(plan_path: Path, output_root: Path) -> Path:
    plan = ProfilePlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    spec = ExperimentSpec.model_validate(plan.experiment)
    rebuilt = build_profile_plan(
        spec,
        profile_id=plan.profile_id,
        candidate_concurrency=plan.candidate_concurrency,
        max_spend_usd=plan.max_spend_usd,
        profile_timeout_seconds=plan.profile_timeout_seconds,
        estimated_profile_cost_usd=plan.estimated_profile_cost_usd,
        sample_task_count=plan.workload.sample_task_count,
        objective=plan.objective,
    )
    if rebuilt != plan:
        raise ProfileWorkerError("profile plan does not match its embedded experiment")
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise ProfileWorkerError("profile worker requires HF_TOKEN")
    if spec.remote is None:
        raise ProfileWorkerError("profile worker requires remote execution settings")
    destination = output_root / plan.artifacts.prefix
    if destination.exists():
        raise ProfileWorkerError("profile artifact prefix already exists")
    destination.mkdir(parents=True)
    write_json(destination / "plan.json", plan.model_dump(mode="json"))
    secrets: str | tuple[str, ...] = token
    try:
        profile = new_unselected_profile(plan)
        bound_spec, desired_endpoint = bind_profile_target(plan)
        run_lock = build_run_lock(
            bound_spec,
            model_id=plan.cell.model,
            deployment_id=plan.cell.deployment,
            agent_id=plan.cell.agent,
            run_id=f"profile-{plan.profile_id}",
            allow_provider=True,
        )
        deadline = time.monotonic() + plan.profile_timeout_seconds
        require_executable("git")
        require_executable("uv")
        secrets = run_secret_values(run_lock, token)
        with tempfile.TemporaryDirectory(prefix="harbor-hf-profile-") as temporary:
            harbor_source = Path(temporary) / "harbor"
            prepare_locked_source(
                spec.remote.harbor.source, harbor_source, SubprocessRunner()
            )
            with _profile_transport(
                plan,
                run_lock,
                destination,
                token,
                deadline,
                desired_endpoint=desired_endpoint,
            ) as transport:
                _verify_smoke(plan, transport, token, deadline)
                points = _run_ladder(
                    plan,
                    run_lock,
                    transport,
                    harbor_source,
                    token,
                    destination,
                    deadline,
                )
                profile = profile.model_copy(update={"points": points})
        selected = select_profile(profile)
        write_json(destination / "profile.json", selected.model_dump(mode="json"))
        selected_digest = canonical_digest(selected)
        _finalize_profile(destination, secrets)
        (destination / "_SELECTED").write_text(selected_digest + "\n", encoding="utf-8")
        return destination
    except Exception as error:
        failure = {
            "error_type": type(error).__name__,
            "message": _redact_message(str(error), secrets),
        }
        write_json(
            destination / "failure.json",
            failure,
        )
        _finalize_profile(destination, secrets)
        write_json(destination / "_FAILED", failure)
        raise


@contextmanager
def _profile_transport(
    plan: ProfilePlan,
    run_lock: RunLock,
    destination: Path,
    token: str,
    deadline: float,
    *,
    desired_endpoint: DesiredEndpoint | None,
) -> Iterator[ProfileTransport]:
    if isinstance(plan.deployment, ProviderTarget):
        with ProfileTransport.for_provider(
            plan.deployment,
            token=token,
            evidence_path=destination / "provider-requests.jsonl",
            deadline=deadline,
        ) as transport:
            yield transport
        return
    deployment = run_lock.deployment
    if isinstance(deployment, ProviderTarget):
        raise ProfileWorkerError("profile target changed while binding its endpoint")
    endpoint = deployment.endpoint
    if endpoint is None:
        raise ProfileWorkerError("profile endpoint binding is missing")
    manager = EndpointManager(endpoint.namespace, endpoint.name, SubprocessRunner())
    if desired_endpoint is not None:
        provisioner = EndpointProvisioner(HuggingFaceEndpointAdapter(token=token))
        existing = provisioner.inspect(desired_endpoint)
        if existing is not None:
            provisioner.create_or_adopt(
                desired_endpoint,
                timeout_seconds=min(900, _remaining(deadline)),
            )
            launch_cleanup_watchdog(run_lock, endpoint, token)
        else:
            launch_cleanup_watchdog(run_lock, endpoint, token)
            provisioner.create_or_adopt(
                desired_endpoint,
                timeout_seconds=min(900, _remaining(deadline)),
            )
    else:
        baseline = manager.describe()
        validate_endpoint_model(run_lock, baseline)
        require_paused_endpoint(baseline)
        launch_cleanup_watchdog(run_lock, endpoint, token)
    baseline = manager.describe()
    validate_endpoint_model(run_lock, baseline)
    require_paused_endpoint(baseline)
    try:
        base_url = resume_and_probe_endpoint(
            destination,
            destination / "events.jsonl",
            run_lock,
            manager,
            token,
            readiness_timeout_seconds=min(3600, _remaining(deadline)),
        )
        yield ProfileTransport.for_endpoint(base_url, endpoint.served_model_name)
    finally:
        manager.pause_and_verify()


def _verify_smoke(
    plan: ProfilePlan,
    transport: ProfileTransport,
    token: str,
    deadline: float,
) -> None:
    with transport.scope("smoke") as (base_url, model_name, capability):
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
            detail = chat.error or "empty output"
            raise ProfileWorkerError(f"chat smoke failed: {detail}")
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
            detail = tool.error or "no tool call"
            raise ProfileWorkerError(f"tool smoke failed: {detail}")
        _verify_token_limits(plan, base_url, model_name, token, deadline)


def _verify_token_limits(
    plan: ProfilePlan,
    base_url: str,
    model_name: str,
    token: str,
    deadline: float,
) -> None:
    available_input = (
        plan.identity.server_context_tokens - plan.identity.max_output_tokens
    )
    if available_input < 1:
        raise ProfileWorkerError("output limit leaves no input context")
    samples: list[tuple[int, int]] = []
    for repeats in (256, 1024):
        observation = _request(
            plan,
            base_url,
            model_name,
            token,
            _capacity_prompt(repeats),
            tools=False,
            timeout=_remaining(deadline),
            max_tokens=1,
            allow_length=True,
        )
        if not observation.success:
            detail = observation.error or "calibration request failed"
            raise ProfileWorkerError(f"context calibration failed: {detail}")
        samples.append((repeats, observation.input_tokens))
    repeat_delta = samples[1][0] - samples[0][0]
    token_delta = samples[1][1] - samples[0][1]
    if token_delta <= 0:
        raise ProfileWorkerError("context calibration did not increase token usage")
    tokens_per_repeat = token_delta / repeat_delta
    fixed_tokens = samples[0][1] - samples[0][0] * tokens_per_repeat
    target_input = max(1, available_input - 256)
    repeats = max(1, math.floor((target_input - fixed_tokens) / tokens_per_repeat))
    capacity = _request(
        plan,
        base_url,
        model_name,
        token,
        _capacity_prompt(repeats),
        tools=False,
        timeout=_remaining(deadline),
        max_tokens=plan.identity.max_output_tokens,
        allow_length=True,
    )
    if not capacity.success:
        detail = capacity.error or "capacity request failed"
        raise ProfileWorkerError(f"token limit smoke failed: {detail}")
    if capacity.input_tokens + plan.identity.max_output_tokens < (
        plan.identity.server_context_tokens - 512
    ):
        raise ProfileWorkerError("token limit smoke did not reach declared context")


def _capacity_prompt(repeats: int) -> str:
    return "Read all tokens, then reply with exactly OK.\n" + "x " * repeats


def _run_ladder(
    plan: ProfilePlan,
    run_lock: RunLock,
    transport: ProfileTransport,
    harbor_source: Path,
    token: str,
    destination: Path,
    deadline: float,
) -> list[ProfilePoint]:
    points: list[ProfilePoint] = []
    previous_rates: list[float] = []
    for concurrency in plan.candidate_concurrency:
        if time.monotonic() >= deadline:
            point = _skipped_point(concurrency, "profile time budget exhausted")
            points.append(point)
            _write_point(destination, point, [], 0)
            break
        result = _run_point(
            plan,
            run_lock,
            transport,
            harbor_source,
            token,
            concurrency,
            repetition=1,
            destination=destination,
            deadline=deadline,
        )
        point = _summarize_point(
            concurrency,
            result.observations,
            elapsed_ms=result.elapsed_ms,
            repetition=1,
        )
        points.append(point)
        _write_point(destination, point, result.observations, result.elapsed_ms)
        if not _point_passes_objective(plan, point):
            _verify_smoke(plan, transport, token, deadline)
            retry = _run_point(
                plan,
                run_lock,
                transport,
                harbor_source,
                token,
                concurrency,
                repetition=2,
                destination=destination,
                deadline=deadline,
            )
            retried_point = _summarize_point(
                concurrency,
                retry.observations,
                elapsed_ms=retry.elapsed_ms,
                repetition=2,
            )
            points.append(retried_point)
            _write_point(
                destination, retried_point, retry.observations, retry.elapsed_ms
            )
            if not _point_passes_objective(plan, retried_point):
                break
            point = retried_point
        rate = _point_ladder_rate(plan, point)
        if rate is not None:
            previous_rates.append(rate)
        if (
            rate is not None
            and len(previous_rates) >= 3
            and previous_rates[-1] <= previous_rates[-2] <= previous_rates[-3]
        ):
            break
    boundary = [
        point.concurrency for point in points if _point_passes_objective(plan, point)
    ][-2:]
    points.extend(
        _run_boundary_repetitions(
            plan,
            run_lock,
            transport,
            harbor_source,
            token,
            destination,
            deadline,
            boundary,
            points,
        )
    )
    return points


def _run_boundary_repetitions(
    plan: ProfilePlan,
    run_lock: RunLock,
    transport: ProfileTransport,
    harbor_source: Path,
    token: str,
    destination: Path,
    deadline: float,
    boundary: list[int],
    existing_points: list[ProfilePoint],
) -> list[ProfilePoint]:
    points: list[ProfilePoint] = []
    for concurrency in boundary:
        completed_repetitions = {
            point.repetition
            for point in existing_points
            if point.concurrency == concurrency
        }
        for repetition in range(2, plan.workload.boundary_repetitions + 1):
            if repetition in completed_repetitions:
                continue
            if time.monotonic() >= deadline:
                point = _skipped_point(
                    concurrency,
                    "profile time budget exhausted during boundary repetition",
                    repetition=repetition,
                )
                points.append(point)
                _write_point(destination, point, [], 0)
                break
            result = _run_point(
                plan,
                run_lock,
                transport,
                harbor_source,
                token,
                concurrency,
                repetition=repetition,
                destination=destination,
                deadline=deadline,
            )
            point = _summarize_point(
                concurrency,
                result.observations,
                elapsed_ms=result.elapsed_ms,
                repetition=repetition,
            )
            points.append(point)
            _write_point(destination, point, result.observations, result.elapsed_ms)
    return points


def _point_ladder_rate(plan: ProfilePlan, point: ProfilePoint) -> float | None:
    if plan.objective.kind == "maximum_stable_concurrency":
        return None
    if plan.objective.kind == "maximum_throughput":
        return point.aggregate_output_tokens_per_second or 0
    return point.tasks_per_hour or 0


def _run_point(
    plan: ProfilePlan,
    run_lock: RunLock,
    transport: ProfileTransport,
    harbor_source: Path,
    token: str,
    concurrency: int,
    *,
    repetition: int,
    destination: Path,
    deadline: float,
) -> _PointResult:
    sampled_tasks = _sample_tasks(plan)
    minimum = max(plan.workload.minimum_observations_per_point, 2 * concurrency)
    if isinstance(plan.deployment, ProviderTarget):
        tasks = dict(list(sampled_tasks.items())[:minimum])
        if len(tasks) < minimum:
            raise ProfileWorkerError(
                "provider profile lacks distinct tasks for this concurrency"
            )
        attempts = 1
    else:
        tasks = sampled_tasks
        attempts = math.ceil(minimum / len(tasks))
    point_root = destination / "points" / str(concurrency) / str(repetition)
    jobs_dir = point_root / "harbor-jobs"
    execution_root = point_root / "harbor-execution"
    jobs_dir.mkdir(parents=True)
    execution_root.mkdir()
    adapter = FilesystemHarborExecutionAdapter()
    scope = f"c{concurrency}-r{repetition}"
    with transport.scope(scope) as (base_url, _model_name, capability):
        prepared = adapter.prepare(
            run_lock,
            execution_root,
            jobs_dir,
            base_url,
            harbor_source,
            task_names=list(tasks),
            attempts=attempts,
            concurrency=concurrency,
            expected_task_digests=tasks,
        )
        timeout = _remaining(deadline)
        started = time.monotonic()
        try:
            with harbor_process_environment(
                run_lock,
                token=token,
                inference_base_url=base_url,
                redaction_secrets=(capability,) if capability else (),
            ) as environment:
                outcome = adapter.execute(
                    prepared,
                    harbor_source,
                    jobs_dir,
                    execution_root / "harbor.log",
                    environment=environment,
                    timeout_seconds=timeout,
                    stream_runner=run_streaming,
                    deadline=deadline,
                )
        except HarborTrialFailure:
            elapsed_ms = (time.monotonic() - started) * 1000
            _scrub_capability(point_root, capability)
            compatibility_path = prepared.request_path.with_name(
                "harbor-compatibility.json"
            )
            try:
                bundle = load_compatibility_bundle(compatibility_path, prepared.request)
            except Exception as error:
                return _failed_point_result(tasks, attempts, elapsed_ms, error)
            return _PointResult(
                observations=[_task_observation(trial) for trial in bundle.trials],
                elapsed_ms=elapsed_ms,
            )
        except Exception as error:
            elapsed_ms = (time.monotonic() - started) * 1000
            _scrub_capability(point_root, capability)
            return _failed_point_result(tasks, attempts, elapsed_ms, error)
        elapsed_ms = (time.monotonic() - started) * 1000
        _scrub_capability(point_root, capability)
    if outcome.exit_code != 0 or outcome.verification is None:
        return _PointResult(
            observations=[
                _TaskObservation(
                    False,
                    elapsed_ms,
                    0,
                    0,
                    task_name,
                    f"Harbor exited with status {outcome.exit_code}",
                )
                for task_name in tasks
                for _ in range(attempts)
            ],
            elapsed_ms=elapsed_ms,
        )
    assert outcome.compatibility_path is not None
    bundle = load_compatibility_bundle(outcome.compatibility_path, prepared.request)
    observations = [_task_observation(trial) for trial in bundle.trials]
    return _PointResult(observations=observations, elapsed_ms=elapsed_ms)


def _task_observation(trial: HarborCompatibilityTrial) -> _TaskObservation:
    task_name = trial.task_name
    usage = trial.usage
    timing = trial.timing.trial
    started = _timestamp(timing.started_at)
    finished = _timestamp(timing.finished_at)
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    valid_usage = (
        isinstance(input_tokens, int)
        and not isinstance(input_tokens, bool)
        and input_tokens > 0
        and isinstance(output_tokens, int)
        and not isinstance(output_tokens, bool)
        and output_tokens > 0
    )
    valid_timing = started is not None and finished is not None and finished >= started
    latency_ms = (
        (finished - started).total_seconds() * 1000
        if valid_timing and started is not None and finished is not None
        else 0
    )
    trial_failure = trial.exception_type
    if trial_failure is None and trial.step_exceptions:
        step = trial.step_exceptions[0]
        trial_failure = f"{step.exception_type} in step {step.step_name}"
    success = trial_failure is None and valid_usage and valid_timing
    return _TaskObservation(
        success,
        latency_ms,
        input_tokens if isinstance(input_tokens, int) else 0,
        output_tokens if isinstance(output_tokens, int) else 0,
        task_name,
        (
            None
            if success
            else trial_failure or "missing positive token usage or complete timing"
        ),
    )


def _failed_point_result(
    tasks: dict[str, str], attempts: int, elapsed_ms: float, error: Exception
) -> _PointResult:
    return _PointResult(
        observations=[
            _TaskObservation(
                False,
                elapsed_ms,
                0,
                0,
                task_name,
                type(error).__name__,
            )
            for task_name in tasks
            for _ in range(attempts)
        ],
        elapsed_ms=elapsed_ms,
    )


def _request(
    plan: ProfilePlan,
    base_url: str,
    model_name: str,
    token: str,
    prompt: str,
    *,
    tools: bool,
    timeout: int,
    max_tokens: int | None = None,
    allow_length: bool = False,
) -> _SmokeObservation:
    parameters = (
        dict(plan.deployment.parameters)
        if isinstance(plan.deployment, ProviderTarget)
        else {}
    )
    payload: dict[str, JsonValue] = {
        **parameters,
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": (
            max_tokens
            if max_tokens is not None
            else min(plan.identity.max_output_tokens, 512)
        ),
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
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=max(1, timeout),
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        usage = body.get("usage") or {}
        content = message.get("content")
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        tool_calls = message.get("tool_calls") or []
        input_tokens = _integer(usage.get("prompt_tokens"))
        output_tokens = _integer(usage.get("completion_tokens"))
        saw_content = isinstance(content, str) and bool(content.strip())
        saw_reasoning = isinstance(reasoning, str) and bool(reasoning.strip())
        saw_tool_call = isinstance(tool_calls, list) and bool(tool_calls)
        finish_reason = choice.get("finish_reason")
        semantic = (
            input_tokens > 0
            and output_tokens > 0
            and (
                finish_reason in {"stop", "tool_calls"}
                or (allow_length and finish_reason == "length")
            )
            and (saw_content or saw_reasoning or saw_tool_call)
        )
        return _SmokeObservation(
            semantic,
            input_tokens,
            output_tokens,
            saw_content,
            saw_reasoning,
            saw_tool_call,
            None if semantic else "response failed semantic validation",
        )
    except Exception as error:
        return _SmokeObservation(False, 0, 0, False, False, False, type(error).__name__)


def _summarize_point(
    concurrency: int,
    observations: list[_TaskObservation],
    *,
    elapsed_ms: float,
    repetition: int,
) -> ProfilePoint:
    successful = [observation for observation in observations if observation.success]
    failed = len(observations) - len(successful)
    if not observations or not successful or elapsed_ms <= 0:
        return _failed_point(
            concurrency,
            len(observations),
            "no semantically valid benchmark trials",
            repetition=repetition,
        )
    elapsed_seconds = elapsed_ms / 1000
    latencies = [observation.latency_ms for observation in successful]
    inputs = [observation.input_tokens for observation in successful]
    outputs = [observation.output_tokens for observation in successful]
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
        "aggregate_input_tokens_per_second": sum(inputs) / elapsed_seconds,
        "aggregate_output_tokens_per_second": sum(outputs) / elapsed_seconds,
        "tasks_per_hour": len(successful) * 3600 / elapsed_seconds,
        "session_output_tokens_per_second": statistics.median(
            observation.output_tokens / max(observation.latency_ms / 1000, 0.001)
            for observation in successful
        ),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "latency_ms_p99": _percentile(latencies, 99),
        "prompt_tokens_p50": _percentile(inputs, 50),
        "prompt_tokens_p95": _percentile(inputs, 95),
        "prompt_tokens_max": max(inputs),
        "output_tokens_p50": _percentile(outputs, 50),
        "output_tokens_p95": _percentile(outputs, 95),
        "output_tokens_max": max(outputs),
        "artifact_prefix": f"points/{concurrency}/{repetition}/evidence.json",
    }
    payload["point_sha256"] = canonical_digest(payload)
    return ProfilePoint.model_validate(payload)


def _write_point(
    destination: Path,
    point: ProfilePoint,
    observations: list[_TaskObservation],
    elapsed_ms: float,
) -> None:
    write_json(
        destination / point.artifact_prefix,
        {
            "point": point.model_dump(mode="json", exclude_none=True),
            "elapsed_ms": elapsed_ms,
            "observations": [observation.__dict__ for observation in observations],
        },
    )


def _point_passes_objective(plan: ProfilePlan, point: ProfilePoint) -> bool:
    objective = plan.objective
    return (
        point.status == "completed"
        and point.error_rate <= objective.maximum_error_rate
        and (
            objective.maximum_ttft_ms_p95 is None
            or (
                point.ttft_ms_p95 is not None
                and point.ttft_ms_p95 <= objective.maximum_ttft_ms_p95
            )
        )
        and (
            objective.maximum_tpot_ms_p95 is None
            or (
                point.tpot_ms_p95 is not None
                and point.tpot_ms_p95 <= objective.maximum_tpot_ms_p95
            )
        )
        and (
            objective.minimum_session_output_tokens_per_second is None
            or (
                point.session_output_tokens_per_second is not None
                and point.session_output_tokens_per_second
                >= objective.minimum_session_output_tokens_per_second
            )
        )
    )


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
        "artifact_prefix": f"points/{concurrency}/{repetition}/evidence.json",
        "failure_reason": reason,
    }
    return ProfilePoint(point_sha256=canonical_digest(payload), **payload)


def _skipped_point(
    concurrency: int, reason: str, *, repetition: int = 1
) -> ProfilePoint:
    payload = {
        "concurrency": concurrency,
        "repetition": repetition,
        "status": "skipped",
        "planned_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "error_rate": 0.0,
        "goodput_rate": 0.0,
        "artifact_prefix": f"points/{concurrency}/{repetition}/evidence.json",
        "failure_reason": reason,
    }
    return ProfilePoint(point_sha256=canonical_digest(payload), **payload)


def _sample_tasks(plan: ProfilePlan) -> dict[str, str]:
    digests = plan.benchmark.get("task_digests")
    if not isinstance(digests, dict):
        raise ProfileWorkerError("profile benchmark has no resolved task digests")
    selected = sorted(digests)[: plan.workload.sample_task_count]
    tasks = {
        task: digest for task in selected if isinstance(digest := digests[task], str)
    }
    if len(tasks) != plan.workload.sample_task_count:
        raise ProfileWorkerError("profile benchmark task digests are malformed")
    if canonical_digest(tasks) != plan.workload.sample_tasks_sha256:
        raise ProfileWorkerError("profile benchmark sample does not match its digest")
    return tasks


def _remaining(deadline: float) -> int:
    remaining = math.floor(deadline - time.monotonic())
    if remaining < 1:
        raise ProfileWorkerError("profile time budget exhausted")
    return remaining


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _scrub_capability(root: Path, capability: str | None) -> None:
    if capability is None:
        return
    scrub_secret_paths(root, capability)
    scrub_secret(root, capability)


def _finalize_profile(root: Path, secrets: str | tuple[str, ...]) -> None:
    scrub_secret_paths(root, secrets)
    scrub_secret(root, secrets)
    assert_secret_absent(root, secrets)
    write_checksums(root)


def _redact_message(value: str, secrets: str | tuple[str, ...]) -> str:
    candidates = (secrets,) if isinstance(secrets, str) else secrets
    for secret in candidates:
        value = value.replace(secret, "[REDACTED]")
    return value


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _percentile(values: list[int] | list[float], percentile: int) -> float:
    ordered = sorted(float(value) for value in values)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    return ordered[max(0, index)]
