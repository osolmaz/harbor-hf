from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

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
from harbor_hf.harbor_adapter import (
    FilesystemHarborExecutionAdapter,
    resolve_native_trial_root,
)
from harbor_hf.harbor_adapter.errors import HarborTrialFailure
from harbor_hf.harbor_adapter.exporter import refresh_bundle_artifacts
from harbor_hf.harbor_adapter.models import (
    HarborCompatibilityBundle,
    HarborCompatibilityTrial,
)
from harbor_hf.harbor_adapter.validation import load_compatibility_bundle
from harbor_hf.hf_endpoints import HuggingFaceEndpointAdapter
from harbor_hf.judge_recorder import (
    JUDGE_RECORDER_PORT,
    JudgeEvidenceRecorder,
    JudgeExchange,
    JudgeRecorderSummary,
)
from harbor_hf.models import EndpointRef, ExperimentSpec
from harbor_hf.private_artifacts import sanitize_private_artifact_special_files
from harbor_hf.process import SubprocessRunner, run_streaming
from harbor_hf.profile_worker_transport import (
    ProfileTransport,
    job_ingress_base_url,
    wait_ready,
)
from harbor_hf.profiling import (
    ProfilePlan,
    ProfilePoint,
    ServingProfile,
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
from harbor_hf.trial_evidence import JudgeCalls, JudgeSelection, assemble_trial_evidence
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


class ProfileCleanupUnverified(ProfileWorkerError):
    """Raised when endpoint shutdown cannot be verified."""


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


@dataclass(frozen=True)
class _ProfileRecovery:
    existing_points: list[ProfilePoint]
    selected: ServingProfile | None = None


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
        sample_task_names=plan.workload.sample_task_names,
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
    recovery = _prepare_profile_destination(plan, destination)
    if recovery is None:
        return destination
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
        deadline = _profile_deadline(destination)
        require_executable("git")
        require_executable("uv")
        secrets = run_secret_values(run_lock, token)
        if recovery.selected is not None:
            _require_recovered_profile_cleanup(run_lock)
            _finalize_profile(destination, secrets)
            selected_digest = canonical_digest(recovery.selected)
            (destination / "_SELECTED").write_text(
                selected_digest + "\n", encoding="utf-8"
            )
            return destination
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
                    existing_points=recovery.existing_points,
                )
                profile = profile.model_copy(update={"points": points})
        selected = select_profile(profile)
        write_json(destination / "profile.json", selected.model_dump(mode="json"))
        selected_digest = canonical_digest(selected)
        _finalize_profile(destination, secrets)
        (destination / "_SELECTED").write_text(selected_digest + "\n", encoding="utf-8")
        return destination
    except ProfileCleanupUnverified as error:
        failure = {
            "error_type": type(error).__name__,
            "message": _redact_message(str(error), secrets),
        }
        write_json(destination / "failure.json", failure)
        _finalize_profile(destination, secrets)
        raise
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


def _prepare_profile_destination(
    plan: ProfilePlan, destination: Path
) -> _ProfileRecovery | None:
    if destination.exists():
        return _recover_profile_destination(plan, destination)
    destination.mkdir(parents=True)
    write_json(destination / "plan.json", plan.model_dump(mode="json"))
    started_at = datetime.now(UTC)
    write_json(
        destination / "lifecycle.json",
        {
            "deadline_at": (
                started_at + timedelta(seconds=plan.profile_timeout_seconds)
            ).isoformat(),
            "plan_sha256": plan.plan_sha256,
            "started_at": started_at.isoformat(),
        },
    )
    return _ProfileRecovery(existing_points=[])


def _recover_profile_destination(
    plan: ProfilePlan, destination: Path
) -> _ProfileRecovery | None:
    stored_plan_path = destination / "plan.json"
    if not stored_plan_path.is_file():
        raise ProfileWorkerError("existing profile prefix has no plan")
    stored_plan = ProfilePlan.model_validate_json(
        stored_plan_path.read_text(encoding="utf-8")
    )
    if stored_plan != plan:
        raise ProfileWorkerError("existing profile prefix belongs to another plan")
    markers = [
        marker for marker in ("_SELECTED", "_FAILED") if (destination / marker).exists()
    ]
    if len(markers) > 1:
        raise ProfileWorkerError("existing profile prefix has conflicting markers")
    profile_path = destination / "profile.json"
    if markers == ["_FAILED"]:
        raise ProfileWorkerError("existing profile prefix is terminally failed")
    if profile_path.is_file():
        return _recover_selected_profile(plan, destination, markers)
    if markers:
        raise ProfileWorkerError("selected profile marker has no profile")
    if not (destination / "lifecycle.json").is_file():
        raise ProfileWorkerError("partial profile prefix predates restart recovery")
    return _ProfileRecovery(existing_points=_load_recoverable_points(plan, destination))


def _recover_selected_profile(
    plan: ProfilePlan, destination: Path, markers: list[str]
) -> _ProfileRecovery | None:
    selected = ServingProfile.model_validate_json(
        (destination / "profile.json").read_text(encoding="utf-8")
    )
    _validate_recovered_selection(plan, selected)
    if markers == ["_SELECTED"]:
        marker = (destination / "_SELECTED").read_text(encoding="utf-8").strip()
        if marker != canonical_digest(selected):
            raise ProfileWorkerError("selected profile marker digest does not match")
        return None
    return _ProfileRecovery(existing_points=selected.points, selected=selected)


def _validate_recovered_selection(plan: ProfilePlan, profile: ServingProfile) -> None:
    unselected = new_unselected_profile(plan).model_copy(
        update={"created_at": profile.created_at, "points": profile.points}
    )
    if select_profile(unselected) != profile:
        raise ProfileWorkerError("existing selected profile does not match its plan")


def _profile_deadline(destination: Path) -> float:
    lifecycle_path = destination / "lifecycle.json"
    if not lifecycle_path.is_file():
        return time.monotonic() + 300
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    deadline_at = datetime.fromisoformat(str(lifecycle["deadline_at"]))
    remaining = (deadline_at - datetime.now(UTC)).total_seconds()
    if remaining < 1:
        raise ProfileWorkerError(
            "profile time budget exhausted before restart recovery"
        )
    return time.monotonic() + remaining


def _require_recovered_profile_cleanup(run_lock: RunLock) -> None:
    deployment = run_lock.deployment
    if isinstance(deployment, ProviderTarget):
        return
    endpoint = deployment.endpoint
    if endpoint is None:
        raise ProfileWorkerError("recovered profile endpoint binding is missing")
    try:
        baseline = EndpointManager(
            endpoint.namespace, endpoint.name, SubprocessRunner()
        ).describe()
        validate_endpoint_model(run_lock, baseline)
        require_paused_endpoint(baseline)
    except Exception as error:
        raise ProfileCleanupUnverified(
            "recovered endpoint cleanup is not verified; profile remains nonterminal"
        ) from error


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
        with (
            ProfileTransport.for_provider(
                plan.deployment,
                token=token,
                evidence_path=destination / "provider-requests.jsonl",
                deadline=deadline,
            ) as transport,
            _profile_judge_transport(run_lock, transport, token, deadline) as selected,
        ):
            yield selected
        return
    deployment = run_lock.deployment
    if isinstance(deployment, ProviderTarget):
        raise ProfileWorkerError("profile target changed while binding its endpoint")
    endpoint = deployment.endpoint
    if endpoint is None:
        raise ProfileWorkerError("profile endpoint binding is missing")
    manager = EndpointManager(endpoint.namespace, endpoint.name, SubprocessRunner())
    provisioner = (
        EndpointProvisioner(HuggingFaceEndpointAdapter(token=token))
        if desired_endpoint is not None
        else None
    )
    cleanup_required = desired_endpoint is not None
    try:
        if desired_endpoint is not None:
            assert provisioner is not None
            _prepare_managed_profile_endpoint(
                provisioner,
                desired_endpoint,
                run_lock,
                endpoint,
                token,
                deadline,
            )
        else:
            baseline = manager.describe()
            validate_endpoint_model(run_lock, baseline)
            require_paused_endpoint(baseline)
            cleanup_required = True
            launch_cleanup_watchdog(run_lock, endpoint, token)
        baseline = manager.describe()
        validate_endpoint_model(run_lock, baseline)
        require_paused_endpoint(baseline)
        base_url = resume_and_probe_endpoint(
            destination,
            destination / "events.jsonl",
            run_lock,
            manager,
            token,
            readiness_timeout_seconds=min(3600, _remaining(deadline)),
        )
        transport = ProfileTransport.for_endpoint(base_url, endpoint.served_model_name)
        with _profile_judge_transport(run_lock, transport, token, deadline) as selected:
            yield selected
    finally:
        if cleanup_required:
            _cleanup_profile_endpoint(manager, provisioner, desired_endpoint)


@contextmanager
def _profile_judge_transport(
    run_lock: RunLock,
    transport: ProfileTransport,
    token: str,
    deadline: float,
) -> Iterator[ProfileTransport]:
    if not run_lock.judge_required_tasks:
        yield transport
        return
    recorder = JudgeEvidenceRecorder(token=token, deadline=deadline)
    base_url = job_ingress_base_url(JUDGE_RECORDER_PORT)
    try:
        recorder.start(port=JUDGE_RECORDER_PORT)
        wait_ready(base_url, token, deadline)
        transport.attach_judge_recorder(recorder, base_url)
        yield transport
    finally:
        transport.detach_judge_recorder()
        recorder.close()


def _prepare_managed_profile_endpoint(
    provisioner: EndpointProvisioner,
    desired: DesiredEndpoint,
    run_lock: RunLock,
    endpoint: EndpointRef,
    token: str,
    deadline: float,
) -> None:
    existing = provisioner.inspect(desired)
    if existing is not None:
        provisioner.create_or_adopt(
            desired,
            timeout_seconds=min(900, _remaining(deadline)),
        )
        launch_cleanup_watchdog(run_lock, endpoint, token)
        return
    launch_cleanup_watchdog(run_lock, endpoint, token)
    provisioner.create_or_adopt(
        desired,
        timeout_seconds=min(900, _remaining(deadline)),
    )


def _cleanup_profile_endpoint(
    manager: EndpointManager,
    provisioner: EndpointProvisioner | None,
    desired: DesiredEndpoint | None,
) -> None:
    try:
        if (
            desired is None
            or provisioner is None
            or provisioner.inspect(desired) is not None
        ):
            manager.pause_and_verify()
    except Exception as error:
        raise ProfileCleanupUnverified(
            "endpoint cleanup is not verified; profile remains nonterminal"
        ) from error


def _verify_smoke(
    plan: ProfilePlan,
    transport: ProfileTransport,
    token: str,
    deadline: float,
) -> None:
    with transport.scope("smoke") as (base_url, model_name, capability):
        chat = _smoke_request(
            plan,
            base_url,
            model_name,
            token,
            "Reply with exactly OK.",
            tools=False,
            deadline=deadline,
        )
        if not chat.success or not (chat.saw_content or chat.saw_reasoning):
            detail = chat.error or "empty output"
            raise ProfileWorkerError(f"chat smoke failed: {detail}")
        if plan.reasoning_required and not chat.saw_reasoning:
            raise ProfileWorkerError("reasoning smoke produced no reasoning channel")
        tool = _smoke_request(
            plan,
            base_url,
            model_name,
            token,
            "Use the lookup tool to look up the value for key alpha.",
            tools=True,
            deadline=deadline,
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
        observation = _smoke_request(
            plan,
            base_url,
            model_name,
            token,
            _capacity_prompt(repeats),
            tools=False,
            deadline=deadline,
            max_tokens=1,
            allow_length=True,
            require_visible_output=False,
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
    capacity = _smoke_request(
        plan,
        base_url,
        model_name,
        token,
        _capacity_prompt(repeats),
        tools=False,
        deadline=deadline,
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


def _smoke_request(
    plan: ProfilePlan,
    base_url: str,
    model_name: str,
    token: str,
    prompt: str,
    *,
    tools: bool,
    deadline: float,
    max_tokens: int | None = None,
    allow_length: bool = False,
    require_visible_output: bool = True,
) -> _SmokeObservation:
    attempts = (
        plan.deployment.limits.max_attempts
        if isinstance(plan.deployment, ProviderTarget)
        else 1
    )
    for attempt in range(1, attempts + 1):
        observation = _request(
            plan,
            base_url,
            model_name,
            token,
            prompt,
            tools=tools,
            timeout=_remaining(deadline),
            max_tokens=max_tokens,
            allow_length=allow_length,
            require_visible_output=require_visible_output,
        )
        if observation.success or attempt == attempts:
            return observation
        delay = min(2 ** (attempt - 1), 4)
        if deadline - time.monotonic() <= delay:
            return observation
        time.sleep(delay)
    raise AssertionError("provider smoke attempts must be positive")


def _run_ladder(
    plan: ProfilePlan,
    run_lock: RunLock,
    transport: ProfileTransport,
    harbor_source: Path,
    token: str,
    destination: Path,
    deadline: float,
    *,
    existing_points: list[ProfilePoint] | None = None,
) -> list[ProfilePoint]:
    recovered = _recovered_point_map(existing_points or [])
    consumed: set[tuple[int, int]] = set()
    points: list[ProfilePoint] = []
    previous_rates: list[float] = []
    for concurrency in plan.candidate_concurrency:
        if time.monotonic() >= deadline:
            point = _skipped_point(concurrency, "profile time budget exhausted")
            points.append(point)
            _write_point(destination, point, [], 0)
            break
        point = _obtain_point(
            plan,
            run_lock,
            transport,
            harbor_source,
            token,
            concurrency,
            repetition=1,
            destination=destination,
            deadline=deadline,
            recovered=recovered,
            consumed=consumed,
        )
        points.append(point)
        if not _point_passes_objective(plan, point):
            _verify_smoke(plan, transport, token, deadline)
            retried_point = _obtain_point(
                plan,
                run_lock,
                transport,
                harbor_source,
                token,
                concurrency,
                repetition=2,
                destination=destination,
                deadline=deadline,
                recovered=recovered,
                consumed=consumed,
            )
            points.append(retried_point)
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
            recovered=recovered,
            consumed=consumed,
        )
    )
    _require_all_recovered_points_consumed(recovered, consumed)
    return points


def _recovered_point_map(
    points: list[ProfilePoint],
) -> dict[tuple[int, int], ProfilePoint]:
    recovered = {(point.concurrency, point.repetition): point for point in points}
    if len(recovered) != len(points):
        raise ProfileWorkerError("recovered profile contains duplicate points")
    return recovered


def _require_all_recovered_points_consumed(
    recovered: dict[tuple[int, int], ProfilePoint],
    consumed: set[tuple[int, int]],
) -> None:
    unused = sorted(set(recovered) - consumed)
    if unused:
        raise ProfileWorkerError(f"recovered profile has unreachable points: {unused}")


def _load_recoverable_points(
    plan: ProfilePlan, destination: Path
) -> list[ProfilePoint]:
    points: list[ProfilePoint] = []
    for evidence_path in sorted((destination / "points").glob("*/*/evidence.json")):
        try:
            concurrency = int(evidence_path.parent.parent.name)
            repetition = int(evidence_path.parent.name)
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            point = ProfilePoint.model_validate(evidence["point"])
            observations = [
                _TaskObservation(**observation)
                for observation in evidence.get("observations", [])
            ]
            elapsed_ms = float(evidence.get("elapsed_ms", 0))
        except (KeyError, TypeError, ValueError) as error:
            raise ProfileWorkerError(
                f"recovered point evidence is malformed: {evidence_path}"
            ) from error
        expected_prefix = f"points/{concurrency}/{repetition}/evidence.json"
        if (
            concurrency not in plan.candidate_concurrency
            or point.concurrency != concurrency
            or point.repetition != repetition
            or point.artifact_prefix != expected_prefix
        ):
            raise ProfileWorkerError(
                f"recovered point identity does not match its path: {evidence_path}"
            )
        payload = point.model_dump(
            mode="json", exclude={"point_sha256"}, exclude_none=True
        )
        if canonical_digest(payload) != point.point_sha256:
            raise ProfileWorkerError(
                f"recovered point digest does not match: {evidence_path}"
            )
        if point.status == "skipped":
            if observations or elapsed_ms != 0:
                raise ProfileWorkerError(
                    f"recovered skipped point has observations: {evidence_path}"
                )
        else:
            summarized = _summarize_point(
                concurrency,
                observations,
                elapsed_ms=elapsed_ms,
                repetition=repetition,
            )
            if summarized != point:
                raise ProfileWorkerError(
                    f"recovered point does not match raw observations: {evidence_path}"
                )
        points.append(point)
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
    *,
    recovered: dict[tuple[int, int], ProfilePoint] | None = None,
    consumed: set[tuple[int, int]] | None = None,
) -> list[ProfilePoint]:
    recovered = recovered or {}
    consumed = consumed if consumed is not None else set()
    points: list[ProfilePoint] = []
    for concurrency in boundary:
        concurrency_points = [
            point for point in existing_points if point.concurrency == concurrency
        ]
        eligible_count = sum(
            _point_passes_objective(plan, point) for point in concurrency_points
        )
        repetition = (
            max((point.repetition for point in concurrency_points), default=0) + 1
        )
        while eligible_count < plan.workload.boundary_repetitions:
            if time.monotonic() >= deadline:
                point = _skipped_point(
                    concurrency,
                    "profile time budget exhausted during boundary repetition",
                    repetition=repetition,
                )
                points.append(point)
                _write_point(destination, point, [], 0)
                break
            point = _obtain_point(
                plan,
                run_lock,
                transport,
                harbor_source,
                token,
                concurrency,
                repetition=repetition,
                destination=destination,
                deadline=deadline,
                recovered=recovered,
                consumed=consumed,
            )
            points.append(point)
            if not _point_passes_objective(plan, point):
                break
            eligible_count += 1
            repetition += 1
    return points


def _obtain_point(
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
    recovered: dict[tuple[int, int], ProfilePoint],
    consumed: set[tuple[int, int]],
) -> ProfilePoint:
    key = (concurrency, repetition)
    if point := recovered.get(key):
        consumed.add(key)
        return point
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
    _write_point(destination, point, result.observations, result.elapsed_ms)
    return point


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
    with tempfile.TemporaryDirectory(prefix="harbor-hf-profile-point-") as temporary:
        point_root = Path(temporary) / "point"
        result = _run_staged_point(
            plan,
            run_lock,
            transport,
            harbor_source,
            token,
            concurrency,
            repetition=repetition,
            point_root=point_root,
            deadline=deadline,
        )
        archived = destination / "points" / str(concurrency) / str(repetition)
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(point_root, archived)
        return result


def _run_staged_point(
    plan: ProfilePlan,
    run_lock: RunLock,
    transport: ProfileTransport,
    harbor_source: Path,
    token: str,
    concurrency: int,
    *,
    repetition: int,
    point_root: Path,
    deadline: float,
) -> _PointResult:
    tasks, attempts = _point_workload(plan, concurrency)
    jobs_dir = point_root / "harbor-jobs"
    execution_root = point_root / "harbor-execution"
    jobs_dir.mkdir(parents=True)
    execution_root.mkdir()
    adapter = FilesystemHarborExecutionAdapter()
    scope = f"c{concurrency}-r{repetition}"
    compatibility_path: Path | None = None
    outcome = None
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
            extra_environment_hosts=_profile_judge_hosts(transport),
        )
        timeout = _remaining(deadline)
        started = time.monotonic()
        judge_capability: str | None = None
        try:
            with (
                _profile_point_judge_scope(
                    plan,
                    run_lock,
                    transport,
                    execution_root,
                    scope,
                    tasks,
                    attempts,
                    capability,
                ) as (judge_api_url, judge_capability),
                harbor_process_environment(
                    run_lock,
                    token=token,
                    inference_base_url=base_url,
                    judge_api_url=judge_api_url,
                    redaction_secrets=tuple(
                        value for value in (capability, judge_capability) if value
                    ),
                ) as environment,
            ):
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
            compatibility_path = outcome.compatibility_path
        except HarborTrialFailure:
            compatibility_path = prepared.request_path.with_name(
                "harbor-compatibility.json"
            )
        except Exception as error:
            elapsed_ms = (time.monotonic() - started) * 1000
            _scrub_capability(point_root, capability)
            _scrub_capability(point_root, judge_capability)
            return _failed_point_result(tasks, attempts, elapsed_ms, error)
        elapsed_ms = (time.monotonic() - started) * 1000
        _scrub_capability(point_root, capability)
        _scrub_capability(point_root, judge_capability)
    if outcome is not None and (outcome.exit_code != 0 or outcome.verification is None):
        return _failed_point_result(
            tasks,
            attempts,
            elapsed_ms,
            ProfileWorkerError(f"Harbor exited with status {outcome.exit_code}"),
        )
    if compatibility_path is None:
        return _failed_point_result(
            tasks,
            attempts,
            elapsed_ms,
            ProfileWorkerError("Harbor produced no compatibility bundle"),
        )
    try:
        bundle = load_compatibility_bundle(compatibility_path, prepared.request)
        _assemble_profile_point_evidence(
            plan,
            run_lock,
            bundle,
            jobs_dir,
            compatibility_path,
            execution_root,
            scope,
            token,
            capability,
            judge_capability,
        )
    except Exception as error:
        return _failed_point_result(tasks, attempts, elapsed_ms, error)
    observations = [_task_observation(trial) for trial in bundle.trials]
    return _PointResult(observations=observations, elapsed_ms=elapsed_ms)


def _profile_judge_hosts(transport: ProfileTransport) -> tuple[str, ...]:
    base_url = getattr(transport, "judge_base_url", None)
    if base_url is None:
        return ()
    host = urlparse(base_url).hostname
    if not host:
        raise ProfileWorkerError("profile judge recorder hostname is invalid")
    return (host,)


@contextmanager
def _profile_point_judge_scope(
    plan: ProfilePlan,
    run_lock: RunLock,
    transport: ProfileTransport,
    execution_root: Path,
    scope: str,
    tasks: dict[str, str],
    attempts: int,
    provider_capability: str | None,
) -> Iterator[tuple[str | None, str | None]]:
    judged_tasks = set(tasks) & set(run_lock.judge_required_tasks or [])
    if not judged_tasks:
        yield None, None
        return
    recorder = transport.judge_recorder
    base_url = transport.judge_base_url
    policy = run_lock.trial_evidence
    judge = run_lock.benchmark_judge
    if recorder is None or base_url is None or policy is None or judge is None:
        raise ProfileWorkerError("judged profile transport is incomplete")
    maximum_calls = attempts * len(judged_tasks) * policy.judge_max_calls_per_execution
    if maximum_calls > 4096:
        raise ProfileWorkerError("judged profile point exceeds recorder call capacity")
    identity = f"profile-{plan.profile_id}-{scope}"
    capability = recorder.register_scope(
        execution_id=identity,
        trial_id=identity,
        model=judge.model,
        destination=execution_root / "judge-records",
        policy=policy,
        known_secrets=((provider_capability,) if provider_capability else ()),
        max_calls=maximum_calls,
    )
    try:
        yield recorder.scoped_url(base_url, capability), capability
    finally:
        recorder.revoke_scope(capability)


def _assemble_profile_point_evidence(
    plan: ProfilePlan,
    run_lock: RunLock,
    bundle: HarborCompatibilityBundle,
    jobs_dir: Path,
    compatibility_path: Path,
    execution_root: Path,
    scope: str,
    token: str,
    provider_capability: str | None,
    judge_capability: str | None,
) -> None:
    policy = run_lock.trial_evidence
    if policy is None:
        raise ProfileWorkerError("profile trial evidence policy is missing")
    run_secrets = run_secret_values(run_lock, token)
    base_secrets = (run_secrets,) if isinstance(run_secrets, str) else run_secrets
    secrets = tuple(
        value
        for value in (*base_secrets, provider_capability, judge_capability)
        if value
    )
    judge_records = _split_profile_judge_records(
        plan, run_lock, bundle, jobs_dir, execution_root, scope
    )
    logical_attempts: dict[str, int] = {}
    for native in bundle.trials:
        native_root = resolve_native_trial_root(jobs_dir, native.path)
        logical_attempt = logical_attempts.get(native.task_name, 0) + 1
        logical_attempts[native.task_name] = logical_attempt
        snapshot = native_root / "artifacts" / "workspace" / "app"
        if native.exception_type is not None:
            shutil.rmtree(snapshot, ignore_errors=True)
            continue
        redacted = scrub_secret(native_root, secrets, allow_symlinks=True)
        if redacted:
            write_json(
                native_root / "evidence-redactions.json",
                {"files": redacted},
            )
        assert_secret_absent(native_root, secrets, allow_symlinks=True)
        judge_expected = native.task_name in (run_lock.judge_required_tasks or [])
        assemble_trial_evidence(
            native_root,
            campaign_id=None,
            run_id=f"profile-{plan.profile_id}",
            execution_id=f"profile-{plan.profile_id}-{scope}-{native.trial_id}",
            trial_id=native.trial_id,
            task_name=native.task_name,
            task_digest=native.task_digest,
            logical_attempt=logical_attempt,
            physical_attempt=1,
            judge_expected=judge_expected,
            judge_model=(
                run_lock.benchmark_judge.model
                if judge_expected and run_lock.benchmark_judge is not None
                else None
            ),
            policy=policy,
            judge_records_dir=judge_records.get(native.trial_id),
            known_secrets=secrets,
        )
    refresh_bundle_artifacts(jobs_dir, compatibility_path)


def _split_profile_judge_records(
    plan: ProfilePlan,
    run_lock: RunLock,
    bundle: HarborCompatibilityBundle,
    jobs_dir: Path,
    execution_root: Path,
    scope: str,
) -> dict[str, Path]:
    judged = [
        trial
        for trial in bundle.trials
        if trial.task_name in (run_lock.judge_required_tasks or [])
    ]
    if not judged:
        return {}
    aggregate = execution_root / "judge-records"
    summary = _profile_judge_summary(aggregate)
    assigned = _profile_judge_assignments(judged, jobs_dir)
    zero_call_trials = _profile_zero_call_trials(judged, jobs_dir)
    exchange_dirs = {
        path.name: path
        for path in aggregate.glob("judge-*")
        if path.is_dir() and not path.is_symlink()
    }
    if set(exchange_dirs) != set(assigned) or summary.exchange_count != len(assigned):
        raise ProfileWorkerError("profile judge exchanges cannot be assigned exactly")
    policy = run_lock.trial_evidence
    if policy is None:
        raise ProfileWorkerError("profile trial evidence policy is missing")
    _validate_profile_assignment_counts(assigned, policy.judge_max_calls_per_execution)
    grouped: dict[str, list[str]] = {}
    destinations: dict[str, Path] = {}
    for exchange_id, native in assigned.items():
        destination = execution_root / "profile-judge-records" / native.trial_id
        destination.mkdir(parents=True, exist_ok=True)
        exchange_dir = exchange_dirs[exchange_id]
        metadata_path = exchange_dir / "exchange.json"
        exchange = JudgeExchange.model_validate_json(metadata_path.read_text())
        execution_id = f"profile-{plan.profile_id}-{scope}-{native.trial_id}"
        exchange = exchange.model_copy(
            update={"execution_id": execution_id, "trial_id": native.trial_id}
        )
        write_json(metadata_path, exchange.model_dump(mode="json"))
        shutil.move(str(exchange_dir), destination / exchange_id)
        grouped.setdefault(native.trial_id, []).append(exchange_id)
        destinations[native.trial_id] = destination
    for native in zero_call_trials:
        destination = execution_root / "profile-judge-records" / native.trial_id
        destination.mkdir(parents=True)
        destinations[native.trial_id] = destination
        grouped[native.trial_id] = []
    for trial_id, exchange_ids in grouped.items():
        execution_id = f"profile-{plan.profile_id}-{scope}-{trial_id}"
        trial_summary = JudgeRecorderSummary(
            execution_id=execution_id,
            trial_id=trial_id,
            model=summary.model,
            exchange_count=len(exchange_ids),
            rejected_call_count=0,
            closed_at=summary.closed_at,
        )
        write_json(
            destinations[trial_id] / "recorder.json",
            trial_summary.model_dump(mode="json"),
        )
    (aggregate / "recorder.json").unlink()
    aggregate.rmdir()
    return destinations


def _validate_profile_assignment_counts(
    assigned: dict[str, HarborCompatibilityTrial], maximum_calls: int
) -> None:
    counts: dict[str, int] = {}
    for native in assigned.values():
        counts[native.trial_id] = counts.get(native.trial_id, 0) + 1
    if any(count > maximum_calls for count in counts.values()):
        raise ProfileWorkerError("profile trial exceeds its judge call limit")


def _profile_judge_summary(aggregate: Path) -> JudgeRecorderSummary:
    try:
        summary = JudgeRecorderSummary.model_validate_json(
            (aggregate / "recorder.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ProfileWorkerError("profile judge recorder summary is invalid") from error
    if summary.rejected_call_count:
        raise ProfileWorkerError("profile judge recorder rejected a call")
    return summary


def _profile_judge_assignments(
    judged: list[HarborCompatibilityTrial], jobs_dir: Path
) -> dict[str, HarborCompatibilityTrial]:
    assigned: dict[str, HarborCompatibilityTrial] = {}
    for native in judged:
        if native.exception_type is not None:
            raise ProfileWorkerError("judged profile trial did not complete")
        native_root = resolve_native_trial_root(jobs_dir, native.path)
        verifier = native_root / "verifier"
        selection_path = verifier / "judge-selection.json"
        calls_path = verifier / "judge-calls.json"
        if not selection_path.exists() and not calls_path.exists():
            continue
        try:
            selection = JudgeSelection.model_validate_json(
                (verifier / "judge-selection.json").read_text(encoding="utf-8")
            )
            calls = (
                JudgeCalls.model_validate_json(calls_path.read_text(encoding="utf-8"))
                if calls_path.is_file()
                else JudgeCalls(exchange_ids=[selection.exchange_id])
            )
        except (OSError, ValueError) as error:
            raise ProfileWorkerError(
                "profile judge call assignment is invalid"
            ) from error
        if selection.exchange_id not in calls.exchange_ids:
            raise ProfileWorkerError("profile judge selection is not a recorded call")
        for exchange_id in calls.exchange_ids:
            if exchange_id in assigned:
                raise ProfileWorkerError("profile judge exchange was assigned twice")
            assigned[exchange_id] = native
    return assigned


def _profile_zero_call_trials(
    judged: list[HarborCompatibilityTrial], jobs_dir: Path
) -> list[HarborCompatibilityTrial]:
    return [
        native
        for native in judged
        if not (
            resolve_native_trial_root(jobs_dir, native.path)
            / "verifier"
            / "judge-selection.json"
        ).exists()
        and not (
            resolve_native_trial_root(jobs_dir, native.path)
            / "verifier"
            / "judge-calls.json"
        ).exists()
    ]


def _point_workload(plan: ProfilePlan, concurrency: int) -> tuple[dict[str, str], int]:
    tasks = _sample_tasks(plan)
    minimum = max(plan.workload.minimum_observations_per_point, 2 * concurrency)
    if isinstance(plan.deployment, ProviderTarget):
        if len(tasks) < minimum:
            raise ProfileWorkerError(
                "provider profile lacks distinct tasks for this concurrency"
            )
        return tasks, 1
    return tasks, math.ceil(minimum / len(tasks))


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
    require_visible_output: bool = True,
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
            and (
                not require_visible_output
                or saw_content
                or saw_reasoning
                or saw_tool_call
            )
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
    raw_digests = plan.benchmark.get("task_digests")
    if not isinstance(raw_digests, dict):
        raise ProfileWorkerError("profile benchmark has no resolved task digests")
    digests = cast(dict[str, object], raw_digests)
    selected = plan.workload.sample_task_names
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
    _scrub_mounted_evidence(root, capability, write_manifest=False, allow_symlinks=True)


def _finalize_profile(root: Path, secrets: str | tuple[str, ...]) -> None:
    _scrub_mounted_evidence(root, secrets, write_manifest=True, allow_symlinks=False)


def _scrub_mounted_evidence(
    root: Path,
    secrets: str | tuple[str, ...],
    *,
    write_manifest: bool,
    allow_symlinks: bool,
) -> None:
    attempts = 6
    for attempt in range(1, attempts + 1):
        try:
            sanitize_private_artifact_special_files(root)
            scrub_secret_paths(root, secrets, allow_symlinks=allow_symlinks)
            scrub_secret(root, secrets, allow_symlinks=allow_symlinks)
            assert_secret_absent(root, secrets, allow_symlinks=allow_symlinks)
            if write_manifest:
                write_checksums(root)
            return
        except FileNotFoundError:
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 16))


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
