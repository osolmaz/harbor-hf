from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from typer.testing import CliRunner

from harbor_hf.cli import app
from harbor_hf.endpoints import bind_endpoint
from harbor_hf.harbor_adapter.errors import HarborTrialFailure
from harbor_hf.harbor_adapter.models import HarborCompatibilityTrial
from harbor_hf.models import (
    DeploymentProfile,
    EndpointRef,
    ExperimentSpec,
    ServingProfileBinding,
)
from harbor_hf.profile_preflight import preflight_profile_plan
from harbor_hf.profile_submission import build_profile_submit_command
from harbor_hf.profile_worker import (
    ProfileWorkerError,
    _point_ladder_rate,
    _PointResult,
    _request,
    _run_ladder,
    _run_point,
    _summarize_point,
    _TaskObservation,
    run_profile_worker,
)
from harbor_hf.profiling import (
    ProfilePlan,
    ProfilePoint,
    bind_profile_target,
    build_profile_plan,
    canonical_digest,
    new_unselected_profile,
    select_profile,
)
from harbor_hf.provider_models import (
    ExplicitProviderRoute,
    ProviderLimits,
    ProviderTarget,
)

runner = CliRunner()


def profiled_spec(spec: ExperimentSpec) -> ExperimentSpec:
    return spec.model_copy(
        update={
            "execution": spec.execution.model_copy(
                update={
                    "server_context_tokens": 65_536,
                    "max_output_tokens": 8192,
                    "reasoning_required": True,
                }
            )
        }
    )


def plan(spec: ExperimentSpec) -> ProfilePlan:
    return build_profile_plan(
        profiled_spec(spec),
        profile_id="profile-one",
        candidate_concurrency=[1, 2, 4, 8],
        max_spend_usd="10.00",
        profile_timeout_seconds=3600,
    )


def point(concurrency: int, throughput: float) -> ProfilePoint:
    payload = {
        "concurrency": concurrency,
        "repetition": 1,
        "status": "completed",
        "planned_count": 8,
        "completed_count": 8,
        "failed_count": 0,
        "error_rate": 0.0,
        "goodput_rate": 1.0,
        "aggregate_output_tokens_per_second": float(throughput),
        "tasks_per_hour": float(throughput),
        "artifact_prefix": f"points/{concurrency}/1.json",
    }
    return ProfilePoint.model_validate(
        {"point_sha256": canonical_digest(payload), **payload}
    )


def failed_repetition(concurrency: int, repetition: int) -> ProfilePoint:
    payload = {
        "concurrency": concurrency,
        "repetition": repetition,
        "status": "failed",
        "planned_count": 8,
        "completed_count": 0,
        "failed_count": 8,
        "error_rate": 1.0,
        "goodput_rate": 0.0,
        "artifact_prefix": f"points/{concurrency}/{repetition}/evidence.json",
        "failure_reason": "failed",
    }
    return ProfilePoint.model_validate(
        {"point_sha256": canonical_digest(payload), **payload}
    )


def run_expected_profile(
    plan_path: Path,
    output_root: Path,
    *,
    smoke_fails: bool,
) -> None:
    if smoke_fails:
        with pytest.raises(ProfileWorkerError, match="smoke failed"):
            run_profile_worker(plan_path, output_root)
        destination = output_root / "serving-profiles/profile-one"
        assert (destination / "_FAILED").is_file()
        checksums = json.loads((destination / "checksums.json").read_text())
        assert set(checksums) == {"failure.json", "plan.json"}
        return
    destination = run_profile_worker(plan_path, output_root)
    assert (destination / "_SELECTED").is_file()
    checksums = json.loads((destination / "checksums.json").read_text())
    assert set(checksums) == {"plan.json", "profile.json"}


def test_profile_plan_is_deterministic(remote_spec: ExperimentSpec) -> None:
    first = plan(remote_spec)
    second = plan(remote_spec)

    assert first == second
    assert first.identity.server_context_tokens == 65_536
    assert first.workload.sample_task_count == 1
    assert first.plan_sha256.startswith("sha256:")


def test_profile_selection_prefers_lower_concurrency_on_tie(
    remote_spec: ExperimentSpec,
) -> None:
    profile = new_unselected_profile(plan(remote_spec)).model_copy(
        update={"points": [point(1, 20), point(2, 20), point(4, 10)]}
    )

    selected = select_profile(profile)

    assert selected.selection is not None
    assert selected.selection.concurrency == 1


def test_profile_selection_rejects_tampered_point(remote_spec: ExperimentSpec) -> None:
    profile = new_unselected_profile(plan(remote_spec)).model_copy(
        update={
            "points": [
                point(1, 20).model_copy(
                    update={"aggregate_output_tokens_per_second": 200}
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="point digest"):
        select_profile(profile)


def test_serving_profile_rejects_point_outside_candidate_ladder(
    remote_spec: ExperimentSpec,
) -> None:
    profile = new_unselected_profile(plan(remote_spec))

    with pytest.raises(ValueError, match="points must be in the candidate ladder"):
        type(profile).model_validate(
            profile.model_copy(update={"points": [point(16, 20)]}).model_dump()
        )


def test_maximum_goodput_does_not_discount_failed_tasks_twice(
    remote_spec: ExperimentSpec,
) -> None:
    resolved = new_unselected_profile(plan(remote_spec))

    def measured(
        concurrency: int, tasks_per_hour: float, goodput: float
    ) -> ProfilePoint:
        payload = point(concurrency, tasks_per_hour).model_dump(
            mode="json", exclude={"point_sha256"}, exclude_none=True
        )
        payload.update(
            error_rate=1 - goodput,
            goodput_rate=goodput,
            tasks_per_hour=float(tasks_per_hour),
        )
        return ProfilePoint.model_validate(
            {"point_sha256": canonical_digest(payload), **payload}
        )

    profile = resolved.model_copy(
        update={
            "objective": resolved.objective.model_copy(
                update={"maximum_error_rate": 0.5}
            ),
            "points": [measured(1, 80, 1.0), measured(2, 100, 0.5)],
        }
    )

    selected = select_profile(profile)

    assert selected.selection is not None
    assert selected.selection.concurrency == 2


def test_profile_selection_disqualifies_failed_boundary_repetition(
    remote_spec: ExperimentSpec,
) -> None:
    first = point(1, 10)
    second = point(2, 20)
    repeated = second.model_copy(
        update={
            "repetition": 2,
            "point_sha256": canonical_digest(
                second.model_dump(
                    mode="json",
                    exclude={"point_sha256"},
                    exclude_none=True,
                )
                | {"repetition": 2}
            ),
        }
    )
    profile = new_unselected_profile(plan(remote_spec)).model_copy(
        update={
            "points": [first, second, repeated, failed_repetition(2, 3)],
        }
    )

    selected = select_profile(profile)

    assert selected.selection is not None
    assert selected.selection.concurrency == 1


def test_point_throughput_uses_complete_wall_time() -> None:
    observations = [
        _TaskObservation(True, 1000, 10, 20, f"task-{index}") for index in range(8)
    ]

    result = _summarize_point(1, observations, elapsed_ms=8000, repetition=1)

    assert result.tasks_per_hour == 3600
    assert result.aggregate_output_tokens_per_second == 20
    assert result.ttft_ms_p95 is None
    assert result.tpot_ms_p95 is None


def compatibility_trial(
    task_name: str, *, exception_type: str | None = None
) -> HarborCompatibilityTrial:
    digest = "sha256:" + "1" * 64
    return HarborCompatibilityTrial.model_validate(
        {
            "path": f"job/{task_name}",
            "trial_id": task_name,
            "trial_name": task_name,
            "lock_digest": digest,
            "result_digest": digest,
            "task_name": task_name,
            "task_digest": digest,
            "agent_name": "openclaw",
            "agent_version": "1",
            "exception_type": exception_type,
            "step_exceptions": [],
            "rewards": {"reward": 0},
            "timing": {
                "trial": {
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "finished_at": "2026-01-01T00:00:01+00:00",
                }
            },
            "usage": {
                "input_tokens": 10,
                "cache_tokens": 0,
                "output_tokens": 5,
                "cost_usd": 0,
            },
            "artifacts": [],
        }
    )


def test_profile_point_preserves_individual_harbor_trial_failures(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = plan(remote_spec)
    request_path = tmp_path / "execution" / "harbor-request.json"
    request_path.parent.mkdir()
    prepared = SimpleNamespace(request_path=request_path, request=object())

    class Adapter:
        def prepare(self, *_args: object, **_kwargs: object) -> object:
            return prepared

        def execute(self, *_args: object, **_kwargs: object) -> object:
            raise HarborTrialFailure("one failed", "SandboxError")

    class Transport:
        @contextmanager
        def scope(self, _scope: str) -> Iterator[tuple[str, str, None]]:
            yield "https://endpoint.test", "model", None

    @contextmanager
    def process_environment(
        *_args: object, **_kwargs: object
    ) -> Iterator[dict[str, str]]:
        yield {}

    monkeypatch.setattr(
        "harbor_hf.profile_worker.FilesystemHarborExecutionAdapter", Adapter
    )
    monkeypatch.setattr(
        "harbor_hf.profile_worker.harbor_process_environment", process_environment
    )
    monkeypatch.setattr(
        "harbor_hf.profile_worker._sample_tasks",
        lambda _plan: {"success": "digest", "failure": "digest"},
    )
    monkeypatch.setattr(
        "harbor_hf.profile_worker.load_compatibility_bundle",
        lambda *_args: SimpleNamespace(
            trials=[
                compatibility_trial("success"),
                compatibility_trial("failure", exception_type="SandboxError"),
            ]
        ),
    )

    result = _run_point(
        resolved,
        cast(Any, object()),
        cast(Any, Transport()),
        tmp_path / "harbor",
        "hf_test",
        1,
        repetition=1,
        destination=tmp_path / "profile",
        deadline=10**12,
    )

    assert [observation.success for observation in result.observations] == [True, False]
    assert result.observations[1].error == "SandboxError"


def test_profile_ladder_skips_repetition_used_by_health_retry(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = plan(remote_spec).model_copy(update={"candidate_concurrency": [1]})
    repetitions: list[int] = []

    def run_point(*_args: object, repetition: int, **_kwargs: object) -> _PointResult:
        repetitions.append(repetition)
        success = repetition != 1
        observations = [
            _TaskObservation(success, 1000, 10, 20, f"task-{index}")
            for index in range(8)
        ]
        return _PointResult(observations, 8000)

    monkeypatch.setattr("harbor_hf.profile_worker._run_point", run_point)
    monkeypatch.setattr("harbor_hf.profile_worker._verify_smoke", lambda *_args: None)
    monkeypatch.setattr("harbor_hf.profile_worker._write_point", lambda *_args: None)

    _run_ladder(
        resolved,
        cast(Any, None),
        cast(Any, None),
        tmp_path,
        "token",
        tmp_path,
        float("inf"),
    )

    assert repetitions == [1, 2, 3]


def test_profile_ladder_uses_selected_objective_metric(
    remote_spec: ExperimentSpec,
) -> None:
    resolved = plan(remote_spec)
    throughput = resolved.model_copy(
        update={
            "objective": resolved.objective.model_copy(
                update={"kind": "maximum_throughput"}
            )
        }
    )
    measured = point(1, 100).model_copy(
        update={"aggregate_output_tokens_per_second": 200.0}
    )

    assert _point_ladder_rate(throughput, measured) == 200.0
    stable = resolved.model_copy(
        update={
            "objective": resolved.objective.model_copy(
                update={"kind": "maximum_stable_concurrency"}
            )
        }
    )
    assert _point_ladder_rate(stable, measured) is None


def test_endpoint_smoke_does_not_forward_endpoint_settings(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = plan(remote_spec)
    captured: dict[str, object] = {}

    def post(*_args: object, **kwargs: object) -> httpx.Response:
        payload = kwargs["json"]
        assert isinstance(payload, dict)
        for key, value in payload.items():
            assert isinstance(key, str)
            captured[key] = value
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
            request=httpx.Request("POST", "https://endpoint.test"),
        )

    monkeypatch.setattr("harbor_hf.profile_worker.httpx.post", post)

    observation = _request(
        resolved,
        "https://endpoint.test",
        "model",
        "token",
        "OK",
        tools=False,
        timeout=10,
    )

    assert observation.success
    assert "min_replicas" not in captured
    assert "health_route" not in captured


def test_serving_profile_binding_fails_closed_on_concurrency(
    remote_spec: ExperimentSpec,
) -> None:
    resolved = plan(remote_spec)
    binding = ServingProfileBinding(
        profile_id=resolved.profile_id,
        profile_sha256="sha256:" + "9" * 64,
        artifact_uri="hf://buckets/osolmaz/benchmark-runs/serving-profiles/profile-one/profile.json",
        concurrency=2,
        **resolved.identity.model_dump(mode="python"),
    )
    spec = profiled_spec(remote_spec)
    execution = spec.execution.model_copy(update={"serving_profile": binding})

    with pytest.raises(ValueError, match="concurrent_trials"):
        ExperimentSpec.model_validate(
            spec.model_copy(update={"execution": execution}).model_dump(mode="python")
        )


def test_managed_endpoint_binding_preserves_profile_identity(
    remote_spec: ExperimentSpec,
) -> None:
    deployment = remote_spec.matrix.deployments[0].model_copy(update={"endpoint": None})
    spec = profiled_spec(
        remote_spec.model_copy(
            update={
                "matrix": remote_spec.matrix.model_copy(
                    update={"deployments": [deployment]}
                )
            }
        )
    )
    resolved = build_profile_plan(
        spec,
        profile_id="profile-one",
        candidate_concurrency=[1],
        max_spend_usd="5.00",
        profile_timeout_seconds=3600,
    )
    binding = ServingProfileBinding(
        profile_id=resolved.profile_id,
        profile_sha256="sha256:" + "9" * 64,
        artifact_uri=(
            "hf://buckets/osolmaz/benchmark-runs/serving-profiles/"
            "profile-one/profile.json"
        ),
        concurrency=1,
        **resolved.identity.model_dump(mode="python"),
    )
    profiled = ExperimentSpec.model_validate(
        spec.model_copy(
            update={
                "execution": spec.execution.model_copy(
                    update={"serving_profile": binding}
                )
            }
        ).model_dump(mode="python")
    )

    bound = bind_endpoint(
        profiled,
        deployment_id=deployment.id,
        endpoint=EndpointRef(
            namespace="osolmaz",
            name="managed-profile-endpoint",
            served_model_name="/repository",
        ),
    )

    assert ExperimentSpec.model_validate(bound.model_dump(mode="python")) == bound


def test_profile_submit_command_is_remote_only(remote_spec: ExperimentSpec) -> None:
    command = build_profile_submit_command(
        plan(remote_spec), input_dir="hf://buckets/input", bucket="osolmaz/results"
    )

    assert command[:3] == ["hf", "jobs", "run"]
    assert "profile-worker" in command
    assert "/input/plan.json" in command
    assert not any("llama-server" in argument for argument in command)


def test_provider_profile_submit_command_exposes_recorder(
    remote_spec: ExperimentSpec,
) -> None:
    model = remote_spec.matrix.models[0]
    provider = ProviderTarget(
        id="provider",
        model=model.repo,
        routing=ExplicitProviderRoute(provider="fireworks-ai"),
        limits=ProviderLimits(max_concurrent_requests=2),
    )
    spec = profiled_spec(
        remote_spec.model_copy(
            update={
                "matrix": remote_spec.matrix.model_copy(
                    update={"deployments": [provider]}
                )
            }
        )
    )
    command = build_profile_submit_command(
        build_profile_plan(
            spec,
            profile_id="provider-profile",
            candidate_concurrency=[1, 2],
            max_spend_usd="10.00",
            profile_timeout_seconds=3600,
        ),
        input_dir="hf://buckets/input",
        bucket="osolmaz/results",
    )

    expose = command.index("--expose")
    assert command[expose : expose + 2] == ["--expose", "8000"]


def test_profile_without_endpoint_gets_deterministic_managed_binding(
    remote_spec: ExperimentSpec,
) -> None:
    deployment = remote_spec.matrix.deployments[0].model_copy(update={"endpoint": None})
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={"deployments": [deployment]}
            )
        }
    )
    resolved = plan(spec)

    first, desired = bind_profile_target(resolved)
    second, repeated = bind_profile_target(resolved)

    assert desired is not None
    assert desired == repeated
    assert first == second
    bound = first.matrix.deployments[0]
    assert isinstance(bound, DeploymentProfile)
    assert bound.endpoint is not None
    assert bound.endpoint.name == desired.identity.name
    command = build_profile_submit_command(
        resolved, input_dir="hf://buckets/input", bucket="osolmaz/results"
    )
    assert "harbor-hf-endpoint=" in " ".join(command)


class FakeApi:
    def model_info(self, repo: str, **kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(sha="a" * 40, inference_provider_mapping={})

    def bucket_info(self, bucket_id: str) -> object:
        del bucket_id
        return SimpleNamespace(private=True)

    def list_repo_files(self, *_args: object, **_kwargs: object) -> list[str]:
        return []


class FakeProviderApi(FakeApi):
    def model_info(self, repo: str, **kwargs: object) -> object:
        del repo, kwargs
        return SimpleNamespace(
            sha="a" * 40,
            inference_provider_mapping=[
                SimpleNamespace(provider="fireworks-ai", status="live")
            ],
        )


def test_endpoint_preflight_reports_quota_and_cost(remote_spec: ExperimentSpec) -> None:
    spec = profiled_spec(remote_spec)
    model = spec.matrix.models[0].model_copy(update={"revision": "a" * 40})
    spec = spec.model_copy(
        update={
            "matrix": spec.matrix.model_copy(update={"models": [model]}),
        }
    )
    resolved = build_profile_plan(
        spec,
        profile_id="profile-one",
        candidate_concurrency=[1, 2],
        max_spend_usd="5.00",
        profile_timeout_seconds=3600,
    )
    deployment = resolved.deployment
    assert isinstance(deployment, DeploymentProfile)
    response = {
        "vendors": [
            {
                "name": "aws",
                "regions": [
                    {
                        "name": deployment.region.removeprefix("aws-"),
                        "computes": [
                            {
                                "instanceType": deployment.hardware,
                                "numAccelerators": deployment.accelerator_count,
                                "status": "available",
                                "pricePerHour": 5.0,
                                "quota": {
                                    "maxAccelerators": 2,
                                    "usedAccelerators": 0,
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=response)
        )
    )

    report = preflight_profile_plan(
        resolved, api=FakeApi(), client=client, token="hf_test"
    )

    assert report.available_accelerators == 2
    assert str(report.estimated_cost_usd) == "5.0"


def test_endpoint_preflight_accounts_for_maximum_replicas(
    remote_spec: ExperimentSpec,
) -> None:
    spec = profiled_spec(remote_spec)
    model = spec.matrix.models[0].model_copy(update={"revision": "a" * 40})
    deployment = spec.matrix.deployments[0]
    assert isinstance(deployment, DeploymentProfile)
    deployment = deployment.model_copy(
        update={"parameters": deployment.parameters | {"max_replicas": 2}}
    )
    spec = spec.model_copy(
        update={
            "matrix": spec.matrix.model_copy(
                update={"models": [model], "deployments": [deployment]}
            )
        }
    )
    resolved = build_profile_plan(
        spec,
        profile_id="profile-one",
        candidate_concurrency=[1, 2],
        max_spend_usd="10.00",
        profile_timeout_seconds=3600,
    )
    response = {
        "vendors": [
            {
                "name": "aws",
                "regions": [
                    {
                        "name": deployment.region.removeprefix("aws-"),
                        "computes": [
                            {
                                "instanceType": deployment.hardware,
                                "numAccelerators": deployment.accelerator_count,
                                "status": "available",
                                "pricePerHour": 5.0,
                                "quota": {
                                    "maxAccelerators": 2,
                                    "usedAccelerators": 0,
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=response)
        )
    )

    report = preflight_profile_plan(
        resolved, api=FakeApi(), client=client, token="hf_test"
    )

    assert report.required_accelerators == 2
    assert report.estimated_cost_usd == Decimal("10.0")


def test_managed_endpoint_preflight_uses_remote_namespace(
    remote_spec: ExperimentSpec,
) -> None:
    deployment = remote_spec.matrix.deployments[0].model_copy(update={"endpoint": None})
    assert isinstance(deployment, DeploymentProfile)
    model = remote_spec.matrix.models[0].model_copy(update={"revision": "a" * 40})
    spec = profiled_spec(
        remote_spec.model_copy(
            update={
                "matrix": remote_spec.matrix.model_copy(
                    update={"models": [model], "deployments": [deployment]}
                )
            }
        )
    )
    resolved = build_profile_plan(
        spec,
        profile_id="profile-one",
        candidate_concurrency=[1, 2],
        max_spend_usd="5.00",
        profile_timeout_seconds=3600,
    )
    response = {
        "vendors": [
            {
                "name": "aws",
                "regions": [
                    {
                        "name": deployment.region.removeprefix("aws-"),
                        "computes": [
                            {
                                "instanceType": deployment.hardware,
                                "numAccelerators": deployment.accelerator_count,
                                "status": "available",
                                "pricePerHour": 5.0,
                                "quota": {
                                    "maxAccelerators": 2,
                                    "usedAccelerators": 0,
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=response)
        )
    )

    report = preflight_profile_plan(
        resolved, api=FakeApi(), client=client, token="hf_test"
    )

    assert report.target_kind == "inference-endpoint"
    assert report.available_accelerators == 2


def test_endpoint_preflight_rejects_missing_repository_artifact(
    remote_spec: ExperimentSpec,
) -> None:
    deployment = remote_spec.matrix.deployments[0]
    assert isinstance(deployment, DeploymentProfile)
    deployment = deployment.model_copy(
        update={
            "engine": deployment.engine.model_copy(
                update={"arguments": ["-m", "/repository/missing.gguf"]}
            )
        }
    )
    model = remote_spec.matrix.models[0].model_copy(update={"revision": "a" * 40})
    spec = profiled_spec(
        remote_spec.model_copy(
            update={
                "matrix": remote_spec.matrix.model_copy(
                    update={"models": [model], "deployments": [deployment]}
                )
            }
        )
    )
    resolved = build_profile_plan(
        spec,
        profile_id="profile-one",
        candidate_concurrency=[1, 2],
        max_spend_usd="5.00",
        profile_timeout_seconds=3600,
    )

    with pytest.raises(ValueError, match="missing model artifacts: missing.gguf"):
        preflight_profile_plan(resolved, api=FakeApi(), token="hf_test")


def test_provider_preflight_requires_bounded_full_profile_estimate(
    remote_spec: ExperimentSpec,
) -> None:
    spec = profiled_spec(remote_spec)
    model = spec.matrix.models[0].model_copy(update={"revision": "a" * 40})
    provider = ProviderTarget(
        id="provider",
        model=model.repo,
        routing=ExplicitProviderRoute(provider="fireworks-ai"),
        limits=ProviderLimits(max_concurrent_requests=2),
    )
    spec = spec.model_copy(
        update={
            "matrix": spec.matrix.model_copy(
                update={"models": [model], "deployments": [provider]}
            )
        }
    )
    resolved = build_profile_plan(
        spec,
        profile_id="profile-one",
        candidate_concurrency=[1, 2],
        max_spend_usd="5.00",
        profile_timeout_seconds=3600,
    )

    with pytest.raises(ValueError, match="bounded full-profile cost estimate"):
        preflight_profile_plan(resolved, api=FakeProviderApi(), token="hf_test")


def test_provider_preflight_enforces_profile_spend_cap(
    remote_spec: ExperimentSpec,
) -> None:
    spec = profiled_spec(remote_spec)
    model = spec.matrix.models[0].model_copy(update={"revision": "a" * 40})
    provider = ProviderTarget(
        id="provider",
        model=model.repo,
        routing=ExplicitProviderRoute(provider="fireworks-ai"),
        limits=ProviderLimits(
            max_concurrent_requests=2,
            max_spend_usd=Decimal("10"),
            estimated_wave_cost_usd=Decimal("6"),
        ),
    )
    spec = spec.model_copy(
        update={
            "matrix": spec.matrix.model_copy(
                update={"models": [model], "deployments": [provider]}
            )
        }
    )
    resolved = build_profile_plan(
        spec,
        profile_id="profile-one",
        candidate_concurrency=[1, 2],
        max_spend_usd="5.00",
        profile_timeout_seconds=3600,
        estimated_profile_cost_usd="6",
    )

    with pytest.raises(ValueError, match="exceeds spend cap"):
        preflight_profile_plan(resolved, api=FakeProviderApi(), token="hf_test")


def test_provider_preflight_uses_full_profile_not_wave_estimate(
    remote_spec: ExperimentSpec,
) -> None:
    spec = profiled_spec(remote_spec)
    model = spec.matrix.models[0].model_copy(update={"revision": "a" * 40})
    provider = ProviderTarget(
        id="provider",
        model=model.repo,
        routing=ExplicitProviderRoute(provider="fireworks-ai"),
        limits=ProviderLimits(
            max_concurrent_requests=2,
            max_spend_usd=Decimal("10"),
            estimated_wave_cost_usd=Decimal("1"),
        ),
    )
    spec = spec.model_copy(
        update={
            "matrix": spec.matrix.model_copy(
                update={"models": [model], "deployments": [provider]}
            )
        }
    )
    resolved = build_profile_plan(
        spec,
        profile_id="profile-one",
        candidate_concurrency=[1, 2],
        max_spend_usd="10",
        profile_timeout_seconds=3600,
        estimated_profile_cost_usd="6",
    )

    report = preflight_profile_plan(resolved, api=FakeProviderApi(), token="hf_test")

    assert report.estimated_cost_usd == Decimal("6")


def test_profile_worker_rebuilds_provider_cost_estimate(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = remote_spec.matrix.models[0]
    provider = ProviderTarget(
        id="provider",
        model=model.repo,
        routing=ExplicitProviderRoute(provider="fireworks-ai"),
        limits=ProviderLimits(
            max_concurrent_requests=2,
            max_spend_usd=Decimal("10"),
            estimated_wave_cost_usd=Decimal("1"),
        ),
    )
    spec = profiled_spec(
        remote_spec.model_copy(
            update={
                "matrix": remote_spec.matrix.model_copy(
                    update={"deployments": [provider]}
                )
            }
        )
    )
    resolved = build_profile_plan(
        spec,
        profile_id="profile-one",
        candidate_concurrency=[1, 2],
        max_spend_usd="10",
        profile_timeout_seconds=3600,
        estimated_profile_cost_usd="6",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(resolved.model_dump_json(), encoding="utf-8")

    def rebuild(*_args: object, **kwargs: object) -> ProfilePlan:
        assert kwargs["estimated_profile_cost_usd"] == "6"
        raise RuntimeError("rebuild observed")

    monkeypatch.setattr("harbor_hf.profile_worker.build_profile_plan", rebuild)

    with pytest.raises(RuntimeError, match="rebuild observed"):
        run_profile_worker(plan_path, tmp_path / "output")


def test_profile_plan_cli_writes_local_plan(
    remote_manifest: Path, tmp_path: Path
) -> None:
    manifest = ExperimentSpec.model_validate_json(
        json.dumps(__import__("yaml").safe_load(remote_manifest.read_text()))
    )
    manifest = profiled_spec(manifest)
    source = tmp_path / "experiment.json"
    source.write_text(manifest.model_dump_json(), encoding="utf-8")
    output = tmp_path / "plan.json"

    result = runner.invoke(
        app,
        [
            "profile",
            "plan",
            str(source),
            "--output",
            str(output),
            "--profile-id",
            "profile-one",
            "--max-spend-usd",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert output.is_file()
    assert json.loads(result.stdout)["remote_work"] is False


@pytest.mark.parametrize("smoke_fails", [False, True])
def test_profile_worker_always_pauses_owned_endpoint(
    remote_spec: ExperimentSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    smoke_fails: bool,
) -> None:
    deployment = remote_spec.matrix.deployments[0].model_copy(update={"endpoint": None})
    spec = remote_spec.model_copy(
        update={
            "matrix": remote_spec.matrix.model_copy(
                update={"deployments": [deployment]}
            )
        }
    )
    resolved = plan(spec)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(resolved.model_dump_json(), encoding="utf-8")
    calls: list[str] = []

    class Manager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            calls.append("manager")

        def describe(self) -> dict[str, object]:
            calls.append("describe")
            return {}

        def pause_and_verify(self) -> dict[str, object]:
            calls.append("pause")
            return {}

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr(
        "harbor_hf.profile_worker.prepare_locked_source",
        lambda *_args, **_kwargs: calls.append("source"),
    )
    monkeypatch.setattr(
        "harbor_hf.profile_worker.HuggingFaceEndpointAdapter",
        lambda **_kwargs: object(),
    )

    class Provisioner:
        def __init__(self, _adapter: object) -> None:
            calls.append("provisioner")

        def create_or_adopt(self, *_args: object, **_kwargs: object) -> None:
            calls.append("provision")

    monkeypatch.setattr("harbor_hf.profile_worker.EndpointProvisioner", Provisioner)
    monkeypatch.setattr("harbor_hf.profile_worker.EndpointManager", Manager)
    monkeypatch.setattr(
        "harbor_hf.profile_worker.validate_endpoint_model",
        lambda *_args: calls.append("validated"),
    )
    monkeypatch.setattr(
        "harbor_hf.profile_worker.require_paused_endpoint",
        lambda *_args: calls.append("baseline-paused"),
    )
    monkeypatch.setattr(
        "harbor_hf.profile_worker.launch_cleanup_watchdog",
        lambda *_args: calls.append("watchdog"),
    )
    monkeypatch.setattr(
        "harbor_hf.profile_worker.resume_and_probe_endpoint",
        lambda *_args, **_kwargs: calls.append("resume") or "https://endpoint.test",
    )

    def smoke(*_args: object, **_kwargs: object) -> None:
        calls.append("smoke")
        if smoke_fails:
            raise ProfileWorkerError("smoke failed")

    monkeypatch.setattr("harbor_hf.profile_worker._verify_smoke", smoke)
    monkeypatch.setattr(
        "harbor_hf.profile_worker._run_ladder",
        lambda *_args, **_kwargs: [point(1, 10.0)],
    )

    run_expected_profile(
        plan_path,
        tmp_path / "output",
        smoke_fails=smoke_fails,
    )

    assert calls.index("watchdog") < calls.index("resume")
    assert calls.index("watchdog") < calls.index("provision")
    assert calls.index("provision") < calls.index("resume")
    assert calls[-1] == "pause"
