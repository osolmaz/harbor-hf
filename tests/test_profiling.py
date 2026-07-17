from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

from harbor_hf.cli import app
from harbor_hf.models import DeploymentProfile, ExperimentSpec, ServingProfileBinding
from harbor_hf.profile_preflight import preflight_profile_plan
from harbor_hf.profile_submission import build_profile_submit_command
from harbor_hf.profile_worker import ProfileWorkerError, run_profile_worker
from harbor_hf.profiling import (
    ProfilePlan,
    ProfilePoint,
    build_profile_plan,
    canonical_digest,
    new_unselected_profile,
    select_profile,
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


def test_profile_submit_command_is_remote_only(remote_spec: ExperimentSpec) -> None:
    command = build_profile_submit_command(
        plan(remote_spec), input_dir="hf://buckets/input", bucket="osolmaz/results"
    )

    assert command[:3] == ["hf", "jobs", "run"]
    assert "profile-worker" in command
    assert "/input/plan.json" in command
    assert not any("llama-server" in argument for argument in command)


class FakeApi:
    def model_info(self, repo: str, **kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(sha="a" * 40, inference_provider_mapping={})

    def bucket_info(self, bucket_id: str) -> object:
        del bucket_id
        return SimpleNamespace(private=True)


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
                "regions": [
                    {
                        "name": deployment.region,
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
                ]
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
    resolved = plan(remote_spec)
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

    if smoke_fails:
        with pytest.raises(ProfileWorkerError, match="smoke failed"):
            run_profile_worker(plan_path, tmp_path / "output")
    else:
        destination = run_profile_worker(plan_path, tmp_path / "output")
        assert (destination / "_SELECTED").is_file()

    assert calls.index("watchdog") < calls.index("resume")
    assert calls[-1] == "pause"
