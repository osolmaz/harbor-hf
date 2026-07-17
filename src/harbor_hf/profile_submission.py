from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Protocol, cast

from huggingface_hub import HfApi
from pydantic import BaseModel, ConfigDict

from harbor_hf.models import DeploymentProfile
from harbor_hf.process import SubprocessRunner
from harbor_hf.profiling import ProfilePlan, bind_profile_target
from harbor_hf.provider_proxy import PROVIDER_RECORDER_PORT
from harbor_hf.runs import build_run_lock
from harbor_hf.submission import (
    BucketApi,
    bucket_uri,
    endpoint_lease_label_for,
    ensure_private_job_input_bucket,
    job_secret_names,
    locked_source_command,
    require_private_bucket,
    require_source_secrets,
    stage_job_input,
)

_JOB_ID = re.compile(r"(?<![a-f0-9])[a-f0-9]{24}(?![a-f0-9])")


class TextRunner(Protocol):
    def run_text(self, command: list[str]) -> str: ...


class ProfileSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    artifact_prefix: str
    job_id: str | None
    command: list[str]


def build_profile_submit_command(
    plan: ProfilePlan, *, input_dir: str, bucket: str
) -> list[str]:
    spec, _desired = bind_profile_target(plan)
    if spec.remote is None:
        raise ValueError("profile run requires remote configuration")
    lock = build_run_lock(
        spec,
        model_id=plan.cell.model,
        deployment_id=plan.cell.deployment,
        agent_id=plan.cell.agent,
        run_id=f"profile-{plan.profile_id}",
        allow_provider=True,
    )
    target = lock.deployment
    labels = ["--label", f"harbor-hf-profile={plan.profile_id}"]
    exposed_port: list[str] = []
    if isinstance(target, DeploymentProfile):
        if target.endpoint is None:
            raise ValueError("profile endpoint binding is missing")
        labels.extend(
            [
                "--label",
                "harbor-hf-endpoint="
                + endpoint_lease_label_for(
                    target.endpoint.namespace, target.endpoint.name
                ),
            ]
        )
    else:
        exposed_port = ["--expose", str(PROVIDER_RECORDER_PORT)]
        labels.extend(
            [
                "--label",
                "harbor-hf-provider="
                + hashlib.sha256(target.model.encode()).hexdigest()[:32],
            ]
        )
    secret_args = [
        argument for name in job_secret_names(lock) for argument in ("--secrets", name)
    ]
    job = spec.remote.job
    return [
        "hf",
        "jobs",
        "run",
        "--detach",
        "--namespace",
        job.namespace,
        "--flavor",
        job.flavor,
        "--timeout",
        f"{job.timeout_seconds}s",
        *exposed_port,
        *secret_args,
        *labels,
        "--volume",
        f"{input_dir}:/input:ro",
        "--volume",
        f"{bucket_uri(bucket)}:/output:rw",
        "--",
        job.image,
        *locked_source_command(
            spec.remote.worker,
            "harbor-hf",
            "profile-worker",
            "/input/plan.json",
            "--output-root",
            "/output",
        ),
    ]


def submit_profile(
    plan: ProfilePlan,
    *,
    runner: TextRunner | None = None,
    bucket_api: BucketApi | None = None,
) -> ProfileSubmission:
    spec, _desired = bind_profile_target(plan)
    if spec.remote is None:
        raise ValueError("profile run requires remote configuration")
    lock = build_run_lock(
        spec,
        model_id=plan.cell.model,
        deployment_id=plan.cell.deployment,
        agent_id=plan.cell.agent,
        run_id=f"profile-{plan.profile_id}",
        allow_provider=True,
    )
    require_source_secrets(lock)
    api = bucket_api or cast(BucketApi, HfApi())
    input_bucket = ensure_private_job_input_bucket(spec.remote.job.namespace, api=api)
    require_private_bucket(plan.artifacts.bucket, api=api)
    with tempfile.TemporaryDirectory(prefix="harbor-hf-profile-") as directory:
        staging = Path(directory)
        (staging / "plan.json").write_text(
            json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source = stage_job_input(
            staging,
            bucket=input_bucket,
            identity=plan.profile_id,
            api=api,
        )
    command = build_profile_submit_command(
        plan, input_dir=source, bucket=plan.artifacts.bucket
    )
    output = (runner or SubprocessRunner()).run_text(command)
    match = _JOB_ID.search(output)
    if match is None:
        raise ValueError("HF Jobs profile submission did not return a job ID")
    return ProfileSubmission(
        profile_id=plan.profile_id,
        artifact_prefix=plan.artifacts.prefix,
        job_id=match.group(),
        command=command,
    )
