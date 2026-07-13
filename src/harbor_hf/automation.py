from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from harbor_hf.coordination import coordination_repository
from harbor_hf.models import RemoteExecutionSpec
from harbor_hf.submission import locked_source_command


class AutomationError(RuntimeError):
    """Raised when campaign reconciliation automation is malformed."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AutomationRequest(FrozenModel):
    namespace: str = Field(min_length=1)
    schedule: str = Field(min_length=1)
    remote: RemoteExecutionSpec
    suspended: bool = False


class AutomationInstallation(FrozenModel):
    scheduled_job_id: str
    webhook_id: str
    control_repository: str


class AutomationApi(Protocol):
    def create_scheduled_job(self, **kwargs: object) -> object: ...

    def create_webhook(self, **kwargs: object) -> object: ...


def scheduled_reconciler_command(request: AutomationRequest) -> list[str]:
    return locked_source_command(
        request.remote.worker,
        "harbor-hf",
        "campaign",
        "reconcile-all",
        "--namespace",
        request.namespace,
        "--apply",
    )


def install_automation(
    request: AutomationRequest,
    *,
    api: AutomationApi,
) -> AutomationInstallation:
    if request.remote.job.namespace != request.namespace:
        raise AutomationError("automation and remote Job namespaces must match")
    job = api.create_scheduled_job(
        image=request.remote.job.image,
        command=scheduled_reconciler_command(request),
        schedule=request.schedule,
        suspend=request.suspended,
        concurrency=False,
        secrets=[request.remote.job.token_secret_name],
        flavor=request.remote.job.flavor,
        timeout=request.remote.job.timeout_seconds,
        labels={"harbor-hf-role": "campaign-reconciler"},
        namespace=request.namespace,
    )
    job_id = _required_id(job, "scheduled Job")
    webhook = api.create_webhook(
        job_id=job_id,
        watched=[
            {
                "type": "dataset",
                "name": coordination_repository(request.namespace),
            }
        ],
        domains=["repo.content"],
    )
    return AutomationInstallation(
        scheduled_job_id=job_id,
        webhook_id=_required_id(webhook, "webhook"),
        control_repository=coordination_repository(request.namespace),
    )


def _required_id(value: object, resource: str) -> str:
    identifier = getattr(value, "id", None)
    if not isinstance(identifier, str) or not identifier:
        raise AutomationError(f"{resource} creation returned no ID")
    return identifier
