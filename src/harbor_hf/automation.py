from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Protocol, cast

from huggingface_hub import HfApi
from pydantic import BaseModel, ConfigDict, Field, field_validator

from harbor_hf.coordination import coordination_repository
from harbor_hf.models import RemoteExecutionSpec
from harbor_hf.submission import locked_source_command

_WEBHOOK_DOMAINS = ("repo",)


class AutomationError(RuntimeError):
    """Raised when campaign reconciliation automation is malformed."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AutomationRequest(FrozenModel):
    namespace: str = Field(min_length=1)
    schedule: str = Field(min_length=1)
    remote: RemoteExecutionSpec
    secret_names: list[str] = Field(default_factory=list)
    provider_active_waves: int | None = Field(default=None, ge=1)
    suspended: bool = False

    @field_validator("secret_names")
    @classmethod
    def extra_secrets_are_canonical(cls, value: list[str]) -> list[str]:
        if "HF_TOKEN" in value:
            raise ValueError("automation secret_names must not repeat HF_TOKEN")
        if len(value) != len(set(value)):
            raise ValueError("automation secret names must be unique")
        if any(not name or name != name.strip() or "=" in name for name in value):
            raise ValueError("automation secret names must be environment variables")
        return value


class AutomationInstallation(FrozenModel):
    scheduled_job_id: str
    webhook_id: str
    control_repository: str
    scheduled_job_created: bool
    webhook_created: bool


class AutomationPlan(FrozenModel):
    namespace: str
    schedule: str
    suspended: bool
    image: str
    command: list[str]
    secret_names: list[str]
    provider_active_waves: int | None
    control_repository: str


class AutomationApi(Protocol):
    def list_scheduled_jobs(self, **kwargs: object) -> list[object]: ...

    def create_scheduled_job(self, **kwargs: object) -> object: ...

    def list_webhooks(self, **kwargs: object) -> list[object]: ...

    def create_webhook(self, **kwargs: object) -> object: ...


def scheduled_reconciler_command(request: AutomationRequest) -> list[str]:
    command = locked_source_command(
        request.remote.worker,
        "harbor-hf",
        "campaign",
        "reconcile-all",
        "--namespace",
        request.namespace,
        "--apply",
    )
    if request.provider_active_waves is not None:
        command.extend(["--provider-active-waves", str(request.provider_active_waves)])
    return command


def install_automation(
    request: AutomationRequest,
    *,
    token: str,
    api: AutomationApi | None = None,
) -> AutomationInstallation:
    if not token:
        raise AutomationError("automation installation requires an HF token")
    if request.remote.job.namespace != request.namespace:
        raise AutomationError("automation and remote Job namespaces must match")
    client = api or cast(AutomationApi, HfApi(token=token))
    plan = automation_plan(request)
    job, created_job = _install_scheduled_job(client, request, plan, token)
    job_id = _required_id(job, "scheduled Job")
    webhook, created_webhook = _install_webhook(client, request, job_id)
    return AutomationInstallation(
        scheduled_job_id=job_id,
        webhook_id=_required_id(webhook, "webhook"),
        control_repository=coordination_repository(request.namespace),
        scheduled_job_created=created_job,
        webhook_created=created_webhook,
    )


def _install_scheduled_job(
    client: AutomationApi,
    request: AutomationRequest,
    plan: AutomationPlan,
    token: str,
) -> tuple[object, bool]:
    managed_jobs = [
        job
        for job in client.list_scheduled_jobs(namespace=request.namespace)
        if _managed_job(job, request.namespace)
    ]
    if len(managed_jobs) > 1:
        raise AutomationError("multiple managed reconciliation schedules exist")
    created_job = not managed_jobs
    if managed_jobs:
        job = managed_jobs[0]
        if not _scheduled_job_matches(job, request):
            raise AutomationError(
                "managed reconciliation schedule has configuration drift"
            )
    else:
        job = client.create_scheduled_job(
            image=request.remote.job.image,
            command=plan.command,
            schedule=request.schedule,
            suspend=request.suspended,
            concurrency=False,
            secrets=_automation_secrets(request, token),
            flavor=request.remote.job.flavor,
            timeout=request.remote.job.timeout_seconds,
            labels=_managed_labels(request.namespace),
            namespace=request.namespace,
        )
    return job, created_job


def _install_webhook(
    client: AutomationApi,
    request: AutomationRequest,
    job_id: str,
) -> tuple[object, bool]:
    managed_webhooks = [
        webhook
        for webhook in client.list_webhooks()
        if _managed_webhook(webhook, request.namespace)
    ]
    if len(managed_webhooks) > 1:
        raise AutomationError("multiple managed reconciliation webhooks exist")
    created_webhook = not managed_webhooks
    if managed_webhooks:
        webhook = managed_webhooks[0]
        if not _webhook_matches(webhook, request):
            raise AutomationError(
                "managed reconciliation webhook has configuration drift"
            )
    else:
        webhook = client.create_webhook(
            job_id=job_id,
            watched=[
                {
                    "type": "dataset",
                    "name": coordination_repository(request.namespace),
                }
            ],
            domains=list(_WEBHOOK_DOMAINS),
        )
    return webhook, created_webhook


def automation_plan(request: AutomationRequest) -> AutomationPlan:
    if request.remote.job.namespace != request.namespace:
        raise AutomationError("automation and remote Job namespaces must match")
    return AutomationPlan(
        namespace=request.namespace,
        schedule=request.schedule,
        suspended=request.suspended,
        image=request.remote.job.image,
        command=scheduled_reconciler_command(request),
        secret_names=[
            request.remote.job.token_secret_name,
            *request.secret_names,
        ],
        provider_active_waves=request.provider_active_waves,
        control_repository=coordination_repository(request.namespace),
    )


def _managed_labels(namespace: str) -> dict[str, str]:
    return {
        "harbor-hf-role": "campaign-reconciler",
        "harbor-hf-namespace": namespace,
    }


def _managed_job(value: object, namespace: str) -> bool:
    spec = getattr(value, "job_spec", None)
    return _managed_spec(spec, namespace)


def _managed_spec(spec: object, namespace: str) -> bool:
    labels = getattr(spec, "labels", None)
    return isinstance(labels, Mapping) and all(
        labels.get(key) == expected
        for key, expected in _managed_labels(namespace).items()
    )


def _scheduled_job_matches(value: object, request: AutomationRequest) -> bool:
    spec = getattr(value, "job_spec", None)
    secrets = getattr(spec, "secrets", None)
    secret_names = set(secrets) if isinstance(secrets, Mapping) else set()
    flavor = getattr(spec, "flavor", None)
    flavor_value = getattr(flavor, "value", flavor)
    return (
        getattr(value, "schedule", None) == request.schedule
        and getattr(value, "suspend", None) is request.suspended
        and getattr(value, "concurrency", None) is False
        and getattr(spec, "docker_image", None) == request.remote.job.image
        and getattr(spec, "command", None) == scheduled_reconciler_command(request)
        and flavor_value == request.remote.job.flavor
        and getattr(spec, "timeout", None) == request.remote.job.timeout_seconds
        and secret_names
        == {request.remote.job.token_secret_name, *request.secret_names}
    )


def _managed_webhook(value: object, namespace: str) -> bool:
    job = getattr(value, "job", None)
    labels = getattr(job, "labels", None)
    return isinstance(labels, Mapping) and all(
        labels.get(key) == expected
        for key, expected in _managed_labels(namespace).items()
    )


def _webhook_matches(value: object, request: AutomationRequest) -> bool:
    watched = {
        (getattr(item, "type", None), getattr(item, "name", None))
        for item in getattr(value, "watched", [])
    }
    return (
        watched == {("dataset", coordination_repository(request.namespace))}
        and getattr(value, "domains", None) == list(_WEBHOOK_DOMAINS)
        and getattr(value, "disabled", False) is False
        and _scheduled_spec_matches(getattr(value, "job", None), request)
    )


def _scheduled_spec_matches(spec: object, request: AutomationRequest) -> bool:
    return (
        getattr(spec, "docker_image", None) == request.remote.job.image
        and getattr(spec, "command", None) == scheduled_reconciler_command(request)
        and _managed_spec(spec, request.namespace)
    )


def _required_id(value: object, resource: str) -> str:
    identifier = getattr(value, "id", None)
    if not isinstance(identifier, str) or not identifier:
        raise AutomationError(f"{resource} creation returned no ID")
    return identifier


def _automation_secrets(request: AutomationRequest, token: str) -> dict[str, str]:
    secrets = {request.remote.job.token_secret_name: token}
    for name in request.secret_names:
        value = os.environ.get(name, "")
        if not value:
            raise AutomationError(f"required secret {name} is not available")
        secrets[name] = value
    return secrets
