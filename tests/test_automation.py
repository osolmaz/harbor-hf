from types import SimpleNamespace
from typing import cast

import pytest

from harbor_hf.automation import (
    AutomationError,
    AutomationRequest,
    install_automation,
    scheduled_reconciler_command,
)
from harbor_hf.models import ExperimentSpec


class FakeApi:
    def __init__(self) -> None:
        self.job: dict[str, object] = {}
        self.webhook: dict[str, object] = {}
        self.scheduled_jobs: list[SimpleNamespace] = []
        self.webhooks: list[SimpleNamespace] = []

    def list_scheduled_jobs(self, **kwargs: object) -> list[object]:
        assert kwargs == {"namespace": "osolmaz"}
        return cast(list[object], self.scheduled_jobs)

    def create_scheduled_job(self, **kwargs: object) -> object:
        self.job = kwargs
        value = SimpleNamespace(
            id="scheduled-1",
            schedule=kwargs["schedule"],
            suspend=kwargs["suspend"],
            concurrency=kwargs["concurrency"],
            job_spec=SimpleNamespace(
                docker_image=kwargs["image"],
                command=kwargs["command"],
                flavor=kwargs["flavor"],
                timeout=kwargs["timeout"],
                secrets=kwargs["secrets"],
                labels=kwargs["labels"],
            ),
        )
        self.scheduled_jobs.append(value)
        return value

    def list_webhooks(self, **kwargs: object) -> list[object]:
        assert kwargs == {}
        return cast(list[object], self.webhooks)

    def create_webhook(self, **kwargs: object) -> object:
        self.webhook = kwargs
        watched = cast(list[dict[str, object]], kwargs["watched"])
        value = SimpleNamespace(
            id="webhook-1",
            watched=[SimpleNamespace(**item) for item in watched],
            domains=kwargs["domains"],
            disabled=False,
            job=self.scheduled_jobs[0].job_spec,
        )
        self.webhooks.append(value)
        return value


def _request(spec: ExperimentSpec) -> AutomationRequest:
    assert spec.remote is not None
    return AutomationRequest(
        namespace="osolmaz",
        schedule="*/10 * * * *",
        remote=spec.remote,
    )


def test_builds_digest_pinned_scheduled_reconciler(remote_spec: ExperimentSpec) -> None:
    request = _request(remote_spec)

    command = scheduled_reconciler_command(request)

    assert command[-6:] == [
        "harbor-hf",
        "campaign",
        "reconcile-all",
        "--namespace",
        "osolmaz",
        "--apply",
    ]
    assert request.remote.worker.revision in command[2]


def test_installs_serial_schedule_and_dataset_webhook(
    remote_spec: ExperimentSpec,
) -> None:
    api = FakeApi()

    result = install_automation(_request(remote_spec), token="test-only", api=api)

    assert result.model_dump() == {
        "scheduled_job_id": "scheduled-1",
        "webhook_id": "webhook-1",
        "control_repository": "osolmaz/harbor-hf-coordination",
        "scheduled_job_created": True,
        "webhook_created": True,
    }
    assert api.job["concurrency"] is False
    assert set(cast(dict[str, str], api.job["secrets"])) == {"HF_TOKEN"}
    assert api.job["labels"] == {
        "harbor-hf-role": "campaign-reconciler",
        "harbor-hf-namespace": "osolmaz",
    }
    assert api.webhook == {
        "job_id": "scheduled-1",
        "watched": [{"type": "dataset", "name": "osolmaz/harbor-hf-coordination"}],
        "domains": ["repo.content"],
    }

    repeated = install_automation(_request(remote_spec), token="test-only", api=api)

    assert not repeated.scheduled_job_created
    assert not repeated.webhook_created
    assert len(api.scheduled_jobs) == 1
    assert len(api.webhooks) == 1


def test_rejects_cross_namespace_automation(remote_spec: ExperimentSpec) -> None:
    request = _request(remote_spec).model_copy(update={"namespace": "other"})

    with pytest.raises(AutomationError, match="namespaces must match"):
        install_automation(request, token="test-only", api=FakeApi())


@pytest.mark.parametrize("missing", ["job", "webhook"])
def test_requires_created_resource_ids(
    remote_spec: ExperimentSpec, missing: str
) -> None:
    class MissingApi(FakeApi):
        def create_scheduled_job(self, **kwargs: object) -> object:
            super().create_scheduled_job(**kwargs)
            return SimpleNamespace(id=None if missing == "job" else "scheduled-1")

        def create_webhook(self, **kwargs: object) -> object:
            super().create_webhook(**kwargs)
            return SimpleNamespace(id=None if missing == "webhook" else "webhook-1")

    with pytest.raises(AutomationError, match="creation returned no ID"):
        install_automation(_request(remote_spec), token="test-only", api=MissingApi())


@pytest.mark.parametrize("resource", ["schedule", "webhook"])
def test_rejects_managed_resource_drift(
    remote_spec: ExperimentSpec, resource: str
) -> None:
    api = FakeApi()
    request = _request(remote_spec)
    install_automation(request, token="test-only", api=api)
    if resource == "schedule":
        api.scheduled_jobs[0].schedule = "@daily"
    else:
        api.webhooks[0].domains = ["repo"]

    with pytest.raises(AutomationError, match="configuration drift"):
        install_automation(request, token="test-only", api=api)
