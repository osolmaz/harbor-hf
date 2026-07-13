from types import SimpleNamespace

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

    def create_scheduled_job(self, **kwargs: object) -> object:
        self.job = kwargs
        return SimpleNamespace(id="scheduled-1")

    def create_webhook(self, **kwargs: object) -> object:
        self.webhook = kwargs
        return SimpleNamespace(id="webhook-1")


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

    result = install_automation(_request(remote_spec), api=api)

    assert result.model_dump() == {
        "scheduled_job_id": "scheduled-1",
        "webhook_id": "webhook-1",
        "control_repository": "osolmaz/harbor-hf-coordination",
    }
    assert api.job["concurrency"] is False
    assert api.job["secrets"] == ["HF_TOKEN"]
    assert api.job["labels"] == {"harbor-hf-role": "campaign-reconciler"}
    assert api.webhook == {
        "job_id": "scheduled-1",
        "watched": [
            {"type": "dataset", "name": "osolmaz/harbor-hf-coordination"}
        ],
        "domains": ["repo.content"],
    }


def test_rejects_cross_namespace_automation(remote_spec: ExperimentSpec) -> None:
    request = _request(remote_spec).model_copy(update={"namespace": "other"})

    with pytest.raises(AutomationError, match="namespaces must match"):
        install_automation(request, api=FakeApi())


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
        install_automation(_request(remote_spec), api=MissingApi())
