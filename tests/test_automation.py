from collections.abc import Callable
from copy import deepcopy
from types import SimpleNamespace
from typing import cast

import pytest
from huggingface_hub import WebhookInfo, WebhookWatchedItem
from huggingface_hub._jobs_api import JobSpec

from harbor_hf.automation import (
    AutomationError,
    AutomationRequest,
    _required_id,
    _scheduled_job_matches,
    _scheduled_spec_matches,
    _webhook_matches,
    automation_plan,
    install_automation,
    scheduled_reconciler_command,
)
from harbor_hf.models import ExperimentSpec


class FakeApi:
    def __init__(self) -> None:
        self.job: dict[str, object] = {}
        self.webhook: dict[str, object] = {}
        self.scheduled_jobs: list[SimpleNamespace] = []
        self.webhooks: list[SimpleNamespace | WebhookInfo] = []

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


def _provider_webhook(request: AutomationRequest) -> WebhookInfo:
    return WebhookInfo(
        id="webhook-1",
        url=None,
        job=JobSpec(
            docker_image=request.remote.job.image,
            command=scheduled_reconciler_command(request),
            labels={
                "harbor-hf-role": "campaign-reconciler",
                "harbor-hf-namespace": request.namespace,
            },
        ),
        watched=[
            WebhookWatchedItem(
                type="dataset",
                name="osolmaz/harbor-hf-coordination",
            )
        ],
        domains=["repo"],
        secret=None,
        disabled=False,
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
        "domains": ["repo"],
    }

    repeated = install_automation(_request(remote_spec), token="test-only", api=api)

    assert not repeated.scheduled_job_created
    assert not repeated.webhook_created
    assert len(api.scheduled_jobs) == 1
    assert len(api.webhooks) == 1


def test_installs_extra_controller_secrets_for_private_sources(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(remote_spec).model_copy(
        update={"secret_names": ["GITHUB_TOKEN"]}
    )
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    api = FakeApi()

    install_automation(request, token="hf-secret", api=api)

    assert api.job["secrets"] == {
        "HF_TOKEN": "hf-secret",
        "GITHUB_TOKEN": "github-secret",
    }
    assert automation_plan(request).secret_names == ["HF_TOKEN", "GITHUB_TOKEN"]


def test_install_rejects_missing_extra_controller_secret(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(remote_spec).model_copy(
        update={"secret_names": ["GITHUB_TOKEN"]}
    )
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with pytest.raises(AutomationError, match="required secret GITHUB_TOKEN"):
        install_automation(request, token="hf-secret", api=FakeApi())


def test_install_creates_supported_webhook_from_real_api_contract(
    remote_spec: ExperimentSpec,
) -> None:
    request = _request(remote_spec)

    class RealShapedApi(FakeApi):
        def create_webhook(self, **kwargs: object) -> object:
            self.webhook = kwargs
            webhook = _provider_webhook(request)
            self.webhooks.append(webhook)
            return webhook

    api = RealShapedApi()

    result = install_automation(request, token="test-only", api=api)

    assert result.webhook_created is True
    assert api.webhook == {
        "job_id": "scheduled-1",
        "watched": [{"type": "dataset", "name": "osolmaz/harbor-hf-coordination"}],
        "domains": ["repo"],
    }


def test_install_adopts_real_api_webhook_with_supported_domain(
    remote_spec: ExperimentSpec,
) -> None:
    request = _request(remote_spec)
    api = FakeApi()
    api.webhooks.append(_provider_webhook(request))

    result = install_automation(request, token="test-only", api=api)

    assert result.webhook_created is False
    assert result.webhook_id == "webhook-1"
    assert api.webhook == {}


def test_rejects_cross_namespace_automation(remote_spec: ExperimentSpec) -> None:
    request = _request(remote_spec).model_copy(update={"namespace": "other"})

    with pytest.raises(AutomationError) as captured:
        install_automation(request, token="test-only", api=FakeApi())

    assert str(captured.value) == "automation and remote Job namespaces must match"


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
        api.webhooks[0].domains = ["discussions"]

    with pytest.raises(AutomationError, match="configuration drift"):
        install_automation(request, token="test-only", api=api)


def test_automation_plan_exposes_the_complete_installation_contract(
    remote_spec: ExperimentSpec,
) -> None:
    request = _request(remote_spec).model_copy(
        update={"schedule": "17 */3 * * *", "suspended": True}
    )

    assert automation_plan(request).model_dump() == {
        "namespace": "osolmaz",
        "schedule": "17 */3 * * *",
        "suspended": True,
        "image": "ghcr.io/astral-sh/uv@sha256:" + "0" * 64,
        "command": scheduled_reconciler_command(request),
        "secret_names": ["HF_TOKEN"],
        "control_repository": "osolmaz/harbor-hf-coordination",
    }


def test_install_requires_a_nonempty_token_with_exact_error(
    remote_spec: ExperimentSpec,
) -> None:
    with pytest.raises(AutomationError) as captured:
        install_automation(_request(remote_spec), token="", api=FakeApi())

    assert str(captured.value) == "automation installation requires an HF token"


def test_plan_rejects_cross_namespace_with_exact_error(
    remote_spec: ExperimentSpec,
) -> None:
    request = _request(remote_spec).model_copy(update={"namespace": "other"})

    with pytest.raises(AutomationError) as captured:
        automation_plan(request)

    assert str(captured.value) == "automation and remote Job namespaces must match"


@pytest.mark.parametrize(
    ("resource", "expected"),
    [
        ("job", "scheduled Job creation returned no ID"),
        ("webhook", "webhook creation returned no ID"),
    ],
)
def test_install_reports_the_exact_resource_missing_an_identifier(
    remote_spec: ExperimentSpec, resource: str, expected: str
) -> None:
    class MissingIdentifierApi(FakeApi):
        def create_scheduled_job(self, **kwargs: object) -> object:
            created = super().create_scheduled_job(**kwargs)
            return SimpleNamespace() if resource == "job" else created

        def create_webhook(self, **kwargs: object) -> object:
            created = super().create_webhook(**kwargs)
            return SimpleNamespace() if resource == "webhook" else created

    with pytest.raises(AutomationError) as captured:
        install_automation(
            _request(remote_spec), token="test-only", api=MissingIdentifierApi()
        )

    assert str(captured.value) == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda job: setattr(job, "schedule", "@daily"),
        lambda job: setattr(job, "suspend", 0),
        lambda job: setattr(job, "concurrency", 0),
        lambda job: setattr(job.job_spec, "docker_image", "wrong-image"),
        lambda job: setattr(job.job_spec, "command", ["wrong-command"]),
        lambda job: setattr(job.job_spec, "flavor", "wrong-flavor"),
        lambda job: setattr(job.job_spec, "timeout", 1),
        lambda job: setattr(job.job_spec, "secrets", {}),
        lambda job: setattr(job.job_spec, "secrets", ["HF_TOKEN"]),
    ],
)
def test_every_managed_schedule_field_participates_in_drift_detection(
    remote_spec: ExperimentSpec,
    mutate: Callable[[SimpleNamespace], None],
) -> None:
    api = FakeApi()
    request = _request(remote_spec)
    install_automation(request, token="test-only", api=api)
    mutate(api.scheduled_jobs[0])

    with pytest.raises(AutomationError) as captured:
        install_automation(request, token="test-only", api=api)

    assert str(captured.value) == (
        "managed reconciliation schedule has configuration drift"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda webhook: setattr(
            webhook,
            "watched",
            [SimpleNamespace(type="dataset", name="other/repository")],
        ),
        lambda webhook: setattr(
            webhook,
            "watched",
            [SimpleNamespace(type="model", name="osolmaz/harbor-hf-coordination")],
        ),
        lambda webhook: setattr(webhook, "domains", ["discussions"]),
        lambda webhook: setattr(webhook, "disabled", 0),
        lambda webhook: setattr(webhook.job, "docker_image", "wrong-image"),
        lambda webhook: setattr(webhook.job, "command", ["wrong-command"]),
    ],
)
def test_every_managed_webhook_field_participates_in_drift_detection(
    remote_spec: ExperimentSpec,
    mutate: Callable[[SimpleNamespace], None],
) -> None:
    api = FakeApi()
    request = _request(remote_spec)
    install_automation(request, token="test-only", api=api)
    webhook = cast(SimpleNamespace, api.webhooks[0])
    webhook.job = deepcopy(webhook.job)
    mutate(webhook)

    with pytest.raises(AutomationError) as captured:
        install_automation(request, token="test-only", api=api)

    assert str(captured.value) == (
        "managed reconciliation webhook has configuration drift"
    )


@pytest.mark.parametrize(
    ("resource", "expected"),
    [
        ("schedule", "multiple managed reconciliation schedules exist"),
        ("webhook", "multiple managed reconciliation webhooks exist"),
    ],
)
def test_install_rejects_multiple_managed_resources_with_exact_error(
    remote_spec: ExperimentSpec, resource: str, expected: str
) -> None:
    api = FakeApi()
    request = _request(remote_spec)
    install_automation(request, token="test-only", api=api)
    if resource == "schedule":
        api.scheduled_jobs.append(deepcopy(api.scheduled_jobs[0]))
    else:
        api.webhooks.append(deepcopy(api.webhooks[0]))

    with pytest.raises(AutomationError) as captured:
        install_automation(request, token="test-only", api=api)

    assert str(captured.value) == expected


def test_unmanaged_provider_resources_are_ignored(
    remote_spec: ExperimentSpec,
) -> None:
    class ApiWithDecoys(FakeApi):
        def create_webhook(self, **kwargs: object) -> object:
            self.webhook = kwargs
            watched = cast(list[dict[str, object]], kwargs["watched"])
            value = SimpleNamespace(
                id="webhook-1",
                watched=[SimpleNamespace(**item) for item in watched],
                domains=kwargs["domains"],
                disabled=False,
                job=self.scheduled_jobs[-1].job_spec,
            )
            self.webhooks.append(value)
            return value

    api = ApiWithDecoys()
    api.scheduled_jobs = [
        SimpleNamespace(),
        SimpleNamespace(job_spec=SimpleNamespace(labels=[])),
        SimpleNamespace(
            job_spec=SimpleNamespace(
                labels={
                    "harbor-hf-role": "campaign-reconciler",
                    "harbor-hf-namespace": "other",
                }
            )
        ),
    ]
    api.webhooks = [
        SimpleNamespace(),
        SimpleNamespace(job=SimpleNamespace(labels=[])),
        SimpleNamespace(
            job=SimpleNamespace(
                labels={
                    "harbor-hf-role": "other-role",
                    "harbor-hf-namespace": "osolmaz",
                }
            )
        ),
    ]

    result = install_automation(_request(remote_spec), token="test-only", api=api)

    assert result.scheduled_job_created is True
    assert result.webhook_created is True
    assert api.job["namespace"] == "osolmaz"
    assert api.job["secrets"] == {"HF_TOKEN": "test-only"}
    assert api.webhook["job_id"] == "scheduled-1"


def test_install_constructs_default_hf_client_with_the_install_token(
    remote_spec: ExperimentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens: list[str] = []

    class DefaultApi(FakeApi):
        def __init__(self, *, token: str) -> None:
            super().__init__()
            tokens.append(token)

    monkeypatch.setattr("harbor_hf.automation.HfApi", DefaultApi)

    result = install_automation(_request(remote_spec), token="default-token")

    assert tokens == ["default-token"]
    assert result.scheduled_job_id == "scheduled-1"
    assert result.webhook_id == "webhook-1"


def test_schedule_matching_returns_false_for_every_missing_provider_field(
    remote_spec: ExperimentSpec,
) -> None:
    api = FakeApi()
    request = _request(remote_spec)
    install_automation(request, token="test-only", api=api)
    valid = api.scheduled_jobs[0]
    assert _scheduled_job_matches(valid, request) is True

    missing_paths = [
        ("job_spec",),
        ("job_spec", "secrets"),
        ("job_spec", "flavor"),
        ("schedule",),
        ("suspend",),
        ("concurrency",),
        ("job_spec", "docker_image"),
        ("job_spec", "command"),
        ("job_spec", "timeout"),
    ]
    for path in missing_paths:
        candidate = deepcopy(valid)
        owner = candidate if len(path) == 1 else candidate.job_spec
        delattr(owner, path[-1])
        assert _scheduled_job_matches(candidate, request) is False

    enum_flavor = deepcopy(valid)
    enum_flavor.job_spec.flavor = SimpleNamespace(value=request.remote.job.flavor)
    assert _scheduled_job_matches(enum_flavor, request) is True


def test_webhook_matching_handles_sparse_provider_objects_as_contract_data(
    remote_spec: ExperimentSpec,
) -> None:
    api = FakeApi()
    request = _request(remote_spec)
    install_automation(request, token="test-only", api=api)
    valid = api.webhooks[0]
    assert _webhook_matches(valid, request) is True

    without_disabled = deepcopy(valid)
    del without_disabled.disabled
    assert _webhook_matches(without_disabled, request) is True

    incomplete = [
        SimpleNamespace(),
        SimpleNamespace(watched=[SimpleNamespace()]),
        SimpleNamespace(
            watched=[
                SimpleNamespace(type="dataset", name="osolmaz/harbor-hf-coordination")
            ]
        ),
        SimpleNamespace(
            watched=[
                SimpleNamespace(type="dataset", name="osolmaz/harbor-hf-coordination")
            ],
            domains=["repo"],
        ),
    ]
    assert [_webhook_matches(value, request) for value in incomplete] == [
        False,
        False,
        False,
        False,
    ]


def test_required_identifier_rejects_truthy_nonstr_values() -> None:
    with pytest.raises(AutomationError) as captured:
        _required_id(SimpleNamespace(id=object()), "scheduled Job")

    assert str(captured.value) == "scheduled Job creation returned no ID"


def test_scheduled_spec_matching_treats_a_missing_command_as_drift(
    remote_spec: ExperimentSpec,
) -> None:
    request = _request(remote_spec)
    incomplete = SimpleNamespace(
        docker_image=request.remote.job.image,
        labels={
            "harbor-hf-role": "campaign-reconciler",
            "harbor-hf-namespace": "osolmaz",
        },
    )

    assert _scheduled_spec_matches(incomplete, request) is False
