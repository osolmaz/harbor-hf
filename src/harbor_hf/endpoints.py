from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Annotated, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from harbor_hf.models import DeploymentProfile, DeploymentTarget, ModelProfile

DeploymentDigest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
EndpointName = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")]
_MANAGED_TAG = "harbor-hf-managed"


class EndpointProvisioningError(RuntimeError):
    """Raised when an endpoint cannot be provisioned without losing safety."""


class EndpointNotFound(EndpointProvisioningError):
    """Raised when an endpoint disappears during an exact lifecycle operation."""


class EndpointProviderError(EndpointProvisioningError):
    """Raised for a provider failure known not to have applied a side effect."""


class AmbiguousEndpointCreate(EndpointProvisioningError):
    """Raised when create may have succeeded despite its failed response."""


class AmbiguousEndpointPause(EndpointProvisioningError):
    """Raised when pause may have succeeded despite its failed response."""


class AmbiguousEndpointDelete(EndpointProvisioningError):
    """Raised when delete may have succeeded despite its failed response."""


class EndpointVerificationTimeout(EndpointProvisioningError):
    """Raised when provider state does not converge before the deadline."""


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EndpointScaling(FrozenModel):
    min_replicas: int = Field(ge=0)
    max_replicas: int = Field(ge=1)
    scale_to_zero_timeout: int | None = Field(default=None, ge=0)
    metric: Literal["pendingRequests", "hardwareUsage"] | None = None
    threshold: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def scaling_is_consistent(self) -> EndpointScaling:
        if self.max_replicas < self.min_replicas:
            raise ValueError("maximum replicas must be at least minimum replicas")
        if (self.metric is None) != (self.threshold is None):
            raise ValueError("scaling metric and threshold must be configured together")
        return self


class EndpointCompute(FrozenModel):
    accelerator: str = Field(min_length=1)
    instance_size: str = Field(min_length=1)
    instance_type: str = Field(min_length=1)
    scaling: EndpointScaling


class EndpointProvider(FrozenModel):
    vendor: str = Field(min_length=1)
    region: str = Field(min_length=1)
    account_id: str | None = Field(default=None, min_length=1)


class EndpointImage(FrozenModel):
    url: str = Field(min_length=1)
    health_route: str = Field(pattern=r"^/[^?#]*$")
    port: int | None = Field(default=None, ge=1, le=65535)


class EndpointModel(FrozenModel):
    repository: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    framework: str = Field(min_length=1)
    task: str = Field(min_length=1)
    image: EndpointImage
    command: list[str]
    arguments: list[str]
    environment: dict[str, str]
    secret_names: list[str]

    @field_validator("secret_names")
    @classmethod
    def secret_names_are_canonical(cls, value: list[str]) -> list[str]:
        if any(not name for name in value):
            raise ValueError("secret names must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("secret names must be unique")
        return sorted(value)


class EndpointRoute(FrozenModel):
    domain: str | None = Field(default=None, min_length=1)
    path: str | None = Field(default=None, pattern=r"^/[^?#]*$")


class EndpointConfiguration(FrozenModel):
    model: EndpointModel
    compute: EndpointCompute
    provider: EndpointProvider
    access_type: Literal["public", "authenticated", "private"]
    route: EndpointRoute
    cache_http_responses: bool
    tags: list[str]

    @field_validator("tags")
    @classmethod
    def tags_are_canonical(cls, value: list[str]) -> list[str]:
        if any(not tag for tag in value):
            raise ValueError("endpoint tags must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("endpoint tags must be unique")
        return sorted(value)


class ManagedEndpointIdentity(FrozenModel):
    namespace: str = Field(min_length=1)
    name: EndpointName
    campaign_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    deployment_digest: DeploymentDigest
    tags: list[str]

    @field_validator("tags")
    @classmethod
    def managed_tags_are_canonical(cls, value: list[str]) -> list[str]:
        return sorted(value)

    @model_validator(mode="after")
    def identity_is_deterministic(self) -> ManagedEndpointIdentity:
        expected_name, expected_tags = _managed_identity_values(
            self.namespace, self.campaign_id, self.deployment_digest
        )
        if self.name != expected_name or self.tags != expected_tags:
            raise ValueError("managed endpoint identity is not deterministic")
        return self


class DesiredEndpoint(FrozenModel):
    identity: ManagedEndpointIdentity
    configuration: EndpointConfiguration

    @model_validator(mode="after")
    def configuration_has_managed_tags(self) -> DesiredEndpoint:
        missing = set(self.identity.tags) - set(self.configuration.tags)
        if missing:
            raise ValueError("endpoint configuration is missing managed identity tags")
        return self


class EndpointStatus(FrozenModel):
    state: str = Field(min_length=1)
    ready_replicas: int = Field(ge=0)
    target_replicas: int = Field(ge=0)
    url: str | None = Field(default=None, min_length=1)


class EndpointSnapshot(FrozenModel):
    namespace: str = Field(min_length=1)
    name: EndpointName
    configuration: EndpointConfiguration
    status: EndpointStatus


class ConfigurationMismatch(FrozenModel):
    path: str
    expected: str
    observed: str


class EndpointConfigurationMismatch(EndpointProvisioningError):
    def __init__(self, mismatches: Sequence[ConfigurationMismatch]) -> None:
        self.mismatches = tuple(mismatches)
        paths = ", ".join(mismatch.path for mismatch in self.mismatches)
        super().__init__(f"endpoint effective configuration mismatch: {paths}")


class EndpointIdentityMismatch(EndpointProvisioningError):
    """Raised when a deterministic name resolves to an unmanaged endpoint."""


class EndpointNotPaused(EndpointProvisioningError):
    """Raised when adoption or deletion would interfere with an active endpoint."""


class ProvisioningResult(FrozenModel):
    action: Literal["created", "adopted"]
    snapshot: EndpointSnapshot


class EndpointProvisioningPort(Protocol):
    def create(self, desired: DesiredEndpoint) -> EndpointSnapshot: ...

    def inspect(self, identity: ManagedEndpointIdentity) -> EndpointSnapshot | None: ...

    def pause(self, identity: ManagedEndpointIdentity) -> EndpointSnapshot: ...

    def delete(self, identity: ManagedEndpointIdentity) -> None: ...


class EndpointSettings(FrozenModel):
    accelerator: str = Field(default="gpu", min_length=1)
    instance_type: str | None = Field(default=None, min_length=1)
    min_replicas: int = Field(default=1, ge=0)
    max_replicas: int = Field(default=1, ge=1)
    scale_to_zero_timeout: int | None = Field(default=None, ge=0)
    scaling_metric: Literal["pendingRequests", "hardwareUsage"] | None = None
    scaling_threshold: float | None = Field(default=None, gt=0)
    framework: str = Field(default="custom", min_length=1)
    task: str = Field(default="text-generation", min_length=1)
    health_route: str = Field(default="/health", pattern=r"^/[^?#]*$")
    port: int | None = Field(default=None, ge=1, le=65535)
    endpoint_type: Literal["public", "authenticated", "private"] = "authenticated"
    account_id: str | None = Field(default=None, min_length=1)
    domain: str | None = Field(default=None, min_length=1)
    path: str | None = Field(default=None, pattern=r"^/[^?#]*$")
    cache_http_responses: bool = False
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def settings_are_consistent(self) -> EndpointSettings:
        EndpointScaling(
            min_replicas=self.min_replicas,
            max_replicas=self.max_replicas,
            scale_to_zero_timeout=self.scale_to_zero_timeout,
            metric=self.scaling_metric,
            threshold=self.scaling_threshold,
        )
        return self


def deployment_digest(
    model: ModelProfile, deployment: DeploymentTarget
) -> DeploymentDigest:
    return _digest(
        {
            "model": model.model_dump(mode="json", exclude={"id"}, exclude_none=True),
            "deployment": deployment.model_dump(
                mode="json",
                exclude={"id", "endpoint"},
                exclude_none=True,
            ),
        }
    )


def managed_endpoint_identity(
    *, namespace: str, campaign_id: str, deployment_digest: DeploymentDigest
) -> ManagedEndpointIdentity:
    name, tags = _managed_identity_values(namespace, campaign_id, deployment_digest)
    return ManagedEndpointIdentity(
        namespace=namespace,
        name=name,
        campaign_id=campaign_id,
        deployment_digest=deployment_digest,
        tags=tags,
    )


def _managed_identity_values(
    namespace: str, campaign_id: str, deployment_digest: DeploymentDigest
) -> tuple[str, list[str]]:
    identity_hash = _digest(
        {
            "namespace": namespace,
            "campaign_id": campaign_id,
            "deployment_digest": deployment_digest,
        }
    ).removeprefix("sha256:")
    digest_hash = deployment_digest.removeprefix("sha256:")
    campaign_hash = hashlib.sha256(campaign_id.encode()).hexdigest()
    return (
        f"harbor-hf-{identity_hash[:40]}",
        sorted(
            [
                _MANAGED_TAG,
                f"harbor-hf-campaign-{campaign_hash[:24]}",
                f"harbor-hf-deployment-{digest_hash[:24]}",
            ]
        ),
    )


def build_desired_endpoint(
    *,
    namespace: str,
    campaign_id: str,
    model: ModelProfile,
    deployment: DeploymentProfile,
) -> DesiredEndpoint:
    digest = deployment_digest(model, deployment)
    identity = managed_endpoint_identity(
        namespace=namespace,
        campaign_id=campaign_id,
        deployment_digest=digest,
    )
    settings = EndpointSettings.model_validate(deployment.parameters)
    vendor, separator, region = deployment.region.partition("-")
    if not separator or not region:
        raise ValueError("deployment region must use vendor-region form")
    configuration = EndpointConfiguration(
        model=EndpointModel(
            repository=model.repo,
            revision=model.revision,
            framework=settings.framework,
            task=settings.task,
            image=EndpointImage(
                url=deployment.engine.image,
                health_route=settings.health_route,
                port=settings.port,
            ),
            command=deployment.engine.command,
            arguments=deployment.engine.arguments,
            environment=deployment.engine.environment,
            secret_names=deployment.engine.secret_names,
        ),
        compute=EndpointCompute(
            accelerator=settings.accelerator,
            instance_size=f"x{deployment.accelerator_count}",
            instance_type=(settings.instance_type or f"nvidia-{deployment.hardware}"),
            scaling=EndpointScaling(
                min_replicas=settings.min_replicas,
                max_replicas=settings.max_replicas,
                scale_to_zero_timeout=settings.scale_to_zero_timeout,
                metric=settings.scaling_metric,
                threshold=settings.scaling_threshold,
            ),
        ),
        provider=EndpointProvider(
            vendor=vendor,
            region=region,
            account_id=settings.account_id,
        ),
        access_type=settings.endpoint_type,
        route=EndpointRoute(domain=settings.domain, path=settings.path),
        cache_http_responses=settings.cache_http_responses,
        tags=sorted([*settings.tags, *identity.tags]),
    )
    return DesiredEndpoint(identity=identity, configuration=configuration)


def effective_configuration_mismatches(
    expected: EndpointConfiguration, observed: EndpointConfiguration
) -> tuple[ConfigurationMismatch, ...]:
    mismatches: list[ConfigurationMismatch] = []
    _compare_values(
        expected.model_dump(mode="json"),
        observed.model_dump(mode="json"),
        path="configuration",
        mismatches=mismatches,
    )
    return tuple(mismatches)


def verify_exact_endpoint(desired: DesiredEndpoint, snapshot: EndpointSnapshot) -> None:
    if (
        snapshot.namespace != desired.identity.namespace
        or snapshot.name != desired.identity.name
        or not set(desired.identity.tags).issubset(snapshot.configuration.tags)
    ):
        raise EndpointIdentityMismatch(
            "endpoint does not have the expected deterministic managed identity"
        )
    mismatches = effective_configuration_mismatches(
        desired.configuration, snapshot.configuration
    )
    if mismatches:
        raise EndpointConfigurationMismatch(mismatches)


def require_paused_zero_ready(snapshot: EndpointSnapshot) -> None:
    status = snapshot.status
    if status.state != "paused" or status.ready_replicas != 0:
        raise EndpointNotPaused(
            "endpoint must report state=paused and readyReplica=0; "
            f"observed state={status.state!r}, readyReplica={status.ready_replicas}, "
            f"targetReplica={status.target_replicas}"
        )


class EndpointProvisioner:
    def __init__(
        self,
        port: EndpointProvisioningPort,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.port = port
        self.sleep = sleep
        self.monotonic = monotonic

    def inspect(self, desired: DesiredEndpoint) -> EndpointSnapshot | None:
        snapshot = self.port.inspect(desired.identity)
        if snapshot is not None:
            verify_exact_endpoint(desired, snapshot)
        return snapshot

    def create_or_adopt(
        self,
        desired: DesiredEndpoint,
        *,
        timeout_seconds: float = 300,
        poll_seconds: float = 5,
    ) -> ProvisioningResult:
        existing = self.inspect(desired)
        if existing is not None:
            require_paused_zero_ready(existing)
            return ProvisioningResult(action="adopted", snapshot=existing)
        action: Literal["created", "adopted"] = "created"
        try:
            snapshot = self.port.create(desired)
        except AmbiguousEndpointCreate:
            action = "adopted"
            snapshot = self._wait_until_present(
                desired,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
            )
        try:
            verify_exact_endpoint(desired, snapshot)
        except (EndpointIdentityMismatch, EndpointConfigurationMismatch):
            self._pause_created_identity(
                desired.identity,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
            )
            raise
        try:
            paused = self.pause_and_verify(
                desired,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
            )
        except (EndpointIdentityMismatch, EndpointConfigurationMismatch):
            try:
                self._pause_created_identity(
                    desired.identity,
                    timeout_seconds=timeout_seconds,
                    poll_seconds=poll_seconds,
                )
            except EndpointProvisioningError as cleanup_error:
                raise AmbiguousEndpointPause(
                    "created endpoint cleanup is not verified and must be retried"
                ) from cleanup_error
            raise
        except EndpointProvisioningError:
            try:
                paused = self._pause_created_identity(
                    desired.identity,
                    timeout_seconds=timeout_seconds,
                    poll_seconds=poll_seconds,
                )
                verify_exact_endpoint(desired, paused)
            except EndpointProvisioningError as cleanup_error:
                raise AmbiguousEndpointPause(
                    "created endpoint cleanup is not verified and must be retried"
                ) from cleanup_error
        return ProvisioningResult(action=action, snapshot=paused)

    def pause_and_verify(
        self,
        desired: DesiredEndpoint,
        *,
        timeout_seconds: float = 300,
        poll_seconds: float = 5,
    ) -> EndpointSnapshot:
        current = self.inspect(desired)
        if current is None:
            raise EndpointNotFound("managed endpoint does not exist")
        if _is_paused_zero_ready(current):
            return current
        with suppress(AmbiguousEndpointPause):
            self.port.pause(desired.identity)
        return self._wait_until_paused(
            desired,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )

    def delete(
        self,
        desired: DesiredEndpoint,
        *,
        timeout_seconds: float = 300,
        poll_seconds: float = 5,
    ) -> bool:
        current = self.inspect(desired)
        if current is None:
            return False
        require_paused_zero_ready(current)
        with suppress(AmbiguousEndpointDelete):
            self.port.delete(desired.identity)
        self._wait_until_deleted(
            desired.identity,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        return True

    def _wait_until_present(
        self,
        desired: DesiredEndpoint,
        *,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> EndpointSnapshot:
        deadline = _validated_deadline(self.monotonic, timeout_seconds, poll_seconds)
        while True:
            snapshot = self.inspect(desired)
            if snapshot is not None:
                return snapshot
            if not _sleep_before_deadline(
                self.sleep, self.monotonic, deadline, poll_seconds
            ):
                raise EndpointVerificationTimeout(
                    "ambiguous endpoint create could not be adopted before timeout"
                )

    def _wait_until_paused(
        self,
        desired: DesiredEndpoint,
        *,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> EndpointSnapshot:
        deadline = _validated_deadline(self.monotonic, timeout_seconds, poll_seconds)
        while True:
            snapshot = self.inspect(desired)
            if snapshot is None:
                raise EndpointNotFound("managed endpoint disappeared while pausing")
            if _is_paused_zero_ready(snapshot):
                return snapshot
            if not _sleep_before_deadline(
                self.sleep, self.monotonic, deadline, poll_seconds
            ):
                status = snapshot.status
                raise EndpointVerificationTimeout(
                    "endpoint pause did not reach state=paused and readyReplica=0; "
                    f"state={status.state!r}, readyReplica={status.ready_replicas}, "
                    f"targetReplica={status.target_replicas}"
                )

    def _pause_created_identity(
        self,
        identity: ManagedEndpointIdentity,
        *,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> EndpointSnapshot:
        with suppress(AmbiguousEndpointPause):
            self.port.pause(identity)
        deadline = _validated_deadline(self.monotonic, timeout_seconds, poll_seconds)
        while True:
            snapshot = self.port.inspect(identity)
            if snapshot is None:
                raise EndpointNotFound("created endpoint disappeared during cleanup")
            if _is_paused_zero_ready(snapshot):
                return snapshot
            if not _sleep_before_deadline(
                self.sleep, self.monotonic, deadline, poll_seconds
            ):
                raise EndpointVerificationTimeout(
                    "created endpoint configuration was invalid and cleanup was "
                    "not verified"
                )

    def _wait_until_deleted(
        self,
        identity: ManagedEndpointIdentity,
        *,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> None:
        deadline = _validated_deadline(self.monotonic, timeout_seconds, poll_seconds)
        while self.port.inspect(identity) is not None:
            if not _sleep_before_deadline(
                self.sleep, self.monotonic, deadline, poll_seconds
            ):
                raise EndpointVerificationTimeout(
                    "endpoint deletion was not verified before timeout"
                )


def _compare_values(
    expected: object,
    observed: object,
    *,
    path: str,
    mismatches: list[ConfigurationMismatch],
) -> None:
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        for key in sorted(set(expected) | set(observed)):
            _compare_values(
                expected.get(key),
                observed.get(key),
                path=f"{path}.{key}",
                mismatches=mismatches,
            )
        return
    if expected != observed:
        mismatches.append(
            ConfigurationMismatch(
                path=path,
                expected=_canonical_display(expected),
                observed=_canonical_display(observed),
            )
        )


def _canonical_display(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> DeploymentDigest:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _is_paused_zero_ready(snapshot: EndpointSnapshot) -> bool:
    return snapshot.status.state == "paused" and snapshot.status.ready_replicas == 0


def _validated_deadline(
    monotonic: Callable[[], float], timeout_seconds: float, poll_seconds: float
) -> float:
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("endpoint timeouts and poll intervals must be positive")
    return monotonic() + timeout_seconds


def _sleep_before_deadline(
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    deadline: float,
    poll_seconds: float,
) -> bool:
    remaining = deadline - monotonic()
    if remaining <= 0:
        return False
    sleep(min(poll_seconds, remaining))
    return True
