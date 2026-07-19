from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

from harbor_hf.provider_models import ProviderTarget
from harbor_hf.provider_proxy import PROVIDER_RECORDER_PORT, ProviderEvidenceProxy
from harbor_hf.providers import routed_provider_model


class ProfileTransportError(RuntimeError):
    """Raised when a profile transport cannot be exposed safely."""


_HF_JOB_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class ProfileTransport:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        proxy: ProviderEvidenceProxy | None = None,
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.proxy = proxy

    @classmethod
    def for_endpoint(cls, base_url: str, model_name: str) -> ProfileTransport:
        return cls(base_url=base_url, model_name=model_name)

    @classmethod
    @contextmanager
    def for_provider(
        cls,
        target: ProviderTarget,
        *,
        token: str,
        evidence_path: Path,
        deadline: float,
    ) -> Iterator[ProfileTransport]:
        job_id = os.environ.get("JOB_ID", "")
        if not _HF_JOB_ID.fullmatch(job_id):
            raise ProfileTransportError("provider profiling requires a valid HF Job ID")
        proxy = ProviderEvidenceProxy(target, token=token, evidence_path=evidence_path)
        proxy.start(host="0.0.0.0", port=PROVIDER_RECORDER_PORT)
        base_url = f"https://{job_id}--{PROVIDER_RECORDER_PORT}.hf.jobs"
        try:
            _wait_ready(base_url, token, deadline)
            yield cls(
                base_url=base_url,
                model_name=routed_provider_model(target),
                proxy=proxy,
            )
        finally:
            proxy.close()

    @contextmanager
    def scope(self, scope: str) -> Iterator[tuple[str, str, str | None]]:
        if self.proxy is None:
            yield self.base_url, self.model_name, None
            return
        capability = self.proxy.register_scope(scope)
        try:
            yield (
                self.proxy.scoped_base_url(self.base_url, capability),
                self.model_name,
                capability,
            )
        finally:
            self.proxy.revoke_scope(capability)


def _wait_ready(base_url: str, token: str, deadline: float) -> None:
    ready_deadline = min(deadline, time.monotonic() + 120)
    last_failure = "no response"
    with httpx.Client(follow_redirects=False) as client:
        while time.monotonic() < ready_deadline:
            try:
                response = client.get(
                    f"{base_url}/healthz",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=min(5.0, max(0.1, ready_deadline - time.monotonic())),
                )
                if response.status_code in {401, 403}:
                    raise ProfileTransportError(
                        "provider profile ingress rejected HF authentication"
                    )
                if response.status_code == 200 and response.json() == {"status": "ok"}:
                    return
                last_failure = f"HTTP {response.status_code}"
            except (httpx.TransportError, ValueError) as error:
                last_failure = type(error).__name__
            time.sleep(min(1.0, max(0.0, ready_deadline - time.monotonic())))
    raise ProfileTransportError(
        "provider profile ingress readiness timed out: " + last_failure
    )
