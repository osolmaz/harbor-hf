from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class HarborVerificationPolicy(FrozenModel):
    expected_trials: int | None = Field(default=1, ge=1)
    expected_task_counts: dict[str, int] | None = None
    expected_attempts_per_task: int | None = Field(default=None, ge=1)
    expected_task_names: list[str] | None = None
    expected_task_digests: dict[str, Sha256Digest] | None = None
    expected_agent_name: str | None = None
    expected_agent_version: str | None = None
    expected_model_provider: str | None = None
    expected_model_name: str | None = None


class HarborExecutionRequest(FrozenModel):
    schema_version: str = "harbor-hf/harbor-execution-request/v1alpha1"
    harbor_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    harbor_config: dict[str, JsonValue]
    harbor_config_digest: Sha256Digest
    verification: HarborVerificationPolicy

    @model_validator(mode="after")
    def config_digest_matches_payload(self) -> HarborExecutionRequest:
        digest = sha256_digest(canonical_json_bytes(self.harbor_config))
        if self.harbor_config_digest != digest:
            raise ValueError("Harbor config digest does not match its payload")
        return self

    def config_bytes(self) -> bytes:
        return canonical_json_bytes(self.harbor_config) + b"\n"

    def request_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json")) + b"\n"


class HarborArtifactEntry(FrozenModel):
    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    digest: Sha256Digest


class HarborTimingSummary(FrozenModel):
    started_at: str | None = None
    finished_at: str | None = None


class HarborUsageSummary(FrozenModel):
    input_tokens: int | None = Field(default=None, ge=0)
    cache_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class HarborStepException(FrozenModel):
    step_name: str
    exception_type: str


class HarborCompatibilityTrial(FrozenModel):
    path: str = Field(min_length=1)
    lock_digest: Sha256Digest
    result_digest: Sha256Digest
    task_name: str = Field(min_length=1)
    task_digest: Sha256Digest
    agent_name: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)
    model_provider: str | None = None
    model_name: str | None = None
    exception_type: str | None = None
    step_exceptions: list[HarborStepException] = Field(default_factory=list)
    rewards: dict[str, int | float] | None = None
    timing: HarborTimingSummary
    usage: HarborUsageSummary
    artifacts: list[HarborArtifactEntry]


class HarborCompatibilityJob(FrozenModel):
    path: str = Field(min_length=1)
    lock_digest: Sha256Digest
    result_digest: Sha256Digest
    total_trials: int = Field(ge=0)
    completed_trials: int = Field(ge=0)
    errored_trials: int = Field(ge=0)


class HarborCompatibilityBundle(FrozenModel):
    schema_version: str = "harbor-hf/harbor-compatibility/v1alpha1"
    harbor_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    harbor_version: str = Field(min_length=1)
    request_digest: Sha256Digest
    jobs: list[HarborCompatibilityJob]
    trials: list[HarborCompatibilityTrial]


def ensure_no_policy_conflicts(
    config: Mapping[str, JsonValue], policy: HarborVerificationPolicy
) -> None:
    attempts = config.get("n_attempts")
    concurrency = config.get("n_concurrent_trials")
    retry = config.get("retry")
    datasets = config.get("datasets")
    agents = config.get("agents")
    if attempts != policy.expected_attempts_per_task:
        raise ValueError("Harbor request attempts disagree with verification policy")
    if not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("Harbor request concurrency must be positive")
    if not isinstance(retry, dict) or retry.get("max_retries") != 0:
        raise ValueError("Harbor retries must remain disabled for campaign execution")
    dataset = _only_mapping(datasets, "dataset")
    if dataset.get("task_names") != policy.expected_task_names:
        raise ValueError("Harbor request tasks disagree with verification policy")
    agent = _only_mapping(agents, "agent")
    if agent.get("name") != policy.expected_agent_name:
        raise ValueError("Harbor request agent disagrees with verification policy")
    if agent.get("n_concurrent") != concurrency:
        raise ValueError("Harbor agent concurrency must match trial concurrency")


def _only_mapping(value: JsonValue | None, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"Harbor request must select exactly one {label}")
    selected = value[0]
    if not isinstance(selected, dict):
        raise ValueError(f"Harbor {label} request is malformed")
    return selected
