# Harbor Compatibility Contract

This document freezes the Harbor assumptions used by `harbor-hf` while the
generic Harbor-owned execution protocol is developed upstream. New worker and
campaign executions use this adapter. Historical evidence can still be read by
the isolated legacy reader.

## Ownership Boundary

Harbor owns the job config, task resolution, agent and environment config,
trial execution, locks, results, verifier rewards, exceptions, timing, token
usage, and trial artifact inventory.

`harbor-hf` owns campaign and physical execution identity, Hugging Face
infrastructure, immutable request storage, endpoint cleanup, infrastructure
retries, policy checks, evidence publication, and normalized result rows.

Application modules call `FilesystemHarborExecutionAdapter`. They do not render
Harbor run flags or inspect Harbor result paths. All current Harbor-specific
knowledge lives under `harbor_hf.harbor_adapter`.

## Execution Input

Each physical execution writes two immutable files before Harbor starts:

- `harbor-job.json` is the exact `JobConfig` document consumed by Harbor.
- `harbor-request.json` contains that config, its SHA-256 digest, the pinned
  Harbor revision, and the independent `harbor-hf` verification policy.

The job config is the only source for attempts, concurrency, retry policy,
dataset and task selection, agent identity and parameters, model identity,
environment type and parameters, output path, and allowed model host. Internal
Harbor retries are fixed at zero. Agent concurrency equals trial concurrency.

The process command is deliberately small:

```text
uv run --project HARBOR_SOURCE --locked --no-dev --extra hf-sandbox \
  harbor run --config HARBOR_JOB_JSON --yes
```

Harbor receives `HF_TOKEN`, `OPENAI_API_KEY`, and `OPENAI_BASE_URL` in the
process environment. Secret values are not serialized into either input file.
The adapter checks both files byte-for-byte before execution and again before
accepting output.

## Compatibility Export

After Harbor exits, the adapter runs `harbor_adapter/exporter.py` with the same
pinned Harbor project environment. The exporter imports and validates Harbor's
own `JobLock`, `JobResult`, `TrialLock`, and `TrialResult` models. It emits
`harbor-compatibility.json` with schema
`harbor-hf/harbor-compatibility/v1alpha2`.

The bundle contains:

- Harbor source revision and package version;
- the immutable request digest;
- checksums and progress counts for each Harbor job lock and result;
- checksums, task and agent identity, model identity, exceptions, verifier
  rewards, timing, usage, and a typed private artifact inventory for each trial;
- no exception tracebacks, environment variables, agent config, task content,
  or secret values.

The exporter log is retained as `harbor-export.log`. The normal evidence
redaction, secret scan, checksums, and terminal-marker rules apply to the input,
bundle, log, and raw Harbor artifacts together.

The controller writes `private-artifacts.json` for every direct Harbor trial and
for each complete campaign physical execution. Entries are sorted,
private-only, size-bounded, and checksummed. A successful OpenClaw trial whose
agent execution started must include a session JSONL. Failed and timed-out
trials retain the same requirement record without turning incomplete evidence
into a score. Raw files and this private manifest cannot cross the normalized
result publication boundary.

## Additional Policy

The typed bundle is accepted only when all of these checks pass:

- Harbor revision and request digest match the immutable request;
- the number and names of trials match the fully resolved task set and attempt
  count, including wildcard selectors;
- every trial lock has the expected task content digest;
- agent name, agent version, model provider, and model name match the request;
- trial and multi-step exception fields are empty;
- every trial has at least one finite numeric verifier reward.

A nonzero Harbor exit with a typed trial exception preserves that exception for
campaign retry classification. Other malformed output from a failed process is
reported as the Harbor process failure. A zero exit without a valid typed
bundle cannot publish success or a score.

## Historical Reader

`harbor_hf.harbor_adapter.legacy` preserves the old `lock.json` and
`result.json` reader for existing evidence and audit tools. New execution paths
do not call it. It can be removed from new-write support only after the
Harbor-owned protocol satisfies the migration conditions in the
[refactor plan](harbor-integration-refactor.md).

## Behavioral Baseline

The remote baseline is campaign
`20260714T072108Z-7a2b167238-2bbc0a89fe`, produced with Harbor commit
`bd9e606dcb99eb49de70bd741fd846cae5c7ebd1` and OpenClaw `2026.6.11`.
Its evidence is stored under:

```text
hf://buckets/osolmaz/benchmark-runs/campaigns/
20260714T072108Z-7a2b167238-2bbc0a89fe
```

Exporter parity was checked against a preserved successful physical execution
from that campaign using the same Harbor commit. Harbor `0.17.1` validated one
job and one trial, including task digest, agent/model identity, rewards, usage,
and 12 artifact inventory entries.

The canonical job config was also accepted by the pinned Harbor `JobConfig`
parser through `harbor run --config ... --print-config`. No inference, model
weights, endpoint mutation, or remote benchmark was used for either check.
