# Harbor Integration Refactor Plan

## Status

Phases 0 through 3 are implemented by the Harbor execution adapter. New
executions use one checksummed Harbor job config and a typed compatibility
bundle exported inside the pinned Harbor environment. The legacy filesystem
reader remains available only for historical evidence.

Phases 4 through 9 remain planned because they require a released,
Harbor-owned execution protocol. They must not be simulated as a private
`harbor-hf` protocol: doing that would recreate the ownership duplication this
refactor is intended to remove. See the
[frozen compatibility contract](harbor-integration-contract.md) for the exact
implemented boundary and parity evidence.

## Goal

Make Harbor the single authority for benchmark configuration and trial output,
while keeping `harbor-hf` responsible for Hugging Face infrastructure,
campaign recovery, evidence storage, and result publication.

The refactor must not block benchmark execution. The current pinned Harbor CLI
and filesystem adapter remains supported until the replacement has passed local
contract tests and remote campaign parity tests.

## Current Boundary

Today a `harbor-hf` worker:

1. resolves campaign, model, deployment, agent, and task configuration;
2. builds a Harbor CLI command;
3. runs the pinned Harbor checkout as a subprocess;
4. reads Harbor `lock.json` and `result.json` files from known paths;
5. independently validates task identity, agent identity, exceptions, and
   verifier rewards;
6. wraps the result in campaign execution evidence and publishes artifacts.

This works and keeps Harbor internals out of the control plane, but it copies
knowledge of Harbor configuration fields, result fields, and directory layout.
That knowledge can drift when Harbor changes.

## Target Boundary

The final interaction should use one small, versioned Harbor-owned contract:

```text
harbor-hf campaign lock
        |
        +-- Hugging Face infrastructure envelope
        |
        +-- Harbor execution request
                    |
                    v
              Harbor execution
                    |
                    +-- lifecycle events
                    +-- trial result bundles
                    +-- artifact manifests
```

Harbor remains authoritative for:

- task, agent, environment, verifier, and Harbor job configuration;
- logical attempts and trial identity within one Harbor execution;
- agent sessions, trajectories, verifier output, exceptions, and timing;
- Harbor job and trial locks;
- the file inventory produced by a Harbor execution.

`harbor-hf` remains authoritative for:

- campaign, run, shard, wave, and physical execution identity;
- model weights, serving engine, endpoint or provider, hardware, and region;
- HF Jobs, Sandboxes, Endpoints, Buckets, Datasets, and lifecycle cleanup;
- cross-campaign admission, spending limits, and infrastructure retries;
- immutable evidence publication and normalized cross-run results.

The `harbor-hf` execution record references the Harbor result bundle and its
checksum. It does not maintain an independent copy of every Harbor field.

## Invariants

The refactor must preserve these rules in every phase:

1. Harbor performs one requested logical attempt for each assigned trial.
2. Harbor's internal infrastructure retries are disabled for campaign work.
3. `harbor-hf` creates a new physical execution for every infrastructure retry.
4. A completed Harbor bundle is immutable and belongs to exactly one physical
   execution.
5. Partial or checkpoint evidence cannot produce a score, `_SUCCESS`, or a
   public result row.
6. Secret values and unsanitized task artifacts never enter shared storage.
7. Endpoint cleanup remains part of run correctness.
8. Existing campaign locks and evidence remain readable throughout migration.

## Phase 0: Freeze The Existing Contract

Status: complete.

Purpose: capture the working behavior before moving code.

Deliverables:

- inventory every Harbor CLI flag, environment variable, expected output path,
  and result field used by `harbor-hf`;
- record the exact supported Harbor source commit and OpenClaw version;
- add sanitized golden fixtures for a successful trial, handled trial failure,
  infrastructure failure, retry, and multi-step result;
- preserve the completed remote campaign as the behavioral baseline;
- document which checks are Harbor guarantees and which are additional
  `harbor-hf` policy.

Exit criteria: a compatibility test can explain every Harbor assumption used by
the current worker and fails when any fixture changes unexpectedly.

## Phase 1: Isolate The Current Adapter

Status: complete.

Purpose: make future replacement local to one module without changing behavior.

Deliverables:

- introduce a narrow `HarborExecutionAdapter` boundary;
- move command rendering, subprocess invocation, output discovery, and raw
  Harbor result loading behind that boundary;
- return typed adapter outcomes instead of unstructured dictionaries;
- prevent campaign, recovery, and publication modules from reading Harbor file
  paths directly;
- retain the current command and filesystem behavior exactly.

Tests:

- golden command tests;
- golden result and lock parsing tests;
- path-layout rejection tests;
- parity tests proving the refactor emits byte-equivalent terminal evidence.

Exit criteria: changing Harbor integration requires editing one adapter package,
and the remote command and published evidence are unchanged.

## Phase 2: Use One Canonical Harbor Request

Status: complete.

Purpose: stop reconstructing Harbor configuration in multiple places.

Deliverables:

- resolve each campaign assignment into one serialized Harbor job or trial
  configuration accepted by the pinned Harbor version;
- store that exact configuration and its checksum in the private execution
  input bundle;
- derive the CLI invocation from the serialized configuration rather than from
  a second collection of command flags;
- keep infrastructure-only values in the surrounding `harbor-hf` lock;
- reject attempts, retries, concurrency, task selection, or agent settings that
  disagree with campaign policy.

Tests:

- round-trip validation through the pinned Harbor configuration model;
- digest and tamper tests;
- rejection tests for conflicting retry, attempt, task, and concurrency values;
- comparison tests against the current generated commands.

Exit criteria: the request validated by Harbor is the same immutable request
recorded by `harbor-hf`; no second configuration is reconstructed at runtime.

## Phase 3: Add A Typed Compatibility Exporter

Status: complete.

Purpose: remove broad filesystem and dictionary knowledge before upstream APIs
are available.

Deliverables:

- run a small exporter inside the pinned Harbor environment after execution;
- validate Harbor locks and results with that checkout's own public Pydantic
  models or generated JSON Schemas;
- emit one versioned, sanitized compatibility bundle for `harbor-hf`;
- include Harbor version, schema version, lock checksum, trial identity,
  result, exception, verifier, timing, token usage, and artifact inventory;
- keep additional `harbor-hf` checks for expected task digest, agent identity,
  result completeness, and finite rewards;
- keep the legacy parser as a fallback reader for historical evidence only.

Tests:

- exporter fixtures across every supported Harbor version;
- malformed and unknown schema tests;
- equality tests between legacy parsing and exported bundles;
- remote smoke comparison using separate campaign IDs.

Exit criteria: new executions no longer require `harbor-hf` application code to
understand Harbor's internal directory layout.

## Phase 4: Upstream Generic Harbor Support

Status: waiting on upstream Harbor work and release adoption.

Purpose: replace the compatibility exporter with a stable Harbor-owned
contract.

Upstream work:

- merge the HF Sandbox environment from Harbor PR 1925;
- upstream explicit Bash execution for HF Sandbox commands;
- upstream OpenClaw configuration upload for non-mounted environments;
- propose a versioned execution request and result bundle owned by Harbor;
- propose structured trial lifecycle events and an artifact-sink interface;
- keep endpoint, campaign, Bucket, Dataset, and Hugging Face control-plane code
  out of Harbor core.

The Harbor proposal should be useful to other remote orchestrators and storage
backends. It must not mention `harbor-hf` state or require Harbor to understand
campaign waves and endpoints.

Exit criteria: a released Harbor version can accept the execution request and
produce the result bundle without `harbor-hf` parsing Harbor internals.

## Phase 5: Adopt The Harbor-Owned Protocol

Status: blocked by Phase 4.

Purpose: switch new campaigns to the stable upstream contract safely.

Deliverables:

- negotiate protocol and capability versions before remote mutation;
- fail before endpoint resume when the pinned Harbor version is incompatible;
- support both compatibility and Harbor-owned bundles during migration;
- run the new reader in shadow mode and compare its normalized result with the
  legacy reader;
- record the selected Harbor protocol and capabilities in execution evidence;
- publish equivalent result rows regardless of which supported reader produced
  the validated bundle.

Exit criteria: at least two complete remote campaigns produce equivalent locks,
trial outcomes, artifact inventories, and normalized rows through both readers.

## Phase 6: Run Harbor Once Per Shard

Status: planned after protocol adoption.

Purpose: let Harbor own concurrent trial execution without losing campaign-level
recovery semantics.

Deliverables:

- submit one Harbor execution request containing a bounded shard of trials;
- let Harbor enforce the shard's trial and agent concurrency;
- persist each trial's lifecycle events and completed result bundle as soon as
  they become available;
- map every Harbor trial bundle to its `harbor-hf` physical execution ID;
- after interruption, retain completed bundles and create new physical
  executions only for incomplete trials;
- keep Harbor retries disabled so physical retry history remains explicit.

Tests:

- shard completion and mixed-outcome tests;
- controller and Sandbox kills after every completed-trial count;
- duplicate and delayed lifecycle events;
- proof that completed trials are never rerun after recovery;
- concurrency and throughput comparison against per-trial Harbor subprocesses.

Exit criteria: a killed shard resumes only missing trials and uses fewer Harbor
process startups without changing scores or evidence identity.

## Phase 7: Incremental Artifact Checkpoints

Status: planned after protocol adoption.

Purpose: connect the planned in-progress evidence milestone to Harbor's public
artifact interface.

Deliverables:

- consume Harbor artifact events through the same versioned protocol;
- publish sanitized, checksum-valid private checkpoints while a trial runs;
- preserve the newest valid checkpoint after abrupt worker or Sandbox loss;
- keep checkpoint evidence explicitly incomplete and excluded from scoring;
- compact or link checkpoints into terminal evidence under bounded retention
  rules.

Exit criteria: the Milestone 8 exit criteria in the production campaign plan
pass without Harbor-specific path scraping.

## Phase 8: Pin Released Worker Images

Status: planned after protocol adoption.

Purpose: remove per-Job source bootstrapping from the normal production path.

Deliverables:

- build a digest-pinned worker image containing released `harbor` and
  `harbor-hf` packages plus locked dependencies;
- record package versions, source commits, image digest, lockfile checksums, and
  software bill of materials in runtime evidence;
- run an image smoke before promotion;
- retain a source-checkout mode for development and historical replay;
- reject mutable image tags and incompatible package combinations.

Exit criteria: production Jobs start from one verified image without cloning or
resolving packages, and reproduce the same result as source-checkout mode.

## Phase 9: Remove Legacy Writes

Status: planned after the migration evidence and release-window conditions
below are satisfied.

Purpose: finish migration without losing historical readability.

Removal conditions:

- the Harbor-owned protocol has shipped in a supported Harbor release;
- two complete remote campaigns and one forced-recovery campaign passed parity;
- all active worker images use the new protocol;
- no new evidence has used the legacy writer for one release window;
- rebuild and audit commands can still read historical legacy evidence.

Actions:

- remove legacy command reconstruction and new-write filesystem parsing;
- retain versioned legacy readers for audit and migration;
- remove duplicated Harbor fields from new lock schema versions;
- document the minimum supported Harbor and protocol versions.

Exit criteria: all new execution data crosses one versioned Harbor boundary,
while historical campaigns remain auditable and rebuildable.

## Compatibility And Rollback

- Every protocol and lock change creates a new schema version; historical bytes
  are never rewritten.
- Campaigns remain pinned to the Harbor version and adapter selected at submit
  time.
- A failed migration rolls back by selecting the previous worker image and
  adapter for new campaign IDs; active campaigns keep their locked runtime.
- Readers remain additive until the removal conditions above are met.
- Rollback never changes endpoint cleanup, evidence checksums, or publication
  identity.

## Measures Of Success

The refactor is complete when:

- new `harbor-hf` code contains no Harbor output path discovery outside the
  compatibility package;
- campaign locks contain one canonical Harbor request digest instead of copied
  Harbor configuration fields;
- Harbor validates and emits every new trial result bundle;
- completed trials survive shard-worker interruption without rerun;
- the same raw evidence rebuilds the same normalized result rows;
- unsupported Harbor versions fail before billable resources start;
- production Jobs use digest-pinned release images;
- endpoint cleanup and private evidence guarantees remain unchanged.

## Work Available Before Upstream Merge

Phases 0 through 3 can be completed entirely in `harbor-hf` while the existing
Harbor pin remains in place. The HF Sandbox and OpenClaw fixes can be proposed
upstream in parallel. Phases 5 through 9 should begin only after Harbor accepts
or publishes the required stable contracts.
