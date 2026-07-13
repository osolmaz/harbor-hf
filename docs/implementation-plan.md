# Production Campaign Implementation Plan

## Goal

Extend the proven single-run controller into a remote-only, resumable system
that can run many Harbor benchmarks across models, serving engines, hardware,
and inference providers. The system must preserve complete evidence, publish
comparable results, recover after control-plane failure, and leave no paid
Inference Endpoint active when it has no assigned work.

This plan is ordered by dependency, not calendar time. Each milestone must be
independently releasable and must preserve the current single-run safety
properties.

## Starting Point

The merged implementation already provides the execution kernel for one
resolved matrix cell:

- immutable source, model, image, dataset, task, and agent references;
- permanent run reservations and compare-and-swap endpoint leases;
- a remote HF Job controller and independent endpoint watchdog;
- Harbor execution in HF Sandboxes through public Harbor APIs;
- exact endpoint and trial identity verification;
- Job-local artifact staging, redaction, checksums, and private Bucket
  publication;
- terminal success or failure markers written after verified endpoint cleanup;
- endpoint-backed execution without local inference or task execution.

The next implementation must reuse this kernel. It must not create a second
execution path with weaker validation or cleanup behavior.

## Architectural Decisions

### Reconciliation Instead Of A Long-Running Server

Campaign orchestration is implemented as a stateless reconciler. Each pass:

1. reads immutable campaign plans and append-only events;
2. inspects current HF Jobs, Inference Endpoints, and provider state;
3. derives a projection of campaigns, runs, shards, and deployment waves;
4. reserves a bounded set of idempotent actions;
5. performs those actions and records their outcomes;
6. exits.

A Hub webhook provides prompt reconciliation after control-repository updates.
A scheduled CPU Job provides the recovery path when a webhook is delayed,
missed, or fails. Correctness must not depend on either process retaining
memory between passes.

### Bounded Deployment Waves

A deployment wave is the unit that owns endpoint lifecycle. It groups a bounded
set of compatible shards with one exact deployment digest: model,
revision, engine, image, arguments, environment, hardware, region, scaling,
and runtime policy.

One endpoint-backed wave:

1. acquires the endpoint lease and starts its watchdog;
2. adopts or creates the exact endpoint deployment;
3. resumes and verifies the endpoint once;
4. runs assigned Harbor shards at the locked concurrency;
5. stops admitting work when its duration, cost, or shard bound is reached;
6. drains active work, pauses the endpoint, and verifies zero ready replicas;
7. publishes terminal evidence and releases the lease.

This amortizes model startup across compatible shards without allowing
unbounded endpoint reuse. Endpoint reuse is limited to one campaign by default.
Cross-campaign reuse requires a later explicit policy and is not inferred from
matching endpoint names.

Provider-backed waves use the same run, shard, trial, execution, and artifact
contracts but have no endpoint lease. They record only runtime details exposed
by the provider.

### Storage Responsibilities

Each HF storage primitive has one purpose:

| Store | Contents | Mutation model |
| --- | --- | --- |
| Private control Dataset | Campaign plans, reservations, leases, run-level events, and publisher cursors | Small parent-checked commits |
| Private artifact Bucket | Sessions, logs, Harbor trees, trajectories, archives, and checksums | Unique prefixes with terminal markers |
| Benchmark result Dataset | Normalized run, trial, execution, metric, and artifact tables | Serialized publisher commits |
| Global index Dataset | One discoverable row per published run | Serialized publisher commits |
| Optional Space | Read-only views and authenticated submission requests | No authoritative state |

The Bucket remains the canonical evidence store. Result Datasets are derived,
rebuildable query and presentation layers. No raw session is published to a
public Dataset.

### Ports Around Hugging Face

Domain planning and reconciliation must not depend directly on HF SDK models.
Use narrow typed ports for:

- control state and compare-and-swap commits;
- Jobs submission, inspection, and cancellation;
- endpoint provisioning and lifecycle;
- Inference Provider requests and quota observations;
- Bucket publication and artifact inspection;
- result publication;
- clock and identifier generation.

HF adapters validate untrusted response data at their boundary. The existing
controller behavior moves behind these ports incrementally; there is no
big-bang package rewrite.

## Durable Domain Model

The initial schema is deliberately small. Schema versioning is required before
the first campaign is submitted.

| Entity | Identity | Meaning |
| --- | --- | --- |
| Experiment | User-defined ID plus manifest digest | Requested matrix and policy |
| Campaign plan | Content digest | Fully resolved immutable execution plan |
| Campaign | Generated ID plus plan digest | One submitted execution of a plan |
| Run | Digest of campaign ID and resolved matrix cell | One homogeneous benchmark configuration |
| Shard | Digest of run ID and ordered task-attempt set | Bounded schedulable work |
| Trial | Digest of run ID, task digest, and logical attempt | Benchmark-semantic attempt |
| Execution | Generated ID scoped to one trial | One physical invocation of a logical trial |
| Deployment wave | Generated ID plus deployment digest | Bounded endpoint or provider ownership session |
| Artifact | Digest of typed owner, path, and content | Checksummed evidence object |
| Event | Generated ID, typed subject, kind, and schema version | Append-only state transition evidence |

Submitting the same plan twice creates two campaigns. It does not overwrite or
silently adopt the first campaign. Within a campaign, deterministic run, shard,
and trial IDs make repeated reconciliation idempotent. An infrastructure retry
creates a new execution ID and never changes the logical trial identity.

The plan digest is computed from canonical plan content and is stored in its
envelope and references, not inside the bytes being hashed. Schema versions
belong to independently serialized documents, event envelopes, and table
schemas; they are not part of domain identity. Artifact ownership uses an
explicit `owner_type` and `owner_id` pair.

A shard is only a scheduling batch. An execution belongs to one trial, not to
the shard that happened to schedule it. Retrying a lost shard creates physical
executions only for trials that lack a valid completed execution; completed
trials are not repeated.

### State Projections

Events are authoritative; status fields are rebuildable projections.

Campaign projection:

```text
queued -> active -> draining -> completed
                |          |-> partial
                |          |-> failed
                |          |-> cancelled
                -> cancel_requested -> draining
```

Run and shard projection:

```text
planned -> queued -> active -> verifying -> publishing -> complete
                    |                         |-> invalid
                    |-> retry_wait            |-> failed_infrastructure
                    |-> cancelled
```

Deployment-wave projection:

```text
planned -> acquiring -> provisioning -> ready -> active -> draining
                                                        -> cleaning -> closed
                                                                    -> cleanup_failed
```

A verifier reward of zero is a valid completed benchmark result. `invalid`
means evidence or benchmark semantics failed validation. `failed_infrastructure`
means no valid benchmark result was produced after the allowed physical
executions.

### Event Rules

- Events are immutable and include `schema_version`, `event_id`, `subject_type`,
  `subject_id`, `kind`, `observed_at`, `producer`, and a typed payload.
- `subject_type` identifies the referenced entity kind. `producer` identifies
  the component that recorded the event, such as the reconciler, watchdog,
  wave controller, publisher, or CLI.
- `observed_at` is the controller observation time. A provider-supplied event
  time, when available, is a separate typed payload field.
- Provider timestamps are evidence fields, not replacements for controller
  observation time.
- Event consumers ignore unknown optional fields and reject unsupported major
  schema versions.
- Reconciliation actions have deterministic reservation IDs. A second pass
  adopts the reservation or its remote resource instead of repeating the side
  effect.
- No global lock is held while making a slow provider request. Reserve, release
  the control-store commit, perform the request, then append the result.

The entity schemas and state-machine events must receive a dedicated data-model
review before they are frozen in Milestone 1.

## Control Repository Layout

The control Dataset stores only small coordination records:

```text
schema/
  current.json
campaigns/<campaign-id>/
  request.yaml
  campaign.lock.json
  reservations/<reservation-id>.json
  events/<event-id>.json
coordination/
  endpoints/<endpoint-identity>.json
  reconcilers/<scope>.json
  publishers/<dataset-identity>.json
```

Per-trial progress, logs, and large payloads must not become Dataset commits.
Workers publish those under unique Bucket prefixes and emit only compact
lifecycle events to the control Dataset.

Parent-commit conflicts are expected under concurrency. Adapters reread,
revalidate ownership, and retry with bounded randomized backoff. Persistent
conflicts are surfaced as control-plane errors rather than bypassing a lease.

## Artifact Layout

New campaign execution uses a versioned layout:

```text
campaigns/<campaign-id>/
  campaign.lock.json
  waves/<wave-id>/
    wave.lock.json
    endpoint.snapshot.json
    runtime-environment.json
    events.jsonl
    wave-summary.json
    _SUCCESS or _FAILED or _CANCELLED
  runs/<run-id>/
    run.lock.json
    shards/<shard-id>/
      shard.lock.json
      events.jsonl
      shard-summary.json
      _SUCCESS or _FAILED or _CANCELLED
    trials/<trial-id>/
      trial.lock.json
      executions/<execution-id>/
        manifest.yaml
        events.jsonl
        harbor.log
        harbor-jobs/
        artifacts.tar.gz
        checksums.json
        _SUCCESS or _FAILED or _CANCELLED
      trial-summary.json
      _SUCCESS or _FAILED or _CANCELLED
    run-summary.json
    _SUCCESS or _PARTIAL or _FAILED or _CANCELLED
  campaign-summary.json
  _SUCCESS or _PARTIAL or _FAILED or _CANCELLED
```

Every summary references child checksums. A parent terminal marker is written
only after all required child markers and cleanup evidence are present. Existing
single-run artifact prefixes remain readable and are never rewritten.

## Reconciliation Algorithm

Each pass has an explicit action limit and deadline:

1. Acquire a short reconciler lease for one campaign or scheduling partition.
2. Load the campaign plan, events, reservations, and relevant remote resources.
3. Rebuild projections and verify invariants.
4. Convert desired-versus-observed differences into deterministic actions.
5. Order cleanup and cancellation actions ahead of new billable work.
6. Apply global, endpoint, provider, and spend admission controls.
7. Reserve actions atomically and release the reconciler lease.
8. Execute actions with bounded timeouts.
9. Append success, failure, or ambiguous-outcome events.
10. Requeue ambiguous outcomes for inspection and adoption on the next pass.

The reconciler never assumes that a timed-out create, resume, submit, cancel,
or pause request failed. It inspects deterministic labels and current provider
state before deciding whether to retry.

### Scheduling And Concurrency

Concurrency is enforced at distinct levels:

- maximum active deployment waves globally;
- one lifecycle owner per endpoint identity;
- maximum active waves per provider and hardware pool;
- maximum active Harbor shards within a wave;
- maximum agents and requests admitted to one serving deployment;
- maximum controller retries per shard and physical executions per trial;
- maximum estimated campaign and wave spend.

Serving concurrency is taken from a measured deployment profile. The scheduler
does not infer safe concurrency from GPU names or context-window capacity. A
profile records the workload distribution and goodput criterion used to choose
its limits.

### Cancellation

Cancellation is a durable request, not a process signal:

1. stop reserving new shards;
2. request cancellation of queued and active remote work;
3. allow a policy-controlled grace period or terminate immediately;
4. pause and verify every owned endpoint;
5. publish available evidence and cancellation markers;
6. release leases only after cleanup is verified.

Repeated cancellation requests are idempotent. A campaign with valid completed
trials may finish as `partial`; completed evidence is not deleted.

## Endpoint Provisioning

Deployment profiles become declarative desired state. Their digest covers
all behavior-affecting and cost-affecting configuration, including:

- model repository and full revision;
- engine and digest-pinned image;
- command, ordered arguments, non-secret environment, and secret names;
- provider, region, hardware, accelerator count, and scaling;
- context, output, sequence, batching, and request concurrency limits;
- weight, activation, and KV-cache precision;
- parser, template, attention, MoE, graph, caching, speculative, and reasoning
  controls;
- readiness and health probes.

The provisioner may adopt an endpoint only when it has a managed identity and
its complete effective configuration matches the digest. It must never
mutate an endpoint under another lease to make it match. A deterministic managed
name plus permanent deployment record makes create operations adoptable after
ambiguous API outcomes.

Created endpoints start paused or are paused immediately after provisioning
verification. Deletion is a separate explicit retention policy. Ordinary
campaign completion pauses endpoints and preserves their reproducibility
records.

## Result Publication

One serialized publisher owns each result Dataset. It discovers complete run
markers, verifies the raw checksums, and writes normalized Parquet tables:

- `runs` for the immutable benchmark configuration and aggregate outcome;
- `trials` for logical attempts and verifier results;
- `executions` for infrastructure invocations and retry reasons;
- `metrics` for latency, token, throughput, concurrency, cost, and utilization;
- `artifacts` for checksums, media types, sizes, and canonical evidence paths.

Rows use stable entity IDs and are idempotent. A rerun is a new campaign and
new rows, not an update to historical measurements. Dataset schema versions and
migrations are explicit. The publisher records its source Bucket checksum and
control-repository commit so every table row can be traced back to evidence.

The global index contains only discoverability fields and pointers to the
benchmark-specific Dataset revision. It does not duplicate full trial data.
Composite or manually selected results are labeled and cannot appear as
ordinary complete runs.

## Security And Supply Chain

- Use separate least-privilege token secrets for orchestration, execution, and
  publication where HF permissions allow it.
- Store only secret names in manifests, locks, events, and logs.
- Require private control, input, artifact, and unpublished result stores.
- Preserve digest-pinned images, full source commits, exact package versions,
  and locked dependency installation.
- Redact staged evidence before it reaches shared storage.
- Reject symlinks, traversal, unsafe archive entries, and noncanonical paths.
- Keep public examples synthetic; do not commit public ShellBench task bodies.
- Publish only explicitly selected sanitized result fields.

## Observability And Operational Targets

Every campaign must expose status without reading worker logs. Projections and
metrics include:

- queued, active, retrying, complete, invalid, failed, and cancelled counts;
- endpoint startup, active, idle, drain, and cleanup durations;
- prompt, reasoning, output, and cached token counts when reported;
- TTFT, inter-token latency, request latency, task duration, and aggregate
  throughput when reported;
- physical retry counts and categorized infrastructure failures;
- quoted price, endpoint-active time, and estimated spend;
- last successful reconcile and publisher checkpoints.

Initial operational invariants:

- no endpoint is running without a current lease and live watchdog;
- cleanup actions always take priority over new billable work;
- a lost controller is detected by its watchdog and leaves the endpoint paused;
- committed complete trials are never rerun automatically;
- a run is published at most once per result Dataset revision history;
- all published rows trace to checksummed raw evidence;
- no success marker is emitted while cleanup or validation is incomplete.

Alerting initially uses failed scheduled Jobs, stale leases, cleanup-failure
events, and campaigns with no progress across multiple reconciliation periods.
A dedicated external monitoring service is not required for the first release.

## CLI And Optional Space

The CLI remains the canonical control surface:

```text
harbor-hf campaign plan MANIFEST
harbor-hf campaign submit MANIFEST
harbor-hf campaign status CAMPAIGN_ID
harbor-hf campaign reconcile CAMPAIGN_ID
harbor-hf campaign cancel CAMPAIGN_ID
harbor-hf campaign retry CAMPAIGN_ID --shard SHARD_ID
harbor-hf artifacts verify CAMPAIGN_ID
harbor-hf results publish CAMPAIGN_ID
```

Machine-readable JSON output is required for every command. Mutating commands
support dry-run where meaningful and print the immutable IDs they reserve.

An optional authenticated Space may create campaign requests and display
projections. It calls the same application layer and writes the same control
records. It never stores authoritative state, directly owns an endpoint, or
decides that a run is complete.

## Implementation Milestones

### Milestone 0: Freeze The Single-Run Baseline

Status: complete.

Deliverables:

- preserve current locks, lifecycle, evidence, and cleanup fixtures;
- retain `harbor-hf submit` as the supported single-cell path;
- capture compatibility tests for current artifact and coordination records;
- document the campaign feature as additive until migration is complete.

Exit evidence: the existing remote smoke, artifact audit, lifecycle tests,
mutation gate, and endpoint cleanup verification remain valid.

### Milestone 1: Campaign Schema And Deterministic Planning

Deliverables:

- add versioned campaign, run, shard, trial, execution, wave, event, and
  artifact models;
- add matrix include and exclude rules;
- resolve all selected Harbor tasks and digests without executing them;
- split task-attempt sets deterministically under configured shard bounds;
- produce `campaign.lock.json` and a stable plan digest;
- export JSON Schema and compatibility fixtures;
- add `campaign plan` with human and JSON output.

Tests:

- property tests for ordering-independent plan resolution;
- golden files for schema and lock compatibility;
- rejection tests for mutable, duplicate, missing, and conflicting inputs;
- a 10,000-shard planning test with bounded memory and no remote mutations.

Exit criteria: two clean environments resolve the same immutable inputs to the
same plan digest, run IDs, shard IDs, and trial IDs.

### Milestone 2: Durable Control Plane And Dry Reconciliation

Deliverables:

- add the control Dataset layout and typed event store;
- implement campaign and action reservations with parent-commit checking;
- implement projection rebuilding and invariant validation;
- implement a reconciler that emits an action plan without remote mutation;
- add webhook and scheduled-Job installation commands;
- add `campaign submit`, `status`, and `reconcile --dry-run`;
- record reconciler checkpoints and stale-lease diagnostics.

Tests:

- concurrent reservation and conflict tests;
- replay tests from shuffled, duplicated, and partially unknown events;
- crash tests between reservation, side effect, and outcome recording;
- no-op reconciliation tests proving repeated passes make no changes;
- contract tests against sanitized HF Dataset and Jobs responses.

Exit criteria: repeated and concurrent reconciliation converges to one action
per reservation without launching billable resources.

### Milestone 3: Endpoint Provisioning And Deployment Waves

Deliverables:

- implement exact endpoint create, adopt, inspect, pause, and optional delete;
- add deployment digests and deterministic managed endpoint identities;
- extend the current controller into a bounded wave controller;
- retain the independent watchdog and fail-closed lease behavior;
- run multiple compatible shards under one endpoint startup;
- enforce duration, shard, concurrency, idle, and spend bounds;
- publish wave-level lifecycle and cleanup evidence.

Tests:

- full lifecycle state-machine tests with failures at every provider boundary;
- ambiguous create, resume, pause, and cancellation adoption tests;
- endpoint mismatch and competing-wave rejection tests;
- controller-kill and watchdog-cleanup remote integration tests;
- separate remote smokes for pinned vLLM and llama.cpp profiles.

Exit criteria: one campaign runs at least two shards in one wave, survives a
controller termination test, and finishes every created or resumed endpoint at
`state=paused` with `readyReplica=0`.

### Milestone 4: Recovery, Cancellation, And Admission Control

Deliverables:

- reconcile queued, active, lost, retryable, terminal, and cancelled work;
- distinguish logical attempts from physical retries end to end;
- add global, deployment, provider, and campaign concurrency budgets;
- add hard spend caps and cleanup-first admission control;
- add durable cancellation, drain, retry, and manual-intervention workflows;
- add backoff and quota handling without hiding benchmark failures;
- add campaign summaries and terminal markers.

Tests:

- randomized state-machine and fault-injection tests;
- duplicate, delayed, and out-of-order event tests;
- cancellation at every wave and shard phase;
- quota exhaustion and retry-budget tests;
- remote kill-and-reconcile tests with completed-trial preservation;
- scale simulation across multiple campaigns and deployment digests.

Exit criteria: a multi-model, multi-hardware campaign survives reconciler and
wave-controller termination, resumes without republishing or rerunning valid
trials, respects its spend cap, and leaves all endpoints paused.

### Milestone 5: Inference Providers

Deliverables:

- implement a provider target adapter separate from endpoint deployments;
- preserve provider request, model, routing, quota, retry, usage, and latency
  evidence without inventing hidden runtime details;
- apply provider-specific concurrency and spend budgets;
- run provider-backed shards through the same Harbor and artifact contracts;
- make endpoint and provider runs comparable only on shared observed fields.

Tests:

- provider response and streaming contract tests;
- throttling, timeout, malformed usage, and ambiguous request tests;
- tool-use smoke tests for each supported provider path;
- assertions that endpoint-only evidence remains `not_applicable` or
  `not_reported`, never guessed.

Exit criteria: a provider-backed campaign shard produces a valid Harbor result,
complete evidence, and normalized records without creating an endpoint.

### Milestone 6: Serialized Results Publication

Deliverables:

- freeze reviewed Parquet schemas for all normalized tables;
- implement one leased publisher per destination Dataset;
- verify complete raw evidence and checksums before publishing;
- implement idempotent row generation, partitioning, and compaction;
- publish benchmark-specific revisions and the global index;
- add rebuild, audit, and schema-migration commands.

Tests:

- golden Parquet schema and migration tests;
- duplicate publication and interrupted commit recovery tests;
- raw-to-row traceability audits;
- exclusion tests for partial, invalid, or unsanitized evidence;
- rebuild equality tests from canonical Bucket evidence.

Exit criteria: deleting and rebuilding the derived Dataset produces equivalent
rows and every row points to checksummed evidence and an immutable run lock.

### Milestone 7: Presentation And Upstreaming

Deliverables:

- build a read-only leaderboard Space from normalized Datasets;
- add campaign, run, task, attempt, error, throughput, hardware, and cost views;
- support explicit complete, partial, composite, and manual-result labels;
- document the workflow in Harbor Cookbook;
- upstream only generic Harbor lifecycle or artifact extension points;
- keep package boundaries compatible with a future Harbor monorepo import.

Exit criteria: an external reader can identify the exact configuration,
evidence, result scope, and publication revision behind every displayed score.

## Quality Gates For Every Milestone

- Ruff format and lint pass.
- Ty type checking passes without adding unbounded `Any`.
- Pytest passes with at least 85% coverage and focused tests for every behavior.
- Mutation testing remains at or above 90% for behavior changes.
- `pip-audit`, Slophammer DRY, and Slophammer production checks pass.
- No local model loading, inference, or benchmark task execution occurs.
- Remote tests use explicit markers and verify all touched endpoints are paused.
- Captured fixtures and artifacts contain no credentials or public ShellBench
  task contents.
- Documentation and schema compatibility fixtures change in the same pull
  request as their behavior.
- A final review checks idempotency, ambiguous provider outcomes, cancellation,
  cleanup ordering, artifact publication, and secret handling.

## Migration And Compatibility

1. Add campaign models and commands without changing `submit` behavior.
2. Implement a one-cell campaign adapter that can reproduce a current run lock.
3. Run campaign and single-run remote smokes against separate disposable run
   IDs and compare evidence contracts.
4. Make `submit` call the campaign application layer only after parity tests
   pass; keep its CLI contract as a convenience command.
5. Continue reading legacy single-run prefixes and coordination records.
6. Never rewrite historical artifacts or result rows during migration.
7. Deprecate legacy internal paths only after one released schema version and a
   successful rebuild audit.

Rollback is code-only: stop webhook and scheduled reconciliation, cancel queued
campaign work, let watchdogs pause active endpoints, and continue using the
existing single-run path. Durable campaign plans and evidence remain readable.

## Scaling Boundary

The first production control plane intentionally uses HF Datasets and Buckets,
not a database embedded in a Space. Coordination interfaces must remain
replaceable. Before introducing an external transactional database or workflow
engine, measure:

- parent-commit conflict and retry rates;
- reconciliation latency and no-progress periods;
- control-repository history and projection rebuild cost;
- active campaign, shard, and endpoint counts;
- requirements for transactions spanning independent HF resources.

First reduce contention by partitioning reconciliation and coordination by
campaign, endpoint identity, and publisher destination. Move to a managed
database or workflow engine only when measured Hub coordination limits prevent
the operational targets above. The domain IDs, events, ports, artifact layout,
and result schemas must remain unchanged across that migration.

## Non-Goals

- Supporting benchmark harnesses other than Harbor.
- Running inference or task containers locally.
- Treating a Space as the execution service or source of truth.
- Sharing one endpoint across unrelated campaigns by default.
- Claiming exactly-once remote execution.
- Inferring unreported provider hardware, engine, precision, or cost details.
- Building a transactional database before object-backed reconciliation is
  shown to be insufficient.
