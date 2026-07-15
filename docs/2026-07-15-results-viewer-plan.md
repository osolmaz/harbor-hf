---
title: "Production results viewer plan"
author: "Onur Solmaz <2453968+osolmaz@users.noreply.github.com>"
date: "2026-07-15"
---

# Production Results Viewer Plan

## Purpose

Build a Harbor-compatible results website that is hosted as a Hugging Face
Docker Space and can browse, compare, and inspect historical `harbor-hf`
campaigns. The viewer must preserve the repository's existing evidence,
publication, reproducibility, and privacy boundaries.

This is an incremental presentation-layer project. It does not replace Harbor
as the benchmark runner, the private artifact Bucket as canonical evidence, or
the normalized result Datasets as the query and publication layer.

## Current State

`harbor-hf` already provides the foundation:

- immutable, checksummed run evidence in a private HF Bucket;
- versioned normalized Parquet tables for runs, trials, executions, metrics,
  and safe artifact metadata;
- a global result index that pins each publication to an exact Dataset commit;
- explicit complete, partial, ordinary, composite, and manual result labels;
- a credentialless Gradio Space that validates provenance and displays tables;
- deterministic campaign, run, shard, logical-trial, and physical-execution
  identities.

The current Space is an operational table browser. It directly downloads the
index and all table files for every selected publication on refresh. It does
not provide stable resource URLs, an API boundary, lazy trial detail loading,
trajectory inspection, artifact retrieval, comparison workflows, protected
access to private evidence, or a scalable catalog query path.

## Target State

```text
Harbor jobs and trajectories
            |
            v
harbor-hf verification and publication
            |
            +---- private HF Bucket
            |       canonical evidence and full sessions
            |
            +---- normalized result Datasets
            |       immutable query and presentation projections
            |
            +---- compact catalog snapshots
                    bounded list and comparison queries
                              |
                              v
                  versioned Results API
                              |
                              v
               React application in a Docker Space
```

The Space is a replaceable read-only projection. It never schedules work,
controls endpoints, publishes results, or stores authoritative state.

## Design Principles

1. Keep canonical evidence private, immutable, and content verified.
2. Treat every public table, trajectory, and artifact as a rebuildable
   projection of canonical evidence.
3. Put a versioned API between the frontend and HF storage layouts.
4. Resolve mutable Hub references to immutable commits before reading data.
5. Load list pages from bounded catalog snapshots and load details lazily.
6. Use stable IDs and cursor pagination; never use display names as identity.
7. Fail closed when schemas, checksums, revisions, or provenance disagree.
8. Run the same application in public and protected modes with explicit
   capabilities rather than maintaining separate implementations.
9. Use only public Harbor and Hugging Face interfaces.
10. Reuse Apache-2.0 Harbor viewer code where it is genuinely shared, with
    attribution and a pinned upstream revision. Do not copy or depend on
    Harbor Hub's private implementation.

## Component Boundaries

### Publication Layer

The existing result publisher remains responsible for producing immutable,
versioned normalized tables. Extend it with two derived projections:

- **catalog snapshot**: one compact row per run with aggregate score, status,
  benchmark, model, agent, hardware, runtime, cost, token, timing, and
  provenance fields needed by list and comparison pages;
- **presentation artifact manifest**: one row per viewable artifact with a
  stable artifact ID, visibility, redaction status, media type, size, checksum,
  and immutable source reference.

Catalog snapshots use power-of-two compaction windows, as the current global
index does. A list request reads one bounded snapshot instead of five files per
publication. Per-publication tables remain the detail source and retain their
current trace validation.

Raw task bodies, unsanitized logs, credentials, and full sessions must never be
copied into a public Dataset. Public trajectory publication requires a
deterministic sanitizer version, a redaction report, and a checksum linking the
projection to its private source.

### Results Domain

Move presentation models and validation out of the Gradio-only package into a
shared `harbor_hf.presentation` package. It owns:

- versioned API and Dataset models;
- catalog and publication repositories;
- provenance and relation validation;
- visibility and capability policy;
- service methods used by every presentation frontend.

HF SDK calls, filesystem caches, clocks, and HTTP behavior remain adapters
around this domain. The domain must be testable without network access.

### Results API

Add a read-only FastAPI application with an explicit `/api/v1` contract. The
initial resource model is:

```text
GET /api/v1/health
GET /api/v1/capabilities
GET /api/v1/campaigns
GET /api/v1/campaigns/{campaign_id}
GET /api/v1/runs
GET /api/v1/runs/{run_id}
GET /api/v1/runs/{run_id}/trials
GET /api/v1/runs/{run_id}/metrics
GET /api/v1/compare?run_id=...&run_id=...
GET /api/v1/runs/{run_id}/trials/{trial_id}
GET /api/v1/runs/{run_id}/trials/{trial_id}/executions
GET /api/v1/runs/{run_id}/executions/{execution_id}
GET /api/v1/runs/{run_id}/executions/{execution_id}/trajectory
GET /api/v1/runs/{run_id}/artifacts/{artifact_id}
GET /api/v1/runs/{run_id}/artifacts/{artifact_id}/content
```

Collection endpoints use opaque cursor pagination, deterministic ordering,
bounded page sizes, structured filters, and a documented maximum comparison
set. Detail responses include source Dataset revisions and evidence checksums.

Immutable responses receive ETags and long-lived cache headers. Catalog
responses identify the resolved index commit and use revalidation. Errors use
one documented JSON envelope and never expose storage credentials, local paths,
or raw provider responses.

The artifact content route resolves only manifest-backed artifact IDs. It must
not accept arbitrary paths or Bucket keys. It enforces size and media-type
limits, sends active content as attachments, verifies the content checksum,
and records no secret-bearing URL in logs.

### Web Application

Replace the Gradio table UI with a responsive React application served by the
same Docker container as the API. Required routes are:

```text
/
/campaigns/{campaign_id}
/runs/{run_id}
/runs/{run_id}/compare/{other_run_id}
/trials/{trial_id}
/executions/{execution_id}
```

The first production slice includes:

- searchable and filterable run catalog;
- campaign summary and run matrix;
- run configuration, aggregate metrics, provenance, and task outcome table;
- trial and execution history with retries clearly distinguished;
- structured trajectory timeline with messages, reasoning visibility markers,
  tool calls, tool results, token usage, and timestamps;
- verifier output and safe artifact links;
- side-by-side run comparison with cohort compatibility warnings;
- permanent links that survive index compaction and Space redeployment.

The frontend obtains all data through `/api/v1`; it never reads Parquet or
Bucket paths directly. Missing private capabilities are hidden based on the
capabilities response rather than failing after interaction.

Harbor's open-source viewer is the design and component baseline. Until its
frontend is available as a versioned package, keep any adapted code small,
clearly attributed, and pinned to a reviewed Harbor commit. Do not use a git
submodule or fetch an unpinned branch during the Space build. Prefer an
upstream package or shared viewer protocol when Harbor provides one.

### Access Modes

The same image supports two explicit modes:

| Mode | Space visibility | Credentials | Data exposed |
| --- | --- | --- | --- |
| Public | public | none | public catalog, metrics, provenance, and sanitized artifacts |
| Protected | private or protected | scoped read-only token | public data plus authorized private evidence |

Public mode must continue to construct every Hub client with anonymous access.
Protected mode requires a token scoped only to the result Datasets and artifact
Bucket. The application fails startup if its configured mode and available
credentials disagree.

The HF Space access boundary protects the whole protected application. The
application must not pretend to provide finer per-user authorization until a
real identity and policy layer exists.

### Cache And Scale

The Docker Space uses an ephemeral, bounded, checksum-keyed local cache. Cache
keys include repository, immutable revision, path, and expected checksum.
Mutable branch names never key cached content.

Scale requirements:

- list pages read a compact catalog snapshot, not every publication;
- trial, execution, trajectory, and artifact data load only when requested;
- all collection queries are paginated and bounded;
- repeated reads of immutable content avoid another Hub download;
- cache size and item limits are configurable and enforce least-recently-used
  eviction;
- concurrent cache fills for one object are coalesced;
- a malformed publication is isolated from unrelated valid runs where the
  catalog contract allows it, while the affected detail request fails closed;
- Dataset compaction creates new immutable snapshots and never rewrites a
  published revision.

Do not introduce a database in the first production version. The compact
Parquet catalog and immutable per-run detail files are sufficient until measured
query latency, Dataset size, or update frequency proves otherwise. The API
boundary permits a future indexed store without changing the frontend.

## Repository Layout

Move toward this layout incrementally:

```text
src/harbor_hf/presentation/
  models.py
  policy.py
  repositories.py
  service.py

apps/results-api/
  app.py
  routes/
  adapters/

apps/results-web/
  src/
  package.json

deploy/space/
  Dockerfile
  README.md
  start.sh
```

The existing `space/` remains operational until the Docker Space meets the
replacement gates. Shared Python behavior must move into `src/harbor_hf`; it
must not be duplicated between the legacy and new applications.

## Delivery Plan

### Phase 0: Freeze Contracts And Fixtures

- Record the current Space behavior and normalized schemas as compatibility
  fixtures.
- Add representative complete, failed, partial, retried, composite, and manual
  publications with synthetic, non-sensitive trajectories.
- Define API v1 resources, pagination, error envelopes, capabilities, and
  visibility semantics in an OpenAPI contract.
- Define catalog snapshot and presentation artifact manifest schemas.
- Document the Harbor commit and license obligations for reused viewer code.

Exit criteria: API and projection contracts can be reviewed without a running
Space, and every example contains no benchmark-private content.

### Phase 1: Shared Presentation Domain

- Extract the current Space models, Dataset readers, trace checks, and filters
  into `harbor_hf.presentation`.
- Preserve anonymous reads and exact-revision validation.
- Keep the Gradio Space working through the new service boundary.
- Add catalog snapshot generation and deterministic compaction.

Exit criteria: the existing Space passes unchanged user-level behavior tests,
and catalog generation is reproducible byte-for-byte from fixed inputs.

### Phase 2: Read-Only API

- Implement `/api/v1` over in-memory and HF adapters.
- Add cursor pagination, filtering, comparison compatibility checks, ETags,
  caching, and structured errors.
- Add health, readiness, source revision, and capability reporting.
- Generate and check in the OpenAPI snapshot used by the frontend client.

Exit criteria: contract tests cover every route and failure class, and a large
synthetic catalog does not trigger per-publication fan-out for list requests.

### Phase 3: Artifact And Trajectory Publication

- Define the sanitizer interface and versioned redaction report.
- Publish synthetic trajectory fixtures through the complete private-source to
  public-projection path.
- Implement manifest-only artifact lookup and checksum-verified streaming.
- Add protected-mode Bucket reads and startup policy validation.

Exit criteria: automated secret canaries never appear in public projections,
path traversal and arbitrary-key requests are rejected, and modified artifact
bytes fail checksum validation.

### Phase 4: React Viewer

- Implement catalog, campaign, run, trial, execution, trajectory, provenance,
  artifact, and comparison views.
- Generate the API client from the pinned OpenAPI contract.
- Adapt only the reusable portions of Harbor's Apache-2.0 viewer and retain
  attribution.
- Add accessible loading, empty, partial, error, and restricted states.

Exit criteria: all required URLs are directly loadable and refreshable, no
page reads storage directly, and Playwright verifies desktop and mobile flows.

### Phase 5: Docker Space And Release Automation

- Build one multi-stage Docker image containing the static frontend and API.
- Add public and protected configuration profiles using the same image.
- Publish a versioned deployment tree from a tagged `harbor-hf` release to a
  separate HF Space repository.
- Record application version, source commit, schema versions, and resolved
  Dataset revision in health output and the UI.
- Add rollback to the previous known-good release.

Exit criteria: CI deploys a staging Space, smoke-tests real routes and one
  synthetic trajectory, verifies logs, and promotes the exact tested release
  without rebuilding it.

### Phase 6: Migration And Removal

- Run Gradio and Docker Spaces against the same published fixtures.
- Compare totals, labels, metrics, provenance, and filtering results.
- Make the Docker Space the documented default after parity and reliability
  gates pass.
- Retain the Gradio implementation for one release as a rollback path, then
  remove it without removing the shared presentation domain.

Exit criteria: the Docker Space is the sole documented viewer, historical
permalinks remain valid, and rollback has been exercised.

## Verification Strategy

### Local And CI

- schema and golden-file tests for every projection version;
- domain tests with in-memory repositories and no network access;
- API contract tests generated from the OpenAPI snapshot;
- property tests for cursor stability, ordering, identity, and filter bounds;
- corruption tests for revisions, relations, checksums, and Parquet rows;
- security tests for secret canaries, path traversal, active content, oversized
  files, and error redaction;
- frontend unit tests and Playwright end-to-end tests at desktop and mobile
  viewports;
- load tests over synthetic catalogs at 1K, 10K, and 100K runs;
- license and attribution checks for adapted Harbor files.

Documentation-only planning changes do not require the runtime test suite.
Implementation phases must use the repository's normal Ruff, ty, pytest,
coverage, Slophammer, and relevant frontend checks. Mutation testing remains a
non-blocking diagnostic and is not part of the deployment critical path.

### Hosted Staging

Each release candidate is tested on an HF staging Space without running model
inference. Verification includes:

- Space reaches the expected running stage;
- startup and request logs contain no warnings, credentials, or local paths;
- health and capability responses match deployment mode;
- list, detail, comparison, trajectory, and artifact requests return expected
  content;
- public mode cannot read private fixtures;
- protected mode can read only its scoped fixtures;
- exact source and application revisions appear in responses;
- the promoted release is byte-identical to the tested release.

## Observability And Operations

- Emit structured logs with request ID, route, status, duration, cache result,
  resolved Dataset commit, and application version.
- Never log tokens, signed URLs, raw trajectories, artifact content, or Bucket
  keys outside stable public identifiers.
- Expose health and readiness separately; readiness fails when required catalog
  configuration cannot be resolved and validated.
- Report cache entries, bytes, hits, misses, evictions, and Hub download
  failures without exposing private names in public mode.
- Put timeouts, retries, response limits, cache bounds, and maximum page sizes
  in validated configuration with conservative defaults.
- Pin all Python, JavaScript, Harbor-source, base-image, and build-tool versions
  in a release lock.

## Compatibility And Upstream Path

The API and presentation domain remain `harbor-hf` features until their generic
parts are stable. Potential upstream contributions to Harbor are deliberately
small and independent:

- a stable Harbor result/trajectory viewer protocol;
- publishable Harbor viewer components;
- generic job bundle and artifact interfaces;
- links from a Harbor job to an external immutable evidence store.

`harbor-hf` continues to own HF Jobs, Inference Endpoints, Datasets, Buckets,
Space deployment, campaign orchestration, hardware metadata, and provider
evidence. No upstream change is required to ship the production viewer.

## Production Acceptance Gates

The viewer is production-ready only when all of the following hold:

- historical complete and failed runs have stable, directly loadable URLs;
- campaign, run, trial, and execution identities cannot be confused across
  retries or shards;
- public deployment can operate with no credential;
- no private task body, session, log, or secret canary reaches public storage;
- protected deployment uses only a scoped read credential and exposes no
  finer-grained authorization claim than HF actually enforces;
- list latency is bounded by compact catalog reads rather than total historical
  publication count;
- details are revision pinned, relation validated, and checksum verified;
- the frontend has no knowledge of Dataset or Bucket path layouts;
- staging deployment, smoke testing, promotion, and rollback are automated;
- the Gradio viewer remains available until parity, security, performance, and
  rollback gates pass.

## Explicit Non-Goals

- Reimplementing Harbor Hub's private backend or copying its closed-source UI.
- Making the viewer a campaign controller or benchmark scheduler.
- Using the Space filesystem as durable or authoritative storage.
- Publishing raw ShellBench tasks, unsanitized sessions, or private evidence.
- Adding a database before measured scale requires one.
- Blocking result publication on Space availability.
- Requiring Harbor to merge upstream changes before this viewer can ship.
