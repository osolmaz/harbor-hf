# Harbor-Native Result Publication Plan

## Status

The canonical `v1` publication boundary is implemented. The dual writer,
v2-first reader, migration command, legacy classifications, and superseded
schemas have been removed. The active hosted catalog must be rebuilt only from
verified native evidence and deployed with the matching Results Space release.

## Purpose

Make Harbor the sole authority for benchmark job and trial results while
keeping `harbor-hf` responsible for Hugging Face execution, durable evidence,
publication, and query infrastructure.

The current publisher converts completed evidence into independent `runs`,
`trials`, `executions`, `metrics`, and `artifacts` models. Those tables are
useful for querying, but some fields repeat Harbor-owned concepts such as trial
identity, task identity, rewards, exceptions, timing, usage, and artifacts.
The long-term design must preserve those values without maintaining a second
authoritative benchmark-result contract.

## Decisions

1. A Harbor-native job bundle is the canonical benchmark result.
2. `harbor-hf` owns a small, versioned HF execution envelope around that
   bundle. It does not redefine Harbor jobs, trials, rewards, or trajectories.
3. The private HF Bucket is the canonical evidence store.
4. Public and protected Datasets contain sanitized, rebuildable projections.
5. The Results Space reads projections through the versioned Results API. It
   never becomes an authority or a publication dependency.
6. Harbor Hub upload is an optional sink, not a required backend or source of
   truth.
7. This repository is owned by `osolmaz`, so publication changes cut over in
   place. The canonical contract remains `v1`; no parallel `v2`, fallback
   reader, dual writer, or permanent legacy path survives the cutover.

## Cutover Policy

The current dual-publication implementation is temporary and deprecated. It is
not a compatibility commitment. Deprecation ends with one coordinated release,
not an open-ended window.

The cutover must:

- pause publication writes for a bounded maintenance window;
- snapshot the current Dataset and Space revisions for recovery and audit;
- rebuild active results from verified canonical evidence;
- deploy the canonical writer, reader, API, and Space together;
- delete the superseded schemas, code paths, commands, fixtures, and tests in
  the same change;
- keep the public schema identifier and paths at `v1`.

Historical publications that cannot be traced to a verified native Harbor
bundle are not silently converted. They move to an archival snapshot that the
production API does not read, and return to the active catalog only after a
verified rebuild or rerun. The archive is evidence retention, not a runtime
compatibility layer.

## Ownership Boundary

Harbor owns:

- job and trial configuration;
- job, trial, task, and logical-attempt identity within a Harbor execution;
- agent and model information reported by the harness;
- verifier rewards, exceptions, timing, and token or cost usage;
- sessions, ATIF trajectories, logs, and collected artifact manifests;
- Harbor job and trial locks and the native job-directory layout.

`harbor-hf` owns:

- experiment, campaign, run, shard, wave, and physical-execution identity;
- model-weight revisions and resolved serving profiles;
- inference engine, quantization, context, batching, and concurrency settings;
- endpoint or provider, hardware, accelerator count, region, and HF Job IDs;
- infrastructure retries, lifecycle events, spending policy, and verified
  endpoint cleanup;
- Bucket, Dataset, and Space publication revisions;
- evidence checksums, visibility policy, redaction policy, and publication
  state.

Fields may be copied into a query projection for performance, but their owner
does not change. Projection code must derive Harbor-owned values from a pinned
Harbor result contract rather than reinterpret raw files independently.

## Target Architecture

```text
Harbor execution
    |
    +-- native job bundle ----------------------+
    |                                           |
    +-- harbor-hf execution envelope -----------+--> private HF Bucket
                                                |      canonical evidence
                                                |
                                                +--> verified projector
                                                       |
                                                       +-- public Dataset
                                                       +-- protected Dataset
                                                               |
                                                        Results API and Space
```

The projector is deterministic. Given the same Harbor bundle, HF envelope,
schema version, and sanitizer version, it must produce byte-equivalent logical
rows and stable publication identities.

## Canonical Publication

Each terminal run receives one immutable publication prefix. A run may contain
multiple physical Harbor execution bundles because infrastructure retries are
retained rather than overwritten:

```text
runs/<run-id>/
  trials/<trial-id>/executions/<execution-id>/
    artifacts.tar.gz
    harbor-native-bundle.json
  publication-envelope.v1.json
  checksums.json
  _SUCCESS | _FAILED
```

Each `artifacts.tar.gz` contains the allowlisted Harbor job tree, including
native job and trial results and retained private artifacts. The archive format must
preserve names, modes where relevant, and deterministic ordering. It must not
follow links or include unlisted scratch files.

Today, `harbor-native-bundle.json` is a `harbor-hf` packaging manifest with
`contract_status: compatibility`. It records the Harbor schema and version,
native result paths, and checksums without copying result semantics. The pinned
compatibility exporter derives it from Harbor's public Pydantic models inside
the pinned Harbor environment. A future Harbor-owned export manifest can
replace this temporary packaging contract without changing the HF envelope.

`publication-envelope.v1.json` uses `harbor-hf/publication-envelope/v1`. Its smallest
stable shape contains:

- the run, campaign, and physical-execution IDs;
- the Harbor source revision, package version, bundle schema, paths, and
  checksums for every physical execution bundle;
- the resolved model and serving profile digests;
- provider, region, hardware, accelerator count, and remote execution IDs;
- canonical Bucket prefix and evidence checksum;
- sanitizer and projection versions;
- completion and endpoint-cleanup outcomes.

The envelope references the Harbor bundle. It does not copy task names,
rewards, trial timing, exceptions, trajectories, or artifact entries.

Each physical execution records `bundle_status`. Successful executions use
`verified` and must reference a bundle. Failed or cancelled executions without
a valid bundle use `not_available`. A successful execution without a verified
bundle is invalid for canonical publication and must be rebuilt or rerun; there
is no `legacy_unavailable` status or legacy catalog classification.

`checksums.json` covers every non-marker object. `_SUCCESS` is written last and
only after Harbor output, HF infrastructure evidence, checksums, redaction,
and endpoint cleanup all validate. Publication consumes only an exclusively
successful prefix.

## Dataset Projections

Parquet remains the query format, not the canonical result format. Projection
schemas are versioned implementation contracts for readers and may denormalize
data deliberately.

The minimum public projections are:

- `catalog`: one compact row per published run for lists and comparisons;
- `trials`: sanitized task identity, attempt, outcome, and selected rewards;
- `artifacts`: public-safe metadata and immutable references only.

Additional execution, metric, or aggregate tables are added only when a
measured query requires them. They must not become mandatory merely to mirror
every Harbor model.

Every projected row records:

- publication ID and projection schema version;
- Harbor bundle checksum and Harbor schema version;
- HF envelope checksum and projection code revision;
- exact result Dataset revision after publication.

Catalog windows remain bounded and power-of-two sized. Run details load lazily
from the exact revision named by the catalog row. Historical projection files
are immutable; compaction creates derived windows without changing individual
publication records.

## Privacy And Access

The private Bucket retains complete native results, sessions, trajectories,
logs, task-sensitive files, and canonical checksums. Public projections contain
only fields approved by an explicit allowlist.

Public projection must reject, rather than redact opportunistically:

- credentials, environment values, and authorization headers;
- task bodies, hidden tests, solutions, and private source contents;
- raw sessions, unsanitized trajectories, and unrestricted logs;
- exception tracebacks or artifact bytes not approved for publication.

A protected Results deployment may retrieve private evidence with a
least-privilege token. The public Space remains credentialless. Public and
protected modes share the same API and capability model so privacy logic does
not fork across separate applications.

## Harbor Upstream Contract

Propose a storage-neutral Harbor export interface, for example:

```text
harbor job export JOB_DIR --format bundle-v1 --output OUTPUT_DIR
```

The upstream contract should:

- validate with Harbor's own job, trial, lock, trajectory, and artifact models;
- emit a versioned manifest and deterministic allowlisted archive;
- support completed and failed jobs without inventing HF lifecycle concepts;
- expose structured artifact metadata and checksums;
- avoid Supabase, Hugging Face, campaign, endpoint, and storage assumptions;
- round-trip through Harbor download and local viewer workflows.

This is the only new benchmark-result contract to pursue. `harbor-hf` must not
promote its compatibility bundle into a competing permanent standard.

## Cutover Plan

### Phase 0: Inventory And Freeze

- Map every published field to Harbor, `harbor-hf`, or a derived calculation.
- Retain golden native Harbor bundles for success, verifier failure,
  infrastructure failure, retry, and multi-step execution.
- Mark the dual writer, v2 schemas, v2-first reader, and legacy classifications
  deprecated. Do not add consumers or features to them.
- Freeze publication changes while the cutover branch is prepared.

Exit criterion: every field has one documented owner and the complete removal
surface is enumerated.

Status: complete. Field ownership, native fixtures, and the removal inventory
are recorded and covered by the canonical contract tests.

### Phase 1: Define The Canonical V1 Contract

- Rename the native envelope and projection contract to `v1` in place.
- Define canonical archive, checksum, marker, and content-addressing rules.
- Define public and protected allowlists.
- Ensure the envelope contains no Harbor-owned result fields and can address
  immutable native Harbor bundles without consulting mutable state.
- Update schemas, generated models, API contracts, fixtures, and docs together.

Exit criterion: exactly one writer and one reader implement the canonical `v1`
contract, with no runtime reference to a publication `v2`.

### Phase 2: Build The One-Time Rebuilder

- Build a one-time command that validates private evidence and regenerates the
  active Dataset projections using the canonical `v1` contract.
- Rebuild only runs with verified native Harbor bundles and checksums.
- Produce a report listing runs that cannot be rebuilt and therefore require a
  rerun.
- Keep this command outside the runtime reader and writer paths; remove it after
  the cutover has been verified.

Exit criterion: a dry run produces a deterministic replacement catalog and an
explicit exclusion report without mutating hosted state.

### Phase 3: Snapshot And Rebuild

- Pause publication writes for a bounded maintenance window.
- Record and verify immutable snapshots of the current result Dataset, catalog
  Dataset, and Results Space revisions.
- Preserve the pre-cutover publications in an archival Dataset or immutable
  Dataset revision that production readers do not query.
- Rebuild active results from canonical private evidence. Exclude unverifiable
  rows instead of labeling them as legacy.
- Validate counts, identities, rewards, aggregate scores, timing, artifacts,
  checksums, and privacy allowlists before deployment.

Exit criterion: the replacement active catalog contains only verified native
publications and every row traces to one Harbor bundle and HF envelope.

### Phase 4: Coordinated Cutover

- Deploy the canonical `v1` writer, Results API reader, and Space in one release.
- Atomically switch the active Dataset pointer or configured revision to the
  rebuilt catalog.
- Run hosted API checks, desktop and mobile browser checks, comparison views,
  and one bounded fully remote campaign.
- Verify all Inference Endpoints are paused with zero ready replicas.

Exit criterion: new runs publish once, the production API has one read path,
the Space displays only canonical results, and remote publication succeeds end
to end.

### Phase 5: Delete Superseded Paths

- Remove the dual writer, fallback reader, v2 schemas, v2 models, migration
  aliases, legacy statuses, and mixed-version tests.
- Remove commands such as `migrate-catalog-v2` and any `catalog_v2_*` helper.
- Remove duplicated Harbor-owned result models after the canonical projector no
  longer imports them.
- Remove the one-time rebuilder after its output and recovery snapshot are
  verified.
- Search code, tests, docs, CI, Dataset configuration, and Space configuration
  for remaining superseded references.

Exit criterion: repository-wide checks find no runtime compatibility path,
parallel publication version, or legacy classification.

### Phase 6: Adopt A Harbor-Owned Export

Harbor is an important upstream repository that `harbor-hf` does not own, so its
published compatibility and versioning requirements remain authoritative.

- Replace the compatibility exporter when a released Harbor version provides
  the storage-neutral bundle contract.
- Pin the first supported Harbor release and verify byte and semantic parity.
- Cut over the internal exporter in place; do not retain the temporary exporter
  as a fallback in `harbor-hf`.

Exit criterion: Harbor owns native export end to end and `harbor-hf` contains
only HF orchestration, evidence, and projection logic.

## Verification

Required local tests:

- native Harbor model validation and archive round trips;
- JSON Schema and golden-file validation for the canonical v1 HF envelope;
- deterministic publication IDs, manifests, checksums, and projections;
- rejection of superseded v2, legacy, and mixed-version inputs;
- idempotent retry after interruption between Bucket and Dataset commits;
- corruption, traversal, archive-bomb, symlink, and secret-leak rejection;
- single-version Results API and Space behavior;
- rebuild equality from canonical evidence.

Required remote test:

- run one bounded campaign entirely on Hugging Face infrastructure;
- preserve the native Harbor bundle and v1 envelope in the private Bucket;
- publish and browse sanitized projections;
- download and validate the bundle with the pinned Harbor version;
- verify all Inference Endpoints are paused with zero ready replicas before the
  campaign is declared complete.

No local model loading or inference is part of this migration.

## Operations And Rollback

Publication state must expose the canonical Bucket prefix, Harbor bundle
checksum, envelope checksum, projection revision, and last completed phase.
Retries adopt matching immutable objects and reject conflicting bytes.

The Results Space is never on the write path. Dataset or Space failure cannot
invalidate canonical evidence. A failed projection can be rebuilt after the
run without resuming an endpoint or rerunning a benchmark.

Rollback restores the complete pre-cutover release and points readers to the
verified pre-cutover Dataset snapshot. It does not keep fallback code in the
main branch, delete canonical bundles, rewrite historical Dataset commits, or
mutate successful run prefixes. A failed cutover is fixed and attempted again
as a coordinated release; it does not reintroduce dual reads or writes.

## Completion Criteria

The migration is complete when:

- Harbor is the only authority for benchmark job and trial semantics;
- `harbor-hf` owns only HF execution, provenance, storage, and projections;
- every displayed score traces to one immutable Harbor bundle and HF envelope;
- public datasets can be deleted and rebuilt from canonical evidence;
- the active catalog contains only results with verified native provenance;
- superseded v2, dual-publication, and legacy runtime paths are deleted;
- the sole supported publication contract and public path remain named `v1`;
- Harbor Hub and HF publication can consume the same native bundle
  independently;
- no endpoint is left running after publication or projection work.

## Non-Goals

- Replacing Harbor Hub or requiring its hosted Supabase backend.
- Moving HF endpoint or campaign logic into Harbor.
- Publishing raw sessions or task-sensitive evidence publicly.
- Serving unverifiable historical rows from the active production catalog.
- Preserving the temporary v2 or dual-publication contract for compatibility.
- Making the Results Space part of benchmark correctness.
