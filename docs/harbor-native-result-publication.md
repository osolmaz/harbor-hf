# Harbor-Native Result Publication Plan

## Status

Implemented through the non-breaking dual-publication and v2-reader phases.
The current `harbor-hf/results/*/v1` Parquet publications remain supported and
immutable. New evidence can carry native v2 provenance while historical rows
remain explicitly marked `legacy-v1`. Stopping new v1 query-table writes and a
future Harbor-owned export command remain deferred compatibility work.

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
7. Current v1 publications remain readable indefinitely. New versions use new
   paths and explicit schema identifiers.

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
  publication-envelope.v2.json
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

`publication-envelope.v2.json` uses `harbor-hf/publication-envelope/v2`. Its smallest
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

Each physical execution records `bundle_status`. New successful executions use
`verified` and must reference a bundle. Immutable successes produced before the
compatibility adapter existed use `legacy_unavailable`; failed or cancelled
executions without a valid bundle use `not_available`. The legacy status keeps
in-flight campaigns finalizable during rollout without claiming native
provenance that does not exist. Runs containing a legacy successful execution
remain `legacy-v1` in public catalogs until they are rerun with verified native
bundles.

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

## Migration

### Phase 0: Freeze And Inventory

- Freeze `harbor-hf/results/*/v1` and its readers.
- Map every v1 field to Harbor, `harbor-hf`, or a derived calculation.
- Identify fields whose meaning differs from current Harbor models.
- Add golden native Harbor bundles for success, verifier failure,
  infrastructure failure, retry, and multi-step execution.

Exit criterion: every current field has one documented owner and derivation.

Status: complete. See `docs/result-field-ownership.md`.

### Phase 1: Specify V2

- Define the minimal HF execution envelope and JSON Schema.
- Define canonical archive, checksum, marker, and content-addressing rules.
- Define public and protected allowlists.
- Run a focused schema review before implementation.

Exit criterion: the envelope contains no Harbor-owned result fields and can
address immutable native Harbor bundles without consulting mutable state.

Status: complete. JSON Schemas are checked in under `schemas/`.

### Phase 2: Build The Native Adapter

- Change the compatibility exporter to retain Harbor-native serialized models
  and generate the temporary bundle manifest inside the pinned Harbor runtime.
- Stop application code from parsing Harbor's private directory layout.
- Verify archive extraction limits, checksums, and native-model round trips.
- Keep the legacy reader for historical evidence only.

Exit criterion: new canonical evidence can be validated using the pinned
Harbor models plus the HF envelope, without `harbor-hf` trial-result models.

Status: complete for the compatibility adapter. Replacement by a Harbor-owned
export command remains upstream work and is not required for current runs.

### Phase 3: Dual Publish

- Publish the canonical v2 bundle and envelope alongside existing v1 output.
- Generate v1 and v2-derived projections from the same successful evidence.
- Compare trial identity, rewards, aggregate score, timing, and artifact
  metadata for every fixture and a bounded remote campaign.
- Treat any mismatch as a publication failure; do not choose one silently.

Exit criterion: dual publication is idempotent and produces equivalent visible
results for all supported run shapes.

Status: complete locally; bounded hosted verification is required for release.

### Phase 4: Move Readers

- Teach the Results API to read the v2 catalog and revision-pinned details.
- Keep permanent v1 URLs and readers working.
- Rebuild the six historical demonstration runs as v2 projections without
  replacing their existing v1 publications.
- Verify public and protected Space behavior end to end.

Exit criterion: the Space uses v2 by default and mixed v1/v2 history remains
searchable and comparable.

Status: reader and publisher complete; hosted Dataset migration and Space
verification are required for release.

### Phase 5: Stop New V1 Writes

- Disable v1 publication for new runs after a documented compatibility window.
- Retain v1 schemas, readers, audit commands, and golden fixtures.
- Remove duplicated Harbor-owned write models only when no active path imports
  them.
- Keep rollback capable of re-enabling v1 writes without changing canonical
  evidence.

Exit criterion: new results have one canonical Harbor bundle, one HF envelope,
and rebuildable projections; no production writer maintains a second Harbor
result model.

### Phase 6: Adopt Harbor Export

- Replace the compatibility exporter when a released Harbor version provides
  the storage-neutral bundle contract.
- Pin the first supported Harbor release and verify byte and semantic parity.
- Remove only the temporary exporter path; retain migration readers for its
  historical bundles.

Exit criterion: Harbor owns native export end to end and `harbor-hf` contains
only HF orchestration, evidence, and projection logic.

## Verification

Required local tests:

- native Harbor model validation and archive round trips;
- JSON Schema and golden-file compatibility for the HF envelope;
- deterministic publication IDs, manifests, checksums, and projections;
- v1/v2 semantic parity for all supported fixtures;
- idempotent retry after interruption between Bucket and Dataset commits;
- corruption, traversal, archive-bomb, symlink, and secret-leak rejection;
- mixed-version Results API and Space behavior;
- rebuild equality from canonical evidence.

Required remote test:

- run one bounded campaign entirely on Hugging Face infrastructure;
- preserve the native Harbor bundle and v2 envelope in the private Bucket;
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

Rollback changes the preferred reader or writer version only. It never deletes
canonical bundles, rewrites historical Dataset commits, or mutates successful
run prefixes.

## Completion Criteria

The migration is complete when:

- Harbor is the only authority for benchmark job and trial semantics;
- `harbor-hf` owns only HF execution, provenance, storage, and projections;
- every displayed score traces to one immutable Harbor bundle and HF envelope;
- public datasets can be deleted and rebuilt from canonical evidence;
- current v1 links and historical results remain readable;
- Harbor Hub and HF publication can consume the same native bundle
  independently;
- no endpoint is left running after publication or projection work.

## Non-Goals

- Replacing Harbor Hub or requiring its hosted Supabase backend.
- Moving HF endpoint or campaign logic into Harbor.
- Publishing raw sessions or task-sensitive evidence publicly.
- Rewriting current v1 Dataset history.
- Making the Results Space part of benchmark correctness.
