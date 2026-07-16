# Normalized Result Publication

The private artifact Bucket is canonical evidence and result Datasets are
derived indexes. Campaign reconciliation invokes publication automatically for
a completed campaign; `harbor-hf results publish` exposes the same verified,
idempotent path for explicit operation and recovery.

## Safety boundary

A run is publishable only when its evidence prefix has `_SUCCESS`, has no other
top-level terminal marker, and every non-marker object is present in and matches
`checksums.json`. The checksummed `run-summary.json` must declare that it was
sanitized and must describe an ordinary, complete run. Complete means every
logical trial reached a scored terminal outcome: a valid verifier result, or a
zero-score failed trial after its bounded physical retry budget was exhausted.
The checksummed `run.lock.json` must contain the same immutable `run_id`.

The publisher reads normalized values only from `run-summary.json`. It never
copies evidence object bytes into a Dataset. Raw Harbor job trees, sessions,
trajectories, task source, task instructions, `manifest.yaml`, `harbor.log`, and
`artifacts.tar.gz` are not publishable artifacts. Task rows contain only a task
name and immutable digest; they never contain ShellBench task contents.

## Frozen table schemas

All five benchmark tables use schema version `harbor-hf/results/<table>/v1`.
Fields are flat Parquet columns. Strings are UTF-8, timestamps are UTC
microseconds, counters and sizes are signed 64-bit integers, metric values are
64-bit floating point, and nullable fields are marked below. Every benchmark
row repeats the publication and evidence trace fields. This deliberate
denormalization makes each row independently auditable.

Common trace fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Exact table schema version |
| `publication_id` | string | Deterministic identity of the source run evidence |
| `run_id` | string | Immutable run identity |
| `source_bucket` | string | Canonical private evidence Bucket |
| `source_prefix` | string | Canonical run evidence prefix |
| `source_checksum` | string | Digest of the verified checksum manifest |
| `run_lock_path` | string | Canonical path of the immutable run lock |
| `run_lock_sha256` | string | Verified run-lock object checksum |
| `control_commit` | string | Immutable control Dataset commit used to publish |

`runs` adds campaign, experiment, benchmark, result classification, completion
times, model, deployment, agent, fixed planned-trial denominator, task-outcome
counts, quality, and aggregate child-count fields. `trials` adds the logical
trial identity, task name and digest, attempt, selected execution, and one of
four outcomes: `scored`, `agent_failed`, `benchmark_failed`, or
`infrastructure_exhausted`. Every non-scored trial has a zero reward metric and
selects its final failed execution. `executions` adds the physical execution
identity, trial identity, attempt, runtime kind, `succeeded`, `failed`, or
`cancelled` status, typed failure category, timestamps, retry reason, and
optional remote Job identity. `metrics` adds a deterministic metric
identity, typed owner, name, value, unit, and optional aggregation. `artifacts`
adds a deterministic artifact identity, typed owner, safe artifact kind, private
canonical path, checksum, media type, and size. Artifact rows are metadata and
pointers only.

For endpoint-backed runs, `model_revision` is the Hub revision verified in the
endpoint configuration. HF Inference Providers neither accept nor report a
served Hub commit, so provider-backed rows use `not_observed`. The selected
model-profile revision remains available through the checksummed private run
lock; the public row does not make an unsupported served-weight claim.

The global index uses `harbor-hf/results/index/v1`. It contains one row per
publication: run and campaign IDs, benchmark, ordinary/complete labels,
completion time, model and agent identity, result Dataset and exact revision,
source checksum, and control commit. It contains no trial, execution, metric,
artifact, session, or task-content data.

Task failures remain visible through trial outcomes, execution statuses, retry
counts, the catalog's outcome counts, and its failed-execution count. The
planned trial count is locked before execution and remains the score
denominator. Exhausted failures therefore contribute zero instead of shrinking
the denominator or suppressing valid results from the rest of the run. A run is
`clean` only when every task is scored; otherwise it is `degraded`. A run with
no valid completed trial, missing terminal evidence, or inconsistent checksums
is not publishable as an ordinary complete result.

Each publication keeps its immutable index row. The publisher also rewrites
consolidated power-of-two windows containing the newest 1, 2, 4, and so on up
to 2,048 rows in the same parent-checked commit. Readers choose the smallest
window covering their configured limit, so public refresh I/O stays bounded
without deleting the per-publication archive.

The schema review intentionally keeps a few domain-qualified names and repeated
fields. `logical_attempt` and `physical_attempt` preserve the campaign model's
benchmark-versus-infrastructure distinction. `hardware`, `model_id`,
`deployment_id`, and `agent_id` retain resolved manifest vocabulary rather than
creating presentation-only aliases. `result_kind` and `outcome` remain explicit
labels so ordinary complete rows cannot be mistaken for future composite,
manual, partial, or invalid records. `run_lock_path` and `source_checksum` are
repeated because every row must be independently traceable without relying on a
join or an implicit filename convention.

## Determinism and storage layout

Rows are sorted by stable IDs. Metric and artifact IDs are hashes of their
stable owner and measurement or artifact identity. A publication ID is a hash
of the run ID, source location, verified source checksum, and run-lock
checksum. `control_commit` is the Hub commit that last modified the immutable
campaign lock, not the moving control-repository head. Later events and leases
therefore change neither the publication identity nor its Parquet bytes.
Parquet files use deterministic paths:

```text
data/<table>/schema=v1/campaign=<campaign-id>/<publication-id>.parquet
publications/<publication-id>.json
```

The receipt path makes repeated publication a no-op. Rebuilding regenerates the
same row models and paths from canonical evidence. Adoption requires the
existing receipt and every receipt-declared file to match the newly generated
canonical bytes exactly. Conflicting historical publications are rejected
instead of being silently adopted. Auditing compares rows while checking
referential and evidence-trace invariants.

Every result projection records a canonical digest of the complete model,
deployment, and agent profiles from its verified run lock after removing only
local profile IDs and endpoint resource identity. The served model name remains
part of the digest. Composed publications reference every source at an exact
Dataset revision and source checksum, then require the source-owned profile
digests to match. Corrections may rename profiles and run on a different
otherwise-identical endpoint, but they cannot change weights, serving
arguments, inference parameters, or agent configuration.

## Serialized commits and interruption recovery

The Hub adapter acquires a publisher lease for the benchmark Dataset, rereads
its head, and creates one parent-checked commit containing all five Parquet
files and the deterministic receipt. Parent conflicts are reread and retried;
an existing matching receipt is adopted. Publisher claims expire after 15
minutes so a killed process cannot block later publication forever. A live
publisher releases its claim immediately.

The global index is a second leased, parent-checked commit made only after the
benchmark revision is known. If interruption occurs between commits, retrying
adopts the benchmark receipt and completes only the missing index commit. No
remote integration is required to test this behavior; the adapter contract is
covered with in-memory mocks.
An existing publication with no consolidated windows is repaired idempotently
on adoption; later publications update all windows in the normal index commit.

The active pre-release contract is cut over in place under `v1`; superseded
shapes are not supported by production readers. Historical immutable
publications are never rewritten. They must be rebuilt from canonical evidence
or rerun before they can enter the active catalog.
