# Normalized Result Publication

Milestone 6 treats the private artifact Bucket as canonical evidence and result
Datasets as derived indexes. Publication is intentionally an application-layer
slice: campaign reconciliation and CLI wiring will call it later.

## Safety boundary

A run is publishable only when its evidence prefix has `_SUCCESS`, has no other
top-level terminal marker, and every non-marker object is present in and matches
`checksums.json`. The checksummed `run-summary.json` must declare that it was
sanitized and must describe an ordinary, complete run. The checksummed
`run.lock.json` must contain the same immutable `run_id`.

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
times, model, deployment, agent, and aggregate child-count fields. `trials` adds
the logical trial identity, task name and digest, attempt, selected execution,
and outcome. `executions` adds the physical execution identity, trial identity,
attempt, runtime kind, status, timestamps, retry reason, and optional remote Job
identity. `metrics` adds a deterministic metric identity, typed owner, name,
value, unit, and optional aggregation. `artifacts` adds a deterministic artifact
identity, typed owner, safe artifact kind, private canonical path, checksum,
media type, and size. Artifact rows are metadata and pointers only.

The global index uses `harbor-hf/results/index/v1`. It contains one row per
publication: run and campaign IDs, benchmark, ordinary/complete labels,
completion time, model and agent identity, result Dataset and exact revision,
source checksum, and control commit. It contains no trial, execution, metric,
artifact, session, or task-content data.

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
of the run ID, source location, verified source checksum, run-lock checksum,
and control commit. Parquet files use deterministic paths:

```text
data/<table>/schema=v1/campaign=<campaign-id>/<publication-id>.parquet
publications/<publication-id>.json
```

The receipt path makes repeated publication a no-op. Rebuilding regenerates the
same row models and paths from canonical evidence, and auditing compares those
rows while checking referential and evidence-trace invariants.

## Serialized commits and interruption recovery

The Hub adapter acquires a publisher lease for the benchmark Dataset, rereads
its head, and creates one parent-checked commit containing all five Parquet
files and the deterministic receipt. Parent conflicts are reread and retried;
an existing matching receipt is adopted. The lease is always released.

The global index is a second leased, parent-checked commit made only after the
benchmark revision is known. If interruption occurs between commits, retrying
adopts the benchmark receipt and completes only the missing index commit. No
remote integration is required to test this behavior; the adapter contract is
covered with in-memory mocks.

Schema changes require a new table version and a new path. Historical Parquet
files are never rewritten. Application callers must explicitly select a
supported version during rebuild; version migration and CLI exposure can be
added without changing the frozen v1 row contracts.
