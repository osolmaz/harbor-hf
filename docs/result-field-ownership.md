# Result Field Ownership

This inventory fixes the authority boundary between Harbor and `harbor-hf`.
Dataset columns may repeat values for query performance, but repetition does
not transfer ownership.

## Harbor-Owned Values

| Published field | Canonical source | Projection rule |
| --- | --- | --- |
| `trial_id` | Harbor `TrialResult.id` | Copy from the pinned compatibility export. |
| `task_name`, `task_digest` | Harbor trial lock and result | Public allowlist only; never publish task bodies. |
| verifier metric names and values | Harbor `VerifierResult.rewards` | Preserve names and numeric values without reinterpretation. |
| trial outcome and selected result | Harbor trial result | Derive only through the pinned Harbor models. |
| Harbor timing and usage | Harbor job and trial results | Add query columns only when the public contract explicitly allows them. |
| Harbor exceptions | Harbor job and trial results | Keep complete values private; public projections use approved classifications only. |
| native artifacts and sessions | Harbor job directory | Keep bytes in the private Bucket; public rows contain allowlisted metadata only. |

`harbor-hf` does not define replacement models for these concepts. Until
Harbor provides a storage-neutral export contract, the compatibility exporter
runs inside the pinned Harbor environment and records native serialized model
paths and digests in `harbor-native-bundle.json`.

## harbor-hf-Owned Values

| Published field | Canonical source |
| --- | --- |
| campaign and run IDs | campaign and run locks |
| physical execution ID and attempt | execution lock |
| physical execution bundle status | verified bundle presence or `not_available` for failed or cancelled execution |
| infrastructure status and retry reason | campaign recovery events |
| provider, region, hardware, and accelerator count | resolved deployment lock |
| model repository, revision, engine, quantization, context, and concurrency | resolved model and deployment profiles |
| remote HF Job, Endpoint, Dataset, Bucket, and Space identity | HF control-plane evidence |
| endpoint cleanup outcome | terminal campaign decision after all waves close |
| source, archive, envelope, and projection checksums | immutable evidence and publication manifests |
| sanitizer and projector versions | `harbor-hf` publication contract |

## Derived Query Values

The `runs`, `trials`, `executions`, `metrics`, and `artifacts` Parquet tables are
query projections, not canonical evidence. The cutover replaces their contract
in place under the `v1` identifier; superseded shapes are not retained by the
production reader.

The following catalog values are derived:

- `score`: mean selected trial reward under the documented reward-selection rule;
- `passed_trials`: selected trial rewards greater than or equal to `1.0`;
- `duration_seconds`: run completion time minus run creation time;
- `infrastructure_failures`: count of physical executions classified as infrastructure failures;
- row counts: counts of validated projection rows.

Every catalog row points to a checksummed projection manifest. That manifest
binds the derived tables to one canonical v1 execution envelope and its Harbor
archive digests. A successful run without verified native bundle provenance is
excluded from the active catalog until it is rebuilt or rerun.

## Privacy Boundary

Public publication rejects credentials, environment values, task bodies,
hidden tests, solutions, raw sessions, unrestricted logs, trajectories,
tracebacks, and artifact bytes. Those remain in the private Bucket under the
run's immutable evidence prefix. A public Dataset contains only allowlisted
rows and manifest references; the public Space has no Bucket credential.
