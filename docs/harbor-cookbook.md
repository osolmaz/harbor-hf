# Fully Hosted Harbor Evaluations on Hugging Face

This recipe runs a reproducible Harbor campaign entirely on hosted Hugging Face
infrastructure. A local operator plans and submits immutable work, but does not
load a model, run inference, or execute a benchmark task locally. HF Jobs run
the controllers, Harbor runs tasks in HF Sandboxes, Inference Endpoints or
Inference Providers serve model requests, and an HF Bucket keeps canonical
evidence.

The optional result Space is a read-only view. It does not schedule work,
decide whether work is complete, hold credentials, or replace the control
Dataset, artifact Bucket, or normalized result Datasets.

> **Safety rule:** every endpoint-backed wave must finish with the Inference
> Endpoint reporting `paused` and `readyReplica=0`. This applies after success,
> failure, timeout, cancellation, controller loss, and publication failure. A
> terminal success marker is invalid until that state has been observed and
> recorded.

## Hosted architecture

```text
operator CLI
    |
    | immutable request and parent-checked reservations
    v
private control Dataset <----- Hub webhook / scheduled CPU reconcile Job
    |                                      |
    | bounded action reservations          | stateless recovery
    v                                      v
HF wave-controller Job ------------> independent watchdog Job
    |                                      |
    |                                      +---- pause endpoint on owner loss
    |
    +---- endpoint path ----> Inference Endpoint ----+
    |                                                |
    +---- provider path ----> Inference Provider ----+--> Harbor HF Sandboxes
                                                         |
                                                         v
private artifact Bucket (canonical checksummed evidence)
    |
    | serialized, verified publication
    v
normalized result Dataset ----> global index Dataset ----> read-only Space
```

The control Dataset stores small plans, leases, reservations, and events. The
Bucket stores raw evidence under unique prefixes. Result and index Datasets are
derived and rebuildable. The Space reads only those normalized Datasets at
immutable revisions.

## 1. Prepare immutable inputs

Start from [`examples/shellbench.yaml`](../examples/shellbench.yaml). Replace
every placeholder with a durable reference:

- full source commits for Harbor and `harbor-hf`;
- a full model commit;
- digest-pinned serving and controller images;
- a digest-pinned benchmark Dataset and the complete task-name-to-digest map;
- exact agent package versions or source commits;
- an endpoint deployment profile or an Inference Provider target;
- private control, input, artifact, and unpublished result storage;
- campaign, wave, shard, retry, concurrency, idle-time, duration, and spend
  bounds.

Record secret *names* in the manifest, never secret values. Give orchestration,
execution, and publication separate least-privilege HF tokens where the Hub
permission model permits it. Put those values only in the relevant HF Job or
Endpoint secret store. Do not put a token in a manifest, lock, event, log,
Dataset row, Bucket object, test fixture, or result Space.

Plan twice in clean checkouts when introducing a new benchmark or deployment:

```bash
uv run harbor-hf validate campaign.yaml
uv run harbor-hf campaign plan campaign.yaml --format json > campaign-plan.json
```

Planning performs no inference and creates no remote compute. Preserve the plan
digest, run IDs, shard IDs, trial IDs, source commits, model revision,
deployment digest, image digests, and resolved task digests from the output.

## 2. Submit and reconcile

Preview the control records before writing them:

```bash
uv run harbor-hf campaign submit campaign.yaml --dry-run
uv run harbor-hf campaign submit campaign.yaml
uv run harbor-hf campaign status CAMPAIGN_ID --namespace NAMESPACE --format json
uv run harbor-hf campaign reconcile CAMPAIGN_ID --namespace NAMESPACE --dry-run
```

Submission creates a new campaign even when the plan digest already exists. A
reconciler pass rereads the immutable plan and append-only events, derives the
current projection, reserves a bounded set of idempotent actions, performs
them, records outcomes, and exits. A Hub webhook reduces latency; a scheduled
CPU Job is the recovery path. Correctness does not depend on either retaining
memory.

Do not treat a timed-out create, resume, submit, cancel, or pause request as a
confirmed failure. The next pass must inspect deterministic remote identities
and adopt the observed resource or action before retrying.

## 3A. Endpoint-backed execution path

An endpoint deployment digest covers the model revision, engine, image,
command, ordered arguments, non-secret environment, secret names, provider,
region, hardware, accelerator count, scaling, context and batching limits,
precision, parser and template controls, caching, speculative decoding, and
health probes.

For each bounded deployment wave, the controller and watchdog must:

1. acquire the endpoint lease with a parent-checked control commit;
2. start the independent watchdog and verify its readiness handshake;
3. adopt or create only the deterministic managed endpoint with the exact
   deployment digest;
4. require the endpoint to begin paused with zero ready replicas;
5. resume it once, re-verify the complete effective configuration after every
   target replica is ready, and probe the declared health route;
6. run only the wave's assigned Harbor shards at the locked concurrency;
7. stop admitting work at the first duration, shard, idle, or spend bound;
8. drain active work, pause the endpoint, observe `readyReplica=0`, write
   cleanup evidence, and only then release the lease.

The watchdog pauses the endpoint when the controller exits or loses ownership.
Cleanup actions take priority over new billable work. Ordinary completion
pauses the endpoint; deletion is a separate, explicit retention action.

## 3B. Inference Provider execution path

A provider-backed wave uses the same campaign, run, shard, logical-trial,
physical-execution, Harbor, and artifact contracts. It does not create or lease
an Inference Endpoint. Record the requested provider and model, routing data,
request identity when exposed, retry and throttle observations, reported usage,
latency, and quoted or observed cost.

Do not infer a hidden engine, image, region, hardware, precision, cache policy,
or token count. Endpoint-only fields must be `not_applicable` and unreported
provider fields must be `not_reported`. Compare endpoint and provider runs only
on fields observed for both.

## 4. Monitor and cancel safely

Use projections instead of scraping worker logs:

```bash
uv run harbor-hf campaign status CAMPAIGN_ID --namespace NAMESPACE --format json
```

Inspect queued, active, retrying, complete, invalid, failed, and cancelled
counts; physical retries; categorized infrastructure failures; endpoint
startup, active, idle, drain, and cleanup durations; observed throughput and
latency; estimated spend; and the most recent reconciler and publisher
checkpoints.

Cancellation is a durable control request:

```bash
uv run harbor-hf campaign cancel CAMPAIGN_ID --namespace NAMESPACE
```

After that request, reconciliation must stop admitting shards, cancel queued
and active Jobs, drain or terminate according to policy, pause every owned
endpoint, verify zero ready replicas, publish the evidence that exists, and
release leases only after cleanup. Repeating cancellation is safe. Valid
completed trials are retained, and the campaign may end as `partial`.

If cancellation returns before cleanup finishes, keep reconciling and checking
status. Do not consider cancellation finished while any owned endpoint is
running, any cleanup reservation is unresolved, or a watchdog reports lost
ownership.

## 5. Verify canonical artifacts

Run verification before publication:

```bash
uv run harbor-hf artifacts verify CAMPAIGN_ID --namespace NAMESPACE --format json
```

For each run, verify the immutable `run.lock.json`, terminal marker, normalized
summary, every object listed by `checksums.json`, and every child checksum
referenced by a parent summary. Reject traversal, symlinks, unsafe archive
members, conflicting markers, missing files, extra files, mismatched task
digests, non-finite verifier values, secret material, and unsanitized task or
session content.

Endpoint-backed wave evidence must include the exact endpoint snapshot, runtime
environment, lifecycle events, final pause observation, and zero-ready-replica
observation. A verifier reward of zero is a valid result; missing or invalid
evidence is not.

Canonical evidence remains under a unique Bucket prefix such as:

```text
campaigns/<campaign-id>/
  campaign.lock.json
  waves/<wave-id>/...                 # lifecycle and cleanup evidence
  runs/<run-id>/
    run.lock.json
    shards/<shard-id>/...
    trials/<trial-id>/
      executions/<execution-id>/...  # physical retries never overwrite
    run-summary.json
    _SUCCESS | _PARTIAL | _FAILED | _CANCELLED
```

## 6. Publish derived result tables

Publish only after artifact verification succeeds:

```bash
uv run harbor-hf results publish CAMPAIGN_ID --namespace NAMESPACE --format json
```

One leased publisher serializes commits to each destination. It writes flat,
versioned Parquet tables for runs, logical trials, physical executions,
metrics, and safe artifact metadata. It then writes one global index row that
points to the benchmark Dataset and its exact commit. Retrying adopts an
existing matching publication receipt instead of duplicating rows.

Ordinary complete runs are the default comparable result class. Partial,
composite, and manually selected results require an explicit publication path
and must retain their labels. They must never be inserted into an ordinary
complete leaderboard cohort. Raw Harbor sessions, trajectories, task bodies,
logs, manifests, and archives remain in the private Bucket and are never copied
to a public Dataset.

Every published score must be traceable through these fields:

| Layer | Required provenance |
| --- | --- |
| Index row | publication, run and campaign IDs; result kind and outcome; result Dataset and exact revision; source checksum; control commit |
| Run row | benchmark, model and agent revisions; deployment identity; provider, region and hardware; source Bucket and prefix; run-lock checksum |
| Trial row | task digest, logical attempt, selected physical execution and verifier metric owner |
| Execution row | physical attempt, runtime kind, remote Job identity when reported, timestamps, status and retry reason |
| Metric row | stable metric ID, typed owner, name, value, unit and aggregation |
| Artifact row | safe metadata path, media type, size and checksum; never raw evidence bytes |

Audit or rebuild compares these derived rows with the canonical Bucket evidence.
Deleting and rebuilding a result Dataset must produce equivalent normalized
rows and stable publication paths.

## 7. Deploy the optional read-only Space

The repository's [`space/`](../space/) directory is ready to copy into a
separate HF Space repository. Creating that remote Space is an operator action;
the campaign controller and result publisher must never create or mutate it.

Set only public environment variables:

```text
HARBOR_HF_INDEX_DATASET=organization/harbor-results-index
HARBOR_HF_INDEX_REVISION=main
HARBOR_HF_MAX_PUBLICATIONS=250
HARBOR_HF_SPACE_TITLE=Harbor evaluation results
```

Do not configure a token. The Space forces anonymous Hub reads, resolves the
configured index revision to an immutable commit on each refresh, and reads
each result publication at the exact commit in its index row. It fails closed
on a schema or provenance mismatch and stores no authoritative state.

The campaign, run, task, attempt, error, throughput, hardware, cost, and
provenance tabs all show `OUTCOME · KIND`, for example
`COMPLETE · ORDINARY`, `PARTIAL · COMPOSITE`, or `COMPLETE · MANUAL`.

## Final operational checklist

- The campaign plan and all behavior-affecting references are immutable.
- No model or benchmark task ran on the operator machine.
- The control Dataset, input stores, artifact Bucket, and unpublished results
  were private.
- Every endpoint-backed wave has verified `state=paused` and
  `readyReplica=0`; provider-backed waves created no endpoint.
- Artifact verification passed against canonical checksums and exact task
  digests.
- Publication receipts, Dataset revisions, control commits, and evidence
  checksums are recorded.
- Partial, composite, and manual results are labeled and excluded from the
  ordinary-complete cohort.
- The result Space has no credentials and remains a replaceable read-only view.
