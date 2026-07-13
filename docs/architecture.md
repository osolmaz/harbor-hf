# Architecture

## Purpose

`harbor-hf` is the control plane around Harbor. Harbor owns tasks, agents,
environments, verification, trajectories, and trial results. `harbor-hf` owns
experiment expansion, Hugging Face resource lifecycle, reproducibility
metadata, raw artifact retention, and result publication.

The package must remain useful as an independent Harbor plugin and be shaped so
that it could later move into a Harbor monorepo package without architectural
changes.

## Components

```text
experiment.yaml
      |
      v
planner -> resolved run locks -> controller
                               |          |          |
                      HF Endpoints   HF Providers   HF Jobs/Sandboxes
                    (vLLM/llama.cpp)  (large models)       |
                               |          |                |
                               +----------+ Harbor trials -+
                                              |
                                    private HF bucket
                                              |
                                       publisher job
                                              |
                                   versioned HF Datasets
                                              |
                                      leaderboard Space
```

### Deployment Strategy

Inference Endpoints are the primary serving path. Engine choice is independent
of resource type: normal experiments may deploy vLLM, llama.cpp, or another
supported engine on separate endpoints. A Red Hat vLLM recipe and our llama.cpp
recipe are therefore endpoint engine profiles, not different Hugging Face
resource types. Using the Red Hat recipe does not mean routing requests through
Hugging Face Inference Providers.

Inference Providers are a secondary path for models that are too large or too
expensive to host on a dedicated endpoint. Provider-backed runs use the same
Harbor agent contract, but do not pretend that unreported runtime, hardware, or
quantization details are known. Endpoint and provider profiles remain distinct
in manifests, locks, metrics, and result tables.

### Planner

The planner validates an experiment, expands its matrix, and produces one
homogeneous run per model, deployment, and agent cell. Planning must not create
remote resources. Every resolved run receives a content digest before it is
submitted.

### Controller

The submission CLI stages an immutable manifest and run lock and starts an HF
Job. That remote Job is the controller: it starts endpoint inference only when
work is ready, submits bounded Harbor trials to HF Sandboxes, records lifecycle
events, and pauses the endpoint in a `finally` path. The submitting machine does
not execute benchmark tasks. Provider-backed runs will skip endpoint
provisioning but retain request, quota, retry, and accounting state.

### Harbor Adapter

Harbor remains the only benchmark execution engine. The adapter translates a
resolved run into Harbor job configuration and registers public lifecycle hooks
for incremental artifact publication. It must not patch Harbor internals.

### Artifact Store

A private HF Storage Bucket is the complete evidence archive. Requested and
resolved configuration, endpoint snapshots, Harbor output, trajectories,
sessions, verifier records, logs, and checksums are written under an immutable
run prefix. Trial archives are uploaded as trials finish. A `_SUCCESS` marker is
written only after validation and resource cleanup.

### Results Publisher

One serialized publisher converts completed runs into normalized Parquet tables.
Benchmark-specific Dataset repositories contain run, trial, execution, metric,
and artifact tables. A small global index contains one row per completed run.
The bucket is canonical evidence; Datasets are the query and presentation layer.

## Identity

The stable hierarchy is:

1. An experiment groups a requested matrix.
2. A run represents one homogeneous matrix cell.
3. A trial represents one task and logical attempt.
4. An execution represents one physical invocation, including infrastructure
   retries.

Retries never replace previous executions. Composite or manually selected
results must be labeled explicitly and must not appear as single-run results.

## Reproducibility Boundary

A resolved run lock records:

- benchmark source, revision, task-set digest, and verifier digest;
- Harbor, agent, tool, prompt, and skill revisions;
- model, tokenizer, chat-template, and generation-config revisions;
- weight format and quantization separately from activation and KV precision;
- resource type and, when known, provider, region, hardware, accelerator count,
  replica policy, and driver information;
- serving engine name, version, build or commit, immutable container image
  digest, complete arguments, non-secret environment, and secret names;
- CUDA, framework, attention, reasoning-parser, and other serving-library
  versions reported by the runtime;
- chat template source and digest, context and output limits, batching and
  sequence capacity, and activation and KV-cache precision;
- prefix caching, speculative decoding or MTP, CUDA graph mode, attention
  backend, MoE backend, and other behavior-affecting engine controls;
- context, output, batching, concurrency, retry, timeout, and sampling settings;
- requested provider configuration and the effective configuration returned by
  the provider.

Secret values are never recorded. Manifests store only the names of secrets
that must be injected by the remote platform.

### Canonical Configuration Artifacts

Each run preserves four separate configuration records:

| Artifact | Authority |
| --- | --- |
| `manifest.yaml` | Immutable copy of what the user requested. |
| `run.lock.json` | Resolved revisions, matrix cell, policy, and reproducibility contract fixed before execution. |
| `endpoint.snapshot.json` | Effective endpoint or provider configuration observed after readiness. |
| `runtime-environment.json` | Versions and feature controls reported from inside the serving runtime. |

The endpoint snapshot never overwrites the requested manifest or resolved lock.
Differences between requested and effective values remain explicit and are
published as comparison fields.

Runtime evidence uses a status alongside nullable values. `reported` means the
value came from a named probe or provider response, `not_reported` means the
remote service did not expose it, `not_applicable` means the field does not
apply to that serving path, and `unknown` means collection was expected but
failed. The system never infers a hidden vLLM version, quantization, hardware,
or other provider detail from model behavior.

## Failure And Cleanup

Runs progress through `planned`, `submitted`, `provisioning`, `running`,
`verifying`, `publishing`, and a terminal state. Every transition is an
append-only event.

For endpoint-backed runs, the controller starts a separate companion HF Job
watchdog before it resumes the endpoint. The controller requires a readiness
label written from inside the watchdog's monitoring process; a submitted or
merely running Job is not sufficient. The watchdog observes the controller Job
and pauses the endpoint after the controller terminates or its own deadline
expires. The controller also pauses the endpoint after its last active shard in
a `finally` path. Cleanup success is part of endpoint run completion, not an
optional maintenance action. Provider-backed runs have no endpoint lease but
still close worker resources and record final usage and request state.

The worker verifies `status.state = paused` and `readyReplica = 0` before it
writes `_SUCCESS`. The final snapshot also records `targetReplica`, which may
remain nonzero on a paused endpoint. HF Sandbox environments are killed by
Harbor, and their idle timeout limits abandoned resources if the controller is
terminated. The idle timeout must exceed the longest uninterrupted agent or
verifier command because an active streaming command does not necessarily
refresh the Sandbox idle timer.

## Boundaries

- No local model loading or inference.
- No benchmark-specific behavior in package code.
- No raw sessions in public Dataset repositories.
- No secret values in manifests, logs, locks, or artifacts.
- No state that exists only on the submitting machine.
- No direct writes from trial workers to shared Dataset Git repositories.
