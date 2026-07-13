# Experiment Manifest

This specification defines the initial portable experiment format consumed by
`harbor-hf`. An experiment is one YAML file with a required `Experiment`
resource. The format is pre-release and identified as `harbor-hf/v1alpha1`.

## Minimal Shape

```yaml
api_version: harbor-hf/v1alpha1
kind: Experiment
metadata:
  name: example
benchmark:
  dataset: terminal-bench@2.0
  dataset_digest: sha256:0000000000000000000000000000000000000000000000000000000000000000
  task_names: [example-task]
  task_digests:
    example-task: sha256:0000000000000000000000000000000000000000000000000000000000000000
matrix:
  models:
    - id: model
      repo: organization/model
      revision: 0000000000000000000000000000000000000000
      weights:
        format: safetensors
        quantization:
          method: compressed-tensors
          scheme: fp8
  deployments:
    - id: h200
      hardware: h200
      region: aws-us-east-1
      engine:
        name: vllm
        image: registry/image@sha256:0000000000000000000000000000000000000000000000000000000000000000
  agents:
    - id: agent
      name: terminus-2
      revision: bd9e606dcb99eb49de70bd741fd846cae5c7ebd1
      revision_kind: harbor-source
      reported_version: 2.0.0
artifacts:
  bucket: organization/benchmark-runs
publishing:
  dataset: organization/terminal-bench-results
remote:
  job:
    namespace: organization
    image: registry/controller@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
  worker:
    repository: organization/harbor-hf
    revision: 0123456789abcdef0123456789abcdef01234567
  harbor:
    source:
      repository: osolmaz/harbor
      revision: bd9e606dcb99eb49de70bd741fd846cae5c7ebd1
```

Unknown fields are rejected. Use `harbor-hf validate PATH` before submission.

## Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `api_version` | Yes | Manifest schema version. |
| `kind` | Yes | Must be `Experiment`. |
| `metadata` | Yes | Human identity and labels. |
| `benchmark` | Yes | Harbor dataset and task selection. |
| `matrix` | Yes | Models, deployments, and agents to combine. |
| `execution` | No | Attempt, concurrency, and timeout policy. |
| `artifacts` | Yes | Private raw bucket destination. |
| `publishing` | Yes | Result Dataset destinations. |
| `remote` | For submission | HF Job and pinned worker runtime configuration. |

### Metadata

`metadata.name` is a lowercase identifier containing letters, digits, and
hyphens. `metadata.labels` is optional non-executing metadata.

### Benchmark

`benchmark.dataset` is a Harbor dataset reference. Remote runs require its
immutable `@sha256:<64 hex>` form. `task_names` defaults to `["*"]` and remains
the selection passed to Harbor. `task_digests` enumerates the complete resolved
selection as task name to content digest. Every selection must match at least
one pinned task, and every pinned task must match a selection. A task content
digest covers its instructions, environment, verifier, and other task files.

### Matrix

The initial candidate set is the Cartesian product of `models`, `deployments`,
and `agents`. IDs must be unique within each dimension. Optional `include` rules
keep cells that match at least one rule, then `exclude` rules remove matching
cells. Omitted dimensions in a rule are wildcards. Rules use exact profile IDs,
reject unknown IDs, and may not remove every cell.

Remote model revisions must be full 40-character commit IDs, and serving images
must use `@sha256:<64 hex>` content digests. `weights.format` describes the
weight container, such as Safetensors or GGUF. Optional `weights.quantization`
records the quantization method and scheme; unquantized weights omit it.
Activation and KV-cache precision belong to the deployment profile because they
are runtime choices.

Deployment `engine.environment` contains non-secret values. `secret_names`
contains environment-variable names that the remote Job or Endpoint must inject.
Secret-like keys and keys declared in `secret_names` are rejected from
`engine.environment`; credentials must be injected by the remote platform.

Provider-specific values belong in `parameters`. They must be representable as
JSON and are preserved in the resolved lock. Secret-like keys are rejected
recursively from deployment and agent parameter mappings. Top-level agent
parameter keys cannot be empty, contain `=`, or have surrounding whitespace,
because Harbor parses them as command-line `KEY=VALUE` pairs.

An endpoint-backed deployment used for submission also has an `endpoint`
binding with `namespace`, `name`, and the OpenAI-compatible
`served_model_name`. The binding identifies an existing endpoint; planning does
not inspect or resume it.

Engine identity is more than `engine.name`. Resolution and submission preserve
the engine version and build or commit, immutable image digest, container
command, full arguments, non-secret environment, secret names, runtime and
driver versions, parser and chat-template identity, cache precision, batching
limits, and feature controls
such as prefix caching, speculation or MTP, CUDA graphs, attention backend, and
MoE backend. Values observed after startup are stored separately from requested
values so a provider default cannot silently change the run definition.

The current `v1alpha1` deployment shape represents Hugging Face Inference
Endpoints. vLLM and llama.cpp are independent engine choices within that
endpoint type. A planned discriminated Inference Providers profile will cover
models that are too large or expensive to host on a dedicated endpoint; it will
not require or imply a particular serving engine. Until that profile is
implemented, provider-routed manifests are rejected rather than silently
treated as endpoint deployments.

### Execution

`attempts` counts independent logical attempts. Infrastructure retries do not
consume attempt ordinals. `concurrent_trials` limits Harbor trial concurrency;
`max_trials_per_shard` deterministically bounds the number of task-attempt pairs
in one campaign shard and defaults to 64. `max_shards_per_wave` bounds compatible
shards assigned under one endpoint startup and defaults to 8. Provider request
concurrency is part of the deployment profile. Timeout values are in seconds.
`timeout_seconds` is a wall-clock limit for Harbor execution; on expiry, the
controller terminates the Harbor process group and immediately enters verified
endpoint cleanup.

Every task selected by `benchmark.task_names` is passed to Harbor. The resolved
`task_digests` map gives exact and glob selections a deterministic trial count.
The controller requires every pinned task and attempt, rejects unpinned task
names, and compares each trial's Harbor `lock.json` task digest with the run
lock. It then validates every resulting trial for exceptions and finite numeric
verifier rewards.

Agent revisions declare how they are enforced. `package` passes the revision to
an installed agent and requires an exact numeric package version rather than a
tag or version range; Harbor must report that same version.
`harbor-source` means the agent implementation is part of Harbor: its revision
must equal `remote.harbor.source.revision`, no package version is passed, and
`reported_version` records the semantic version Harbor must report.

### Artifacts

`artifacts.bucket` identifies private raw storage for complete run evidence.

### Publishing

`publishing.dataset` identifies the versioned, benchmark-specific publication.
`index_dataset` optionally identifies the global run catalog.

### Remote Execution

`remote.job` pins the HF Job namespace, digest-pinned controller image, hardware
flavor, timeout, and `HF_TOKEN` secret injection. The token secret name is fixed
because the HF CLI can resolve it from the authenticated local credential
without putting a token value in the command. `remote.worker` pins this package
to an exact GitHub commit. `remote.harbor.source` likewise pins Harbor to an
exact GitHub commit and configures the HF Sandbox flavor and idle timeout.
Source revisions must be full lowercase 40-character Git commit IDs. The
current source transport accepts GitHub `owner/name` references or HTTPS GitHub
URLs. The controller checks out both revisions directly and runs them with
`uv --locked`;
missing or stale lock files fail before endpoint-backed benchmark execution
begins. The pinned Harbor revision must expose the `hf-sandbox` optional
dependency; the worker verifies that capability before it resumes the endpoint.

For endpoint-backed runs, `remote.job.namespace` must equal the selected
endpoint namespace. Submission creates or verifies a private
`<namespace>/harbor-hf-coordination` Dataset repository. The watchdog uses an
initialization commit as the first parent in a new repository, then uses an
endpoint-specific file committed against an expected parent revision as an
atomic lease and removes it with the same compare-and-swap protocol only after
verified cleanup.

The controller Job timeout is limited to 85,800 seconds. The remaining 600
seconds within HF Jobs' 86,400-second maximum are reserved for watchdog startup
and verified endpoint cleanup. It must also exceed `execution.timeout_seconds`
by at least 4,800 seconds, reserving time for source bootstrap, watchdog
readiness, endpoint startup, and controller cleanup. The endpoint is not resumed until the watchdog
has completed its source bootstrap and published a readiness handshake.
Endpoint readiness has its own 3,600-second allowance and does not consume or
inherit the Harbor execution timeout.
Readiness requires every positive `targetReplica` to be represented by a ready
replica. The controller then probes the endpoint's reported `healthRoute`
instead of assuming a custom image uses `/health`.
Before starting the watchdog, the controller requires a paused endpoint with
zero ready replicas. Before resuming and again before benchmarking, it compares
the observable endpoint model, custom image, container command, complete
ordered serving arguments, complete non-secret environment, secret key names,
provider region, hardware, accelerator count, and declared replica limits with
the resolved deployment. A mismatch is a run failure, not a warning.

The HF Sandbox idle timeout must exceed the longest uninterrupted agent or
verifier command. A command can keep one streaming SDK request open without
resetting the Sandbox idle timer; if the timer expires first, the remote job is
terminated mid-command. The default is 3,600 seconds. Set it above the
benchmark's agent timeout, but never above the controller Job timeout. Manifest
validation enforces that upper bound so an abandoned Sandbox cannot outlive the
controller's configured lifecycle.

Only secret names are serialized. The configured token is forwarded through the
HF Jobs secret mechanism to the controller and its cleanup watchdog, then
inherited by Harbor through process environment. Its value is absent from
commands, locks, and evidence. Before archiving, secret values are redacted from
both file contents and path components using bounded-memory streaming. Symbolic
links are rejected before evidence is read, modified, hashed, or archived.
Prefixed API, access, private key, and personal access token (`PAT`) names are
treated as secrets, including camel-case and uppercase environment forms.

Harbor's raw job tree is created on Job-local storage rather than the bucket
mount. Before remote work, the worker creates a permanent run reservation with
a parent-commit compare-and-swap in the private coordination repository. Bucket
references are canonicalized before deriving the reservation, so equivalent
URI spellings cannot reserve the same destination independently. Only
the finalized, scrubbed tree is copied to its reserved Bucket prefix, and the
root terminal marker is copied last. Nested task markers are preserved. If the
controller is killed before finalization, raw sessions and logs disappear with
the Job instead of remaining in the bucket. Submission queries both the
configured artifact Bucket and the managed `jobs-artifacts` input Bucket and
refuses to start a Job unless both are private. It uploads manifests and locks
under a content-addressed Job input prefix and mounts that exact Bucket
subdirectory read-only.

## Loading And Resolution

Validation checks the requested document. Planning expands the matrix and
computes a digest from canonical JSON. Remote validation and submission reject
mutable dataset, task, model, serving-image, source, and agent references. The
caller resolves them before submission; the separate lock preserves the exact
selected matrix cell without rewriting the requested document.

Campaign planning additionally sorts resolved cells and tasks, creates one
logical trial per task and requested attempt, splits those trials into bounded
shards, and content-addresses every cell and shard. The campaign plan digest is
derived from resolved execution semantics; the separate manifest digest still
identifies the exact requested document. Reordering equivalent profile lists or
task-digest mappings therefore does not change the campaign plan digest.

Every submitted run writes `manifest.yaml`, `run.lock.json`,
`endpoint.snapshot.json`, and `runtime-environment.json`. Provider-backed runs
must mark hidden details as `not_reported`; failed collection is `unknown`, and
irrelevant fields are `not_applicable`. These statuses accompany null values
rather than being stored as fake version or hardware strings.

Before remote work, the worker reconstructs the selected matrix cell from the
manifest and compares every field with the submitted run lock. A matching
manifest digest alone is not sufficient.

## Not Covered

The manifest does not define Harbor tasks, verifier behavior, agent internals,
leaderboard presentation, secret storage, or provider credentials. Those remain
owned by Harbor, the selected agent, Hugging Face, or the presentation layer.
