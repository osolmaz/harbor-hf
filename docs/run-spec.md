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
matrix:
  models:
    - id: model
      repo: organization/model
      revision: replace-with-immutable-revision
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
        image: registry/image@sha256:replace-with-digest
  agents:
    - id: agent
      name: terminus-2
      revision: 0123456789abcdef0123456789abcdef01234567
      revision_kind: harbor-source
      reported_version: 2.0.0
artifacts:
  bucket: organization/benchmark-runs
publishing:
  dataset: organization/terminal-bench-results
remote:
  job:
    namespace: organization
  worker:
    repository: organization/harbor-hf
    revision: 0123456789abcdef0123456789abcdef01234567
  harbor:
    source:
      repository: harbor-framework/harbor
      revision: 0123456789abcdef0123456789abcdef01234567
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

`benchmark.dataset` is a Harbor dataset reference. `task_names` defaults to
`["*"]`. The planner cannot calculate a trial count for wildcard selection
until the dataset is resolved; a later resolved lock must contain every task
digest.

### Matrix

The initial alpha format takes the Cartesian product of `models`, `deployments`,
and `agents`. IDs must be unique within each dimension. A future format may add
explicit inclusion and exclusion rules without changing the resolved run model.

Model revisions and runtime image references should be immutable commit or
content digests. `weights.format` describes the weight container, such as
Safetensors or GGUF. Optional `weights.quantization` records the quantization
method and scheme; unquantized weights omit it. Activation and KV-cache
precision belong to the deployment profile because they are runtime choices.

Deployment `engine.environment` contains non-secret values. `secret_names`
contains environment-variable names that the remote Job or Endpoint must inject.
Secret values are invalid manifest content.

Provider-specific values belong in `parameters`. They must be representable as
JSON and are preserved in the resolved lock.

An endpoint-backed deployment used for submission also has an `endpoint`
binding with `namespace`, `name`, and the OpenAI-compatible
`served_model_name`. The binding identifies an existing endpoint; planning does
not inspect or resume it.

Engine identity is more than `engine.name`. Resolution and submission preserve
the engine version and build or commit, immutable image digest, full arguments,
non-secret environment, secret names, runtime and driver versions, parser and
chat-template identity, cache precision, batching limits, and feature controls
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
provider request concurrency is part of the deployment profile. Timeout values
are in seconds. `timeout_seconds` is a wall-clock limit for Harbor execution;
on expiry, the controller terminates the Harbor process group and immediately
enters verified endpoint cleanup.

Every task selected by `benchmark.task_names` is passed to Harbor. Exact task
names have a deterministic expected trial count of tasks multiplied by
attempts. Glob selections are resolved by Harbor; the controller requires at
least one result and validates every resulting trial for exceptions and numeric
verifier rewards.

Agent revisions declare how they are enforced. `package` passes the revision to
an installed agent and requires Harbor to report that same version.
`harbor-source` means the agent implementation is part of Harbor: its revision
must equal `remote.harbor.source.revision`, no package version is passed, and
`reported_version` records the semantic version Harbor must report.

### Artifacts

`artifacts.bucket` identifies private raw storage for complete run evidence.

### Publishing

`publishing.dataset` identifies the versioned, benchmark-specific publication.
`index_dataset` optionally identifies the global run catalog.

### Remote Execution

`remote.job` pins the HF Job namespace, controller image, hardware flavor,
timeout, and secret variable name. `remote.worker` pins this package to an exact
GitHub commit. `remote.harbor.source` likewise pins Harbor to an exact GitHub
commit and configures the HF Sandbox flavor and idle timeout. Source revisions
must be full lowercase 40-character Git commit IDs. The controller checks out
both revisions directly and runs them with `uv --locked`; missing or stale lock
files fail before endpoint-backed benchmark execution begins.

The controller Job timeout is limited to 85,800 seconds. The remaining 600
seconds within HF Jobs' 86,400-second maximum are reserved for watchdog startup
and verified endpoint cleanup. The endpoint is not resumed until the watchdog
has completed its source bootstrap and published a readiness handshake.

The HF Sandbox idle timeout must exceed the longest uninterrupted agent or
verifier command. A command can keep one streaming SDK request open without
resetting the Sandbox idle timer; if the timer expires first, the remote job is
terminated mid-command. The default is 3,600 seconds. Set it above the
benchmark's agent timeout while keeping the controller Job timeout as the
outer bound.

Only secret names are serialized. The configured token is forwarded through the
HF Jobs secret mechanism to the controller and its cleanup watchdog, then
inherited by Harbor through process environment. Its value is absent from
commands, locks, and evidence.

## Loading And Resolution

Validation checks only the requested document. Planning expands the matrix and
computes a digest from canonical JSON. Submission will resolve mutable names to
immutable revisions, query effective provider configuration, and write a
separate lock. The requested document is never rewritten with resolved values.

Every submitted run writes `manifest.yaml`, `run.lock.json`,
`endpoint.snapshot.json`, and `runtime-environment.json`. Provider-backed runs
must mark hidden details as `not_reported`; failed collection is `unknown`, and
irrelevant fields are `not_applicable`. These statuses accompany null values
rather than being stored as fake version or hardware strings.

## Not Covered

The manifest does not define Harbor tasks, verifier behavior, agent internals,
leaderboard presentation, secret storage, or provider credentials. Those remain
owned by Harbor, the selected agent, Hugging Face, or the presentation layer.
