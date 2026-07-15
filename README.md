# harbor-hf

`harbor-hf` is a Harbor companion CLI for reproducible benchmark execution on
Hugging Face infrastructure. It plans experiment matrices, manages remote
inference and task environments, preserves complete run evidence, and publishes
queryable results without running model inference locally.

The project is in early development. The CLI validates and expands experiment
matrices, submits individual runs, and operates durable multi-shard campaigns.
Benchmark tasks can come from content-addressed Harbor packages or from a
public or private GitHub repository pinned to an exact commit and
repository-relative task path. Private sources reference a named HF Job secret;
credential values never enter manifests, locks, commands, or clone URLs.
Stateless reconciler Jobs provision or adopt Inference Endpoints, submit bounded
deployment waves, run Harbor tasks in HF Sandboxes, verify canonical evidence,
and publish normalized result Datasets. The remote worker archives evidence to
an HF Bucket and verifies that the endpoint is paused before declaring success.
It refuses lifecycle ownership unless the endpoint starts paused with zero ready
replicas. It then starts an independent HF Job watchdog, waits for its readiness
handshake, and resumes the endpoint. The watchdog pauses the endpoint if the
controller exits or is killed. An ambiguous readiness-label response does not
release the lease; the watchdog keeps control until it has verified the endpoint
is paused.
Controllers and watchdogs targeting the same endpoint share an atomic lease in
the namespace's private `harbor-hf-coordination` Dataset repository. The
watchdog acquires the lease with a parent-commit compare-and-swap before it
advertises readiness and releases it only after verified endpoint cleanup. A
competing watchdog fails before its controller can resume or pause the endpoint.

## Browse Results

The public [Harbor Results Space](https://huggingface.co/spaces/osolmaz/harbor-results)
compares published benchmark runs and exposes stable campaign, run, trial, and
execution URLs. Its versioned API reads one bounded catalog snapshot for list
pages and loads revision-pinned run details on demand. The public deployment has
no credential and exposes only sanitized normalized tables and artifact
metadata; complete sessions and canonical evidence remain in the private HF
Bucket.

The [results viewer plan](docs/2026-07-15-results-viewer-plan.md) defines the
long-term API, storage, privacy, deployment, and Harbor-upstream boundaries.

## Install

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required for
development installs.

```bash
git clone https://github.com/osolmaz/harbor-hf.git
cd harbor-hf
uv sync
```

## Plan An Experiment

Start from [the ShellBench example](examples/shellbench.yaml), replace its
placeholder revisions and destinations, then validate it:

```bash
uv run harbor-hf validate examples/shellbench.yaml
uv run harbor-hf plan examples/shellbench.yaml
```

`plan` performs no remote operations. It prints the resolved matrix cells and a
digest of the requested experiment.

Resolve the same manifest into deterministic campaign runs, shards, and trials:

```bash
uv run harbor-hf campaign plan experiment.yaml
uv run harbor-hf campaign plan experiment.yaml --format json
uv run harbor-hf campaign schema --output campaign.schema.json
```

Campaign planning also performs no remote operations. It requires the complete
task-digest map, applies matrix inclusion and exclusion rules, and prints a
content-addressed plan that can later be submitted more than once as distinct
campaigns.

Persist a campaign request in the private coordination Dataset, then inspect or
dry-run its next reconciliation actions:

```bash
uv run harbor-hf campaign submit experiment.yaml --dry-run
uv run harbor-hf campaign submit experiment.yaml
uv run harbor-hf campaign status CAMPAIGN_ID --namespace NAMESPACE
uv run harbor-hf campaign reconcile CAMPAIGN_ID --namespace NAMESPACE --dry-run
uv run harbor-hf campaign reconcile CAMPAIGN_ID --namespace NAMESPACE --apply
uv run harbor-hf campaign reconcile-all --namespace NAMESPACE --apply
uv run harbor-hf automation install automation.yaml
```

Submission writes the immutable request, campaign lock, permanent reservation,
and first typed event in one parent-checked commit. Reconciliation rebuilds
state from append-only events and groups compatible shards by their exact model
and deployment digest. An apply pass performs a bounded set of reserved actions,
observes worker evidence, finalizes terminal summaries, and publishes verified
results. The dry-run command does not submit Jobs or resume an endpoint.

## Submit A Remote Run

Remote submission requires an endpoint binding and exact 40-character commits
for both `harbor-hf` and a Harbor revision that provides the `hf-sandbox`
extra. The model revision must also be a full commit, serving and controller
images must be pinned by SHA-256 digest, package agents must use exact versions,
and the benchmark must include a digest-pinned dataset plus its complete
resolved task digest map. Mutable references are rejected before submission.
Both checkouts
execute with their committed `uv.lock` files in locked mode without development
dependency groups. Preview the sanitized HF Job command first:

```bash
uv run harbor-hf submit experiment.yaml --dry-run
uv run harbor-hf submit experiment.yaml
```

If a matrix dimension has more than one profile, select it explicitly with
`--model`, `--deployment`, or `--agent`. Submission sends the manifest and
resolved lock to an HF Job. The local machine does not execute the task or load
the model.

The Job writes evidence under
`runs/<experiment>/<run-id>/` in the configured private HF Bucket. `_SUCCESS`
is written only after every requested Harbor attempt is exception-free, has the
expected task content digest and finite numeric verifier results, and the
Inference Endpoint reports `paused` with zero ready replicas. Failures write
`_FAILED` after attempting the same cleanup.
The controller verifies the endpoint's model, custom image, container command,
complete ordered serving arguments, complete non-secret environment, secret
names, provider region, hardware, accelerator count, and declared scaling
limits while the endpoint is paused and again after every target replica is
ready. It then probes the endpoint's reported health route before Harbor starts.
Harbor writes raw sessions and logs only to Job-local storage. The controller
redacts and validates that staging tree before publishing it to the bucket.
Each direct Harbor trial and each campaign physical execution receives a
bounded `private-artifacts.json` inventory with a logical kind, size, and
SHA-256 digest for every retained file. A successful OpenClaw execution cannot
publish success without a captured session JSONL.
Failed executions retain an explicit missing-session requirement for diagnosis,
and the controller copies `_SUCCESS` or `_FAILED` last.
Submission creates or verifies the namespace-level coordination repository and
refuses to use it if it is public. It separately verifies that the configured
artifact Bucket and the `jobs-artifacts` input Bucket are private. Input bundles
are uploaded to content-addressed prefixes and mounted by `hf://` URI, so a
retry reuses the exact immutable manifest and lock files without an implicit
local-directory synchronization step.
Each run prefix receives a permanent compare-and-swap reservation before remote
work, so duplicate run IDs cannot overwrite or invalidate one another.

An experiment expands into homogeneous runs. Each run contains one benchmark
revision, model revision, deployment profile, agent profile, and execution
policy. Harbor remains responsible for task execution and verification.

The [architecture](docs/architecture.md) describes the execution and storage
boundaries. The [run specification](docs/run-spec.md) defines the portable
manifest. The [endpoint provisioning contract](docs/endpoint-provisioning.md)
documents deterministic endpoint ownership, and the
[result publication contract](docs/result-publication.md) freezes normalized
Parquet schemas and the evidence-safety boundary. The
[fully hosted Harbor Cookbook guide](docs/harbor-cookbook.md) covers campaign
operation and result publication. The
[Harbor compatibility contract](docs/harbor-integration-contract.md) freezes
the current checksummed request and typed result boundary. The
[Harbor integration refactor plan](docs/harbor-integration-refactor.md) tracks
the migration to a stable Harbor-owned protocol. The
[Harbor-native result publication plan](docs/harbor-native-result-publication.md)
defines the hard cutover from duplicated result models to canonical Harbor
bundles, a minimal Hugging Face execution envelope, and rebuildable query
projections. The [field ownership inventory](docs/result-field-ownership.md)
records each projected value's authority, and checked-in JSON Schemas under
[`schemas/`](schemas/) define the single canonical publication contract under
the `v1` identifier. The
[implementation plan](docs/implementation-plan.md) tracks the complete path to
remote execution.

## License

[Apache-2.0](LICENSE)
