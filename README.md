# harbor-hf

`harbor-hf` is a Harbor companion CLI for reproducible benchmark execution on
Hugging Face infrastructure. It plans experiment matrices, manages remote
inference and task environments, preserves complete run evidence, and publishes
queryable results without running model inference locally.

The project is in early development. The CLI validates and expands experiment
matrices and can submit one resolved matrix cell to an HF Job. The remote worker
controls an existing Inference Endpoint, runs Harbor tasks in HF Sandboxes,
archives evidence to an HF Bucket, and verifies that the endpoint is paused
before declaring success. Before resuming an endpoint, it starts an independent
HF Job watchdog, waits for its readiness handshake, and then resumes the
endpoint. The watchdog pauses the endpoint if the controller exits or is killed.
Controllers and watchdogs targeting the same endpoint share an atomic lease in
the namespace's private `harbor-hf-coordination` Dataset repository. The
watchdog acquires the lease with a parent-commit compare-and-swap before it
advertises readiness and releases it only after verified endpoint cleanup. A
competing watchdog fails before its controller can resume or pause the endpoint.

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

## Submit A Remote Run

Remote submission requires an endpoint binding and exact 40-character commits
for both `harbor-hf` and a Harbor revision that provides the `hf-sandbox`
extra. The controller image must be pinned by SHA-256 digest. Both checkouts
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
is written only after every requested Harbor attempt is exception-free, has
finite numeric verifier results, and the Inference Endpoint reports `paused`
with zero ready replicas. Failures write `_FAILED` after attempting the same cleanup.
The controller verifies the endpoint's model, custom image, container command,
complete ordered serving arguments, complete non-secret environment, provider
region, hardware, accelerator count, and declared scaling limits while the
endpoint is paused and again after every target replica is ready. It then probes
the endpoint's reported health route before Harbor starts.
Harbor writes raw sessions and logs only to Job-local storage. The controller
redacts and validates that staging tree before publishing it to the bucket, and
copies `_SUCCESS` or `_FAILED` last.
Submission creates or verifies the namespace-level coordination repository and
refuses to use it if it is public. It separately verifies that the configured
artifact Bucket and the implicit `jobs-artifacts` input Bucket are private.
Each run prefix receives a permanent compare-and-swap reservation before remote
work, so duplicate run IDs cannot overwrite or invalidate one another.

An experiment expands into homogeneous runs. Each run contains one benchmark
revision, model revision, deployment profile, agent profile, and execution
policy. Harbor remains responsible for task execution and verification.

The [architecture](docs/architecture.md) describes the execution and storage
boundaries. The [run specification](docs/run-spec.md) defines the portable
manifest, and the [implementation plan](docs/implementation-plan.md) tracks the
path to remote execution.

## License

[Apache-2.0](LICENSE)
