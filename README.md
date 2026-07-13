# harbor-hf

`harbor-hf` is a Harbor companion CLI for reproducible benchmark execution on
Hugging Face infrastructure. It plans experiment matrices, manages remote
inference and task environments, preserves complete run evidence, and publishes
queryable results without running model inference locally.

The project is in early development. The current CLI validates the initial
experiment format and expands its model, deployment, and agent matrix. It does
not yet create remote resources.

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

An experiment expands into homogeneous runs. Each run contains one benchmark
revision, model revision, deployment profile, agent profile, and execution
policy. Harbor remains responsible for task execution and verification.

The [architecture](docs/architecture.md) describes the execution and storage
boundaries. The [run specification](docs/run-spec.md) defines the portable
manifest, and the [implementation plan](docs/implementation-plan.md) tracks the
path to remote execution.

## License

[Apache-2.0](LICENSE)

