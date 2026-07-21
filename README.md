<p align="center">
  <img alt="harbor-hf" src="assets/harbor-hf-logo.svg" width="440">
</p>

`harbor-hf` is a Harbor companion CLI for running benchmark campaigns on
Hugging Face infrastructure. It takes an experiment manifest describing a
matrix of models, deployments, and agents, executes every cell remotely on HF
Jobs, Inference Endpoints, and Sandboxes, and publishes verified, queryable
result tables — without loading a model or running a task on your machine.

Three properties hold across every run:

- **Nothing mutable executes.** Manifests pin exact commits for code, models,
  and benchmarks, SHA-256 digests for images, and content digests for every
  task. Anything less is rejected before submission.
- **No endpoint outlives its work.** An independent watchdog Job holds a
  compare-and-swap lease on every Inference Endpoint and pauses it if the
  controller dies. Success is declared only after the endpoint is verified
  paused with zero ready replicas.
- **Evidence before results.** Sessions, logs, verifier output, and checksums
  are redacted, validated, and archived to a private HF Bucket before a run can
  publish. Published tables always trace back to canonical evidence.

## Browse Results

The public [Harbor Results Space](https://huggingface.co/spaces/osolmaz/harbor-results)
compares final benchmark evaluations and exposes stable campaign, run, trial,
and execution URLs. It serves only sanitized normalized tables and artifact
metadata; complete sessions and canonical evidence stay in the private HF
Bucket.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/osolmaz/harbor-hf.git
cd harbor-hf
uv sync
```

Remote operations authenticate through your Hugging Face token
(`hf auth login`, or set `HF_TOKEN`). The account needs access to HF Jobs,
Inference Endpoints, and Buckets in the target namespace. On first submission,
`harbor-hf` creates a private `harbor-hf-coordination` Dataset in the
namespace to hold campaign state, and verifies that it and the artifact
Buckets are private before doing any work.

## Plan an Experiment

Start from [the ShellBench example](examples/shellbench.yaml), replace its
placeholder revisions and destinations, then validate and resolve it:

```bash
uv run harbor-hf validate experiment.yaml
uv run harbor-hf plan experiment.yaml
uv run harbor-hf campaign plan experiment.yaml
```

Planning is entirely local. `plan` prints the resolved matrix cells and the
experiment digest; `campaign plan` resolves the same manifest into
deterministic, content-addressed runs, shards, and trials. The manifest format
is defined in the [run specification](docs/run-spec.md), and
`campaign schema` exports the plan and lock JSON Schemas.

## Run a Campaign

```bash
uv run harbor-hf campaign submit experiment.yaml
uv run harbor-hf campaign status CAMPAIGN_ID --namespace NAMESPACE
uv run harbor-hf campaign reconcile CAMPAIGN_ID --namespace NAMESPACE --apply
```

`submit` persists an immutable campaign request in the coordination Dataset.
Each `reconcile` pass is stateless: it rebuilds campaign state from
append-only events, provisions or adopts the endpoints the next shards need,
submits bounded deployment waves as HF Jobs, observes worker evidence, and
exits. `reconcile-all --apply` does one pass over every campaign in the
namespace. Every mutating command takes `--dry-run` to preview its actions
without touching remote resources.

Install the managed automation so reconciliation runs without you:

```bash
uv run harbor-hf automation install automation.yaml --schedule "<cron>" \
  --provider-active-waves 2
```

For a bounded campaign queue, repeat `--campaign-id CAMPAIGN_ID` to keep each
automation pass scoped to that queue instead of scanning historical campaigns.

This sets up a scheduled HF Job plus a control webhook, so campaigns make
progress promptly after every state change and recover when a webhook is
missed. Set `--provider-active-waves` to the live serving quota when several
campaigns share one inference provider.

Operate a running campaign with:

```bash
uv run harbor-hf campaign cancel CAMPAIGN_ID --namespace NAMESPACE
uv run harbor-hf campaign retry CAMPAIGN_ID --shard SHARD_ID --namespace NAMESPACE
uv run harbor-hf campaign resume CAMPAIGN_ID --namespace NAMESPACE --cleanup-verified
uv run harbor-hf campaign seal CAMPAIGN_ID --namespace NAMESPACE
```

`cancel` records a durable cancellation and drains work; `retry` requests an
immediate retry for a shard's retryable trials; `resume` records that an
operator verified endpoint cleanup after a manual-intervention stop; `seal`
closes out a drained partial campaign by recording its failed retries as
zero-score outcomes. The
[Harbor Cookbook](docs/harbor-cookbook.md) walks through full campaign
operation end to end.

## Publish Results

```bash
uv run harbor-hf artifacts verify CAMPAIGN_ID --namespace NAMESPACE
uv run harbor-hf results publish CAMPAIGN_ID --namespace NAMESPACE
```

`artifacts verify` checks publishable run evidence against every declared
checksum. `results publish` verifies evidence again and writes the normalized
Parquet tables that the Results Space serves. `results catalog` records
append-only promote or withdraw decisions for a publication in the primary
catalog. The
[result publication contract](docs/result-publication.md) freezes the table
schemas, and the checked-in [JSON Schemas](schemas/) define the canonical
publication contract.

## Submit a Single Run

One resolved matrix cell can run outside a campaign:

```bash
uv run harbor-hf submit experiment.yaml --dry-run
uv run harbor-hf submit experiment.yaml
```

If a matrix dimension has more than one profile, select the cell explicitly
with `--model`, `--deployment`, or `--agent`. The Job writes its evidence
under `runs/<experiment>/<run-id>/` in the configured private Bucket and marks
it `_SUCCESS` or `_FAILED` only after endpoint cleanup is verified.

## Architecture

The [architecture overview](docs/architecture.md) describes the execution and
storage boundaries: benchmark tasks come from content-addressed Harbor
packages or commit-pinned GitHub repositories, agents run in HF Sandboxes,
models serve from Inference Endpoints or Inference Providers, and all
coordination happens through parent-checked commits to the private
coordination Dataset — there is no server to keep alive. The
[endpoint provisioning contract](docs/endpoint-provisioning.md) documents
deterministic endpoint ownership. The
[deployment profiling contract](docs/deployment-profiling.md) defines the
powers-of-two concurrency method, immutable profile evidence, stopping rules,
and selection criteria used before a full campaign. The proposed [trial
evidence bundle](docs/trial-evidence-bundle.md) and its [implementation
plan](docs/trial-evidence-implementation-plan.md) define complete post-agent
workspace capture and exact verifier-judge records.

## License

[Apache-2.0](LICENSE)
