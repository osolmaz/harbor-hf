# Deployment Profiling

This specification defines how Harbor-HF chooses benchmark concurrency for one
exact serving deployment. A selected profile is one immutable JSON document
stored with its raw measurements in the private artifact Bucket.

The general measurement method is engine-independent. Harbor-HF applies it to
remote Inference Endpoints and Inference Providers; it never loads a model or
runs benchmark tasks on the operator machine.

## Minimal Profile

```json
{
  "schema_version": "harbor-hf/serving-profile/v1",
  "profile_id": "ornith-35b-q4-h200-64k-20260718",
  "created_at": "2026-07-18T01:00:00Z",
  "identity": {
    "model_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    "deployment_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    "agent_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    "benchmark_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    "harbor_runtime_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    "server_context_tokens": 65536,
    "max_output_tokens": 8192,
    "reasoning_required": true,
    "sample_task_count": 8,
    "sample_task_names": ["task-a", "task-b", "task-c", "task-d", "task-e", "task-f", "task-g", "task-h"],
    "sample_tasks_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  },
  "objective": {
    "kind": "maximum_goodput",
    "maximum_error_rate": 0.0
  },
  "workload": {
    "kind": "benchmark",
    "sample_task_count": 8,
    "sample_task_names": ["task-a", "task-b", "task-c", "task-d", "task-e", "task-f", "task-g", "task-h"],
    "sample_tasks_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    "minimum_observations_per_point": 8,
    "boundary_repetitions": 3
  },
  "candidate_concurrency": [1, 2, 4, 8, 16, 32, 64],
  "points": [],
  "selection": null,
  "artifacts": {
    "bucket": "organization/benchmark-runs",
    "prefix": "serving-profiles/ornith-35b-q4-h200-64k-20260718"
  }
}
```

The checked-in
[`serving-profile-v1.schema.json`](../schemas/serving-profile-v1.schema.json)
is the machine-readable contract. The
[`serving-profile.json`](../examples/serving-profile.json) example validates
against it.

## Identity And Reuse

`model_sha256`, `deployment_sha256`, and `agent_sha256` are canonical profile
digests. Together they cover the model repository and revision, weight format
and quantization, serving engine and image, ordered arguments, hardware,
replicas, activation and KV precision, context and batching limits, chat
template, reasoning parser, sampling, caching, and speculative decoding.
The deployment digest excludes the endpoint resource reference. A managed
endpoint receives its deterministic name only after planning, and that
transient address must not change the serving configuration identity.
`harbor_runtime_sha256`, reasoning mode, exact sampled task names, and the
sampled-task digest bind the selected concurrency to the exact Harbor client
runtime and benchmark workload used to measure it. Plans may choose an explicit
cohort when some benchmark tasks require sandbox capabilities unavailable on
the profiling backend. That cohort is immutable evidence, not a runtime skip
list.

`benchmark_sha256` covers the benchmark revision, task digests, and the sampled
workload distribution. It prevents a short synthetic sweep from being treated
as proof of ShellBench task throughput.

The profile's `plan.json` contains the resolved model, deployment, agent, and
benchmark profiles used to derive these digests. The final profile remains
small and queryable while the plan preserves every behavior-affecting value.

A campaign may reuse a selection only when all four digests and both token
limits match exactly. A runtime, quantization, template, reasoning, hardware,
context, output, or benchmark change requires a new profile. A profile is not
portable merely because the model name and GPU name are unchanged.

## Candidate Ladder

Run a verified smoke request, then test concurrency in ascending powers of two:

```text
c1, c2, c4, c8, c16, c32, c64, c128, ...
```

The ladder is not capped at `c64`. Extend it while aggregate throughput or
goodput improves and the deployment remains stable. After identifying the
last-good and first-bad powers of two, optional refinement points such as `c24`
may be added between them.

Use at least `max(8, 2 * concurrency)` observations at each point; the declared
minimum is only a floor. Repeat boundary candidates at least three times. Keep
client request concurrency, server sequence capacity, Harbor trial concurrency,
active shards, and replica count separate in the evidence.

Provider profiles use distinct benchmark tasks for every observation at a point
so independent trials retain independent recorder retry budgets. Before the
ladder starts, calibrated requests approach the declared context boundary while
submitting the full declared output limit; a profile fails if either limit is
not accepted.

## Workload

Profile against the workload the full campaign will run. For benchmark-speed
selection, use a representative task sample with the same agent, tools,
reasoning mode, context limit, output limit, sampling, and sandbox shape.

Synthetic request tests may separately characterize prefill and decode
capacity, but they do not select Harbor task concurrency by themselves. Record
observed prompt and output token distributions rather than claiming that every
request exercised the configured context limit.

## Measurements

Each completed benchmark point records:

- planned, completed, and failed request or task counts;
- aggregate input and output tokens per second;
- task completions per hour when profiling benchmark work;
- per-session output tokens per second;
- p50, p95, and p99 trial latency;
- error and goodput rates;
- peak device memory when observable;
- the raw artifact prefix and checksum.

It also records observed p50, p95, and maximum prompt and output tokens. These
measure active workload shape and must not be replaced with configured limits.

TTFT and TPOT are reported only when a streaming recorder measures them. A
non-streaming full-response duration must never be labeled TTFT or TPOT.

Successful HTTP responses are insufficient. The ladder runs the pinned Harbor
task sample through the declared agent and sandbox, then verifies token
accounting, task completion, endpoint logs, agent exits, truncation, timeouts,
and hidden 4xx or 5xx responses.

## Stopping Rules

Confirm and stop the ascending ladder when one of these conditions occurs:

- allocation failure, OOM protection, or unsafe memory headroom;
- repeated request failures, malformed output, timeouts, or endpoint errors;
- aggregate throughput or goodput is flat or lower at two successive points;
- declared p95 or p99 latency limits are exceeded;
- per-session decode speed falls below the declared minimum;
- queueing dominates without increasing completed work.

Retry one failed point after a health probe before declaring the boundary. Keep
failed and skipped points with explicit reasons. Never lower safety controls to
force a larger concurrency result.

## Selection

Choose one objective before the run:

- `maximum_throughput`: greatest aggregate output throughput;
- `maximum_goodput`: greatest completed work satisfying every declared limit;
- `maximum_stable_concurrency`: highest repeatedly stable point;
- `interactive`: greatest goodput satisfying interactive latency and
  per-session decode limits.

The selection names the winning concurrency, criterion, supporting point
digests, and rationale. The full campaign's `execution.concurrent_trials` must
equal `selection.concurrency`.

Higher concurrency is not automatically better. Prefer the lower point when
two candidates are within measurement noise unless repeated evidence shows a
material throughput advantage.

## Storage

Store profiles under one private Bucket prefix:

```text
serving-profiles/<profile-id>/
  plan.json
  points/<concurrency>/<repetition>/evidence.json
  points/<concurrency>/<repetition>/harbor-execution/
  profile.json
  checksums.json
  _SELECTED | _FAILED
```

Write `profile.json`, `checksums.json`, and the terminal marker only after the
endpoint is paused and reports zero ready replicas. The profile points and raw
logs are immutable. A retry appends a new repetition or creates a new profile;
it never overwrites prior evidence.

The final campaign evidence records the profile Bucket URI and SHA-256 digest
through `execution.serving_profile`. Manifest validation rejects a mismatched
selection concurrency or serving identity before campaign planning.

## CLI

The production CLI exposes:

```text
harbor-hf profile plan EXPERIMENT --profile-id ID --max-spend-usd USD \
  --estimated-profile-cost-usd USD \
  --timeout-seconds 3600 --output plan.json
harbor-hf profile preflight plan.json
harbor-hf profile run plan.json
harbor-hf profile select profile.json --output selected-profile.json
```

`plan` resolves exact identities and creates the candidate ladder without
remote work. The plan embeds the immutable experiment, so the remote worker
does not depend on mutable local state. `preflight` verifies the model revision,
private Bucket, provider route or endpoint compute, current accelerator quota,
hourly price, worst-case profile cost, and declared spend cap. Unknown endpoint
quota fails closed. Provider profiles require an explicit estimate for the full
profile through `--estimated-profile-cost-usd`; this is distinct from the
deployment's campaign-wave estimate. Preflight rejects it when it exceeds
either the provider or profile spend cap. Endpoint profiles omit this option.

`run` submits one Hugging Face Job. For an Inference Endpoint, the worker
requires a paused baseline, starts the cleanup watchdog before resume, keeps
one endpoint lease across the whole ladder, and pauses and verifies zero ready
replicas on every exit path. It first verifies ordinary chat, the reasoning
channel when required, and a forced tool call. It then records content-free
request observations for the endpoint or Inference Provider, tests ascending
powers of two by running the sampled benchmark tasks through Harbor and the
declared agent, and repeats the last two viable boundary points until each has
three successful measurements. Failed health-check attempts remain in the raw
evidence but do not replace those measurements. Provider points use the same
distinct task set at every concurrency so workload composition cannot affect
selection. Cleanup uncertainty leaves the profile nonterminal for operator or
watchdog recovery. The operator machine never loads model weights or performs
inference.

`select` recomputes every point digest before choosing the winner. A campaign
can bind the resulting profile under `execution.serving_profile`; validation
then requires exact model, deployment, agent, benchmark, context, output, and
concurrency agreement. The binding is propagated into run locks and campaign
digests.

Do not split `profile run` into independent Jobs or campaigns per point. The
profiler reuses one safely leased endpoint across the ladder and retains the
watchdog and verified-pause guarantees.

## Not Covered

This format does not define benchmark tasks, model quality, verifier scoring,
autoscaling across multiple replicas, or provider pricing. It selects one
serving configuration for one declared workload. The full benchmark remains
the authority for task quality.
