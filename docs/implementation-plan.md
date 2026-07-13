# Implementation Plan

## Goal

Deliver a remote-only, resumable system that can run many Harbor benchmarks
across models and hardware, preserve complete evidence, and publish comparable
results without leaving paid inference resources active when unused.

## Phase 1: Specification And Planning

- Stabilize experiment, resolved-run, trial, execution, event, and artifact
  models.
- Model endpoint deployments and provider-routed inference as distinct serving
  target types; keep endpoint engine selection independent of target type.
- Add explicit matrix inclusion and exclusion rules.
- Resolve Harbor datasets and calculate task-set digests without execution.
- Generate immutable run IDs and lock files.
- Add JSON Schema export and compatibility tests.

Exit criteria: the same manifest produces the same resolved plan and digest on
two clean machines when its remote inputs have not changed.

## Phase 2: Hugging Face Resource Lifecycle

- Implement authenticated clients for HF Jobs, Storage Buckets, and Inference
  Endpoints.
- Treat Inference Endpoints as the primary serving path and support independent
  vLLM and llama.cpp engine profiles.
- Add endpoint create, resume, readiness, pause, and inspection operations.
- Capture requested and effective endpoint configuration with redaction.
- Implement endpoint leases and an independent stale-resource watchdog.
- Add an Inference Providers adapter for models that are too large or expensive
  for dedicated endpoints, without assuming knowledge of hidden serving
  internals.
- Add dry-run cost and resource summaries.

Exit criteria: mocked failure tests cover every lifecycle transition, and a
remote smoke test for each endpoint engine always leaves its endpoint paused.
The provider fallback passes a separate tool-use smoke test without creating an
endpoint.

## Phase 3: Harbor Execution

- Translate resolved runs to Harbor jobs using public Harbor APIs.
- Adopt the HF Sandbox environment after its Harbor integration is available.
- Register trial lifecycle hooks for incremental uploads.
- Preserve Harbor locks, results, trajectories, verifier artifacts, and agent
  sessions.
- Separate logical attempts from physical infrastructure retries.

Exit criteria: a bounded public benchmark shard can be interrupted and resumed
without rerunning valid completed trials.

## Phase 4: Artifact And Dataset Publication

- Implement the immutable bucket prefix and per-trial compressed archives.
- Add checksums, redaction checks, and `_SUCCESS` publication semantics.
- Define Parquet schemas for runs, trials, executions, metrics, and artifacts.
- Serialize Dataset commits through a publisher Job.
- Publish a small cross-benchmark run index.

Exit criteria: every published row traces back to a checksummed raw artifact and
failed or partial runs cannot appear as complete.

## Phase 5: Scale And Operations

- Split experiments into bounded shards below remote timeout limits.
- Reconcile queued, active, failed, and completed shards idempotently.
- Add global, per-endpoint, and per-provider concurrency budgets.
- Add spend limits, cancellation, backoff, and provider quota handling.
- Measure controller recovery and endpoint idle time.

Exit criteria: a multi-model, multi-hardware experiment survives controller
termination and resumes without duplicate billing for completed work.

## Phase 6: Presentation And Upstreaming

- Feed a leaderboard Space from normalized Dataset tables.
- Add run, task, attempt, error, throughput, hardware, and cost views.
- Document the workflow in Harbor Cookbook.
- Submit only generic Harbor extension points upstream.
- Keep package boundaries compatible with a future Harbor monorepo import.

Exit criteria: an external reader can reproduce a published configuration and
distinguish complete, partial, composite, and manually submitted results.

## Initial Non-Goals

- Supporting benchmark harnesses other than Harbor.
- Running inference or task containers locally.
- Reusing endpoints across unrelated experiments.
- Building a transactional database before object-backed reconciliation proves
  insufficient.
- Treating the leaderboard Space as an execution service.
