# Trial evidence bundle implementation plan

## Status

Proposed. This plan implements the [trial evidence bundle
specification](trial-evidence-bundle.md) for new Harbor-HF executions.

The work changes evidence capture, judge transport, execution validation,
private artifact finalization, and recovery through publication. It does not change
benchmark rewards or task rubrics.

## Required outcome

A new scored physical execution must preserve all of the following before its
terminal marker is written:

- the complete post-agent `/app` workspace;
- every retained agent session and trajectory;
- the exact HTTP request and response bodies for each verifier judge call;
- the verifier scorecard and reward together with standard output and standard
  error;
- a strict manifest that binds those files to the locked execution and task;
- root checksums that cover the manifest and every referenced file.

The verifier must grade the workspace copy that Harbor froze after the agent
stopped. Harbor-HF must never reconstruct output files from agent tool calls.

A missing evidence component causes a physical execution failure. It never
turns a verifier reward into a model failure. A scored task can publish only
from a complete bundle.

## Current behavior

The current worker already preserves much of the surrounding execution state:

- `FilesystemHarborExecutionAdapter` writes immutable `harbor-job.json` and
  `harbor-request.json` inputs.
- Harbor downloads declared task artifacts and retains verifier logs under each
  trial directory.
- OpenClaw integrations preserve session and trajectory JSONL files.
- `ProviderEvidenceProxy` records content-free agent inference metadata for
  provider-backed runs.
- `private_artifacts.py` classifies retained files and validates their bounds
  and hashes.
- `evidence.py` redacts known secret values, creates deterministic archives,
  and writes root checksum manifests.
- campaign workers publish each physical execution under a unique Bucket path
  before closing the logical trial.
- result publication checks artifact paths and sizes against checksums.

Four gaps prevent a complete scoring audit:

- the generated Harbor job does not require `/app` as a post-agent artifact;
- the final workspace is not stored as one independently verifiable snapshot;
- judge requests and responses are reduced to scorecards instead of retained;
- the private artifact requirement model requires an OpenClaw session but does
  not require workspace and judge evidence together with verifier records for
  scored executions.

The implementation should extend these existing boundaries. It should not add
a second execution engine or parse agent sessions to infer workspace state, and
it must not patch Harbor internals.

## Target flow

```text
locked experiment
      |
      v
Harbor JobConfig declares /app as a trial artifact
      |
      v
agent runs in HF Sandbox
      |
      v
Harbor stops agent descendants and freezes /app
      |
      +------------------------------+
      |                              |
      v                              v
frozen workspace becomes       native sessions and
verifier input                 trajectories retained
      |
      v
verifier calls scoped judge recorder
      |
      +--> exact request/response files
      |
      v
Harbor writes reward and scorecard with stdout and stderr
      |
      v
Harbor-HF builds workspace archive and trial manifest
      |
      v
secret scan -> deep validation -> checksums -> terminal marker
      |
      v
private HF Bucket -> publication verification -> result tables
```

## Component ownership

The implementation keeps existing ownership boundaries.

### Harbor

Harbor remains responsible for:

- agent and verifier lifecycle;
- stopping agent processes before artifact collection;
- collecting `/app` through the public artifact API;
- making the frozen artifact copy available to the verifier;
- retaining native session and trajectory files alongside verifier and lock
  files plus result files;
- reporting typed trial timing and exceptions together with rewards.

Harbor-HF must use `JobConfig.artifacts` and other public models. If the pinned
Harbor release cannot preserve a required workspace property, the change must
be proposed upstream and pinned by commit before Harbor-HF depends on it.

### Harbor-HF

Harbor-HF is responsible for:

- adding the locked workspace artifact declaration to generated Harbor jobs;
- enforcing workspace and judge evidence limits;
- running the temporary judge recorder;
- packaging the frozen workspace;
- assembling and validating trial evidence manifests;
- preventing secret-bearing exact evidence from reaching durable storage;
- classifying evidence failures and scheduling retries;
- publishing and verifying the private bundle;
- exposing local verification and restore commands.

### Benchmark repository

A benchmark remains responsible for:

- declaring which selected tasks use the external judge;
- writing ordinary Harbor verifier outputs;
- retaining the recorder exchange ID used for its scorecard;
- avoiding rubric behavior in Harbor-HF package code.

ShellBench should update its shared verifier helper once. Harbor-HF must not
carry task-specific scorecard parsing or a list of ShellBench task names.

## Design decisions

### One workspace authority

The frozen Harbor artifact is the only post-agent workspace authority. Session
logs remain conversation evidence and are never a file reconstruction source.

The workspace is captured before verification. Packaging can happen after
Harbor exits because it consumes Harbor's frozen host-side artifact copy.

### Full `/app` capture

Every new remote benchmark execution captures `/app`. The experiment manifest
states the root explicitly, and validation accepts only `/app`.

The first implementation does not offer output-only or disabled modes. A final
or component result cannot weaken the contract. Diagnostic campaigns use the
same evidence path so behavior does not diverge by publication role.

### Immutable judge route

The verifier receives a scoped recorder URL as `AGENT_JUDGE_API_URL`. Direct
judge access is not a fallback. A recorder failure causes an evidence or
infrastructure retry.

The existing content-free agent recorder policy remains unchanged. Judge
routes use a separate full-content policy and separate route namespace even
when both roles share one temporary HTTP server.

### Exact evidence stays exact

Workspace archives and judge bodies are not redacted. They are scanned for the
actual secret values injected into the execution. A match destroys partial
exact evidence and blocks publication.

Ordinary logs may continue to use bounded redaction. The finalizer must not run
the current `scrub_secret` rewrite across accepted workspace archives or judge
body files.

### Evidence failure is distinct

Add `evidence` to the physical retry categories. This category covers capture
and packaging plus recorder and manifest completeness failures.

A configuration limit that the existing workspace necessarily exceeds is
`configuration`, because repeating the same execution cannot fix it. A
transient disk, transfer, or recorder failure is `evidence` and can be retried.

### One active schema

The experiment stays `harbor-hf/v1alpha1`, and the active run-lock identifier
stays unchanged. New writers add the required evidence policy in place.
Generated manifests and locks change together with schemas and fixtures plus
the corresponding docs. New execution code has no path that omits the policy.

The trial evidence bundle is a new resource and starts at
`harbor-hf/trial-evidence/v1`.

## Manifest and lock changes

### Experiment models

Add strict models in `src/harbor_hf/models.py`:

```python
class TrialEvidencePolicy(StrictModel):
    workspace_root: Literal["/app"]
    workspace_max_nodes: int
    workspace_max_file_bytes: int
    workspace_max_total_bytes: int
    workspace_max_archive_bytes: int
    workspace_capture_timeout_seconds: int
    judge_max_request_bytes: int
    judge_max_response_bytes: int
    judge_max_calls_per_execution: int


class ArtifactStoreSpec(StrictModel):
    bucket: str
    trial_evidence: TrialEvidencePolicy
```

Use positive integer bounds and explicit maximum platform ceilings. Keep byte
fields in bytes and time fields in seconds. Do not accept human-size strings in
the stored schema.

Extend `BenchmarkJudgeSpec` with:

```python
task_names: list[TaskName]
exclude_task_names: list[TaskName] = []
```

These fields use the existing task selector rules. Planning resolves them to an
exact sorted set after benchmark selection. Validation must reject unmatched
patterns, duplicates, or patterns outside the selected task set.

Required tests in `tests/test_models.py` and `tests/test_io.py` include:

- minimal valid policy;
- each missing required field;
- zero and negative numbers, non-integers, or values above the ceiling;
- any workspace root other than `/app`;
- unknown policy fields;
- judge selectors that match no selected task;
- exclusions outside the included judge set;
- equivalent YAML round trips;
- experiment digest changes after any policy value changes.

Update `examples/shellbench.yaml`, every test fixture, generated schema output,
`docs/run-spec.md`, and README examples that show `artifacts` or a benchmark
judge.

### Resolved identities

Copy the evidence policy into `RunLock` and campaign run locks. Store the exact
resolved judge-required task names, rather than selector patterns, alongside
the locked judge configuration.

The following identities must include canonical evidence policy content:

- experiment digest;
- run ID;
- campaign plan digest;
- run lock digest;
- serving-profile benchmark identity;
- profile plan identity when profiling executes judged tasks.

Add consistency checks that reconstruct the selected matrix cell and compare
the evidence policy byte for byte with the lock. A lock with omitted or changed
evidence values must fail before endpoint resume or provider request admission.

Update:

- `src/harbor_hf/runs.py`;
- `src/harbor_hf/campaigns.py`;
- `src/harbor_hf/profile_submission.py`;
- `src/harbor_hf/profile_worker.py` and profile planning code where benchmark
  identity is stored;
- campaign lock and plan JSON Schemas;
- digest and binding tests.

Do not infer defaults while reading a locked execution. Defaults belong in
manifest construction tools. The resolved lock always carries complete values.

## Harbor workspace collection

### Job configuration

Update `build_execution_request` in
`src/harbor_hf/harbor_adapter/adapter.py` to add this public Harbor job field:

```json
{
  "artifacts": [
    {
      "source": "/app",
      "destination": "workspace/app"
    }
  ]
}
```

The final shape must be accepted by the exact pinned Harbor `JobConfig` model.
Use Harbor's current public `ArtifactConfig` spelling when it differs from the
illustrative JSON.

`destination` keeps the downloaded copy under the native trial artifact tree
without relying on source-derived host paths. The effective artifact set still
includes Harbor's conventional `/logs/artifacts` directory.

Add this declaration for every remote execution. Do not let a benchmark task
or agent parameter remove, replace, or narrow it. Harbor's artifact overlap
validation must reject a task declaration that collides with `workspace/app`.

`HarborExecutionRequest` and its digest cover the artifact declaration. Existing
input immutability checks then detect any modification before or after Harbor
runs.

### Capture ordering contract

Confirm the pinned Harbor lifecycle with tests and a source-level compatibility
check:

1. agent execution returns;
2. Harbor stops agent-owned processes;
3. Harbor downloads trial artifacts;
4. Harbor makes those artifacts available to a separate verifier or starts the
   verifier after collection;
5. verifier execution begins;
6. Harbor downloads verifier logs and writes the result.

Document the tested Harbor commit in the implementation report. If Harbor does
not stop descendants before collection, add that behavior upstream. Do not add
process-killing shell commands around Harbor from Harbor-HF.

### Frozen verifier input

The verifier must read the captured workspace, not a different copy of live
agent state. Prefer Harbor's separate-verifier environment support when it is
available for HF Sandbox tasks.

If the initial Harbor API runs the verifier in the same Sandbox, require these
properties before release:

- capture completes before verifier start;
- no agent process remains alive;
- the raw artifact copy is outside the verifier-writable `/app` tree;
- later verifier changes cannot change the captured host copy;
- Harbor's artifact manifest records successful `/app` collection.

The implementation plan should add a Harbor integration fixture where the
agent leaves one value in `/app/output/value.txt` and the verifier changes the
live file. The archived workspace must retain the pre-verifier value.

### Collection failure mapping

Inspect Harbor's native artifact manifest after every trial. The `/app` entry
must have `status: ok` and point to a directory. Missing, skipped, collided, or
failed entries become evidence failures before the score is accepted.

Do not accept a valid scorecard when workspace collection failed.

## Workspace package implementation

### New module

Create `src/harbor_hf/trial_evidence.py`. Keep it independent from the HF SDK and Harbor internals as well as clocks and
process execution.

The module should contain strict Pydantic models and pure filesystem functions:

- `TrialEvidenceManifest`;
- `WorkspaceEvidence`;
- `AgentEvidence`;
- `JudgeEvidence`;
- `VerifierEvidence`;
- `CompletionEvidence`;
- `EvidenceFileRef`;
- workspace index row models;
- judge exchange reference models;
- manifest assembly;
- structural validation;
- digest validation;
- deep validation;
- safe workspace restoration.

Inject clocks and limit policies into functions that need them. Do not read
environment variables in this module.

### Snapshot input discovery

The assembler locates the one Harbor trial represented by the physical
execution through the validated compatibility bundle. It then locates:

```text
<trial>/artifacts/workspace/app
```

Before traversal, require:

- the trial directory is a real directory;
- the artifact manifest names the `/app` source and expected destination;
- the destination is not a symlink;
- exactly one workspace destination exists;
- execution and trial identity plus task name and digest match the locks.

Do not find a workspace by taking the first directory named `app`.

### Stable traversal

Walk the workspace with descriptor-relative operations where Python and the
platform allow them. Never follow symlinks.

For each regular file:

1. read metadata without following links;
2. reject a file over the locked per-file limit;
3. stream the file through SHA-256 and the secret matcher;
4. record byte count from bytes actually read;
5. read metadata again;
6. fail when identity, type, size, or modification state changed during read.

The agent should already be stopped, so a change indicates a lifecycle bug or a
stray process. It must not be hidden by retrying the read inside one execution.

Track node and aggregate byte counts during traversal. Stop at the first locked
limit breach. Remove all temporary outputs before returning failure.

### Path handling

Use POSIX relative paths rooted at the captured `app` directory. Reject:

- names that cannot be represented as valid UTF-8;
- names outside Unicode NFC;
- empty components;
- NUL and backslash;
- path traversal components;
- duplicate normalized names;
- absolute or escaping symlink targets;
- symlink cycles and dangling targets;
- sockets and devices as well as FIFOs or mount boundaries.

Record directories, regular files, and safe symlinks. Materialize hard-linked
files as independent regular files.

Write JSONL rows in canonical compact JSON with sorted keys and a final newline.
The root row is first. Other rows are sorted by UTF-8 path bytes.

### Deterministic archive writer

Add `zstandard` as a locked runtime dependency. Write a pax tar stream through
one Zstandard compression thread at level 6.

Normalize outer tar metadata exactly as the specification requires. Preserve
workspace permission bits. Use the file index as the archive member plan rather
than traversing the filesystem a second time without checks.

Use temporary files beside the final destination:

```text
.workspace-files.jsonl.tmp-<random>
.workspace.tar.zst.tmp-<random>
```

After writing:

- flush and fsync both files;
- reopen and deep-validate the archive against the index;
- compute final sizes and SHA-256 digests;
- atomically rename the index and archive;
- fsync the parent directory where supported.

An exception removes every temporary and final file created by that attempt.

### Workspace manifest checks

Deep validation must prove:

- archive and index outer digests match references;
- member count equals index count;
- member order and paths match;
- each member type and mode match;
- file bytes match size and SHA-256;
- symlink targets match and remain inside the root;
- no extra archive member exists;
- aggregate counts match `WorkspaceEvidence`.

Add property-based tests or generated case matrices for path and member
validation. Keep archive fixtures small and build them during tests rather than
checking large binaries into Git.

### Raw workspace retention

After the package validates, remove the raw `artifacts/workspace/app` copy from
the retained private tree to avoid storing the workspace twice. Keep Harbor's
`artifacts/manifest.json`, which records that collection succeeded.

Refresh the Harbor compatibility artifact inventory after removal and package
creation. It must list the archive, index, evidence manifest, and native
artifact manifest with current digests.

If operators prefer browseable raw files later, add a separate extraction tool.
Do not retain both formats in the first production implementation.

## Judge recorder implementation

### Recorder structure

Refactor `src/harbor_hf/provider_proxy.py` so transport mechanics are shared by
two explicit policies:

- agent provider recording, which remains content-free;
- verifier judge recording, which retains exact bounded bodies.

A clean internal split is:

```text
recording_server.py       HTTP server, scope registry, relay, lifecycle
provider_proxy.py         content-free agent request policy
judge_proxy.py            full-content verifier judge policy
```

Keep public names stable only where current package callers need them during the
same change. New production code must use the shared server and explicit route
roles. Do not add a direct-judge fallback.

### Route model

Extend scope registration to bind:

- random capability;
- execution ID;
- trial ID;
- route role (`agent` or `judge`);
- locked target;
- request-attempt or call limit;
- evidence destination;
- activation and expiry state.

Use distinct URL namespaces:

```text
/scopes/<capability>/v1/chat/completions
/judge-scopes/<capability>/v1/chat/completions
```

A capability registered for one namespace must fail in the other. The raw
capability never reaches an event, lock, artifact, session URL, exception, or
process log. Record only its SHA-256 digest.

### Wave transport lifecycle

Start the hosted recorder when either condition is true:

- the model deployment uses an HF Inference Provider;
- at least one task in the wave requires a benchmark judge.

This changes submission and worker transport conditions for endpoint-backed
waves. Update:

- `src/harbor_hf/submission.py`;
- `src/harbor_hf/profile_submission.py`;
- `src/harbor_hf/wave_worker.py`;
- `src/harbor_hf/profile_worker_transport.py`;
- direct-run transport setup in `src/harbor_hf/worker.py`;
- HF Job `--expose` command tests.

One temporary server can host both policies. Endpoint-backed runs use only the
judge route. Provider-backed judged runs use separate agent and judge
capabilities.

The external readiness check must test the server through authenticated HF Job
ingress before any Harbor execution starts. A readiness failure remains an
infrastructure failure because no benchmark process ran.

### Judge scope lifecycle

Register one judge capability before starting the corresponding Harbor
execution. Write its digest and transport type to an execution-local route
record. Pass its scoped URL separately from the agent inference base URL.

Revoke the scope after Harbor and the compatibility exporter finish, including
all exception paths. Closing a wave revokes every remaining scope before the
server stops.

A revoked scope returns a bounded content-free 404. It does not forward or
create a new judge attempt.

### Harbor process environment

Change `harbor_process_environment` in `src/harbor_hf/runs.py` to accept two
base URLs:

- `inference_base_url` for the agent;
- `judge_api_url` for the verifier.

When a task is judge-required, set:

```text
AGENT_JUDGE_API_URL=<scoped recorder URL>/v1/chat/completions
AGENT_JUDGE_MODEL=<locked judge model>
AGENT_JUDGE_API_KEY=<HF Job ingress credential>
```

The upstream judge API URL remains in the immutable lock and is known only to
the trusted recorder. The verifier does not receive it.

The ingress credential remains an environment value. It must not be serialized
into the Harbor request, OpenClaw config, workspace, judge evidence, or process
command.

For deterministic tasks, omit all three judge variables. This prevents a task
outside the resolved judge-required set from making an accidental paid call.

### Request validation

Before forwarding, the judge policy must:

- require `POST` on the exact chat-completions path;
- require a valid JSON object within the locked request byte limit;
- reject compressed request bodies;
- reject `stream: true`;
- require the locked model or insert it only under the typed
  `model_enforced` transformation;
- reject caller-supplied URLs, headers, credentials, or provider routing
  extensions not allowed by the judge contract;
- enforce the per-execution call limit before upstream I/O;
- scan raw request bytes for every known injected secret;
- allocate the next exchange ID atomically;
- reject the call if its evidence destination is unavailable.

The recorder stores exact received and forwarded body bytes. If it changes the
model field, it records both bodies and `transformation: model_enforced`.

### Response handling

Use `Accept-Encoding: identity` upstream where supported. Bound captured
response bytes. Preserve the exact upstream body before content decoding and
the exact body delivered to the verifier. Decode a bounded copy for secret
scanning and semantic validation when the upstream still applies a supported
content encoding.

Allowlist metadata headers at all four stages: received request, forwarded
request, upstream response, and delivered response. Follow the exact bundle
specification. Drop
request and response authorization, cookies, set-cookie, tracing baggage, and
unrecognized headers. Record upstream and delivered HTTP status separately.

Scan the full bounded response for known injected secrets before it is written
or delivered. If a match occurs, close the upstream response, return a local
content-free error, delete body temporaries, and mark the execution for
operator review.

Record provider errors, rate limits, timeouts, client disconnects, malformed
responses, and recorder errors as complete exchange attempts whenever the
required body files can be written safely.

The existing provider recorder continues recording content-free semantic usage
and latency. Judge exchange records may include timing and request IDs but do
not feed model throughput metrics.

### Atomic exchange writes

Write body files and `exchange.json` under an execution-local temporary
directory. `exchange.json` is written last. Rename the complete directory to
`judge/judge-NNNN` while holding the scope lock.

A process crash can leave a temporary directory. Finalization removes it and
marks judge capture incomplete. It never renames a partial attempt into place.

### Selection record

Return `X-Harbor-Judge-Exchange-ID` to the verifier. Update the shared
ShellBench verifier helper to write `judge-selection.json` beside its
scorecard.

Define and publish the tiny selection schema in Harbor-HF so other benchmarks
can conform. The benchmark helper remains responsible for stating which
exchange response produced its scorecard.

For one-call verifiers, Harbor-HF may validate that exactly one complete
exchange exists, but it must still require the selection file. Implicitly
selecting the only call would create a second contract and hide missing helper
behavior.

Add integration fixtures for one-call and retry behavior, multi-call and
deterministic verifiers, plus missing-selection verifiers.

## Trial evidence assembly

### Assembly location

Run assembly after Harbor exits and the compatibility bundle validates, but
before `_finalize_execution` redacts logs, writes private artifact manifests,
creates `artifacts.tar.gz`, or writes root checksums.

For campaign execution, call the assembler in `_execute_trial` after
`verification.json` and before `build_private_artifact_manifest`.

Apply the same ordering to:

- direct single runs in `worker.py`;
- profile trial executions that produce benchmark scores;
- recovered physical executions before adoption;
- any correction or diagnostic path that publishes a scored Harbor trial.

Use one application service function for all paths. Do not duplicate bundle
requirements in each worker.

### Input model

The recorder writes completed judge attempt directories under
`<execution-root>/judge-records/` because the native Harbor trial directory is
not known when the scope is registered. After the compatibility adapter
identifies the one native trial, the assembler moves those completed directories
into `<trial>/evidence/judge/` with an atomic rename on the same Job-local
filesystem. It rejects duplicate exchange IDs, temporary attempt directories,
and records for another execution. The source directory must be empty and
removed before final checksums are written.

The assembler receives typed inputs:

```python
assemble_trial_evidence(
    *,
    execution: ExecutionLock,
    run: RunLock,
    harbor_trial: HarborTrialRecord,
    trial_dir: Path,
    judge_records_dir: Path | None,
    policy: TrialEvidencePolicy,
    clock: Clock,
) -> TrialEvidenceManifest
```

`HarborTrialRecord` should come from the validated compatibility adapter rather
than raw path guesses. Extend the adapter model with the native trial path and
required artifact references if necessary.

The assembler must not import HF SDK clients or inspect the Bucket.

### Requirement derivation

Derive requirements from locked execution facts:

- `/app` workspace is required whenever an agent environment was created;
- session and trajectory are required when OpenClaw execution started;
- verifier logs are required when verifier execution started;
- judge exchange and selection are required only for exact judge-required
  tasks;
- scorecard and reward are required for a scored Harbor trial;
- verifier stdout and stderr files are required, including empty files.

Do not derive judge expectation by looking for `judge_model` in a scorecard.
The run lock is authoritative.

### Verifier log normalization

Harbor currently may omit an empty standard-error file. Add a normalization
step that creates explicit empty `test-stdout.txt` and `test-stderr.txt` files
only when Harbor reports that the verifier ran and the corresponding stream was
empty. Do not fabricate files when verifier execution never started.

Preserve task-specific verifier files through a sorted `logs` list. Scorecard
and reward fields point to their canonical native files.

### Manifest writing

Assemble references only after all component files have final names and exact
digests. Validate the complete Pydantic model, serialize canonical indented
JSON with sorted keys, write through a temporary file, and rename it last.

Immediately run structural and digest validation plus deep validation against
the assembled bundle. A worker-created manifest that its own validator rejects is an evidence
failure.

### Compatibility export refresh

Packaging adds files and removes the raw workspace directory after Harbor's
compatibility exporter ran. Call `refresh_retained_bundle` after assembly and
before the final private artifact manifest.

Extend artifact kinds with:

- `trial_evidence`;
- `workspace_archive`;
- `workspace_index`;
- `judge_request`;
- `judge_response`;
- `judge_exchange`.

Update classification in `harbor_adapter/exporter.py`, private retention
priority, result artifact metadata, presentation tests, and schema fixtures.

## Secret containment

### Known secret set

Build the scan set from values already available to the remote worker:

- `HF_TOKEN`;
- agent inference credentials;
- judge upstream credentials;
- private benchmark source credentials;
- recorder route capabilities;
- endpoint or provider credentials injected by the selected deployment;
- any additional locked secret names present in the worker environment.

Do not serialize this set. Pass it through process memory and temporary
mode-0600 files only where a subprocess needs bounded scanning.

Ignore empty values. Reject unexpectedly short configured secret values at
submission where possible, because they cannot be scanned without excessive
false positives.

### Finalization split

Replace the single `_redact_unit` behavior for trial evidence with two paths:

- mutable ordinary logs are scrubbed and then scanned;
- exact evidence is scanned and either retained byte for byte or deleted as a
  failed component.

Implement path-based classification from the trial evidence manifest. Do not
rely only on filename suffixes.

Before writing a terminal marker, run `assert_secret_absent` over the complete
execution root. The function must remain streaming and bounded. It should
report only the containing evidence category and a sanitized relative path,
never the matching value.

### Secret failure cleanup

When a known secret is detected in exact evidence:

1. revoke agent and judge capabilities;
2. delete workspace and judge temporary files;
3. delete completed exact files for that rejected component;
4. retain only a content-free failure record and event;
5. mark the execution `evidence` with operator review required;
6. do not copy the execution to the ordinary terminal success path;
7. ensure the Bucket receives no secret-bearing bytes if failure evidence is
   published.

The system cannot rotate a user's HF or provider token itself. Operator output
must name the secret environment variable, not its value, and instruct the
operator to rotate it before retrying.

### Bucket staging rule

Keep all unsanitized and unvalidated evidence on Job-local storage. Publish to
the Bucket only after secret scanning, deep validation, private artifact
manifest creation, and root checksum creation.

The current unique-prefix and terminal-marker behavior already supports this
rule. Add tests proving no Bucket adapter call receives a source tree after a
secret match.

## Private artifact requirements

### Requirement model

Replace the single-purpose `PrivateArtifactRequirement` shape with a strict
name enum that covers:

- `openclaw_session_jsonl`;
- `openclaw_trajectory_jsonl`;
- `trial_evidence_manifest`;
- `workspace_archive`;
- `workspace_file_index`;
- `judge_exchange`;
- `verifier_scorecard`;
- `verifier_reward`;
- `verifier_stdout`;
- `verifier_stderr`;
- `judge_selection`.

Each requirement records `required`, `satisfied`, and sorted paths. Judge
requirements are per locked task. Workspace requirements are per execution
lifecycle state.

`build_private_artifact_manifest` accepts a typed evidence expectation instead
of inferring all requirements from filenames. Keep inference only for facts
owned by validated Harbor results, such as whether OpenClaw execution started.

### Size limits

Workspace archive limits are separate from ordinary private artifact limits.
A valid 1 GiB workspace archive must not be discarded by the current 64 MiB
per-file sanitizer.

Extend private artifact policy so typed workspace archive and index files use
the locked workspace ceilings. Judge body files use locked judge ceilings.
Ordinary logs keep their existing bounds.

Aggregate execution limits account for all files but must be high enough to
hold one maximum workspace plus sessions, judge records, and verifier logs.
Calculate this ceiling from the locked component limits with checked integer
arithmetic. Reject overflow and plans above platform storage policy before
execution.

A required file can never be evicted by retention priority. If the bundle is
too large, the execution fails. Priority-based removal remains available only
for optional failed-run diagnostics.

### Symlink policy

The outer retained evidence tree must still contain no symlinks. Safe workspace
symlinks exist only as entries inside `workspace.tar.zst` and rows in the file
index.

Package the workspace before calling outer-tree symlink sanitization. Remove the
raw collected workspace tree afterward. Any other symlink under the execution
root remains a rejection.

## Failure and retry integration

### Retry category

Add `evidence` to `RetryCategory` in `src/harbor_hf/control.py` and every strict
schema or `TypeAdapter` that validates it.

Map failures as follows:

- transient archive I/O, inconsistent capture, missing recorder output, partial
  exchange, missing selection, and manifest write failure become `evidence`;
- a locked size or node ceiling breach becomes `configuration`;
- an unavailable hosted recorder before Harbor starts remains `transient` or
  infrastructure under the existing startup policy;
- known secret detection becomes `evidence` plus manual intervention;
- a judge provider HTTP error with a complete recorded exchange remains under
  verifier or benchmark semantics;
- an agent exception remains `agent` only when evidence capture itself is
  complete.

### Retry behavior

An evidence retry creates a new execution ID. It never reruns only the verifier
against an untracked workspace and never overwrites the failed bundle.

The next execution repeats the complete task because the original agent state
may no longer exist. This keeps task and judge evidence bound to one physical
execution.

Use existing campaign retry limits. Add typed event reasons for `evidence-incomplete`, `secret-detected`, and
`evidence-validated` executions.

If the current event enum should remain smaller, represent these as typed
payload reasons on `execution.failed` and document the choice. Do not encode
reason details into free-form exception strings.

### Logical outcomes

A logical task can become `scored` only from a complete execution bundle.

After evidence retries exhaust, use `infrastructure_exhausted` under the
existing fixed denominator. The result row records no verifier reward source
from the rejected execution and cannot imply that the model answered
incorrectly.

Agent and benchmark failures may still produce task-level zero outcomes when
their retained failure evidence meets the applicable requirements. Missing
workspace or session evidence changes the failure to evidence or infrastructure
rather than silently preserving the semantic category.

### Recovery adoption

Extend terminal execution recovery checks to validate:

- root checksums;
- private artifact manifest;
- trial evidence schema;
- all referenced digests and sizes;
- workspace archive deep consistency;
- judge selection and exchange identity;
- execution and trial identity plus task and attempt identity;
- required component status.

Recovery must not adopt an old scored execution without the new bundle. New
campaigns use one evidence contract. Historical completed campaigns remain
immutable and are handled only by historical audit tools.

## Publication changes

### Publication gate

Extend `verify_campaign_artifacts` and result publication so every selected
scored execution requires a complete trial evidence manifest.

Publication checks:

- the manifest path is present in `private-artifacts.json`;
- the manifest digest matches root checksums;
- every file reference is covered by root checksums;
- task and execution identity match normalized rows;
- completion is `complete`;
- judge-capable tasks have a closed recorder summary;
- every observed judge call has a complete exchange and every scored judge call is selected;
- a zero-call deterministic verifier branch has no judge selection;
- deterministic tasks declare no judge exchange requirement;
- raw evidence remains in a private Bucket;
- public artifact rows contain metadata only.

A publication that fails these checks is rejected. The publisher does not
repair, infer, or downgrade missing evidence.

### Result fields

Add compact evidence fields to private execution summaries and normalized
execution rows:

| Field | Meaning |
| --- | --- |
| `evidence_status` | `complete`, `incomplete`, or `not_available` for historical rows. |
| `evidence_manifest_sha256` | Digest of `trial-evidence.json`. |
| `workspace_archive_sha256` | Digest of the workspace archive when captured. |
| `judge_exchange_count` | Number of complete judge exchange attempts. |
| `selected_judge_exchange_count` | Number used by the verifier scorecard. |

For the new write path, `not_available` is invalid. If public table schema policy
forbids adding these fields immediately, retain them in private execution
summaries first and expose them in the next coordinated table update. The
publication gate must still be active before any new score is cataloged.

Raw workspace, session, prompt, response, and scorecard contents never enter
public Parquet tables or the Results Space API.

### Artifact metadata

Publish private artifact metadata rows for the manifest, workspace archive,
file index, judge records, scorecard, and reward. Keep the existing
`classification: private` contract.

The Results Space may show that evidence exists, its size, and its digest. The
artifact content endpoint remains disabled for private evidence.

### Retention and deletion

Keep trial evidence for as long as any active catalog entry or publication
references it. Result withdrawal does not delete private evidence.

Do not add per-file deletion for workspace, judge, or verifier records. A later
retention command may delete only a complete execution or run prefix after it
proves:

- every referencing publication is withdrawn;
- no campaign recovery or correction still references the prefix;
- the operator named the exact immutable run or execution ID;
- the Bucket path matches the locked artifact destination;
- a content-free deletion record can be committed before removal.

The first implementation should omit automated expiry. Measure storage growth
from complete campaigns and define an operator retention policy before adding
lifecycle deletion. This avoids silently breaking old score audits to reduce
Bucket usage.

Add tests that active publications block deletion planning and that partial
component deletion is never proposed.

## CLI and operator tools

### Local validation

Add an `evidence` Typer group:

```bash
harbor-hf evidence verify EXECUTION_DIR
harbor-hf evidence verify EXECUTION_DIR --deep
harbor-hf evidence show EXECUTION_DIR
harbor-hf evidence restore EXECUTION_DIR --destination PATH
```

Behavior:

- `verify` performs structural and digest checks;
- `--deep` also streams and validates the workspace archive and JSONL records;
- `--format` selects human, JSON, or quiet output;
- `show` prints a content-free summary with identities and statuses plus sizes
  and digests;
- `restore` requires successful deep validation and an empty destination.

Commands use local files only. Bucket download remains an explicit `hf cp` or
`hf sync` operation so verification has no hidden network side effect.

Support `--format human|json|quiet` consistently with existing CLI commands.
JSON output uses a strict result schema and never includes private file content.

### Campaign artifact verification

Extend:

```bash
harbor-hf artifacts verify CAMPAIGN_ID --namespace NAMESPACE
```

with evidence counts and failure paths. The command should report:

- planned physical executions inspected;
- scored executions with complete bundles;
- non-scored executions and their applicable evidence status;
- workspace bytes and archive bytes;
- judge exchange counts;
- missing or invalid requirement names;
- first failing private path without showing content.

Add `--deep-trial-evidence` for an operator-requested remote deep pass if the
normal publication verifier uses digest validation only. Document expected
cost and streaming behavior.

### Operator errors

Errors should identify:

- campaign and run IDs plus trial and execution IDs;
- failed evidence component;
- typed reason code;
- whether a retry is safe;
- whether secret rotation is required.

They must never print request bodies, response bodies, workspace contents,
route capabilities, or secret values.

## Schema files

Add checked-in JSON Schemas:

```text
schemas/trial-evidence-v1.schema.json
schemas/judge-exchange-v1.schema.json
schemas/judge-selection-v1.schema.json
```

Generate them from strict Pydantic models, then review the generated output for:

- `additionalProperties: false` at every object;
- exact required fields;
- digest, identifier, and path patterns;
- discriminated component status shapes;
- integer minimums and maximums;
- no accidental nullable fields;
- no default that lets a required evidence field disappear.

Add schema generation and drift checks to the existing quality gate. A manual
schema edit without matching Pydantic code must fail CI.

Keep one realistic fixture under `tests/fixtures/trial-evidence/complete/` with
small files. Build archive bytes deterministically during fixture setup or
verify a checked-in archive digest explicitly.

## Test plan

### Workspace unit tests

Cover at least these cases:

- empty `/app` with only the root directory;
- one empty regular file;
- nested directories and executable files;
- binary content containing every byte value;
- UTF-8 NFC names;
- non-NFC and undecodable names;
- spaces, leading dots, and long valid names;
- absolute, parent, empty-segment, backslash, and NUL paths;
- safe relative symlink;
- absolute, escaping, dangling, and cyclic symlink;
- hard links materialized as files;
- socket, FIFO, device, and mount-boundary rejection;
- one file and aggregate byte limits;
- node and archive byte limits;
- workspace mutation between pre-read and post-read metadata;
- disk-full and permission failures;
- interrupted index and archive writes;
- temporary-file cleanup;
- deterministic archive bytes across repeated builds;
- deep validator rejection for reordered, missing, extra, renamed, or modified
  tar members;
- restore refusal for a nonempty destination;
- restore tree digest equality.

Use generated temporary trees. Tests must not require root privileges, network,
or a model.

### Judge recorder unit tests

Cover:

- health and unsupported routes;
- agent and judge route namespace separation;
- capability registration, collision, revocation, and expiry;
- concurrent executions with no evidence mixing;
- exact incoming request byte retention;
- exact forwarded body retention;
- locked-model enforcement;
- call count limits;
- request and response byte limits;
- streaming request rejection;
- upstream 200, 4xx, 5xx, timeout, rate limit, and malformed body;
- supported content decoding;
- client disconnect before and after upstream response;
- recorder shutdown during a call;
- atomic attempt directory creation;
- temporary attempt cleanup;
- response header allowlist;
- exchange ID response header;
- no authorization, cookie, capability, or token in evidence;
- known secret in request, response, path, and ordinary header;
- exact response delivered to verifier;
- provider agent ledger remains content-free after refactoring.

Retain the current mutation tests around retry budgets and provider evidence.
Add mutations for every new completion and secret boundary.

### Manifest unit tests

Cover every component status and invalid field combination. Include:

- unknown fields;
- unsupported schema versions;
- invalid timestamps and digests;
- unsafe or escaping references;
- missing files and digest mismatches;
- identity mismatch at each level;
- unsorted session and trajectory references plus exchange and log references
  or requirement lists;
- judge expected with no closed recorder summary;
- closed zero-call judge recorder accepted without a judge selection;
- recorder exchange count mismatch;
- deterministic task with judge selection;
- selected unknown or incomplete exchange;
- complete status with an unsatisfied requirement;
- scored verifier output with incomplete workspace;
- multiple judge selections in the wrong order;
- outer root checksums that omit a referenced file.

### Harbor adapter tests

Add tests proving:

- every generated `JobConfig` declares `/app` capture;
- the declaration participates in request digests;
- task and job artifact collisions fail;
- the pinned Harbor parser accepts the config;
- the artifact manifest reports the expected source and destination;
- a verifier sees the frozen copy;
- script-created files absent from session readbacks still appear in the
  workspace archive;
- verifier-created changes do not alter the archive;
- a failed collection prevents score acceptance.

Do not mock Harbor internals. Use public model parsing and the existing adapter
boundary.

### Worker and recovery tests

Inject failures after each durable boundary:

- agent completion before workspace collection;
- workspace collection before verifier start;
- judge request write before upstream call;
- upstream response before response write;
- exchange completion before verifier result;
- verifier result before workspace packaging;
- archive completion before manifest write;
- manifest completion before secret scan;
- secret scan before checksums;
- checksums before terminal marker;
- terminal execution publication before trial summary publication.

For every crash point, assert whether recovery retries, adopts, or requires
manual intervention. No case may publish a score from partial evidence.

### Publication tests

Build mutation cases that remove or alter each required file and reference.
`results publish` and `artifacts verify` must reject all of them.

Confirm that public tables and API responses contain no raw:

- workspace path contents beyond private artifact metadata;
- session messages;
- prompt text;
- judge response text;
- scorecard reasons if current publication policy keeps them private;
- credentials or capabilities.

### Secret tests

Use synthetic test secrets passed through the same runtime interfaces as real
secrets. Verify exact absence from:

- paths;
- workspace index and archive;
- judge request and response files;
- exchange metadata;
- event logs;
- Harbor logs;
- compatibility exports;
- private artifact manifests;
- root checksums and terminal files;
- exception messages;
- CLI output.

Test a secret split across read chunks. Keep the streaming overlap behavior from
`evidence.py` covered.

### Performance tests

Measure capture and validation with generated trees at these points:

| Tree | Purpose |
| --- | --- |
| 10,000 small files, 100 MiB total | Metadata-heavy workload. |
| 100 files, 1 GiB total | Byte-heavy workload. |
| 100 concurrent judge requests across test scopes | Recorder isolation and lock contention. |
| Maximum allowed judge request and response | Memory bound. |

Set release thresholds after measuring CI and HF Job hardware. At minimum:

- memory use remains bounded by one streaming chunk plus one bounded judge
  body;
- archive creation uses one compression thread per active execution;
- aggregate compression concurrency is capped below worker CPU and disk limits;
- recorder locking does not serialize upstream network time;
- no unbounded list of workspace file contents enters memory.

The worker may use a small evidence-packaging pool separate from agent
concurrency. Record the selected pool size in runtime evidence.

## Remote verification

Local tests cannot prove the HF Job and HF Sandbox boundary. Run bounded remote
canaries before enabling the new write contract for full campaigns.

### Endpoint-backed canary

Use one judge-required task and one deterministic task against a disposable or
already managed paused endpoint. The judged task should create one output file
through a script without reading it back.

Verify:

- the endpoint remains governed by the normal watchdog;
- the hosted judge recorder is reachable from the verifier Sandbox;
- the final workspace contains the script-generated bytes;
- the judge request contains the exact verifier prompt;
- the delivered response matches the recorded body;
- scorecard selection points to the exchange;
- both tasks have complete applicable bundles;
- deep restore reproduces workspace digests;
- endpoint cleanup leaves zero ready replicas.

### Provider-backed canary

Use one small provider-backed task with an external judge. Confirm separate
agent and judge capabilities on the same hosted recorder.

Verify:

- agent request evidence remains content-free;
- judge exchange evidence contains exact bounded bodies;
- route capabilities do not appear in either evidence set;
- trial concurrency cannot mix judge files;
- provider attempt limits and judge call limits remain independent;
- the Job exits with no process or endpoint left running.

### Failure canary

Run a controlled task that causes one workspace capture failure and one missing
judge selection. Confirm both create new physical executions and neither
publishes the rejected score.

A separate synthetic-secret canary should use a disposable fake secret, never a
real credential. Confirm the fake value reaches no Bucket object.

### Release record

Record:

- Harbor-HF commit;
- Harbor commit;
- benchmark commit and task digests;
- HF Job IDs;
- endpoint identity and final paused snapshot when applicable;
- trial and execution IDs;
- bundle and root checksum digests;
- canary wall time, workspace size, archive size, and capture time;
- secret scan and deep validation results.

Store this report in `docs/` or the private campaign evidence and link it from
the pull request.

## Delivery sequence

The work should land in coherent slices. Each slice has tests and leaves the
repository internally consistent.

### Contract slice

Deliver:

- experiment evidence policy;
- judge task selectors;
- lock propagation and digest changes;
- trial evidence, judge exchange, and selection Pydantic models;
- generated JSON Schemas;
- docs and small fixtures.

Exit criteria:

- all strict model and schema tests pass;
- existing new-write fixtures are updated and no fallback defaults are
  accepted;
- planning prints complete locked policy;
- no remote behavior changes yet.

### Workspace slice

Deliver:

- Harbor `/app` artifact declaration;
- lifecycle compatibility checks;
- workspace index, archive, validator, and restore library;
- packaging in direct and campaign execution paths;
- workspace private artifact classification and limits.

Exit criteria:

- a fake agent script creates a file that never appears in session output and
  the archive preserves it exactly;
- verifier changes cannot alter the frozen copy;
- failed workspace capture prevents score acceptance;
- archive construction is deterministic;
- local deep verify and restore pass.

### Judge slice

Deliver:

- shared recorder transport;
- full-content judge policy;
- all-wave recorder startup rules;
- scoped judge environment injection;
- atomic exchange records;
- selection schema and benchmark helper integration.

Exit criteria:

- endpoint and provider worker tests pass;
- agent provider evidence remains content-free;
- every judged scorecard selects a complete exchange;
- direct upstream fallback is removed from new execution code.

### Completion slice

Deliver:

- trial evidence assembly;
- complete private artifact requirements;
- exact-evidence secret handling;
- evidence failure category;
- recovery adoption checks;
- incremental execution publication.

Exit criteria:

- every scored test execution has a complete bundle;
- missing any required component creates an evidence retry;
- secret-bearing exact evidence is absent from publication inputs;
- root checksums cover all references.

### Publication slice

Deliver:

- artifact verification and result publication gates;
- normalized evidence metadata;
- CLI verify, show, and restore commands;
- Results Space metadata changes where applicable;
- operator documentation.

Exit criteria:

- mutation fixtures cannot publish after evidence removal or alteration;
- public tables contain metadata only;
- private references retain exact digests;
- local and remote verification agree on bundle identity.

### Remote release slice

Deliver:

- endpoint-backed canary;
- provider-backed canary;
- failure and fake-secret canaries;
- measured limit and concurrency decisions;
- release record.

Exit criteria:

- all canaries satisfy the remote checks;
- managed endpoints are paused with zero ready replicas;
- no HF Job remains running;
- new campaigns use the complete evidence path exclusively.

## Repository file map

Expected Harbor-HF changes include:

```text
src/harbor_hf/
├── cli.py
├── control.py
├── evidence.py
├── judge_proxy.py                  # New
├── models.py
├── private_artifacts.py
├── provider_proxy.py
├── recording_server.py             # New shared transport
├── runs.py
├── campaigns.py
├── trial_evidence.py                # New schema and workspace package
├── worker.py
├── wave_worker.py
├── profile_worker.py
├── profile_worker_transport.py
├── harbor_adapter/
│   ├── adapter.py
│   ├── exporter.py
│   ├── models.py
│   └── validation.py
└── results.py

schemas/
├── trial-evidence-v1.schema.json
├── judge-exchange-v1.schema.json
└── judge-selection-v1.schema.json

docs/
├── trial-evidence-bundle.md
├── trial-evidence-implementation-plan.md
├── run-spec.md
├── architecture.md
├── harbor-integration-contract.md
├── result-publication.md
└── harbor-cookbook.md

tests/
├── fixtures/trial-evidence/
├── test_trial_evidence.py
├── test_judge_proxy.py
├── test_provider_proxy.py
├── test_harbor_adapter.py
├── test_private_artifacts.py
├── test_wave_worker.py
├── test_worker.py
├── test_recovery.py
├── test_results.py
└── test_cli.py
```

ShellBench needs a separate change to its shared verifier helper and generated
tasks so `judge-selection.json` is written. Any Harbor lifecycle change must be
made upstream in Harbor and consumed through a pinned public revision.

## Quality gates

Run the repository's complete required checks after each behavior slice:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest --cov=src/harbor_hf --cov-fail-under=85
(cd apps/results-web && npm ci && npm run build)
docker build -f deploy/space/Dockerfile .
uv run slophammer-py dry .
uv run pip-audit
uv run slophammer-py check . --baseline
uv run python scripts/check_mutation.py --min-kill-rate 90
```

Additional gates for this feature:

- generated evidence schemas match checked-in files;
- deep verification succeeds for every complete fixture;
- secret mutation tests kill every missing scan boundary;
- provider recorder mutation tests prove agent content is still excluded;
- archive mutation tests reject every changed member and index row;
- remote canary evidence is downloaded and independently verified;
- no endpoint or HF Job remains active after remote tests.

Documentation-only changes need the repository's documentation checks. Code
slices must run the full behavior and mutation gates.

## Operational limits and cost

Before enabling full campaigns, measure workspace size on a representative
115-task ShellBench sample. Report percentiles for node count, raw bytes,
archive bytes, capture duration, and compression ratio.

Use those measurements to confirm the proposed 100,000-node, 2 GiB workspace
ceilings and to estimate Bucket growth. Do not lower limits based only on
average size; one coding task may legitimately contain many generated files.

The campaign planner should estimate maximum evidence storage from:

```text
planned physical executions
× (workspace archive limit + judge response limits + ordinary evidence reserve)
```

This estimate is an operator warning and quota preflight. It does not reserve
or bill storage. A Bucket quota failure remains an infrastructure failure and
cannot be converted into a model score.

Compression work must not compete with active inference request scheduling on
the controller. Use a bounded packaging pool and include packaging time in
worker headroom calculations. Increase remote Job timeout requirements if the
measured upper bound exceeds existing cleanup headroom.

## Monitoring

Add content-free counters and timings to worker events:

- workspace capture started, completed, and failed;
- workspace node, raw-byte, and archive-byte counts;
- workspace capture and deep-validation milliseconds;
- judge scope registered and revoked;
- judge calls started, completed, and failed;
- judge request and response byte counts;
- evidence manifest completed or rejected;
- evidence retry category and reason code;
- secret detection category without value or content.

Do not place workspace paths, prompts, responses, scorecard reasons, or raw
capabilities in operational events.

Campaign status should summarize evidence retries separately from agent,
benchmark, quota, and provider failures. An unusual increase in evidence
failures should stop new wave admission after a bounded threshold and require
operator review.

## Security review checklist

Before remote release, verify:

- the judge recorder is reachable only through authenticated HF Job ingress;
- route capabilities have at least 128 bits of entropy;
- route roles and executions are isolated;
- the verifier cannot select an upstream judge host;
- the recorder is the only component adding upstream authorization;
- authorization and cookie headers are never serialized;
- known secrets are scanned before exact evidence reaches disk where possible
  and always before durable publication;
- partial exact evidence is deleted after secret detection;
- archive extraction cannot write outside its destination;
- symlinks cannot escape `/app`;
- special files and mount boundaries are rejected;
- workspace and judge limits are enforced before allocation or forwarding;
- terminal markers follow evidence validation and endpoint cleanup;
- public result stores contain no private evidence content.

Have a reviewer trace one real execution from lock through Bucket publication
and confirm every credential boundary with concrete files and code paths.

## Documentation updates

Update the following documents when behavior lands:

- `docs/architecture.md` should show the post-agent freeze and judge recorder
  in the execution path.
- `docs/run-spec.md` should define the evidence policy and judge task selectors.
- `docs/harbor-integration-contract.md` should require `/app` artifact capture
  in `harbor-job.json`.
- `docs/result-publication.md` should define the scored-execution evidence gate
  and public metadata fields.
- `docs/harbor-cookbook.md` should show validation and download followed by
  inspection and restore commands.
- `README.md` should link the bundle specification from the architecture
  section after implementation.

The implementation report should state measured behavior and remote canary
identities. It should not repeat the normative bundle specification.

## Completion criteria

The implementation is complete when all of these statements are true:

- every new remote Harbor job locks and captures `/app`;
- Harbor's frozen workspace is the verifier input and the archived authority;
- workspace packaging uses one validated implementation;
- script-generated and binary files are preserved without session
  reconstruction;
- every expected judge call has exact request and response body evidence;
- each scorecard identifies the judge exchange it used;
- every scored execution has a valid `harbor-hf/trial-evidence/v1` manifest;
- evidence requirements are enforced in direct runs, campaigns, profiles,
  retries, recovery, and publication;
- exact evidence containing a known injected secret never reaches the private
  Bucket;
- public result stores expose evidence metadata without private content;
- local deep validation and restore reproduce workspace file digests;
- endpoint-backed and provider-backed remote canaries pass;
- evidence capture failures retry as infrastructure and never become silent
  model zeros;
- all local quality, mutation, security, and remote cleanup gates pass;
- the production path has no direct judge fallback or session-based workspace
  reconstruction.
