# Trial evidence bundle specification

This specification defines the private evidence saved for one physical Harbor
execution. A trial evidence bundle contains the complete `/app` workspace that
existed after the agent stopped, the retained agent session, each recorded judge
exchange, and the verifier records used to produce the reward.

A bundle has one required entry point, `evidence/trial-evidence.json`. Every
other file is referenced from that manifest by a safe path relative to the
Harbor trial root, exact byte count, and SHA-256 digest.

## Bundle structure

A complete scored execution has this shape inside its Harbor execution tree:

```text
harbor-jobs/<job>/<trial>/
├── agent/
│   └── openclaw-sessions/
│       ├── <session-id>.jsonl
│       └── <session-id>.trajectory.jsonl
├── artifacts/
│   ├── manifest.json
│   └── workspace/
│       └── app/                         # Harbor's frozen post-agent copy
├── evidence/
│   ├── trial-evidence.json              # Required entry point
│   ├── workspace.tar.zst                # Complete captured /app tree
│   ├── workspace-files.jsonl            # One record per workspace node
│   └── judge/
│       └── judge-0001/
│           ├── request-received.json
│           ├── request-forwarded.json
│           ├── response-upstream.bin
│           ├── response-delivered.bin
│           └── exchange.json
└── verifier/
    ├── scorecard.json
    ├── reward.txt
    ├── test-stdout.txt
    ├── test-stderr.txt
    └── judge-selection.json
```

The raw `artifacts/workspace/app` directory exists while Harbor runs the
verifier. Harbor-HF packages that directory into `workspace.tar.zst` after
Harbor exits and before private artifact finalization. Harbor-HF may remove the
raw workspace directory after the archive and file index pass validation. The
archive and index are the durable workspace evidence.

A physical execution that fails before one of these components exists still
writes `trial-evidence.json` when the worker can do so. Each component then has
an explicit capture status and reason. Such a bundle cannot support a scored
outcome.

## Minimal complete manifest

This example omits optional response metadata and additional verifier logs. The
paths are relative to the Harbor trial root that contains the `evidence/`,
`agent/`, and `verifier/` directories.

```json
{
  "schema_version": "harbor-hf/trial-evidence/v1",
  "execution_id": "exec-0123456789abcdef",
  "trial_id": "trial-0123456789abcdef",
  "task_name": "example-task",
  "task_digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
  "logical_attempt": 1,
  "physical_attempt": 1,
  "captured_at": "2026-07-21T18:04:12.123456+00:00",
  "workspace": {
    "status": "captured",
    "root": "/app",
    "archive": {
      "path": "evidence/workspace.tar.zst",
      "size_bytes": 18432,
      "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
      "media_type": "application/vnd.harbor-hf.workspace+tar+zstd"
    },
    "file_index": {
      "path": "evidence/workspace-files.jsonl",
      "size_bytes": 4096,
      "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
      "media_type": "application/x-ndjson"
    },
    "entry_count": 18,
    "regular_file_count": 11,
    "regular_file_bytes": 9021,
    "archive_format": "pax",
    "compression": "zstd",
    "compression_level": 6
  },
  "agent": {
    "status": "captured",
    "sessions": [
      {
        "path": "agent/openclaw-sessions/2f1251d2.jsonl",
        "size_bytes": 12842,
        "sha256": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
        "media_type": "application/x-ndjson"
      }
    ],
    "trajectories": [
      {
        "path": "agent/openclaw-sessions/2f1251d2.trajectory.jsonl",
        "size_bytes": 19304,
        "sha256": "sha256:5555555555555555555555555555555555555555555555555555555555555555",
        "media_type": "application/x-ndjson"
      }
    ]
  },
  "judge": {
    "status": "captured",
    "expected": true,
    "exchanges": [
      {
        "exchange_id": "judge-0001",
        "attempt": 1,
        "record": {
          "path": "evidence/judge/judge-0001/exchange.json",
          "size_bytes": 1732,
          "sha256": "sha256:6666666666666666666666666666666666666666666666666666666666666666",
          "media_type": "application/json"
        }
      }
    ]
  },
  "verifier": {
    "status": "captured",
    "scorecard": {
      "path": "verifier/scorecard.json",
      "size_bytes": 3120,
      "sha256": "sha256:7777777777777777777777777777777777777777777777777777777777777777",
      "media_type": "application/json"
    },
    "reward": {
      "path": "verifier/reward.txt",
      "size_bytes": 4,
      "sha256": "sha256:8888888888888888888888888888888888888888888888888888888888888888",
      "media_type": "text/plain"
    },
    "judge_selection": {
      "path": "verifier/judge-selection.json",
      "size_bytes": 96,
      "sha256": "sha256:9999999999999999999999999999999999999999999999999999999999999999",
      "media_type": "application/json"
    },
    "stdout": {
      "path": "verifier/test-stdout.txt",
      "size_bytes": 0,
      "sha256": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "media_type": "text/plain"
    },
    "stderr": {
      "path": "verifier/test-stderr.txt",
      "size_bytes": 0,
      "sha256": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "media_type": "text/plain"
    },
    "selected_judge_exchange_id": "judge-0001"
  },
  "completion": {
    "status": "complete",
    "requirements": [
      "workspace",
      "agent_session",
      "agent_trajectory",
      "judge_exchange",
      "verifier_scorecard",
      "verifier_reward",
      "verifier_stdout",
      "verifier_stderr",
      "judge_selection"
    ]
  }
}
```

## Object model

A trial evidence bundle belongs to one physical execution. It never combines
files from separate execution IDs, even when those executions belong to the
same logical trial.

The bundle records five stable concepts:

- execution identity;
- post-agent workspace state;
- agent conversation records;
- judge HTTP exchanges;
- verifier output and the judge exchange selected for scoring.

The manifest references immutable files. It does not embed workspace contents,
agent messages, prompts, responses, or scorecards. This keeps the entry point
small enough to validate before loading large private evidence.

## Manifest fields

| Field | Required | Type | Meaning |
| --- | --- | --- | --- |
| `schema_version` | Yes | string | Exact bundle schema version. |
| `execution_id` | Yes | string | Harbor-HF physical execution identity. |
| `trial_id` | Yes | string | Logical trial identity. |
| `task_name` | Yes | string | Exact Harbor task name. |
| `task_digest` | Yes | string | SHA-256 task content digest from the locked run. |
| `logical_attempt` | Yes | integer | Benchmark-semantic attempt, starting at 1. |
| `physical_attempt` | Yes | integer | Infrastructure execution attempt, starting at 1. |
| `captured_at` | Yes | string | UTC time at which the manifest was finalized. |
| `workspace` | Yes | object | Complete `/app` snapshot state. |
| `agent` | Yes | object | Session and trajectory state. |
| `judge` | Yes | object | Recorded verifier-judge exchanges. |
| `verifier` | Yes | object | Scorecard and reward, plus logs and judge selection. |
| `completion` | Yes | object | Required evidence and final completeness decision. |

Unknown fields are validation errors throughout this schema. There is no open
metadata map. Adding a new evidence concept requires changing the active schema
and its validators together.

### Schema version

`schema_version` must equal `harbor-hf/trial-evidence/v1`. Readers must reject
any other value.

The version identifies the manifest schema, workspace archive rules, file index
rules, judge exchange contract, and completeness rules as one unit.

### Identity fields

`execution_id` and `trial_id` must match the execution and trial directories
that contain the bundle. They must also match `execution.lock.json`, the Harbor
compatibility export, and the campaign lock when those records exist.

`task_name`, `task_digest`, and `logical_attempt` must match the locked trial.
`physical_attempt` must match the execution ordering recorded by Harbor-HF. A
bundle cannot be reassigned by copying it under another execution path.

`task_digest` uses `sha256:<64 lowercase hex characters>`. The digest is the
Harbor task digest and is independent of the workspace archive digest.

### Capture time

`captured_at` must be an RFC 3339 timestamp with an explicit UTC offset. Writers
use UTC. It records when the completed manifest was written, not when the agent
stopped or when the judge request began. Those times remain in Harbor timing
records and judge exchange metadata.

## File references

Every referenced evidence file uses this shape:

```json
{
  "path": "evidence/workspace.tar.zst",
  "size_bytes": 18432,
  "sha256": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
  "media_type": "application/vnd.harbor-hf.workspace+tar+zstd"
}
```

Rules:

- `path` is relative to the Harbor trial root that contains the `evidence/`
  directory.
- An absolute path, empty segment, `.`, `..`, backslash or NUL, and any URI
  scheme are forbidden.
- A reference must resolve inside that Harbor trial root.
- A referenced path must be a regular file. Outer evidence references cannot
  point to symlinks, directories, devices, sockets, or FIFOs.
- `size_bytes` is the exact file length and must be nonnegative.
- `sha256` is computed over the exact file bytes with no text normalization.
- `media_type` must be one of the types allowed for that field.
- Digest mismatch, size mismatch, missing content, and an unexpected file type
  are fatal validation errors.

Paths to native records begin with `agent/` or `verifier/`. Paths created by
this format begin with `evidence/`.

## Workspace evidence

The workspace component captures the complete `/app` tree after the agent and
all of its child processes have stopped, and before the verifier begins.

### Workspace states

`workspace.status` is one of:

| Value | Meaning |
| --- | --- |
| `captured` | Archive and file index are complete and validated. |
| `not_created` | The agent environment never reached workspace creation. |
| `capture_failed` | A workspace existed but could not be captured safely. |
| `secret_detected` | A known injected credential appeared in a workspace path or file. |

A `captured` workspace requires every field shown in the minimal example.
Other states require `reason_code` and must omit the archive and index together
with count and compression fields.

`reason_code` is a short machine value. Allowed values are:

- `environment_not_started`;
- `agent_not_started`;
- `workspace_missing`;
- `workspace_changed_during_capture`;
- `unsupported_path`;
- `unsupported_file_type`;
- `unsafe_symlink`;
- `file_limit_exceeded`;
- `byte_limit_exceeded`;
- `disk_full`;
- `archive_failed`;
- `index_failed`;
- `known_secret_found`;
- `unknown_capture_error`.

Free-form exception text does not belong in the bundle manifest. Harbor-HF
records a redacted exception type and message in its normal execution failure
record.

### Capture boundary

The agent is finished before capture begins. Harbor must stop the agent process
and its descendants before collecting `/app`. A background process that still
changes workspace files causes `workspace_changed_during_capture`.

The frozen copy collected by Harbor is the verifier input. A verifier running
in a separate environment receives that copy at `/app`. A verifier sharing the
agent environment must start only after the copy finishes, and the archived
copy remains authoritative if the verifier modifies its own workspace.

A score is reproducible from the post-agent state. Verifier-created files are
not part of `workspace.tar.zst`; they are verifier evidence.

### Workspace limits

The locked evidence policy defines these bounds before execution:

- maximum number of workspace nodes;
- maximum bytes in one regular file;
- maximum total regular-file bytes;
- maximum archive bytes;
- maximum capture duration.

A writer must not omit files to fit a limit. Crossing any bound fails workspace
capture and makes the physical execution ineligible for a score. Limits are
configuration and infrastructure controls, not benchmark scoring rules.

The initial production defaults are:

| Limit | Default |
| --- | ---: |
| Workspace nodes | 100,000 |
| One regular file | 512 MiB |
| Total regular-file bytes | 2 GiB |
| Compressed archive | 2 GiB |
| Capture duration | 300 seconds |

A manifest may set lower values. Higher values require an explicit operator
policy in the resolved run lock so storage cost cannot change silently.

### File index

`workspace-files.jsonl` is UTF-8 JSON Lines. The first row describes the root
and uses `path: "."`. Remaining rows are sorted by the UTF-8 bytes of `path`.
There is exactly one row per archived node.

A directory row has this shape:

```json
{"mode":493,"path":"output","type":"directory"}
```

A regular-file row has this shape:

```json
{"mode":420,"path":"output/final.json","sha256":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size_bytes":827,"type":"file"}
```

A symbolic-link row has this shape:

```json
{"mode":511,"path":"output/latest.json","target":"final.json","type":"symlink"}
```

Each row is a strict object. Unknown fields are errors.

Common field rules:

- `path` is relative to `/app`, except for the root value `.`.
- Paths use `/` and valid UTF-8 in Unicode NFC form.
- Absolute paths, empty segments, `.`, `..`, backslashes, NUL, and duplicate
  paths are forbidden.
- `type` is `directory`, `file`, or `symlink`.
- `mode` is the POSIX permission and executable bit mask in the range 0 through
  4095.

A file row requires `size_bytes` and `sha256`. A directory row forbids them. A
symlink row requires `target` and forbids file size and digest fields.

Symlink targets must resolve inside `/app` when interpreted from the symlink's
parent. Absolute targets, escaping targets, cycles, and dangling targets are
invalid. The snapshotter never follows a symlink while reading workspace
content.

Hard-linked regular files are stored as independent regular-file members. The
bundle does not preserve inode identity. Extended attributes, ACLs, device
nodes, sockets, FIFOs, mount points, Linux capabilities, user IDs, group IDs,
and original modification times are outside the workspace format. Tasks that
need those properties must expose them through declared sidecar or verifier
artifacts instead of relying on this snapshot.

### Workspace archive

`workspace.tar.zst` is a POSIX pax tar archive compressed with Zstandard. It
contains one top-level `app/` directory. Member paths correspond exactly to the
file index after adding the `app/` prefix.

Archive rules:

- members appear in file-index order;
- regular-file member bytes match the file-index digest and size;
- directories and safe symbolic links are explicit members;
- tar user and group IDs are zero;
- tar user and group names are empty;
- modification times are zero;
- permission bits come from the file index;
- hard links, sparse members, devices, sockets, FIFOs, pax path escapes, and
  unknown member types are forbidden;
- the Zstandard frame uses compression level 6 and one compression thread;
- the writer records its Zstandard library version in execution runtime
  evidence, outside the bundle manifest.

Normalized ownership and times make the archive independent of transient
Sandbox metadata. File bytes, paths, node types, permissions, and safe symlink
targets remain exact.

The validator must compare the archive with every file-index row. Matching only
the outer archive digest is insufficient for deep validation.

### Workspace restoration

A restore operation creates a new empty destination. It must refuse to merge
into a nonempty directory.

Restoration proceeds in this order:

1. validate the manifest, archive reference, and file-index reference;
2. validate every file-index row and all aggregate counts;
3. stream the archive while checking member order, paths, types, sizes, and
   digests;
4. create directories and regular files without following symlinks;
5. create validated symlinks after their targets exist;
6. apply recorded permission bits;
7. verify the restored tree against the file index.

The restore command performs no remote fetch. Operators first download the
private execution evidence and then run:

```bash
harbor-hf evidence verify PATH/TO/EXECUTION
harbor-hf evidence restore PATH/TO/EXECUTION --destination ./restored
```

## Agent evidence

`agent.status` is one of `captured`, `not_started`, or `capture_failed`.

A captured OpenClaw execution requires at least one session JSONL and its
matching trajectory JSONL. Every reference includes an exact digest and byte
count. Session and trajectory files remain in Harbor's native agent directory;
the evidence manifest does not duplicate their bytes.

A session file must contain only valid JSON objects separated by newlines. A
trajectory must also contain valid JSON objects and must match the session
identity recorded by OpenClaw. Empty files do not satisfy the requirement.

Multiple session pairs are allowed when the agent creates them during one
physical execution. References are sorted by path. Harbor-HF must not choose the
largest session and discard the others.

`not_started` is valid only when Harbor reports that agent execution never
began. `capture_failed` requires a reason code. Neither state can support a
scored execution.

The bundle preserves model-visible messages and tool records returned by the
agent runtime. It does not claim to reconstruct files from those records. The
workspace archive is the authority for file contents.

## Judge evidence

Judge evidence records OpenAI-compatible HTTP calls made by the verifier. The
trusted Harbor-HF recorder receives each call through an execution-scoped
capability route and forwards it to the locked judge provider.

### Judge expectation

`judge.expected` comes from the resolved benchmark judge policy for the exact
task. A benchmark that mixes deterministic and LLM-scored tasks must lock the
set of task names or task digests that require a judge.

`judge.status` is one of:

| Value | Meaning |
| --- | --- |
| `captured` | Every expected judge call has a complete exchange record. |
| `not_expected` | The locked task uses no external judge. |
| `not_called` | A judge was expected but the verifier made no call. |
| `capture_failed` | A call occurred but its evidence is incomplete. |
| `secret_detected` | A known credential appeared in a request or response body. |

`not_expected` requires `expected: false` and an empty exchange list.
`captured` requires one or more exchanges when `expected` is true. Other states
cannot support a scored execution for a judge-required task.

### Judge directory

Each HTTP attempt receives a unique execution-local ID in increasing order:
`judge-0001`, `judge-0002`, and so on. Retried calls are never overwritten.

An exchange directory contains:

| File | Required | Meaning |
| --- | --- | --- |
| `request-received.json` | Yes | Exact HTTP request body received from the verifier. |
| `request-forwarded.json` | Yes | Exact HTTP request body sent upstream. |
| `response-upstream.bin` | When upstream responded | Exact decoded upstream response bytes before local transformation. |
| `response-delivered.bin` | Yes | Exact response bytes returned to the verifier. |
| `exchange.json` | Yes | Strict metadata and references for this attempt. |

The two request bodies may be byte-identical. Both remain required because the
recorder enforces the locked model and may normalize a provider route in later
implementations. Any difference must be explained by typed transformation
fields in `exchange.json`.

`response-upstream.bin` contains the exact upstream HTTP body bytes after
transfer framing and before content decoding. It is absent when no upstream
response began. The delivered body contains the exact bytes returned to the
verifier, including the recorder's bounded local error body when needed.

### Exchange record

A successful exchange record has this shape:

```json
{
  "schema_version": "harbor-hf/judge-exchange/v1",
  "exchange_id": "judge-0001",
  "execution_id": "exec-0123456789abcdef",
  "trial_id": "trial-0123456789abcdef",
  "attempt": 1,
  "started_at": "2026-07-21T18:03:58.120000+00:00",
  "finished_at": "2026-07-21T18:04:02.871000+00:00",
  "provider": "hf-inference-provider",
  "upstream_url": "https://router.huggingface.co/v1/chat/completions",
  "requested_model": "deepseek-ai/DeepSeek-V3.2",
  "forwarded_model": "deepseek-ai/DeepSeek-V3.2",
  "request_received_headers": {
    "content-type": "application/json"
  },
  "request_forwarded_headers": {
    "accept-encoding": "identity",
    "content-type": "application/json"
  },
  "request_received": {
    "path": "request-received.json",
    "size_bytes": 42112,
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "media_type": "application/json"
  },
  "request_forwarded": {
    "path": "request-forwarded.json",
    "size_bytes": 42112,
    "sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "media_type": "application/json"
  },
  "response_upstream": {
    "path": "response-upstream.bin",
    "size_bytes": 2901,
    "sha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "media_type": "application/json"
  },
  "response_delivered": {
    "path": "response-delivered.bin",
    "size_bytes": 2901,
    "sha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "media_type": "application/json"
  },
  "upstream_http_status": 200,
  "delivered_http_status": 200,
  "upstream_request_id": "req-8f07b4",
  "upstream_response_headers": {
    "content-type": "application/json",
    "x-request-id": "req-8f07b4"
  },
  "delivered_response_headers": {
    "content-type": "application/json",
    "x-harbor-judge-exchange-id": "judge-0001"
  },
  "transport_status": "complete",
  "transformation": "none"
}
```

Rules:

- Body references in `exchange.json` are relative to that exchange directory,
  must use one safe filename, and must resolve to regular files.
- `schema_version` must equal `harbor-hf/judge-exchange/v1`.
- Identity fields must match the parent trial evidence manifest.
- `attempt` starts at 1 and exchange records are contiguous.
- Times use RFC 3339 with UTC offsets, and finish cannot precede start.
- The requested model must match the locked judge model.
- The forwarded model must also match the locked judge model.
- `stream: true` requests are rejected in this schema. Judge calls are bounded
  non-streaming JSON responses.
- Request and response bodies are bounded by the locked judge evidence policy.
- `upstream_url` must equal the locked credential-free judge URL. User info,
  query parameters, and fragments are forbidden.
- `request_received_headers` and `request_forwarded_headers` are allowlisted
  maps. They may contain `accept`, `accept-encoding`, `content-encoding`,
  `content-length`, `content-type`, and `user-agent`.
- `upstream_http_status` is required when an upstream response began and absent
  otherwise. `delivered_http_status` is always required.
- `upstream_request_id` is optional because providers do not always return one.
- The two response-header maps may contain `content-encoding`, `content-length`,
  `content-type`, `request-id`, `retry-after`, `x-amzn-requestid`,
  `x-harbor-judge-exchange-id`, `x-ratelimit-limit-requests`,
  `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-requests`,
  `x-ratelimit-remaining-tokens`, `x-ratelimit-reset`,
  `x-ratelimit-reset-requests`, and `x-request-id`.
- Header names are lowercase. Any header outside the allowlist is omitted from
  evidence. Authorization, proxy authorization, cookies, and set-cookie are
  always omitted.
- `transport_status` is `complete`, `upstream_error`, `timeout`,
  `client_disconnected`, `response_too_large`, or `recorder_error`.
- `transformation` is `none`, `model_enforced`, `provider_route_normalized`, or
  `local_error_response`.

The recorder saves the exact HTTP body bytes. Parsing and reserializing JSON is
not an acceptable substitute. Metadata may contain parsed values for
validation, but the body reference remains authoritative.

### Judge route isolation

A raw route capability is authorization and must never appear in evidence. The
execution may retain its SHA-256 digest in normal route metadata.

One capability maps to one physical execution and one role. Agent inference
routes remain content-free. Judge routes record full bodies. A capability for
one role or execution cannot call another route.

The recorder rejects:

- unknown, revoked, expired, or malformed capabilities;
- requests after the per-execution call limit;
- an unapproved model;
- a request larger than the locked limit;
- streaming judge requests;
- unsupported paths or HTTP methods;
- a direct upstream URL supplied by the caller.

The recorder is temporary and runs inside the remote HF Job. Endpoint-backed
and provider-backed waves both start it when the benchmark has a judge policy.

### Score selection

The gateway returns the exchange ID in the allowlisted
`X-Harbor-Judge-Exchange-ID` response header. A conforming verifier writes this
ID to `verifier/judge-selection.json`:

```json
{
  "schema_version": "harbor-hf/judge-selection/v1",
  "exchange_id": "judge-0001"
}
```

The trial evidence manifest copies the selected ID into
`verifier.selected_judge_exchange_id`. The selected exchange must exist, be
complete, belong to the same execution, and contain the response used to build
`scorecard.json`.

When a verifier uses more than one judge call, it may write an ordered
`exchange_ids` array instead of one `exchange_id`. The manifest then stores the
same ordered list in `selected_judge_exchange_ids`. Exactly one of the singular
or plural forms is allowed.

A scorecard without a valid selection record cannot support a scored outcome
for a judge-required task.

## Verifier evidence

`verifier.status` is `captured`, `not_started`, or `capture_failed`.

A captured verifier requires:

- the native Harbor reward file;
- all native scorecards or deterministic result files required by the task;
- verifier standard output and standard error files, including empty files;
- a judge selection record when the task requires a judge;
- Harbor's verifier timing and exception state in the native trial result.

Every retained file has a digest and size reference. The verifier object may
contain an additional sorted `logs` list for task-specific regular files under
`verifier/`.

The reward remains a Harbor-owned value. Harbor-HF does not reinterpret task
rubrics while creating the bundle. It verifies that reward evidence, scorecard
evidence, judge selection, and Harbor's typed trial result agree structurally.

A deterministic task sets `judge.expected: false` and omits judge selection.
Its deterministic test results and reward still have to be captured.

## Completion decision

`completion.status` is `complete` or `incomplete`. `requirements` is a sorted
list derived from the locked task and execution policy.

Allowed requirement names are:

- `workspace`;
- `agent_session`;
- `agent_trajectory`;
- `judge_exchange`;
- `verifier_scorecard`;
- `verifier_reward`;
- `verifier_stdout`;
- `verifier_stderr`;
- `judge_selection`.

A normally scored OpenClaw task with an external judge requires every item. A
deterministic task omits `judge_exchange` and `judge_selection`. A task that
fails before the agent starts records the applicable statuses but cannot have a
complete scored bundle.

A scored public task outcome is valid only when `completion.status` is
`complete`. Missing or rejected evidence causes a physical execution failure
in the `evidence` category. Harbor-HF retries according to the locked
infrastructure retry policy.

After retries are exhausted, the logical task may become
`infrastructure_exhausted` under the existing fixed-denominator scoring policy.
It must never be presented as a scored model answer. A run may publish that
explicit degraded outcome only when the failure evidence itself is complete.

## Secret handling

The bundle never contains API keys, access tokens, passwords, cookies, private
keys, route capabilities, or authorization headers.

Harbor-HF already knows the actual values injected into the worker, Harbor,
agent, source helper, and judge gateway. It checks those exact byte strings
before any workspace archive, judge body, or final bundle is written to durable
storage.

Workspace handling follows these rules:

1. scan every path component and regular-file body for each known injected
   secret before archive creation;
2. do not follow symlinks while scanning;
3. stop capture immediately when a match is found;
4. delete any partial archive, index, and temporary file;
5. write only a content-free `known_secret_found` failure outside the rejected
   workspace evidence;
6. prevent Bucket publication for that physical execution.

Judge handling scans the bounded in-memory request before forwarding it and
scans the bounded response before writing or delivering it. A match produces a
content-free local error and invalidates the execution.

Exact workspace and judge evidence is never rewritten with `[REDACTED]`.
Rewriting would destroy the evidence needed to reproduce the score. Other
ordinary logs may continue to use the existing redaction pipeline, but a
secret-bearing trial evidence component is discarded and the execution is
invalid.

Secret detection uses actual injected values as the mandatory rule. Optional
high-confidence pattern detectors may reject additional content, but they must
not rewrite exact evidence or reveal the matching value in an error.

All trial evidence remains in the configured private HF Bucket. Public result
Datasets may expose only evidence type, size, digest, completion status, and the
private Bucket-relative path. They never contain workspace files, sessions,
prompts, judge responses, or scorecards.

## Atomicity and immutability

Writers use a temporary file in the destination directory, flush it, and rename
it atomically. `trial-evidence.json` is written last inside the bundle. The
physical execution's root checksum manifest and terminal marker are written
after trial evidence validation.

A partial archive, partial judge exchange, or manifest without its referenced
files does not satisfy completion. Retry logic creates a new physical execution
ID and never overwrites the incomplete execution.

Once a physical execution has `_SUCCESS`, `_FAILED`, or `_CANCELLED`, its files
are immutable. Recovery may adopt a completed execution only after verifying
its root checksums, trial evidence manifest, inner references, execution
identity, and task identity.

## Retention and access

Trial evidence is private even when normalized result metadata is public. The
configured HF Bucket must remain private for the entire write and retention
lifecycle. Access uses the Bucket's normal namespace permissions; the bundle
contains no bearer URL or embedded credential.

A cataloged result must retain every evidence file referenced by its published
artifact metadata. Evidence cannot expire while the result remains active in
the catalog. Withdrawing a result removes it from comparison views but does not
delete evidence automatically.

Deletion is an explicit whole-prefix operation after the result is withdrawn
and no publication, correction, audit, or campaign recovery record references
the execution. Deleting selected workspace, judge, or verifier files would
break the manifest and is forbidden. Retention tools delete the execution or
run as one checked unit and record only content-free deletion metadata.

The storage provider supplies encryption at rest and transport security.
Harbor-HF is responsible for private-bucket validation, content checksums, and
preventing private bytes from entering public result stores.

## Validation

Validation has three levels.

### Structural validation

Structural validation reads only small JSON and JSONL records. It checks:

- supported schema versions;
- strict field types and unknown-field rejection;
- safe relative references;
- identity agreement;
- valid states and required fields per state;
- sorted and unique references;
- allowed media types;
- valid digest and timestamp syntax;
- workspace index row syntax and aggregate counts;
- judge exchange ordering and selected exchange references;
- completion requirements.

### Digest validation

Digest validation reads every referenced regular file and checks its exact size
and SHA-256 digest. It also checks that the parent execution checksum manifest
covers the trial evidence files.

### Deep validation

Deep validation streams `workspace.tar.zst` and compares every member with
`workspace-files.jsonl`. It parses every session and trajectory line, validates
each judge exchange body and metadata reference, and checks native Harbor
verifier records.

A run cannot publish a scored execution until deep validation has succeeded at
least once on the remote worker. `harbor-hf artifacts verify` repeats structural
and digest validation against the private Bucket before result publication. An
operator can request another deep pass for audit or recovery.

Any of these conditions is fatal:

- a required component is absent;
- a path escapes its allowed root;
- a file size or digest differs;
- the workspace archive and index disagree;
- a workspace node type is unsupported;
- a symlink is unsafe;
- an expected judge exchange is absent or incomplete;
- a scorecard selects an unknown exchange;
- an identity differs from the locked execution;
- a known secret is found;
- a scored execution declares incomplete evidence.

## Failure classification

Evidence failures use the physical retry category `evidence`. They do not
change the verifier reward and are never classified as agent or benchmark
failures.

| Failure | Physical category | Retry behavior |
| --- | --- | --- |
| Workspace changes during capture | `evidence` | Retry with a new execution. |
| Archive or index write failure | `evidence` | Retry after normal worker cleanup. |
| Missing expected judge exchange | `evidence` | Retry; do not accept the scorecard. |
| Judge recorder unavailable | `evidence` | Retry without calling the upstream judge directly. |
| Known secret detected | `evidence` | Stop publication and require operator review before another attempt. |
| Workspace policy limit exceeded | `configuration` | Stop until policy or task contents are corrected. |
| Malformed benchmark judge declaration | `configuration` | Reject before remote work. |
| Judge provider timeout with complete error evidence | Existing judge or benchmark policy | Preserve the exchange and apply the locked verifier behavior. |

A direct judge fallback is forbidden. Missing recorder evidence cannot be
repaired by rerunning only the judge after the task workspace has disappeared.

## Runtime lifecycle

One physical execution follows this order:

1. Harbor-HF locks the task, agent, model, runtime, evidence policy, and judge
   policy.
2. Harbor starts the agent environment.
3. The agent runs while Harbor preserves its native session and trajectory.
4. Harbor stops the agent and its descendants.
5. Harbor freezes `/app` through its public artifact API.
6. The verifier receives the frozen workspace.
7. Judge-required verifier calls pass through the execution-scoped recorder.
8. Harbor downloads verifier logs and writes its native result.
9. Harbor-HF packages and deep-validates workspace evidence.
10. Harbor-HF assembles and validates `trial-evidence.json`.
11. Harbor-HF scans exact evidence for known injected secrets.
12. Harbor-HF writes private artifact inventories, root checksums, and the
    terminal marker.
13. Harbor-HF publishes the immutable execution prefix to the private Bucket.
14. Result publication verifies the bundle again before accepting a scored row.

No verifier or publisher reads the live agent workspace after step 5.

## Evidence policy in the experiment

The active `harbor-hf/v1alpha1` experiment format adds one required evidence
policy under `artifacts` for remote benchmark runs:

```yaml
artifacts:
  bucket: osolmaz/benchmark-runs
  trial_evidence:
    workspace_root: /app
    workspace_max_nodes: 100000
    workspace_max_file_bytes: 536870912
    workspace_max_total_bytes: 2147483648
    workspace_max_archive_bytes: 2147483648
    workspace_capture_timeout_seconds: 300
    judge_max_request_bytes: 33554432
    judge_max_response_bytes: 33554432
    judge_max_calls_per_execution: 4
```

`workspace_root` must equal `/app`. The field is explicit so the lock and
operator output show the capture boundary. Arbitrary roots are rejected.

All numeric fields are required, positive, and copied into the resolved run and
campaign locks. Planning includes them in experiment and run identity. Changing
a limit creates a different experiment digest and run ID.

The worker may enforce a lower platform ceiling. It must reject the plan before
remote execution rather than silently lowering a locked value.

The benchmark judge declaration also identifies exactly which selected tasks
require judge recording:

```yaml
benchmark:
  judge:
    protocol: openai-compatible
    api_url: https://router.huggingface.co/v1/chat/completions
    model: deepseek-ai/DeepSeek-V3.2
    api_key_secret_name: HF_TOKEN
    task_names:
      - "*"
    exclude_task_names:
      - "3954d9-recall-shipment-traceback"
      - "9f5c9a-pii-bucket-writers"
```

Patterns use the same task-name matching rules as benchmark selection. Every
included pattern must match a selected task. Exclusions must also match selected
tasks and cannot name a task outside an included pattern. The resolved run lock
stores the exact sorted judge-required task names, so runtime behavior does not
depend on pattern matching.

## Extension policy

The first schema has no arbitrary extension map. New workspace node types,
judge protocols, archive formats, and evidence components require an explicit
schema change with validation and replay tests.

Additional verifier logs may appear through the typed `verifier.logs` list.
This is the only open list of file references. Each referenced file still has a
fixed reference shape, a private classification, and a safe location under the
native verifier directory.

## Boundaries

This format records one physical execution. It does not define benchmark
rubrics, reinterpret rewards, reproduce a model provider's hidden internal
state, or promise access to reasoning that the provider did not return.

The workspace contract covers `/app` file paths, bytes, supported node types,
and permission bits. It does not capture process memory, environment variables,
open network connections, `/root`, `/tmp`, `/proc`, or external services.

The bundle is sufficient to inspect the complete submitted workspace and the
exact judge HTTP bodies used for scoring. Re-executing a nondeterministic judge
may return a different answer; the preserved response remains the authority for
the historical score.
