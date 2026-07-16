---
title: "Provider evidence recorder plan"
author: "Onur Solmaz <2453968+osolmaz@users.noreply.github.com>"
date: "2026-07-16"
---

# Provider Evidence Recorder Plan

## Status

Planned. The current loopback-only proxy cannot serve agents running in a
separate HF Sandbox. New provider-backed production campaigns remain blocked
until Phases 1 through 3 pass their local contracts and remote canary.

## Purpose

Preserve content-free evidence for the model requests made by provider-backed
Harbor trials without requiring a permanent service or a private Harbor fork.
The recorder must remain reachable when the wave controller and OpenClaw run in
separate Hugging Face execution environments.

This plan replaces the current loopback-only provider proxy. There is no
parallel legacy transport and no new schema version: provider-backed runs cut
over to the remote recorder after the canary and failure tests pass.

## Current Problem

The wave controller currently starts the recorder on its own `127.0.0.1`
interface and gives that address to OpenClaw. Harbor runs OpenClaw in a separate
HF Sandbox. In that Sandbox, `127.0.0.1` refers to the Sandbox, not the
controller Job, so every model request fails with `ECONNREFUSED` before it can
reach the recorder or the inference provider.

OpenClaw could call the provider directly, but that would bypass the component
that records routing, retries, latency, quota headers, token usage, and provider
errors. Complete request evidence is a Harbor-HF requirement, so direct access
is retained only as an explicit diagnostic probe and is not the benchmark
execution path.

## Target Design

Each provider-backed wave runs one temporary evidence recorder inside its HF
Job. The Job exposes the recorder through Hugging Face's authenticated Job
ingress. Each trial receives an opaque route on that recorder and uses it as its
OpenAI-compatible model base URL.

```text
HF Job                                      HF Sandbox
+-------------------------+                 +------------------+
| Harbor-HF wave worker   |                 | Harbor           |
|                         | authenticated   |   -> OpenClaw     |
| provider recorder :8000 |<----------------|                  |
|          |              |                 +------------------+
+----------|--------------+
           |
           | authenticated provider request
           v
  HF router -> selected inference provider
```

The recorder forwards the request, streams the provider response back to
OpenClaw, and writes the existing content-free `provider-requests.jsonl`
evidence. It does not run inference and it does not store prompts, tool
arguments, model output, or credentials.

Endpoint-backed waves do not expose this port. The recorder exists only for the
lifetime of a provider-backed wave and disappears when the Job exits.

## Component Boundaries

Harbor continues to own task execution, agent lifecycle, sessions, verifier
results, and trial artifacts. OpenClaw continues to use an ordinary
OpenAI-compatible model endpoint.

Harbor-HF owns:

- starting and stopping the recorder;
- requesting authenticated ingress for provider-backed HF Jobs;
- generating, registering, and revoking opaque trial routes;
- checking recorder readiness before admitting trial work;
- forwarding requests to the locked HF provider route;
- recording and publishing sanitized provider evidence;
- classifying recorder, ingress, and provider failures correctly.

No Harbor change is required for this design. A future Harbor-owned agent
sidecar lifecycle could allow the same recorder interface to run beside
OpenClaw inside each Sandbox, but Harbor-HF must not simulate that API with an
OpenClaw-specific wrapper.

## Security Model

The recorder is private even though it has a routable URL:

1. HF Job ingress requires an HF bearer token with access to the Job.
2. Every trial receives a random capability route with at least 128 bits of
   entropy; a trial ID alone is not authorization.
3. The recorder maps each capability to exactly one active trial and rejects
   unknown, expired, or cross-trial routes.
4. Capabilities are revoked when the trial finishes and all remaining
   capabilities are revoked while the wave drains.
5. Raw capabilities, bearer tokens, upstream authorization headers, prompts,
   tool arguments, and response text never enter logs or published evidence.
6. Request size, response capture size, request attempts, concurrency, timeout,
   and spend limits continue to be enforced before forwarding.

The HF token used to enter the Job is distinct in purpose from the recorder's
upstream provider credential, even when both values originate from the same HF
secret. The recorder is the only component that adds upstream authorization.

## Runtime Contract

Provider-backed Job submission declares one fixed recorder port. At worker
startup, Harbor-HF:

1. validates the built-in HF Job identity;
2. binds the recorder to `0.0.0.0` on the declared port;
3. constructs the authenticated ingress URL from the Job identity and port;
4. checks a content-free health route through the external ingress;
5. registers one opaque capability before starting each physical execution;
6. gives only that scoped base URL to the corresponding OpenClaw process;
7. revokes the capability after the execution reaches a terminal state;
8. stops admission and closes the recorder while the wave drains.

A failed external readiness check is an infrastructure failure. It must not be
reported as an agent or benchmark failure, and it must not consume a provider
request attempt because no provider request was sent.

The runtime environment record identifies the transport as the hosted
provider recorder and records the Job ID, port, and ingress hostname. It stores
only digests for capability routes. Existing provider evidence and publication
schemas remain authoritative.

## Failure Handling

The worker distinguishes four boundaries:

| Boundary | Example | Classification |
| --- | --- | --- |
| Recorder startup | bind failure or invalid Job identity | configuration |
| HF Job ingress | unavailable route or authentication rejection | infrastructure |
| Recorder forwarding | timeout or broken upstream connection | transient provider transport |
| Provider response | rate limit, quota, or model error | existing provider category |

Retries create a new physical execution when infrastructure ownership is
ambiguous. The recorder's per-trial request-attempt budget still prevents an
identical model request from being forwarded more times than the immutable run
policy permits. Evidence is flushed before terminal markers are written.

## Implementation Sequence

### Phase 1: Reachable Recorder

- allow the recorder to bind an explicit host and port;
- add a content-free health route;
- expose the fixed port only on provider-backed HF Jobs;
- derive and validate the Job ingress URL inside the remote worker;
- perform an authenticated external readiness check;
- replace the loopback URL passed to provider-backed Harbor executions.

Exit criteria: an OpenClaw tool-use canary in an HF Sandbox reaches the
selected inference provider through the recorder and produces non-empty,
validated provider request evidence.

### Phase 2: Trial Isolation

- replace trial-name routes with random registered capabilities;
- bind each capability to one physical execution and its attempt budget;
- revoke capabilities on completion, cancellation, timeout, and wave drain;
- redact capabilities from process output, runtime records, and artifacts;
- reject requests for inactive or mismatched executions before forwarding.

Exit criteria: concurrent trials cannot use each other's recorder routes, and
secret-scanning tests prove that no capability or credential reaches evidence.

### Phase 3: Failure And Load Verification

- test delayed ingress readiness, invalid authorization, recorder termination,
  interrupted streams, cancellation, and controller shutdown;
- run concurrent streaming and tool-calling tests at powers-of-two concurrency;
- verify that recorder throughput does not become the limiting benchmark
  resource at the selected campaign concurrency;
- confirm that every successful, failed, retried, and cancelled execution has
  the expected request evidence and terminal markers;
- run a one-task remote canary before the first full provider campaign.

Exit criteria: the full provider-backed campaign can run at its selected
concurrency without connection refusals, mixed evidence, leaked secrets, or
unclassified transport failures.

### Phase 4: Optional Harbor-Native Colocation

If Harbor later exposes a public lifecycle API for agent-side helper processes,
propose a small upstream contract that can start a helper beside the agent,
wait for readiness, collect declared artifacts, and guarantee cleanup. Move the
same recorder behind that contract only after remote parity tests pass.

This phase is optional. Until Harbor owns such an API, the authenticated
per-wave recorder remains the production design. Harbor-HF does not maintain a
custom Harbor fork or an OpenClaw-only bootstrap shim.

## Required Tests

- unit tests for explicit binding, health responses, ingress URL validation,
  capability registration, expiry, revocation, and redaction;
- adapter tests proving that only provider-backed Jobs expose the recorder;
- contract tests with the controller and agent on different network hosts;
- streaming, reasoning, and multi-turn tool-call tests that preserve all
  request fields required by the selected provider;
- fault tests at every startup, ingress, forwarding, streaming, and shutdown
  boundary;
- concurrency tests that prove trial evidence cannot mix;
- a remote one-task canary using the exact worker revision intended for the
  full campaign.

The remote canary is a release gate for this transport. A direct provider probe
is useful diagnostics, but it does not satisfy the gate because it bypasses the
recorder and Harbor Sandbox boundary.

## Cutover And Rollback

After local contracts and the remote canary pass, all new provider-backed
campaigns use the hosted recorder. The loopback-only path is removed rather
than retained behind a compatibility flag. Existing immutable run artifacts
remain readable because their evidence schema does not change.

Rollback means reverting the worker revision and pausing new provider-backed
campaign admission. It does not mean silently bypassing evidence capture or
falling back to direct provider calls. Endpoint-backed campaigns are unaffected.

## Success Criteria

The recorder work is complete when:

- separate HF Job and Sandbox environments communicate without special Harbor
  behavior;
- provider requests retain complete content-free routing, retry, quota, usage,
  latency, and error evidence;
- no prompt, response, tool argument, capability, or credential appears in
  provider evidence;
- concurrent trials remain isolated and reproducible;
- transport failures are not scored as benchmark failures;
- the first full provider-backed ShellBench campaign completes through the
  recorder at the selected concurrency;
- the implementation uses one production transport with no legacy shim.
