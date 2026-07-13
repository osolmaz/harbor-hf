# Endpoint Provisioning

Milestone 3 endpoint provisioning is implemented as a domain service and a
narrow Hugging Face adapter. This slice deliberately stops before deployment
wave execution: it does not resume endpoints, submit shards, modify the
campaign reconciler, or add CLI commands.

## Boundary

The provisioning boundary exposes five lifecycle behaviors:

- create one exact desired endpoint;
- inspect one deterministic managed identity;
- adopt an existing exact endpoint through inspection;
- pause an endpoint;
- delete an endpoint only when an explicit caller selects the retention action.

The provider port itself stays narrower: create, inspect, pause, and delete are
remote calls; adoption is a verified application-layer decision based on
inspection.

The application service owns adoption, comparison, polling, and cleanup
verification. The Hugging Face adapter owns SDK calls, provider error
classification, secret-value resolution, and validation of untrusted response
objects. Domain code does not import Hugging Face SDK response models.

## Deployment Identity

The deployment digest covers the complete model and deployment profiles while
excluding display-only profile IDs and a legacy pre-bound endpoint reference.
It therefore changes for behavior-affecting or cost-affecting deployment input,
but not when an equivalent profile is renamed.

A managed endpoint name is a deterministic hash of:

```text
namespace + campaign ID + deployment digest
```

The name is `harbor-hf-` followed by 40 lowercase hexadecimal characters. The
endpoint also carries deterministic managed, campaign, and deployment tags.
Adoption requires the expected namespace, name, and all managed tags. Matching
configuration at an unrelated or untagged endpoint is not enough.

## Exact Effective Configuration

The desired and observed effective configurations use the same frozen typed
model. Exact comparison covers:

- model repository and full revision;
- framework, task, digest-pinned custom image, health route, and port;
- ordered container command and arguments;
- complete non-secret environment and secret-name set;
- accelerator, instance size, instance type, replica bounds, scale-to-zero
  timeout, scaling metric, and threshold;
- provider vendor, region, and optional account identity;
- endpoint access type, route, HTTP caching, and complete tags.

Runtime controls such as context, batching, precision, parsers, caching, and
speculation remain exact because their ordered engine arguments and environment
are part of the compared model configuration and deployment digest. Unknown
endpoint parameters are rejected instead of being silently omitted. Secret
values are resolved only inside the adapter immediately before create and are
never included in desired state or snapshots.

All mismatching fields are reported together using stable dotted paths. The
provisioner never updates a mismatched endpoint to make it fit.

## Create And Adoption Rules

Provisioning first inspects the deterministic identity:

1. An absent endpoint is created once.
2. An exact existing endpoint is adopted only if it already reports
   `state=paused` and `readyReplica=0`.
3. An active existing endpoint is rejected without pausing it; lifecycle lease
   ownership belongs to the later wave controller.
4. A create timeout, transport failure, conflict, or server error is treated as
   ambiguous. The provisioner repeatedly inspects the same deterministic
   identity and adopts it only after managed identity and complete effective
   configuration verification. It never issues a second create during that
   attempt.
5. A successfully created or ambiguously created endpoint is paused and then
   inspected until `state=paused` and `readyReplica=0` are both observed.

`targetReplica` is retained as evidence and may remain nonzero while paused.
It is not used as a substitute for the ready-replica check.

If a create response describes a mismatched resource, the provisioner first
attempts paused-zero-ready cleanup of the deterministic name, then rejects the
resource. A provider response that cannot be validated is never converted into
domain state.

## Pause And Delete

Pause is idempotent and is complete only after a fresh inspection reports both
paused state and zero ready replicas. Ambiguous pause responses are resolved by
inspection. Inspection failures are errors, not evidence that an endpoint is
absent.

Delete is separate from ordinary cleanup. It requires an exact, already
paused-zero-ready endpoint and verifies that subsequent inspection returns not
found. Ambiguous delete responses are likewise resolved through inspection.

## Integration Boundary

The future deployment wave controller should build `DesiredEndpoint` from its
locked model and deployment profiles, acquire the endpoint lease before any
lifecycle mutation, call `create_or_adopt`, and retain the returned final
snapshot as provisioning evidence. It must continue to use the independent
watchdog before resume. This provisioning slice does not grant lifecycle
ownership and does not replace the existing watchdog or lease behavior.

Tests use only in-memory ports and sanitized SDK response contracts. They do
not load models, call remote endpoints, create paid resources, or benchmark.
