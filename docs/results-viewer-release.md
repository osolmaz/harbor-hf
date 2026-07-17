# Harbor Results Viewer Release

## Canonical V1 Cutover

- Source revision: `add8aab285544cdc96cd155a87b7e279cb470ab3`
- Space revision: `dd27179b1c8b76a7391ea243e9faf8e245d4fbab`
- Active catalog revision: `b5fbf189b64ee63c0deaf2bb9f997e27d9378b22`
- Active ShellBench result revision: `bb97497b1e63405d4e460892cf40a44eb4400d40`

The active Dataset and Space use only the canonical publication `v1` contract.
The primary catalog contains five final 115-task ShellBench evaluations. The
audit catalog contains all 17 immutable publications: five finals, five base
components, five correction components, and two diagnostics. Composed result
pages link directly to their source publications.

The pre-cutover Dataset and Space revisions below remain immutable and directly
addressable for audit. Production readers do not query them. Hosted health,
primary and audit lists, permanent source routes, comparison, and browser checks
passed after cutover at desktop and mobile viewport sizes.

## Pre-Cutover Release

- Space: <https://huggingface.co/spaces/osolmaz/harbor-results>
- Runtime: public Docker Space on `cpu-basic`
- Source revision: `c5fef6b0c025f90098a40c7fe1c337716a473a1d`
- Space revision: `4facb5778f1239fe9578d1eff36d09255df4a36e`
- Credentials: none
- Index Dataset: `osolmaz/benchmark-run-index`
- Catalog revision: `d2708c4b379f1886d53409c09dcb771538fbaf09`
- ShellBench result revision: `3ca396225803d5ce65dcf58ad41d7e0f719b3d0d`
- Smoke result revision: `05720340ef06c19f035bd7a71e46cdf67d93a975`

The public projection contains six complete runs, 24 logical trials, 25
physical executions, verifier metrics, serving configuration, immutable model
and agent revisions, hardware metadata, safe artifact metadata, and checksummed
provenance. Run scores are `100%`, `50%`, `50%`, `50%`, `33.3%`, and `0%`.

The complete evidence remains in the private `osolmaz/benchmark-runs` Bucket:
2,691 files and 49,281,998 bytes at deployment time. The Space has no token and
cannot read that Bucket. Raw task bodies, sessions, trajectories, logs, and
artifact bytes were not copied into the public datasets or Space repository.

## Architecture

The result publisher writes a discovery index and two bounded catalog scopes
into the global index Dataset. The primary scope contains only final logical
evaluations; the audit scope also contains base, correction, and diagnostic
publications. Composed result pages link directly to their source publications.
Append-only promotion and withdrawal decisions change only the primary
projection.

The published projections are:

- primary catalog windows used by default for scores and comparisons;
- audit catalog windows selected explicitly by operators;
- discovery index windows with immutable result Dataset references;
- run- and publication-keyed lookup rows with aggregate score, role, and
  provenance metadata.

Both use deterministic power-of-two sizes from 1 through 2,048 rows. A catalog
request downloads one bounded Parquet snapshot. Opening a run then downloads
one immutable run-keyed lookup and only that publication's revision-pinned run,
trial, execution, metric, and artifact tables. Publication-keyed lookups keep
composed-result source links stable after list-window compaction. Catalog
aggregates are recomputed and compared with detail tables before a detail
response is returned.

The Docker Space serves a versioned FastAPI contract and one React application.
It provides filtered and paginated run and campaign lists, stable detail URLs,
task-level comparison, deterministic server-side result sorting, ETags,
structured errors, an OpenAPI snapshot, and fail-closed private artifact and
trajectory routes. The responsive filter toolbar fits phone, tablet,
narrow-laptop, and desktop widths. Run lists default to decreasing score.

The visual implementation was informed by Harbor's Apache-2.0 viewer at commit
`3914ab318b2dfc8d6f7e73e3587d5be401a79d89`. No Harbor source file was copied.
The coding-agent leaderboard at commit
`28435a2381d4a502591755ff2204c8a16c26fa35` was reviewed for product-level
leaderboard and comparison patterns only.

## Verification

- Python: 1,517 tests with 91.10% coverage.
- Frontend: TypeScript and Vite production build.
- Hosted API: health, catalog, run detail, comparison, restricted content, and
  permanent-route smoke requests.
- Hosted browser: eight Playwright tests passed on desktop Chrome and iPhone 13
  profiles, including toolbar overflow checks from 320px through 1,280px and
  score sorting in both directions.
- Runtime: Space reached `RUNNING`; application startup completed cleanly.
- Privacy: Space variables are public configuration only and the secret list is
  empty.

Some rollouts remained in `APP_STARTING` after their images built successfully,
without producing container logs. The identical images started and passed their
browser suites in local Docker. Factory rebuilds cleared the stale Space
container state; the replacements started Uvicorn normally and passed the
hosted suite. The only request-time warning states that Hub reads are anonymous,
which is the intended public deployment boundary.

The public capability response deliberately reports trajectory and artifact
content access as unavailable. A future protected deployment can add sanitized
trajectory projections and scoped private reads without weakening this public
boundary.
