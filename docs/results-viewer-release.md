# Harbor Results Viewer Release

## Deployment

- Space: <https://huggingface.co/spaces/osolmaz/harbor-results>
- Runtime: public Docker Space on `cpu-basic`
- Space revision: `a93a672e4ee160c691e86f0f5fd5b26cef69d62f`
- Credentials: none
- Index Dataset: `osolmaz/benchmark-run-index`
- Catalog revision: `d30eb71a421642d37eb635840e5c132340084181`
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

The result publisher now writes two bounded projections into the global index
Dataset:

- discovery index windows with immutable result Dataset references;
- compact catalog windows with aggregate score and run metadata.

Both use deterministic power-of-two sizes from 1 through 2,048 rows. A catalog
request downloads one bounded Parquet snapshot. Opening a run then downloads
only that publication's revision-pinned run, trial, execution, metric, and
artifact tables. Catalog aggregates are recomputed and compared with detail
tables before a detail response is returned.

The Docker Space serves a versioned FastAPI contract and one React application.
It provides filtered and paginated run and campaign lists, stable detail URLs,
task-level comparison, ETags, structured errors, an OpenAPI snapshot, and
fail-closed private artifact and trajectory routes.

The visual implementation was informed by Harbor's Apache-2.0 viewer at commit
`3914ab318b2dfc8d6f7e73e3587d5be401a79d89`. No Harbor source file was copied.
The coding-agent leaderboard at commit
`28435a2381d4a502591755ff2204c8a16c26fa35` was reviewed for product-level
leaderboard and comparison patterns only.

## Verification

- Python: 1,414 tests with branch coverage above the required 85% gate.
- Frontend: TypeScript and Vite production build.
- Hosted API: health, catalog, run detail, comparison, restricted content, and
  permanent-route smoke requests.
- Hosted browser: Playwright on desktop Chrome and iPhone 13 profiles.
- Runtime: Space reached `RUNNING`; application startup completed cleanly.
- Privacy: Space variables are public configuration only and the secret list is
  empty.

The first rollout remained in `APP_STARTING` after its image built successfully,
without producing container logs. The identical image started and passed its
browser suite in local Docker. One factory rebuild cleared the stale Space
container state; the replacement started Uvicorn normally and passed the hosted
suite. The only request-time warning states that Hub reads are anonymous, which
is the intended public deployment boundary.

The public capability response deliberately reports trajectory and artifact
content access as unavailable. A future protected deployment can add sanitized
trajectory projections and scoped private reads without weakening this public
boundary.
