---
title: Harbor Results
colorFrom: gray
colorTo: blue
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
---

# Harbor results

This is a read-only operational view over normalized `harbor-hf` result and
index Datasets. It does not submit campaigns, control endpoints, publish rows,
hold credentials, or store authoritative state. The artifact Bucket and control
Dataset remain canonical.

Configure the Space with public, non-secret environment variables:

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `HARBOR_HF_INDEX_DATASET` | Yes | — | Public global index Dataset as `namespace/name` |
| `HARBOR_HF_INDEX_REVISION` | No | `main` | Index branch, tag, or commit to resolve on refresh |
| `HARBOR_HF_MAX_PUBLICATIONS` | No | `250` | Newest indexed publications to load, from 1 to 2,000 |
| `HARBOR_HF_SPACE_TITLE` | No | `Harbor results` | Page title |

The loader always asks the Hub for anonymous access (`token=False`). Do not add
an HF token or another credential to this Space. The configured index revision
is resolved to a commit before reads, and every result Dataset is read at the
exact revision recorded in its index row.

The expected normalized layouts are:

```text
index Dataset:
  data/index/schema=v1/<publication-id>.parquet

result Dataset:
  data/<runs|trials|executions|metrics|artifacts>/schema=v1/
    campaign=<campaign-id>/<publication-id>.parquet
```

Malformed rows, schema-version mismatches, conflicting index entries, and
provenance mismatches fail closed: the refresh shows an error and no result
tables.
