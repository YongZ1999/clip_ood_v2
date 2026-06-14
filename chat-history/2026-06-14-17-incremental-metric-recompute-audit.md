# Incremental Metric Recompute Audit

Date: 2026-06-14

## Context

Strict incremental metrics should be traceable not only to stored `metrics`
fields but also to the underlying `accuracy_matrix`. The summary script now
needs to detect mismatches between stored Transfer/Average/Last values and
values recomputed from the matrix.

## Changes

Updated `scripts/summarize_incremental_metrics.py`:

- recomputes Transfer, Average, and Last from `accuracy_matrix`;
- adds columns:
  - `computed Transfer`
  - `computed Average`
  - `computed Last`
  - `Transfer stored-computed`
  - `Average stored-computed`
  - `Last stored-computed`
- keeps the aggregate table focused on method-level mean/std, while per-run rows
  carry the stored-vs-computed audit.

Updated `experiments/README_publication_repro.md`:

- strict incremental metrics section now states that main claims should only use
  rows with negligible stored-minus-computed deltas.

## Verification

Ran:

```bash
python scripts/summarize_incremental_metrics.py \
  experiments/main_parallel_20260315_151605/combined \
  --output_csv /private/tmp/incremental_metrics_recompute_20260614/summary.csv \
  --output_markdown /private/tmp/incremental_metrics_recompute_20260614/summary.md \
  --aggregate_csv /private/tmp/incremental_metrics_recompute_20260614/aggregate.csv \
  --aggregate_markdown /private/tmp/incremental_metrics_recompute_20260614/aggregate.md \
  --expected_tasks 'aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397'
```

For the local historical LoRA rows, stored and recomputed metrics matched:

```text
lora_nsp:     Transfer -0.0000, Average +0.0000, Last +0.0000
lora_vanilla: Transfer +0.0000, Average +0.0000, Last -0.0000
```

These local historical rows remain smoke-test evidence only, not current paper
evidence.
