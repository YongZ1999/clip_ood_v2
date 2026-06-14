# Incremental Aggregate Summary

Date: 2026-06-14

## Context

`scripts/summarize_incremental_metrics.py` could summarize individual
`*_results.json` files, but paper tables need mean/std over seeds by method.

## Changes

Updated `scripts/summarize_incremental_metrics.py`:

- added `--aggregate_csv`;
- added `--aggregate_markdown`;
- groups per-run rows by `method`;
- reports aggregate `n`, `seeds`, `K`, missing expected tasks, Transfer,
  Average, Last, forgetting, and matrix warnings;
- formats multi-run values as `mean +/- std`, and single-run values as a single
  score.

Updated `experiments/README_publication_repro.md`:

- strict incremental metrics command now includes aggregate outputs;
- notes that main training-side claims should use aggregate rows whose
  underlying per-run rows have full 10-task sequence, no missing tasks, and no
  matrix warnings.

## Verification

Ran:

```bash
python scripts/summarize_incremental_metrics.py \
  experiments/main_parallel_20260315_151605/combined \
  --output_csv /private/tmp/incremental_metrics_summary_20260614/summary.csv \
  --output_markdown /private/tmp/incremental_metrics_summary_20260614/summary.md \
  --aggregate_csv /private/tmp/incremental_metrics_summary_20260614/aggregate.csv \
  --aggregate_markdown /private/tmp/incremental_metrics_summary_20260614/aggregate.md \
  --expected_tasks 'aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397'
```

The smoke test produced both per-run and aggregate tables. Local historical
results are single-seed, so the aggregate values are single scores rather than
mean/std. These smoke-test results remain non-paper evidence.
