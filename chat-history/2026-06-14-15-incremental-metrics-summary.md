# Incremental Metrics Summary Script

Date: 2026-06-14

## Context

The remaining publication gates include strict incremental Transfer/Average/Last
and forgetting metrics. Existing `*_results.json` files already contain:

- `metrics.transfer`
- `metrics.average`
- `metrics.last`
- `forgetting_rate`
- `accuracy_matrix`
- task sequence metadata

The package needed a reusable summarizer so final training-side results can be
converted into auditable tables without hand-copying metrics.

## Changes

Added `scripts/summarize_incremental_metrics.py`.

The script accepts result JSON files or directories containing `*_results.json`
files and writes:

- CSV summary;
- Markdown summary.

Output columns:

- `path`
- `method`
- `seed`
- `K`
- `missing expected tasks`
- `Transfer`
- `Average`
- `Last`
- `Forgetting`
- `matrix warning`

The script validates that `accuracy_matrix` is a `K x K` matrix where `K` is the
number of task names found in the result metadata.

Updated `scripts/verify_publication_package.sh` so the new script is included in
Python syntax checks.

Updated `experiments/README_publication_repro.md` with the strict incremental
metrics command and interpretation boundary.

## Verification

Ran a smoke test on existing local results:

```bash
python scripts/summarize_incremental_metrics.py \
  experiments/main_parallel_20260315_151605/combined \
  --output_csv /private/tmp/incremental_metrics_summary_20260614/summary.csv \
  --output_markdown /private/tmp/incremental_metrics_summary_20260614/summary.md \
  --expected_tasks 'aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397'
```

Example output:

```text
lora_nsp     Transfer=63.00 Average=68.53 Last=74.11 Forgetting=-13.62
lora_vanilla Transfer=62.30 Average=68.48 Last=74.12 Forgetting=-14.29
```

Also ran:

```bash
python -m py_compile scripts/summarize_incremental_metrics.py
bash scripts/verify_publication_package.sh
```

Both passed.

## Boundary

The smoke-test local results are not current paper evidence. The script is a
reusable summarizer for the final strict incremental results once the remote
training ablation or a dedicated incremental run completes.
