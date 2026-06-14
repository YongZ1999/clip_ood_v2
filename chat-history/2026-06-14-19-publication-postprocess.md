# Publication Result Postprocess

Date: 2026-06-14

## Context

The publication package now has separate scripts for:

- classifier-side training summary;
- classifier-side training audit;
- strict incremental metrics summary;
- strict incremental metrics audit;
- local package verification.

These need a single postprocess entry point for when remote result files become
available.

## Changes

Added `scripts/postprocess_publication_results.sh`.

Environment variables:

- `TRAINING_DIR`, default `experiments/joint_training_ablation_20260614`;
- `INCREMENTAL_DIR`, default `experiments/joint_incremental_metrics_20260614`;
- `EXPECTED_SEEDS`, default `42 43 44`;
- `EXPECTED_TASKS`, default X-TAIL 10-task order;
- `ALLOW_MISSING`, default `0`.

When training `*_seed*.json` files exist, the script:

- generates `summary.csv` and `summary.md` with
  `scripts/summarize_joint_classifier_replay.py`;
- runs `scripts/audit_training_ablation.py`.

When incremental `*_results.json` files exist, the script:

- generates `incremental_summary.csv`;
- generates `incremental_summary.md`;
- generates `incremental_aggregate.csv`;
- generates `incremental_aggregate.md`;
- runs `scripts/audit_incremental_metrics.py`.

It then runs `bash scripts/verify_publication_package.sh`.

Updated:

- `scripts/verify_publication_package.sh` now shell-checks the postprocess
  script with `bash -n`;
- `experiments/README_publication_repro.md` documents the postprocess command.

## Verification

Ran an absent-results dry run:

```bash
ALLOW_MISSING=1 \
TRAINING_DIR=/private/tmp/postprocess_training_empty_20260614 \
INCREMENTAL_DIR=/private/tmp/postprocess_incremental_empty_20260614 \
bash scripts/postprocess_publication_results.sh
```

It correctly reported both result groups as pending, ran the publication package
verifier, and exited successfully.

Final paper evidence should not use `ALLOW_MISSING=1`.
