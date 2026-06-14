# Per-Seed Summary Audit

Date: 2026-06-14

## Context

The publication goal requires that any claim of LR-RGDA or LR-RGDA+ZS improving
over LADA be supported not only by mean/std but also by every required seed.
The summary script already reported mean delta, min delta, and all-positive
flags, but it did not expose the per-seed margins or missing expected seeds.

## Changes

Updated `scripts/summarize_joint_classifier_replay.py`:

- added `--expected_seeds`;
- added `missing expected seeds`;
- added `LR-RGDA - LADA by seed`;
- added `LR-RGDA+ZS - LADA by seed`;
- computes deltas per result file rather than by zipping separately aggregated
  metric lists.

Updated both launchers:

- `scripts/run_joint_classifier_replay.sh`
- `scripts/run_joint_training_ablation.sh`

Both now pass `--expected_seeds "${SEEDS}"` into the summary script.

Updated `experiments/README_publication_repro.md` to document the missing-seed
and per-seed delta columns.

## Verification

Synthetic summary test with only seeds 42/43 and expected seeds 42/43/44:

```text
missing expected seeds = 44
LR-RGDA - LADA by seed = 42:1.00, 43:0.40
LR-RGDA+ZS - LADA by seed = 42:1.50, 43:0.70
```

Ran:

```bash
python scripts/summarize_joint_classifier_replay.py \
  --input_dir /private/tmp/joint_summary_synth_20260614 \
  --output_csv /private/tmp/joint_summary_synth_20260614/summary.csv \
  --output_markdown /private/tmp/joint_summary_synth_20260614/summary.md \
  --expected_seeds '42 43 44'
bash -n scripts/run_joint_classifier_replay.sh
bash -n scripts/run_joint_training_ablation.sh
python -m py_compile scripts/summarize_joint_classifier_replay.py
DRY_RUN=1 OUT_DIR=/private/tmp/joint_classifier_replay_dry_run_expected_20260614 \
  CLASSIFIER_FEATURE_TRANSFORMS='test' REPLAY_MODES='real' SEEDS='42 43' \
  bash scripts/run_joint_classifier_replay.sh
DRY_RUN=1 OUT_DIR=/private/tmp/joint_training_dry_run_expected_20260614 \
  GPUS='0 1 2 3' SEEDS='42 43' \
  bash scripts/run_joint_training_ablation.sh
git diff --check
```

All checks passed.

## Remote Status

Another low-frequency SSH query failed:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

The joint training ablation remains pending.
