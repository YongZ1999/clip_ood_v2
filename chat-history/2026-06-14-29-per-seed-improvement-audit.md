# Per-Seed Improvement Audit

**Date**: 2026-06-14

## Context

The publication goal requires stable multi-seed evidence, not only a positive
aggregate mean. The strict audits already required the full
`lora_nsp_fd_cd` configuration to improve over `lora_vanilla` and `lora_nsp`,
but aggregate means could still hide a seed-level regression.

## Changes

Updated:

```text
scripts/summarize_joint_classifier_replay.py
scripts/audit_training_ablation.py
scripts/audit_incremental_metrics.py
scripts/selftest_publication_audits.py
experiments/README_publication_repro.md
```

Details:

- `summarize_joint_classifier_replay.py` now writes per-seed classifier metric
  columns:

```text
CLIP-ZS by seed
LR-RGDA by seed
LADA by seed
LR-RGDA+ZS by seed
LADA+ZS by seed
```

- `audit_training_ablation.py` now checks `lora_nsp_fd_cd` against
  `lora_vanilla` and `lora_nsp` for each matched seed when these per-seed
  columns are present.
- `audit_incremental_metrics.py` now checks `lora_nsp_fd_cd` against the same
  baselines on Transfer/Average/Last for every matched seed, using the per-run
  incremental summary rows.
- `selftest_publication_audits.py` now includes seed-flip negative cases where
  the aggregate mean improves but seed 42 regresses; both strict audits must
  reject those cases.
- The README now states that strict audits require aggregate and per-seed
  improvements.

## Verification

Generated a temporary classifier-summary check under:

```text
/tmp/training_by_seed_column_check
```

Confirmed `summary.csv` includes the new per-seed metric columns and
`audit_training_ablation.py` parses them.

Ran:

```bash
python scripts/selftest_publication_audits.py
python -m py_compile \
  scripts/summarize_joint_classifier_replay.py \
  scripts/audit_training_ablation.py \
  scripts/audit_incremental_metrics.py \
  scripts/selftest_publication_audits.py

python scripts/audit_training_ablation.py \
  --summary_csv /tmp/training_by_seed_column_check/summary.csv \
  --expected_seeds "42 43 44"

bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
git diff --check
```

Current package status remains:

```text
Summary: PASS=5, PENDING=2
publication package checks passed
```

Remote status was queried once and still failed with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```
