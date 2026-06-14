# Incremental Full-Test Split Fix

**Date**: 2026-06-14

## Context

The publication goal requires strict LADA/X-TAIL-aligned evaluation on the full
test split. The new strict incremental launcher called
`src/experiments/run_continual_learning.py`, whose legacy evaluation helper used
`max_samples=1000` by default. That default is useful for smoke tests but is not
acceptable for publication-level Transfer/Average/Last/forgetting evidence.

## Changes

Updated:

```text
src/experiments/run_continual_learning.py
scripts/run_joint_incremental_ablation.sh
scripts/audit_protocol_config.py
experiments/README_publication_repro.md
```

Details:

- `run_continual_learning.py` now accepts:

```bash
--eval_max_samples N
```

- `N <= 0` means no per-dataset test cap and therefore the full test split.
- The legacy default remains `1000` for old scripts and smoke-style runs.
- `scripts/run_joint_incremental_ablation.sh` now passes:

```bash
--eval_max_samples 0
```

- `scripts/audit_protocol_config.py` now checks that the publication incremental
  launcher and README both contain this full-test setting.
- The reproducibility README now documents that `--eval_max_samples 0` is the
  publication setting and avoids the old 1000-example cap.

## Verification

Dry-run command check:

```bash
DRY_RUN=1 \
OUT_DIR=/tmp/joint_incremental_fulltest_dry_run \
GPUS="0 1 2 3" \
bash scripts/run_joint_incremental_ablation.sh
```

Confirmed generated command includes:

```bash
--eval_max_samples 0
```

Ran:

```bash
python -m py_compile src/experiments/run_continual_learning.py scripts/audit_protocol_config.py
python scripts/audit_protocol_config.py
bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
git diff --check
```

Current package status remains:

```text
Summary: PASS=5, PENDING=2
publication package checks passed
```

The two pending gates still require real remote result files.

Remote status was queried once and still failed with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```
