# Training Per-Seed Evidence Required

**Date**: 2026-06-14

## Context

The previous per-seed improvement audit made use of per-seed classifier metric
columns when they were present. However, a stale or manually constructed
classifier-side training `summary.csv` without `... by seed` columns could still
fall back to aggregate-only checks. That is too weak for the publication goal,
which requires stable multi-seed evidence.

## Changes

Updated:

```text
scripts/audit_training_ablation.py
scripts/selftest_publication_audits.py
experiments/README_publication_repro.md
```

Behavior now:

- `audit_training_ablation.py` requires per-seed values for every expected seed
  on:

```text
CLIP-ZS
LR-RGDA
LR-RGDA+ZS
```

- A training summary that has aggregate metrics but lacks the corresponding
  `... by seed` columns is rejected.
- The self-test now includes a legacy aggregate-only training summary and
  verifies that the strict training audit fails it.
- The README now documents that classifier-side training evidence must be
  parseable both in aggregate and for every expected seed.

## Verification

Ran:

```bash
python scripts/selftest_publication_audits.py
python -m py_compile scripts/audit_training_ablation.py scripts/selftest_publication_audits.py
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
