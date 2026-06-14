# Strict Improvement Audit

**Date**: 2026-06-14

## Context

The publication goal requires LoRA-NSP + classifier evidence to exceed or
substantially improve over the baselines under the audited protocol. The
incremental and training-side audits previously printed warnings when the full
configuration failed to improve, but still exited with status 0. The publication
dashboard would classify such output as `FAIL`, but direct strict audit commands
and intermediate postprocess steps were not hard-failing early enough.

## Changes

Updated:

```text
scripts/audit_incremental_metrics.py
scripts/audit_training_ablation.py
experiments/README_publication_repro.md
```

Behavior now:

- `scripts/audit_incremental_metrics.py` exits nonzero if
  `lora_nsp_fd_cd` fails to improve over `lora_vanilla` or `lora_nsp` on
  Transfer/Average/Last.
- `scripts/audit_training_ablation.py` exits nonzero if `lora_nsp_fd_cd` fails
  to improve over `lora_vanilla` or `lora_nsp` on CLIP-ZS, LR-RGDA, or
  LR-RGDA+ZS.
- The README now states that these strict audits fail when required improvements
  are not satisfied.

## Synthetic Audit Checks

A temporary synthetic positive incremental result directory was generated under:

```text
/tmp/incremental_audit_synthetic_pass
```

It contains 4 methods x 3 seeds x 10 tasks, with `lora_nsp_fd_cd` better than
both baselines. The audit passed:

```text
INCREMENTAL METRICS AUDIT PASSED
```

A temporary synthetic negative incremental result directory was generated under:

```text
/tmp/incremental_audit_synthetic_fail
```

It contains the same method/seed/task coverage, but `lora_nsp_fd_cd` is worse
than both baselines. The audit now exits with status 1 and reports:

```text
INCREMENTAL METRICS AUDIT FAILED: required improvement not satisfied
```

This confirms the strict incremental gate will not pass non-improving results.

## Verification

Ran:

```bash
python scripts/audit_incremental_metrics.py \
  --summary_csv /tmp/incremental_audit_synthetic_pass/incremental_summary.csv \
  --aggregate_csv /tmp/incremental_audit_synthetic_pass/incremental_aggregate.csv \
  --expected_seeds "42 43 44" \
  --expected_k 10

python scripts/audit_incremental_metrics.py \
  --summary_csv /tmp/incremental_audit_synthetic_fail/incremental_summary.csv \
  --aggregate_csv /tmp/incremental_audit_synthetic_fail/incremental_aggregate.csv \
  --expected_seeds "42 43 44" \
  --expected_k 10

python -m py_compile scripts/audit_incremental_metrics.py scripts/audit_training_ablation.py
bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
git diff --check
```

Current package status:

```text
Summary: PASS=5, PENDING=2
publication package checks passed
```

The two pending gates still require real remote result files.

Remote status was queried once and still failed with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```
