# Incremental Result-Level Full-Test Audit

**Date**: 2026-06-14

## Context

The previous fix made the publication incremental launcher pass
`--eval_max_samples 0`, ensuring future launched runs use the full X-TAIL test
split. However, the strict incremental summary and audit still did not inspect
the result JSON args. A manually copied or stale capped result could therefore
enter `incremental_summary.csv` without being rejected by the strict audit.

## Changes

Updated:

```text
scripts/summarize_incremental_metrics.py
scripts/audit_incremental_metrics.py
scripts/selftest_publication_audits.py
experiments/README_publication_repro.md
```

Details:

- `summarize_incremental_metrics.py` now writes an `eval max samples` per-run
  column from `result["args"]["eval_max_samples"]`.
- `audit_incremental_metrics.py` now fails if any per-run row has missing,
  unparsable, or positive `eval max samples`.
- `eval max samples <= 0` is required for full-test-split publication evidence.
- `selftest_publication_audits.py` now includes a capped-result negative case
  with `eval_max_samples=1000` and verifies that the strict incremental audit
  rejects it.
- The README now documents the result-level full-test check.

## Verification

Generated a temporary synthetic incremental result directory under:

```text
/tmp/incremental_fulltest_column_check
```

Confirmed `incremental_summary.csv` now includes:

```text
path,method,seed,K,eval max samples,...
```

and rows with full-test results contain:

```text
...,10,0,...
```

Ran:

```bash
python scripts/selftest_publication_audits.py
python -m py_compile \
  scripts/summarize_incremental_metrics.py \
  scripts/audit_incremental_metrics.py \
  scripts/selftest_publication_audits.py

python scripts/audit_incremental_metrics.py \
  --summary_csv /tmp/incremental_fulltest_column_check/incremental_summary.csv \
  --aggregate_csv /tmp/incremental_fulltest_column_check/incremental_aggregate.csv \
  --expected_seeds "42 43 44" \
  --expected_k 10

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
