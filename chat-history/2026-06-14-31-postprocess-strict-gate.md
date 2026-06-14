# Postprocess Strict Publication Gate

**Date**: 2026-06-14

## Context

The remote postprocess command generated classifier-side and incremental
summaries, ran their strict audits, and then ran the local package verifier.
However, it did not explicitly run the full dashboard in strict mode at the end.
For the publication goal, the final postprocess command should fail if any gate
is still pending after results are processed.

## Changes

Updated:

```text
scripts/postprocess_publication_results.sh
experiments/README_publication_repro.md
```

Behavior now:

- With `ALLOW_MISSING=0`, postprocess ends with:

```bash
python scripts/audit_publication_gates.py --strict
```

- With `ALLOW_MISSING=1`, postprocess keeps the local dry-run behavior and skips
  the final strict dashboard, because missing results are expected in that mode.
- The README now documents that `ALLOW_MISSING=1` must not be used for final
  paper evidence.

## Verification

Ran dry-run postprocess with missing result directories:

```bash
ALLOW_MISSING=1 \
TRAINING_DIR=/tmp/postprocess_missing_training \
INCREMENTAL_DIR=/tmp/postprocess_missing_incremental \
bash scripts/postprocess_publication_results.sh
```

Confirmed output includes:

```text
[postprocess] strict publication gate skipped because ALLOW_MISSING=1
[postprocess] done
```

Ran:

```bash
bash -n scripts/postprocess_publication_results.sh
python -m py_compile scripts/audit_publication_gates.py
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
