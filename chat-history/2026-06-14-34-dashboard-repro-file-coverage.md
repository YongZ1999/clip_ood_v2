# Dashboard Reproducibility File Coverage

**Date**: 2026-06-14

## Context

The local verifier and remote upload package now depend on several launchers,
summarizers, audits, self-tests, the remote helper, the result ledger, and the
publication README. The publication dashboard's reproducibility-file gate still
checked only the earliest small subset of scripts. This could let the dashboard
report reproducibility as passing even if a newer self-test or helper script was
missing.

## Changes

Updated:

```text
scripts/audit_publication_gates.py
experiments/README_publication_repro.md
```

The dashboard reproducibility gate now checks presence of:

- classifier replay launcher;
- classifier-side training launcher;
- strict incremental launcher;
- classifier and incremental summarizers;
- paper/protocol/result/training/incremental audit scripts;
- postprocess script;
- remote helper;
- publication audit self-test;
- publication launcher self-test;
- remote package self-test;
- package verifier;
- publication README;
- result ledger.

The README now states that the dashboard's reproducibility-file gate covers
launchers, summarizers, audits, self-tests, the remote helper, the README, and
the result ledger.

## Verification

Ran:

```bash
python -m py_compile scripts/audit_publication_gates.py
python scripts/audit_publication_gates.py
bash scripts/verify_publication_package.sh
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
