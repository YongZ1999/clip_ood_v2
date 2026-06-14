# Remote Upload Preflight

**Date**: 2026-06-14

## Context

`scripts/selftest_remote_package.py` verifies that the remote upload file list
contains every local dependency needed by publication postprocessing and
verification. However, a user could still call `scripts/remote_training_ablation.sh
upload` directly without first running the verifier. The upload command itself
now performs this local preflight.

## Changes

Updated:

```text
scripts/remote_training_ablation.sh
experiments/README_publication_repro.md
```

Behavior now:

```bash
bash scripts/remote_training_ablation.sh upload
```

first runs:

```bash
python scripts/selftest_remote_package.py
```

before creating the tar stream or attempting SSH. If the upload file list is
stale or a required file is missing locally, upload fails before any remote
state is touched.

The README now documents this upload preflight.

## Verification

Ran:

```bash
python scripts/selftest_remote_package.py
bash -n scripts/remote_training_ablation.sh
python -m py_compile scripts/selftest_remote_package.py
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
