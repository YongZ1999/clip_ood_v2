# Remote Package Self-Test

**Date**: 2026-06-14

## Context

The remote helper uploads a selected file list before launching or
postprocessing publication experiments. If that list misses a verifier,
postprocess, source, paper, README, or ledger dependency, the failure may only
appear after SSH recovers. A local self-test was added to catch upload-package
drift before remote execution.

## Changes

Added:

```text
scripts/selftest_remote_package.py
```

Updated:

```text
scripts/remote_training_ablation.sh
scripts/verify_publication_package.sh
experiments/README_publication_repro.md
```

The self-test parses the `FILES=(...)` array in
`scripts/remote_training_ablation.sh` and verifies that the remote upload package
contains every required local dependency for publication postprocessing and
verification, including:

- launchers and postprocess scripts;
- all audit and self-test scripts;
- `main_joint.py`;
- strict incremental entry point;
- locally modified classifier/model/trainer files;
- paper draft;
- reproducibility README;
- result ledger.

It also verifies that each required upload file exists on disk.

The verifier now includes:

```text
[verify] remote package selftest
REMOTE PACKAGE SELFTEST PASSED
```

## Verification

Ran:

```bash
python scripts/selftest_remote_package.py
python -m py_compile scripts/selftest_remote_package.py
bash -n scripts/remote_training_ablation.sh
bash -n scripts/verify_publication_package.sh
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
