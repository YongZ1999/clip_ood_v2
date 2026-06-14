# Remote OUT_DIR Forwarding

**Date**: 2026-06-14

## Context

The remote helper documentation says `OUT_DIR` can be overridden. The `status`
and `postprocess` commands already used the helper-level `OUT_DIR`, but
`launch` did not forward it into `scripts/run_joint_training_ablation.sh`. This
could make a custom training launch write to the default directory while status
and postprocess looked in the overridden directory.

## Changes

Updated:

```text
scripts/remote_training_ablation.sh
scripts/selftest_remote_package.py
experiments/README_publication_repro.md
```

Behavior now:

```bash
OUT_DIR=experiments/custom_training \
bash scripts/remote_training_ablation.sh launch
```

launches the remote training script with:

```bash
OUT_DIR='experiments/custom_training' nohup scripts/run_joint_training_ablation.sh ...
```

`scripts/selftest_remote_package.py` now checks that the remote helper forwards:

- `OUT_DIR` into the training launcher;
- `INCREMENTAL_DIR` as `OUT_DIR` into the incremental launcher;
- both directories into postprocess.

The README now states that training launch, status, and postprocess all use the
same `OUT_DIR`.

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
