# Remote Launch-All Helper

**Date**: 2026-06-14

## Context

Two publication gates still require remote results:

- classifier-side LoRA-NSP training ablation;
- strict incremental Transfer/Average/Last/forgetting.

The remote helper previously required separate `launch` and `launch_incremental`
calls. That is workable, but easy to partially execute and then forget one of
the pending gates. A combined command was added for the common publication path.

## Changes

Updated:

```text
scripts/remote_training_ablation.sh
scripts/selftest_remote_package.py
experiments/README_publication_repro.md
```

New commands:

```bash
bash scripts/remote_training_ablation.sh launch_all
bash scripts/remote_training_ablation.sh status_all
```

Behavior:

- `launch_all` runs the classifier-side training launch and then the strict
  incremental launch.
- `status_all` prints classifier-side training status and strict incremental
  status.

`scripts/selftest_remote_package.py` now checks that the remote helper exposes
these combined commands, in addition to checking upload files and environment
forwarding.

The README now documents the combined command sequence:

```bash
bash scripts/remote_training_ablation.sh upload
bash scripts/remote_training_ablation.sh launch_all
bash scripts/remote_training_ablation.sh status_all
```

## Verification

Ran:

```bash
python scripts/selftest_remote_package.py
bash -n scripts/remote_training_ablation.sh
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
