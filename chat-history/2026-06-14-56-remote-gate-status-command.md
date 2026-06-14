# Remote Gate Status Command

**Date**: 2026-06-14

## Context

The publication goal remains active. The final postprocess path now writes
publication gate artifacts, but the remote helper status commands only exposed
training and incremental job status. After a remote postprocess run, checking the
final gate artifact still required manually logging into the server.

## Changes

Updated:

```text
scripts/remote_training_ablation.sh
scripts/selftest_remote_package.py
experiments/README_publication_repro.md
```

Added:

```bash
bash scripts/remote_training_ablation.sh status_gates
```

The command prints the remote `experiments/publication_gate_status.json` fields:

```text
strict
summary
generated_at_utc
```

`status_all` now runs `status_gates` after classifier-side training status and
strict incremental status.

The remote package self-test now checks that the helper keeps `status_gates`,
the gate artifact path, and the `generated_at_utc` field query.

The README now documents `status_gates` and notes that `status_all` includes it.

## Verification

Ran:

```bash
bash -n scripts/remote_training_ablation.sh
python scripts/selftest_remote_package.py
bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
bash scripts/remote_training_ablation.sh status_gates
```

Result:

```text
REMOTE PACKAGE SELFTEST PASSED
publication package checks passed
Summary: PASS=5, PENDING=2
```

Remote status still failed from this local environment:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

## Next Action

When SSH access works, run:

```bash
bash scripts/remote_training_ablation.sh upload
bash scripts/remote_training_ablation.sh launch_all
bash scripts/remote_training_ablation.sh status_all
bash scripts/remote_training_ablation.sh postprocess
bash scripts/remote_training_ablation.sh status_gates
```

The goal remains incomplete until strict publication gates pass with real
training-side and incremental result summaries and the final status artifact
records `strict: true`.
