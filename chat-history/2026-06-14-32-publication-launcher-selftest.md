# Publication Launcher Self-Test

**Date**: 2026-06-14

## Context

The publication verifier already checked audit behavior, but it did not execute
the launchers in dry-run mode. A string-based protocol audit can catch many
issues, but it does not prove the shell launchers expand to commands with the
required publication arguments.

## Changes

Added:

```text
scripts/selftest_publication_launchers.py
```

Updated:

```text
scripts/verify_publication_package.sh
scripts/remote_training_ablation.sh
experiments/README_publication_repro.md
```

The launcher self-test runs dry-run versions of:

- `scripts/run_joint_classifier_replay.sh`
- `scripts/run_joint_training_ablation.sh`
- `scripts/run_joint_incremental_ablation.sh`

It checks command logs for required protocol arguments, including:

- classifier replay: `--classifier_feature_transform test`, LADA settings,
  `gmm_k=4`, `gaussian_samples_per_class=16`, and GMM mean replay flags;
- classifier-side training ablation: `--tune_vision_encoder true`, LADA/LR-RGDA
  settings, `lora_nsp_fd_cd`, `fd_weight=1.0`, `cd_weight=1.0`;
- strict incremental ablation: full 10-task X-TAIL sequence, `--num_shots 16`,
  `--eval_max_samples 0`, `lora_nsp_fd_cd`, `fd_weight=1.0`, `cd_weight=1.0`.

The verifier now includes:

```text
[verify] publication launcher selftest
PUBLICATION LAUNCHER SELFTEST PASSED
```

The remote upload helper now includes the launcher self-test so remote
postprocessing verifies the same launcher contracts after upload.

## Verification

Ran:

```bash
python scripts/selftest_publication_launchers.py
python -m py_compile scripts/selftest_publication_launchers.py
bash -n scripts/verify_publication_package.sh
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
