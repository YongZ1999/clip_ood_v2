# Launcher Grid Self-Test

**Date**: 2026-06-14

## Context

The publication goal remains active. The launcher self-test previously checked
representative dry-run commands for the replay, training, and strict incremental
launchers, but it did not verify the full expected configuration grid.

That left a local reproducibility gap: a launcher could accidentally drop or
mislabel one configuration while the representative `lora_nsp_fd_cd` command
still looked correct.

## Changes

Updated:

```text
scripts/selftest_publication_launchers.py
experiments/README_publication_repro.md
```

The launcher self-test now checks that:

- the classifier replay dry-run produces real, raw-GMM sample, sphere-GMM sample,
  and raw-GMM mean replay logs;
- classifier replay manifests retain the test transform, replay modes, LADA,
  LR-RGDA, and GMM settings;
- training dry-runs produce logs for all four configurations:
  `lora_vanilla`, `lora_nsp`, `lora_nsp_fd`, and `lora_nsp_fd_cd`;
- training manifests and logs preserve each configuration's LoRA type, FD weight,
  CD weight, and experiment name;
- strict incremental dry-runs produce logs for all four configurations and
  preserve the expected task sequence, full-test setting, method, FD/CD weights,
  and experiment names.

The README now documents that launcher self-tests cover the full dry-run grid and
manifest arguments, not only a single representative command.

## Verification

Ran:

```bash
python scripts/selftest_publication_launchers.py
python -m py_compile scripts/selftest_publication_launchers.py
bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
PUBLICATION LAUNCHER SELFTEST PASSED
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
```

The goal remains incomplete until strict publication gates pass with real
training-side and incremental result summaries.
