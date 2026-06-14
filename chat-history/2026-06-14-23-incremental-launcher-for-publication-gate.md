# Strict Incremental Launcher for Publication Gate

**Date**: 2026-06-14

## Context

The publication dashboard had two remaining pending gates:

- training-side LoRA-NSP ablation;
- strict incremental Transfer/Average/Last/forgetting.

The classifier-side training ablation already had a launcher
(`scripts/run_joint_training_ablation.sh`), but the strict incremental gate only
had summarization/audit scripts. It still needed a dedicated launcher that
generates `*_results.json` files with `accuracy_matrix` for the four LoRA
configurations and three seeds.

## Changes

Added:

```text
scripts/run_joint_incremental_ablation.sh
```

The script runs:

- `lora_vanilla`
- `lora_nsp`
- `lora_nsp_fd`
- `lora_nsp_fd_cd`

with seeds `42 43 44` over the full 10-task X-TAIL sequence. It calls
`src/experiments/run_continual_learning.py`, stores each run under
`${OUT_DIR}/runs/${experiment}_seed${seed}`, and copies a uniquely named
`${experiment}_seed${seed}_results.json` into `${OUT_DIR}` for
`scripts/summarize_incremental_metrics.py`.

Updated:

```text
src/experiments/run_continual_learning.py
scripts/summarize_incremental_metrics.py
scripts/remote_training_ablation.sh
scripts/audit_publication_gates.py
scripts/verify_publication_package.sh
experiments/README_publication_repro.md
```

Key behavior:

- `run_continual_learning.py` now accepts `--experiment_name` and uses it as the
  final result stem.
- `summarize_incremental_metrics.py` now prefers `args.experiment_name` before
  `args.method`, so FD/CD variants are not collapsed into a single `lora_nsp`
  aggregate.
- `remote_training_ablation.sh` now supports:

```bash
bash scripts/remote_training_ablation.sh launch_incremental
bash scripts/remote_training_ablation.sh status_incremental
```

and uploads the new launcher plus the modified incremental entry point.

## Verification

Ran:

```bash
DRY_RUN=1 \
OUT_DIR=/tmp/joint_incremental_ablation_dry_run \
GPUS="0 1 2 3" \
bash scripts/run_joint_incremental_ablation.sh

python -m py_compile \
  src/experiments/run_continual_learning.py \
  scripts/summarize_incremental_metrics.py \
  scripts/audit_publication_gates.py

bash -n scripts/run_joint_incremental_ablation.sh
bash -n scripts/remote_training_ablation.sh
bash -n scripts/verify_publication_package.sh

bash scripts/verify_publication_package.sh
```

The dry run produced the expected command pattern, including:

```bash
python src/experiments/run_continual_learning.py \
  --root /data1/open_datasets/X-TAIL \
  --task_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
  --num_shots 16 \
  --iterations 800 \
  --batch_size 64 \
  --method lora_nsp \
  --fd_weight 1.0 \
  --cd_weight 1.0 \
  --seed 42 \
  --experiment_name lora_nsp_fd_cd
```

`verify_publication_package.sh` still reports:

```text
Summary: PASS=5, PENDING=2
publication package checks passed
```

The two pending gates remain pending because the real remote training and
incremental result files have not been generated yet. This change only completes
the local execution path for the strict incremental gate.

## Next Commands When SSH Recovers

```bash
bash scripts/remote_training_ablation.sh upload
bash scripts/remote_training_ablation.sh launch
bash scripts/remote_training_ablation.sh launch_incremental
bash scripts/remote_training_ablation.sh status
bash scripts/remote_training_ablation.sh status_incremental
bash scripts/remote_training_ablation.sh postprocess
```
