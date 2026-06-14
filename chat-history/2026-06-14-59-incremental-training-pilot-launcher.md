# Incremental Training Pilot Launcher

**Date**: 2026-06-14

## Context

After fixing the strict incremental runner to use `LoRANSPTrainer`, the next
step is to test whether text encoder tuning and LoRA-NSP are positive in a
small LADA-style incremental setting before launching the full 10-task,
3-seed gate.

The goal of this step is not to prove the final paper claim. It is a fast
decision gate:

```text
Should we spend the full remote budget on vision+text LoRA-NSP?
```

## Added Launcher

Added:

```text
scripts/run_incremental_training_pilot.sh
```

Default pilot protocol:

```text
tasks = aircraft caltech101
seeds = 42
num_shots = 16
iterations = 800
batch_size = 64
eval_max_samples = 0
full test split = yes
```

The pilot runs five configurations:

```text
lora_vanilla_vision
lora_nsp_vision
lora_vanilla_vision_text
lora_nsp_vision_text
lora_nsp_fd_cd_vision_text
```

Configuration meanings:

```text
lora_vanilla_vision:
  method=lora_vanilla, tune_vision=true, tune_text=false, fd=0, cd=0

lora_nsp_vision:
  method=lora_nsp, tune_vision=true, tune_text=false, fd=0, cd=0

lora_vanilla_vision_text:
  method=lora_vanilla, tune_vision=true, tune_text=true, fd=0, cd=0

lora_nsp_vision_text:
  method=lora_nsp, tune_vision=true, tune_text=true, fd=0, cd=0

lora_nsp_fd_cd_vision_text:
  method=lora_nsp, tune_vision=true, tune_text=true, fd=1, cd=1
```

Outputs:

```text
experiments/incremental_training_pilot_20260614/
  manifest.txt
  logs/
  runs/
  *_results.json
  incremental_summary.csv
  incremental_summary.md
  incremental_aggregate.csv
  incremental_aggregate.md
```

The launcher reuses:

```text
src/experiments/run_continual_learning.py
scripts/summarize_incremental_metrics.py
```

so the pilot result schema matches the strict incremental summarizer.

## Remote Helper

Updated:

```text
scripts/remote_training_ablation.sh
scripts/selftest_remote_package.py
```

New remote commands:

```bash
bash scripts/remote_training_ablation.sh upload
bash scripts/remote_training_ablation.sh launch_incremental_pilot
bash scripts/remote_training_ablation.sh status_incremental_pilot
```

Remote defaults:

```text
PILOT_DIR = experiments/incremental_training_pilot_20260614
PILOT_LOG = experiments/incremental_training_pilot_20260614_master.log
```

The upload package now includes:

```text
scripts/run_incremental_training_pilot.sh
```

## Verification

Ran:

```bash
bash -n \
  scripts/run_incremental_training_pilot.sh \
  scripts/remote_training_ablation.sh \
  scripts/run_joint_incremental_ablation.sh

python -m py_compile \
  scripts/selftest_publication_launchers.py \
  scripts/selftest_remote_package.py \
  src/experiments/run_continual_learning.py

DRY_RUN=1 OUT_DIR=/private/tmp/incremental_training_pilot_dry_run_20260614 \
  SEEDS='42' GPUS='0 1 2 3 4' \
  bash scripts/run_incremental_training_pilot.sh

python scripts/selftest_publication_launchers.py
python scripts/selftest_remote_package.py
bash scripts/verify_publication_package.sh
```

Results:

```text
PUBLICATION LAUNCHER SELFTEST PASSED
REMOTE PACKAGE SELFTEST PASSED
publication package checks passed
```

The full verifier remains:

```text
PASS=5, PENDING=2
```

The pending gates require real remote result files and are expected.

## Dry-Run Command Checks

Vision-only NSP expands to:

```text
python src/experiments/run_continual_learning.py
  --task_sequence aircraft caltech101
  --eval_max_samples 0
  --method lora_nsp
  --fd_weight 0.0
  --cd_weight 0.0
  --tune_vision_encoder true
  --tune_text_encoder false
  --experiment_name lora_nsp_vision
```

Vision+text NSP+FD+CD expands to:

```text
python src/experiments/run_continual_learning.py
  --task_sequence aircraft caltech101
  --eval_max_samples 0
  --method lora_nsp
  --fd_weight 1.0
  --cd_weight 1.0
  --tune_vision_encoder true
  --tune_text_encoder true
  --experiment_name lora_nsp_fd_cd_vision_text
```

## How To Interpret The Pilot

Primary comparisons:

```text
lora_nsp_vision - lora_vanilla_vision
  Tests whether NSP helps under vision-only incremental tuning.

lora_vanilla_vision_text - lora_vanilla_vision
  Tests whether text tuning helps even without NSP.

lora_nsp_vision_text - lora_vanilla_vision_text
  Tests whether NSP still helps when text tuning is enabled.

lora_nsp_fd_cd_vision_text - lora_nsp_vision_text
  Tests whether FD/CD add value after vision+text NSP is active.
```

Decision rule:

```text
If vision+text NSP improves Average/Last without a large Transfer collapse,
run the full 10-task 3-seed strict incremental gate with vision+text enabled.

If vision+text helps but FD/CD hurts, keep the text-tuning result and drop or
retune FD/CD before the full run.

If vision+text is unstable or worse than vision-only on the pilot, run the full
gate conservatively with vision-only NSP first, then revisit text tuning.
```

This pilot still evaluates the training-side incremental contribution only.
It does not yet test the best LR-RGDA+ZS inference classifier inside the
incremental protocol.
