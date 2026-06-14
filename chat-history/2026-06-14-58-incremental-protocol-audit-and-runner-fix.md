# Incremental Protocol Audit and Runner Fix

**Date**: 2026-06-14

## Context

After confirming the best inference-side classifier recipe
(`LR-RGDA + multi-centroid + classifier fine-tuning + test_transform +
GMM component-mean replay + alpha=0.05`), the next question is whether the full
method can beat LADA under the LADA-style incremental protocol:

```text
X-TAIL 10 tasks
16-shot
Transfer / Average / Last / Forgetting
multiple seeds
```

The project has two intended innovation layers:

```text
Training side: LoRA-NSP, now including possible vision+text encoder tuning.
Inference side: LR-RGDA+ZS statistical replay classifier.
```

These should be validated separately before claiming that their benefits stack.

## Audit Findings

### 1. Inference-side evidence is strong but not incremental evidence

The current best classifier evidence comes from joint classifier replay:

```text
LR-RGDA+ZS alpha=0.05: 78.44 +/- 0.21
LADA+ZS alpha=0.05:    77.03 +/- 0.38
LADA:                  76.08 +/- 0.35
```

This supports a compact statistical replay claim, but it does not yet satisfy
the LADA incremental metric protocol.

### 2. `main_incremental.py` is not currently a reliable publication entry

`main_incremental.py` has the conceptual shape of the desired LR-RGDA
incremental evaluation, but it has significant drift:

```text
uses old single-center build_stats_dict_from_features
uses train-transform classifier construction
does not expose the best classifier recipe
does not expose GMM mean replay
does not expose LR-RGDA classifier fine-tuning
does not expose LADA comparison/fair LADA+ZS alpha
stores a schema that is not consumed by summarize_incremental_metrics.py
```

It also contains a visible indentation error around the global class-name
collection region. The file still passes `py_compile`, but the main incremental
logic is not a safe basis for the current publication gate.

### 3. `src/experiments/run_continual_learning.py` was the strict incremental gate entry

The publication launcher uses:

```text
scripts/run_joint_incremental_ablation.sh
  -> src/experiments/run_continual_learning.py
```

Before this session, that runner used `src.models.trainer.Trainer`. This meant
the strict incremental gate was not using the full `LoRANSPTrainer` task-finalize
path:

```text
train current task
extract current task covariances
merge LoRA weights into the base model
update vision/text covariance history
update projection matrices for the next task
```

Therefore it was insufficient as a strict LoRA-NSP incremental evaluation entry.

### 4. `main_incremental_lada.py` is the closest LADA/DPT research entry

`main_incremental_lada.py` includes LADA, DPT replay modes, and independent
LR-RGDA/LR-RGDA+ZS metrics. However, it still needs cleanup before being used as
the final comparison:

```text
LADA+ZS evaluation still uses the older mask-gated fusion path.
LR-RGDA still uses old single-center/no-fit classifier construction.
It does not emit the publication summarizer schema.
It is useful for LADA/DPT mechanics, not yet the final LR-RGDA+ZS best recipe.
```

## Technical Decision

Use `run_continual_learning.py` for the first incremental gate:

```text
Does LoRA-NSP training, especially vision+text tuning, improve LADA-style
incremental Transfer/Average/Last/forgetting?
```

Do not yet mix in LR-RGDA+ZS. The classifier-side incremental stacking should be
a second-stage entry after the training-side best configuration is identified.

## Changes Made

Updated:

```text
src/experiments/run_continual_learning.py
scripts/run_joint_incremental_ablation.sh
scripts/audit_protocol_config.py
scripts/selftest_publication_launchers.py
```

### `run_continual_learning.py`

Changed the incremental runner to use:

```text
LoRANSPTrainer
```

instead of:

```text
Trainer
```

The runner now:

```text
exposes --tune_vision_encoder true/false
exposes --tune_text_encoder true/false
exposes --text_lora_rank
exposes --max_zs_classes
exposes --aux_weight, --sce_a, --sce_b
loads the distillation reference loader once per run, not once per task
stores all task accuracies as fractions in [0, 1]
uses the deterministic update_loader for NSP covariance extraction
calls trainer.finalize_task_for_incremental()
updates vision covariance history when vision LoRA is enabled
updates text covariance history when text LoRA is enabled
```

For `lora_nsp`, each task now follows the intended LoRA-NSP sequence:

```text
train task
extract vision/text covariances
merge and reset LoRA modules
update projection matrices for the next task
evaluate all tasks
```

For `lora_vanilla` and `lora_sgp`, the runner still merges/resets the LoRA
modules after each task, but does not update NSP covariance history.

### `run_joint_incremental_ablation.sh`

Added explicit environment controls:

```text
TUNE_VISION_ENCODER=true
TUNE_TEXT_ENCODER=true
```

These are passed to `run_continual_learning.py` and recorded in the manifest.
This makes vision-only vs vision+text runs auditable.

Example override for a vision-only pilot:

```bash
TUNE_TEXT_ENCODER=false bash scripts/run_joint_incremental_ablation.sh
```

## Verification

Ran:

```bash
python -m py_compile \
  src/experiments/run_continual_learning.py \
  scripts/selftest_publication_launchers.py \
  scripts/audit_protocol_config.py

bash -n scripts/run_joint_incremental_ablation.sh

DRY_RUN=1 OUT_DIR=/private/tmp/incremental_lora_nsp_trainer_dry_run_20260614 \
  SEEDS='42' GPUS='0 1 2 3' EXPECTED_TASKS='aircraft caltech101' \
  bash scripts/run_joint_incremental_ablation.sh

python scripts/selftest_publication_launchers.py
python scripts/audit_protocol_config.py
bash scripts/verify_publication_package.sh
```

Results:

```text
PUBLICATION LAUNCHER SELFTEST PASSED
PROTOCOL CONFIG AUDIT PASSED
publication package checks passed
```

The full verifier status remains:

```text
PASS=5, PENDING=2
```

The two pending gates are expected:

```text
Training-side LoRA-NSP ablation
Strict incremental Transfer/Average/Last/forgetting
```

They require real remote result files, not more local wiring.

## Recommended Experiment Plan

### Phase 1: Training-side incremental pilot

Run a small two-task pilot first:

```text
aircraft -> caltech101
seed 42
full test split
```

Compare at least:

```text
lora_vanilla, vision-only
lora_nsp, vision-only
lora_vanilla, vision+text
lora_nsp, vision+text
lora_nsp_fd_cd, vision+text
```

Purpose:

```text
Check whether text tuning and NSP are actually positive under LADA-style
incremental metrics before launching the full 10-task 3-seed grid.
```

### Phase 2: Full training-side gate

If Phase 1 is positive, run the existing strict launcher with:

```text
10 X-TAIL tasks
seeds 42,43,44
full test split
vision+text tuning
```

This answers whether LoRA-NSP improves incremental Transfer/Average/Last.

### Phase 3: Incremental classifier stacking

Only after identifying the best training-side model, add a second incremental
evaluation entry for:

```text
ZS
LR-RGDA
LR-RGDA+ZS alpha=0.05
LADA
LADA+ZS alpha=0.05
```

The LR-RGDA side should use the confirmed best classifier components:

```text
classifier_feature_transform = test
num_centers = 4
rgda_train_iter = 200
alpha = 0.05
current task real deterministic features
old task compact statistics / GMM component means
```

This phase should not reuse `main_incremental.py` as-is.

## Claim Boundary

After this session, the valid claim is only:

```text
The strict incremental training runner now uses the intended LoRA-NSP
task-finalization and projection-update path, and exposes auditable vision/text
tuning controls.
```

Still not established:

```text
LoRA-NSP beats LADA on incremental metrics.
Vision+text tuning beats vision-only under the full protocol.
LR-RGDA+ZS best classifier improves incremental Average/Last.
The combined method is publication-complete.
```
