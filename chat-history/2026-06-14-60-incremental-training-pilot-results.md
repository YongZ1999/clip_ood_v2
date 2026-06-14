# Incremental Training Pilot Results

**Date**: 2026-06-14

## Context

We ran the small LADA-style two-task incremental training pilot after fixing
`src/experiments/run_continual_learning.py` to use `LoRANSPTrainer` and task
finalization.

This pilot is a decision gate for training-side direction, not a final paper
claim. It tests whether the full strict 10-task, 3-seed incremental run should
spend compute on vision+text tuning and LoRA-NSP variants.

Remote project:

```text
/home/raoxuan/projects/project_clip_continual_learning
```

Pilot output directory:

```text
experiments/incremental_training_pilot_20260614
```

Protocol:

```text
tasks = aircraft caltech101
seed = 42
num_shots = 16
iterations = 800
batch_size = 64
eval_max_samples = 0
test split = full
```

## Remote Run Notes

The original `lora_vanilla_vision` process failed because GPU0 was occupied and
CUDA had only about 36 MiB free. This was an infrastructure/OOM issue, not a
code failure.

It was rerun manually with the same protocol after GPU0 became available:

```text
CLIP_USE_SAFETENSORS=0
CLIP_LOCAL_FILES_ONLY=1
CUDA_VISIBLE_DEVICES=0
method = lora_vanilla
tune_vision_encoder = true
tune_text_encoder = false
fd_weight = 0.0
cd_weight = 0.0
```

The rerun completed and produced:

```text
experiments/incremental_training_pilot_20260614/lora_vanilla_vision_seed42_results.json
```

## Summary Table

Generated remotely with:

```bash
python scripts/summarize_incremental_metrics.py \
  experiments/incremental_training_pilot_20260614 \
  --output_csv experiments/incremental_training_pilot_20260614/incremental_summary.csv \
  --output_markdown experiments/incremental_training_pilot_20260614/incremental_summary.md \
  --aggregate_csv experiments/incremental_training_pilot_20260614/incremental_aggregate.csv \
  --aggregate_markdown experiments/incremental_training_pilot_20260614/incremental_aggregate.md \
  --expected_tasks "aircraft caltech101"
```

Aggregate result:

| method | n | seed | K | Transfer | Average | Last |
|---|---:|---:|---:|---:|---:|---:|
| lora_nsp_fd_cd_vision_text | 1 | 42 | 2 | 89.29 | 72.01 | 73.49 |
| lora_nsp_vision | 1 | 42 | 2 | 82.88 | 65.67 | 68.73 |
| lora_nsp_vision_text | 1 | 42 | 2 | 88.84 | 70.35 | 71.34 |
| lora_vanilla_vision | 1 | 42 | 2 | 82.27 | 62.71 | 62.21 |
| lora_vanilla_vision_text | 1 | 42 | 2 | 89.33 | 69.16 | 68.25 |

The summarizer reported no missing expected tasks, no task-order warning, no
matrix warning, and stored metrics matched recomputed metrics.

## Accuracy Matrices

Rows are after each training step. Columns are `aircraft`, `caltech101`.

```text
lora_vanilla_vision:
  after aircraft:  [44.2, 82.3]
  after caltech101:[27.8, 96.6]

lora_nsp_vision:
  after aircraft:  [42.3, 82.9]
  after caltech101:[41.1, 96.3]

lora_vanilla_vision_text:
  after aircraft:  [50.8, 89.3]
  after caltech101:[40.0, 96.5]

lora_nsp_vision_text:
  after aircraft:  [49.9, 88.8]
  after caltech101:[46.6, 96.1]

lora_nsp_fd_cd_vision_text:
  after aircraft:  [51.8, 89.3]
  after caltech101:[50.0, 97.0]
```

## Primary Comparisons

Vision-only NSP vs vision-only vanilla:

```text
lora_nsp_vision - lora_vanilla_vision
Transfer: +0.61
Average:  +2.96
Last:     +6.52
```

Interpretation: under this two-task pilot, LoRA-NSP mainly improves retention
of the first task. The final aircraft accuracy after training caltech101 rises
from 27.8 to 41.1, while caltech101 remains high.

Text tuning without NSP:

```text
lora_vanilla_vision_text - lora_vanilla_vision
Transfer: +7.06
Average:  +6.45
Last:     +6.04
```

Interpretation: tuning the text encoder is strongly positive in this pilot,
even without NSP. It improves first-step aircraft, pre-training transfer to
caltech101, and final average performance.

NSP under vision+text tuning:

```text
lora_nsp_vision_text - lora_vanilla_vision_text
Transfer: -0.49
Average:  +1.20
Last:     +3.09
```

Interpretation: when text tuning is enabled, NSP still helps final retention
and Last, with a small Transfer drop.

FD/CD on top of vision+text NSP:

```text
lora_nsp_fd_cd_vision_text - lora_nsp_vision_text
Transfer: +0.45
Average:  +1.66
Last:     +2.15
```

Interpretation: in this pilot, FD/CD is beneficial rather than harmful after
vision+text NSP is active.

Best pilot configuration:

```text
lora_nsp_fd_cd_vision_text
Transfer = 89.29
Average  = 72.01
Last     = 73.49
```

## Current Scientific Reading

This pilot supports the training-side direction:

```text
vision+text LoRA-NSP + FD/CD
```

It is the best observed configuration in the two-task pilot and improves over
the conservative vision-only vanilla baseline by:

```text
Transfer: +7.02
Average:  +9.30
Last:     +11.28
```

It also improves over vision-only NSP by:

```text
Transfer: +6.41
Average:  +6.34
Last:     +4.76
```

Important boundary:

```text
This does not yet prove superiority over LADA paper metrics.
This does not yet prove the full 10-task incremental result.
This does not yet prove that LR-RGDA+ZS stacking improves incremental metrics.
```

The result is strong enough to justify a serious discussion before launching
the full 10-task, 3-seed strict incremental gate.

## Recommended Next Discussion

Before launching the full run, decide whether the full training-side gate should
include:

```text
primary config:
  lora_nsp_fd_cd_vision_text

minimal ablation companion:
  lora_nsp_vision_text

optional conservative baseline:
  lora_nsp_vision
```

My current recommendation is:

```text
Run the full strict 10-task, 3-seed gate with lora_nsp_fd_cd_vision_text as the
primary candidate, and include lora_nsp_vision_text if compute allows. Do not
launch the LR-RGDA+ZS incremental stacking experiment until the training-side
full gate identifies a stable checkpoint/config.
```

