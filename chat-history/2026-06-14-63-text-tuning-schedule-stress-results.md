# Text Tuning Schedule Stress Test Results

**Date**: 2026-06-14

## Context

This entry records Stage 2 of:

```text
chat-history/2026-06-14-61-text-tuning-schedule-goal.md
```

The purpose was to stress-test the Stage 1 text tuning schedule signal on a
longer 4-task incremental sequence before launching the strict 10-task,
3-seed full gate.

Fixed training-side baseline:

```text
LoRA-NSP + tune vision encoder + tune text encoder + FD/CD
```

## Protocol

Remote project:

```text
/home/raoxuan/projects/project_clip_continual_learning
```

Remote output:

```text
experiments/incremental_text_schedule_stress_20260614
```

Launcher:

```bash
bash scripts/remote_training_ablation.sh launch_text_schedule_stress
```

Protocol:

```text
task_sequence = aircraft caltech101 dtd eurosat
seed = 42
num_shots = 16
iterations = 800
batch_size = 64
eval_max_samples = 0
method = lora_nsp
tune_vision_encoder = true
fd_weight = 1.0
cd_weight = 1.0
```

Thread limits:

```text
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
```

Configurations:

```text
text_always:
  all tasks train text LoRA with text_lr = 1e-4

text_task1_lr_1_5:
  task 1 trains text LoRA with text_lr = 1e-4
  tasks 2..K train text LoRA with text_lr = 2e-5

text_task1_freeze:
  task 1 trains text LoRA with text_lr = 1e-4
  tasks 2..K freeze the already-adapted/merged text encoder
```

## Formal Summary

Generated remotely with:

```bash
python scripts/summarize_incremental_metrics.py \
  experiments/incremental_text_schedule_stress_20260614 \
  --output_csv experiments/incremental_text_schedule_stress_20260614/incremental_summary.csv \
  --output_markdown experiments/incremental_text_schedule_stress_20260614/incremental_summary.md \
  --aggregate_csv experiments/incremental_text_schedule_stress_20260614/incremental_aggregate.csv \
  --aggregate_markdown experiments/incremental_text_schedule_stress_20260614/incremental_aggregate.md \
  --expected_tasks "aircraft caltech101 dtd eurosat"
```

The summarizer reported:

```text
no missing expected tasks
no task order warning
no matrix warning
stored metrics matched recomputed metrics
```

Aggregate:

| method | n | seeds | K | Transfer | Average | Last |
|---|---:|---|---:|---:|---:|---:|
| text_always | 1 | 42 | 4 | 58.19 | 62.73 | 76.07 |
| text_task1_freeze | 1 | 42 | 4 | 56.53 | 61.57 | 75.30 |
| text_task1_lr_1_5 | 1 | 42 | 4 | 57.68 | 62.90 | 77.01 |

## Accuracy Matrices

Rows are after each training step. Columns are:

```text
aircraft, caltech101, dtd, eurosat
```

`text_always`:

```text
after aircraft:   [47.85, 91.12, 44.50, 36.17]
after caltech101: [45.54, 96.19, 44.92, 41.31]
after dtd:        [43.38, 96.39, 73.29, 38.77]
after eurosat:    [43.59, 96.11, 73.23, 91.37]
```

`text_task1_freeze`:

```text
after aircraft:   [48.03, 90.71, 45.51, 34.11]
after caltech101: [47.22, 95.90, 45.15, 33.60]
after dtd:        [46.68, 95.38, 68.68, 32.94]
after eurosat:    [46.38, 95.25, 68.20, 91.36]
```

`text_task1_lr_1_5`:

```text
after aircraft:   [48.03, 91.52, 45.09, 37.37]
after caltech101: [47.22, 96.75, 44.56, 39.09]
after dtd:        [46.41, 96.39, 72.28, 33.59]
after eurosat:    [47.01, 96.27, 72.52, 92.25]
```

## Per-Task Text Schedule Metadata

`text_always`:

```text
aircraft:   tune_text=true, text_lr=1e-4, scale=1.0
caltech101: tune_text=true, text_lr=1e-4, scale=1.0
dtd:        tune_text=true, text_lr=1e-4, scale=1.0
eurosat:    tune_text=true, text_lr=1e-4, scale=1.0
```

`text_task1_freeze`:

```text
aircraft:   tune_text=true,  text_lr=1e-4, scale=1.0
caltech101: tune_text=false, text_lr=0,    scale=0.0
dtd:        tune_text=false, text_lr=0,    scale=0.0
eurosat:    tune_text=false, text_lr=0,    scale=0.0
```

`text_task1_lr_1_5`:

```text
aircraft:   tune_text=true, text_lr=1e-4, scale=1.0
caltech101: tune_text=true, text_lr=2e-5, scale=0.2
dtd:        tune_text=true, text_lr=2e-5, scale=0.2
eurosat:    tune_text=true, text_lr=2e-5, scale=0.2
```

## Main Comparisons

`text_task1_lr_1_5 - text_always`:

```text
Transfer: -0.51
Average:  +0.17
Last:     +0.94
```

Final row:

```text
aircraft:   +3.42
caltech101: +0.16
dtd:        -0.71
eurosat:    +0.88
```

Interpretation:

```text
LR/5 sacrifices a small amount of Transfer but improves final retention/Last.
The final improvement is mainly aircraft retention and eurosat final accuracy.
dtd final accuracy is slightly lower than always, but the gap is below 1 point.
```

`text_task1_freeze - text_always`:

```text
Transfer: -1.66
Average:  -1.16
Last:     -0.77
```

Final row:

```text
aircraft:   +2.79
caltech101: -0.85
dtd:        -5.02
eurosat:    -0.01
```

Interpretation:

```text
Freezing improves aircraft retention, but it strongly hurts dtd adaptation and
is worse on all aggregate metrics. It should not enter the full gate as a
primary method.
```

`text_task1_lr_1_5 - text_task1_freeze`:

```text
Transfer: +1.15
Average:  +1.33
Last:     +1.71
```

Interpretation:

```text
Continuing text adaptation at low LR is clearly better than fully freezing
the text encoder after task 1.
```

## Scientific Reading

Stage 2 partially confirms the Stage 1 low-LR consolidation signal:

```text
task 1: full text LR
tasks 2..K: text LR / 5
```

The evidence is stronger for final retention than for Transfer:

```text
LR/5 is best on Last and Average.
Always is best on Transfer.
Freeze is not competitive.
```

The decisive point is the final accuracy matrix:

```text
LR/5 improves final aircraft retention substantially over always:
  47.01 vs 43.59

LR/5 also improves final eurosat:
  92.25 vs 91.37

LR/5 only slightly reduces final dtd:
  72.52 vs 73.23
```

This matches the intended use of the schedule:

```text
not maximum pre-training transfer to future tasks,
but better final continual performance after all tasks are learned.
```

## Full-Gate Recommendation

Use the LR/5 schedule as the primary full-gate training-side method:

```text
primary:
  text_task1_lr_1_5

schedule:
  task 1 text_lr = 1e-4
  tasks 2..K text_lr = 2e-5
```

Use always-tune as the minimal companion ablation:

```text
companion:
  text_always

schedule:
  all tasks text_lr = 1e-4
```

Do not include `text_task1_freeze` in the full strict gate unless there is a
separate analysis-only reason:

```text
freeze loses Transfer, Average, and Last in Stage 2;
its aircraft retention advantage does not compensate for dtd degradation.
```

## Full-Gate Launcher

Added:

```text
scripts/run_incremental_text_schedule_full_gate.sh
```

Default protocol:

```text
task_sequence = aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397
seeds = 42 43 44
num_shots = 16
iterations = 800
batch_size = 64
eval_max_samples = 0
method = lora_nsp
tune_vision_encoder = true
fd_weight = 1.0
cd_weight = 1.0
configs = text_task1_lr_1_5, text_always
```

Remote helper commands:

```bash
bash scripts/remote_training_ablation.sh upload
bash scripts/remote_training_ablation.sh launch_text_schedule_full_gate
bash scripts/remote_training_ablation.sh status_text_schedule_full_gate
```

Recommended launch:

```bash
TEXT_FULL_GATE_DIR=experiments/incremental_text_schedule_full_gate_20260614 \
TEXT_FULL_GATE_LOG=experiments/incremental_text_schedule_full_gate_20260614_master.log \
bash scripts/remote_training_ablation.sh launch_text_schedule_full_gate
```

This launcher intentionally avoids the older broad 4-config incremental grid.
At this point the clean full-gate comparison is:

```text
LR/5 schedule vs always-tune schedule
```

## Limitations

Stage 2 is still:

```text
one seed
four tasks
one task order
```

It is enough to choose the full-gate schedule, but not enough to claim a
paper-level win over LADA or over always-tune. The paper-level training claim
requires the 10-task, 3-seed full gate.

