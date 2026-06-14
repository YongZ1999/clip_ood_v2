# Text Schedule Current Progress and Best Settings

**Date**: 2026-06-14

## Purpose

This entry summarizes the current progress and the best settings identified so
far for the training-side CLIP continual learning path:

```text
LoRA-NSP + vision/text tuning + FD/CD
```

It consolidates the discussion and evidence from:

```text
chat-history/2026-06-14-60-incremental-training-pilot-results.md
chat-history/2026-06-14-61-text-tuning-schedule-goal.md
chat-history/2026-06-14-62-text-tuning-schedule-pilot-results.md
chat-history/2026-06-14-63-text-tuning-schedule-stress-results.md
```

## High-Level Research State

The project currently has two complementary innovation layers:

```text
1. Training-side innovation:
   LoRA-NSP + vision/text continual adaptation + FD/CD + text LR schedule

2. Inference-side innovation:
   LR-RGDA + zero-shot ensemble classifier under compact statistical replay
```

The current text schedule work belongs to the training-side layer. It should be
evaluated first under strict incremental learning before stacking LR-RGDA + ZS
on top.

## Strongest Training-Side Baseline Before Text Schedule

Earlier 2-task pilot:

```text
task_sequence = aircraft caltech101
seed = 42
num_shots = 16
iterations = 800
batch_size = 64
eval_max_samples = 0
```

Best baseline from that pilot:

| method | Transfer | Average | Last |
|---|---:|---:|---:|
| lora_nsp_fd_cd_vision_text | 89.29 | 72.01 | 73.49 |

Key conclusion:

```text
vision + text tuning is strongly positive;
LoRA-NSP remains useful when text tuning is enabled;
FD/CD helps on top of LoRA-NSP with vision/text tuning.
```

Therefore all text schedule experiments should use:

```text
method = lora_nsp
tune_vision_encoder = true
tune_text_encoder = true
fd_weight = 1.0
cd_weight = 1.0
```

## Why Text Schedule Was Studied

The central question was:

```text
Should the CLIP text encoder continue to be updated at the same learning rate
for every incremental task, or should text-side updates be reduced/frozen after
the first task?
```

Motivation:

```text
The first task may adapt the text encoder to the downstream few-shot,
zero-shot-style classification format. After this adaptation, a stable or
slowly changing text encoder may act as a semantic anchor and reduce forgetting.
```

Important interpretation:

```text
This is not about whether text tuning is useful at all. Text tuning is useful.
The real question is how much text plasticity should remain after task 1.
```

## Implemented Schedule Controls

Implemented CLI:

```text
--text_tuning_schedule always|never|freeze_after|low_lr_after
--text_schedule_switch_task 1
--text_lr_scale_after_task 0.2
```

Semantics:

```text
always:
  every task trains text LoRA with base text LR

never:
  no text LoRA is attached/trained

freeze_after:
  tasks <= switch_task train text LoRA
  tasks > switch_task freeze the already-adapted/merged text encoder

low_lr_after:
  tasks <= switch_task train text LoRA with base text LR
  tasks > switch_task train text LoRA with base_text_lr * scale
```

Correctness detail:

```text
freeze_after preserves task-1 merged text adaptation.
It does not revert to the original CLIP text encoder.
```

Trainer behavior:

```text
vision LoRA and text LoRA use separate optimizer parameter groups;
frozen text LoRA is excluded from optimizer;
frozen/adapted text encoder is still used as the no-grad semantic anchor in CD;
per-task schedule metadata is saved in result JSON under text_schedule.
```

## Stage 1: 2-Task Text Schedule Pilot

Protocol:

```text
task_sequence = aircraft caltech101
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

Aggregate:

| method | Transfer | Average | Last |
|---|---:|---:|---:|
| text_task1_lr_1_5 | 91.97 | 71.40 | 72.22 |
| text_task1_freeze | 90.71 | 70.92 | 72.00 |
| text_always | 91.56 | 70.86 | 71.73 |
| text_task1_lr_1_10 | 90.91 | 70.57 | 71.61 |
| vision_only_fd_cd | 86.94 | 65.26 | 67.24 |

Stage 1 reading:

```text
LR/5 after task 1 was best on Transfer, Average, and Last.
Freeze-after-task1 improved Last over always but lost Transfer.
LR/10 was worse than LR/5, suggesting later text adaptation remains useful.
Vision-only FD/CD was clearly weaker, confirming that text tuning itself is
important.
```

Stage 1 best setting:

```text
text_task1_lr_1_5:
  task 1 text_lr = 1e-4
  task 2 text_lr = 2e-5
```

## Stage 2: 4-Task Stress Test

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

Configs:

```text
text_always:
  all tasks text_lr = 1e-4

text_task1_lr_1_5:
  task 1 text_lr = 1e-4
  tasks 2..K text_lr = 2e-5

text_task1_freeze:
  task 1 text_lr = 1e-4
  tasks 2..K text frozen
```

Aggregate:

| method | seed | K | Transfer | Average | Last |
|---|---:|---:|---:|---:|---:|
| text_always | 42 | 4 | 58.19 | 62.73 | 76.07 |
| text_task1_freeze | 42 | 4 | 56.53 | 61.57 | 75.30 |
| text_task1_lr_1_5 | 42 | 4 | 57.68 | 62.90 | 77.01 |

Main comparison:

```text
text_task1_lr_1_5 - text_always:
  Transfer -0.51
  Average  +0.17
  Last     +0.94
```

Final-row comparison:

```text
text_task1_lr_1_5 - text_always:
  aircraft   +3.42
  caltech101 +0.16
  dtd        -0.71
  eurosat    +0.88
```

Freeze comparison:

```text
text_task1_freeze - text_always:
  Transfer -1.66
  Average  -1.16
  Last     -0.77
```

Stage 2 reading:

```text
LR/5 after task 1 is best on Last and Average.
Always-tune is best on Transfer.
Freeze-after-task1 is not competitive.
```

Important interpretation:

```text
LR/5 is not maximizing pre-training transfer to future tasks.
Its value is better final continual performance after all tasks are learned.
It improves final old-task retention, especially aircraft, without materially
hurting later-task adaptation.
```

## Current Best Training-Side Setting

The current best schedule to carry into the strict full gate is:

```text
name = text_task1_lr_1_5

method = lora_nsp
tune_vision_encoder = true
tune_text_encoder = true
fd_weight = 1.0
cd_weight = 1.0
text_tuning_schedule = low_lr_after
text_schedule_switch_task = 1
text_lr_scale_after_task = 0.2

task 1 text_lr = 1e-4
tasks 2..K text_lr = 2e-5
```

The minimal companion baseline is:

```text
name = text_always

method = lora_nsp
tune_vision_encoder = true
tune_text_encoder = true
fd_weight = 1.0
cd_weight = 1.0
text_tuning_schedule = always

all tasks text_lr = 1e-4
```

Do not use `text_task1_freeze` as a full-gate primary method:

```text
It helps aircraft retention but hurts aggregate metrics and dtd adaptation.
It can remain an analysis-only result.
```

## Full-Gate Launcher Prepared

Added:

```text
scripts/run_incremental_text_schedule_full_gate.sh
```

Remote helper supports:

```bash
bash scripts/remote_training_ablation.sh launch_text_schedule_full_gate
bash scripts/remote_training_ablation.sh status_text_schedule_full_gate
```

Default full-gate protocol:

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

Recommended launch:

```bash
TEXT_FULL_GATE_DIR=experiments/incremental_text_schedule_full_gate_20260614 \
TEXT_FULL_GATE_LOG=experiments/incremental_text_schedule_full_gate_20260614_master.log \
bash scripts/remote_training_ablation.sh launch_text_schedule_full_gate
```

Status:

```bash
bash scripts/remote_training_ablation.sh status_text_schedule_full_gate
```

The launcher was uploaded to the remote server and the remote helper was
verified to contain the full-gate entry points.

## Verification Status

Local verification passed:

```bash
bash scripts/verify_publication_package.sh
```

Result:

```text
publication package checks passed
PASS=5, PENDING=2
```

The two pending items are existing publication gates:

```text
Training-side LoRA-NSP ablation
Strict incremental Transfer/Average/Last/forgetting
```

They are not failures introduced by the text schedule work.

## Current Claim Boundary

What we can say now:

```text
On the 2-task pilot and 4-task stress test, the LR/5 text schedule is the best
candidate for final continual performance. It improves Last over always-tune
and avoids the under-adaptation problem of freezing the text encoder after
task 1.
```

What we cannot claim yet:

```text
We cannot claim paper-level superiority over LADA or over always-tune until the
strict 10-task, 3-seed full gate completes.
```

Next decisive evidence:

```text
Run full gate:
  text_task1_lr_1_5 vs text_always

Protocol:
  10 X-TAIL tasks
  seeds 42,43,44
  full test split
  Transfer / Average / Last
```

