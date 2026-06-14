# Text Tuning Schedule Pilot Results

**Date**: 2026-06-14

## Context

This entry records execution of:

```text
chat-history/2026-06-14-61-text-tuning-schedule-goal.md
```

The goal was to implement task-wise text encoder tuning schedules and run the
Stage 1 two-task pilot on top of the current strongest training-side baseline:

```text
LoRA-NSP + tune vision + tune text + FD/CD
```

## Implemented Code Support

Updated:

```text
src/experiments/run_continual_learning.py
src/trainers/lora_nsp_trainer.py
scripts/run_incremental_text_schedule_pilot.sh
scripts/remote_training_ablation.sh
scripts/selftest_publication_launchers.py
scripts/selftest_remote_package.py
```

Added schedule CLI:

```text
--text_tuning_schedule always|never|freeze_after|low_lr_after
--text_schedule_switch_task 1
--text_lr_scale_after_task 0.2
```

Semantics:

```text
always:
  every task trains text LoRA with base LR

never:
  no text LoRA is attached/trained

freeze_after:
  tasks <= switch_task train text LoRA
  tasks > switch_task freeze the already-adapted/merged text encoder

low_lr_after:
  tasks <= switch_task train text LoRA with base LR
  tasks > switch_task train text LoRA with base_lr * scale
```

Important implementation detail:

```text
freeze_after preserves the task-1 merged text adaptation. It does not revert to
the original CLIP text encoder.
```

Trainer behavior:

```text
vision LoRA and text LoRA use separate optimizer parameter groups;
frozen text LoRA is excluded from optimizer;
frozen/adapted text encoder is still used as a no-grad semantic anchor in CD;
per-task schedule metadata is saved in result JSON under text_schedule.
```

The schedule pilot launcher also sets BLAS/OMP thread limits:

```text
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
```

This was added after the first remote run overloaded CPU linear algebra.

## Verification

Local checks passed:

```bash
python -m py_compile \
  src/experiments/run_continual_learning.py \
  src/trainers/lora_nsp_trainer.py \
  scripts/selftest_publication_launchers.py \
  scripts/selftest_remote_package.py

bash -n \
  scripts/run_incremental_text_schedule_pilot.sh \
  scripts/remote_training_ablation.sh

python scripts/selftest_publication_launchers.py
python scripts/selftest_remote_package.py
bash scripts/verify_publication_package.sh
```

Verifier result:

```text
publication package checks passed
PASS=5, PENDING=2
```

The two pending gates are the existing real training/full incremental result
files, not failures introduced by this schedule work.

## Remote Execution Notes

Remote project:

```text
/home/raoxuan/projects/project_clip_continual_learning
```

Output directory:

```text
experiments/incremental_text_schedule_pilot_20260614
```

The first launch was aborted and archived:

```text
experiments/incremental_text_schedule_pilot_20260614_aborted_20260614_192713
experiments/incremental_text_schedule_pilot_20260614_master_aborted_20260614_192713.log
```

Reason:

```text
Five parallel jobs entered projection matrix updates at the same time.
Each process used very high CPU thread counts, pushing remote load average above
700 and making progress too slow.
```

Fix:

```text
Added OMP/MKL/OPENBLAS/NUMEXPR thread limits to the launcher and relaunched.
```

The relaunched `text_always` job initially failed on GPU0 because another
process occupied GPU memory. It was rerun manually on GPU0 after it became
available, with the same protocol and thread limits.

## Stage 1 Protocol

Fixed protocol:

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

Configurations:

```text
text_always:
  task1 text_lr = 1e-4
  task2 text_lr = 1e-4

text_task1_freeze:
  task1 text_lr = 1e-4
  task2 text frozen

text_task1_lr_1_5:
  task1 text_lr = 1e-4
  task2 text_lr = 2e-5

text_task1_lr_1_10:
  task1 text_lr = 1e-4
  task2 text_lr = 1e-5

vision_only_fd_cd:
  text encoder frozen / no text LoRA
```

## Results

Generated remotely with:

```bash
python scripts/summarize_incremental_metrics.py \
  experiments/incremental_text_schedule_pilot_20260614 \
  --output_csv experiments/incremental_text_schedule_pilot_20260614/incremental_summary.csv \
  --output_markdown experiments/incremental_text_schedule_pilot_20260614/incremental_summary.md \
  --aggregate_csv experiments/incremental_text_schedule_pilot_20260614/incremental_aggregate.csv \
  --aggregate_markdown experiments/incremental_text_schedule_pilot_20260614/incremental_aggregate.md \
  --expected_tasks "aircraft caltech101"
```

Aggregate result:

| method | n | seed | K | Transfer | Average | Last |
|---|---:|---:|---:|---:|---:|---:|
| text_task1_lr_1_5 | 1 | 42 | 2 | 91.97 | 71.40 | 72.22 |
| text_task1_freeze | 1 | 42 | 2 | 90.71 | 70.92 | 72.00 |
| text_always | 1 | 42 | 2 | 91.56 | 70.86 | 71.73 |
| text_task1_lr_1_10 | 1 | 42 | 2 | 90.91 | 70.57 | 71.61 |
| vision_only_fd_cd | 1 | 42 | 2 | 86.94 | 65.26 | 67.24 |

The summarizer reported:

```text
no missing expected tasks
no task order warning
no matrix warning
stored metrics matched recomputed metrics
```

## Accuracy Matrices

Rows are after each training step. Columns are `aircraft`, `caltech101`.

```text
text_always:
  after aircraft:   [48.4, 91.6]
  after caltech101: [47.0, 96.5]

text_task1_freeze:
  after aircraft:   [49.0, 90.7]
  after caltech101: [47.9, 96.1]

text_task1_lr_1_10:
  after aircraft:   [48.2, 90.9]
  after caltech101: [47.1, 96.1]

text_task1_lr_1_5:
  after aircraft:   [49.2, 92.0]
  after caltech101: [47.9, 96.6]

vision_only_fd_cd:
  after aircraft:   [39.6, 86.9]
  after caltech101: [38.4, 96.1]
```

## Main Comparisons

Best schedule vs always tune:

```text
text_task1_lr_1_5 - text_always
Transfer: +0.41
Average:  +0.54
Last:     +0.49
```

Freeze after task 1 vs always tune:

```text
text_task1_freeze - text_always
Transfer: -0.85
Average:  +0.06
Last:     +0.27
```

LR/10 after task 1 vs always tune:

```text
text_task1_lr_1_10 - text_always
Transfer: -0.65
Average:  -0.29
Last:     -0.12
```

Best schedule vs vision-only FD/CD:

```text
text_task1_lr_1_5 - vision_only_fd_cd
Transfer: +5.03
Average:  +6.15
Last:     +4.98
```

## Scientific Reading

The Stage 1 pilot supports the low-LR consolidation hypothesis:

```text
task 1: full text LR
task 2: text LR / 5
```

This schedule is best on all three aggregate metrics in the 2-task pilot:

```text
Transfer = 91.97
Average  = 71.40
Last     = 72.22
```

The improvement over always-tune is modest but consistent:

```text
Transfer +0.41
Average  +0.54
Last     +0.49
```

Freeze-after-task1 also improves Last over always-tune but loses Transfer:

```text
This suggests text drift may exist, but completely freezing text may be too
conservative for later-task adaptation.
```

LR/10 is worse than LR/5:

```text
This suggests later text adaptation remains useful; reducing the LR too much can
under-adapt the second task.
```

Vision-only FD/CD remains clearly weaker:

```text
This confirms that the text encoder contribution is not explained by FD/CD
alone.
```

## Current Recommendation

Proceed to Stage 2 with a 4-task stress test.

Recommended Stage 2 task sequence:

```text
aircraft caltech101 dtd eurosat
```

Recommended Stage 2 configs:

```text
text_always
text_task1_lr_1_5
text_task1_freeze
```

Rationale:

```text
text_task1_lr_1_5 is the Stage 1 winner;
text_always is the strong baseline;
text_task1_freeze tests whether retention wins over plasticity over more than
two tasks.
```

Do not yet use this Stage 1 result as a paper-level claim:

```text
It is one seed, two tasks, one task order.
It is enough to justify Stage 2, not enough to claim full X-TAIL stability or
superiority over LADA.
```

## Next Action

Implement or parameterize a Stage 2 launcher for:

```text
tasks = aircraft caltech101 dtd eurosat
configs = text_always text_task1_lr_1_5 text_task1_freeze
seed = 42
num_shots = 16
iterations = 800
batch_size = 64
eval_max_samples = 0
OMP/MKL/OPENBLAS/NUMEXPR threads = 4
```

If Stage 2 confirms LR/5 improves or ties Last/Average without harming new-task
learning, then the full strict 10-task, 3-seed gate should use:

```text
primary:
  LoRA-NSP + FD/CD + text_task1_lr_1_5

companion:
  LoRA-NSP + FD/CD + text_always
```

