# Goal: Text Encoder Tuning Schedule for LoRA-NSP Incremental CLIP

**Date**: 2026-06-14

## 0. Goal Mode Objective

Execute a rigorous, staged evaluation of task-wise text encoder tuning
schedules for the current strongest CLIP continual learning training baseline:

```text
LoRA-NSP + vision encoder tuning + text encoder tuning + FD/CD
```

The goal is to decide the training-side configuration that should enter the
strict LADA-style full incremental gate:

```text
X-TAIL 10 tasks
16-shot
full test split
3 seeds: 42, 43, 44
Transfer / Average / Last
```

The goal is not yet to claim final superiority over LADA. The immediate output
must be a defensible recommendation for the full incremental run:

```text
primary text schedule
minimal companion ablation
exact launcher/command
recorded evidence and limitations in chat-history
```

## 1. Research Motivation

Recent pilots show that jointly tuning the vision encoder and text encoder
substantially improves LoRA-NSP incremental learning. This creates a new
question:

```text
Should the text encoder continue to be updated at the same learning rate for
every incremental task, or should text-side updates be reduced/frozen after an
early adaptation phase?
```

The motivating hypothesis is:

```text
The first task may adapt the CLIP text encoder to the downstream few-shot
zero-shot-style classification format. After that adaptation, a stable or
slowly changing text encoder may act as a semantic anchor and reduce forgetting.
```

This is a training-side question and should be kept separate from the
inference-side LR-RGDA + zero-shot ensemble work.

## 2. Current Baseline Evidence

The strongest prior 2-task training pilot was recorded in:

```text
chat-history/2026-06-14-60-incremental-training-pilot-results.md
```

Protocol:

```text
task_sequence = aircraft caltech101
seed = 42
num_shots = 16
iterations = 800
batch_size = 64
eval_max_samples = 0
```

Result summary:

| method | Transfer | Average | Last |
|---|---:|---:|---:|
| lora_nsp_fd_cd_vision_text | 89.29 | 72.01 | 73.49 |
| lora_nsp_vision_text | 88.84 | 70.35 | 71.34 |
| lora_vanilla_vision_text | 89.33 | 69.16 | 68.25 |
| lora_nsp_vision | 82.88 | 65.67 | 68.73 |
| lora_vanilla_vision | 82.27 | 62.71 | 62.21 |

Reading:

```text
vision+text tuning is strongly positive;
LoRA-NSP remains useful when text tuning is enabled;
FD/CD helps on top of vision+text LoRA-NSP;
therefore text schedule experiments should use lora_nsp_fd_cd_vision_text as
the fixed baseline.
```

## 3. Hypotheses

### H1: Task-1 Text Adaptation

The first task teaches a reusable downstream classification format rather than
only task-specific class semantics.

Prediction:

```text
task 1 text tuning followed by freezing retains much of the text-tuning benefit
and reduces old-task forgetting.
```

Expected signature:

```text
old-task final retention improves;
Transfer stays close to always-tune;
new-task final accuracy may drop slightly.
```

### H2: Continual Text Plasticity

Every new task needs text-side adaptation.

Prediction:

```text
always tuning the text encoder remains best because later tasks need full text
plasticity.
```

Expected signature:

```text
freeze-after-task1 hurts later tasks;
Average and Last drop even if first-task retention improves.
```

### H3: Low-LR Text Consolidation

The text encoder needs strong early adaptation but conservative later updates.

Prediction:

```text
task 1 uses base text LR;
tasks 2..K use base text LR / 5 or / 10;
this balances semantic anchoring and later-task adaptation.
```

Expected signature:

```text
low-LR-after-task1 matches or improves Last/Average over always-tune;
old-task retention improves;
new-task final accuracy does not collapse.
```

Current prior after Stage 1:

```text
Most likely best: task1 base LR, later text LR / 5.
Strong baseline: always tune text.
Analysis-only candidate: task1 then freeze.
Less likely: later text LR / 10.
Weak reference: never tune text.
```

## 4. Implemented CLI and Semantics

Code support has been implemented in:

```text
src/experiments/run_continual_learning.py
src/trainers/lora_nsp_trainer.py
scripts/run_incremental_text_schedule_pilot.sh
scripts/run_incremental_text_schedule_stress.sh
scripts/remote_training_ablation.sh
scripts/selftest_publication_launchers.py
scripts/selftest_remote_package.py
```

New CLI:

```text
--text_tuning_schedule always|never|freeze_after|low_lr_after
--text_schedule_switch_task 1
--text_lr_scale_after_task 0.2
```

Semantics:

```text
always:
  all tasks train text LoRA with base text LR

never:
  no text LoRA is attached/trained

freeze_after:
  tasks <= switch_task train text LoRA
  tasks > switch_task freeze the already-adapted/merged text encoder

low_lr_after:
  tasks <= switch_task train text LoRA with base text LR
  tasks > switch_task train text LoRA with base_text_lr * scale
```

Important correctness condition:

```text
freeze_after preserves the task-1 merged text adaptation.
It must not revert to the original CLIP text encoder after task 1.
```

Implementation behavior:

```text
vision LoRA and text LoRA use separate optimizer parameter groups;
frozen text LoRA is excluded from optimizer;
frozen/adapted text encoder is still used as a no-grad semantic anchor in CD;
per-task schedule metadata is saved in result JSON under text_schedule.
```

Thread limits are required in remote launchers:

```text
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
```

These limits were added after projection matrix updates overloaded CPU BLAS
threads during the first pilot launch.

## 5. Verification Already Passed

Local verification after implementation:

```bash
python -m py_compile \
  src/experiments/run_continual_learning.py \
  src/trainers/lora_nsp_trainer.py \
  scripts/selftest_publication_launchers.py \
  scripts/selftest_remote_package.py

bash -n \
  scripts/run_incremental_text_schedule_pilot.sh \
  scripts/run_incremental_text_schedule_stress.sh \
  scripts/remote_training_ablation.sh

python scripts/selftest_publication_launchers.py
python scripts/selftest_remote_package.py
bash scripts/verify_publication_package.sh
```

Verifier status:

```text
publication package checks passed
PASS=5, PENDING=2
```

The two pending items are pre-existing real training/full incremental result
gates, not schedule-code failures.

## 6. Stage 1: 2-Task Pilot

Status:

```text
completed and summarized
```

Record:

```text
chat-history/2026-06-14-62-text-tuning-schedule-pilot-results.md
```

Remote output:

```text
experiments/incremental_text_schedule_pilot_20260614
```

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
  no text LoRA
```

Aggregate:

| method | n | seed | K | Transfer | Average | Last |
|---|---:|---:|---:|---:|---:|---:|
| text_task1_lr_1_5 | 1 | 42 | 2 | 91.97 | 71.40 | 72.22 |
| text_task1_freeze | 1 | 42 | 2 | 90.71 | 70.92 | 72.00 |
| text_always | 1 | 42 | 2 | 91.56 | 70.86 | 71.73 |
| text_task1_lr_1_10 | 1 | 42 | 2 | 90.91 | 70.57 | 71.61 |
| vision_only_fd_cd | 1 | 42 | 2 | 86.94 | 65.26 | 67.24 |

Accuracy matrices:

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

Main comparisons:

```text
text_task1_lr_1_5 - text_always:
  Transfer +0.41
  Average  +0.54
  Last     +0.49

text_task1_freeze - text_always:
  Transfer -0.85
  Average  +0.06
  Last     +0.27

text_task1_lr_1_10 - text_always:
  Transfer -0.65
  Average  -0.29
  Last     -0.12

text_task1_lr_1_5 - vision_only_fd_cd:
  Transfer +5.03
  Average  +6.15
  Last     +4.98
```

Stage 1 reading:

```text
The low-LR consolidation hypothesis is supported.
The best pilot schedule is task1 full text LR, later text LR / 5.
The improvement over always-tune is modest but consistent.
Freezing may help retention but appears too conservative.
LR/10 under-adapts compared with LR/5.
Vision-only FD/CD is clearly weaker, confirming the text encoder benefit.
```

Stage 1 limitation:

```text
one seed, two tasks, one task order;
enough to justify Stage 2;
not enough for a paper-level claim or LADA comparison.
```

## 7. Stage 2: 4-Task Stress Test

Status as of 2026-06-14:

```text
launched remotely
no result JSON yet
three configs are still running at Step 1/4
```

Remote project:

```text
/home/raoxuan/projects/project_clip_continual_learning
```

Output:

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

Configurations:

```text
text_always:
  --text_tuning_schedule always
  --text_lr_scale_after_task 1.0

text_task1_lr_1_5:
  --text_tuning_schedule low_lr_after
  --text_schedule_switch_task 1
  --text_lr_scale_after_task 0.2

text_task1_freeze:
  --text_tuning_schedule freeze_after
  --text_schedule_switch_task 1
  --text_lr_scale_after_task 0.0
```

Why these tasks:

```text
aircraft: fine-grained and hard
caltech101: broad object categories and easier
dtd: texture recognition and different visual/semantic structure
eurosat: remote sensing and strong domain shift
```

Primary Stage 2 analysis:

```text
1. Last across all 4 tasks.
2. Average across all 4 tasks.
3. Final retention for task 1 and task 2.
4. Accuracy matrix degradation as more tasks are learned.
5. New-task final accuracy under freeze/low-LR schedules.
6. Whether LR/5 improves retention without sacrificing dtd/eurosat adaptation.
```

Stage 2 decision rule:

```text
If text_task1_lr_1_5 beats or ties always on Last/Average and does not harm
later-task final accuracy by more than about 1 point, use LR/5 as the primary
full-gate schedule.

If always beats LR/5 clearly, remove schedule complexity and use always-tune as
the full-gate baseline.

If freeze improves retention but hurts later domain adaptation, keep it as an
analysis result only.
```

Status command:

```bash
ssh raoxuan@10.20.34.30 '
cd /home/raoxuan/projects/project_clip_continual_learning &&
echo RESULTS &&
find experiments/incremental_text_schedule_stress_20260614 -maxdepth 1 -name "*_results.json" -printf "%f\n" | sort &&
echo PROCS &&
pgrep -af "run_incremental_text_schedule_stress|incremental_text_schedule_stress_20260614|text_tuning_schedule" || true &&
echo LOGS &&
for f in experiments/incremental_text_schedule_stress_20260614/logs/*_seed42.log; do
  echo "==== $(basename "$f") ===="
  grep -aE "Step [1234]/4|Task train schedule|Projection matrices updated|Evaluating on all|Acc =|Transfer:|Average:|Last:|All results saved|Traceback|OutOfMemory|RuntimeError|ValueError" "$f" | tail -80 || true
done
'
```

Summarizer once all three result JSONs exist:

```bash
ssh raoxuan@10.20.34.30 '
cd /home/raoxuan/projects/project_clip_continual_learning &&
source ~/miniconda3/etc/profile.d/conda.sh &&
conda activate raoxuan &&
python scripts/summarize_incremental_metrics.py \
  experiments/incremental_text_schedule_stress_20260614 \
  --output_csv experiments/incremental_text_schedule_stress_20260614/incremental_summary.csv \
  --output_markdown experiments/incremental_text_schedule_stress_20260614/incremental_summary.md \
  --aggregate_csv experiments/incremental_text_schedule_stress_20260614/incremental_aggregate.csv \
  --aggregate_markdown experiments/incremental_text_schedule_stress_20260614/incremental_aggregate.md \
  --expected_tasks "aircraft caltech101 dtd eurosat" &&
cat experiments/incremental_text_schedule_stress_20260614/incremental_aggregate.md
'
```

After summarization, create:

```text
chat-history/2026-06-14-63-text-tuning-schedule-stress-results.md
```

Required contents:

```text
protocol
aggregate table
accuracy matrices
per-task text schedule metadata
LR/5 vs always comparison
freeze vs always comparison
retention/new-task adaptation analysis
full-gate recommendation
limitations
```

## 8. Stage 3: Full Strict 10-Task, 3-Seed Gate

Launch only after Stage 2 is summarized and a schedule is selected.

Strict protocol:

```text
task_sequence = full X-TAIL 10-task sequence
seeds = 42 43 44
num_shots = 16
eval_max_samples = 0
method = lora_nsp
tune_vision_encoder = true
fd_weight = 1.0
cd_weight = 1.0
text schedule = selected by Stage 2
```

Recommended full-gate grid if Stage 2 confirms LR/5:

```text
primary:
  lora_nsp_fd_cd_vision_text_task1_lr_1_5

minimal companion:
  lora_nsp_fd_cd_vision_text_always
```

Recommended full-gate grid if Stage 2 rejects LR/5:

```text
primary:
  lora_nsp_fd_cd_vision_text_always

minimal companion:
  only the best Stage 2 alternative if it reveals a clear trade-off
```

Avoid running many full 10-task variants unless Stage 2 is genuinely ambiguous.
The goal is statistical clarity, not a large uncontrolled grid.

Full-gate decision requirements:

```text
1. Report mean +/- std over seeds 42, 43, 44.
2. Report per-seed values, not only aggregate means.
3. Use full test split, not eval subsampling.
4. Compare against the correct LADA-aligned baseline.
5. Do not claim superiority over LADA unless the strict full gate supports it.
```

## 9. Interaction With LR-RGDA + ZS

Do not stack LR-RGDA + ZS into the text schedule stress test yet.

Boundary:

```text
text schedule = training-side innovation;
LR-RGDA + ZS = inference-side innovation.
```

Recommended order:

```text
1. Select stable training-side schedule with strict incremental evidence.
2. Save/document resulting checkpoints and configs.
3. Then evaluate LR-RGDA + ZS on top of the stable training setup.
```

This keeps the research story clean:

```text
Training-side contribution:
  LoRA-NSP + vision/text continual adaptation + conservative text schedule

Inference-side contribution:
  LR-RGDA + ZS ensemble classifier under compact statistical replay
```

## 10. Completion Criteria for This Goal

The goal is complete only when all of the following are true:

```text
1. Stage 2 either completes and is summarized, or there is a documented
   infrastructure reason that prevents completion and a concrete recovery plan.

2. A chat-history entry records Stage 2 results/status:
   chat-history/2026-06-14-63-text-tuning-schedule-stress-results.md

3. The final recommendation for the strict full gate is explicit:
   LR/5, always, freeze-analysis-only, or another justified choice.

4. The full-gate launcher/command is identified.

5. Local verification is rerun after any final code/script edits:
   bash scripts/verify_publication_package.sh

6. The final answer clearly states what should be run next in goal mode.
```

Do not mark this goal complete solely because Stage 1 supports LR/5. Stage 1 is
only a screening result.

## 11. Immediate Next Actions in Goal Mode

1. Check remote Stage 2 status.
2. If all three result JSONs exist, run the summarizer.
3. Parse accuracy matrices and `text_schedule` metadata from each JSON.
4. Write `chat-history/2026-06-14-63-text-tuning-schedule-stress-results.md`.
5. Apply the Stage 2 decision rule.
6. If a full-gate schedule is selected, prepare or verify the full 10-task,
   3-seed launcher.
7. Run local verification.

The current best prior entering Stage 2 is:

```text
text_task1_lr_1_5:
  task 1 text_lr = 1e-4
  tasks 2..K text_lr = 2e-5
```

The current strongest fixed baseline remains:

```text
text_always:
  all tasks text_lr = 1e-4
```

