# Training Launcher Audit Improvements

Date: 2026-06-14

## Context

The joint training ablation is the main remaining gate for deciding whether
LoRA-NSP can remain a primary empirical contribution. The launcher must produce
auditable result directories and fail loudly if any background job fails.

## Issues Found

During a local smoke attempt, setting `SEEDS=""` still used the default seed list
because Bash `${SEEDS:-...}` treats an empty string as unset. This briefly
started local training commands. They failed immediately because the local
environment does not have `transformers` installed:

```text
ModuleNotFoundError: No module named 'transformers'
```

This exposed a more important launcher issue: background training failures could
still let the script continue to the summary stage.

## Changes

Updated `scripts/run_joint_training_ablation.sh`:

- writes `${OUT_DIR}/manifest.txt` before launching jobs;
- records timestamp, host, working directory, Python path, conda environment,
  root, output directory, seeds, iterations, batch size, GPUs, dry-run flag,
  CLIP loading flags, git commit/status, common args, and config grid;
- supports `DRY_RUN=1`, which writes per-config command logs without invoking
  `main_joint.py`;
- tracks background job PIDs explicitly;
- exits nonzero if any background training job fails, preventing failed runs
  from being silently summarized.

Updated `scripts/summarize_joint_classifier_replay.py`:

- added minimum delta columns;
- added `all LR-RGDA > LADA` and `all LR-RGDA+ZS > LADA` columns.

Updated `experiments/README_publication_repro.md`:

- documents the new manifest;
- documents `DRY_RUN=1`;
- documents min-delta and all-positive summary columns.

## Verification

Ran:

```bash
bash -n scripts/run_joint_training_ablation.sh
python -m py_compile scripts/summarize_joint_classifier_replay.py
DRY_RUN=1 OUT_DIR=/private/tmp/joint_training_dry_run_20260614 GPUS='0 1 2 3' \
  bash scripts/run_joint_training_ablation.sh
python scripts/summarize_joint_classifier_replay.py \
  --input_dir /private/tmp/joint_summary_synth_20260614 \
  --output_csv /private/tmp/joint_summary_synth_20260614/summary.csv \
  --output_markdown /private/tmp/joint_summary_synth_20260614/summary.md
git diff --check
```

All checks passed. The dry run produced `manifest.txt`, launch logs, command
logs, and an empty summary without calling `main_joint.py`.

The synthetic summary confirmed:

```text
min LR-RGDA - LADA = 0.40
all LR-RGDA > LADA = yes
min LR-RGDA+ZS - LADA = 0.70
all LR-RGDA+ZS > LADA = yes
```

## Remote Status

Another low-frequency SSH query failed:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

The training ablation remains pending.
