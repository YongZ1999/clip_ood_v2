# Classifier Replay Launcher Audit Improvements

Date: 2026-06-14

## Context

The inference-side replay matrix is the strongest current evidence for the
publication claim. Its launcher should therefore provide the same audit trail as
the joint training ablation launcher.

## Changes

Updated `scripts/run_joint_classifier_replay.sh`:

- defaults `CLASSIFIER_FEATURE_TRANSFORMS` to `test`, matching the fair main
  protocol;
- auto-activates the `raoxuan` conda environment when available and no conda
  environment is already active;
- exports `CLIP_USE_SAFETENSORS=0` and `CLIP_LOCAL_FILES_ONLY=1` by default;
- writes `${OUT_DIR}/manifest.txt` before launching runs;
- records timestamp, host, working directory, Python path, conda environment,
  dataset root, output directory, GPU, seeds, feature transforms, replay modes,
  mean-ablation flag, dry-run flag, CLIP loading flags, git commit/status, and
  common args;
- skips existing JSON result files so interrupted runs can resume safely;
- supports `DRY_RUN=1`, writing command logs without invoking `main_joint.py`;
- tees the final summary into `${OUT_DIR}/logs/summary.log`.

Updated `scripts/summarize_joint_classifier_replay.py`:

- changed the markdown first-column label from `Replay` to `Experiment`, since
  the script now summarizes both classifier replay and training ablation
  directories.

Updated `experiments/README_publication_repro.md`:

- documents the replay launcher manifest;
- documents replay launcher dry-run usage;
- states that the replay launcher now defaults to the fair `test` transform.

## Verification

Ran:

```bash
bash -n scripts/run_joint_classifier_replay.sh
bash -n scripts/run_joint_training_ablation.sh
python -m py_compile scripts/summarize_joint_classifier_replay.py
DRY_RUN=1 \
  OUT_DIR=/private/tmp/joint_classifier_replay_dry_run_20260614 \
  CLASSIFIER_FEATURE_TRANSFORMS='test' \
  REPLAY_MODES='real gmm_raw_mean' \
  bash scripts/run_joint_classifier_replay.sh
git diff --check
```

All checks passed. The classifier replay dry run produced:

- `manifest.txt`
- `logs/launch.log`
- per-run command logs for `test_real` and `test_gmm_raw_mean` across seeds
  42/43/44
- `summary.csv`
- `summary.md`
- `logs/summary.log`

## Remote Status

Another low-frequency SSH query failed:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

The joint training ablation remains pending.
