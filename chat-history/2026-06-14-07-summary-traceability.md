# Summary Traceability Improvements

Date: 2026-06-14

## Context

The publication goal requires every claim against LADA to be auditable by seed,
feature source, and exact result directory. The existing summary script reported
classifier averages but did not expose the number of completed seeds or the
direct LR-RGDA versus LADA margins.

## Changes

Updated `scripts/summarize_joint_classifier_replay.py` to include:

- `n`: number of completed result files contributing to the experiment row;
- `seeds`: seed ids parsed from the JSON result or filename;
- `LR-RGDA - LADA`: matched per-run margin summarized as mean/std;
- `LR-RGDA+ZS - LADA`: matched per-run ensemble margin summarized as mean/std.

Updated `experiments/README_publication_repro.md` to:

- document the new summary columns;
- recommend using delta columns to check whether improvements are positive for
  every seed, not only positive on average;
- replace the `scp` plus `ssh` training launch with a single-connection stdin
  upload command, which is more suitable while SSH is unstable;
- remove already completed storage-budget and DPT-boundary items from the
  remaining publication gates.

## Verification

Ran:

```bash
python -m py_compile scripts/summarize_joint_classifier_replay.py
bash -n scripts/run_joint_training_ablation.sh
git diff --check
python scripts/summarize_joint_classifier_replay.py \
  --input_dir /private/tmp/joint_summary_empty_20260614 \
  --output_csv /private/tmp/joint_summary_empty_20260614/summary.csv \
  --output_markdown /private/tmp/joint_summary_empty_20260614/summary.md
```

All checks passed. The empty-directory smoke test produced the expected markdown
header with the new traceability and delta columns.

## Remote Status

Another low-frequency SSH status query still failed with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

The joint training ablation remains pending until the remote connection becomes
available.
