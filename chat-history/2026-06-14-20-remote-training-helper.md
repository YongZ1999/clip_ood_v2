# Remote Training Helper

Date: 2026-06-14

## Context

SSH to `10.20.34.30` has been unstable. Previous remote launch instructions used
multiple connections or required manually copying several scripts, which creates
an avoidable failure surface.

## Changes

Added `scripts/remote_training_ablation.sh`.

Subcommands:

- `upload`: uploads the relevant scripts, ledger, and reproducibility README via
  one tar-over-SSH stream;
- `launch`: runs `bash -n scripts/run_joint_training_ablation.sh`, starts it with
  `nohup`, and prints matching processes;
- `status`: prints matching processes, completed `*_seed*.json` files, and the
  tail of `logs/launch.log`;
- `postprocess`: runs `scripts/postprocess_publication_results.sh` on the remote
  result directory.

Environment variables:

- `REMOTE`, default `raoxuan@10.20.34.30`;
- `REMOTE_DIR`, default `/home/raoxuan/projects/project_clip_continual_learning`;
- `OUT_DIR`, default `experiments/joint_training_ablation_20260614`;
- `MASTER_LOG`, default `experiments/joint_training_ablation_20260614_master.log`.

Updated `experiments/README_publication_repro.md` to use:

```bash
bash scripts/remote_training_ablation.sh upload
bash scripts/remote_training_ablation.sh launch
bash scripts/remote_training_ablation.sh status
```

Updated `scripts/verify_publication_package.sh` to run `bash -n` on the remote
helper.

## Boundary

The helper does not retry or poll aggressively. It keeps SSH usage explicit and
low-frequency because repeated SSH attempts have been failing with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```
