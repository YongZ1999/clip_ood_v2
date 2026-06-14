# Incremental Protocol Audit and Remote Upload Closure

**Date**: 2026-06-14

## Context

After adding `scripts/run_joint_incremental_ablation.sh`, the local verifier
could syntax-check the new launcher, but the protocol audit still did not inspect
it. The remote upload helper also did not include all scripts used by
`verify_publication_package.sh`, which could make remote postprocessing depend
on stale files already present on the server.

## Changes

Updated:

```text
scripts/audit_protocol_config.py
scripts/remote_training_ablation.sh
```

`audit_protocol_config.py` now audits:

- `scripts/run_joint_classifier_replay.sh`
- `scripts/run_joint_training_ablation.sh`
- `scripts/run_joint_incremental_ablation.sh`
- `experiments/README_publication_repro.md`

The new incremental checks cover:

- full expected task sequence plumbing via `EXPECTED_TASKS`;
- `--num_shots 16`;
- four publication configurations:
  `lora_vanilla`, `lora_nsp`, `lora_nsp_fd`, `lora_nsp_fd_cd`;
- stable `--experiment_name` labels;
- incremental summarization through `scripts/summarize_incremental_metrics.py`;
- CLIP offline loading flags.

`remote_training_ablation.sh` now uploads the full local publication audit and
launcher package needed by remote postprocessing, including:

- `main_joint.py`
- classifier/training/incremental launchers;
- all audit scripts called by the verifier;
- `src/experiments/run_continual_learning.py`;
- locally modified model/classifier/trainer files;
- `paper_draft.tex`;
- publication README and result ledger.

## Verification

Ran:

```bash
python scripts/audit_protocol_config.py
python scripts/audit_publication_gates.py
bash -n scripts/remote_training_ablation.sh
bash -n scripts/run_joint_incremental_ablation.sh
python -m py_compile scripts/audit_protocol_config.py scripts/audit_publication_gates.py
bash scripts/verify_publication_package.sh
git diff --check
```

Current verifier status remains:

```text
Summary: PASS=5, PENDING=2
publication package checks passed
```

The remaining pending gates are still real-result gates:

- training-side LoRA-NSP ablation summary;
- strict incremental Transfer/Average/Last/forgetting summary.

Remote status was queried once and still failed with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```
