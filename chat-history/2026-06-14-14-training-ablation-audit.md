# Training Ablation Audit

Date: 2026-06-14

## Context

The joint training ablation is the main remaining empirical gate for deciding
whether LoRA-NSP remains a primary contribution. The result summary should be
audited mechanically once the remote run completes.

## Changes

Added `scripts/audit_training_ablation.py`.

The script reads a training ablation `summary.csv` and checks:

- required experiments are present:
  - `lora_vanilla`
  - `lora_nsp`
  - `lora_nsp_fd`
  - `lora_nsp_fd_cd`
- expected seeds are present;
- `n` is at least the expected seed count;
- `CLIP-ZS`, `LR-RGDA`, and `LR-RGDA+ZS` are parseable;
- the full `lora_nsp_fd_cd` configuration improves over `lora_vanilla` and
  `lora_nsp` on those classifier-side metrics.

The script supports `--allow_missing`, which reports the audit as pending when
the summary file is not available yet.

Updated `scripts/verify_publication_package.sh`:

- runs `python scripts/audit_training_ablation.py --allow_missing`;
- includes the new audit script in Python syntax checks.

Updated `experiments/README_publication_repro.md` with the strict command to run
after the remote training summary exists.

## Verification

Created a synthetic summary at:

```text
/private/tmp/training_ablation_audit_synth_20260614/summary.csv
```

Ran:

```bash
python scripts/audit_training_ablation.py \
  --summary_csv /private/tmp/training_ablation_audit_synth_20260614/summary.csv \
  --expected_seeds '42 43 44'
python scripts/audit_training_ablation.py \
  --summary_csv /private/tmp/training_ablation_audit_synth_20260614/missing.csv \
  --allow_missing
bash scripts/verify_publication_package.sh
git diff --check
```

The synthetic full-result audit passed. The missing-summary path reported:

```text
TRAINING ABLATION AUDIT PENDING
```

The publication package verifier still passed and now prints the pending
training-ablation status while remote results are absent.

## Boundary

This is a classifier-side training ablation gate. It does not replace the
remaining strict incremental Average/Last/forgetting metrics.
