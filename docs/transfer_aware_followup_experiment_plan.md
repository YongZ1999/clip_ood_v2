# Transfer-aware follow-up experiment plan

## Objective

The transfer-aware E1 main table is complete: LoRA-NF has a three-seed
advantage over Standard LoRA and the native LADA reproduction in
Transfer/Average/Last.  Do not continue to tune classifier hyperparameters for
main-table gains: D1/D2 established diminishing returns.

This plan adds only the evidence that became newly necessary after changing
classification evaluation to `preserve_aspect + zs_predicted_seen`:

1. an inference-only ablation for the new gate (E6-TA); and
2. a new-protocol distillation-component ablation with retrieval (E2-TA).

All follow-ups are **single-seed explanatory experiments**, not replacements
for the three-seed E1 table.  Use seed 42 consistently because the matching
full FD+CD D1 artifact trajectory already exists.

## Fixed reference configuration

Use the transfer-aware LoRA-NF 16-shot configuration from E1/D1 unchanged:

```text
OpenAI CLIP ViT-B/16; LoRA-NF rank=4; hard NSP eps=.20, weight=.02;
800 iterations; AdamW lr=1e-4; FD=1; CD=2;
M=4; RGDA rank=32; fit=200; gmm_sample; alpha=.05;
preserve_aspect; zs_predicted_seen; MSCOCO 5K retrieval after every task.
```

The matching full-distillation seed-42 reference is already available at:

```text
experiments/paper_transfer_aware/dev_seed42/
```

It is called **C3 (FD=1, CD=2)** in the tables below.  Do not rerun C3.

## R1 / E6-TA: transfer-aware routing ablation — no encoder training

### Question

Does the `zs_predicted_seen` gate itself protect forward transfer compared with
the historical classwise ZS+LR-RGDA fusion?

### Execution

Reuse the ten D1 artifacts under
`experiments/paper_transfer_aware/dev_seed42/async_eval/`.  Run the offline
evaluator twice, with all classifier settings identical:

| Condition | Routing | M | replay | fit | alpha | eval seed |
|---|---|---:|---|---:|---:|---:|
| E6-classwise | `classwise` | 4 | gmm_sample | 200 | .05 | 42 |
| E6-gated | `zs_predicted_seen` | 4 | gmm_sample | 200 | .05 | 42 |

Record ZS and Ensemble Transfer/Average/Last. Retrieval is intentionally not
rerun: a classifier routing change cannot change image/text embeddings or
retrieval rankings.

### Interpretation

The gate is supported if it improves or preserves Transfer without a material
Average/Last loss.  This is a single-seed inference ablation; report it as
such, not as a replacement for E1's three-seed result.

## R2 / E2-TA: distillation components with retrieval — three encoder runs

### Question

Under the new classification protocol, which of feature distillation (FD) and
cross-modal distillation (CD) preserves classification and retrieval?

### Conditions

Run the following three LoRA-NF 16-shot seed-42 jobs.  Every flag other than
`fd_weight`/`cd_weight`, experiment name and output path must match the fixed
reference configuration.

| ID | FD | CD | Purpose |
|---|---:|---:|---|
| C0 | 0 | 0 | no-distillation reference |
| C1 | 1 | 0 | FD-only |
| C2 | 0 | 2 | CD-only |
| C3 | 1 | 2 | existing D1 full reference; do **not** rerun |

Run C0/C1/C2 concurrently on three otherwise idle GPUs.  Enable the ordinary
inline classification evaluation and MSCOCO 5K retrieval after every task.
Save results to `experiments/paper_transfer_aware/E2_distill_seed42/` and logs
to `logs/paper_transfer_aware/E2_distill_seed42/`.

### Required report

For C0/C1/C2/C3, create `E2_TA_SUMMARY.md` containing:

- Ensemble and ZS Transfer/Average/Last;
- MSCOCO 5K I2T/T2I R@1/5/10, both Average-over-tasks and Last;
- deltas relative to C3;
- exact command/configuration, commit SHA, and any failure.

Do not claim statistical significance from one seed.  The purpose is to test
whether the old component conclusion persists under the new protocol and to
identify which component best preserves retrieval.  If a large, coherent trend
appears, only that key comparison may later be expanded to multiple seeds.

## Optional R3 / E3-TA: adapter family under the new protocol

Do this only if the paper will place LoRA, LoRA-Null, Gradient-projected LoRA,
and LoRA-NF in one **new-protocol** ranking table.  The new E1 already provides
LoRA and LoRA-NF.  In that case, run only the missing LoRA-Null and
Gradient-projected LoRA 16-shot seed-42 conditions with the fixed
transfer-aware evaluation, then decide whether their ranking warrants multiple
seeds.  Otherwise retain the historical E3 table as a clearly labelled legacy
protocol ablation and do not spend compute here.

## Explicit non-runs

- Do not rerun retrieval-only studies: retrieval preprocessing and embeddings
  did not change.
- Do not rerun E4 NSP hyperparameter sweeps: D1/D2 already show classifier
  tuning saturation, and the old E4 is sufficient for parameter motivation.
- Do not rerun native LADA or frozen CLIP merely because the LoRA-NF evaluator
  changed; their methods do not use this LR-RGDA gate.
- Do not run new three-seed main tables unless a single-seed follow-up shows a
  large, pre-specified effect.
