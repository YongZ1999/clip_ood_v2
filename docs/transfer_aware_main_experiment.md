# Transfer-aware E1 main experiment

## Purpose

This protocol evaluates the final transfer-oriented LoRA-NF inference design
without introducing a new ablation suite. It targets the observed pattern of a
strong Last score but a lower absolute Transfer score.

The new results must be written to `experiments/paper_transfer_aware/`, never
over the existing `experiments/paper_formal/` results.

## What changes

### 1. Standard CLIP classification-test preprocessing

`--eval_resize_mode preserve_aspect` uses:

```text
Resize(shorter_edge=224, bicubic) -> CenterCrop(224)
```

instead of the historical `Resize((224, 224))` image warp. Training transforms
and retrieval preprocessing are unchanged. The old behavior remains available
as `--eval_resize_mode legacy_square` and is still the program default for
reproducibility.

### 2. Zero-shot-predicted-seen ensemble gate

`--ensemble_routing zs_predicted_seen` evaluates each image as follows, with
`K_t` the number of classes learned through task `t`:

```text
p_zs = argmax(ZS logits)
if p_zs >= K_t:
    final prediction = ZS prediction
else:
    final prediction = classwise maxshift ZS + LR-RGDA ensemble prediction
```

The condition is based only on the model's zero-shot prediction, never the
unknown ground-truth task or label. This differs from the old `classwise`
fusion, which adjusted seen-class columns for every sample, including samples
that ZS had already assigned to a future class.

## Frozen classifier settings

The experiment intentionally retains the previously validated LR-RGDA recipe:

```text
num_centers=4, rgda_rank=32, rgda_train_iter=200,
rgda_fit_source=gmm_sample, alpha=0.05, maxshift
```

`rank=15`, `gmm_mean`, and fewer fitting iterations are hypotheses, not yet
validated improvements. Changing them concurrently would make a single final
main-table rerun uninterpretable. They are therefore not part of this protocol.

## Required runs

The fair internal E1 comparison requires:

| Method | Shots | Seeds | Training runs |
|---|---|---|---:|
| LoRA-NF | 16-shot, full-shot | 42, 43, 44 | 6 |
| Standard LoRA | 16-shot, full-shot | 42, 43, 44 | 6 |

Total: **12 encoder-training runs**. Frozen CLIP needs no retraining. Native
LADA also needs no retraining because this branch changes only the internal
classification evaluator and LR-RGDA inference. Before placing a local LADA
number in the same table, verify its evaluator: the official `LADA/` trainer
uses standard aspect-preserving preprocessing, whereas any legacy wrapper that
calls shared `get_transforms` must be explicitly re-evaluated with
`preserve_aspect`.

For a quick method-only diagnostic, set `RUN_STANDARD_LORA=0`; this runs only
the six LoRA-NF jobs. Do not use that reduced run to update a LoRA-vs-LoRA-NF
comparison table.

## Launch

After activating the server environment and checking out this branch:

```bash
bash scripts/run_transfer_aware_main_table.sh
```

The launcher uses GPU IDs `0,1,2,3,4,5` by default. It starts six jobs in the
first wave and queues one matched job per GPU for the second wave. It logs all
commands under `logs/paper_transfer_aware/` and writes JSON/retrieval results
under `experiments/paper_transfer_aware/E1_main/`.

Use `DRY_RUN=1` to inspect the six-GPU schedule and fully expanded commands
without starting training.

Useful overrides:

```bash
# Only measure the new LoRA-NF result first (six jobs).
RUN_STANDARD_LORA=0 bash scripts/run_transfer_aware_main_table.sh

# Use a different GPU subset or X-TAIL/retrieval roots.
GPU_IDS=1,2,3 ROOT=/data1/open_datasets/X-TAIL \
bash scripts/run_transfer_aware_main_table.sh
```

## Interpretation guardrails

- Compare new LoRA-NF with new Standard LoRA only; their test preprocessing and
  routing are identical.
- Compare against the pre-existing LADA results only after confirming the
  class names, templates, task order, and LADA native preprocessing remain
  aligned.
- Report all three Transfer/Average/Last metrics and retrieval Average/Last.
  A Transfer increase that substantially damages Last is a trade-off, not an
  automatic improvement.
- Keep old `paper_formal` results intact as the historical protocol record.
