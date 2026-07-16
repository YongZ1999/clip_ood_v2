#!/usr/bin/env bash
# Native LADA-only E7 replay. Does not rerun completed SigLIP2 LoRA-NF seeds.
# Usage: bash scripts/run_siglip2_lada_native.sh GPU_INDEX [SEED ...]
set -euo pipefail

GPU="${1:?Usage: bash scripts/run_siglip2_lada_native.sh GPU_INDEX [SEED ...]}"
shift
ROOT="${XTAIL_ROOT:-/data1/open_datasets/X-TAIL}"
OUT="${SIGLIP2_OUTPUT_DIR:-experiments/paper_formal/E7_siglip2}"
MODEL="google/siglip2-base-patch16-224"
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(42 43 44)
fi

for seed in "${SEEDS[@]}"; do
  python -u scripts/main_incremental_lada_native.py \
    --root "$ROOT" --model_name "$MODEL" \
    --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --num_shots 16 --batch_size 64 --eval_batch_size 128 \
    --lr 1e-3 --weight_decay 5e-4 \
    --lada_k 16 --prototype_k 4 --image_prototypes_weight_coef 64 \
    --text_adapter_dim 16 --text_adapter_scale 0.1 \
    --experiment_name "E7__siglip2__lada_native__16shot__seed${seed}" --seed "$seed" \
    --gpu "$GPU" --output_dir "$OUT"
done
