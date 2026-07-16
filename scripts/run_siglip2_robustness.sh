#!/usr/bin/env bash
# SigLIP 2 robustness main table: 16-shot X-TAIL, 3 seeds.
# Usage: bash scripts/run_siglip2_robustness.sh GPU_INDEX
set -euo pipefail

GPU="${1:?Usage: bash scripts/run_siglip2_robustness.sh GPU_INDEX}"
ROOT="${XTAIL_ROOT:-/data1/open_datasets/X-TAIL}"
OUT="${SIGLIP2_OUTPUT_DIR:-experiments/paper_formal/E7_siglip2}"
MODEL="google/siglip2-base-patch16-224"
SEEDS=(42 43 44)

COMMON=(
  --root "$ROOT"
  --model_name "$MODEL"
  --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397
  --num_shots 16 --batch_size 32 --eval_batch_size 128 --iterations 800
  --optimizer adamw --lr 1e-4 --weight_decay 3e-5
  --scheduler cosine_with_warmup --warmup_ratio 0.1
  --lora_rank 4 --lora_alpha 4 --lora_dropout 0.0
  --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2
  --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01
  --rgda_rank 32 --rgda_alpha1 0.2 --rgda_alpha2 2.0 --rgda_alpha3 0.5
  --rgda_fit_source gmm_sample --alpha 0.05 --ensemble_normalize maxshift
  --gpu "$GPU" --output_dir "$OUT"
)

for seed in "${SEEDS[@]}"; do
  # Proposed method: the same LoRA-NF + FD/CD + LR-RGDA configuration as E1.
  python -u main_incremental.py "${COMMON[@]}" \
    --experiment_name "E7__siglip2__lora_nf__16shot__seed${seed}" --seed "$seed" \
    --lora_type lora_nsp --init_mode lora_nsp --use_dora false \
    --nsp_eps 0.20 --nsp_weight 0.02 \
    --tune_vision_encoder true --tune_text_encoder true --text_adapter_type matched \
    --text_tuning_schedule always --fd_weight 1.0 --cd_weight 2.0 --cd_temperature 4.0 \
    --disable_lada

  # LADA-style SigLIP2 port: frozen visual encoder + AdaptFormer text tuner +
  # label-specific prototype memory.  It is architecture-adapted, not the
  # original OpenAI-CLIP-only official LADA code.
  python -u main_incremental.py "${COMMON[@]}" \
    --experiment_name "E7__siglip2__lada_style__16shot__seed${seed}" --seed "$seed" \
    --lora_type lora_nsp --init_mode lora_nsp --use_dora false \
    --tune_vision_encoder false --tune_text_encoder true \
    --text_adapter_type lada_adaptformer --text_adapter_dim 16 --text_adapter_scale 0.1 \
    --text_tuning_schedule always --fd_weight 0 --cd_weight 0 \
    --enable_lada
done
