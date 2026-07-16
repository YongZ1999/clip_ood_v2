#!/bin/bash
# Wave 1 chain launcher: each GPU runs its queue sequentially, all GPUs in parallel
set -euo pipefail

LOGDIR="logs/paper_formal"
mkdir -p "$LOGDIR"
ROOT="/data1/open_datasets/X-TAIL"
DS="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"

COMMON="--root $ROOT --dataset_sequence $DS --num_shots 16 --batch_size 32 --eval_batch_size 128 --iterations 800 --train_budget_mode uniform --optimizer adamw --lr 1e-4 --weight_decay 3e-5 --scheduler cosine_with_warmup --warmup_ratio 0.1 --eta_min 0.0 --lora_rank 4 --lora_alpha 4 --lora_dropout 0.0 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 --projection_param_mode full --null_init_mode none --nsp_eps 0.20 --nsp_weight 0.02 --reference_dataset flickr8k --reference_batch_size 32 --fd_weight 1.0 --cd_weight 2.0 --cd_divergence kl_forward --cd_temperature 4.0 --aux_weight 0.0 --tune_vision_encoder true --tune_text_encoder true --text_lora_rank 4 --text_tuning_schedule always --text_classifier_mode lada_hybrid --classifier_feature_transform test --rgda_rank 32 --rgda_alpha1 0.2 --rgda_alpha2 2.0 --rgda_alpha3 0.5 --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 --rgda_fit_source gmm_sample --alpha 0.05 --ensemble_normalize maxshift --temperature 1.0 --enable_retrieval_eval --retrieval_datasets mscoco_2014_5k,flickr30k_hf --retrieval_batch_size 128 --retrieval_recall_ks 1,5,10 --retrieval_max_images 0 --disable_lada"

run_exp() {
    local gpu=$1 name=$2; shift 2
    echo "[$(date '+%H:%M:%S')] GPU$gpu: $name starting" | tee -a "$LOGDIR/_queue.log"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u main_incremental.py $COMMON --gpu 0 "$@" \
      --experiment_name "$name" > "$LOGDIR/${name}.log" 2>&1
    echo "[$(date '+%H:%M:%S')] GPU$gpu: $name DONE (exit $?)" | tee -a "$LOGDIR/_queue.log"
}

# ====== GPU 0 queue ======
(run_exp 0 E3__lora_null__16shot__seed42 \
  --lora_type lora_nsp --init_mode lora_nsp --null_init_mode history_init_only \
  --seed 42 --output_dir experiments/paper_formal/E3_adapters
 run_exp 0 E2__C0__16shot__seed42 \
  --lora_type lora_nsp --init_mode lora_nsp \
  --fd_weight 0 --cd_weight 0 \
  --seed 42 --output_dir experiments/paper_formal/E2_components
 run_exp 0 E2__C2__16shot__seed42 \
  --lora_type lora_nsp --init_mode lora_nsp \
  --fd_weight 0 --cd_weight 2 --cd_temperature 4.0 \
  --seed 42 --output_dir experiments/paper_formal/E2_components
) &

# ====== GPU 1 queue ======
(run_exp 1 E3__lora_null__16shot__seed43 \
  --lora_type lora_nsp --init_mode lora_nsp --null_init_mode history_init_only \
  --seed 43 --output_dir experiments/paper_formal/E3_adapters
 run_exp 1 E2__C0__16shot__seed43 \
  --lora_type lora_nsp --init_mode lora_nsp \
  --fd_weight 0 --cd_weight 0 \
  --seed 43 --output_dir experiments/paper_formal/E2_components
 run_exp 1 E2__C2__16shot__seed43 \
  --lora_type lora_nsp --init_mode lora_nsp \
  --fd_weight 0 --cd_weight 2 --cd_temperature 4.0 \
  --seed 43 --output_dir experiments/paper_formal/E2_components
) &

# ====== GPU 2 queue ======
(run_exp 2 E3__lora_null__16shot__seed44 \
  --lora_type lora_nsp --init_mode lora_nsp --null_init_mode history_init_only \
  --seed 44 --output_dir experiments/paper_formal/E3_adapters
 run_exp 2 E2__C0__16shot__seed44 \
  --lora_type lora_nsp --init_mode lora_nsp \
  --fd_weight 0 --cd_weight 0 \
  --seed 44 --output_dir experiments/paper_formal/E2_components
 run_exp 2 E2__C2__16shot__seed44 \
  --lora_type lora_nsp --init_mode lora_nsp \
  --fd_weight 0 --cd_weight 2 --cd_temperature 4.0 \
  --seed 44 --output_dir experiments/paper_formal/E2_components
) &

# ====== GPU 3 queue ======
(run_exp 3 E3__gradproj__16shot__seed42 \
  --lora_type lora_nsp --init_mode lora_nsp --use_gradient_projection \
  --seed 42 --output_dir experiments/paper_formal/E3_adapters
 run_exp 3 E2__C1__16shot__seed42 \
  --lora_type lora_nsp --init_mode lora_nsp \
  --fd_weight 1 --cd_weight 0 \
  --seed 42 --output_dir experiments/paper_formal/E2_components
) &

# ====== GPU 4 queue ======
(run_exp 4 E3__gradproj__16shot__seed43 \
  --lora_type lora_nsp --init_mode lora_nsp --use_gradient_projection \
  --seed 43 --output_dir experiments/paper_formal/E3_adapters
 run_exp 4 E2__C1__16shot__seed43 \
  --lora_type lora_nsp --init_mode lora_nsp \
  --fd_weight 1 --cd_weight 0 \
  --seed 43 --output_dir experiments/paper_formal/E2_components
) &

# ====== GPU 5 queue ======
(run_exp 5 E3__gradproj__16shot__seed44 \
  --lora_type lora_nsp --init_mode lora_nsp --use_gradient_projection \
  --seed 44 --output_dir experiments/paper_formal/E3_adapters
 run_exp 5 E2__C1__16shot__seed44 \
  --lora_type lora_nsp --init_mode lora_nsp \
  --fd_weight 1 --cd_weight 0 \
  --seed 44 --output_dir experiments/paper_formal/E2_components
) &

echo "Wave 1 chain launched (E3 LoRA-Null ×3 + GradProj ×3 + E2 C0/C1/C2 ×3)"
echo "Queue log: tail -f $LOGDIR/_queue.log"
wait
echo "=== Wave 1 ALL DONE ===" | tee -a "$LOGDIR/_queue.log"