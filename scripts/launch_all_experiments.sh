#!/bin/bash
# ALL experiments — Wave 1+2+3, balanced across 6 GPUs
set -euo pipefail

LOGDIR="logs/paper_formal"
mkdir -p "$LOGDIR"
ROOT="/data1/open_datasets/X-TAIL"
DS="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"
Q="$LOGDIR/_all_queue.log"

# --- common base for main_incremental.py ---
C0="--root $ROOT --dataset_sequence $DS --batch_size 32 --eval_batch_size 128 --iterations 800 --train_budget_mode uniform --optimizer adamw --lr 1e-4 --weight_decay 3e-5 --scheduler cosine_with_warmup --warmup_ratio 0.1 --eta_min 0.0 --lora_rank 4 --lora_alpha 4 --lora_dropout 0.0 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 --projection_param_mode full --null_init_mode none --nsp_eps 0.20 --nsp_weight 0.02 --reference_dataset flickr8k --reference_batch_size 32 --fd_weight 1.0 --cd_weight 2.0 --cd_divergence kl_forward --cd_temperature 4.0 --aux_weight 0.0 --tune_vision_encoder true --tune_text_encoder true --text_lora_rank 4 --text_tuning_schedule always --text_classifier_mode lada_hybrid --classifier_feature_transform test --rgda_rank 32 --rgda_alpha1 0.2 --rgda_alpha2 2.0 --rgda_alpha3 0.5 --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 --rgda_fit_source gmm_sample --alpha 0.05 --ensemble_normalize maxshift --temperature 1.0 --enable_retrieval_eval --retrieval_datasets mscoco_2014_5k,flickr30k_hf --retrieval_batch_size 128 --retrieval_recall_ks 1,5,10 --retrieval_max_images 0 --disable_lada"

run() {
    local gpu=$1 name=$2; shift 2
    echo "[$(date '+%m-%d %H:%M')] GPU$gpu START $name" | tee -a "$Q"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u main_incremental.py $C0 --gpu 0 "$@" \
      --experiment_name "$name" > "$LOGDIR/${name}.log" 2>&1
    echo "[$(date '+%m-%d %H:%M')] GPU$gpu DONE  $name (exit=$?)" | tee -a "$Q"
}

# --- LADA experiments (separate entry point) ---
run_lada() {
    local gpu=$1 name=$2 seed=$3 shots=$4; shift 4
    local ns="--num_shots $shots"
    [ "$shots" = "full" ] && ns=""  # LADA script has no --full_shot, skip
    echo "[$(date '+%m-%d %H:%M')] GPU$gpu START $name" | tee -a "$Q"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/legacy/main_incremental_lada.py \
      --root "$ROOT" --dataset_sequence $DS --batch_size 32 --iterations 800 \
      --lora_type lora_nsp --lada_k 16 --lada_beta 1.0 --lada_alpha 1.0 \
      --enable_dpt --prototype_k 4 --seed "$seed" --gpu 0 \
      > "$LOGDIR/${name}.log" 2>&1
    echo "[$(date '+%m-%d %H:%M')] GPU$gpu DONE  $name (exit=$?)" | tee -a "$Q"
}

# ============================================================
# GPU 0 (8 runs)
# ============================================================
(run 0 E3__lora_null__s42 --num_shots 16 --output_dir experiments/paper_formal/E3_adapters --lora_type lora_nsp --init_mode lora_nsp --null_init_mode history_init_only --seed 42
 run 0 E2__C0__s42 --num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 0 --seed 42
 run 0 E1__lora__fs__s42 --full_shot --output_dir experiments/paper_formal/E1_main --lora_type lora_vanilla --init_mode lora_vanilla --seed 42
 run 0 E4__eps002__s43 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --nsp_eps 0.02 --seed 43
 run 0 E4__eps002__s44 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --nsp_eps 0.02 --seed 44
 run 0 E5__cd0__s44 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 0 --seed 44
 run 0 E5__cd4p0__s42 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 4.0 --seed 42
 run_lada 0 LADA__16shot__s42 42 16
) &

# ============================================================
# GPU 1 (8 runs)
# ============================================================
(run 1 E3__gradproj__s42 --num_shots 16 --output_dir experiments/paper_formal/E3_adapters --lora_type lora_nsp --init_mode lora_nsp --use_gradient_projection --seed 42
 run 1 E2__C1__s42 --num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 1 --cd_weight 0 --seed 42
 run 1 E1__lora_nf__fs__s42 --full_shot --output_dir experiments/paper_formal/E1_main --lora_type lora_nsp --init_mode lora_nsp --seed 42
 run 1 E4__eps005__s43 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --nsp_eps 0.05 --seed 43
 run 1 E4__weight0__s42 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --nsp_weight 0 --seed 42
 run 1 E4__weight0__s44 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --nsp_weight 0 --seed 44
 run 1 E5__cd0__s42 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 0 --seed 42
 run_lada 1 LADA__16shot__s43 43 16
) &

# ============================================================
# GPU 2 (8 runs)
# ============================================================
(run 2 E3__lora_null__s43 --num_shots 16 --output_dir experiments/paper_formal/E3_adapters --lora_type lora_nsp --init_mode lora_nsp --null_init_mode history_init_only --seed 43
 run 2 E2__C2__s42 --num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 2 --cd_temperature 4.0 --seed 42
 run 2 E1__lora__fs__s43 --full_shot --output_dir experiments/paper_formal/E1_main --lora_type lora_vanilla --init_mode lora_vanilla --seed 43
 run 2 E1__lora_nf__fs__s44 --full_shot --output_dir experiments/paper_formal/E1_main --lora_type lora_nsp --init_mode lora_nsp --seed 44
 run 2 E4__eps010__s43 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --nsp_eps 0.10 --seed 43
 run 2 E4__weight008__s43 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --nsp_weight 0.08 --seed 43
 run 2 E5__cd_temp1p0__s43 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_temperature 1.0 --cd_weight 2.0 --seed 43
 run_lada 2 LADA__16shot__s44 44 16
) &

# ============================================================
# GPU 3 (8 runs)
# ============================================================
(run 3 E3__gradproj__s43 --num_shots 16 --output_dir experiments/paper_formal/E3_adapters --lora_type lora_nsp --init_mode lora_nsp --use_gradient_projection --seed 43
 run 3 E2__C0__s43 --num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 0 --seed 43
 run 3 E1__lora__fs__s44 --full_shot --output_dir experiments/paper_formal/E1_main --lora_type lora_vanilla --init_mode lora_vanilla --seed 44
 run 3 E1__lora_nf__fs__s43 --full_shot --output_dir experiments/paper_formal/E1_main --lora_type lora_nsp --init_mode lora_nsp --seed 43
 run 3 E4__weight004__s43 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --nsp_weight 0.04 --seed 43
 run 3 E4__weight016__s43 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --nsp_weight 0.16 --seed 43
 run 3 E5__cd_weight4p0__s43 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 4.0 --seed 43
 run 3 E5__cd_temp8p0__s43 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 2.0 --cd_temperature 8.0 --seed 43
) &

# ============================================================
# GPU 4 (8 runs)
# ============================================================
(run 4 E3__lora_null__s44 --num_shots 16 --output_dir experiments/paper_formal/E3_adapters --lora_type lora_nsp --init_mode lora_nsp --null_init_mode history_init_only --seed 44
 run 4 E2__C0__s44 --num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 0 --seed 44
 run 4 E2__C2__s43 --num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 2 --cd_temperature 4.0 --seed 43
 run 4 E4__layers_attn__s43 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --lora_target_modules q_proj,k_proj,v_proj,out_proj --seed 43
 run 4 E4__layers_ffn__s43 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --lora_target_modules fc1,fc2 --seed 43
 run 4 E5__cd_weight0p5__s43 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 0.5 --seed 43
 run 4 E5__cd_weight1p0__s43 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 1.0 --seed 43
 run 4 E5__cd_temp2p0__s43 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 2.0 --cd_temperature 2.0 --seed 43
) &

# ============================================================
# GPU 5 (8 runs)
# ============================================================
(run 5 E3__gradproj__s44 --num_shots 16 --output_dir experiments/paper_formal/E3_adapters --lora_type lora_nsp --init_mode lora_nsp --use_gradient_projection --seed 44
 run 5 E2__C1__s43 --num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 1 --cd_weight 0 --seed 43
 run 5 E2__C1__s44 --num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 1 --cd_weight 0 --seed 44
 run 5 E2__C2__s44 --num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 2 --cd_temperature 4.0 --seed 44
 run 5 E4__weight0__s43 --num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp --nsp_weight 0 --seed 43
 run 5 E5__cd0__s43 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 0 --seed 43
 run 5 E5__cd_temp8p0__s42 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 2.0 --cd_temperature 8.0 --seed 42
 run 5 E5__cd_temp8p0__s44 --num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 2.0 --cd_temperature 8.0 --seed 44
) &

echo "=== ALL 48 runs queued (E1+E2+E3+E4+E5 + LADA + seed completion) ==="
echo "  Per GPU: 0:8, 1:8, 2:8, 3:8, 4:8, 5:8"
echo "  Monitor: tail -f $Q"
wait
echo "=== ALL EXPERIMENTS DONE ===" | tee -a "$Q"