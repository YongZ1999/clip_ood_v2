#!/bin/bash
# ALL remaining experiments — balanced across 6 GPUs (6-7 runs each)
set -euo pipefail

LOGDIR="logs/paper_formal"
mkdir -p "$LOGDIR"
ROOT="/data1/open_datasets/X-TAIL"
DS="aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"
Q="$LOGDIR/_all_queue.log"

C="--root $ROOT --dataset_sequence $DS --batch_size 32 --eval_batch_size 128 --iterations 800 --train_budget_mode uniform --optimizer adamw --lr 1e-4 --weight_decay 3e-5 --scheduler cosine_with_warmup --warmup_ratio 0.1 --eta_min 0.0 --lora_rank 4 --lora_alpha 4 --lora_dropout 0.0 --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 --projection_param_mode full --null_init_mode none --nsp_eps 0.20 --nsp_weight 0.02 --reference_dataset flickr8k --reference_batch_size 32 --fd_weight 1.0 --cd_weight 2.0 --cd_divergence kl_forward --cd_temperature 4.0 --aux_weight 0.0 --tune_vision_encoder true --tune_text_encoder true --text_lora_rank 4 --text_tuning_schedule always --text_classifier_mode lada_hybrid --classifier_feature_transform test --rgda_rank 32 --rgda_alpha1 0.2 --rgda_alpha2 2.0 --rgda_alpha3 0.5 --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 --rgda_fit_source gmm_sample --alpha 0.05 --ensemble_normalize maxshift --temperature 1.0 --enable_retrieval_eval --retrieval_datasets mscoco_2014_5k,flickr30k_hf --retrieval_batch_size 128 --retrieval_recall_ks 1,5,10 --retrieval_max_images 0 --disable_lada"

run() {
    local gpu=$1 name=$2; shift 2
    echo "[$(date '+%m-%d %H:%M')] GPU$gpu START $name" | tee -a "$Q"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u main_incremental.py $C --gpu 0 "$@" \
      --experiment_name "$name" > "$LOGDIR/${name}.log" 2>&1
    echo "[$(date '+%m-%d %H:%M')] GPU$gpu DONE  $name (exit=$?)" | tee -a "$Q"
}

# ============ Shared experiment definitions ============
# E3 methods
E3_NULL='--num_shots 16 --output_dir experiments/paper_formal/E3_adapters --lora_type lora_nsp --init_mode lora_nsp --null_init_mode history_init_only'
E3_GP='--num_shots 16 --output_dir experiments/paper_formal/E3_adapters --lora_type lora_nsp --init_mode lora_nsp --use_gradient_projection'

# E2 components
E2_C0='--num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 0'
E2_C1='--num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 1 --cd_weight 0'
E2_C2='--num_shots 16 --output_dir experiments/paper_formal/E2_components --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0 --cd_weight 2 --cd_temperature 4.0'

# E1 full-shot
E1_FS_LORA='--full_shot --output_dir experiments/paper_formal/E1_main --lora_type lora_vanilla --init_mode lora_vanilla'
E1_FS_NF='--full_shot --output_dir experiments/paper_formal/E1_main --lora_type lora_nsp --init_mode lora_nsp'

# E4 nsp_eps
E4_EPS='--num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp'
# E4 nsp_weight
E4_WT='--num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp'
# E4 layers
E4_LAYER='--num_shots 16 --output_dir experiments/paper_formal/E4_lora_nf_hparams --lora_type lora_nsp --init_mode lora_nsp'

# E5
E5_CD='--num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0'
E5_TEMP='--num_shots 16 --output_dir experiments/paper_formal/E5_distillation --lora_type lora_nsp --init_mode lora_nsp --fd_weight 0'

# ============ GPU 0 (7 runs) ============
(run 0 E3__lora_null__16shot__seed42 $E3_NULL --seed 42
 run 0 E2__C0__16shot__seed42       $E2_C0 --seed 42
 run 0 E1__lora__fullshot__seed42   $E1_FS_LORA --seed 42
 run 0 E1__lora_nf__fullshot__seed43 $E1_FS_NF --seed 43
 run 0 E4__eps0p02__seed43          $E4_EPS --seed 43 --nsp_eps 0.02
 run 0 E4__weight0p04__seed43       $E4_WT --seed 43 --nsp_weight 0.04
 run 0 E5__cd_temp8p0__seed43       $E5_TEMP --seed 43 --cd_temperature 8.0 --cd_weight 2.0
) &

# ============ GPU 1 (7 runs) ============
(run 1 E3__gradproj__16shot__seed42 $E3_GP --seed 42
 run 1 E2__C1__16shot__seed42       $E2_C1 --seed 42
 run 1 E1__lora__fullshot__seed43   $E1_FS_LORA --seed 43
 run 1 E1__lora_nf__fullshot__seed44 $E1_FS_NF --seed 44
 run 1 E4__eps0p05__seed43          $E4_EPS --seed 43 --nsp_eps 0.05
 run 1 E4__weight0p08__seed43       $E4_WT --seed 43 --nsp_weight 0.08
 run 1 E5__cd_temp1p0__seed43       $E5_TEMP --seed 43 --cd_temperature 1.0 --cd_weight 2.0
) &

# ============ GPU 2 (6 runs) ============
(run 2 E3__lora_null__16shot__seed43 $E3_NULL --seed 43
 run 2 E2__C2__16shot__seed42       $E2_C2 --seed 42
 run 2 E1__lora__fullshot__seed44   $E1_FS_LORA --seed 44
 run 2 E4__eps0p10__seed43          $E4_EPS --seed 43 --nsp_eps 0.10
 run 2 E4__weight0p16__seed43       $E4_WT --seed 43 --nsp_weight 0.16
 run 2 E5__cd_temp2p0__seed43       $E5_TEMP --seed 43 --cd_temperature 2.0 --cd_weight 2.0
) &

# ============ GPU 3 (6 runs) ============
(run 3 E3__gradproj__16shot__seed43 $E3_GP --seed 43
 run 3 E2__C1__16shot__seed43       $E2_C1 --seed 43
 run 3 E1__lora_nf__fullshot__seed42 $E1_FS_NF --seed 42
 run 3 E4__weight0__seed43          $E4_WT --seed 43 --nsp_weight 0
 run 3 E4__layers_attn__seed43      $E4_LAYER --seed 43 --lora_target_modules q_proj,k_proj,v_proj,out_proj
 run 3 E5__cd_weight0__seed43       $E5_CD --seed 43 --cd_weight 0
) &

# ============ GPU 4 (6 runs) ============
(run 4 E3__lora_null__16shot__seed44 $E3_NULL --seed 44
 run 4 E2__C0__16shot__seed43       $E2_C0 --seed 43
 run 4 E2__C2__16shot__seed43       $E2_C2 --seed 43
 run 4 E4__layers_ffn__seed43       $E4_LAYER --seed 43 --lora_target_modules fc1,fc2
 run 4 E5__cd_weight0p5__seed43     $E5_CD --seed 43 --cd_weight 0.5
 run 4 E5__cd_weight4p0__seed43     $E5_CD --seed 43 --cd_weight 4.0
) &

# ============ GPU 5 (5 runs) ============
(run 5 E3__gradproj__16shot__seed44 $E3_GP --seed 44
 run 5 E2__C1__16shot__seed44       $E2_C1 --seed 44
 run 5 E2__C0__16shot__seed44       $E2_C0 --seed 44
 run 5 E2__C2__16shot__seed44       $E2_C2 --seed 44
 run 5 E5__cd_weight1p0__seed43     $E5_CD --seed 43 --cd_weight 1.0
) &

echo "=== ALL 37 runs queued, balanced ==="
echo "  GPU0:7  GPU1:7  GPU2:6  GPU3:6  GPU4:6  GPU5:5"
echo "  Monitor: tail -f $Q"
wait
echo "=== ALL DONE ===" | tee -a "$Q"