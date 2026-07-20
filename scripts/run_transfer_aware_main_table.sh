#!/usr/bin/env bash
# Run the transfer-aware E1 main table on a multi-GPU server.
#
# Required before launch: activate the project's Python environment, then run
# this script from the repository root.  It assigns one sequential worker to
# each GPU, so jobs beyond the number of GPUs are queued automatically.

set -uo pipefail

ROOT="${ROOT:-/data1/open_datasets/X-TAIL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/paper_transfer_aware}"
LOG_ROOT="${LOG_ROOT:-logs/paper_transfer_aware}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
RUN_STANDARD_LORA="${RUN_STANDARD_LORA:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RETRIEVAL_DATASETS="${RETRIEVAL_DATASETS:-mscoco_2014_5k}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_NAME="${MODEL_NAME:-openai/clip-vit-base-patch16}"

# The server's OpenAI CLIP cache provides pytorch_model.bin rather than
# model.safetensors.  These exports retain the known-good offline loading
# behavior, while src/models/clip.py also has a model-family-aware default.
export CLIP_USE_SAFETENSORS="${CLIP_USE_SAFETENSORS:-0}"
export CLIP_LOCAL_FILES_ONLY="${CLIP_LOCAL_FILES_ONLY:-1}"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ ${#GPUS[@]} -eq 0 || -z "${GPUS[0]}" ]]; then
  echo "GPU_IDS must contain at least one physical GPU id, e.g. 0,1,2,3,4,5" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/E1_main" "${LOG_ROOT}"

TASKS=(aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397)
SEEDS=(42 43 44)
JOBS=()

# The ordering gives GPUs 0-2 the 16-shot seeds and GPUs 3-5 the full-shot
# seeds in the first wave.  Each worker then picks up the matched Standard
# LoRA job in the second wave, so all six GPUs remain balanced.
for method in lora_nf lora; do
  for shot in 16shot fullshot; do
    for seed in "${SEEDS[@]}"; do
      JOBS+=("${method}:${shot}:${seed}")
    done
  done
done

if [[ "${RUN_STANDARD_LORA}" != "1" ]]; then
  JOBS=("${JOBS[@]:0:6}")
fi

run_one() {
  local physical_gpu="$1"
  local method="$2"
  local shot="$3"
  local seed="$4"
  local name="E1TA__${method}__${shot}__seed${seed}"
  local log_file="${LOG_ROOT}/${name}__gpu${physical_gpu}.log"
  local output_dir="${OUTPUT_ROOT}/E1_main"
  local -a cmd

  cmd=(
    "${PYTHON_BIN}" -u main_incremental.py
    --root "${ROOT}"
    --model_name "${MODEL_NAME}"
    --dataset_sequence "${TASKS[@]}"
    --batch_size 32
    --eval_batch_size 128
    --iterations 800
    --train_budget_mode uniform
    --optimizer adamw
    --lr 1e-4
    --weight_decay 3e-5
    --scheduler cosine_with_warmup
    --warmup_ratio 0.1
    --eta_min 0.0
    --use_dora false
    --lora_rank 4
    --lora_alpha 4
    --lora_dropout 0.0
    --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2
    --projection_param_mode full
    --reference_dataset flickr8k
    --reference_batch_size 32
    --fd_weight 1.0
    --cd_weight 2.0
    --cd_divergence kl_forward
    --cd_temperature 4.0
    --aux_weight 0.0
    --tune_vision_encoder true
    --tune_text_encoder true
    --text_lora_rank 4
    --text_tuning_schedule always
    --text_classifier_mode lada_hybrid
    --classifier_feature_transform test
    --rgda_rank 32
    --rgda_alpha1 0.2
    --rgda_alpha2 2.0
    --rgda_alpha3 0.5
    --num_centers 4
    --rgda_train_iter 200
    --rgda_train_lr 0.01
    --rgda_fit_source gmm_sample
    --alpha 0.05
    --ensemble_normalize maxshift
    --ensemble_routing zs_predicted_seen
    --eval_resize_mode preserve_aspect
    --no-alpha_sensitivity
    --enable_retrieval_eval
    --retrieval_datasets "${RETRIEVAL_DATASETS}"
    --retrieval_batch_size 128
    --retrieval_recall_ks 1,5,10
    --retrieval_max_images 0
    --disable_lada
    --seed "${seed}"
    --gpu 0
    --output_dir "${output_dir}"
    --experiment_name "${name}"
  )

  if [[ "${shot}" == "16shot" ]]; then
    cmd+=(--num_shots 16)
  else
    cmd+=(--full_shot)
  fi

  if [[ "${method}" == "lora_nf" ]]; then
    cmd+=(
      --lora_type lora_nsp
      --init_mode lora_nsp
      --null_init_mode none
      --nsp_eps 0.20
      --nsp_weight 0.02
    )
  elif [[ "${method}" == "lora" ]]; then
    cmd+=(
      --lora_type lora_vanilla
      --init_mode lora_vanilla
      --null_init_mode none
    )
  else
    echo "Unsupported method: ${method}" >&2
    return 2
  fi

  {
    echo "=== ${name} | physical GPU ${physical_gpu} | $(date '+%F %T %Z') ==="
    echo "git: $(git rev-parse --short HEAD 2>/dev/null || true)"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${physical_gpu}"
    printf '%q ' "${cmd[@]}"
    echo
  } | tee "${log_file}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: command logged but not executed." | tee -a "${log_file}"
    return 0
  fi

  # A process with one GPU exposed must use --gpu 0 inside that namespace.
  CUDA_VISIBLE_DEVICES="${physical_gpu}" "${cmd[@]}" >> "${log_file}" 2>&1
}

worker() {
  local worker_index="$1"
  local physical_gpu="${GPUS[$worker_index]}"
  local job_index job method shot seed
  local worker_status=0

  for ((job_index=worker_index; job_index<${#JOBS[@]}; job_index+=${#GPUS[@]})); do
    job="${JOBS[$job_index]}"
    IFS=':' read -r method shot seed <<< "${job}"
    if ! run_one "${physical_gpu}" "${method}" "${shot}" "${seed}"; then
      echo "FAILED: ${job} on GPU ${physical_gpu}; continuing with its queued job." \
        | tee -a "${LOG_ROOT}/launcher.log"
      worker_status=1
    else
      echo "DONE: ${job} on GPU ${physical_gpu}" | tee -a "${LOG_ROOT}/launcher.log"
    fi
  done
  return "${worker_status}"
}

{
  echo "=== transfer-aware E1 launcher | $(date '+%F %T %Z') ==="
  echo "branch: $(git branch --show-current 2>/dev/null || true)"
  echo "commit: $(git rev-parse HEAD 2>/dev/null || true)"
  echo "GPUs: ${GPU_IDS}; jobs: ${#JOBS[@]}; RUN_STANDARD_LORA=${RUN_STANDARD_LORA}; DRY_RUN=${DRY_RUN}"
  echo "model: ${MODEL_NAME}; CLIP_USE_SAFETENSORS=${CLIP_USE_SAFETENSORS}; CLIP_LOCAL_FILES_ONLY=${CLIP_LOCAL_FILES_ONLY}"
  echo "Protocol: preserve_aspect + zs_predicted_seen; classifier settings remain mc4ft200."
} | tee "${LOG_ROOT}/launcher.log"

PIDS=()
for worker_index in "${!GPUS[@]}"; do
  worker "${worker_index}" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || status=1
done

if [[ "${status}" -eq 0 ]]; then
  echo "All transfer-aware E1 jobs completed." | tee -a "${LOG_ROOT}/launcher.log"
else
  echo "One or more transfer-aware E1 jobs failed; inspect ${LOG_ROOT}." \
    | tee -a "${LOG_ROOT}/launcher.log"
fi
exit "${status}"
