#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
ROOT="${ROOT:-/data1/open_datasets/X-TAIL}"
SEEDS="${SEEDS:-42 43 44}"
OUT_DIR="${OUT_DIR:-experiments/joint_classifier_replay}"
RUN_MEAN_ABLATION="${RUN_MEAN_ABLATION:-1}"

COMMON_ARGS=(
  --id_datasets ALL
  --ood_datasets
  --root "${ROOT}"
  --gpu "${GPU}"
  --iterations 0
  --num_shots 16
  --batch_size 64
  --enable_lada
  --lada_k 16
  --lada_beta 1.0
  --lada_train_iter 200
  --lada_train_lr 0.01
  --num_centers 4
  --rgda_rank 32
  --rgda_train_iter 200
  --rgda_train_lr 0.01
  --gmm_k 4
  --gaussian_samples_per_class 16
  --output_dir "${OUT_DIR}"
)

mkdir -p "${OUT_DIR}/logs"

run_one() {
  local name="$1"
  local seed="$2"
  shift 2
  echo "[$(date '+%F %T')] ${name}, seed=${seed}"
  python main_joint.py \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" \
    --experiment_name "${name}" \
    "$@" \
    > "${OUT_DIR}/logs/${name}_seed${seed}.log" 2>&1
}

for seed in ${SEEDS}; do
  run_one real "${seed}"
  run_one gmm_raw "${seed}" \
    --use_gaussian_features --gmm_fit_space raw --gmm_sample_mode sample
  run_one gmm_sphere "${seed}" \
    --use_gaussian_features --gmm_fit_space sphere --gmm_sample_mode sample
  if [[ "${RUN_MEAN_ABLATION}" == "1" ]]; then
    run_one gmm_raw_mean "${seed}" \
      --use_gaussian_features --gmm_fit_space raw --gmm_sample_mode mean
  fi
done

python scripts/summarize_joint_classifier_replay.py \
  --input_dir "${OUT_DIR}" \
  --output_csv "${OUT_DIR}/summary.csv" \
  --output_markdown "${OUT_DIR}/summary.md"

