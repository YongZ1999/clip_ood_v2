#!/bin/bash
# Tier 1: 快速筛选 — 投影/初始化方法对比
# 8种方法 × 3任务 × seed=42
# 用法: bash scripts/run_tier1.sh <gpu_id> [exp_id]
#   不指定 exp_id 时跑全部

GPU=${1:-1}
SINGLE_EXP=$2  # 可选：只跑单个实验

PYTHON="/home/raoxuan/ENTER/envs/raoxuan/bin/python"
BASE_DIR="/home/raoxuan/projects/project_clip_continual_learning"
BASE_CMD="cd ${BASE_DIR} && ${PYTHON} main_incremental.py \
  --dataset_sequence aircraft caltech101 dtd \
  --root /data1/open_datasets/X-TAIL \
  --gpu ${GPU} \
  --batch_size 32 \
  --iterations 800 \
  --num_workers 4 \
  --seed 42"

run_experiment() {
  local exp_id=$1
  local lora_type=$2
  local init_mode=$3
  local extra_args=$4
  local log_file="${BASE_DIR}/experiments/tier1_${exp_id}.log"
  
  echo "[$(date '+%H:%M:%S')] ===== Exp ${exp_id} ===== "
  echo "  lora_type=${lora_type}, init_mode=${init_mode}, extra=${extra_args}"
  
  # git tag 标记代码状态
  cd ${BASE_DIR}
  git tag -f "tier1_${exp_id}_start" 2>/dev/null
  
  # 运行
  eval "${BASE_CMD} --lora_type ${lora_type} --init_mode ${init_mode} ${extra_args}" > "${log_file}" 2>&1
  local exit_code=$?
  
  git diff "tier1_${exp_id}_start" > "${BASE_DIR}/experiments/tier1_${exp_id}.diff" 2>/dev/null
  
  if [ $exit_code -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ✅ Exp ${exp_id} OK! Log: tier1_${exp_id}.log"
  else
    echo "[$(date '+%H:%M:%S')] ❌ Exp ${exp_id} FAILED (exit=${exit_code})"
    tail -10 "${log_file}"
  fi
}

# ====== 实验矩阵 ======
# 注意: lora_type 决定模型类型(是否有 projection 能力)
#       init_mode 决定初始化/训练策略
#       weight_kind 仅对 lora_sgp 生效（soft projection 时）
#
# | ID | 方法                | lora_type     | init_mode         | extra              |
# |----|---------------------|--------------|-------------------|--------------------|
# | 1.1| LoRA Vanilla        | lora_vanilla | lora_vanilla      |                    |
# | 1.2| LoRA-NSP (baseline) | lora_nsp     | lora_nsp          |                    |
# | 1.3| Band-pass P         | lora_sgp     | lora_nsp          | --weight_kind band_pass |
# | 1.4| High-cut P          | lora_sgp     | lora_nsp          | --weight_kind high_cut  |
# | 1.5| Proj-Σ tail         | lora_nsp     | proj_sigma_tail   |                    |
# | 1.6| Proj-Σ middle       | lora_nsp     | proj_sigma_middle |                    |
# | 1.7| SVD-W tail          | lora_nsp     | weight_svd_tail   |                    |
# | 1.8| SVD-W middle        | lora_nsp     | weight_svd_middle |                    |

declare -A EXPERIMENTS
EXPERIMENTS=(
  ["1.1_lora_vanilla"]="lora_vanilla|lora_vanilla|"
  ["1.2_lora_nsp"]="lora_nsp|lora_nsp|"
  ["1.3_band_pass"]="lora_sgp|lora_nsp|--weight_kind band_pass"
  ["1.4_high_cut"]="lora_sgp|lora_nsp|--weight_kind high_cut"
  ["1.5_proj_sigma_tail"]="lora_nsp|proj_sigma_tail|"
  ["1.6_proj_sigma_middle"]="lora_nsp|proj_sigma_middle|"
  ["1.7_weight_svd_tail"]="lora_nsp|weight_svd_tail|"
  ["1.8_weight_svd_middle"]="lora_nsp|weight_svd_middle|"
)

echo "=========================================="
echo "Tier 1 Experiment Batch - GPU ${GPU}"
echo "Started: $(date)"
echo "=========================================="

# 如果指定了单个实验
if [ -n "$SINGLE_EXP" ]; then
  KEY="${SINGLE_EXP}"
  if [ -n "${EXPERIMENTS[$KEY]}" ]; then
    IFS='|' read -r lt im ea <<< "${EXPERIMENTS[$KEY]}"
    run_experiment "$KEY" "$lt" "$im" "$ea"
  else
    echo "Unknown experiment: ${SINGLE_EXP}"
    echo "Available: ${!EXPERIMENTS[@]}"
    exit 1
  fi
else
  # 顺序执行全部
  for key in "1.1_lora_vanilla" "1.2_lora_nsp" "1.3_band_pass" "1.4_high_cut" \
             "1.5_proj_sigma_tail" "1.6_proj_sigma_middle" \
             "1.7_weight_svd_tail" "1.8_weight_svd_middle"; do
    IFS='|' read -r lt im ea <<< "${EXPERIMENTS[$key]}"
    run_experiment "$key" "$lt" "$im" "$ea"
  done
fi

echo ""
echo "=========================================="
echo "Tier 1 Complete!"
echo "Finished: $(date)"
echo "=========================================="
