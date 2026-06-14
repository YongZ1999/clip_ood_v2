#!/usr/bin/env bash
set -euo pipefail

TRAINING_DIR="${TRAINING_DIR:-experiments/joint_training_ablation_20260614}"
INCREMENTAL_DIR="${INCREMENTAL_DIR:-experiments/joint_incremental_metrics_20260614}"
EXPECTED_SEEDS="${EXPECTED_SEEDS:-42 43 44}"
EXPECTED_TASKS="${EXPECTED_TASKS:-aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

mkdir -p "${TRAINING_DIR}" "${INCREMENTAL_DIR}"

if compgen -G "${TRAINING_DIR}/*_seed*.json" >/dev/null; then
  echo "[postprocess] classifier-side training ablation summary"
  python scripts/summarize_joint_classifier_replay.py \
    --input_dir "${TRAINING_DIR}" \
    --output_csv "${TRAINING_DIR}/summary.csv" \
    --output_markdown "${TRAINING_DIR}/summary.md" \
    --expected_seeds "${EXPECTED_SEEDS}"

  echo "[postprocess] classifier-side training ablation audit"
  if [[ "${ALLOW_MISSING}" == "1" ]]; then
    python scripts/audit_training_ablation.py \
      --summary_csv "${TRAINING_DIR}/summary.csv" \
      --expected_seeds "${EXPECTED_SEEDS}" \
      --allow_missing
  else
    python scripts/audit_training_ablation.py \
      --summary_csv "${TRAINING_DIR}/summary.csv" \
      --expected_seeds "${EXPECTED_SEEDS}"
  fi
else
  echo "[postprocess] classifier-side training ablation pending: no ${TRAINING_DIR}/*_seed*.json files found"
  if [[ "${ALLOW_MISSING}" != "1" ]]; then
    exit 1
  fi
fi

if compgen -G "${INCREMENTAL_DIR}/*_results.json" >/dev/null; then
  echo "[postprocess] strict incremental metrics summary"
  python scripts/summarize_incremental_metrics.py \
    "${INCREMENTAL_DIR}" \
    --output_csv "${INCREMENTAL_DIR}/incremental_summary.csv" \
    --output_markdown "${INCREMENTAL_DIR}/incremental_summary.md" \
    --aggregate_csv "${INCREMENTAL_DIR}/incremental_aggregate.csv" \
    --aggregate_markdown "${INCREMENTAL_DIR}/incremental_aggregate.md" \
    --expected_tasks "${EXPECTED_TASKS}"

  echo "[postprocess] strict incremental metrics audit"
  python scripts/audit_incremental_metrics.py \
    --summary_csv "${INCREMENTAL_DIR}/incremental_summary.csv" \
    --aggregate_csv "${INCREMENTAL_DIR}/incremental_aggregate.csv" \
    --expected_seeds "${EXPECTED_SEEDS}" \
    --expected_k 10
else
  echo "[postprocess] strict incremental metrics pending: no ${INCREMENTAL_DIR}/*_results.json files found"
  if [[ "${ALLOW_MISSING}" != "1" ]]; then
    exit 1
  fi
fi

echo "[postprocess] publication package verification"
if [[ "${ALLOW_MISSING}" == "1" ]]; then
  bash scripts/verify_publication_package.sh
else
  STRICT_GATES=1 bash scripts/verify_publication_package.sh
fi

if [[ "${ALLOW_MISSING}" == "1" ]]; then
  echo "[postprocess] strict publication gate skipped because ALLOW_MISSING=1"
else
  echo "[postprocess] strict publication gate"
  python scripts/audit_publication_gates.py \
    --strict \
    --output_json experiments/publication_gate_status.json \
    --output_markdown experiments/publication_gate_status.md
fi

echo "[postprocess] done"
