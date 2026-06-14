#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-raoxuan@10.20.34.30}"
REMOTE_DIR="${REMOTE_DIR:-/home/raoxuan/projects/project_clip_continual_learning}"
OUT_DIR="${OUT_DIR:-experiments/joint_training_ablation_20260614}"
MASTER_LOG="${MASTER_LOG:-experiments/joint_training_ablation_20260614_master.log}"
INCREMENTAL_DIR="${INCREMENTAL_DIR:-experiments/joint_incremental_metrics_20260614}"
INCREMENTAL_LOG="${INCREMENTAL_LOG:-experiments/joint_incremental_metrics_20260614_master.log}"
PILOT_DIR="${PILOT_DIR:-experiments/incremental_training_pilot_20260614}"
PILOT_LOG="${PILOT_LOG:-experiments/incremental_training_pilot_20260614_master.log}"
TEXT_SCHEDULE_DIR="${TEXT_SCHEDULE_DIR:-experiments/incremental_text_schedule_pilot_20260614}"
TEXT_SCHEDULE_LOG="${TEXT_SCHEDULE_LOG:-experiments/incremental_text_schedule_pilot_20260614_master.log}"
TEXT_STRESS_DIR="${TEXT_STRESS_DIR:-experiments/incremental_text_schedule_stress_20260614}"
TEXT_STRESS_LOG="${TEXT_STRESS_LOG:-experiments/incremental_text_schedule_stress_20260614_master.log}"
TEXT_FULL_GATE_DIR="${TEXT_FULL_GATE_DIR:-experiments/incremental_text_schedule_full_gate_20260614}"
TEXT_FULL_GATE_LOG="${TEXT_FULL_GATE_LOG:-experiments/incremental_text_schedule_full_gate_20260614_master.log}"

SSH_OPTS=(
  -o ConnectTimeout=8
  -o BatchMode=yes
)

FILES=(
  main_joint.py
  scripts/remote_training_ablation.sh
  scripts/run_joint_classifier_replay.sh
  scripts/run_joint_training_ablation.sh
  scripts/run_joint_incremental_ablation.sh
  scripts/run_incremental_training_pilot.sh
  scripts/run_incremental_text_schedule_pilot.sh
  scripts/run_incremental_text_schedule_stress.sh
  scripts/run_incremental_text_schedule_full_gate.sh
  scripts/run_lada_dpt_ablation.sh
  scripts/summarize_joint_classifier_replay.py
  scripts/summarize_joint_alpha_sweep.py
  scripts/summarize_incremental_metrics.py
  scripts/audit_training_ablation.py
  scripts/audit_incremental_metrics.py
  scripts/audit_paper_claims.py
  scripts/audit_protocol_config.py
  scripts/audit_publication_gates.py
  scripts/audit_result_ledger.py
  scripts/postprocess_publication_results.sh
  scripts/selftest_publication_audits.py
  scripts/selftest_publication_launchers.py
  scripts/selftest_remote_package.py
  scripts/verify_publication_package.sh
  src/experiments/run_continual_learning.py
  src/classifiers/gaussian_statistics.py
  src/lada/lada_classifier.py
  src/utils/main_utils.py
  src/models/clip.py
  src/models/trainer.py
  src/trainers/lora_nsp_trainer.py
  paper_writing/paper-template/paper_draft.tex
  experiments/README_publication_repro.md
  experiments/result_ledger.json
)

usage() {
  cat <<EOF
Usage: $0 {upload|launch|status|launch_incremental|status_incremental|launch_incremental_pilot|status_incremental_pilot|launch_text_schedule_pilot|status_text_schedule_pilot|launch_text_schedule_stress|status_text_schedule_stress|launch_text_schedule_full_gate|status_text_schedule_full_gate|status_gates|launch_all|status_all|postprocess}

Environment:
  REMOTE=${REMOTE}
  REMOTE_DIR=${REMOTE_DIR}
  OUT_DIR=${OUT_DIR}
  MASTER_LOG=${MASTER_LOG}
  INCREMENTAL_DIR=${INCREMENTAL_DIR}
  INCREMENTAL_LOG=${INCREMENTAL_LOG}
  PILOT_DIR=${PILOT_DIR}
  PILOT_LOG=${PILOT_LOG}
  TEXT_SCHEDULE_DIR=${TEXT_SCHEDULE_DIR}
  TEXT_SCHEDULE_LOG=${TEXT_SCHEDULE_LOG}
  TEXT_STRESS_DIR=${TEXT_STRESS_DIR}
  TEXT_STRESS_LOG=${TEXT_STRESS_LOG}
  TEXT_FULL_GATE_DIR=${TEXT_FULL_GATE_DIR}
  TEXT_FULL_GATE_LOG=${TEXT_FULL_GATE_LOG}
EOF
}

remote_run() {
  ssh "${SSH_OPTS[@]}" "${REMOTE}" "$@"
}

upload() {
  python scripts/selftest_remote_package.py
  tar czf - "${FILES[@]}" | remote_run "cd '${REMOTE_DIR}' && tar xzf - && chmod +x scripts/run_joint_training_ablation.sh scripts/run_joint_incremental_ablation.sh scripts/run_incremental_training_pilot.sh scripts/run_incremental_text_schedule_pilot.sh scripts/run_incremental_text_schedule_stress.sh scripts/run_incremental_text_schedule_full_gate.sh scripts/postprocess_publication_results.sh scripts/verify_publication_package.sh"
}

launch() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    bash -n scripts/run_joint_training_ablation.sh &&
    OUT_DIR='${OUT_DIR}' nohup scripts/run_joint_training_ablation.sh > '${MASTER_LOG}' 2>&1 &
    sleep 3
    pgrep -af 'run_joint_training_ablation|main_joint.py' || true
  "
}

status() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    pgrep -af 'run_joint_training_ablation|main_joint.py' || true &&
    find '${OUT_DIR}' -maxdepth 1 -name '*_seed*.json' -printf '%f\n' 2>/dev/null | sort &&
    tail -40 '${OUT_DIR}/logs/launch.log' 2>/dev/null || true
  "
}

launch_incremental() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    bash -n scripts/run_joint_incremental_ablation.sh &&
    OUT_DIR='${INCREMENTAL_DIR}' nohup scripts/run_joint_incremental_ablation.sh > '${INCREMENTAL_LOG}' 2>&1 &
    sleep 3
    pgrep -af 'run_joint_incremental_ablation|run_continual_learning.py' || true
  "
}

status_incremental() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    pgrep -af 'run_joint_incremental_ablation|run_continual_learning.py' || true &&
    find '${INCREMENTAL_DIR}' -maxdepth 1 -name '*_results.json' -printf '%f\n' 2>/dev/null | sort &&
    tail -40 '${INCREMENTAL_DIR}/logs/launch.log' 2>/dev/null || true
  "
}

launch_incremental_pilot() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    bash -n scripts/run_incremental_training_pilot.sh &&
    OUT_DIR='${PILOT_DIR}' nohup scripts/run_incremental_training_pilot.sh > '${PILOT_LOG}' 2>&1 &
    sleep 3
    pgrep -af 'run_incremental_training_pilot|run_continual_learning.py' || true
  "
}

status_incremental_pilot() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    pgrep -af 'run_incremental_training_pilot|run_continual_learning.py' || true &&
    find '${PILOT_DIR}' -maxdepth 1 -name '*_results.json' -printf '%f\n' 2>/dev/null | sort &&
    tail -40 '${PILOT_DIR}/logs/launch.log' 2>/dev/null || true
  "
}

launch_text_schedule_pilot() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    bash -n scripts/run_incremental_text_schedule_pilot.sh &&
    OUT_DIR='${TEXT_SCHEDULE_DIR}' nohup scripts/run_incremental_text_schedule_pilot.sh > '${TEXT_SCHEDULE_LOG}' 2>&1 &
    sleep 3
    pgrep -af 'run_incremental_text_schedule_pilot|run_continual_learning.py' || true
  "
}

status_text_schedule_pilot() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    pgrep -af 'run_incremental_text_schedule_pilot|run_continual_learning.py' || true &&
    find '${TEXT_SCHEDULE_DIR}' -maxdepth 1 -name '*_results.json' -printf '%f\n' 2>/dev/null | sort &&
    tail -40 '${TEXT_SCHEDULE_DIR}/logs/launch.log' 2>/dev/null || true
  "
}

launch_text_schedule_stress() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    bash -n scripts/run_incremental_text_schedule_stress.sh &&
    OUT_DIR='${TEXT_STRESS_DIR}' nohup scripts/run_incremental_text_schedule_stress.sh > '${TEXT_STRESS_LOG}' 2>&1 &
    sleep 3
    pgrep -af 'run_incremental_text_schedule_stress|run_continual_learning.py' || true
  "
}

status_text_schedule_stress() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    pgrep -af 'run_incremental_text_schedule_stress|run_continual_learning.py' || true &&
    find '${TEXT_STRESS_DIR}' -maxdepth 1 -name '*_results.json' -printf '%f\n' 2>/dev/null | sort &&
    tail -40 '${TEXT_STRESS_DIR}/logs/launch.log' 2>/dev/null || true
  "
}

launch_text_schedule_full_gate() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    bash -n scripts/run_incremental_text_schedule_full_gate.sh &&
    OUT_DIR='${TEXT_FULL_GATE_DIR}' nohup scripts/run_incremental_text_schedule_full_gate.sh > '${TEXT_FULL_GATE_LOG}' 2>&1 &
    sleep 3
    pgrep -af 'run_incremental_text_schedule_full_gate|run_continual_learning.py' || true
  "
}

status_text_schedule_full_gate() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    pgrep -af 'run_incremental_text_schedule_full_gate|run_continual_learning.py' || true &&
    find '${TEXT_FULL_GATE_DIR}' -maxdepth 1 -name '*_results.json' -printf '%f\n' 2>/dev/null | sort &&
    tail -40 '${TEXT_FULL_GATE_DIR}/logs/launch.log' 2>/dev/null || true
  "
}

status_gates() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    if [[ -f experiments/publication_gate_status.json ]]; then
      python -c 'import json; d=json.load(open(\"experiments/publication_gate_status.json\")); print(\"strict:\", d.get(\"strict\")); print(\"summary:\", d.get(\"summary\")); print(\"generated_at_utc:\", d.get(\"generated_at_utc\"))'
    else
      echo 'experiments/publication_gate_status.json missing'
    fi
  "
}

launch_all() {
  launch
  launch_incremental
}

status_all() {
  echo "[remote] classifier-side training status"
  status
  echo "[remote] strict incremental status"
  status_incremental
  echo "[remote] incremental pilot status"
  status_incremental_pilot
  echo "[remote] text schedule pilot status"
  status_text_schedule_pilot
  echo "[remote] text schedule stress status"
  status_text_schedule_stress
  echo "[remote] text schedule full gate status"
  status_text_schedule_full_gate
  echo "[remote] publication gate status"
  status_gates
}

postprocess() {
  remote_run "
    cd '${REMOTE_DIR}' &&
    TRAINING_DIR='${OUT_DIR}' INCREMENTAL_DIR='${INCREMENTAL_DIR}' bash scripts/postprocess_publication_results.sh
  "
}

case "${1:-}" in
  upload)
    upload
    ;;
  launch)
    launch
    ;;
  status)
    status
    ;;
  launch_incremental)
    launch_incremental
    ;;
  status_incremental)
    status_incremental
    ;;
  launch_incremental_pilot)
    launch_incremental_pilot
    ;;
  status_incremental_pilot)
    status_incremental_pilot
    ;;
  launch_text_schedule_pilot)
    launch_text_schedule_pilot
    ;;
  status_text_schedule_pilot)
    status_text_schedule_pilot
    ;;
  launch_text_schedule_stress)
    launch_text_schedule_stress
    ;;
  status_text_schedule_stress)
    status_text_schedule_stress
    ;;
  launch_text_schedule_full_gate)
    launch_text_schedule_full_gate
    ;;
  status_text_schedule_full_gate)
    status_text_schedule_full_gate
    ;;
  status_gates)
    status_gates
    ;;
  launch_all)
    launch_all
    ;;
  status_all)
    status_all
    ;;
  postprocess)
    postprocess
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
