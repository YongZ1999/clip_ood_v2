#!/usr/bin/env bash
set -euo pipefail

RUN_LATEX="${RUN_LATEX:-0}"
STRICT_GATES="${STRICT_GATES:-0}"

echo "[verify] paper claim audit"
python scripts/audit_paper_claims.py \
  paper_writing/paper-template/paper_draft.tex \
  experiments/README_publication_repro.md

echo "[verify] result ledger audit"
python scripts/audit_result_ledger.py

echo "[verify] protocol config audit"
python scripts/audit_protocol_config.py

echo "[verify] training ablation audit"
python scripts/audit_training_ablation.py --allow_missing

echo "[verify] incremental metrics audit"
python scripts/audit_incremental_metrics.py --allow_missing

echo "[verify] publication gates"
python scripts/audit_publication_gates.py \
  --output_json experiments/publication_gate_status.json \
  --output_markdown experiments/publication_gate_status.md

echo "[verify] python syntax"
python -m py_compile \
  scripts/audit_incremental_metrics.py \
  scripts/audit_paper_claims.py \
  scripts/audit_publication_gates.py \
  scripts/audit_protocol_config.py \
  scripts/audit_result_ledger.py \
  scripts/audit_training_ablation.py \
  scripts/selftest_publication_audits.py \
  scripts/selftest_publication_launchers.py \
  scripts/selftest_remote_package.py \
  scripts/summarize_incremental_metrics.py \
  scripts/summarize_joint_classifier_replay.py \
  main_joint.py \
  src/experiments/run_continual_learning.py \
  src/models/clip.py \
  src/models/trainer.py \
  src/trainers/lora_nsp_trainer.py \
  src/classifiers/gaussian_statistics.py \
  src/lada/lada_classifier.py

echo "[verify] shell syntax"
bash -n scripts/run_joint_classifier_replay.sh
bash -n scripts/run_joint_training_ablation.sh
bash -n scripts/run_joint_incremental_ablation.sh
bash -n scripts/run_incremental_text_schedule_full_gate.sh
bash -n scripts/run_lada_dpt_ablation.sh
bash -n scripts/postprocess_publication_results.sh
bash -n scripts/remote_training_ablation.sh

echo "[verify] ledger json"
python -m json.tool experiments/result_ledger.json >/dev/null

echo "[verify] publication audit selftest"
python scripts/selftest_publication_audits.py

echo "[verify] publication launcher selftest"
python scripts/selftest_publication_launchers.py

echo "[verify] remote package selftest"
python scripts/selftest_remote_package.py

echo "[verify] whitespace"
git diff --check

if [[ "${RUN_LATEX}" == "1" ]]; then
  echo "[verify] latex"
  (
    cd paper_writing/paper-template
    pdflatex -interaction=nonstopmode -halt-on-error paper_draft.tex
    pdflatex -interaction=nonstopmode -halt-on-error paper_draft.tex
    if grep -q "destination with the same identifier" paper_draft.log; then
      echo "duplicate PDF destination warning found in paper_draft.log" >&2
      exit 1
    fi
    if grep -Eq "LaTeX Warning: (Reference|Citation).*undefined|There were undefined references|Citation .* undefined" paper_draft.log; then
      echo "undefined LaTeX reference or citation found in paper_draft.log" >&2
      exit 1
    fi
    if grep -Eq "Label\\(s\\) may have changed|Rerun to get cross-references right|Rerun to get outlines right" paper_draft.log; then
      echo "LaTeX rerun warning remained after two pdflatex passes" >&2
      exit 1
    fi
  )
fi

if [[ "${STRICT_GATES}" == "1" ]]; then
  echo "[verify] strict publication gates"
  python scripts/audit_publication_gates.py \
    --strict \
    --output_json experiments/publication_gate_status.json \
    --output_markdown experiments/publication_gate_status.md
fi

echo "[verify] publication package checks passed"
