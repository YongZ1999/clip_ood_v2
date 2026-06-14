# Publication Gate Dashboard

Date: 2026-06-14

## Context

The package now has many individual audits. A single dashboard is useful for
showing which publication gates are already satisfied and which remain pending.

## Changes

Added `scripts/audit_publication_gates.py`.

The script checks:

- paper claim boundary via `audit_paper_claims.py`;
- inference replay result ledger via `audit_result_ledger.py`;
- training-side LoRA-NSP ablation via `audit_training_ablation.py --allow_missing`;
- strict incremental Transfer/Average/Last/forgetting via
  `audit_incremental_metrics.py --allow_missing`;
- required reproducibility scripts and README presence.

Default mode reports pending gates but exits zero unless a gate fails. Strict
mode exits nonzero if any gate is pending:

```bash
python scripts/audit_publication_gates.py --strict
```

Updated `scripts/verify_publication_package.sh` to run the gate dashboard.

Updated `experiments/README_publication_repro.md` with dashboard commands and
strict-mode usage.

## Verification

Ran:

```bash
python scripts/audit_publication_gates.py \
  --output_json /private/tmp/publication_gate_status_20260614/status.json \
  --output_markdown /private/tmp/publication_gate_status_20260614/status.md
python -m py_compile scripts/audit_publication_gates.py
```

Output:

```text
Paper claim boundary: PASS
Inference replay result ledger: PASS
Training-side LoRA-NSP ablation: PENDING
Strict incremental Transfer/Average/Last/forgetting: PENDING
Reproducibility scripts and README: PASS

Summary: PASS=3, PENDING=2
```

This matches the current state: inference-side evidence and paper boundaries are
audited locally; training-side and strict incremental results still require
remote execution.
