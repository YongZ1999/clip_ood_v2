# LADA DPT Boundary Gate

Date: 2026-06-14

## Context

The LADA official DPT distinction is one of the highest-risk claim-boundary
issues. It was already checked inside the paper claim audit, but it should also
appear as an explicit publication gate in the dashboard.

## Changes

Updated `scripts/audit_publication_gates.py`:

- added `LADA official DPT boundary` gate;
- checks visible paper text for:
  - compact GMM classifier rebuilding is not equivalent to official LADA DPT;
  - the current experiment does not conflate classifier rebuilding with
    training-time replay dynamics;
- checks `experiments/README_publication_repro.md` for:
  - `Relation to Official LADA DPT`;
  - current replay matrix is not official DPT reproduction;
  - claims should not say the current experiments reproduce official DPT.

Updated `experiments/README_publication_repro.md`:

- dashboard description now lists the independent LADA-official-DPT boundary
  gate.

## Verification

Ran:

```bash
python scripts/audit_publication_gates.py \
  --output_json /private/tmp/publication_gate_status_dpt_20260614/status.json \
  --output_markdown /private/tmp/publication_gate_status_dpt_20260614/status.md
python -m py_compile scripts/audit_publication_gates.py
```

Current dashboard:

```text
Paper claim boundary: PASS
Inference replay result ledger: PASS
LADA official DPT boundary: PASS
Training-side LoRA-NSP ablation: PENDING
Strict incremental Transfer/Average/Last/forgetting: PENDING
Reproducibility scripts and README: PASS

Summary: PASS=4, PENDING=2
```
