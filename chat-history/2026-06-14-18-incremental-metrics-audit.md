# Incremental Metrics Audit

Date: 2026-06-14

## Context

The strict incremental metrics summarizer can now produce per-run and aggregate
Transfer/Average/Last/forgetting tables. A separate audit is needed to enforce
publication gates once final multi-seed results are available.

## Changes

Added `scripts/audit_incremental_metrics.py`.

The audit checks:

- required methods are present:
  - `lora_vanilla`
  - `lora_nsp`
  - `lora_nsp_fd`
  - `lora_nsp_fd_cd`
- all expected seeds are present;
- each per-run row has the expected `K=10`;
- no expected tasks are missing;
- no `accuracy_matrix` shape warning is present;
- stored-vs-computed Transfer/Average/Last deltas are within tolerance;
- aggregate Transfer/Average/Last of `lora_nsp_fd_cd` are compared against
  `lora_vanilla` and `lora_nsp`.

The script supports `--allow_missing` so the package verifier can report the
incremental audit as pending until final results exist.

Updated `scripts/verify_publication_package.sh`:

- runs `python scripts/audit_incremental_metrics.py --allow_missing`;
- includes the script in Python syntax checks.

Updated `experiments/README_publication_repro.md` with the strict audit command.

## Verification

Created synthetic per-run and aggregate summaries under:

```text
/private/tmp/incremental_audit_synth_20260614
```

Ran:

```bash
python scripts/audit_incremental_metrics.py \
  --summary_csv /private/tmp/incremental_audit_synth_20260614/incremental_summary.csv \
  --aggregate_csv /private/tmp/incremental_audit_synth_20260614/incremental_aggregate.csv \
  --expected_seeds '42 43 44' \
  --expected_k 10
python scripts/audit_incremental_metrics.py \
  --summary_csv /private/tmp/incremental_audit_synth_20260614/missing_summary.csv \
  --aggregate_csv /private/tmp/incremental_audit_synth_20260614/missing_aggregate.csv \
  --allow_missing
bash scripts/verify_publication_package.sh
python -m py_compile scripts/audit_incremental_metrics.py
git diff --check
```

The strict synthetic audit passed. The missing-result path reports:

```text
INCREMENTAL METRICS AUDIT PENDING
```

The publication package verifier still passes while final incremental results
are absent.

## Bug Found and Fixed

Initial synthetic testing failed because `parse_score` did not accept leading
`+` signs such as `+0.0000`. This was fixed so stored-minus-computed deltas from
`summarize_incremental_metrics.py` parse correctly.
