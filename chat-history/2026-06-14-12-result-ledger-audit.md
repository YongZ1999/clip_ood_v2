# Result Ledger Audit

Date: 2026-06-14

## Context

The paper draft and reproducibility README both contain completed inference-side
numbers. To make table traceability more explicit, the numbers need a
machine-readable ledger that maps paper rows to result directories, source JSON
files, seeds, protocol fields, and interpretation.

## Changes

Added `experiments/result_ledger.json`.

The ledger currently tracks:

- `tab:gmm_mean_replay`
  - real 16-shot features
  - GMM raw component means
  - GMM raw samples
  - GMM sphere samples
- `tab:storage_budget`
  - real 16-shot features
  - LADA `k=16` centers
  - GMM `k=4` component means
  - GMM `k=4` spherical parameters

For the replay matrix, each row records:

- paper label;
- README label;
- remote output directory;
- expected source JSON files;
- display values;
- protocol fields;
- claim interpretation;
- per-seed deltas for the strongest component-mean replay row.

Added `scripts/audit_result_ledger.py`.

The audit checks that:

- ledger JSON is parseable;
- listed paper table ids appear in the draft;
- replay display numbers appear in the paper and README;
- storage-budget numbers appear in the paper and README;
- storage-budget protocol explicitly marks that it counts replay-source storage,
  not final classifier parameters;
- key paper protocol boundary text remains visible.

Updated `experiments/README_publication_repro.md` with the result-ledger audit
command.

## Verification

Ran:

```bash
python scripts/audit_result_ledger.py
python -m py_compile scripts/audit_result_ledger.py
python -m json.tool experiments/result_ledger.json >/private/tmp/result_ledger_pretty.json
```

The ledger audit passed:

```text
audited ledger: experiments/result_ledger.json
tables: 2
RESULT LEDGER AUDIT PASSED
```

## Boundary

This ledger is a traceability aid. It does not replace re-summarizing the remote
JSON files when the remote result directories are available.
