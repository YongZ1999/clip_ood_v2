# Ledger Delta And Storage Audit

**Date**: 2026-06-14

## Context

The publication goal remains active. The result ledger audit already checked
that the main classifier table values appeared in the paper and README, but it
did not yet machine-check two important pieces of the current defensible claim:

- the per-seed margins over LADA for compact component-mean replay;
- the storage values derived from the ledger protocol.

Both are central to the storage-dependent claim boundary.

## Changes

Updated:

```text
experiments/README_publication_repro.md
paper_writing/paper-template/paper_draft.tex
scripts/audit_result_ledger.py
scripts/selftest_publication_audits.py
```

The README now includes an explicit per-seed margin table for the strongest
compact component-mean replay setting:

```text
LR-RGDA - LADA:     +0.96, +0.99, +0.59
LR-RGDA+ZS - LADA:  +1.18, +1.19, +0.74
```

The paper text now labels those two margin sequences explicitly, so the numbers
are not only present but attributable to the correct classifier comparison.

The result ledger audit now:

- checks that every ledger `per_seed_delta` value is positive;
- checks that rounded per-seed deltas appear in both the paper and README;
- recomputes storage values from the ledger protocol for supported formulas;
- fails if the ledger storage string disagrees with the computed MiB value.

The publication audit self-test now includes synthetic ledger fixtures that
verify negative per-seed deltas and incorrect storage values are rejected.

## Verification

Ran:

```bash
python scripts/audit_result_ledger.py
python scripts/selftest_publication_audits.py
python -m py_compile scripts/audit_result_ledger.py scripts/selftest_publication_audits.py
bash scripts/verify_publication_package.sh
RUN_LATEX=1 bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
RESULT LEDGER AUDIT PASSED
PUBLICATION AUDIT SELFTEST PASSED
publication package checks passed
Output written on paper_draft.pdf (17 pages, 281893 bytes).
Summary: PASS=5, PENDING=2
```

Remote status still failed from the local environment:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

## Next Action

The remaining true completion work is still remote:

```bash
bash scripts/remote_training_ablation.sh upload
bash scripts/remote_training_ablation.sh launch_all
bash scripts/remote_training_ablation.sh status_all
bash scripts/remote_training_ablation.sh postprocess
```

The goal remains incomplete until the strict dashboard passes with real
training-side and incremental summaries.
