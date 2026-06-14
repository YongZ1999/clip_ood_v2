# Incremental Task Order Audit

**Date**: 2026-06-14

## Context

The publication goal remains active. The strict incremental gate already checked
that all expected tasks were present, that the result matrix had the expected
shape, and that stored Transfer/Average/Last metrics matched recomputation from
the accuracy matrix.

However, a result with the correct task set but the wrong task order could still
pass those checks. That would change the meaning of the incremental matrix and
therefore the Transfer/Average/Last evidence.

## Changes

Updated:

```text
scripts/summarize_incremental_metrics.py
scripts/audit_incremental_metrics.py
scripts/selftest_publication_audits.py
experiments/README_publication_repro.md
```

The incremental summarizer now writes a `task order warning` column in both
per-run and aggregate summaries when the task set matches but the order differs
from the expected X-TAIL sequence.

The strict incremental audit now fails if:

- the per-run or aggregate summaries are missing the `task order warning` column;
- any row reports a task-order warning.

The publication audit self-test now includes a synthetic shuffled-task fixture
that has all ten expected tasks but in reverse order. The fixture must fail the
strict incremental audit.

The README now documents that strict incremental evidence requires the expected
task order, not only the expected task set.

## Verification

Ran:

```bash
python scripts/selftest_publication_audits.py
python -m py_compile \
  scripts/summarize_incremental_metrics.py \
  scripts/audit_incremental_metrics.py \
  scripts/selftest_publication_audits.py
bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
PUBLICATION AUDIT SELFTEST PASSED
publication package checks passed
Summary: PASS=5, PENDING=2
```

Remote status still failed from this local environment:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

## Next Action

The remaining true completion work still requires remote execution:

```bash
bash scripts/remote_training_ablation.sh upload
bash scripts/remote_training_ablation.sh launch_all
bash scripts/remote_training_ablation.sh status_all
bash scripts/remote_training_ablation.sh postprocess
```

The goal remains incomplete until the strict dashboard passes with real
training-side and incremental summaries.
