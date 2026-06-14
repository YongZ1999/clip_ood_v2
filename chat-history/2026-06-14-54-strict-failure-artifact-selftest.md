# Strict Failure Artifact Self-Test

**Date**: 2026-06-14

## Context

The publication goal remains active. The final postprocess path now writes
strict gate artifacts, and the self-test already covered the strict success
artifact path through a self-test-only override.

One more artifact behavior mattered for reproducibility: if the final strict
dashboard fails because a publication gate is still pending or failed, it should
still leave a `strict: true` JSON snapshot that records the failing state. That
prevents remote failures from being visible only in terminal logs.

## Changes

Updated:

```text
scripts/selftest_publication_audits.py
experiments/README_publication_repro.md
```

The publication audit self-test now runs:

```bash
python scripts/audit_publication_gates.py \
  --strict \
  --output_json <tmp>/gate_status_strict_fail.json \
  --output_markdown <tmp>/gate_status_strict_fail.md
```

without the self-test override. The command must exit nonzero under the current
missing-result state, but it must still write a JSON artifact with:

```text
strict: true
summary containing PENDING
```

The README now documents that final strict dashboard failures still write a
`strict: true` snapshot before exiting nonzero.

## Verification

Ran:

```bash
python scripts/selftest_publication_audits.py
python -m py_compile scripts/selftest_publication_audits.py
bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
PUBLICATION AUDIT SELFTEST PASSED
publication package checks passed
strict: False
summary: {'PASS': 5, 'PENDING': 2}
Summary: PASS=5, PENDING=2
```

Remote status still failed from this local environment:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

## Next Action

When SSH access works, run:

```bash
bash scripts/remote_training_ablation.sh upload
bash scripts/remote_training_ablation.sh launch_all
bash scripts/remote_training_ablation.sh status_all
bash scripts/remote_training_ablation.sh postprocess
```

The goal remains incomplete until strict publication gates pass with real
training-side and incremental result summaries and the final status artifact
records `strict: true`.
