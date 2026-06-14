# Strict Postprocess Artifact

**Date**: 2026-06-14

## Context

The publication goal remains active. The gate status JSON now records a `strict`
flag, but remote postprocess still had a gap: `verify_publication_package.sh`
wrote a normal non-strict gate snapshot, and the final strict dashboard command
only printed to the terminal.

That meant a successful final remote postprocess would not necessarily leave a
`strict: true` JSON/Markdown artifact behind.

## Changes

Updated:

```text
scripts/postprocess_publication_results.sh
scripts/selftest_remote_package.py
experiments/README_publication_repro.md
```

When `ALLOW_MISSING=0`, postprocess now runs the strict dashboard with explicit
artifact output paths:

```bash
python scripts/audit_publication_gates.py \
  --strict \
  --output_json experiments/publication_gate_status.json \
  --output_markdown experiments/publication_gate_status.md
```

The remote package self-test now checks that postprocess keeps the strict gate
and both artifact output paths.

The README now documents that:

- normal verifier runs write `strict: false`;
- successful final postprocess runs overwrite the status snapshot with
  `strict: true`.

## Verification

Ran:

```bash
python scripts/selftest_remote_package.py
bash -n scripts/postprocess_publication_results.sh
python -m py_compile scripts/selftest_remote_package.py
bash scripts/verify_publication_package.sh
ALLOW_MISSING=1 bash scripts/postprocess_publication_results.sh
python scripts/audit_publication_gates.py
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
REMOTE PACKAGE SELFTEST PASSED
publication package checks passed
ALLOW_MISSING=1 postprocess skipped strict publication gate as expected
schema_version: 1
strict: false
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
training-side and incremental result summaries and the status artifact records
`strict: true`.
