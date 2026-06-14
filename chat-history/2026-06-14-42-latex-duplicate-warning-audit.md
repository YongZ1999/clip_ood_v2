# LaTeX Duplicate Warning Audit

**Date**: 2026-06-14

## Context

The duplicate PDF destination warnings in the paper were fixed, but the verifier
only checked that `pdflatex` exited successfully. To prevent table/reference
anchor regressions, the LaTeX verifier now rejects duplicate destination
warnings explicitly.

## Changes

Updated:

```text
scripts/verify_publication_package.sh
experiments/README_publication_repro.md
```

When `RUN_LATEX=1`, the verifier now runs `pdflatex` and then checks
`paper_writing/paper-template/paper_draft.log` for:

```text
destination with the same identifier
```

If that warning appears, the verifier exits nonzero.

The README now documents that `RUN_LATEX=1` fails on compile errors and
duplicate PDF destination warnings.

## Verification

Ran:

```bash
RUN_LATEX=1 bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
git diff --check
```

Result:

```text
publication package checks passed
Output written on paper_draft.pdf (17 pages, 281891 bytes).
Summary: PASS=5, PENDING=2
```

No duplicate destination warnings are present under the new check.

Remote status was queried once and still failed with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```
