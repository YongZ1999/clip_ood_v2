# LaTeX Verifier Check

**Date**: 2026-06-14

## Context

Most recent work strengthened the reproducibility package and audit scripts.
The publication goal also requires the paper draft itself to remain buildable.
The package verifier was therefore run with PDF compilation enabled.

## Verification

Ran:

```bash
RUN_LATEX=1 bash scripts/verify_publication_package.sh
```

Result:

```text
publication package checks passed
Output written on paper_draft.pdf (17 pages, 281851 bytes).
```

The verifier still reports:

```text
Summary: PASS=5, PENDING=2
```

The two pending gates remain:

- training-side LoRA-NSP ablation;
- strict incremental Transfer/Average/Last/forgetting.

## Nonfatal LaTeX Warnings

The PDF build succeeds. Current nonfatal warnings include:

- font substitution for `OMS/cmtt/m/n`;
- overfull/underfull boxes;
- duplicate destination warnings for table anchors;
- `h` float specifier changed to `ht`.

These are layout/LaTeX hygiene issues, not current blockers for the publication
gate package. They can be cleaned before submission polishing, after the missing
remote result gates are resolved.

Remote status was queried once and still failed with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```
