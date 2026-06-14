# LaTeX Duplicate Destination Fix

**Date**: 2026-06-14

## Context

The paper compiled successfully, but LaTeX emitted duplicate PDF destination
warnings for table anchors:

```text
destination with the same identifier (name{table.1}) has been already used
destination with the same identifier (name{table.2}) has been already used
destination with the same identifier (name{table.3}) has been already used
```

This was not a publication-gate blocker, but it is a PDF hygiene issue and can
affect internal links.

## Changes

Updated:

```text
paper_writing/paper-template/paper_draft.tex
```

Changes:

- load hyperref with `hypertexnames=false`;
- define a stable table hyperlink counter prefix via `\theHtable`.

This makes table destinations unique under the current NeurIPS-style table
environment and removes the duplicate destination warnings.

## Verification

Ran direct LaTeX compile:

```bash
cd paper_writing/paper-template
pdflatex -interaction=nonstopmode -halt-on-error paper_draft.tex
```

Confirmed no duplicate destination warnings remain.

Ran full verifier:

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

Remaining LaTeX warnings are nonfatal layout/font warnings:

- font substitution for `OMS/cmtt/m/n`;
- overfull/underfull boxes;
- `h` float specifier changed to `ht`.

Remote status was queried once and still failed with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```
