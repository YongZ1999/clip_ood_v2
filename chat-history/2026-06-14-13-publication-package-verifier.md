# Publication Package Verifier

Date: 2026-06-14

## Context

The reproducibility package now has several individual checks:

- paper claim audit;
- result ledger audit;
- Python syntax checks;
- launcher shell syntax checks;
- result ledger JSON validation;
- whitespace checks;
- optional LaTeX compile.

These need a single local entry point so the package can be checked before
remote launches, paper edits, or final claim audits.

## Changes

Added `scripts/verify_publication_package.sh`.

Default checks:

```bash
bash scripts/verify_publication_package.sh
```

The script runs:

- `python scripts/audit_paper_claims.py paper_writing/paper-template/paper_draft.tex`
- `python scripts/audit_result_ledger.py`
- `python -m py_compile` on audit/summarizer/core touched Python files
- `bash -n` on replay/training/DPT launchers
- `python -m json.tool experiments/result_ledger.json`
- `git diff --check`

Optional PDF check:

```bash
RUN_LATEX=1 bash scripts/verify_publication_package.sh
```

Updated `experiments/README_publication_repro.md` with both commands.

## Verification

Ran:

```bash
bash scripts/verify_publication_package.sh
```

It passed:

```text
CLAIM AUDIT PASSED
RESULT LEDGER AUDIT PASSED
[verify] publication package checks passed
```
