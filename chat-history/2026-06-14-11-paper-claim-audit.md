# Paper Claim Audit Script

Date: 2026-06-14

## Context

The paper draft contains old TODO-heavy sections hidden inside `\iffalse ...
\fi`. Simple `rg TODO` is too noisy because it reports hidden draft text.
The publication goal needs a mechanical check over visible paper text only, so
future edits do not accidentally reintroduce unsupported claims.

## Changes

Added `scripts/audit_paper_claims.py`.

The script:

- reads `paper_writing/paper-template/paper_draft.tex`;
- ignores text hidden inside `\iffalse ... \fi`;
- ignores comment-only lines;
- fails on visible `\todo{...}` commands;
- fails on visible overclaims such as unconditional state-of-the-art language,
  Bayes-optimal classifier language, adaptive-router claims, near-zero OOD
  confidence claims, or positive official-LADA-DPT equivalence claims;
- requires visible boundary text stating that:
  - LADA remains stronger under real 16-shot feature storage;
  - LR-RGDA is not a universal replacement for LADA;
  - compact GMM classifier rebuilding is not official LADA DPT;
  - the storage table counts replay-source storage, not final classifier
    parameters;
  - the fair classifier protocol uses `classifier_feature_transform=test`.

Updated `experiments/README_publication_repro.md` with the audit command and
the checks it performs.

## Verification

Ran:

```bash
python scripts/audit_paper_claims.py paper_writing/paper-template/paper_draft.tex
python -m py_compile scripts/audit_paper_claims.py
git diff --check
```

The audit passed:

```text
visible lines: 933
hidden lines ignored: 137
hidden TODO commands ignored: 16
CLAIM AUDIT PASSED
```

## Remote Status

Remote SSH remains unavailable:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

The joint training ablation remains pending.
