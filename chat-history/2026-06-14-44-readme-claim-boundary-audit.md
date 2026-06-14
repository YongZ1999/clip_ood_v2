# README Claim Boundary Audit

**Date**: 2026-06-14

## Context

The publication goal remains active. The remaining completion blockers are still
the real remote training-side LoRA-NSP ablation and strict incremental metrics.

The local reproducibility package now audits claim boundaries in both:

```text
paper_writing/paper-template/paper_draft.tex
experiments/README_publication_repro.md
```

This prevents the reproducibility note from drifting into stronger claims than
the paper draft permits.

## Changes

Updated:

```text
scripts/audit_paper_claims.py
scripts/audit_publication_gates.py
scripts/verify_publication_package.sh
scripts/selftest_publication_audits.py
experiments/README_publication_repro.md
```

The claim audit now:

- accepts multiple paths;
- uses README-specific required boundary phrases for Markdown files;
- keeps the original paper-specific required phrases for TeX files;
- allows explicitly negated warning contexts such as "do not claim";
- rejects positive overclaims in README fixtures through self-tests.

The publication gate dashboard and verifier now run the paper claim boundary
audit over both the paper draft and the reproducibility README.

## Verification

Ran:

```bash
python scripts/audit_paper_claims.py \
  paper_writing/paper-template/paper_draft.tex \
  experiments/README_publication_repro.md

bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
RUN_LATEX=1 bash scripts/verify_publication_package.sh
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
CLAIM AUDIT PASSED
publication package checks passed
Output written on paper_draft.pdf (17 pages, 281891 bytes).
Summary: PASS=5, PENDING=2
```

Remote status still failed from the local environment:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

## Next Action

When SSH access works, run the remote launcher and postprocess:

```bash
bash scripts/remote_training_ablation.sh upload
bash scripts/remote_training_ablation.sh launch_all
bash scripts/remote_training_ablation.sh status_all
bash scripts/remote_training_ablation.sh postprocess
```

The goal remains incomplete until the strict dashboard passes with no pending
gates.
