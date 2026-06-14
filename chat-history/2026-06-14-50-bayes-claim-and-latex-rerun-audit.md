# Bayes Claim And LaTeX Rerun Audit

**Date**: 2026-06-14

## Context

The publication goal remains active. While auditing claim-boundary coverage, the
paper still contained an appendix section and theorem titled "Bayes Optimality of
RGDA". The theorem itself was a conditional MAP equivalence under a regularized
Gaussian model, but the title could be read as a stronger unconditional
optimality claim.

The claim audit also only blocked hyphenated `state-of-the-art` and
`Bayes-optimal` phrases, leaving common unhyphenated variants uncovered.

## Changes

Updated:

```text
paper_writing/paper-template/paper_draft.tex
scripts/audit_paper_claims.py
scripts/selftest_publication_audits.py
scripts/verify_publication_package.sh
experiments/README_publication_repro.md
```

The appendix section/theorem title was changed from Bayes optimality language to
a narrower conditional statement:

```text
MAP Equivalence of RGDA Under a Regularized Gaussian Model
MAP Equivalence of RGDA
```

The claim audit now also blocks:

```text
state of the art
Bayes Optimality
Bayes optimal
```

The publication audit self-test now includes a negative fixture with
unhyphenated leaderboard and Bayes-optimality-style language.

The LaTeX verifier now runs `pdflatex` twice and fails if rerun or
cross-reference warnings remain after the second pass, in addition to the
existing compile, duplicate destination, and undefined reference/citation
checks.

The README was updated to document the two-pass LaTeX verifier and to avoid
spelling forbidden claim phrases in its "do not claim" section.

## Verification

Ran:

```bash
python scripts/audit_paper_claims.py \
  paper_writing/paper-template/paper_draft.tex \
  experiments/README_publication_repro.md
python scripts/selftest_publication_audits.py
bash scripts/verify_publication_package.sh
RUN_LATEX=1 bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
CLAIM AUDIT PASSED
PUBLICATION AUDIT SELFTEST PASSED
publication package checks passed
Output written on paper_draft.pdf (17 pages, 281940 bytes).
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
training-side and incremental result summaries.
