# LADA DPT Boundary Clarification

Date: 2026-06-14

## Context

The active publication goal requires strict alignment with LADA while avoiding
overclaiming. The completed classifier replay matrix compares LR-RGDA and LADA
under matched real-feature and GMM statistical replay sources. This is useful
evidence for compact replay-source classifier construction, but it is not the
same protocol as official LADA DPT.

## Change Made

Added an explicit protocol boundary to:

- `paper_writing/paper-template/paper_draft.tex`
- `experiments/README_publication_repro.md`

The new text states that the current `joint_classifier_*` experiments are
controlled classifier-rebuild comparisons:

- the feature source is fixed before classifier fitting;
- LR-RGDA and LADA are rebuilt from the same real or GMM-derived features;
- both classifiers are evaluated on the same full test split;
- the goal is to isolate classifier inductive bias under matched replay-source
  storage.

It also states that official LADA DPT is a training-time replay procedure:

- old-class prototypes or GMM samples are replayed during incremental training;
- current-task data remains real;
- label memories are updated and frozen incrementally;
- replay affects the training trajectory, not only the final classifier fit.

## Claim Boundary

Current classifier-replay results can support:

> LR-RGDA/zero-shot ensemble is stronger than a LADA-style label-memory
> classifier under matched compact GMM replay sources.

They should not support:

> The current classifier-rebuild replay experiments reproduce or surpass
> official LADA DPT.

That stronger claim requires a separate DPT-aligned experiment.

## Verification

Ran:

```bash
bash -n scripts/run_joint_training_ablation.sh
git diff --check
pdflatex -interaction=nonstopmode -halt-on-error paper_draft.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_draft.tex
```

All commands passed. Remaining LaTeX warnings are non-fatal and were already
present in the draft class/appendix structure.

## Remote Status

Tried one low-frequency remote status query for the training ablation. SSH still
failed with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

The training-side ablation remains pending.
