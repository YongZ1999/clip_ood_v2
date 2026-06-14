# LaTeX Reference Verifier And Goal Status

**Date**: 2026-06-14

## Context

The publication goal remains active:

```text
将 LoRA-NSP + LR-RGDA/集成分类器推进到可发表研究标准：在严格对齐 LADA 协议和无数据泄漏评估下，明确超越或实质性改进 LADA 的核心实验设置、方法边界、消融证据、论文叙事与复现实验包。
```

The current gate dashboard remains:

```text
PASS=5, PENDING=2
```

Passed gates:

```text
Paper claim boundary
Inference replay result ledger
Fair protocol configuration
LADA official DPT boundary
Reproducibility scripts and README
```

Pending gates:

```text
Training-side LoRA-NSP ablation
Strict incremental Transfer/Average/Last/forgetting
```

## Changes

Updated:

```text
scripts/verify_publication_package.sh
experiments/README_publication_repro.md
```

When `RUN_LATEX=1`, the verifier now rejects undefined LaTeX references and
citations in `paper_draft.log`, in addition to compile errors and duplicate PDF
destination warnings.

The check intentionally matches reference/citation warning patterns instead of
the bare word `undefined`, so the existing nonfatal font substitution warning
does not fail the verifier.

## Verification

Ran:

```bash
RUN_LATEX=1 bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
publication package checks passed
Output written on paper_draft.pdf (17 pages, 281891 bytes).
Summary: PASS=5, PENDING=2
```

Remote status still failed from the local environment:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

## Next Action

Once SSH access works, run:

```bash
bash scripts/remote_training_ablation.sh upload
bash scripts/remote_training_ablation.sh launch_all
bash scripts/remote_training_ablation.sh status_all
bash scripts/remote_training_ablation.sh postprocess
```

The goal should not be marked complete until the strict dashboard passes with no
pending gates.
