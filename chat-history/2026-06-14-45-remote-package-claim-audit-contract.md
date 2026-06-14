# Remote Package Claim Audit Contract

**Date**: 2026-06-14

## Context

The publication goal remains active. The local verifier now audits claim
boundaries in both the paper draft and the reproducibility README, but the remote
package self-test previously only checked that required files were uploaded.

This turn tightened the remote reproducibility contract so a remote package
cannot silently drop README claim-boundary coverage.

## Changes

Updated:

```text
scripts/selftest_remote_package.py
experiments/README_publication_repro.md
```

The remote package self-test now checks that:

- `verify_publication_package.sh` audits both `paper_draft.tex` and
  `README_publication_repro.md`;
- `audit_publication_gates.py` also uses both files for the paper claim boundary
  gate;
- `postprocess_publication_results.sh` still runs the verifier and strict
  publication dashboard after remote result postprocessing.

The README now documents that the remote package self-test covers these final
verification contracts in addition to upload file coverage.

## Verification

Ran:

```bash
python scripts/selftest_remote_package.py
python -m py_compile scripts/selftest_remote_package.py
bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py --strict
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
REMOTE PACKAGE SELFTEST PASSED
publication package checks passed
Summary: PASS=5, PENDING=2
```

The strict dashboard failed as expected because the two true result gates remain
pending:

```text
Training-side LoRA-NSP ablation
Strict incremental Transfer/Average/Last/forgetting
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
training and incremental result summaries.
