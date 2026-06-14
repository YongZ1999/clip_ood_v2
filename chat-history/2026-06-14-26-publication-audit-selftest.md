# Publication Audit Self-Test

**Date**: 2026-06-14

## Context

The strict training and incremental audits now fail when the full
`lora_nsp_fd_cd` configuration does not improve over `lora_vanilla` or
`lora_nsp`. To prevent future edits from accidentally relaxing that behavior, a
local self-test was added to the publication verifier.

## Changes

Added:

```text
scripts/selftest_publication_audits.py
```

Updated:

```text
scripts/verify_publication_package.sh
scripts/remote_training_ablation.sh
```

The self-test creates temporary synthetic data under the system temp directory:

- training-side positive and negative `summary.csv` files;
- incremental positive and negative `*_results.json` files with full 10-task
  `accuracy_matrix` fields.

It verifies:

- `scripts/audit_training_ablation.py` passes a positive case;
- `scripts/audit_training_ablation.py` fails a complete but non-improving case;
- `scripts/audit_incremental_metrics.py` passes a positive case;
- `scripts/audit_incremental_metrics.py` fails a complete but non-improving
  case.

The synthetic files are only regression tests for audit behavior. They are not
paper results and are not written into `experiments/`.

## Verification

Ran:

```bash
python scripts/selftest_publication_audits.py
python -m py_compile \
  scripts/selftest_publication_audits.py \
  scripts/audit_training_ablation.py \
  scripts/audit_incremental_metrics.py
bash -n scripts/verify_publication_package.sh
bash -n scripts/remote_training_ablation.sh
bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
git diff --check
```

`verify_publication_package.sh` now includes:

```text
[verify] publication audit selftest
PUBLICATION AUDIT SELFTEST PASSED
```

Current package status remains:

```text
Summary: PASS=5, PENDING=2
publication package checks passed
```

The two pending gates still require real remote result files.

Remote status was queried once and still failed with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```
