# Training Protocol Summary Audit

**Date**: 2026-06-14

## Context

The publication goal remains active. The classifier-side training ablation gate
already required all four LoRA configurations, all expected seeds, per-seed
metrics, and positive improvements for `lora_nsp_fd_cd`.

However, the training summary did not yet expose key protocol fields from each
`main_joint.py` result JSON. A summary could therefore contain parseable accuracy
values while hiding an accidental protocol mismatch.

## Changes

Updated:

```text
scripts/summarize_joint_classifier_replay.py
scripts/audit_training_ablation.py
scripts/selftest_publication_audits.py
experiments/README_publication_repro.md
```

The classifier-side training summary now records these protocol columns:

```text
num shots
classifier feature transform
enable lada
lada k
num centers
tune vision encoder
```

The strict training-side audit now requires the publication protocol:

```text
num shots = 16
classifier feature transform = test
enable lada = True
lada k = 16
num centers = 4
tune vision encoder = True
```

The publication audit self-test now includes a bad-protocol fixture where the
classifier feature transform is `train`; the strict training audit must reject
it.

The README now documents that classifier-side training evidence requires both
per-seed accuracy evidence and matching protocol fields.

## Verification

Ran:

```bash
python scripts/selftest_publication_audits.py
python -m py_compile \
  scripts/summarize_joint_classifier_replay.py \
  scripts/audit_training_ablation.py \
  scripts/selftest_publication_audits.py
bash scripts/verify_publication_package.sh
python scripts/audit_publication_gates.py
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
PUBLICATION AUDIT SELFTEST PASSED
publication package checks passed
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
