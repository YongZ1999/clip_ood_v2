# Strict Artifact Self-Test

**Date**: 2026-06-14

## Context

The publication goal remains active. The final postprocess path now writes
strict gate artifacts, but the publication audit self-test only covered normal
non-strict artifact generation. Since the real strict dashboard currently fails
until remote results exist, there was no local self-test for the successful
`strict: true` artifact path.

## Changes

Updated:

```text
scripts/audit_publication_gates.py
scripts/selftest_publication_audits.py
```

`audit_publication_gates.py` now supports a deliberately named self-test-only
environment variable:

```text
PUBLICATION_GATES_TREAT_PENDING_AS_PASS_FOR_SELFTEST=1
```

When set, pending gates are converted to pass after all normal gate checks have
run. This is only used by the self-test to exercise strict artifact generation
without needing real remote result files.

The publication audit self-test now:

- generates a normal artifact and verifies `strict: false`;
- generates a strict artifact under the self-test override;
- verifies the strict artifact records `strict: true`;
- verifies the strict self-test artifact has no pending gates.

Normal strict behavior is unchanged: without the self-test override, pending
gates still make `python scripts/audit_publication_gates.py --strict` exit
nonzero.

## Verification

Ran:

```bash
python scripts/selftest_publication_audits.py
python -m py_compile scripts/audit_publication_gates.py scripts/selftest_publication_audits.py
python scripts/audit_publication_gates.py --strict
bash scripts/verify_publication_package.sh
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
PUBLICATION AUDIT SELFTEST PASSED
python scripts/audit_publication_gates.py --strict exited nonzero as expected
publication package checks passed
strict: False
summary: {'PASS': 5, 'PENDING': 2}
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
training-side and incremental result summaries and the status artifact records
`strict: true`.
