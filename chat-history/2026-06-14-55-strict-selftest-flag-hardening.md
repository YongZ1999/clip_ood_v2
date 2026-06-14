# Strict Self-Test Flag Hardening

**Date**: 2026-06-14

## Context

The publication goal remains active. The strict artifact self-test needed a way
to exercise the `strict: true` success path while real remote results are still
missing. The first implementation used an environment variable to treat pending
gates as pass in the self-test.

That created an avoidable risk: if the same environment variable leaked into a
real verifier or remote postprocess environment, strict gates could be
accidentally relaxed.

## Changes

Updated:

```text
scripts/audit_publication_gates.py
scripts/selftest_publication_audits.py
```

Removed the environment-variable override and replaced it with an explicit
hidden self-test CLI flag:

```bash
python scripts/audit_publication_gates.py \
  --strict \
  --selftest_treat_pending_as_pass
```

The flag is only used inside `scripts/selftest_publication_audits.py`. Normal
verifier, postprocess, and remote helper paths do not pass it.

The self-test still covers:

- normal artifact generation with `strict: false`;
- strict failure artifact generation with pending gates preserved;
- strict success artifact generation under the explicit self-test flag.

## Verification

Ran:

```bash
python scripts/selftest_publication_audits.py
python -m py_compile scripts/audit_publication_gates.py scripts/selftest_publication_audits.py
python scripts/audit_publication_gates.py --strict
python scripts/audit_publication_gates.py --strict --selftest_treat_pending_as_pass \
  --output_json /tmp/strict_selftest_gate.json \
  --output_markdown /tmp/strict_selftest_gate.md
bash scripts/verify_publication_package.sh
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
PUBLICATION AUDIT SELFTEST PASSED
normal strict dashboard exited nonzero as expected
self-test strict artifact: strict=True, summary={'PASS': 7}
publication package checks passed
current real artifact: strict=False, summary={'PASS': 5, 'PENDING': 2}
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
training-side and incremental result summaries and the final status artifact
records `strict: true` without the self-test flag.
