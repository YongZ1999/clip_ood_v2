# Gate Artifact Metadata

**Date**: 2026-06-14

## Context

The publication goal remains active. The verifier already writes gate status
snapshots to:

```text
experiments/publication_gate_status.json
experiments/publication_gate_status.md
```

The JSON snapshot previously contained only `gates` and `summary`. That made it
less useful as a remote postprocess artifact because it did not record the schema
version, generation time, or whether strict mode was used.

## Changes

Updated:

```text
scripts/audit_publication_gates.py
scripts/selftest_publication_audits.py
experiments/README_publication_repro.md
```

The gate JSON artifact now includes:

```text
schema_version
generated_at_utc
strict
gates
summary
```

The publication audit self-test now generates temporary gate JSON/Markdown
artifacts and checks that the JSON metadata fields and Markdown dashboard content
are present.

The README now documents the artifact metadata fields so remote postprocess
outputs can be audited after the fact.

## Verification

Ran:

```bash
python scripts/selftest_publication_audits.py
python -m py_compile scripts/audit_publication_gates.py scripts/selftest_publication_audits.py
bash scripts/verify_publication_package.sh
python -m json.tool experiments/publication_gate_status.json
python scripts/audit_publication_gates.py
bash scripts/remote_training_ablation.sh status_all
```

Result:

```text
PUBLICATION AUDIT SELFTEST PASSED
publication package checks passed
schema_version: 1
strict: false
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
