# Gate Status Artifacts

**Date**: 2026-06-14

## Context

`scripts/audit_publication_gates.py` supports JSON and Markdown outputs, but the
main verifier previously only printed the dashboard to stdout. Persisting the
current dashboard makes the publication package easier to inspect after local or
remote postprocessing.

## Changes

Updated:

```text
scripts/verify_publication_package.sh
experiments/README_publication_repro.md
```

`verify_publication_package.sh` now writes:

```text
experiments/publication_gate_status.json
experiments/publication_gate_status.md
```

when it runs the publication dashboard.

The README now documents these gate status artifacts.

## Verification

Ran:

```bash
bash scripts/verify_publication_package.sh
sed -n '1,40p' experiments/publication_gate_status.md
python -m json.tool experiments/publication_gate_status.json
git diff --check
```

The generated dashboard reports:

```text
Summary: PASS=5, PENDING=2
```

The JSON artifact contains the same seven gates and summary:

```json
{
  "PASS": 5,
  "PENDING": 2
}
```

Remote status was queried once and still failed with:

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```
