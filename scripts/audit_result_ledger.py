#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("experiments/result_ledger.json"),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path("paper_writing/paper-template/paper_draft.tex"),
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("experiments/README_publication_repro.md"),
    )
    return parser.parse_args()


def tex_number(display):
    return display.replace(" +/- ", r"{\pm}")


def normalized(text):
    return re.sub(r"\s+", " ", text)


def require_contains(haystack, needle, label, failures):
    if needle not in haystack:
        failures.append(f"{label}: missing `{needle}`")


def rounded_margin(value):
    return f"{float(value):+.2f}"


def storage_mib(protocol, formula):
    classes = int(protocol["classes"])
    feature_dim = int(protocol["feature_dim"])
    if protocol["dtype"] != "float32":
        raise ValueError(f"unsupported storage dtype: {protocol['dtype']}")
    bytes_per_value = 4
    normalized_formula = formula.replace(" ", "")
    if normalized_formula == "C*16*d":
        values = classes * 16 * feature_dim
    elif normalized_formula == "C*4*d":
        values = classes * 4 * feature_dim
    elif normalized_formula == "C*4*(d+2)":
        values = classes * 4 * (feature_dim + 2)
    else:
        raise ValueError(f"unsupported storage formula: {formula}")
    return f"{values * bytes_per_value / (1024 ** 2):.2f} MiB"


def audit_replay_protocol(protocol, failures):
    expected = {
        "datasets": "X-TAIL 10 datasets",
        "shots": 16,
        "seeds": [42, 43, 44],
        "classifier_feature_transform": "test",
        "full_test_split": True,
        "feature_source_matched_between_lr_rgda_and_lada": True,
    }
    for key, expected_value in expected.items():
        actual = protocol.get(key)
        if actual != expected_value:
            failures.append(
                f"replay protocol mismatch for {key}: "
                f"ledger has {actual!r}, expected {expected_value!r}"
            )


def audit_replay_row(row, failures):
    required_metrics = ["CLIP-ZS", "LR-RGDA", "LADA", "LR-RGDA+ZS", "LADA+ZS"]
    display = row.get("display", {})
    missing_metrics = [metric for metric in required_metrics if metric not in display]
    extra_metrics = [metric for metric in display if metric not in required_metrics]
    if missing_metrics:
        failures.append(
            f"{row.get('readme_label', row.get('paper_label'))}: missing display metrics "
            + ", ".join(missing_metrics)
        )
    if extra_metrics:
        failures.append(
            f"{row.get('readme_label', row.get('paper_label'))}: unexpected display metrics "
            + ", ".join(extra_metrics)
        )

    source_files = row.get("source_files", [])
    if len(source_files) != 3:
        failures.append(
            f"{row.get('readme_label', row.get('paper_label'))}: expected 3 source files, "
            f"found {len(source_files)}"
        )
    for seed in (42, 43, 44):
        token = f"seed{seed}"
        matches = [source for source in source_files if token in source]
        if len(matches) != 1:
            failures.append(
                f"{row.get('readme_label', row.get('paper_label'))}: expected exactly one "
                f"source file containing {token}, found {len(matches)}"
            )

    if not row.get("remote_output_dir"):
        failures.append(f"{row.get('readme_label', row.get('paper_label'))}: missing remote_output_dir")

    for metric, deltas in row.get("per_seed_delta", {}).items():
        seeds = sorted(str(seed) for seed in deltas)
        if seeds != ["42", "43", "44"]:
            failures.append(
                f"{row.get('readme_label', row.get('paper_label'))}: {metric} deltas have "
                f"seeds {seeds}, expected ['42', '43', '44']"
            )


def main():
    args = parse_args()
    ledger = json.loads(args.ledger.read_text())
    paper = args.paper.read_text()
    readme = args.readme.read_text()
    paper_norm = normalized(paper)
    readme_norm = normalized(readme)
    failures = []

    require_contains(paper, "classifier\\_feature\\_transform=test", "paper protocol", failures)
    require_contains(
        paper,
        "not as a claim that compact GMM classifier rebuilding is equivalent to official LADA DPT",
        "paper LADA DPT boundary",
        failures,
    )

    for table in ledger["tables"]:
        require_contains(paper, table["table_id"], f"paper table {table['table_id']}", failures)
        if table["table_id"] == "tab:gmm_mean_replay":
            audit_replay_protocol(table["protocol"], failures)
            for row in table["rows"]:
                audit_replay_row(row, failures)
                require_contains(paper_norm, row["paper_label"], f"paper row {row['paper_label']}", failures)
                require_contains(readme_norm, row["readme_label"], f"readme row {row['readme_label']}", failures)
                for metric, display in row["display"].items():
                    require_contains(readme_norm, display, f"readme {row['readme_label']} {metric}", failures)
                    require_contains(
                        paper,
                        tex_number(display),
                        f"paper {row['paper_label']} {metric}",
                        failures,
                    )
                for source_file in row["source_files"]:
                    require_contains(
                        json.dumps(row),
                        source_file,
                        f"ledger source file for {row['readme_label']}",
                        failures,
                    )
                for metric, deltas in row.get("per_seed_delta", {}).items():
                    require_contains(paper_norm, metric, f"paper per-seed delta {metric}", failures)
                    require_contains(readme_norm, metric, f"readme per-seed delta {metric}", failures)
                    for seed, delta in deltas.items():
                        rounded = rounded_margin(delta)
                        require_contains(
                            paper_norm,
                            rounded,
                            f"paper per-seed delta {metric} seed {seed}",
                            failures,
                        )
                        require_contains(
                            readme_norm,
                            rounded,
                            f"readme per-seed delta {metric} seed {seed}",
                            failures,
                        )
                        if float(delta) <= 0:
                            failures.append(
                                f"ledger per-seed delta must be positive: {metric} seed {seed}={delta}"
                            )
        elif table["table_id"] == "tab:storage_budget":
            protocol = table["protocol"]
            if not protocol.get("counts_replay_source_not_final_classifier"):
                failures.append("storage budget protocol must mark replay-source boundary")
            for row in table["rows"]:
                expected_storage = storage_mib(protocol, row["formula"])
                if row["storage"] != expected_storage:
                    failures.append(
                        f"storage budget mismatch for {row['paper_label']}: "
                        f"ledger has {row['storage']}, computed {expected_storage}"
                    )
                require_contains(paper_norm, row["storage"], f"paper storage {row['paper_label']}", failures)
                require_contains(readme_norm, row["storage"], f"readme storage {row['paper_label']}", failures)

    print(f"audited ledger: {args.ledger}")
    print(f"tables: {len(ledger['tables'])}")
    if failures:
        print("RESULT LEDGER AUDIT FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("RESULT LEDGER AUDIT PASSED")


if __name__ == "__main__":
    main()
