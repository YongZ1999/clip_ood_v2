#!/usr/bin/env python3
import csv
import json
import subprocess
import tempfile
from pathlib import Path


EXPECTED_TASKS = "aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397".split()
EXPECTED_SEEDS = (42, 43, 44)
METHODS = ("lora_vanilla", "lora_nsp", "lora_nsp_fd", "lora_nsp_fd_cd")


def run(command, expect_success=True, env=None):
    proc = subprocess.run(command, text=True, capture_output=True, env=env)
    output = (proc.stdout + proc.stderr).strip()
    if expect_success and proc.returncode != 0:
        raise SystemExit(
            "command failed unexpectedly:\n"
            + " ".join(command)
            + "\n"
            + output
        )
    if not expect_success and proc.returncode == 0:
        raise SystemExit(
            "command succeeded unexpectedly:\n"
            + " ".join(command)
            + "\n"
            + output
        )
    return output


def mean_value(values):
    return sum(values) / len(values)


def seed_values_for(raw):
    if isinstance(raw, dict):
        return [float(raw[seed]) for seed in EXPECTED_SEEDS]
    return [float(raw) for _ in EXPECTED_SEEDS]


def format_by_seed(values):
    return ", ".join(f"{seed}:{value:.2f}" for seed, value in zip(EXPECTED_SEEDS, values))


def append_seed_score(raw, seed, value):
    return raw + f", {seed}:{value:.2f}"


def first_score(raw):
    return float(str(raw).split()[0])


def write_readme_claim_fixture(path, overclaim=False):
    if overclaim:
        text = """
# Publication Notes

Use `--classifier_feature_transform test`.
LR-RGDA is state of the art, provides Bayes Optimality, and surpass official LADA DPT.
The paper proposes Bayesian Ensemble Classifiers in a holistic framework that co-optimizes training and inference.
LR-RGDA/zero-shot ensemble is not a universal replacement for LADA.
LADA is stronger when real historical 16-shot features are available.
These are not official LADA DPT reproductions.
Storage excludes final classifier parameters.
"""
    else:
        text = """
# Publication Notes

Use `--classifier_feature_transform test`.
LR-RGDA/zero-shot ensemble is not a universal replacement for LADA.
LADA is stronger when real historical 16-shot features are available.
These are not official LADA DPT reproductions.
Replay-source storage is reported excluding final classifier parameters.
Do not claim state-of-the-art.
They should not say that these experiments reproduce or surpass official LADA DPT.
"""
    path.write_text(text.strip() + "\n")


def write_paper_claim_fixture(path, malformed_hidden_block=False):
    hidden_suffix = "\\iffalse\nhidden TODO"
    if not malformed_hidden_block:
        hidden_suffix += "\n\\fi"
    text = rf"""
classifier\_feature\_transform=test
LR-RGDA is not as a universal replacement for LADA.
LADA remains stronger when real historical 16-shot features are available.
This is not as a claim that compact GMM classifier rebuilding is equivalent to official LADA DPT.
The storage table counts replay-source storage, not the final classifier parameters.
{hidden_suffix}
"""
    path.write_text(text.strip() + "\n")


def write_result_ledger_fixture(
    root,
    negative_delta=False,
    bad_storage=False,
    bad_replay_protocol=False,
):
    delta = -0.20 if negative_delta else 0.20
    storage = "34.37 MiB" if bad_storage else "34.38 MiB"
    ledger = {
        "version": 1,
        "updated_at": "2026-06-14",
        "paper": str(root / "paper.tex"),
        "readme": str(root / "README.md"),
        "tables": [
            {
                "table_id": "tab:gmm_mean_replay",
                "description": "fixture",
                "protocol": {
                    "datasets": "X-TAIL 10 datasets",
                    "shots": 16,
                    "seeds": [42, 43, 44],
                    "classifier_feature_transform": "train" if bad_replay_protocol else "test",
                    "full_test_split": True,
                    "feature_source_matched_between_lr_rgda_and_lada": True,
                },
                "rows": [
                    {
                        "paper_label": "GMM raw component means",
                        "readme_label": "test_gmm_raw_mean_fixed",
                        "remote_output_dir": "/tmp",
                        "source_files": [
                            "test_gmm_raw_mean_fixed_seed42.json",
                            "test_gmm_raw_mean_fixed_seed43.json",
                            "test_gmm_raw_mean_fixed_seed44.json",
                        ],
                        "display": {
                            "CLIP-ZS": "56.40 +/- 0.00",
                            "LR-RGDA": "76.93 +/- 0.16",
                            "LADA": "76.08 +/- 0.35",
                            "LR-RGDA+ZS": "77.12 +/- 0.13",
                            "LADA+ZS": "76.08 +/- 0.35",
                        },
                        "per_seed_delta": {
                            "LR-RGDA - LADA": {
                                "42": delta,
                                "43": 0.99,
                                "44": 0.59,
                            }
                        },
                    }
                ],
            },
            {
                "table_id": "tab:storage_budget",
                "description": "fixture",
                "protocol": {
                    "classes": 1100,
                    "feature_dim": 512,
                    "dtype": "float32",
                    "counts_replay_source_not_final_classifier": True,
                },
                "rows": [
                    {
                        "paper_label": "Real 16-shot features",
                        "items_per_class": "16 features",
                        "formula": "C * 16 * d",
                        "storage": storage,
                    }
                ],
            },
        ],
    }
    paper = f"""
classifier\\_feature\\_transform=test
not as a claim that compact GMM classifier rebuilding is equivalent to official LADA DPT
\\label{{tab:gmm_mean_replay}}
GMM raw component means
$56.40{{\\pm}}0.00$ $76.93{{\\pm}}0.16$ $76.08{{\\pm}}0.35$
$77.12{{\\pm}}0.13$ $76.08{{\\pm}}0.35$
LR-RGDA - LADA {delta:+.2f} +0.99 +0.59
\\label{{tab:storage_budget}}
Real 16-shot features {storage}
"""
    readme = f"""
test_gmm_raw_mean_fixed 56.40 +/- 0.00 76.93 +/- 0.16 76.08 +/- 0.35
77.12 +/- 0.13 76.08 +/- 0.35
LR-RGDA - LADA {delta:+.2f} +0.99 +0.59
Real 16-shot features {storage}
"""
    (root / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (root / "paper.tex").write_text(paper.strip() + "\n")
    (root / "README.md").write_text(readme.strip() + "\n")


def check_claim_audit(tmp):
    pass_readme = tmp / "claim_pass.md"
    fail_readme = tmp / "claim_fail.md"
    pass_paper = tmp / "claim_pass.tex"
    malformed_paper = tmp / "claim_malformed_hidden.tex"
    write_readme_claim_fixture(pass_readme)
    write_readme_claim_fixture(fail_readme, overclaim=True)
    write_paper_claim_fixture(pass_paper)
    write_paper_claim_fixture(malformed_paper, malformed_hidden_block=True)
    run(["python", "scripts/audit_paper_claims.py", str(pass_readme)])
    run(["python", "scripts/audit_paper_claims.py", str(pass_paper)])
    fail_output = run(
        ["python", "scripts/audit_paper_claims.py", str(fail_readme)],
        expect_success=False,
    )
    if "CLAIM AUDIT FAILED" not in fail_output:
        raise SystemExit("claim negative self-test did not report failure")
    malformed_output = run(
        ["python", "scripts/audit_paper_claims.py", str(malformed_paper)],
        expect_success=False,
    )
    if "malformed hidden block structure" not in malformed_output:
        raise SystemExit("claim audit did not reject malformed hidden blocks")


def check_publication_gate_artifacts(tmp):
    output_json = tmp / "gate_status.json"
    output_markdown = tmp / "gate_status.md"
    strict_fail_json = tmp / "gate_status_strict_fail.json"
    strict_fail_markdown = tmp / "gate_status_strict_fail.md"
    strict_json = tmp / "gate_status_strict.json"
    strict_markdown = tmp / "gate_status_strict.md"
    run(
        [
            "python",
            "scripts/audit_publication_gates.py",
            "--output_json",
            str(output_json),
            "--output_markdown",
            str(output_markdown),
        ]
    )
    data = json.loads(output_json.read_text())
    required_keys = {"schema_version", "generated_at_utc", "strict", "gates", "summary"}
    missing = required_keys - set(data)
    if missing:
        raise SystemExit(
            "publication gate artifact missing keys: " + ", ".join(sorted(missing))
        )
    if data["schema_version"] != 1:
        raise SystemExit("publication gate artifact has unexpected schema_version")
    if data["strict"] is not False:
        raise SystemExit("publication gate artifact strict flag should be false")
    if not isinstance(data["gates"], list) or not data["gates"]:
        raise SystemExit("publication gate artifact has no gates")
    if not isinstance(data["summary"], dict) or not data["summary"]:
        raise SystemExit("publication gate artifact has no summary")
    markdown = output_markdown.read_text()
    if "| Gate | Status | Next action |" not in markdown or "Summary:" not in markdown:
        raise SystemExit("publication gate markdown artifact missing dashboard content")
    run(
        [
            "python",
            "scripts/audit_publication_gates.py",
            "--strict",
            "--output_json",
            str(strict_fail_json),
            "--output_markdown",
            str(strict_fail_markdown),
        ],
        expect_success=False,
    )
    strict_fail_data = json.loads(strict_fail_json.read_text())
    if strict_fail_data.get("strict") is not True:
        raise SystemExit("strict failure artifact did not record strict=true")
    if "PENDING" not in strict_fail_data.get("summary", {}):
        raise SystemExit("strict failure artifact did not preserve pending summary")
    run(
        [
            "python",
            "scripts/audit_publication_gates.py",
            "--strict",
            "--selftest_treat_pending_as_pass",
            "--output_json",
            str(strict_json),
            "--output_markdown",
            str(strict_markdown),
        ],
    )
    strict_data = json.loads(strict_json.read_text())
    if strict_data.get("strict") is not True:
        raise SystemExit("strict publication gate artifact did not record strict=true")
    if "PENDING" in strict_data.get("summary", {}):
        raise SystemExit("strict self-test artifact still contains pending gates")


def check_result_ledger_audit(tmp):
    pass_dir = tmp / "ledger_pass"
    negative_delta_dir = tmp / "ledger_negative_delta"
    bad_storage_dir = tmp / "ledger_bad_storage"
    bad_protocol_dir = tmp / "ledger_bad_protocol"
    pass_dir.mkdir()
    negative_delta_dir.mkdir()
    bad_storage_dir.mkdir()
    bad_protocol_dir.mkdir()
    write_result_ledger_fixture(pass_dir)
    write_result_ledger_fixture(negative_delta_dir, negative_delta=True)
    write_result_ledger_fixture(bad_storage_dir, bad_storage=True)
    write_result_ledger_fixture(bad_protocol_dir, bad_replay_protocol=True)
    run(
        [
            "python",
            "scripts/audit_result_ledger.py",
            "--ledger",
            str(pass_dir / "ledger.json"),
            "--paper",
            str(pass_dir / "paper.tex"),
            "--readme",
            str(pass_dir / "README.md"),
        ]
    )
    negative_output = run(
        [
            "python",
            "scripts/audit_result_ledger.py",
            "--ledger",
            str(negative_delta_dir / "ledger.json"),
            "--paper",
            str(negative_delta_dir / "paper.tex"),
            "--readme",
            str(negative_delta_dir / "README.md"),
        ],
        expect_success=False,
    )
    if "per-seed delta must be positive" not in negative_output:
        raise SystemExit("ledger negative-delta self-test did not report failure")
    storage_output = run(
        [
            "python",
            "scripts/audit_result_ledger.py",
            "--ledger",
            str(bad_storage_dir / "ledger.json"),
            "--paper",
            str(bad_storage_dir / "paper.tex"),
            "--readme",
            str(bad_storage_dir / "README.md"),
        ],
        expect_success=False,
    )
    if "storage budget mismatch" not in storage_output:
        raise SystemExit("ledger storage self-test did not report failure")
    protocol_output = run(
        [
            "python",
            "scripts/audit_result_ledger.py",
            "--ledger",
            str(bad_protocol_dir / "ledger.json"),
            "--paper",
            str(bad_protocol_dir / "paper.tex"),
            "--readme",
            str(bad_protocol_dir / "README.md"),
        ],
        expect_success=False,
    )
    if "replay protocol mismatch" not in protocol_output:
        raise SystemExit("ledger protocol self-test did not report failure")


def write_training_summary(path, values, classifier_feature_transform="test"):
    source_dir = path.parent / f"{path.stem}_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    headers = [
        "experiment",
        "n",
        "seeds",
        "missing expected seeds",
        "source files",
        "num shots",
        "classifier feature transform",
        "enable lada",
        "lada k",
        "num centers",
        "tune vision encoder",
        "CLIP-ZS",
        "LR-RGDA",
        "LADA",
        "LR-RGDA+ZS",
        "LADA+ZS",
        "CLIP-ZS by seed",
        "LR-RGDA by seed",
        "LADA by seed",
        "LR-RGDA+ZS by seed",
        "LADA+ZS by seed",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for method in METHODS:
            clip_values = seed_values_for(values[method])
            lr_values = [value + 10.0 for value in clip_values]
            lada_values = [value + 9.0 for value in clip_values]
            ensemble_values = [value + 10.5 for value in clip_values]
            for seed, clip, lr, lada, ensemble in zip(
                EXPECTED_SEEDS,
                clip_values,
                lr_values,
                lada_values,
                ensemble_values,
            ):
                source_path = source_dir / f"{method}_seed{seed}.json"
                source_path.write_text(
                    json.dumps(
                        {
                            "experiment_name": method,
                            "seed": seed,
                            "arguments": {
                                "num_shots": 16,
                                "classifier_feature_transform": classifier_feature_transform,
                                "enable_lada": True,
                                "lada_k": 16,
                                "num_centers": 4,
                                "tune_vision_encoder": True,
                            },
                            "metrics": {
                                "id": {
                                    "average": {
                                        "zero_shot": clip,
                                        "lr_rgda": lr,
                                        "lada": lada,
                                        "ours_ensemble": ensemble,
                                        "lada_zs": lada,
                                    }
                                }
                            },
                        },
                        indent=2,
                    )
                    + "\n"
                )
            writer.writerow(
                {
                    "experiment": method,
                    "n": "3",
                    "seeds": "42,43,44",
                    "missing expected seeds": "",
                    "source files": (
                        f"42:{source_dir / f'{method}_seed42.json'}, "
                        f"43:{source_dir / f'{method}_seed43.json'}, "
                        f"44:{source_dir / f'{method}_seed44.json'}"
                    ),
                    "num shots": "16",
                    "classifier feature transform": classifier_feature_transform,
                    "enable lada": "True",
                    "lada k": "16",
                    "num centers": "4",
                    "tune vision encoder": "True",
                    "CLIP-ZS": f"{mean_value(clip_values):.2f} +/- 0.10",
                    "LR-RGDA": f"{mean_value(lr_values):.2f} +/- 0.10",
                    "LADA": f"{mean_value(lada_values):.2f} +/- 0.10",
                    "LR-RGDA+ZS": f"{mean_value(ensemble_values):.2f} +/- 0.10",
                    "LADA+ZS": f"{mean_value(lada_values):.2f} +/- 0.10",
                    "CLIP-ZS by seed": format_by_seed(clip_values),
                    "LR-RGDA by seed": format_by_seed(lr_values),
                    "LADA by seed": format_by_seed(lada_values),
                    "LR-RGDA+ZS by seed": format_by_seed(ensemble_values),
                    "LADA+ZS by seed": format_by_seed(lada_values),
                }
            )


def write_legacy_training_summary(path, values):
    headers = [
        "experiment",
        "n",
        "seeds",
        "missing expected seeds",
        "CLIP-ZS",
        "LR-RGDA",
        "LADA",
        "LR-RGDA+ZS",
        "LADA+ZS",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for method in METHODS:
            writer.writerow(
                {
                    "experiment": method,
                    "n": "3",
                    "seeds": "42,43,44",
                    "missing expected seeds": "",
                    "CLIP-ZS": f"{values[method]:.2f} +/- 0.10",
                    "LR-RGDA": f"{values[method] + 10.0:.2f} +/- 0.10",
                    "LADA": f"{values[method] + 9.0:.2f} +/- 0.10",
                    "LR-RGDA+ZS": f"{values[method] + 10.5:.2f} +/- 0.10",
                    "LADA+ZS": f"{values[method] + 9.0:.2f} +/- 0.10",
                }
            )


def write_incremental_results(root, values, eval_max_samples=0, shuffle_task_order=False):
    root.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        for seed in EXPECTED_SEEDS:
            raw_value = values[method]
            if isinstance(raw_value, dict):
                value = float(raw_value[seed])
            else:
                value = float(raw_value) + (seed - 43) * 0.001
            matrix = [[value for _ in EXPECTED_TASKS] for _ in EXPECTED_TASKS]
            data = {
                "args": {
                    "method": "lora_nsp" if method != "lora_vanilla" else "lora_vanilla",
                    "experiment_name": method,
                    "seed": seed,
                    "task_sequence": (
                        list(reversed(EXPECTED_TASKS)) if shuffle_task_order else EXPECTED_TASKS
                    ),
                    "eval_max_samples": eval_max_samples,
                },
                "metrics": {
                    "transfer": value * 100.0,
                    "average": value * 100.0,
                    "last": value * 100.0,
                },
                "forgetting_rate": (1.0 - value) * 100.0,
                "accuracy_matrix": matrix,
            }
            (root / f"{method}_seed{seed}_results.json").write_text(
                json.dumps(data, indent=2) + "\n"
            )


def summarize_incremental(root):
    run(
        [
            "python",
            "scripts/summarize_incremental_metrics.py",
            str(root),
            "--output_csv",
            str(root / "incremental_summary.csv"),
            "--output_markdown",
            str(root / "incremental_summary.md"),
            "--aggregate_csv",
            str(root / "incremental_aggregate.csv"),
            "--aggregate_markdown",
            str(root / "incremental_aggregate.md"),
            "--expected_tasks",
            " ".join(EXPECTED_TASKS),
        ]
    )


def check_training_audit(tmp):
    pass_csv = tmp / "training_pass.csv"
    fail_csv = tmp / "training_fail.csv"
    seed_flip_csv = tmp / "training_seed_flip.csv"
    legacy_csv = tmp / "training_legacy_missing_by_seed.csv"
    bad_protocol_csv = tmp / "training_bad_protocol.csv"
    stale_aggregate_csv = tmp / "training_stale_aggregate.csv"
    extra_seed_csv = tmp / "training_extra_seed.csv"
    extra_config_csv = tmp / "training_extra_config.csv"
    duplicate_experiment_csv = tmp / "training_duplicate_experiment.csv"
    duplicate_seed_cell_csv = tmp / "training_duplicate_seed_cell.csv"
    missing_source_csv = tmp / "training_missing_source.csv"
    source_mismatch_csv = tmp / "training_source_mismatch.csv"
    write_training_summary(
        pass_csv,
        {
            "lora_vanilla": 50.0,
            "lora_nsp": 51.0,
            "lora_nsp_fd": 52.0,
            "lora_nsp_fd_cd": 54.0,
        },
    )
    write_training_summary(
        fail_csv,
        {
            "lora_vanilla": 54.0,
            "lora_nsp": 53.0,
            "lora_nsp_fd": 52.0,
            "lora_nsp_fd_cd": 50.0,
        },
    )
    write_training_summary(
        seed_flip_csv,
        {
            "lora_vanilla": {42: 50.0, 43: 50.0, 44: 50.0},
            "lora_nsp": {42: 51.0, 43: 51.0, 44: 51.0},
            "lora_nsp_fd": {42: 52.0, 43: 52.0, 44: 52.0},
            "lora_nsp_fd_cd": {42: 49.0, 43: 56.0, 44: 57.0},
        },
    )
    write_training_summary(
        bad_protocol_csv,
        {
            "lora_vanilla": 50.0,
            "lora_nsp": 51.0,
            "lora_nsp_fd": 52.0,
            "lora_nsp_fd_cd": 54.0,
        },
        classifier_feature_transform="train",
    )
    write_training_summary(
        stale_aggregate_csv,
        {
            "lora_vanilla": 50.0,
            "lora_nsp": 51.0,
            "lora_nsp_fd": 52.0,
            "lora_nsp_fd_cd": 54.0,
        },
    )
    write_training_summary(
        extra_seed_csv,
        {
            "lora_vanilla": 50.0,
            "lora_nsp": 51.0,
            "lora_nsp_fd": 52.0,
            "lora_nsp_fd_cd": 54.0,
        },
    )
    write_training_summary(
        extra_config_csv,
        {
            "lora_vanilla": 50.0,
            "lora_nsp": 51.0,
            "lora_nsp_fd": 52.0,
            "lora_nsp_fd_cd": 54.0,
        },
    )
    write_training_summary(
        duplicate_experiment_csv,
        {
            "lora_vanilla": 50.0,
            "lora_nsp": 51.0,
            "lora_nsp_fd": 52.0,
            "lora_nsp_fd_cd": 54.0,
        },
    )
    write_training_summary(
        duplicate_seed_cell_csv,
        {
            "lora_vanilla": 50.0,
            "lora_nsp": 51.0,
            "lora_nsp_fd": 52.0,
            "lora_nsp_fd_cd": 54.0,
        },
    )
    write_training_summary(
        missing_source_csv,
        {
            "lora_vanilla": 50.0,
            "lora_nsp": 51.0,
            "lora_nsp_fd": 52.0,
            "lora_nsp_fd_cd": 54.0,
        },
    )
    write_training_summary(
        source_mismatch_csv,
        {
            "lora_vanilla": 50.0,
            "lora_nsp": 51.0,
            "lora_nsp_fd": 52.0,
            "lora_nsp_fd_cd": 54.0,
        },
    )
    stale_rows = list(csv.DictReader(stale_aggregate_csv.open(newline="")))
    stale_headers = list(stale_rows[0].keys())
    for row in stale_rows:
        if row["experiment"] == "lora_nsp_fd_cd":
            row["LR-RGDA+ZS"] = "99.99 +/- 0.00"
    with stale_aggregate_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stale_headers)
        writer.writeheader()
        writer.writerows(stale_rows)
    extra_seed_rows = list(csv.DictReader(extra_seed_csv.open(newline="")))
    extra_seed_headers = list(extra_seed_rows[0].keys())
    for row in extra_seed_rows:
        row["n"] = "4"
        row["seeds"] = "42,43,44,45"
        row["source files"] += f", 45:/tmp/{row['experiment']}_seed45.json"
        for metric in ("CLIP-ZS", "LR-RGDA", "LADA", "LR-RGDA+ZS", "LADA+ZS"):
            row[f"{metric} by seed"] = append_seed_score(
                row[f"{metric} by seed"],
                45,
                first_score(row[metric]),
            )
    with extra_seed_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=extra_seed_headers)
        writer.writeheader()
        writer.writerows(extra_seed_rows)
    extra_config_rows = list(csv.DictReader(extra_config_csv.open(newline="")))
    extra_config_headers = list(extra_config_rows[0].keys())
    extra_config_row = dict(extra_config_rows[0])
    extra_config_row["experiment"] = "lora_extra"
    extra_config_rows.append(extra_config_row)
    with extra_config_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=extra_config_headers)
        writer.writeheader()
        writer.writerows(extra_config_rows)
    duplicate_experiment_rows = list(csv.DictReader(duplicate_experiment_csv.open(newline="")))
    duplicate_experiment_headers = list(duplicate_experiment_rows[0].keys())
    duplicate_experiment_rows.append(dict(duplicate_experiment_rows[0]))
    with duplicate_experiment_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=duplicate_experiment_headers)
        writer.writeheader()
        writer.writerows(duplicate_experiment_rows)
    duplicate_seed_cell_rows = list(csv.DictReader(duplicate_seed_cell_csv.open(newline="")))
    duplicate_seed_cell_headers = list(duplicate_seed_cell_rows[0].keys())
    for row in duplicate_seed_cell_rows:
        if row["experiment"] == "lora_nsp_fd_cd":
            row["seeds"] = "42,42,43,44"
            row["LR-RGDA+ZS by seed"] = append_seed_score(
                row["LR-RGDA+ZS by seed"],
                42,
                first_score(row["LR-RGDA+ZS"]),
            )
    with duplicate_seed_cell_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=duplicate_seed_cell_headers)
        writer.writeheader()
        writer.writerows(duplicate_seed_cell_rows)
    missing_source_rows = list(csv.DictReader(missing_source_csv.open(newline="")))
    missing_source_headers = list(missing_source_rows[0].keys())
    for row in missing_source_rows:
        if row["experiment"] == "lora_nsp_fd_cd":
            row["source files"] = row["source files"].replace(
                "lora_nsp_fd_cd_seed42.json",
                "missing_lora_nsp_fd_cd_seed42.json",
            )
    with missing_source_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=missing_source_headers)
        writer.writeheader()
        writer.writerows(missing_source_rows)
    mismatch_source = tmp / "training_source_mismatch_sources" / "lora_nsp_fd_cd_seed42.json"
    mismatch_data = json.loads(mismatch_source.read_text())
    mismatch_data["metrics"]["id"]["average"]["ours_ensemble"] = -1.0
    mismatch_source.write_text(json.dumps(mismatch_data, indent=2) + "\n")
    write_legacy_training_summary(
        legacy_csv,
        {
            "lora_vanilla": 50.0,
            "lora_nsp": 51.0,
            "lora_nsp_fd": 52.0,
            "lora_nsp_fd_cd": 54.0,
        },
    )
    run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(pass_csv),
            "--expected_seeds",
            "42 43 44",
        ]
    )
    fail_output = run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(fail_csv),
            "--expected_seeds",
            "42 43 44",
        ],
        expect_success=False,
    )
    if "TRAINING ABLATION AUDIT FAILED" not in fail_output:
        raise SystemExit("training negative self-test did not report failure")
    seed_flip_output = run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(seed_flip_csv),
            "--expected_seeds",
            "42 43 44",
        ],
        expect_success=False,
    )
    if "seed 42" not in seed_flip_output:
        raise SystemExit("training seed-flip self-test did not report seed-level failure")
    protocol_output = run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(bad_protocol_csv),
            "--expected_seeds",
            "42 43 44",
        ],
        expect_success=False,
    )
    if "classifier feature transform=train" not in protocol_output:
        raise SystemExit("training bad-protocol self-test did not report protocol failure")
    stale_output = run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(stale_aggregate_csv),
            "--expected_seeds",
            "42 43 44",
        ],
        expect_success=False,
    )
    if "aggregate LR-RGDA+ZS" not in stale_output or "by-seed mean" not in stale_output:
        raise SystemExit("training stale-aggregate self-test did not report mismatch")
    extra_seed_output = run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(extra_seed_csv),
            "--expected_seeds",
            "42 43 44",
        ],
        expect_success=False,
    )
    if "unexpected seeds 45" not in extra_seed_output:
        raise SystemExit("training extra-seed self-test did not report unexpected seed")
    extra_config_output = run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(extra_config_csv),
            "--expected_seeds",
            "42 43 44",
        ],
        expect_success=False,
    )
    if "unexpected experiments: lora_extra" not in extra_config_output:
        raise SystemExit("training extra-config self-test did not report unexpected config")
    duplicate_experiment_output = run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(duplicate_experiment_csv),
            "--expected_seeds",
            "42 43 44",
        ],
        expect_success=False,
    )
    if "duplicate experiments: lora_vanilla" not in duplicate_experiment_output:
        raise SystemExit("training duplicate-experiment self-test did not report duplicate")
    duplicate_seed_cell_output = run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(duplicate_seed_cell_csv),
            "--expected_seeds",
            "42 43 44",
        ],
        expect_success=False,
    )
    if (
        "duplicate seeds 42" not in duplicate_seed_cell_output
        or "duplicate LR-RGDA+ZS by seed values for 42" not in duplicate_seed_cell_output
    ):
        raise SystemExit("training duplicate-seed-cell self-test did not report duplicate")
    missing_source_output = run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(missing_source_csv),
            "--expected_seeds",
            "42 43 44",
        ],
        expect_success=False,
    )
    if "source file for seed 42 does not exist" not in missing_source_output:
        raise SystemExit("training missing-source self-test did not report missing file")
    source_mismatch_output = run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(source_mismatch_csv),
            "--expected_seeds",
            "42 43 44",
        ],
        expect_success=False,
    )
    if "source file seed 42 LR-RGDA+ZS" not in source_mismatch_output:
        raise SystemExit("training source-mismatch self-test did not report metric mismatch")
    legacy_output = run(
        [
            "python",
            "scripts/audit_training_ablation.py",
            "--summary_csv",
            str(legacy_csv),
            "--expected_seeds",
            "42 43 44",
        ],
        expect_success=False,
    )
    if "by seed values" not in legacy_output:
        raise SystemExit("training legacy self-test did not report missing per-seed evidence")


def check_incremental_audit(tmp):
    pass_dir = tmp / "incremental_pass"
    fail_dir = tmp / "incremental_fail"
    capped_dir = tmp / "incremental_capped"
    seed_flip_dir = tmp / "incremental_seed_flip"
    shuffled_dir = tmp / "incremental_shuffled_tasks"
    stale_aggregate_dir = tmp / "incremental_stale_aggregate"
    extra_seed_dir = tmp / "incremental_extra_seed"
    extra_method_dir = tmp / "incremental_extra_method"
    duplicate_run_dir = tmp / "incremental_duplicate_run"
    duplicate_aggregate_dir = tmp / "incremental_duplicate_aggregate"
    duplicate_aggregate_seed_dir = tmp / "incremental_duplicate_aggregate_seed"
    missing_path_dir = tmp / "incremental_missing_path"
    write_incremental_results(
        pass_dir,
        {
            "lora_vanilla": 0.50,
            "lora_nsp": 0.52,
            "lora_nsp_fd": 0.53,
            "lora_nsp_fd_cd": 0.56,
        },
    )
    write_incremental_results(
        fail_dir,
        {
            "lora_vanilla": 0.56,
            "lora_nsp": 0.54,
            "lora_nsp_fd": 0.53,
            "lora_nsp_fd_cd": 0.52,
        },
    )
    write_incremental_results(
        capped_dir,
        {
            "lora_vanilla": 0.50,
            "lora_nsp": 0.52,
            "lora_nsp_fd": 0.53,
            "lora_nsp_fd_cd": 0.56,
        },
        eval_max_samples=1000,
    )
    write_incremental_results(
        seed_flip_dir,
        {
            "lora_vanilla": {42: 0.50, 43: 0.50, 44: 0.50},
            "lora_nsp": {42: 0.51, 43: 0.51, 44: 0.51},
            "lora_nsp_fd": {42: 0.52, 43: 0.52, 44: 0.52},
            "lora_nsp_fd_cd": {42: 0.49, 43: 0.56, 44: 0.57},
        },
    )
    write_incremental_results(
        shuffled_dir,
        {
            "lora_vanilla": 0.50,
            "lora_nsp": 0.52,
            "lora_nsp_fd": 0.53,
            "lora_nsp_fd_cd": 0.56,
        },
        shuffle_task_order=True,
    )
    write_incremental_results(
        stale_aggregate_dir,
        {
            "lora_vanilla": 0.50,
            "lora_nsp": 0.52,
            "lora_nsp_fd": 0.53,
            "lora_nsp_fd_cd": 0.56,
        },
    )
    write_incremental_results(
        extra_seed_dir,
        {
            "lora_vanilla": 0.50,
            "lora_nsp": 0.52,
            "lora_nsp_fd": 0.53,
            "lora_nsp_fd_cd": 0.56,
        },
    )
    write_incremental_results(
        extra_method_dir,
        {
            "lora_vanilla": 0.50,
            "lora_nsp": 0.52,
            "lora_nsp_fd": 0.53,
            "lora_nsp_fd_cd": 0.56,
        },
    )
    write_incremental_results(
        duplicate_run_dir,
        {
            "lora_vanilla": 0.50,
            "lora_nsp": 0.52,
            "lora_nsp_fd": 0.53,
            "lora_nsp_fd_cd": 0.56,
        },
    )
    write_incremental_results(
        duplicate_aggregate_dir,
        {
            "lora_vanilla": 0.50,
            "lora_nsp": 0.52,
            "lora_nsp_fd": 0.53,
            "lora_nsp_fd_cd": 0.56,
        },
    )
    write_incremental_results(
        duplicate_aggregate_seed_dir,
        {
            "lora_vanilla": 0.50,
            "lora_nsp": 0.52,
            "lora_nsp_fd": 0.53,
            "lora_nsp_fd_cd": 0.56,
        },
    )
    write_incremental_results(
        missing_path_dir,
        {
            "lora_vanilla": 0.50,
            "lora_nsp": 0.52,
            "lora_nsp_fd": 0.53,
            "lora_nsp_fd_cd": 0.56,
        },
    )
    for method in METHODS:
        source = extra_seed_dir / f"{method}_seed42_results.json"
        data = json.loads(source.read_text())
        data["args"]["seed"] = 45
        (extra_seed_dir / f"{method}_seed45_results.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )
    for seed in EXPECTED_SEEDS:
        source = extra_method_dir / f"lora_vanilla_seed{seed}_results.json"
        data = json.loads(source.read_text())
        data["args"]["experiment_name"] = "lora_extra"
        (extra_method_dir / f"lora_extra_seed{seed}_results.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )
    summarize_incremental(pass_dir)
    summarize_incremental(fail_dir)
    summarize_incremental(capped_dir)
    summarize_incremental(seed_flip_dir)
    summarize_incremental(shuffled_dir)
    summarize_incremental(stale_aggregate_dir)
    summarize_incremental(extra_seed_dir)
    summarize_incremental(extra_method_dir)
    summarize_incremental(duplicate_run_dir)
    summarize_incremental(duplicate_aggregate_dir)
    summarize_incremental(duplicate_aggregate_seed_dir)
    summarize_incremental(missing_path_dir)
    stale_aggregate_csv = stale_aggregate_dir / "incremental_aggregate.csv"
    stale_rows = list(csv.DictReader(stale_aggregate_csv.open(newline="")))
    stale_headers = list(stale_rows[0].keys())
    for row in stale_rows:
        if row["method"] == "lora_nsp_fd_cd":
            row["Average"] = "99.99 +/- 0.00"
    with stale_aggregate_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stale_headers)
        writer.writeheader()
        writer.writerows(stale_rows)
    duplicate_run_csv = duplicate_run_dir / "incremental_summary.csv"
    duplicate_run_rows = list(csv.DictReader(duplicate_run_csv.open(newline="")))
    duplicate_run_headers = list(duplicate_run_rows[0].keys())
    duplicate_run_rows.append(dict(duplicate_run_rows[0]))
    with duplicate_run_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=duplicate_run_headers)
        writer.writeheader()
        writer.writerows(duplicate_run_rows)
    duplicate_aggregate_csv = duplicate_aggregate_dir / "incremental_aggregate.csv"
    duplicate_aggregate_rows = list(csv.DictReader(duplicate_aggregate_csv.open(newline="")))
    duplicate_aggregate_headers = list(duplicate_aggregate_rows[0].keys())
    duplicate_aggregate_rows.append(dict(duplicate_aggregate_rows[0]))
    with duplicate_aggregate_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=duplicate_aggregate_headers)
        writer.writeheader()
        writer.writerows(duplicate_aggregate_rows)
    duplicate_aggregate_seed_csv = duplicate_aggregate_seed_dir / "incremental_aggregate.csv"
    duplicate_aggregate_seed_rows = list(csv.DictReader(duplicate_aggregate_seed_csv.open(newline="")))
    duplicate_aggregate_seed_headers = list(duplicate_aggregate_seed_rows[0].keys())
    for row in duplicate_aggregate_seed_rows:
        if row["method"] == "lora_nsp_fd_cd":
            row["seeds"] = "42,42,43,44"
    with duplicate_aggregate_seed_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=duplicate_aggregate_seed_headers)
        writer.writeheader()
        writer.writerows(duplicate_aggregate_seed_rows)
    missing_path_csv = missing_path_dir / "incremental_summary.csv"
    missing_path_rows = list(csv.DictReader(missing_path_csv.open(newline="")))
    missing_path_headers = list(missing_path_rows[0].keys())
    for row in missing_path_rows:
        if row["method"] == "lora_nsp_fd_cd" and row["seed"] == "42":
            row["path"] = str(missing_path_dir / "missing_lora_nsp_fd_cd_seed42_results.json")
    with missing_path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=missing_path_headers)
        writer.writeheader()
        writer.writerows(missing_path_rows)
    run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(pass_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(pass_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ]
    )
    fail_output = run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(fail_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(fail_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ],
        expect_success=False,
    )
    if "INCREMENTAL METRICS AUDIT FAILED" not in fail_output:
        raise SystemExit("incremental negative self-test did not report failure")
    if "Forgetting" not in fail_output:
        raise SystemExit("incremental negative self-test did not check forgetting")
    capped_output = run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(capped_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(capped_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ],
        expect_success=False,
    )
    if "expected <=0 for full test split" not in capped_output:
        raise SystemExit("incremental capped-eval self-test did not report full-test failure")
    seed_flip_output = run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(seed_flip_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(seed_flip_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ],
        expect_success=False,
    )
    if "seed 42" not in seed_flip_output:
        raise SystemExit("incremental seed-flip self-test did not report seed-level failure")
    shuffled_output = run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(shuffled_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(shuffled_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ],
        expect_success=False,
    )
    if "task order warning" not in shuffled_output:
        raise SystemExit("incremental shuffled-task self-test did not report order failure")
    stale_output = run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(stale_aggregate_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(stale_aggregate_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ],
        expect_success=False,
    )
    if "aggregate Average" not in stale_output or "per-run mean" not in stale_output:
        raise SystemExit("incremental stale-aggregate self-test did not report mismatch")
    extra_seed_output = run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(extra_seed_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(extra_seed_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ],
        expect_success=False,
    )
    if "unexpected seeds 45" not in extra_seed_output:
        raise SystemExit("incremental extra-seed self-test did not report unexpected seed")
    extra_method_output = run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(extra_method_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(extra_method_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ],
        expect_success=False,
    )
    if "unexpected per-run methods: lora_extra" not in extra_method_output:
        raise SystemExit("incremental extra-method self-test did not report per-run method")
    duplicate_run_output = run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(duplicate_run_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(duplicate_run_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ],
        expect_success=False,
    )
    if "duplicate per-run method/seed rows" not in duplicate_run_output:
        raise SystemExit("incremental duplicate-run self-test did not report duplicate")
    duplicate_aggregate_output = run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(duplicate_aggregate_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(duplicate_aggregate_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ],
        expect_success=False,
    )
    if "duplicate aggregate methods" not in duplicate_aggregate_output:
        raise SystemExit("incremental duplicate-aggregate self-test did not report duplicate")
    duplicate_aggregate_seed_output = run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(duplicate_aggregate_seed_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(duplicate_aggregate_seed_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ],
        expect_success=False,
    )
    if "aggregate duplicate seeds 42" not in duplicate_aggregate_seed_output:
        raise SystemExit("incremental duplicate-aggregate-seed self-test did not report duplicate")
    missing_path_output = run(
        [
            "python",
            "scripts/audit_incremental_metrics.py",
            "--summary_csv",
            str(missing_path_dir / "incremental_summary.csv"),
            "--aggregate_csv",
            str(missing_path_dir / "incremental_aggregate.csv"),
            "--expected_seeds",
            "42 43 44",
            "--expected_k",
            "10",
        ],
        expect_success=False,
    )
    if "raw result path does not exist" not in missing_path_output:
        raise SystemExit("incremental missing-path self-test did not report missing raw JSON")


def main():
    with tempfile.TemporaryDirectory(prefix="publication_audit_selftest_") as raw_tmp:
        tmp = Path(raw_tmp)
        check_claim_audit(tmp)
        check_publication_gate_artifacts(tmp)
        check_result_ledger_audit(tmp)
        check_training_audit(tmp)
        check_incremental_audit(tmp)
    print("PUBLICATION AUDIT SELFTEST PASSED")


if __name__ == "__main__":
    main()
