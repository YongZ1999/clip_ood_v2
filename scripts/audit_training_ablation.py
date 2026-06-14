#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean


EXPECTED_EXPERIMENTS = (
    "lora_vanilla",
    "lora_nsp",
    "lora_nsp_fd",
    "lora_nsp_fd_cd",
)

PRIMARY_METRICS = (
    "CLIP-ZS",
    "LR-RGDA",
    "LR-RGDA+ZS",
)

EXPECTED_PROTOCOL = {
    "num shots": "16",
    "classifier feature transform": "test",
    "enable lada": "True",
    "lada k": "16",
    "num centers": "4",
    "tune vision encoder": "True",
}

SOURCE_METRICS = {
    "CLIP-ZS": "zero_shot",
    "LR-RGDA": "lr_rgda",
    "LR-RGDA+ZS": "ours_ensemble",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("experiments/joint_training_ablation_20260614/summary.csv"),
    )
    parser.add_argument("--expected_seeds", default="42 43 44")
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def parse_seed_list(raw):
    return {seed for seed in raw.replace(",", " ").split() if seed}


def parse_seed_tokens(raw):
    return [seed for seed in raw.replace(",", " ").split() if seed]


def duplicate_items(items):
    seen = set()
    duplicates = []
    for item in items:
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    return sorted(set(duplicates))


def parse_score(value):
    if not value:
        return None
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", value)
    return float(match.group(1)) if match else None


def parse_seed_scores(value):
    scores = {}
    if not value:
        return scores
    for item in value.split(","):
        if ":" not in item:
            continue
        seed, raw_score = item.split(":", 1)
        score = parse_score(raw_score)
        if score is not None:
            scores[seed.strip()] = score
    return scores


def parse_seed_score_tokens(value):
    seeds = []
    if not value:
        return seeds
    for item in value.split(","):
        if ":" not in item:
            continue
        seed, _ = item.split(":", 1)
        seed = seed.strip()
        if seed:
            seeds.append(seed)
    return seeds


def parse_seed_sources(value):
    sources = {}
    if not value:
        return sources
    for item in value.split(","):
        if ":" not in item:
            continue
        seed, source = item.split(":", 1)
        seed = seed.strip()
        source = source.strip()
        if seed and source:
            sources[seed] = source
    return sources


def read_row_list(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_rows(path):
    return {row["experiment"]: row for row in read_row_list(path)}


def metric_delta(rows, metric, lhs, rhs):
    lhs_value = parse_score(rows[lhs].get(metric, ""))
    rhs_value = parse_score(rows[rhs].get(metric, ""))
    if lhs_value is None or rhs_value is None:
        return None
    return lhs_value - rhs_value


def mean_seed_score(scores, expected_seeds):
    values = [scores[seed] for seed in sorted(expected_seeds) if seed in scores]
    if not values:
        return None
    return mean(values)


def source_metric_value(source_data, metric):
    raw_key = SOURCE_METRICS[metric]
    metrics = source_data.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    id_metrics = metrics.get("id", {})
    if not isinstance(id_metrics, dict):
        return None
    average = id_metrics.get("average", {})
    if not isinstance(average, dict) or raw_key not in average:
        return None
    return float(average[raw_key])


def main():
    args = parse_args()
    if not args.summary_csv.exists():
        message = f"training summary missing: {args.summary_csv}"
        if args.allow_missing:
            print(f"TRAINING ABLATION AUDIT PENDING: {message}")
            return
        raise SystemExit(message)

    row_list = read_row_list(args.summary_csv)
    rows = {row["experiment"]: row for row in row_list}
    expected_seeds = parse_seed_list(args.expected_seeds)
    failures = []
    warnings = []

    seen_experiments = set()
    duplicate_experiments = []
    for row in row_list:
        experiment = row.get("experiment", "")
        if experiment in seen_experiments:
            duplicate_experiments.append(experiment)
        seen_experiments.add(experiment)
    if duplicate_experiments:
        failures.append(
            "duplicate experiments: " + ", ".join(sorted(set(duplicate_experiments)))
        )

    missing_experiments = [name for name in EXPECTED_EXPERIMENTS if name not in rows]
    if missing_experiments:
        failures.append(f"missing experiments: {', '.join(missing_experiments)}")
    unexpected_experiments = sorted(set(rows) - set(EXPECTED_EXPERIMENTS))
    if unexpected_experiments:
        failures.append(f"unexpected experiments: {', '.join(unexpected_experiments)}")

    for name in EXPECTED_EXPERIMENTS:
        if name not in rows:
            continue
        row = rows[name]
        seeds = parse_seed_list(row.get("seeds", ""))
        duplicate_seeds = duplicate_items(parse_seed_tokens(row.get("seeds", "")))
        if duplicate_seeds:
            failures.append(
                f"{name}: duplicate seeds {', '.join(duplicate_seeds)}"
            )
        missing_seeds = expected_seeds - seeds
        if missing_seeds:
            failures.append(f"{name}: missing expected seeds {', '.join(sorted(missing_seeds))}")
        unexpected_seeds = seeds - expected_seeds
        if unexpected_seeds:
            failures.append(
                f"{name}: unexpected seeds {', '.join(sorted(unexpected_seeds))}"
            )
        try:
            n = int(row.get("n", "0") or "0")
        except ValueError:
            n = 0
        if n != len(expected_seeds):
            failures.append(f"{name}: n={n}, expected exactly {len(expected_seeds)}")
        source_files = parse_seed_sources(row.get("source files", ""))
        missing_source_seeds = expected_seeds - set(source_files)
        if missing_source_seeds:
            failures.append(
                f"{name}: missing source files for "
                f"{', '.join(sorted(missing_source_seeds))}"
            )
        unexpected_source_seeds = set(source_files) - expected_seeds
        if unexpected_source_seeds:
            failures.append(
                f"{name}: unexpected source file seeds "
                f"{', '.join(sorted(unexpected_source_seeds))}"
            )
        for seed in sorted(expected_seeds):
            source = source_files.get(seed)
            if source and not Path(source).exists():
                failures.append(
                    f"{name}: source file for seed {seed} does not exist: {source}"
                )
            elif source:
                with Path(source).open() as handle:
                    source_data = json.load(handle)
                source_experiment = source_data.get("experiment_name")
                if source_experiment != name:
                    failures.append(
                        f"{name}: source file seed {seed} experiment_name="
                        f"{source_experiment}, expected {name}"
                    )
                source_seed = str(source_data.get("seed", ""))
                if source_seed != seed:
                    failures.append(
                        f"{name}: source file seed {seed} has seed={source_seed}"
                    )
                arguments = (
                    source_data.get("arguments", {})
                    if isinstance(source_data.get("arguments"), dict)
                    else {}
                )
                for raw_column, summary_column in (
                    ("num_shots", "num shots"),
                    ("classifier_feature_transform", "classifier feature transform"),
                    ("enable_lada", "enable lada"),
                    ("lada_k", "lada k"),
                    ("num_centers", "num centers"),
                    ("tune_vision_encoder", "tune vision encoder"),
                ):
                    source_value = str(arguments.get(raw_column, ""))
                    expected_value = EXPECTED_PROTOCOL[summary_column]
                    if source_value != expected_value:
                        failures.append(
                            f"{name}: source file seed {seed} {raw_column}="
                            f"{source_value}, expected {expected_value}"
                        )
                for metric in PRIMARY_METRICS:
                    by_seed = parse_seed_scores(row.get(f"{metric} by seed", ""))
                    summary_value = by_seed.get(seed)
                    raw_value = source_metric_value(source_data, metric)
                    if raw_value is None:
                        failures.append(
                            f"{name}: source file seed {seed} missing {metric}"
                        )
                    elif summary_value is not None and abs(raw_value - summary_value) > 0.015:
                        failures.append(
                            f"{name}: source file seed {seed} {metric}="
                            f"{raw_value:.2f}, summary={summary_value:.2f}"
                        )
        for column, expected_value in EXPECTED_PROTOCOL.items():
            value = row.get(column, "")
            if not value:
                failures.append(f"{name}: missing protocol column {column}")
            elif value != expected_value:
                failures.append(f"{name}: {column}={value}, expected {expected_value}")
        for metric in PRIMARY_METRICS:
            if parse_score(row.get(metric, "")) is None:
                failures.append(f"{name}: missing or unparsable {metric}")
            by_seed = parse_seed_scores(row.get(f"{metric} by seed", ""))
            duplicate_metric_seeds = duplicate_items(
                parse_seed_score_tokens(row.get(f"{metric} by seed", ""))
            )
            if duplicate_metric_seeds:
                failures.append(
                    f"{name}: duplicate {metric} by seed values for "
                    f"{', '.join(duplicate_metric_seeds)}"
                )
            missing_metric_seeds = expected_seeds - set(by_seed)
            if missing_metric_seeds:
                failures.append(
                    f"{name}: missing {metric} by seed values for "
                    f"{', '.join(sorted(missing_metric_seeds))}"
                )
            unexpected_metric_seeds = set(by_seed) - expected_seeds
            if unexpected_metric_seeds:
                failures.append(
                    f"{name}: unexpected {metric} by seed values for "
                    f"{', '.join(sorted(unexpected_metric_seeds))}"
                )
            aggregate_value = parse_score(row.get(metric, ""))
            recomputed_value = mean_seed_score(by_seed, expected_seeds)
            if aggregate_value is not None and recomputed_value is not None:
                if abs(aggregate_value - recomputed_value) > 0.015:
                    failures.append(
                        f"{name}: aggregate {metric}={aggregate_value:.2f}, "
                        f"by-seed mean={recomputed_value:.2f}"
                    )

    if not failures:
        full = "lora_nsp_fd_cd"
        for baseline in ("lora_vanilla", "lora_nsp"):
            for metric in PRIMARY_METRICS:
                delta = metric_delta(rows, metric, full, baseline)
                if delta is None:
                    failures.append(f"cannot compare {full} vs {baseline} on {metric}")
                    continue
                status = "improves" if delta > 0 else "does not improve"
                print(f"{full} - {baseline} {metric}: {delta:+.2f} ({status})")
                if delta <= 0:
                    warnings.append(f"{full} does not improve over {baseline} on {metric}: {delta:+.2f}")
                full_by_seed = parse_seed_scores(rows[full].get(f"{metric} by seed", ""))
                base_by_seed = parse_seed_scores(rows[baseline].get(f"{metric} by seed", ""))
                if full_by_seed and base_by_seed:
                    for seed in sorted(expected_seeds):
                        if seed not in full_by_seed or seed not in base_by_seed:
                            failures.append(
                                f"cannot compare {full} vs {baseline} on {metric} seed {seed}"
                            )
                            continue
                        seed_delta = full_by_seed[seed] - base_by_seed[seed]
                        seed_status = "improves" if seed_delta > 0 else "does not improve"
                        print(
                            f"{full} - {baseline} {metric} seed {seed}: "
                            f"{seed_delta:+.2f} ({seed_status})"
                        )
                        if seed_delta <= 0:
                            warnings.append(
                                f"{full} does not improve over {baseline} on "
                                f"{metric} seed {seed}: {seed_delta:+.2f}"
                            )

    print(f"audited training summary: {args.summary_csv}")
    print(f"expected experiments: {', '.join(EXPECTED_EXPERIMENTS)}")
    print(f"expected seeds: {', '.join(sorted(expected_seeds))}")

    if failures:
        print("TRAINING ABLATION AUDIT FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    if warnings:
        print("TRAINING ABLATION AUDIT FAILED: required improvement not satisfied")
        for warning in warnings:
            print(f"- {warning}")
        raise SystemExit(1)

    print("TRAINING ABLATION AUDIT PASSED")


if __name__ == "__main__":
    main()
