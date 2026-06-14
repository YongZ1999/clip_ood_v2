#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from statistics import mean


EXPECTED_METHODS = (
    "lora_vanilla",
    "lora_nsp",
    "lora_nsp_fd",
    "lora_nsp_fd_cd",
)

HIGHER_IS_BETTER_METRICS = ("Transfer", "Average", "Last")
LOWER_IS_BETTER_METRICS = ("Forgetting",)
PRIMARY_METRICS = HIGHER_IS_BETTER_METRICS + LOWER_IS_BETTER_METRICS
DELTA_COLUMNS = (
    "Transfer stored-computed",
    "Average stored-computed",
    "Last stored-computed",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("experiments/joint_incremental_metrics_20260614/incremental_summary.csv"),
    )
    parser.add_argument(
        "--aggregate_csv",
        type=Path,
        default=Path("experiments/joint_incremental_metrics_20260614/incremental_aggregate.csv"),
    )
    parser.add_argument("--expected_seeds", default="42 43 44")
    parser.add_argument("--expected_k", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def parse_list(raw):
    return {item for item in raw.replace(",", " ").split() if item}


def parse_tokens(raw):
    return [item for item in raw.replace(",", " ").split() if item]


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
    match = re.match(r"\s*([+-]?\d+(?:\.\d+)?)", value)
    return float(match.group(1)) if match else None


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_by_method(path):
    return {row["method"]: row for row in read_rows(path)}


def mean_score(rows, metric):
    values = [parse_score(row.get(metric, "")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return mean(values)


def main():
    args = parse_args()
    missing_paths = [path for path in (args.summary_csv, args.aggregate_csv) if not path.exists()]
    if missing_paths:
        message = "missing incremental metric files: " + ", ".join(str(path) for path in missing_paths)
        if args.allow_missing:
            print(f"INCREMENTAL METRICS AUDIT PENDING: {message}")
            return
        raise SystemExit(message)

    expected_seeds = parse_list(args.expected_seeds)
    per_run = read_rows(args.summary_csv)
    aggregate_rows = read_rows(args.aggregate_csv)
    aggregate = {row["method"]: row for row in aggregate_rows}
    failures = []
    warnings = []

    seen_aggregate_methods = set()
    duplicate_aggregate_methods = []
    for row in aggregate_rows:
        method = row.get("method", "")
        if method in seen_aggregate_methods:
            duplicate_aggregate_methods.append(method)
        seen_aggregate_methods.add(method)
    if duplicate_aggregate_methods:
        failures.append(
            "duplicate aggregate methods: "
            + ", ".join(sorted(set(duplicate_aggregate_methods)))
        )

    seen_runs = set()
    duplicate_runs = []
    for row in per_run:
        key = (row.get("method", ""), row.get("seed", ""))
        if key in seen_runs:
            duplicate_runs.append(key)
        seen_runs.add(key)
    if duplicate_runs:
        failures.append(
            "duplicate per-run method/seed rows: "
            + ", ".join(
                f"{method} seed {seed}"
                for method, seed in sorted(set(duplicate_runs))
            )
        )

    missing_methods = [method for method in EXPECTED_METHODS if method not in aggregate]
    if missing_methods:
        failures.append(f"missing aggregate methods: {', '.join(missing_methods)}")
    unexpected_aggregate_methods = sorted(set(aggregate) - set(EXPECTED_METHODS))
    if unexpected_aggregate_methods:
        failures.append(
            f"unexpected aggregate methods: {', '.join(unexpected_aggregate_methods)}"
        )

    per_run_methods = {row.get("method", "") for row in per_run if row.get("method", "")}
    unexpected_per_run_methods = sorted(per_run_methods - set(EXPECTED_METHODS))
    if unexpected_per_run_methods:
        failures.append(
            f"unexpected per-run methods: {', '.join(unexpected_per_run_methods)}"
        )

    per_run_by_method = {method: [] for method in EXPECTED_METHODS}
    for row in per_run:
        if row.get("method") in per_run_by_method:
            per_run_by_method[row["method"]].append(row)

    for method in EXPECTED_METHODS:
        rows = per_run_by_method.get(method, [])
        if not rows:
            failures.append(f"{method}: no per-run rows")
            continue
        seeds = {row.get("seed", "") for row in rows if row.get("seed", "")}
        missing_seeds = expected_seeds - seeds
        if missing_seeds:
            failures.append(f"{method}: missing expected seeds {', '.join(sorted(missing_seeds))}")
        unexpected_seeds = seeds - expected_seeds
        if unexpected_seeds:
            failures.append(f"{method}: unexpected seeds {', '.join(sorted(unexpected_seeds))}")
        if len(rows) != len(expected_seeds):
            failures.append(f"{method}: n={len(rows)}, expected exactly {len(expected_seeds)}")
        for row in rows:
            raw_path = row.get("path", "")
            if not raw_path:
                failures.append(f"{method}: missing raw result path")
            elif not Path(raw_path).exists():
                failures.append(f"{method}: raw result path does not exist: {raw_path}")
            if row.get("K") != str(args.expected_k):
                failures.append(f"{method}: {row.get('path')} K={row.get('K')}, expected {args.expected_k}")
            eval_max_samples = parse_score(row.get("eval max samples", ""))
            if eval_max_samples is None:
                failures.append(f"{method}: {row.get('path')} missing eval max samples")
            elif eval_max_samples > 0:
                failures.append(
                    f"{method}: {row.get('path')} eval max samples={eval_max_samples:g}, "
                    "expected <=0 for full test split"
                )
            if row.get("missing expected tasks"):
                failures.append(f"{method}: {row.get('path')} missing tasks {row.get('missing expected tasks')}")
            if "task order warning" not in row:
                failures.append(f"{method}: {row.get('path')} missing task order warning column")
            elif row.get("task order warning"):
                failures.append(
                    f"{method}: {row.get('path')} task order warning {row.get('task order warning')}"
                )
            if row.get("matrix warning"):
                failures.append(f"{method}: {row.get('path')} matrix warning {row.get('matrix warning')}")
            for metric in PRIMARY_METRICS:
                if parse_score(row.get(metric, "")) is None:
                    failures.append(f"{method}: {row.get('path')} missing or unparsable {metric}")
            for column in DELTA_COLUMNS:
                value = parse_score(row.get(column, ""))
                if value is None:
                    failures.append(f"{method}: {row.get('path')} missing {column}")
                elif abs(value) > args.tolerance:
                    failures.append(
                        f"{method}: {row.get('path')} {column}={value:+.4f}, "
                        f"tolerance={args.tolerance}"
                    )

    if not failures:
        for method, row in aggregate.items():
            if "task order warning" not in row:
                failures.append(f"{method}: aggregate missing task order warning column")
            elif row.get("task order warning"):
                failures.append(f"{method}: aggregate task order warning {row.get('task order warning')}")
            if method in EXPECTED_METHODS:
                aggregate_seeds = parse_list(row.get("seeds", ""))
                duplicate_aggregate_seeds = duplicate_items(parse_tokens(row.get("seeds", "")))
                if duplicate_aggregate_seeds:
                    failures.append(
                        f"{method}: aggregate duplicate seeds "
                        f"{', '.join(duplicate_aggregate_seeds)}"
                    )
                missing_aggregate_seeds = expected_seeds - aggregate_seeds
                if missing_aggregate_seeds:
                    failures.append(
                        f"{method}: aggregate missing expected seeds "
                        f"{', '.join(sorted(missing_aggregate_seeds))}"
                    )
                unexpected_aggregate_seeds = aggregate_seeds - expected_seeds
                if unexpected_aggregate_seeds:
                    failures.append(
                        f"{method}: aggregate unexpected seeds "
                        f"{', '.join(sorted(unexpected_aggregate_seeds))}"
                    )
                try:
                    aggregate_n = int(row.get("n", "0") or "0")
                except ValueError:
                    aggregate_n = 0
                if aggregate_n != len(expected_seeds):
                    failures.append(
                        f"{method}: aggregate n={aggregate_n}, expected exactly "
                        f"{len(expected_seeds)}"
                    )

    if not failures:
        for method in EXPECTED_METHODS:
            aggregate_row = aggregate.get(method)
            if not aggregate_row:
                continue
            for metric in PRIMARY_METRICS:
                aggregate_value = parse_score(aggregate_row.get(metric, ""))
                recomputed_value = mean_score(per_run_by_method[method], metric)
                if aggregate_value is None or recomputed_value is None:
                    failures.append(f"{method}: cannot verify aggregate {metric}")
                    continue
                if abs(aggregate_value - recomputed_value) > 0.015:
                    failures.append(
                        f"{method}: aggregate {metric}={aggregate_value:.2f}, "
                        f"per-run mean={recomputed_value:.2f}"
                    )

    if not failures:
        full = "lora_nsp_fd_cd"
        for baseline in ("lora_vanilla", "lora_nsp"):
            for metric in HIGHER_IS_BETTER_METRICS:
                full_value = parse_score(aggregate[full].get(metric, ""))
                base_value = parse_score(aggregate[baseline].get(metric, ""))
                if full_value is None or base_value is None:
                    failures.append(f"cannot compare {full} vs {baseline} on {metric}")
                    continue
                delta = full_value - base_value
                status = "improves" if delta > 0 else "does not improve"
                print(f"{full} - {baseline} {metric}: {delta:+.2f} ({status})")
                if delta <= 0:
                    warnings.append(f"{full} does not improve over {baseline} on {metric}: {delta:+.2f}")
                full_by_seed = {
                    row.get("seed", ""): parse_score(row.get(metric, ""))
                    for row in per_run_by_method[full]
                }
                base_by_seed = {
                    row.get("seed", ""): parse_score(row.get(metric, ""))
                    for row in per_run_by_method[baseline]
                }
                for seed in sorted(expected_seeds):
                    full_seed_value = full_by_seed.get(seed)
                    base_seed_value = base_by_seed.get(seed)
                    if full_seed_value is None or base_seed_value is None:
                        failures.append(
                            f"cannot compare {full} vs {baseline} on {metric} seed {seed}"
                        )
                        continue
                    seed_delta = full_seed_value - base_seed_value
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
            for metric in LOWER_IS_BETTER_METRICS:
                full_value = parse_score(aggregate[full].get(metric, ""))
                base_value = parse_score(aggregate[baseline].get(metric, ""))
                if full_value is None or base_value is None:
                    failures.append(f"cannot compare {full} vs {baseline} on {metric}")
                    continue
                delta = full_value - base_value
                status = "improves" if delta < 0 else "does not improve"
                print(f"{full} - {baseline} {metric}: {delta:+.2f} ({status})")
                if delta >= 0:
                    warnings.append(
                        f"{full} does not reduce {metric} vs {baseline}: {delta:+.2f}"
                    )
                full_by_seed = {
                    row.get("seed", ""): parse_score(row.get(metric, ""))
                    for row in per_run_by_method[full]
                }
                base_by_seed = {
                    row.get("seed", ""): parse_score(row.get(metric, ""))
                    for row in per_run_by_method[baseline]
                }
                for seed in sorted(expected_seeds):
                    full_seed_value = full_by_seed.get(seed)
                    base_seed_value = base_by_seed.get(seed)
                    if full_seed_value is None or base_seed_value is None:
                        failures.append(
                            f"cannot compare {full} vs {baseline} on {metric} seed {seed}"
                        )
                        continue
                    seed_delta = full_seed_value - base_seed_value
                    seed_status = "improves" if seed_delta < 0 else "does not improve"
                    print(
                        f"{full} - {baseline} {metric} seed {seed}: "
                        f"{seed_delta:+.2f} ({seed_status})"
                    )
                    if seed_delta >= 0:
                        warnings.append(
                            f"{full} does not reduce {metric} vs {baseline} "
                            f"seed {seed}: {seed_delta:+.2f}"
                        )

    print(f"audited incremental per-run summary: {args.summary_csv}")
    print(f"audited incremental aggregate summary: {args.aggregate_csv}")
    print(f"expected methods: {', '.join(EXPECTED_METHODS)}")
    print(f"expected seeds: {', '.join(sorted(expected_seeds))}")
    print(f"expected K: {args.expected_k}")

    if failures:
        print("INCREMENTAL METRICS AUDIT FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    if warnings:
        print("INCREMENTAL METRICS AUDIT FAILED: required improvement not satisfied")
        for warning in warnings:
            print(f"- {warning}")
        raise SystemExit(1)

    print("INCREMENTAL METRICS AUDIT PASSED")


if __name__ == "__main__":
    main()
