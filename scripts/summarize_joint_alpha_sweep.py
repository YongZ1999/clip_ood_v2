#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_markdown", type=Path, required=True)
    parser.add_argument("--experiment", default="", help="Optional experiment_name filter.")
    parser.add_argument("--expected_seeds", default="")
    return parser.parse_args()


def parse_seed_list(raw):
    return [seed for seed in raw.replace(",", " ").split() if seed]


def fmt_score(values):
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{mean(values):.2f} +/- {stdev(values):.2f}"


def best_pair(sweep):
    if not sweep:
        return None
    return max(((float(alpha), float(acc)) for alpha, acc in sweep), key=lambda item: item[1])


def main():
    args = parse_args()
    expected_seeds = set(parse_seed_list(args.expected_seeds))
    records = []
    lr_grid = defaultdict(list)
    lada_grid = defaultdict(list)

    for path in sorted(args.input_dir.glob("*_seed*.json")):
        with path.open() as handle:
            data = json.load(handle)
        experiment = data.get("experiment_name") or path.stem.rsplit("_seed", 1)[0]
        if args.experiment and experiment != args.experiment:
            continue
        seed = str(data.get("seed") or path.stem.rsplit("_seed", 1)[-1])
        sensitivity = data.get("alpha_sensitivity", {}).get("id", {})
        lr_sweep = sensitivity.get("sweep") or []
        lada_sweep = sensitivity.get("lada_sweep") or []
        if not lr_sweep and not lada_sweep:
            continue
        lr_best = best_pair(lr_sweep)
        lada_best = best_pair(lada_sweep)
        averages = data.get("metrics", {}).get("id", {}).get("average", {})
        arguments = data.get("arguments", {})
        for alpha, acc in lr_sweep:
            lr_grid[float(alpha)].append(float(acc))
        for alpha, acc in lada_sweep:
            lada_grid[float(alpha)].append(float(acc))
        records.append(
            {
                "experiment": experiment,
                "seed": seed,
                "source file": str(path),
                "fixed alpha": str(arguments.get("alpha", "")),
                "fixed lada alpha": str(arguments.get("lada_alpha", "")),
                "LR-RGDA": f"{float(averages.get('lr_rgda', 0.0)):.2f}",
                "LADA": f"{float(averages.get('lada', 0.0)):.2f}",
                "fixed LR-RGDA+ZS": f"{float(averages.get('ours_ensemble', 0.0)):.2f}",
                "fixed LADA+ZS": f"{float(averages.get('lada_zs', 0.0)):.2f}",
                "best LR alpha": f"{lr_best[0]:.2f}" if lr_best else "",
                "best LR-RGDA+ZS": f"{lr_best[1]:.2f}" if lr_best else "",
                "best LADA alpha": f"{lada_best[0]:.2f}" if lada_best else "",
                "best LADA+ZS": f"{lada_best[1]:.2f}" if lada_best else "",
            }
        )

    if not records:
        raise SystemExit(f"No alpha_sensitivity records found in {args.input_dir}")

    present_seeds = {record["seed"] for record in records}
    missing_seeds = sorted(expected_seeds - present_seeds)

    grid_rows = []
    all_alphas = sorted(set(lr_grid) | set(lada_grid))
    for alpha in all_alphas:
        grid_rows.append(
            {
                "alpha": f"{alpha:.2f}",
                "LR-RGDA+ZS": fmt_score(lr_grid.get(alpha, [])),
                "LADA+ZS": fmt_score(lada_grid.get(alpha, [])),
                "LR n": str(len(lr_grid.get(alpha, []))),
                "LADA n": str(len(lada_grid.get(alpha, []))),
            }
        )

    best_lr_alpha, best_lr_values = max(lr_grid.items(), key=lambda item: mean(item[1]))
    best_lada_alpha, best_lada_values = max(lada_grid.items(), key=lambda item: mean(item[1]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        headers = list(records[0].keys())
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)

    grid_csv = args.output_csv.with_name(args.output_csv.stem + "_grid.csv")
    with grid_csv.open("w", newline="") as handle:
        headers = ["alpha", "LR-RGDA+ZS", "LADA+ZS", "LR n", "LADA n"]
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(grid_rows)

    lines = [
        "# Joint Ensemble Alpha Sweep",
        "",
        "Selection note: this is a test-set sensitivity/oracle summary unless the "
        "sweep is driven by a held-out validation protocol.",
        "",
        f"- seeds: {','.join(sorted(present_seeds))}",
        f"- missing expected seeds: {','.join(missing_seeds)}",
        f"- best aggregate LR-RGDA+ZS alpha: {best_lr_alpha:.2f} ({fmt_score(best_lr_values)})",
        f"- best aggregate LADA+ZS alpha: {best_lada_alpha:.2f} ({fmt_score(best_lada_values)})",
        "",
        "## Per Seed",
        "",
        "| " + " | ".join(records[0].keys()) + " |",
        "| " + " | ".join("---" for _ in records[0]) + " |",
    ]
    for record in records:
        lines.append("| " + " | ".join(record[key] for key in records[0]) + " |")
    lines.extend(
        [
            "",
            "## Aggregate Grid",
            "",
            "| alpha | LR-RGDA+ZS | LADA+ZS | LR n | LADA n |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in grid_rows:
        lines.append(
            f"| {row['alpha']} | {row['LR-RGDA+ZS']} | {row['LADA+ZS']} | "
            f"{row['LR n']} | {row['LADA n']} |"
        )
    args.output_markdown.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
