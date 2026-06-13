#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = (
    ("zero_shot", "CLIP-ZS"),
    ("lr_rgda", "LR-RGDA"),
    ("lada", "LADA"),
    ("ours_ensemble", "LR-RGDA+ZS"),
    ("lada_zs", "LADA+ZS"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_markdown", type=Path, required=True)
    return parser.parse_args()


def format_score(values):
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{mean(values):.2f} +/- {stdev(values):.2f}"


def main():
    args = parse_args()
    grouped = defaultdict(lambda: defaultdict(list))

    for path in sorted(args.input_dir.glob("*_seed*.json")):
        with path.open() as handle:
            result = json.load(handle)
        experiment = result.get("experiment_name") or path.stem
        averages = result["metrics"]["id"]["average"]
        for key, _ in METRICS:
            if key in averages:
                grouped[experiment][key].append(float(averages[key]))

    rows = []
    for experiment in sorted(grouped):
        row = {"experiment": experiment}
        for key, label in METRICS:
            values = grouped[experiment].get(key, [])
            row[label] = format_score(values) if values else ""
        rows.append(row)

    headers = ["experiment"] + [label for _, label in METRICS]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Replay | " + " | ".join(headers[1:]) + " |",
        "|---|" + "|".join("---:" for _ in headers[1:]) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + row["experiment"] + " | "
            + " | ".join(row[h] for h in headers[1:]) + " |"
        )
    args.output_markdown.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
