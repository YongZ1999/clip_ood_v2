#!/usr/bin/env python3
"""Aggregate Transfer/Average/Last JSON files across random seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=PATH",
                        help="Repeat once per seed; equal labels are aggregated.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_item(item):
    if "=" not in item:
        raise ValueError(f"Invalid input {item!r}; expected LABEL=PATH")
    label, raw_path = item.split("=", 1)
    path = Path(raw_path)
    if not label or not path.is_file():
        raise ValueError(f"Invalid input {item!r}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload.get("summary_metrics"))
    if not isinstance(metrics, dict) or not {"transfer", "average", "last"}.issubset(metrics):
        raise ValueError(f"{path}: missing metrics.transfer/average/last")
    return label, str(path), {key: float(metrics[key]) for key in ("transfer", "average", "last")}


def stat(values):
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def fmt(value):
    return f"{value['mean']:.2f}" if value["n"] == 1 else f"{value['mean']:.2f} ± {value['std']:.2f}"


def main():
    args = parse_args()
    grouped = defaultdict(list)
    for item in args.input:
        label, path, metrics = parse_item(item)
        grouped[label].append({"path": path, "metrics": metrics})

    summary = {}
    for label, runs in grouped.items():
        summary[label] = {
            "runs": runs,
            "metrics": {key: stat([run["metrics"][key] for run in runs])
                        for key in ("transfer", "average", "last")},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps({"schema_version": 1, "summary": summary}, indent=2), encoding="utf-8")
    lines = ["# Continual Classification Summary", "", "| Method | Transfer | Average | Last |", "|---|---:|---:|---:|"]
    for label, value in summary.items():
        metrics = value["metrics"]
        lines.append(
            f"| {label} | {fmt(metrics['transfer'])} | {fmt(metrics['average'])} | {fmt(metrics['last'])} |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.with_suffix('.json')} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
