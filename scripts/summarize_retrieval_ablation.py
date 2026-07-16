#!/usr/bin/env python3
"""Aggregate per-task retrieval JSON files into paper-ready ablation tables.

The main incremental runner already evaluates retrieval after every task.  This
script is deliberately read-only: it validates coverage and summarizes existing
artifacts, so adding retrieval results to an ablation never requires rerunning
an encoder whose ``*_retrieval.json`` is complete.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("i2t_r@1", "t2i_r@1", "i2t_r@5", "t2i_r@5", "i2t_r@10", "t2i_r@10")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", action="append", required=True, metavar="LABEL=PATH",
        help="Repeat for every seed artifact. Identical LABEL values are aggregated as seeds.",
    )
    parser.add_argument("--output", type=Path, required=True,
                        help="Output stem; .json and .md are written.")
    parser.add_argument("--expected-tasks", type=int, default=10,
                        help="Required per-dataset task count (0 disables the coverage check).")
    return parser.parse_args()


def parse_inputs(items):
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --input {item!r}; expected LABEL=PATH")
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not label or not path.is_file():
            raise ValueError(f"Invalid retrieval input: {item!r}")
        parsed.append((label, path))
    return parsed


def load_rows(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list or a payload containing rows")
    required = {"step", "task", "dataset", *METRICS}
    missing = [idx for idx, row in enumerate(rows) if not required.issubset(row)]
    if missing:
        raise ValueError(f"{path}: rows missing required retrieval fields at indices {missing[:5]}")
    return rows


def summarize_run(path, expected_tasks):
    by_dataset = defaultdict(list)
    for row in load_rows(path):
        by_dataset[row["dataset"]].append(row)

    result = {}
    for dataset, rows in by_dataset.items():
        rows = sorted(rows, key=lambda row: (int(row["step"]), str(row["task"])))
        steps = [int(row["step"]) for row in rows]
        if len(set(steps)) != len(steps):
            raise ValueError(f"{path}: {dataset} has duplicate retrieval steps: {steps}")
        if expected_tasks and len(rows) != expected_tasks:
            raise ValueError(
                f"{path}: {dataset} has {len(rows)} task rows; expected {expected_tasks}")
        averages = {metric: sum(float(row[metric]) for row in rows) / len(rows) for metric in METRICS}
        last = {metric: float(rows[-1][metric]) for metric in METRICS}
        result[dataset] = {"num_tasks": len(rows), "average": averages, "last": last}
    return result


def mean_std(values):
    mean = statistics.fmean(values)
    # Match historical paper tables: std over the fixed seed set, not an
    # unbiased estimator of a hypothetical larger seed population.
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(values)}


def aggregate(parsed_inputs, expected_tasks):
    grouped = defaultdict(list)
    for label, path in parsed_inputs:
        grouped[label].append({"path": str(path), "datasets": summarize_run(path, expected_tasks)})

    summary = {}
    for label, runs in grouped.items():
        datasets = sorted({dataset for run in runs for dataset in run["datasets"]})
        if any(set(run["datasets"]) != set(datasets) for run in runs):
            raise ValueError(f"{label}: seeds do not cover identical retrieval datasets")
        summary[label] = {"runs": runs, "datasets": {}}
        for dataset in datasets:
            dataset_summary = {}
            for phase in ("average", "last"):
                metrics = {
                    metric: mean_std([run["datasets"][dataset][phase][metric] for run in runs])
                    for metric in METRICS
                }
                metrics["mean_recall_r@1"] = mean_std([
                    (run["datasets"][dataset][phase]["i2t_r@1"] +
                     run["datasets"][dataset][phase]["t2i_r@1"]) / 2.0
                    for run in runs
                ])
                dataset_summary[phase] = metrics
            summary[label]["datasets"][dataset] = dataset_summary
    return summary


def fmt(stat):
    if stat["n"] == 1:
        return f"{stat['mean']:.2f}"
    return f"{stat['mean']:.2f} ± {stat['std']:.2f}"


def markdown_report(summary):
    lines = ["# Retrieval Ablation Summary", ""]
    for label, value in summary.items():
        lines += [f"## {label}", "", f"Seeds / artifacts: {len(value['runs'])}", ""]
        lines += [
            "| Dataset | Avg I2T R@1 | Avg T2I R@1 | Avg mR@1 | Last I2T R@1 | Last T2I R@1 | Last mR@1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for dataset, values in value["datasets"].items():
            avg, last = values["average"], values["last"]
            lines.append(
                f"| {dataset} | {fmt(avg['i2t_r@1'])} | {fmt(avg['t2i_r@1'])} | "
                f"{fmt(avg['mean_recall_r@1'])} | {fmt(last['i2t_r@1'])} | "
                f"{fmt(last['t2i_r@1'])} | {fmt(last['mean_recall_r@1'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    summary = aggregate(parse_inputs(args.input), args.expected_tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps({"schema_version": 1, "summary": summary}, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown_report(summary), encoding="utf-8")
    print(f"Wrote {args.output.with_suffix('.json')} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
