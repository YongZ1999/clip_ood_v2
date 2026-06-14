#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--output_markdown", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--selftest_treat_pending_as_pass",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def run_check(command):
    proc = subprocess.run(command, text=True, capture_output=True)
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode, output


def gate(name, status, evidence, action=""):
    return {
        "gate": name,
        "status": status,
        "evidence": evidence,
        "next_action": action,
    }


def contains_all(path, patterns):
    text = path.read_text()
    missing = [pattern for pattern in patterns if pattern not in text]
    return missing


def status_from_audit(command, pass_text, pending_text=None):
    code, output = run_check(command)
    if code == 0 and pass_text in output:
        return "PASS", output
    if pending_text and code == 0 and pending_text in output:
        return "PENDING", output
    return "FAIL", output


def main():
    args = parse_args()
    gates = []

    status, evidence = status_from_audit(
        [
            "python",
            "scripts/audit_paper_claims.py",
            "paper_writing/paper-template/paper_draft.tex",
            "experiments/README_publication_repro.md",
        ],
        "CLAIM AUDIT PASSED",
    )
    gates.append(
        gate(
            "Paper claim boundary",
            status,
            evidence,
            "Remove visible TODOs or overclaims from paper_draft.tex.",
        )
    )

    status, evidence = status_from_audit(
        ["python", "scripts/audit_result_ledger.py"],
        "RESULT LEDGER AUDIT PASSED",
    )
    gates.append(
        gate(
            "Inference replay result ledger",
            status,
            evidence,
            "Update experiments/result_ledger.json or paper/README table values.",
        )
    )

    status, evidence = status_from_audit(
        ["python", "scripts/audit_protocol_config.py"],
        "PROTOCOL CONFIG AUDIT PASSED",
    )
    gates.append(
        gate(
            "Fair protocol configuration",
            status,
            evidence,
            "Keep launcher and README protocol fields aligned with the fair test-transform comparison.",
        )
    )

    dpt_missing = []
    dpt_missing.extend(
        f"paper: {item}"
        for item in contains_all(
            Path("paper_writing/paper-template/paper_draft.tex"),
            [
                "not as a claim that compact GMM classifier rebuilding is equivalent to official LADA DPT",
                "training-time replay dynamics",
            ],
        )
    )
    dpt_missing.extend(
        f"README: {item}"
        for item in contains_all(
            Path("experiments/README_publication_repro.md"),
            [
                "## Relation to Official LADA DPT",
                "not official LADA DPT reproductions",
                "reproduction of, or an improvement over, the official LADA DPT protocol",
            ],
        )
    )
    gates.append(
        gate(
            "LADA official DPT boundary",
            "PASS" if not dpt_missing else "FAIL",
            "paper and README distinguish classifier-rebuild replay from official DPT"
            if not dpt_missing
            else "missing boundary text: " + "; ".join(dpt_missing),
            "Keep the DPT boundary visible in paper_draft.tex and README_publication_repro.md."
            if not dpt_missing
            else "Restore the DPT boundary paragraph in paper_draft.tex and README_publication_repro.md.",
        )
    )

    status, evidence = status_from_audit(
        ["python", "scripts/audit_training_ablation.py", "--allow_missing"],
        "TRAINING ABLATION AUDIT PASSED",
        "TRAINING ABLATION AUDIT PENDING",
    )
    gates.append(
        gate(
            "Training-side LoRA-NSP ablation",
            status,
            evidence,
            "Run scripts/run_joint_training_ablation.sh on the remote server and postprocess summary.csv.",
        )
    )

    status, evidence = status_from_audit(
        ["python", "scripts/audit_incremental_metrics.py", "--allow_missing"],
        "INCREMENTAL METRICS AUDIT PASSED",
        "INCREMENTAL METRICS AUDIT PENDING",
    )
    gates.append(
        gate(
            "Strict incremental Transfer/Average/Last/forgetting",
            status,
            evidence,
            "Generate incremental_summary.csv and incremental_aggregate.csv from final *_results.json files.",
        )
    )

    required_files = [
        Path("scripts/run_joint_classifier_replay.sh"),
        Path("scripts/run_joint_training_ablation.sh"),
        Path("scripts/run_joint_incremental_ablation.sh"),
        Path("scripts/summarize_joint_classifier_replay.py"),
        Path("scripts/summarize_incremental_metrics.py"),
        Path("scripts/audit_training_ablation.py"),
        Path("scripts/audit_incremental_metrics.py"),
        Path("scripts/audit_paper_claims.py"),
        Path("scripts/audit_protocol_config.py"),
        Path("scripts/audit_result_ledger.py"),
        Path("scripts/postprocess_publication_results.sh"),
        Path("scripts/remote_training_ablation.sh"),
        Path("scripts/selftest_publication_audits.py"),
        Path("scripts/selftest_publication_launchers.py"),
        Path("scripts/selftest_remote_package.py"),
        Path("scripts/verify_publication_package.sh"),
        Path("experiments/README_publication_repro.md"),
        Path("experiments/result_ledger.json"),
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    gates.append(
        gate(
            "Reproducibility scripts and README",
            "PASS" if not missing else "FAIL",
            "all required files present" if not missing else "missing: " + ", ".join(missing),
            "Restore missing reproducibility scripts/files.",
        )
    )

    if args.selftest_treat_pending_as_pass:
        for item in gates:
            if item["status"] == "PENDING":
                item["status"] = "PASS"
                item["evidence"] += "\nSELFTEST OVERRIDE: pending treated as pass"

    status_counts = {}
    for item in gates:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    lines = [
        "| Gate | Status | Next action |",
        "|---|---:|---|",
    ]
    for item in gates:
        lines.append(f"| {item['gate']} | {item['status']} | {item['next_action']} |")
    lines.append("")
    lines.append(
        "Summary: "
        + ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
    )

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "strict": args.strict,
        "gates": gates,
        "summary": status_counts,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))

    if any(item["status"] == "FAIL" for item in gates):
        raise SystemExit(1)
    if args.strict and any(item["status"] == "PENDING" for item in gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
