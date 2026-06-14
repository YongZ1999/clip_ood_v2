#!/usr/bin/env python3
import re
from pathlib import Path


REMOTE_HELPER = Path("scripts/remote_training_ablation.sh")
VERIFY_SCRIPT = Path("scripts/verify_publication_package.sh")
GATE_AUDIT = Path("scripts/audit_publication_gates.py")
POSTPROCESS_SCRIPT = Path("scripts/postprocess_publication_results.sh")

REQUIRED_UPLOADS = {
    "main_joint.py",
    "scripts/remote_training_ablation.sh",
    "scripts/run_joint_classifier_replay.sh",
    "scripts/run_joint_training_ablation.sh",
    "scripts/run_joint_incremental_ablation.sh",
    "scripts/run_incremental_training_pilot.sh",
    "scripts/run_incremental_text_schedule_pilot.sh",
    "scripts/run_incremental_text_schedule_stress.sh",
    "scripts/run_incremental_text_schedule_full_gate.sh",
    "scripts/run_lada_dpt_ablation.sh",
    "scripts/summarize_joint_classifier_replay.py",
    "scripts/summarize_incremental_metrics.py",
    "scripts/audit_training_ablation.py",
    "scripts/audit_incremental_metrics.py",
    "scripts/audit_paper_claims.py",
    "scripts/audit_protocol_config.py",
    "scripts/audit_publication_gates.py",
    "scripts/audit_result_ledger.py",
    "scripts/postprocess_publication_results.sh",
    "scripts/selftest_publication_audits.py",
    "scripts/selftest_publication_launchers.py",
    "scripts/selftest_remote_package.py",
    "scripts/verify_publication_package.sh",
    "src/experiments/run_continual_learning.py",
    "src/classifiers/gaussian_statistics.py",
    "src/lada/lada_classifier.py",
    "src/models/clip.py",
    "src/models/trainer.py",
    "src/trainers/lora_nsp_trainer.py",
    "paper_writing/paper-template/paper_draft.tex",
    "experiments/README_publication_repro.md",
    "experiments/result_ledger.json",
}


def parse_files_array(path):
    text = path.read_text()
    match = re.search(r"^FILES=\(\n(?P<body>.*?)^\)", text, re.MULTILINE | re.DOTALL)
    if not match:
        raise SystemExit(f"could not find FILES array in {path}")
    files = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        files.append(line.strip("'\""))
    return set(files)


def main():
    uploads = parse_files_array(REMOTE_HELPER)
    helper_text = REMOTE_HELPER.read_text()
    verify_text = VERIFY_SCRIPT.read_text()
    gate_text = GATE_AUDIT.read_text()
    postprocess_text = POSTPROCESS_SCRIPT.read_text()
    missing_from_upload = sorted(REQUIRED_UPLOADS - uploads)
    missing_on_disk = sorted(path for path in REQUIRED_UPLOADS if not Path(path).exists())
    failures = []
    if missing_from_upload:
        failures.append(
            "missing from remote upload FILES:\n"
            + "\n".join(f"- {path}" for path in missing_from_upload)
        )
    if missing_on_disk:
        failures.append(
            "required upload files missing on disk:\n"
            + "\n".join(f"- {path}" for path in missing_on_disk)
        )
    required_helper_snippets = (
        "launch_all|status_all",
        "launch_text_schedule_pilot",
        "status_text_schedule_pilot",
        "launch_text_schedule_stress",
        "status_text_schedule_stress",
        "launch_text_schedule_full_gate",
        "status_text_schedule_full_gate",
        "status_gates)",
        "status_gates()",
        "launch_all()",
        "status_all()",
        "experiments/publication_gate_status.json",
        "generated_at_utc:",
        "OUT_DIR='${OUT_DIR}' nohup scripts/run_joint_training_ablation.sh",
        "OUT_DIR='${INCREMENTAL_DIR}' nohup scripts/run_joint_incremental_ablation.sh",
        "OUT_DIR='${PILOT_DIR}' nohup scripts/run_incremental_training_pilot.sh",
        "OUT_DIR='${TEXT_SCHEDULE_DIR}' nohup scripts/run_incremental_text_schedule_pilot.sh",
        "OUT_DIR='${TEXT_STRESS_DIR}' nohup scripts/run_incremental_text_schedule_stress.sh",
        "OUT_DIR='${TEXT_FULL_GATE_DIR}' nohup scripts/run_incremental_text_schedule_full_gate.sh",
        "TRAINING_DIR='${OUT_DIR}' INCREMENTAL_DIR='${INCREMENTAL_DIR}' bash scripts/postprocess_publication_results.sh",
    )
    missing_snippets = [
        snippet for snippet in required_helper_snippets if snippet not in helper_text
    ]
    if missing_snippets:
        failures.append(
            "remote helper missing required environment forwarding:\n"
            + "\n".join(f"- {snippet}" for snippet in missing_snippets)
        )
    required_verifier_snippets = (
        "paper_writing/paper-template/paper_draft.tex",
        "experiments/README_publication_repro.md",
        "python scripts/audit_paper_claims.py",
        "STRICT_GATES",
        "--strict",
    )
    missing_verifier_snippets = [
        snippet for snippet in required_verifier_snippets if snippet not in verify_text
    ]
    if missing_verifier_snippets:
        failures.append(
            "verifier missing paper/README claim audit coverage:\n"
            + "\n".join(f"- {snippet}" for snippet in missing_verifier_snippets)
        )
    required_gate_snippets = (
        "paper_writing/paper-template/paper_draft.tex",
        "experiments/README_publication_repro.md",
        "scripts/audit_paper_claims.py",
    )
    missing_gate_snippets = [
        snippet for snippet in required_gate_snippets if snippet not in gate_text
    ]
    if missing_gate_snippets:
        failures.append(
            "publication gate audit missing paper/README claim coverage:\n"
            + "\n".join(f"- {snippet}" for snippet in missing_gate_snippets)
        )
    required_postprocess_snippets = (
        "STRICT_GATES=1 bash scripts/verify_publication_package.sh",
        "python scripts/audit_publication_gates.py",
        "--strict",
        "--output_json experiments/publication_gate_status.json",
        "--output_markdown experiments/publication_gate_status.md",
    )
    missing_postprocess_snippets = [
        snippet for snippet in required_postprocess_snippets if snippet not in postprocess_text
    ]
    if missing_postprocess_snippets:
        failures.append(
            "postprocess script missing final verification gates:\n"
            + "\n".join(f"- {snippet}" for snippet in missing_postprocess_snippets)
        )
    if failures:
        raise SystemExit("\n\n".join(failures))
    print("REMOTE PACKAGE SELFTEST PASSED")


if __name__ == "__main__":
    main()
