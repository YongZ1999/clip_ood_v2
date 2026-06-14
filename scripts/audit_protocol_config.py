#!/usr/bin/env python3
from pathlib import Path


CHECKS = (
    (
        "scripts/run_joint_classifier_replay.sh",
        (
            'CLASSIFIER_FEATURE_TRANSFORMS="${CLASSIFIER_FEATURE_TRANSFORMS:-test}"',
            'SEEDS="${SEEDS:-42 43 44}"',
            "--enable_lada",
            "--lada_k 16",
            "--num_centers 4",
            "--gmm_k 4",
            "--gaussian_samples_per_class 16",
            "--expected_seeds",
            "CLIP_USE_SAFETENSORS",
            "CLIP_LOCAL_FILES_ONLY",
        ),
    ),
    (
        "scripts/run_joint_training_ablation.sh",
        (
            "--classifier_feature_transform test",
            'SEEDS="${SEEDS:-42 43 44}"',
            "--enable_lada",
            "--lada_k 16",
            "--num_centers 4",
            "--tune_vision_encoder true",
            'CONFIG_NAMES=(lora_vanilla lora_nsp lora_nsp_fd lora_nsp_fd_cd)',
            "--expected_seeds",
            "CLIP_USE_SAFETENSORS",
            "CLIP_LOCAL_FILES_ONLY",
        ),
    ),
    (
        "scripts/run_joint_incremental_ablation.sh",
        (
            "EXPECTED_TASKS",
            'SEEDS="${SEEDS:-42 43 44}"',
            "--task_sequence ${EXPECTED_TASKS}",
            "--num_shots 16",
            "--eval_max_samples 0",
            "--tune_vision_encoder",
            "--tune_text_encoder",
            'CONFIG_NAMES=(lora_vanilla lora_nsp lora_nsp_fd lora_nsp_fd_cd)',
            'CONFIG_METHODS=(lora_vanilla lora_nsp lora_nsp lora_nsp)',
            "--experiment_name",
            "lora_nsp_fd_cd",
            "summarize_incremental_metrics.py",
            "CLIP_USE_SAFETENSORS",
            "CLIP_LOCAL_FILES_ONLY",
        ),
    ),
    (
        "experiments/README_publication_repro.md",
        (
            "Use `--classifier_feature_transform test` for the main protocol.",
            "Build LR-RGDA and LADA from the same feature source.",
            "Do not mix old `test_gmm_raw_mean_seed42.json` with fixed mean replay results.",
            "CLASSIFIER_FEATURE_TRANSFORMS=\"test\"",
            "lora_vanilla",
            "lora_nsp_fd_cd",
            "run_joint_incremental_ablation.sh",
            "launch_incremental",
            "--eval_max_samples 0",
            "not official LADA DPT reproductions",
        ),
    ),
)


def main():
    failures = []
    for path_text, needles in CHECKS:
        path = Path(path_text)
        if not path.exists():
            failures.append(f"missing file: {path}")
            continue
        text = path.read_text()
        for needle in needles:
            if needle not in text:
                failures.append(f"{path}: missing `{needle}`")

    print("audited protocol files:")
    for path_text, _ in CHECKS:
        print(f"- {path_text}")

    if failures:
        print("PROTOCOL CONFIG AUDIT FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("PROTOCOL CONFIG AUDIT PASSED")


if __name__ == "__main__":
    main()
