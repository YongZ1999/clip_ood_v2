#!/usr/bin/env python3
import subprocess
import tempfile
from pathlib import Path


EXPECTED_TASKS = "aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397"


def run(command, env=None):
    proc = subprocess.run(command, text=True, capture_output=True, env=env)
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        raise SystemExit(
            "command failed unexpectedly:\n"
            + " ".join(command)
            + "\n"
            + output
        )
    return output


def assert_contains(path, needles):
    text = path.read_text()
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(
            f"{path} missing expected launcher arguments:\n"
            + "\n".join(f"- {needle}" for needle in missing)
            + "\n\ncontent:\n"
            + text
        )


def assert_exists(paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("missing expected launcher dry-run logs:\n" + "\n".join(f"- {path}" for path in missing))


def shell_env(overrides):
    import os

    env = os.environ.copy()
    env.update(overrides)
    return env


def check_classifier_launcher(tmp):
    out_dir = tmp / "classifier"
    run(
        ["bash", "scripts/run_joint_classifier_replay.sh"],
        env=shell_env(
            {
                "DRY_RUN": "1",
                "OUT_DIR": str(out_dir),
                "SEEDS": "42",
                "CLASSIFIER_FEATURE_TRANSFORMS": "test",
                "REPLAY_MODES": "real gmm_raw gmm_sphere gmm_raw_mean",
                "RUN_MEAN_ABLATION": "1",
            }
        ),
    )
    assert_exists(
        [
            out_dir / "logs" / "test_real_seed42.log",
            out_dir / "logs" / "test_gmm_raw_seed42.log",
            out_dir / "logs" / "test_gmm_sphere_seed42.log",
            out_dir / "logs" / "test_gmm_raw_mean_seed42.log",
        ]
    )
    assert_contains(
        out_dir / "manifest.txt",
        [
            "classifier_feature_transforms: test",
            "seeds: 42",
            "replay_modes: real gmm_raw gmm_sphere gmm_raw_mean",
            "run_mean_ablation: 1",
            "--num_shots 16",
            "--enable_lada",
            "--lada_k 16",
            "--num_centers 4",
            "--gmm_k 4",
            "--gaussian_samples_per_class 16",
        ],
    )
    assert_contains(
        out_dir / "logs" / "test_real_seed42.log",
        [
            "--id_datasets ALL",
            "--num_shots 16",
            "--enable_lada",
            "--lada_k 16",
            "--num_centers 4",
            "--classifier_feature_transform test",
            "--experiment_name test_real",
        ],
    )
    assert_contains(
        out_dir / "logs" / "test_gmm_raw_mean_seed42.log",
        [
            "--use_gaussian_features",
            "--gmm_k 4",
            "--gaussian_samples_per_class 16",
            "--gmm_fit_space raw",
            "--gmm_sample_mode mean",
        ],
    )
    assert_contains(
        out_dir / "logs" / "test_gmm_sphere_seed42.log",
        [
            "--use_gaussian_features",
            "--gmm_fit_space sphere",
            "--gmm_sample_mode sample",
        ],
    )


def check_training_launcher(tmp):
    out_dir = tmp / "training"
    run(
        ["bash", "scripts/run_joint_training_ablation.sh"],
        env=shell_env(
            {
                "DRY_RUN": "1",
                "OUT_DIR": str(out_dir),
                "SEEDS": "42",
                "GPUS": "0 1 2 3",
            }
        ),
    )
    expected_configs = {
        "lora_vanilla": ("lora_vanilla", "0.0", "0.0"),
        "lora_nsp": ("lora_nsp", "0.0", "0.0"),
        "lora_nsp_fd": ("lora_nsp", "1.0", "0.0"),
        "lora_nsp_fd_cd": ("lora_nsp", "1.0", "1.0"),
    }
    assert_exists(
        [out_dir / "logs" / f"{name}_seed42.log" for name in expected_configs]
    )
    assert_contains(
        out_dir / "manifest.txt",
        [
            "lora_vanilla lora_type=lora_vanilla fd=0.0 cd=0.0",
            "seeds: 42",
            "lora_nsp lora_type=lora_nsp fd=0.0 cd=0.0",
            "lora_nsp_fd lora_type=lora_nsp fd=1.0 cd=0.0",
            "lora_nsp_fd_cd lora_type=lora_nsp fd=1.0 cd=1.0",
            "--classifier_feature_transform test",
            "--enable_lada",
            "--lada_k 16",
            "--num_centers 4",
        ],
    )
    for name, (lora_type, fd_weight, cd_weight) in expected_configs.items():
        assert_contains(
            out_dir / "logs" / f"{name}_seed42.log",
            [
                f"--lora_type {lora_type}",
                f"--fd_weight {fd_weight}",
                f"--cd_weight {cd_weight}",
                f"--experiment_name {name}",
            ],
        )
    assert_contains(
        out_dir / "logs" / "lora_nsp_fd_cd_seed42.log",
        [
            "--id_datasets ALL",
            "--num_shots 16",
            "--tune_vision_encoder true",
            "--enable_lada",
            "--lada_k 16",
            "--num_centers 4",
            "--classifier_feature_transform test",
            "--lora_type lora_nsp",
            "--fd_weight 1.0",
            "--cd_weight 1.0",
            "--experiment_name lora_nsp_fd_cd",
        ],
    )


def check_incremental_launcher(tmp):
    out_dir = tmp / "incremental"
    run(
        ["bash", "scripts/run_joint_incremental_ablation.sh"],
        env=shell_env(
            {
                "DRY_RUN": "1",
                "OUT_DIR": str(out_dir),
                "SEEDS": "42",
                "GPUS": "0 1 2 3",
                "EXPECTED_TASKS": EXPECTED_TASKS,
            }
        ),
    )
    expected_configs = {
        "lora_vanilla": ("lora_vanilla", "0.0", "0.0"),
        "lora_nsp": ("lora_nsp", "0.0", "0.0"),
        "lora_nsp_fd": ("lora_nsp", "1.0", "0.0"),
        "lora_nsp_fd_cd": ("lora_nsp", "1.0", "1.0"),
    }
    assert_exists(
        [out_dir / "logs" / f"{name}_seed42.log" for name in expected_configs]
    )
    assert_contains(
        out_dir / "manifest.txt",
        [
            f"expected_tasks: {EXPECTED_TASKS}",
            "seeds: 42",
            "lora_vanilla method=lora_vanilla fd=0.0 cd=0.0",
            "lora_nsp method=lora_nsp fd=0.0 cd=0.0",
            "lora_nsp_fd method=lora_nsp fd=1.0 cd=0.0",
            "lora_nsp_fd_cd method=lora_nsp fd=1.0 cd=1.0",
            f"--task_sequence {EXPECTED_TASKS}",
            "--eval_max_samples 0",
            "--tune_vision_encoder true",
            "--tune_text_encoder true",
        ],
    )
    for name, (method, fd_weight, cd_weight) in expected_configs.items():
        assert_contains(
            out_dir / "logs" / f"{name}_seed42.log",
            [
                f"--method {method}",
                f"--fd_weight {fd_weight}",
                f"--cd_weight {cd_weight}",
                f"--experiment_name {name}",
            ],
        )
    assert_contains(
        out_dir / "logs" / "lora_nsp_fd_cd_seed42.log",
        [
            f"--task_sequence {EXPECTED_TASKS}",
            "--num_shots 16",
            "--eval_max_samples 0",
            "--tune_vision_encoder true",
            "--tune_text_encoder true",
            "--method lora_nsp",
            "--fd_weight 1.0",
            "--cd_weight 1.0",
            "--experiment_name lora_nsp_fd_cd",
        ],
    )


def check_incremental_pilot_launcher(tmp):
    out_dir = tmp / "incremental_pilot"
    run(
        ["bash", "scripts/run_incremental_training_pilot.sh"],
        env=shell_env(
            {
                "DRY_RUN": "1",
                "OUT_DIR": str(out_dir),
                "SEEDS": "42",
                "GPUS": "0 1 2 3 4",
            }
        ),
    )
    expected_configs = {
        "lora_vanilla_vision": ("lora_vanilla", "false", "0.0", "0.0"),
        "lora_nsp_vision": ("lora_nsp", "false", "0.0", "0.0"),
        "lora_vanilla_vision_text": ("lora_vanilla", "true", "0.0", "0.0"),
        "lora_nsp_vision_text": ("lora_nsp", "true", "0.0", "0.0"),
        "lora_nsp_fd_cd_vision_text": ("lora_nsp", "true", "1.0", "1.0"),
    }
    assert_exists(
        [out_dir / "logs" / f"{name}_seed42.log" for name in expected_configs]
    )
    assert_contains(
        out_dir / "manifest.txt",
        [
            "expected_tasks: aircraft caltech101",
            "seeds: 42",
            "lora_vanilla_vision method=lora_vanilla fd=0.0 cd=0.0 tune_vision=true tune_text=false",
            "lora_nsp_vision method=lora_nsp fd=0.0 cd=0.0 tune_vision=true tune_text=false",
            "lora_vanilla_vision_text method=lora_vanilla fd=0.0 cd=0.0 tune_vision=true tune_text=true",
            "lora_nsp_vision_text method=lora_nsp fd=0.0 cd=0.0 tune_vision=true tune_text=true",
            "lora_nsp_fd_cd_vision_text method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=true",
            "--task_sequence aircraft caltech101",
            "--eval_max_samples 0",
        ],
    )
    for name, (method, tune_text, fd_weight, cd_weight) in expected_configs.items():
        assert_contains(
            out_dir / "logs" / f"{name}_seed42.log",
            [
                "--tune_vision_encoder true",
                f"--tune_text_encoder {tune_text}",
                f"--method {method}",
                f"--fd_weight {fd_weight}",
                f"--cd_weight {cd_weight}",
                f"--experiment_name {name}",
            ],
        )


def check_text_schedule_pilot_launcher(tmp):
    out_dir = tmp / "text_schedule_pilot"
    run(
        ["bash", "scripts/run_incremental_text_schedule_pilot.sh"],
        env=shell_env(
            {
                "DRY_RUN": "1",
                "OUT_DIR": str(out_dir),
                "SEEDS": "42",
                "GPUS": "0 1 2 3 4",
            }
        ),
    )
    expected_configs = {
        "text_always": ("always", "true", "1", "1.0"),
        "text_task1_freeze": ("freeze_after", "true", "1", "0.0"),
        "text_task1_lr_1_5": ("low_lr_after", "true", "1", "0.2"),
        "text_task1_lr_1_10": ("low_lr_after", "true", "1", "0.1"),
        "vision_only_fd_cd": ("never", "false", "1", "0.0"),
    }
    assert_exists(
        [out_dir / "logs" / f"{name}_seed42.log" for name in expected_configs]
    )
    assert_contains(
        out_dir / "manifest.txt",
        [
            "expected_tasks: aircraft caltech101",
            "seeds: 42",
            "text_always method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=true text_schedule=always",
            "text_task1_freeze method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=true text_schedule=freeze_after",
            "text_task1_lr_1_5 method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=true text_schedule=low_lr_after switch_task=1 text_lr_scale=0.2",
            "text_task1_lr_1_10 method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=true text_schedule=low_lr_after switch_task=1 text_lr_scale=0.1",
            "vision_only_fd_cd method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=false text_schedule=never",
            "--task_sequence aircraft caltech101",
            "--eval_max_samples 0",
            "--method lora_nsp",
            "--fd_weight 1.0",
            "--cd_weight 1.0",
            "--tune_vision_encoder true",
        ],
    )
    for name, (schedule, tune_text, switch_task, scale) in expected_configs.items():
        assert_contains(
            out_dir / "logs" / f"{name}_seed42.log",
            [
                "--method lora_nsp",
                "--fd_weight 1.0",
                "--cd_weight 1.0",
                "--tune_vision_encoder true",
                f"--tune_text_encoder {tune_text}",
                f"--text_tuning_schedule {schedule}",
                f"--text_schedule_switch_task {switch_task}",
                f"--text_lr_scale_after_task {scale}",
                f"--experiment_name {name}",
            ],
        )


def check_text_schedule_stress_launcher(tmp):
    out_dir = tmp / "text_schedule_stress"
    run(
        ["bash", "scripts/run_incremental_text_schedule_stress.sh"],
        env=shell_env(
            {
                "DRY_RUN": "1",
                "OUT_DIR": str(out_dir),
                "SEEDS": "42",
                "GPUS": "0 1 2",
            }
        ),
    )
    expected_configs = {
        "text_always": ("always", "true", "1", "1.0"),
        "text_task1_freeze": ("freeze_after", "true", "1", "0.0"),
        "text_task1_lr_1_5": ("low_lr_after", "true", "1", "0.2"),
    }
    assert_exists(
        [out_dir / "logs" / f"{name}_seed42.log" for name in expected_configs]
    )
    assert_contains(
        out_dir / "manifest.txt",
        [
            "expected_tasks: aircraft caltech101 dtd eurosat",
            "seeds: 42",
            "text_always method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=true text_schedule=always",
            "text_task1_freeze method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=true text_schedule=freeze_after",
            "text_task1_lr_1_5 method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=true text_schedule=low_lr_after switch_task=1 text_lr_scale=0.2",
            "--task_sequence aircraft caltech101 dtd eurosat",
            "--eval_max_samples 0",
            "--method lora_nsp",
            "--fd_weight 1.0",
            "--cd_weight 1.0",
            "--tune_vision_encoder true",
        ],
    )
    for name, (schedule, tune_text, switch_task, scale) in expected_configs.items():
        assert_contains(
            out_dir / "logs" / f"{name}_seed42.log",
            [
                "--method lora_nsp",
                "--fd_weight 1.0",
                "--cd_weight 1.0",
                "--tune_vision_encoder true",
                f"--tune_text_encoder {tune_text}",
                f"--text_tuning_schedule {schedule}",
                f"--text_schedule_switch_task {switch_task}",
                f"--text_lr_scale_after_task {scale}",
                f"--experiment_name {name}",
            ],
        )


def check_text_schedule_full_gate_launcher(tmp):
    out_dir = tmp / "text_schedule_full_gate"
    run(
        ["bash", "scripts/run_incremental_text_schedule_full_gate.sh"],
        env=shell_env(
            {
                "DRY_RUN": "1",
                "OUT_DIR": str(out_dir),
                "SEEDS": "42 43 44",
                "GPUS": "0 1",
            }
        ),
    )
    expected_configs = {
        "text_task1_lr_1_5": ("low_lr_after", "true", "1", "0.2"),
        "text_always": ("always", "true", "1", "1.0"),
    }
    expected_tasks = (
        "aircraft caltech101 dtd eurosat flowers food101 mnist "
        "oxford_pets stanford_cars sun397"
    )
    assert_exists(
        [
            out_dir / "logs" / f"{name}_seed{seed}.log"
            for name in expected_configs
            for seed in (42, 43, 44)
        ]
    )
    assert_contains(
        out_dir / "manifest.txt",
        [
            f"expected_tasks: {expected_tasks}",
            "seeds: 42 43 44",
            "text_task1_lr_1_5 method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=true text_schedule=low_lr_after switch_task=1 text_lr_scale=0.2",
            "text_always method=lora_nsp fd=1.0 cd=1.0 tune_vision=true tune_text=true text_schedule=always switch_task=1 text_lr_scale=1.0",
            f"--task_sequence {expected_tasks}",
            "--eval_max_samples 0",
            "--method lora_nsp",
            "--fd_weight 1.0",
            "--cd_weight 1.0",
            "--tune_vision_encoder true",
        ],
    )
    for name, (schedule, tune_text, switch_task, scale) in expected_configs.items():
        assert_contains(
            out_dir / "logs" / f"{name}_seed42.log",
            [
                "--method lora_nsp",
                "--fd_weight 1.0",
                "--cd_weight 1.0",
                "--tune_vision_encoder true",
                f"--tune_text_encoder {tune_text}",
                f"--text_tuning_schedule {schedule}",
                f"--text_schedule_switch_task {switch_task}",
                f"--text_lr_scale_after_task {scale}",
                f"--experiment_name {name}",
            ],
        )


def main():
    with tempfile.TemporaryDirectory(prefix="publication_launcher_selftest_") as raw_tmp:
        tmp = Path(raw_tmp)
        check_classifier_launcher(tmp)
        check_training_launcher(tmp)
        check_incremental_launcher(tmp)
        check_incremental_pilot_launcher(tmp)
        check_text_schedule_pilot_launcher(tmp)
        check_text_schedule_stress_launcher(tmp)
        check_text_schedule_full_gate_launcher(tmp)
    print("PUBLICATION LAUNCHER SELFTEST PASSED")


if __name__ == "__main__":
    main()
