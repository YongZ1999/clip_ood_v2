"""
持续学习主实验运行脚本
支持 LADA 论文的评估协议
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import sys
import argparse
import json
import torch
import numpy as np
from typing import List, Dict

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.trainers.lora_nsp_trainer import LoRANSPTrainer
from src.utils.reference_loader import load_reference_dataset
from src.utils.continual_metrics import ContinualLearningMetrics
from utils_data import get_xtail_trainloader, get_xtail_testloader, get_transforms


# LADA 论文的 10 个数据集（字母顺序）
LADA_TASK_SEQUENCE = [
    "aircraft",
    "caltech101", 
    "dtd",
    "eurosat",
    "flowers",
    "food101",
    "mnist",
    "oxford_pets",
    "stanford_cars",
    "sun397"
]


def parse_args():
    parser = argparse.ArgumentParser(description="X-TAIL Continual Learning Experiments")
    
    # 数据集配置
    parser.add_argument("--root", type=str, 
                       default="/home/raoxuan/projects/data/X-TAIL/",
                       help="X-TAIL 数据集根目录")
    parser.add_argument("--task_sequence", type=str, nargs='+',
                       default=LADA_TASK_SEQUENCE,
                       help="任务序列（默认使用 LADA 字母顺序）")
    parser.add_argument("--num_shots", type=int, default=16,
                       help="每类别的训练样本数 (16-shot 或 full)")
    parser.add_argument("--full_shot", action="store_true",
                       help="使用 full-shot 设置（覆盖 num_shots）")
    
    # 训练配置
    parser.add_argument("--method", type=str, 
                       choices=["zeroshot", "lora_vanilla", "lora_sgp", "lora_nsp"],
                       default="lora_nsp",
                       help="训练方法: zeroshot, lora_vanilla (基线), lora_sgp, lora_nsp")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-5)
    parser.add_argument("--eval_max_samples", type=int, default=1000,
                       help="Max test samples per dataset; use 0 or negative for the full test split")
    
    # LoRA 参数
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--nsp_eps", type=float, default=0.05)
    parser.add_argument("--nsp_weight", type=float, default=0.02)
    parser.add_argument("--weight_temp", type=float, default=1.0)
    parser.add_argument("--weight_kind", type=str, default="log1p")
    parser.add_argument("--weight_p", type=float, default=1.0)
    parser.add_argument("--tune_vision_encoder", type=lambda x: x.lower() == "true", default=True,
                       help="Whether to attach/train vision-side LoRA modules.")
    parser.add_argument("--tune_text_encoder", type=lambda x: x.lower() == "true", default=True,
                       help="Whether to attach/train text-side LoRA modules.")
    parser.add_argument("--text_lora_rank", type=int, default=4)
    parser.add_argument("--max_zs_classes", type=int, default=128)
    parser.add_argument("--text_tuning_schedule", type=str, default=None,
                       choices=["always", "never", "freeze_after", "low_lr_after"],
                       help="Task-wise text tuning schedule. Defaults to always when tune_text_encoder=true, never otherwise.")
    parser.add_argument("--text_schedule_switch_task", type=int, default=1,
                       help="1-indexed last task that uses the base text LR for freeze_after/low_lr_after schedules.")
    parser.add_argument("--text_lr_scale_after_task", type=float, default=0.2,
                       help="Text LR multiplier after switch_task when text_tuning_schedule=low_lr_after.")
    
    # 蒸馏参数
    parser.add_argument("--fd_weight", type=float, default=1.0)
    parser.add_argument("--cd_weight", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=0.0)
    parser.add_argument("--sce_a", type=float, default=0.5)
    parser.add_argument("--sce_b", type=float, default=0.5)
    parser.add_argument("--reference_dataset", type=str, default="flickr8k")
    parser.add_argument("--reference_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # OOD 检测配置
    parser.add_argument("--enable_ood_eval", action="store_true", default=False,
                       help="启用 OOD 检测评估（目前简化版暂不支持）")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default="experiments/continual_learning")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Optional stable run label used for result file names and summaries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()


def prepare_args_for_method(args):
    """根据方法类型准备 args"""
    normalize_text_schedule_args(args)
    if args.method == "zeroshot":
        # Zero-shot 不需要训练
        pass
    elif args.method == "lora_vanilla":
        # 普通 LoRA 基线（无 SGP/NSP）
        args.lora_type = "lora_vanilla"
        if args.lora_alpha is None:
            args.lora_alpha = args.lora_rank
        args.init_mode = "lora_vanilla"
    elif args.method == "lora_sgp":
        # LoRA + SGP（软投影）
        args.lora_type = "lora_sgp"
        args.init_mode = "lora_sgp"
    elif args.method == "lora_nsp":
        # LoRA + NSP（硬投影）
        args.lora_type = "lora_nsp"
        args.init_mode = "lora_nsp"
    
    return args


def normalize_text_schedule_args(args):
    """Resolve backward-compatible text schedule defaults before model creation."""
    if args.text_tuning_schedule is None:
        args.text_tuning_schedule = "always" if args.tune_text_encoder else "never"

    if args.text_schedule_switch_task < 1:
        raise ValueError("--text_schedule_switch_task must be >= 1")
    if args.text_lr_scale_after_task < 0:
        raise ValueError("--text_lr_scale_after_task must be >= 0")

    # Text LoRA must be attached from the start for freeze_after/low_lr_after so
    # task-1 text adaptation can be merged and preserved for later tasks.
    if args.text_tuning_schedule == "never":
        args.tune_text_encoder = False
    else:
        args.tune_text_encoder = True

    return args


def text_schedule_for_task(args, task_index_zero_based: int):
    """Return (train_text, text_lr, scale) for the current 0-based task index."""
    task_number = task_index_zero_based + 1
    schedule = args.text_tuning_schedule
    base_lr = args.lr

    if schedule == "never":
        return False, 0.0, 0.0
    if schedule == "always":
        return True, base_lr, 1.0
    if task_number <= args.text_schedule_switch_task:
        return True, base_lr, 1.0
    if schedule == "freeze_after":
        return False, 0.0, 0.0
    if schedule == "low_lr_after":
        scale = args.text_lr_scale_after_task
        return scale > 0, base_lr * scale, scale
    raise ValueError(f"Unsupported text_tuning_schedule: {schedule}")


def normalize_accuracy(accuracy: float) -> float:
    """Store all incremental accuracies as fractions in [0, 1]."""
    return accuracy / 100.0 if accuracy > 1.0 else accuracy


def evaluate_all_tasks(trainer, task_sequence: List[str], root: str,
                       batch_size: int, max_samples: int = 1000) -> Dict:
    """在所有任务上评估模型"""
    results = {'accuracies': {}}
    max_num_per_dataset = None if max_samples is None or max_samples <= 0 else max_samples
    
    _, test_transform = get_transforms("aircraft")
    
    for task_name in task_sequence:
        test_loader, class_names, _ = get_xtail_testloader(
            root=root,
            dataset_sequence=[task_name],
            transform_test=test_transform,
            batch_size=batch_size,
            max_num_per_dataset=max_num_per_dataset
        )
        
        accuracy = normalize_accuracy(trainer.evaluate(test_loader, class_names))
        results['accuracies'][task_name] = accuracy
        print(f"  {task_name}: Acc = {accuracy*100:.1f}%")
    
    return results


def run_continual_learning(args):
    """运行持续学习实验"""
    print("="*80)
    print("X-TAIL Continual Learning Experiment")
    print("="*80)
    print(f"Method: {args.method}")
    print(f"Task Sequence: {args.task_sequence}")
    print(f"Num Shots: {'full' if args.full_shot else args.num_shots}")
    print(f"Eval max samples per dataset: {'full' if args.eval_max_samples <= 0 else args.eval_max_samples}")
    print(f"Text tuning schedule: {args.text_tuning_schedule or ('always' if args.tune_text_encoder else 'never')}")
    print("="*80)
    
    # 准备参数
    args = prepare_args_for_method(args)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化评估指标记录器
    metrics_tracker = ContinualLearningMetrics(args.task_sequence)
    
    # 初始化训练器。Use LoRANSPTrainer for every LoRA variant so NSP runs the
    # same task-finalization path used by the main incremental code.
    trainer = LoRANSPTrainer(args)
    reference_loader = None
    if args.fd_weight > 0 or args.cd_weight > 0:
        reference_loader = load_reference_dataset(
            args, trainer.model_pretrain, trainer.processor, args.device)
    
    # Step 0: 评估 Zero-shot 性能
    print("\n[Step 0] Evaluating Zero-shot Performance...")
    zeroshot_results = evaluate_all_tasks(
        trainer, args.task_sequence, args.root, args.batch_size, args.eval_max_samples
    )
    
    zeroshot_accs = {task: zeroshot_results['accuracies'].get(task, 0) 
                     for task in args.task_sequence}
    metrics_tracker.update(0, zeroshot_accs)
    
    print("\nZero-shot Accuracies:")
    for task, acc in zeroshot_accs.items():
        print(f"  {task}: {acc*100:.1f}%")
    
    # 如果只需要 zero-shot，直接结束
    if args.method == "zeroshot":
        print("\nZero-shot evaluation completed.")
        metrics_tracker.print_summary()
        metrics_tracker.save(os.path.join(args.output_dir, "zeroshot_results.json"))
        return
    
    # 增量学习循环
    per_task_text_schedule = []
    for step, task_name in enumerate(args.task_sequence):
        print(f"\n{'='*80}")
        print(f"[Step {step+1}/{len(args.task_sequence)}] Training on: {task_name}")
        print(f"{'='*80}")

        train_text_this_task, text_lr_this_task, text_lr_scale = text_schedule_for_task(args, step)
        per_task_text_schedule.append({
            "task": task_name,
            "task_index": step + 1,
            "tune_text_encoder": bool(train_text_this_task),
            "text_lr": float(text_lr_this_task),
            "text_lr_scale": float(text_lr_scale),
        })
        print(
            "Text schedule for this task: "
            f"tune_text_encoder={train_text_this_task}, "
            f"text_lr={text_lr_this_task:.6g}, "
            f"scale={text_lr_scale:.6g}"
        )
        
        # 获取当前任务的数据加载器
        train_transform, test_transform = get_transforms(task_name)
        num_shots = None if args.full_shot else args.num_shots
        
        train_loader, update_loader, _, class_names = get_xtail_trainloader(
            root=args.root,
            dataset_name=task_name,
            transform_train=train_transform,
            transform_test=test_transform,
            num_shots=num_shots,
            batch_size=args.batch_size
        )
        
        # 训练当前任务
        print(f"\nTraining {task_name}...")
        
        trainer.train(train_loader, class_names, reference_loader,
                      aux_weight=args.aux_weight,
                      train_text_encoder=train_text_this_task,
                      text_lr=text_lr_this_task)
        
        # 增量学习关键步骤：将 LoRA 参数归并到主模型，并在 NSP 模式下
        # 用当前任务的确定性 update loader 更新下一任务的投影矩阵。
        print(f"\nFinalizing task adapters...")
        if getattr(args, "lora_type", "") == "lora_nsp":
            if trainer.has_vision_lora:
                covariances = trainer.extract_layer_covariances(update_loader)
            if trainer.has_text_lora and train_text_this_task:
                text_covariances = trainer.extract_text_covariances(class_names)
            trainer.finalize_task_for_incremental()
            if trainer.has_vision_lora:
                trainer.update_covariance_history(covariances)
            if trainer.has_text_lora and train_text_this_task:
                trainer.update_text_covariance_history(text_covariances)
        else:
            trainer.finalize_task_for_incremental()
        
        # 在所有任务上评估
        print(f"\nEvaluating on all {len(args.task_sequence)} tasks...")
        eval_results = evaluate_all_tasks(
            trainer, args.task_sequence, args.root, args.batch_size, args.eval_max_samples
        )
        
        # 更新指标记录器
        metrics_tracker.update(step, eval_results['accuracies'])
        
        # 保存中间结果
        intermediate_results = {
            'step': step + 1,
            'task': task_name,
            'accuracies': {k: float(v) for k, v in eval_results['accuracies'].items()},
            'text_schedule': per_task_text_schedule[-1],
        }
        
        with open(os.path.join(args.output_dir, f"step_{step+1}_{task_name}.json"), 'w') as f:
            json.dump(intermediate_results, f, indent=2)
    
    # 打印最终总结
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    metrics_tracker.print_summary()
    
    # 保存最终结果
    final_results = {
        'args': {k: str(v) if not isinstance(v, (int, float, bool, list, dict)) else v 
                 for k, v in vars(args).items()},
        'text_schedule': per_task_text_schedule,
        'metrics': metrics_tracker.get_summary(),
        'per_task_metrics': metrics_tracker.calculate_per_task_metrics(),
        'accuracy_matrix': metrics_tracker.get_accuracy_matrix().tolist()
    }
    
    result_stem = args.experiment_name or args.method
    output_file = os.path.join(args.output_dir, f"{result_stem}_results.json")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nAll results saved to: {args.output_dir}")
    
    return final_results


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 运行实验
    results = run_continual_learning(args)
    
    return results


if __name__ == "__main__":
    main()
