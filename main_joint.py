"""
联合微调 (Joint Fine-tuning) 入口脚本

功能：
1. 将 id_datasets 拼接成大数据集进行联合训练
2. 训练结束后，在 id_datasets 和 ood_datasets 上同时评估：
   - 零样本分类器 (Zero-shot) 准确率
   - 集成分类器 (Ensemble) 准确率
3. 保存结果 JSON

用法示例：
    python main_joint.py --id_datasets ALL --iterations 5000 --alpha 0.5
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # 屏蔽 transformers 版本提示

import torch
import argparse
import json
import logging
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset

from src.trainers.lora_nsp_trainer import LoRANSPTrainer
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.classifiers.gaussian_statistics import build_multi_center_stats_dict
from src.utils.reference_loader import load_reference_dataset

from src.utils.main_utils import (
    fix_random_seed,
    get_zeroshot_classifier,
    evaluate_dataset,
    batch_evaluate_datasets,
)
from utils_data import get_xtail_trainloader, get_transforms


# X-TAIL 全部 10 个数据集（LADA 论文协议）
ALL_XTAIL_DATASETS = [
    "aircraft", "caltech101", "dtd", "eurosat", "flowers",
    "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Joint Fine-tuning for CLIP Continual Learning")

    # 数据集相关参数
    parser.add_argument("--id_datasets", type=str, nargs='+',
                        default=ALL_XTAIL_DATASETS,
                        help="List of ID datasets for training and ID evaluation. "
                             "Use 'ALL' to select all 10 X-TAIL datasets.")
    parser.add_argument("--ood_datasets", type=str, nargs='*', default=None,
                        help="List of OOD datasets for OOD evaluation. If not specified, "
                             "automatically computed as X-TAIL \\ id_datasets.")
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL",
                        help="Root directory of the dataset.")
    parser.add_argument("--num_shots", type=int, default=16,
                        help="Number of shots for few-shot learning.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and testing.")
    parser.add_argument("--max_num_per_test_dataset", type=int, default=1000,
                        help="Maximum number of samples per test dataset.")

    # 训练基础参数
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    # 设备参数
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (e.g., 0 for cuda:0). Sets CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", type=str,
                        default=None,
                        help="Device to use for training. Overrides --gpu if set.")
    parser.add_argument("--iterations", type=int, default=800,
                        help="Number of training iterations.")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=3e-5,
                        help="Weight decay for optimizer.")

    # LoRA 相关参数
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="Rank for LoRA adaptation.")
    parser.add_argument("--lora_type", type=str, default="lora_vanilla",
                        choices=["lora_vanilla", "lora_sgp", "lora_nsp"],
                        help="Type of LoRA adaptation.")
    parser.add_argument("--nsp_eps", type=float, default=0.05,
                        help="Epsilon parameter for NSP.")
    parser.add_argument("--nsp_weight", type=float, default=0.02,
                        help="Weight parameter for NSP.")
    parser.add_argument("--weight_temp", type=float, default=1.0,
                        help="Temperature parameter for weight.")
    parser.add_argument("--weight_kind", type=str, default="log1p")
    parser.add_argument("--weight_p", type=float, default=1.0,
                        help="P parameter for weight function.")

    # 参考数据集参数（默认不使用蒸馏）
    parser.add_argument("--reference_dataset", type=str, default="flickr8k",
                        help="Reference dataset for distillation. Set to 'flickr8k' to enable.")
    parser.add_argument("--reference_batch_size", type=int, default=32,
                        help="Batch size for reference dataset.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading.")

    # 损失函数权重参数
    parser.add_argument("--fd_weight", type=float, default=1.0,
                        help="Weight for feature distillation loss.")
    parser.add_argument("--cd_weight", type=float, default=1.0,
                        help="Weight for cross-modal distillation loss.")
    parser.add_argument("--aux_weight", type=float, default=0.0,
                        help="Weight for auxiliary linear classifier loss (0=disabled). "
                             "Adds a linear head on features during training to improve "
                             "feature separability for downstream LR-RGDA.")

    # 模型微调开关
    parser.add_argument("--tune_student", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to fine-tune the student model (default: True). "
                             "Set to False to skip training and only evaluate.")

    # Alpha 敏感性分析
    parser.add_argument("--alpha_sensitivity", action='store_true', default=False,
                        help="If set, evaluate ensemble accuracy at multiple alpha values "
                             "(0 to 1.0) to analyze sensitivity.")

    # 分类器参数
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for LR-RGDA classifier in ensemble.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for zero-shot classifier.")

    # LR-RGDA 构建参数
    parser.add_argument("--rgda_rank", type=int, default=32,
                        help="Rank for LR-RGDA low-rank decomposition.")
    parser.add_argument("--rgda_alpha1", type=float, default=0.2,
                        help="qda_reg_alpha1 for LR-RGDA.")
    parser.add_argument("--rgda_alpha2", type=float, default=2.0,
                        help="qda_reg_alpha2 for LR-RGDA.")
    parser.add_argument("--rgda_alpha3", type=float, default=0.5,
                        help="qda_reg_alpha3 for LR-RGDA.")
    parser.add_argument("--num_centers", type=int, default=1,
                        help="Number of centers per class for multi-center LR-RGDA.\n"
                             "1=standard single-center, >1=k-means multi-center.")

    args = parser.parse_args()

    # 支持 'ALL' 简写
    if len(args.id_datasets) == 1 and args.id_datasets[0].upper() == 'ALL':
        args.id_datasets = ALL_XTAIL_DATASETS

    # 解析设备：优先 --device，否则用 --gpu 指定 cuda:N
    if args.device is None:
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # 自动计算 OOD 数据集为 X-TAIL 补集
    if args.ood_datasets is None:
        args.ood_datasets = [d for d in ALL_XTAIL_DATASETS if d not in args.id_datasets]
    logging.info(f"ID  datasets: {args.id_datasets}")
    logging.info(f"OOD datasets: {args.ood_datasets}")

    return args


def main(args):
    if args.seed is not None:
        fix_random_seed(args.seed)

    # ========== 1. 初始化 ==========
    logging.info("\n=== Initializing Joint Fine-tuning ===")
    trainer = LoRANSPTrainer(args)
    model = trainer.model
    processor = trainer.processor

    tune_student = args.tune_student

    # ========== 2. 完整评估函数（中间评估 + 最终评估共用）==========
    def run_full_evaluation(eval_model, tag=""):
        logging.info(f"\n=== Extracting Features for LR-RGDA {tag}===")
        all_feats = []
        all_lbls = []
        feat_offset = 0

        for d_name in args.id_datasets:
            train_transform, _ = get_transforms(d_name)
            tr_loader, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=train_transform, transform_test=None,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            from src.utils.feature_extractor import extract_features
            features, labels = extract_features(eval_model, tr_loader, args.device)
            features = features / features.norm(dim=-1, keepdim=True)
            all_feats.append(features)
            all_lbls.append(labels + feat_offset)
            feat_offset += len(c_names)

        all_features = torch.cat(all_feats)
        all_labels = torch.cat(all_lbls)

        stats_dict, center_means = build_multi_center_stats_dict(
            all_features, all_labels, M=args.num_centers
        )

        lr_rgda_classifier = LRRGDAClassifier(
            stats_dict=stats_dict, device=args.device,
            rank=args.rgda_rank,
            qda_reg_alpha1=args.rgda_alpha1,
            qda_reg_alpha2=args.rgda_alpha2,
            qda_reg_alpha3=args.rgda_alpha3,
            temperature=1.0,
            M=args.num_centers, center_means=center_means,
        )

        num_id_classes = len(all_class_names)
        zs_classifier = get_zeroshot_classifier(eval_model, processor,
                                                 all_class_names, args.device)

        logging.info(f"\n=== Evaluating ID Datasets {tag}===")
        id_zs, id_rgda, id_ens = [], [], []

        id_dataset_offset_map = {}
        eval_offset = 0
        for d_name in args.id_datasets:
            _, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=None, transform_test=None,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            id_dataset_offset_map[d_name] = eval_offset
            eval_offset += len(c_names)

        eval_offset = 0
        for d_name in args.id_datasets:
            zs_acc, rgda_acc, ens_acc, c_len, _ = evaluate_dataset(
                args, d_name, eval_model, zs_classifier, lr_rgda_classifier,
                num_id_classes, eval_offset
            )
            id_zs.append(zs_acc)
            id_rgda.append(rgda_acc)
            id_ens.append(ens_acc)
            eval_offset += c_len
            logging.info(f"[{tag} ID] {d_name:<12s} | ZS: {zs_acc:5.1f}% | "
                         f"RGDA: {rgda_acc:5.1f}% | Ensemble: {ens_acc:5.1f}%")

        id_zs_avg = sum(id_zs) / len(id_zs)
        id_rgda_avg = sum(id_rgda) / len(id_rgda)
        id_ens_avg = sum(id_ens) / len(id_ens)

        # 打印格式化总表（中间评估 + 最终评估共用）
        print(f"\n[{tag} ID Summary]")
        print(f"{'Dataset':<15s} | {'Zero-shot':>9s} | {'LR-RGDA':>9s} | {'Ensemble':>9s}")
        print("-" * 55)
        for d_name, zs, rgda, ens in zip(args.id_datasets, id_zs, id_rgda, id_ens):
            print(f"{d_name:<15s} | {zs:>7.1f}%  | {rgda:>7.1f}%  | {ens:>7.1f}%")
        print("-" * 55)
        print(f"{'Average':<15s} | {id_zs_avg:>7.1f}%  | {id_rgda_avg:>7.1f}%  | {id_ens_avg:>7.1f}%")

        return (id_zs, id_rgda, id_ens, id_zs_avg, id_rgda_avg, id_ens_avg,
                id_dataset_offset_map, num_id_classes, zs_classifier, lr_rgda_classifier)

    all_class_names = []
    if tune_student:
        # 只有在微调时才加载参考数据集
        reference_loader = load_reference_dataset(args, trainer.model_pretrain,
                                                  processor, args.device)

        # 2a. 准备联合训练数据
        logging.info("\n=== Preparing Joint Training Data ===")

        class LocalShiftDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, shift):
                self.base = base_dataset
                self.shift = shift
            def __getitem__(self, idx):
                img, label = self.base[idx]
                return img, label + self.shift
            def __len__(self):
                return len(self.base)

        all_shifted_datasets = []
        current_offset = 0

        for d_name in args.id_datasets:
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            shifted_ds = LocalShiftDataset(tr_loader.dataset, current_offset)
            all_shifted_datasets.append(shifted_ds)
            all_class_names.extend(c_names)
            current_offset += len(c_names)

        merged_dataset = ConcatDataset(all_shifted_datasets)
        merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

        # 2b. 训练模型
        logging.info("\n=== Training (Joint Fine-tuning) ===")
        checkpoint_dir = f"experiments/checkpoints/joint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 预计算零样本分类器和各数据集的标签偏移，用于中间评估
        zs_for_eval = get_zeroshot_classifier(model, processor, all_class_names, args.device)
        dataset_offsets = {}
        offset = 0
        for d_name in args.id_datasets:
            dataset_offsets[d_name] = offset
            _, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=None, transform_test=None,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            offset += len(c_names)

        def eval_callback(current_model, step):
            logging.info(f"\n{'='*60}")
            logging.info(f"Intermediate Evaluation at Iter {step}")
            logging.info(f"{'='*60}")
            run_full_evaluation(current_model, tag=f"Iter {step}")

        model = trainer.train(merged_loader, all_class_names, reference_loader,
                              eval_interval=500, eval_callback=eval_callback,
                              aux_weight=args.aux_weight)

        # 2c. 合并 LoRA 权重（训练结束后只需一次）
        logging.info("\n=== Merging LoRA Weights for Joint Evaluation ===")
        trainer.finalize_task_for_incremental()
        logging.info(f"\nCheckpoints saved to: {checkpoint_dir}")

        # 训练完成后执行完整评估（使用合并后的模型）
        (id_zs_accs, id_rgda_accs, id_ens_accs,
         id_zs_avg, id_rgda_avg, id_ens_avg,
         id_dataset_offset, num_id_classes,
         zeroshot_classifier, lr_rgda_classifier) = run_full_evaluation(model, tag="Final")

    else:
        logging.info("\n=== Skipping Fine-tuning (tune_student=False) ===")
        for d_name in args.id_datasets:
            _, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=None, transform_test=None,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            all_class_names.extend(c_names)

        (id_zs_accs, id_rgda_accs, id_ens_accs,
         id_zs_avg, id_rgda_avg, id_ens_avg,
         id_dataset_offset, num_id_classes,
         zeroshot_classifier, lr_rgda_classifier) = run_full_evaluation(model, tag="No-tune")

    # ========== 4. 评估 OOD 数据集 ==========
    logging.info("\n=== Evaluating OOD Datasets (Ensemble with alpha=%.1f) ===" % args.alpha)
    ood_zs_accs = []
    ood_rgda_accs = []
    ood_ens_accs = []

    for d_name in args.ood_datasets:
        if d_name in id_dataset_offset:
            zs_acc, rgda_acc, ens_acc, _, _ = evaluate_dataset(
                args, d_name, model, zeroshot_classifier, lr_rgda_classifier,
                num_id_classes, id_dataset_offset[d_name]
            )
        else:
            # 构建 ID + OOD 联合零样本分类器，使 ensemble 能正确选择
            _, test_transform = get_transforms(d_name)
            _, te_loader, _, ood_c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=None, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            combined_class_names = all_class_names + ood_c_names
            combined_zeroshot = get_zeroshot_classifier(model, processor, combined_class_names, args.device)
            total_classes = len(combined_class_names)

            zs_acc, rgda_acc, ens_acc, _, _ = evaluate_dataset(
                args, d_name, model, combined_zeroshot, lr_rgda_classifier,
                num_id_classes, num_id_classes
            )

        ood_zs_accs.append(zs_acc)
        ood_rgda_accs.append(rgda_acc)
        ood_ens_accs.append(ens_acc)
        logging.info(f"[OOD] {d_name:<12s} | ZS: {zs_acc:5.1f}% | "
                     f"RGDA: {rgda_acc:5.1f}% | Ensemble: {ens_acc:5.1f}%")

    # ========== 6. 打印指标报告 ==========
    all_zs_accs = id_zs_accs + ood_zs_accs
    all_rgda_accs = id_rgda_accs + ood_rgda_accs
    all_ens_accs = id_ens_accs + ood_ens_accs
    all_dataset_names = args.id_datasets + args.ood_datasets

    id_zs_avg = sum(id_zs_accs) / len(id_zs_accs)
    id_rgda_avg = sum(id_rgda_accs) / len(id_rgda_accs)
    id_ens_avg = sum(id_ens_accs) / len(id_ens_accs)
    if len(ood_zs_accs) > 0:
        ood_zs_avg = sum(ood_zs_accs) / len(ood_zs_accs)
        ood_rgda_avg = sum(ood_rgda_accs) / len(ood_rgda_accs)
        ood_ens_avg = sum(ood_ens_accs) / len(ood_ens_accs)
    else:
        ood_zs_avg = ood_rgda_avg = ood_ens_avg = 0.0
    total_zs_avg = sum(all_zs_accs) / len(all_zs_accs)
    total_rgda_avg = sum(all_rgda_accs) / len(all_rgda_accs)
    total_ens_avg = sum(all_ens_accs) / len(all_ens_accs)

    print("\n" + "=" * 110)
    print("JOINT FINE-TUNING RESULTS")
    print("=" * 110)

    # ID 数据集详细结果
    print("\n[ID Datasets — Fine-tuned]")
    print(f"{'Dataset':<15s} | {'Zero-shot':>9s} | {'LR-RGDA':>9s} | {'Ensemble':>9s}")
    print("-" * 55)
    for d_name, zs, rgda, ens in zip(args.id_datasets,
                                     id_zs_accs, id_rgda_accs, id_ens_accs):
        print(f"{d_name:<15s} | {zs:>7.1f}%  | {rgda:>7.1f}%  | {ens:>7.1f}%")
    print("-" * 55)
    print(f"{'ID Average':<15s} | {id_zs_avg:>7.1f}%  | {id_rgda_avg:>7.1f}%  | {id_ens_avg:>7.1f}%")

    # OOD 数据集详细结果
    if len(args.ood_datasets) > 0:
        print(f"\n[OOD Datasets — Non-fine-tuned]")
        print(f"{'Dataset':<15s} | {'Zero-shot':>9s} | {'LR-RGDA':>9s} | {'Ensemble':>9s}")
        print("-" * 55)
        for d_name, zs, rgda, ens in zip(args.ood_datasets,
                                         ood_zs_accs, ood_rgda_accs, ood_ens_accs):
            print(f"{d_name:<15s} | {zs:>7.1f}%  | {rgda:>7.1f}%  | {ens:>7.1f}%")
        print("-" * 55)
        print(f"{'OOD Average':<15s} | {ood_zs_avg:>7.1f}%  | {ood_rgda_avg:>7.1f}%  | {ood_ens_avg:>7.1f}%")
        print("-" * 55)

    # 全部数据集总平均
    print(f"\n[All Datasets — Total Average]")
    print(f"{'All Average':<15s} | {total_zs_avg:>7.1f}%  | {total_rgda_avg:>7.1f}%  | {total_ens_avg:>7.1f}%")
    print("=" * 110)

    # ========== 7. Alpha 敏感性分析（批处理，如 debug_classifier_router.py）==========
    if args.alpha_sensitivity:
        print("\n" + "=" * 110)
        print("[Alpha 敏感性分析] 所有 ID 数据集的 feature 拼在一起统一扫描")
        print("=" * 110)

        id_result = batch_evaluate_datasets(
            args.id_datasets, model, processor, zeroshot_classifier,
            lr_rgda_classifier, num_id_classes, args=args,
            alpha_sensitivity=True, n_alpha_samples=21
        )

        alphas = [s[0] for s in id_result["sensitivity"]]
        accs = [s[1] for s in id_result["sensitivity"]]

        # 找最佳 alpha
        best_idx = max(range(len(accs)), key=lambda i: accs[i])
        best_alpha, best_acc = alphas[best_idx], accs[best_idx]

        print(f"{'Alpha':>10} | {'Overall Acc (ID合拼)':>20s}")
        print("-" * 40)
        for a, acc in zip(alphas, accs):
            marker = " ← best" if a == best_alpha else ""
            print(f"{a:10.2f} | {acc:18.2f}%{marker}")
        print("-" * 40)
        print(f"最佳 Alpha: {best_alpha:.2f}, 最高准确率: {best_acc:.2f}%")

        # 同时也拼 OOD 做敏感性分析
        if args.ood_datasets:
            ood_result = batch_evaluate_datasets(
                args.ood_datasets, model, processor, zeroshot_classifier,
                lr_rgda_classifier, num_id_classes, args=args,
                alpha_sensitivity=True, n_alpha_samples=21
            )
            ood_accs = [s[1] for s in ood_result["sensitivity"]]
            best_ood_idx = max(range(len(ood_accs)), key=lambda i: ood_accs[i])
            print(f"\n{'Alpha':>10} | {'Overall Acc (OOD合拼)':>20s}")
            print("-" * 40)
            for a, acc in zip(alphas, ood_accs):
                marker = " ← best" if a == alphas[best_ood_idx] else ""
                print(f"{a:10.2f} | {acc:18.2f}%{marker}")
            print("-" * 40)

    # ========== 8. 保存结果 JSON ==========
    save_results = {
        "mode": "Joint Fine-tuning",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "arguments": vars(args),
        "id_datasets": list(args.id_datasets),
        "ood_datasets": list(args.ood_datasets),
        "metrics": {
            "id": {
                "per_dataset": {
                    "zero_shot": {d: v for d, v in zip(args.id_datasets, id_zs_accs)},
                    "lr_rgda": {d: v for d, v in zip(args.id_datasets, id_rgda_accs)},
                    "ours_ensemble": {d: v for d, v in zip(args.id_datasets, id_ens_accs)},
                },
                "average": {
                    "zero_shot": id_zs_avg,
                    "lr_rgda": id_rgda_avg,
                    "ours_ensemble": id_ens_avg,
                }
            },
            "ood": {
                "per_dataset": {
                    "zero_shot": {d: v for d, v in zip(args.ood_datasets, ood_zs_accs)},
                    "lr_rgda": {d: v for d, v in zip(args.ood_datasets, ood_rgda_accs)},
                    "ours_ensemble": {d: v for d, v in zip(args.ood_datasets, ood_ens_accs)},
                },
                "average": {
                    "zero_shot": ood_zs_avg,
                    "lr_rgda": ood_rgda_avg,
                    "ours_ensemble": ood_ens_avg,
                }
            },
            "total": {
                "per_dataset": {
                    "zero_shot": {d: v for d, v in zip(all_dataset_names, all_zs_accs)},
                    "lr_rgda": {d: v for d, v in zip(all_dataset_names, all_rgda_accs)},
                    "ours_ensemble": {d: v for d, v in zip(all_dataset_names, all_ens_accs)},
                },
                "average": {
                    "zero_shot": total_zs_avg,
                    "lr_rgda": total_rgda_avg,
                    "ours_ensemble": total_ens_avg,
                }
            }
        }
    }

    # 如果有敏感性分析，追加保存
    if args.alpha_sensitivity:
        id_result = batch_evaluate_datasets(
            args.id_datasets, model, processor, zeroshot_classifier,
            lr_rgda_classifier, num_id_classes, args=args,
            alpha_sensitivity=True, n_alpha_samples=21
        )
        sens_data = {
            "id": {
                "overall": id_result["overall"],
                "sweep": [(a, acc) for a, acc in id_result["sensitivity"]],
            }
        }
        if args.ood_datasets:
            ood_result = batch_evaluate_datasets(
                args.ood_datasets, model, processor, zeroshot_classifier,
                lr_rgda_classifier, num_id_classes, args=args,
                alpha_sensitivity=True, n_alpha_samples=21
            )
            sens_data["ood"] = {
                "overall": ood_result["overall"],
                "sweep": [(a, acc) for a, acc in ood_result["sensitivity"]],
            }
        save_results["alpha_sensitivity"] = sens_data

    os.makedirs("experiments", exist_ok=True)
    save_path = f"experiments/joint_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(save_path, 'w') as f:
        json.dump(save_results, f, indent=4)
    logging.info(f"\n联合微调结果已保存至: {save_path}")


if __name__ == "__main__":
    command_line_args = parse_args()
    main(command_line_args)
