"""
增量微调 (Incremental Fine-tuning) 入口脚本

功能：
1. 按 dataset_sequence 逐任务增量训练
2. 每个任务进行 LoRA-NSP 训练 → 零空间投影 → 特征提取 → LR-RGDA 分类器构建
3. 自动评估所有已学任务，记录 Transfer / Average / Last 指标矩阵
4. 保存结果 JSON

用法示例：
    python main_incremental.py \\
        --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \\
        --num_shots 16 --batch_size 32 --iterations 800 \\
        --lora_type lora_nsp --alpha 0.05
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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
    get_full_stats,
    print_paper_metrics,
)
from utils_data import get_xtail_trainloader, get_xtail_classnames, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Incremental Fine-tuning for CLIP Continual Learning")

    # 数据集相关参数
    parser.add_argument("--id_datasets", type=str, nargs='+',
                        default=["aircraft", "caltech101", "dtd", "eurosat", "flowers",
                                 "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"],
                        help="List of all ID datasets (for reference).")
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL",
                        help="Root directory of the dataset.")
    parser.add_argument("--num_shots", type=int, default=16,
                        help="Number of shots for few-shot learning.")
    parser.add_argument("--full_shot", action="store_true", default=False,
                        help="Use full dataset instead of few-shot (overrides --num_shots).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and testing.")
    parser.add_argument("--eval_max_samples", type=int, default=0,
                        help="Maximum number of test samples per dataset (0=full split).")

    # 增量学习特定参数
    parser.add_argument("--dataset_sequence", type=str, nargs='+',
                        default=["aircraft", "caltech101", "dtd", "eurosat", "flowers",
                                 "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"],
                        help="Sequence of datasets, one task per dataset. "
                             "Each element is a single dataset name.")

    # 训练基础参数
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (e.g., 0 for cuda:0). Sets CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", type=str,
                        default=None,
                        help="Device to use for training. Overrides --gpu if set.")
    parser.add_argument("--iterations", type=int, default=800,
                        help="Number of training iterations per task.")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=3e-5,
                        help="Weight decay for optimizer.")

    # LoRA 相关参数
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="Rank for LoRA adaptation.")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA alpha scaling factor. Defaults to lora_rank when None.")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout rate (only effective for lora_vanilla).")
    parser.add_argument("--lora_type", type=str, default="lora_nsp",
                        choices=["lora_vanilla", "lora_sgp", "lora_nsp"],
                        help="Type of LoRA adaptation (for backward compat).")
    parser.add_argument("--init_mode", type=str, default="lora_nsp",
                        choices=["lora_nsp", "lora_vanilla",
                                 "proj_sigma_tail", "proj_sigma_middle",
                                 "weight_svd_tail", "weight_svd_middle"],
                        help="""Adapter initialization mode.
    lora_nsp:        (default) runtime P + random A/B init (current LoRA-NSP)
    lora_vanilla:    standard LoRA, no P, no structured init
    proj_sigma_tail: Σ's smallest eigenvectors → project W → init A/B
    proj_sigma_middle: Σ's middle eigenvectors → project W → init A/B
    weight_svd_tail: W's smallest singular components → init A/B
    weight_svd_middle: W's middle singular components → init A/B""")
    parser.add_argument("--nsp_eps", type=float, default=0.05,
                        help="Epsilon parameter for NSP.")
    parser.add_argument("--nsp_weight", type=float, default=0.02,
                        help="Weight parameter for NSP.")
    parser.add_argument("--weight_temp", type=float, default=1.0,
                        help="Temperature parameter for weight.")
    parser.add_argument("--weight_kind", type=str, default="log1p")
    parser.add_argument("--weight_p", type=float, default=1.0,
                        help="P parameter for weight function.")

    # 参考数据集参数
    parser.add_argument("--reference_dataset", type=str, default="flickr8k",
                        help="Reference dataset for training.")
    parser.add_argument("--reference_batch_size", type=int, default=32,
                        help="Batch size for reference dataset.")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="Number of workers for data loading.")

    # 损失函数权重参数
    parser.add_argument("--fd_weight", type=float, default=1.0,
                        help="Weight for feature distillation loss (0=disabled).")
    parser.add_argument("--cd_weight", type=float, default=1.0,
                        help="Weight for cross-modal distillation loss (0=disabled).")
    parser.add_argument("--aux_weight", type=float, default=1.0,
                        help="Weight for auxiliary linear classifier loss (0=disabled). "
                             "Adds a linear head on features during training to improve "
                             "feature separability for downstream LR-RGDA.")
    parser.add_argument("--sce_a", type=float, default=0.5,
                        help="Weight for CE in symmetric cross-entropy loss.")
    parser.add_argument("--sce_b", type=float, default=0.5,
                        help="Weight for RCE in symmetric cross-entropy loss.")

    # 分类器参数
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Weight for LR-RGDA classifier in ensemble (paper: 0.05).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for zero-shot classifier.")
    parser.add_argument("--adaptive_ensemble", action='store_true', default=False,
                        help="Use per-sample adaptive alpha based on classifier confidence.")
    parser.add_argument("--classifier_feature_transform", type=str, default="test",
                        choices=["train", "test"],
                        help="Transform for LR-RGDA classifier feature extraction: "
                             "'train'=augmented, 'test'=deterministic.")

    # LR-RGDA 构建参数
    parser.add_argument("--rgda_rank", type=int, default=32,
                        help="Rank for LR-RGDA low-rank decomposition.")
    parser.add_argument("--rgda_alpha1", type=float, default=0.2,
                        help="qda_reg_alpha1 for LR-RGDA.")
    parser.add_argument("--rgda_alpha2", type=float, default=2.0,
                        help="qda_reg_alpha2 for LR-RGDA.")
    parser.add_argument("--rgda_alpha3", type=float, default=0.5,
                        help="qda_reg_alpha3 for LR-RGDA.")
    parser.add_argument("--rgda_train_iter", type=int, default=0,
                        help="LR-RGDA classifier fine-tuning iterations (0=analytical only).")
    parser.add_argument("--rgda_train_lr", type=float, default=0.01,
                        help="LR-RGDA classifier fine-tuning learning rate.")

    # 文本编码器 LoRA 参数
    parser.add_argument("--tune_text_encoder", type=lambda x: x.lower() == 'true', default=True,
                        help="是否同时微调文本编码器（默认 True）。设为 False 则仅微调图像编码器。")
    parser.add_argument("--text_lora_rank", type=int, default=4,
                        help="文本编码器 LoRA rank（默认 4，与 image lora_rank 一致）。")
    parser.add_argument("--max_zs_classes", type=int, default=128,
                        help="Zero-shot 分类器最大类数上限（用于联合训练时的随机采样）。")
    parser.add_argument("--tune_vision_encoder", type=lambda x: x.lower() == 'true', default=True,
                        help="是否微调视觉编码器（默认 True）。设为 False 则仅微调文本编码器（text-only模式）。")
    parser.add_argument("--text_tuning_schedule", type=str, default=None,
                        choices=["always", "never", "freeze_after", "low_lr_after"],
                        help="文本编码器微调调度。None=根据tune_text_encoder自动选择。")
    parser.add_argument("--text_schedule_switch_task", type=int, default=1,
                        help="切换到降低文本LR的任务编号（1-indexed）。")
    parser.add_argument("--text_lr_scale_after_task", type=float, default=0.2,
                        help="switch_task之后文本LR的缩放因子（low_lr_after模式下）。")
    parser.add_argument("--num_centers", type=int, default=1,
                        help="Number of k-means centers per class for multi-center LR-RGDA. 1=single-center.")

    # LADA 分类器参数（用于与 LR-RGDA 对比）
    parser.add_argument("--enable_lada", action="store_true", default=False,
                        help="Enable LADA classifier for comparison.")
    parser.add_argument("--lada_k", type=int, default=16,
                        help="LADA prototypes per class.")
    parser.add_argument("--lada_beta", type=float, default=1.0,
                        help="LADA affinity sharpness.")
    parser.add_argument("--lada_alpha", type=float, default=0.05,
                        help="LADA+ZS ensemble weight.")
    parser.add_argument("--lada_train_iter", type=int, default=0,
                        help="LADA classifier fine-tuning iterations (0=analytical).")
    parser.add_argument("--lada_train_lr", type=float, default=0.01,
                        help="LADA classifier fine-tuning learning rate.")

    # GMM 统计回放参数
    parser.add_argument("--use_gaussian_features", action="store_true", default=False,
                        help="Use GMM pseudo-features for classifier replay.")
    parser.add_argument("--gmm_k", type=int, default=4,
                        help="GMM components per class for statistical replay.")
    parser.add_argument("--gmm_sample_mode", type=str, default="mean",
                        choices=["mean", "sample"],
                        help="GMM replay mode: component means or stochastic samples.")
    parser.add_argument("--gmm_fit_space", type=str, default="raw",
                        choices=["raw", "sphere"],
                        help="Space for GMM fitting: raw features or L2-normalized sphere.")
    parser.add_argument("--gaussian_samples_per_class", type=int, default=16,
                        help="Number of pseudo-samples per class for GMM replay.")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="experiments",
                        help="Output directory for result JSON files.")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Stable experiment label for result file naming and summarizer.")

    args = parser.parse_args()
    # 将 dataset_sequence 转换为嵌套列表格式 [[d1], [d2], ...]
    args.dataset_sequence = [[d] for d in args.dataset_sequence]

    # 解析设备：优先 --device，否则用 --gpu 指定 cuda:N
    if args.device is None:
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    return args


def main(args):
    if args.seed is not None:
        fix_random_seed(args.seed)

    # ========== 1. 初始化 ==========
    logging.info("\n=== Initializing Incremental Learning ===")
    trainer = LoRANSPTrainer(args)
    model = trainer.model
    processor = trainer.processor

    # 解析文本编码器调度的默认值
    if args.text_tuning_schedule is None:
        args.text_tuning_schedule = "always" if args.tune_text_encoder else "never"

    # 验证文本调度参数
    if args.text_schedule_switch_task < 1:
        raise ValueError("--text_schedule_switch_task must be >= 1")
    if args.text_lr_scale_after_task < 0:
        raise ValueError("--text_lr_scale_after_task must be >= 0")

    # 非 never 调度时强制启用文本 LoRA（否则调度无意义）
    if args.text_tuning_schedule == "never":
        args.tune_text_encoder = False
    else:
        args.tune_text_encoder = True

    use_distillation = args.fd_weight > 0 or args.cd_weight > 0
    reference_loader = load_reference_dataset(args, trainer.model_pretrain,
                                              processor, args.device) if use_distillation else None

    # 预收集所有数据集的类名，用于全局 ZS 分类器（text LoRA 每次 merge 后更新）
    global_class_names = []
    for task_datasets in args.dataset_sequence:
        for d_name in task_datasets:
            c_names = get_xtail_classnames(args.root, d_name, args.num_shots)
            global_class_names.extend(c_names)

    # ========== 2. 增量学习循环 ==========
    history_class_names = []  # 记录所有已学类名列表的列表
    global_stats_dict = {}
    global_center_means = None  # 多中心 LR-RGDA 的类中心映射，用局部变量替代 main._global_center_means

    acc_matrix_zs = []
    acc_matrix_rgda = []
    acc_matrix_ens = []
    acc_matrix_lada = [] if args.enable_lada else None
    acc_matrix_lada_zs = [] if args.enable_lada else None

    for i, task_datasets in enumerate(args.dataset_sequence):
        print(f"\n" + "=" * 50)
        print(f"=== Task {i+1}: {task_datasets} ===")
        print("=" * 50)

        # --- 2a. 准备训练数据 ---
        train_loaders = []
        task_class_names = []
        for d_name in task_datasets:
            train_transform, test_transform = get_transforms(d_name)
            num_shots = None if args.full_shot else args.num_shots
            tr_loader, _, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=num_shots, batch_size=args.batch_size
            )
            train_loaders.append(tr_loader)
            task_class_names.extend(c_names)

        merged_dataset = ConcatDataset([loader.dataset for loader in train_loaders])
        # 用于提取协方差的 loader 不打乱顺序
        cov_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=False)
        merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True)

        # --- 2b. 初始化适配器（非标准 LoRA 时在训练前初始化） ---
        if args.init_mode not in ["lora_nsp", "lora_vanilla"]:
            logging.info(f"\n=== Initializing adapters ({args.init_mode}) ===")
            if "proj_sigma" in args.init_mode and args.init_mode != "lora_nsp":
                window = "tail" if "tail" in args.init_mode else "middle"
                if trainer.covariance_history:
                    trainer.model.vision_model.initialize_adapters_from_covariance(
                        trainer.covariance_history, window=window)
                else:
                    logging.info("No covariance history yet, using default random init for Task 1")
            elif "weight_svd" in args.init_mode:
                window = "tail" if "tail" in args.init_mode else "middle"
                trainer.model.vision_model.initialize_adapters_from_weight_svd(window=window)

        # --- 2c. 训练模型 ---
        # 文本编码器调度：根据当前任务决定 text_lr
        task_number = i + 1
        schedule = args.text_tuning_schedule
        text_lr = args.lr  # 默认
        if schedule == "never":
            text_lr = 0.0
        elif schedule in ("freeze_after", "low_lr_after"):
            if task_number > args.text_schedule_switch_task:
                if schedule == "freeze_after":
                    text_lr = 0.0
                else:
                    text_lr = args.lr * args.text_lr_scale_after_task
        train_text_this_task = text_lr > 0
        model = trainer.train(merged_loader, task_class_names, reference_loader,
                              aux_weight=args.aux_weight,
                              train_text_encoder=train_text_this_task,
                              text_lr=text_lr)

        # --- 2d. 任务后处理：合入 + 协方差累积 ---
        if args.init_mode == "lora_nsp":
            print("\n=== Applying Null-Space Projection (NSP) ===")
            # 图像编码器 NSP
            text_covariances = None
            if trainer.has_vision_lora:
                covariances = trainer.extract_layer_covariances(cov_loader)
            # 文本编码器 NSP（仅当本轮实际训练了文本编码器）
            if trainer.has_text_lora and train_text_this_task:
                text_covariances = trainer.extract_text_covariances(task_class_names)
            trainer.finalize_task_for_incremental()
            if trainer.has_vision_lora:
                trainer.update_covariance_history(covariances)
            if trainer.has_text_lora and train_text_this_task:
                trainer.update_text_covariance_history(text_covariances)
        elif "proj_sigma" in args.init_mode:
            print("\n=== Proj-Σ: Extracting covariances + merging ===")
            if trainer.has_vision_lora:
                covariances = trainer.extract_layer_covariances(cov_loader)
                trainer.update_covariance_history(covariances, update_projection=False)
            trainer.finalize_task_for_incremental()
        else:
            print(f"\n=== Merging LoRA Weights (init_mode={args.init_mode}) ===")
            trainer.finalize_task_for_incremental()

        # --- 2e. 提取特征并构建统计字典 ---
        task_features = []
        task_labels = []
        label_offset = sum(len(c_names) for c_names in history_class_names)

        for d_name in task_datasets:
            train_transform, test_transform = get_transforms(d_name)
            tr_loader, tr4update, _, c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=train_transform, transform_test=test_transform,
                num_shots=num_shots, batch_size=args.batch_size
            )
            # 根据 --classifier_feature_transform 选择特征提取的 transform
            # 'test' = 确定性变换（publication protocol），'train' = 随机增强
            feature_loader = tr_loader if args.classifier_feature_transform == "train" else tr4update
            from src.utils.feature_extractor import extract_features
            features, labels = extract_features(model, feature_loader, args.device)
            features = features / features.norm(dim=-1, keepdim=True)
            task_features.append(features)
            task_labels.append(labels + label_offset)

        task_features = torch.cat(task_features)
        task_labels = torch.cat(task_labels)

        # 构建多中心统计（M=1 等价于单中心）
        task_stats_dict, center_means = build_multi_center_stats_dict(
            task_features, task_labels, M=args.num_centers)

        # 累加统计字典和中心均值
        global_stats_dict.update(task_stats_dict)
        if center_means is not None:
            if global_center_means is None:
                global_center_means = {}
            # center_means 的 key 已是全局类ID（来自 task_labels + label_offset）
            global_center_means.update(center_means)

        history_class_names.append(task_class_names)

        # --- 2f. 构建分类器 ---
        # 计算数据集等权的全局协方差
        offset = 0
        per_dataset_covs = []
        for hcn in history_class_names:
            n_classes = len(hcn)
            ds_cov = sum(global_stats_dict[cid].cov for cid in range(offset, offset + n_classes)) / n_classes
            per_dataset_covs.append(ds_cov)
            offset += n_classes

        dataset_balanced_global_cov = sum(per_dataset_covs) / len(per_dataset_covs)

        lr_rgda_classifier = LRRGDAClassifier(
            stats_dict=global_stats_dict,
            device=args.device,
            rank=args.rgda_rank,
            M=args.num_centers,
            center_means=global_center_means,
            qda_reg_alpha1=args.rgda_alpha1,
            qda_reg_alpha2=args.rgda_alpha2,
            qda_reg_alpha3=args.rgda_alpha3,
            temperature=1.0,
            global_cov=dataset_balanced_global_cov,
        )
        # 分类器微调（如果启用）：用 GMM mean replay 生成旧类伪特征，与当前任务真实特征合并
        if args.rgda_train_iter > 0:
            current_task_ids = set(task_labels.unique().tolist())
            old_class_ids = sorted(set(global_stats_dict.keys()) - current_task_ids)

            if old_class_ids:
                logging.info(f"[GMM Replay] Generating pseudo-features for "
                             f"{len(old_class_ids)} old classes (mode={args.gmm_sample_mode})...")
                n_samples = args.gaussian_samples_per_class
                old_feats, old_lbls = [], []

                for cid in old_class_ids:
                    if global_center_means is not None and cid in global_center_means:
                        # 多中心：用 k-means 聚类中心
                        centers = global_center_means[cid].to(args.device)  # [M, D]
                    else:
                        # 单中心：用 stats_dict 的类均值
                        centers = global_stats_dict[cid].mean.unsqueeze(0).to(args.device)  # [1, D]

                    M_c = centers.shape[0]
                    per_center = max(1, n_samples // M_c)
                    remainder = n_samples - per_center * M_c

                    for m in range(M_c):
                        cnt = per_center + (1 if m < remainder else 0)
                        mean = centers[m]
                        if args.gmm_sample_mode == "sample":
                            # 添加少量球形噪声
                            noise = torch.randn(cnt, mean.shape[0], device=mean.device) * 0.05
                            samples = mean.unsqueeze(0) + noise
                        else:
                            # "mean" 模式：直接复制分量均值
                            samples = mean.unsqueeze(0).repeat(cnt, 1)
                        samples = samples / samples.norm(dim=-1, keepdim=True)
                        old_feats.append(samples)
                        old_lbls.append(torch.full((cnt,), cid, dtype=torch.long, device=samples.device))

                replay_features = torch.cat([task_features.to(args.device)] + old_feats, dim=0)
                replay_labels = torch.cat([task_labels.to(args.device)] + old_lbls, dim=0)
            else:
                replay_features = task_features.to(args.device)
                replay_labels = task_labels.to(args.device)

            logging.info(f"[Fit] Fine-tuning LR-RGDA for {args.rgda_train_iter} iters "
                         f"on {replay_features.shape[0]} samples "
                         f"({len(old_class_ids)} old classes replayed)...")
            lr_rgda_classifier.fit(
                replay_features, replay_labels,
                iterations=args.rgda_train_iter,
                lr=args.rgda_train_lr,
            )

        flat_class_names = [name for sublist in history_class_names for name in sublist]
        current_num_classes = len(flat_class_names)
        zeroshot_classifier = get_zeroshot_classifier(model, processor,
                                                      flat_class_names, args.device)

        # --- 2g. 评估所有已学任务 ---
        print("\n=== Evaluating Task ===")
        step_accs_zs, step_accs_rgda, step_accs_ens = [], [], []

        eval_label_offset = 0
        for j in range(i + 1):
            eval_datasets = args.dataset_sequence[j]
            d_name = eval_datasets[0]  # 每个 Task 只有一个数据集

            zs_acc, rgda_acc, ens_acc, lada_acc, lada_zs_acc, c_len, _ = evaluate_dataset(
                args, d_name, model, zeroshot_classifier, lr_rgda_classifier,
                current_num_classes, eval_label_offset
            )
            eval_label_offset += c_len

            print(f"[Tested on Task {j+1}: {d_name:<10s}] -> "
                  f"Zero-shot: {zs_acc:5.1f}% | LR-RGDA: {rgda_acc:5.1f}% | "
                  f"Ensemble: {ens_acc:5.1f}%")

            step_accs_zs.append(zs_acc)
            step_accs_rgda.append(rgda_acc)
            step_accs_ens.append(ens_acc)

        acc_matrix_zs.append(step_accs_zs)
        acc_matrix_rgda.append(step_accs_rgda)
        acc_matrix_ens.append(step_accs_ens)

    # ========== 3. 打印最终结果 ==========
    print("\n=== Training and Evaluation Completed ===")
    print("\n" + "=" * 80)
    print("Final Results Mapping to Paper Tables")
    print("=" * 80)

    task_names = [d[0] for d in args.dataset_sequence]

    print_paper_metrics(acc_matrix_zs, "Zero-shot Baseline", task_names)
    print_paper_metrics(acc_matrix_rgda, "LR-RGDA Only", task_names)
    print_paper_metrics(acc_matrix_ens, f"Ours Ensemble (alpha={args.alpha})", task_names)

    # ========== 4. 保存结果 JSON ==========
    # 格式与 summarize_incremental_metrics.py 兼容
    zs_stats = get_full_stats(acc_matrix_zs)
    rgda_stats = get_full_stats(acc_matrix_rgda)
    ens_stats = get_full_stats(acc_matrix_ens)

    # 将三角矩阵补齐为方阵（summarizer 期望 K x K）
    K = len(task_names)
    def _pad_to_square(triangular_rows):
        square = []
        for r, row in enumerate(triangular_rows):
            padded = list(row) + [0.0] * (K - len(row))
            square.append(padded)
        return square

    output_dir = args.output_dir or "experiments"
    os.makedirs(output_dir, exist_ok=True)
    stem = args.experiment_name or f"incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    file_name = f"{stem}.json"
    file_name_zs = f"{stem}_zs.json"
    file_name_rgda = f"{stem}_rgda.json"
    file_name_ens = f"{stem}_ens.json"
    save_path = os.path.join(output_dir, file_name)

    # Per-classifier result files (summarizer-compatible)
    for suffix, matrix, stats, method_name in [
        ("_zs", acc_matrix_zs, zs_stats, "zero_shot"),
        ("_rgda", acc_matrix_rgda, rgda_stats, "lr_rgda"),
        ("_ens", acc_matrix_ens, ens_stats, "ensemble"),
    ]:
        per_file = os.path.join(output_dir, f"{stem}{suffix}.json")
        per_result = {
            "args": {
                "task_sequence": task_names,
                "seed": args.seed,
                "method": method_name,
                "eval_max_samples": args.eval_max_samples,
                "num_shots": args.num_shots,
            },
            "accuracy_matrix": _pad_to_square(matrix),
            "metrics": {
                "transfer": stats["transfer_total_avg"],
                "average": stats["average_total_avg"],
                "last": stats["last_total_avg"],
            },
            "per_task_metrics": {
                "transfer": stats["transfer"],
                "average_per_task": stats["average_per_task"],
                "last": stats["last"],
            },
        }
        with open(per_file, 'w', encoding='utf-8') as f:
            json.dump(per_result, f, indent=2)

    # 汇总 JSON（兼容旧格式 + 新增信息）
    save_results = {
        "mode": "Incremental Learning",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_order": task_names,
        "arguments": {k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v
                      for k, v in vars(args).items()},
        "args": {
            "task_sequence": task_names,
            "seed": args.seed,
            "method": args.lora_type,
            "eval_max_samples": args.eval_max_samples,
            "num_shots": args.num_shots,
        },
        "accuracy_matrix": _pad_to_square(acc_matrix_ens),
        "metrics": {
            "zero_shot": zs_stats,
            "lr_rgda": rgda_stats,
            "ours_ensemble": ens_stats,
        },
        "summary_metrics": {
            "zero_shot": {k: zs_stats[k] for k in ["transfer_total_avg", "average_total_avg", "last_total_avg"]},
            "lr_rgda": {k: rgda_stats[k] for k in ["transfer_total_avg", "average_total_avg", "last_total_avg"]},
            "ours_ensemble": {k: ens_stats[k] for k in ["transfer_total_avg", "average_total_avg", "last_total_avg"]},
        },
        "per_file_results": {
            "zero_shot": os.path.basename(file_name_zs),
            "lr_rgda": os.path.basename(file_name_rgda),
            "ensemble": os.path.basename(file_name_ens),
        },
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n增量学习结果已保存至: {save_path}")


if __name__ == "__main__":
    command_line_args = parse_args()
    main(command_line_args)
