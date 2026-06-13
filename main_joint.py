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
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from src.trainers.lora_nsp_trainer import LoRANSPTrainer
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.classifiers.gaussian_statistics import build_multi_center_stats_dict
from src.lada import LADAClassifier
from src.utils.reference_loader import load_reference_dataset

from src.utils.main_utils import (
    fix_random_seed,
    get_zeroshot_classifier,
    evaluate_dataset,
    batch_evaluate_datasets,
)
from utils_data import get_xtail_trainloader, get_xtail_classnames, get_transforms


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
    parser.add_argument("--lora_type", type=str, default="lora_nsp",
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
    parser.add_argument("--fd_weight", type=float, default=0.0,
                        help="Weight for feature distillation loss (0=disabled).")
    parser.add_argument("--cd_weight", type=float, default=0.0,
                        help="Weight for cross-modal distillation loss (0=disabled).")
    parser.add_argument("--aux_weight", type=float, default=0.0,
                        help="Weight for auxiliary linear classifier loss (0=disabled). "
                             "Adds a linear head on features during training to improve "
                             "feature separability for downstream LR-RGDA.")
    parser.add_argument("--sce_a", type=float, default=0.5,
                        help="Weight for CE in symmetric cross-entropy loss.")
    parser.add_argument("--sce_b", type=float, default=0.5,
                        help="Weight for RCE in symmetric cross-entropy loss.")

    # 模型微调开关
    parser.add_argument("--tune_student", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to fine-tune the student model (default: True). "
                             "Set to False to skip training and only evaluate.")

    # Alpha 敏感性分析
    parser.add_argument("--alpha_sensitivity", action='store_true', default=False,
                        help="If set, evaluate ensemble accuracy at multiple alpha values "
                             "(0 to 1.0) to analyze sensitivity.")
    parser.add_argument("--adaptive_ensemble", action='store_true', default=False,
                        help="Use per-sample adaptive alpha based on classifier confidence.")

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

    # LADA 分类器参数
    parser.add_argument("--enable_lada", action='store_true', default=False,
                        help="是否启用 LADA 分类器，与 LR-RGDA 同时评估对比。")
    parser.add_argument("--lada_k", type=int, default=16,
                        help="LADA 每类聚类中心数 k（默认 16）。")
    parser.add_argument("--lada_beta", type=float, default=1.0,
                        help="LADA 指数亲和变换的 β 参数（默认 1.0）。")
    parser.add_argument("--lada_alpha", type=float, default=1.0,
                        help="LADA+ZS 集成中 LADA logits 的权重（默认 1.0）。")
    parser.add_argument("--lada_train_iter", type=int, default=0,
                        help="LADA 分类器梯度微调迭代次数（默认 0，不禁用）。")
    parser.add_argument("--lada_train_lr", type=float, default=0.01,
                        help="LADA 分类器梯度微调学习率（默认 0.01）。")

    # LR-RGDA 微调参数
    parser.add_argument("--rgda_train_iter", type=int, default=0,
                        help="LR-RGDA 分类器梯度微调迭代次数（默认 0，不禁用）。")
    parser.add_argument("--rgda_train_lr", type=float, default=0.01,
                        help="LR-RGDA 分类器梯度微调学习率（默认 0.01）。")

    # 高斯特征采样
    parser.add_argument("--use_gaussian_features", action='store_true', default=False,
                        help="用高斯分布采样伪特征替代真实特征来训练分类器。")
    parser.add_argument("--gaussian_samples_per_class", type=int, default=16,
                        help="每类高斯采样数（默认 16）。")
    parser.add_argument("--gmm_k", type=int, default=0,
                        help="每类 spherical GMM 分量数（默认 0，使用正则化高斯；>0 启用 GMM 采样）。")
    parser.add_argument("--gmm_reg", type=float, default=0.0,
                        help="GMM spherical 协方差正则化（加到标量方差上，默认 0.0）。")
    parser.add_argument("--gmm_cov_type", type=str, default="spherical",
                        choices=["spherical", "rank1"],
                        help="GMM 协方差类型（spherical=标量, rank1=球面基+秩1主成分）。")
    parser.add_argument("--gmm_fit_space", type=str, default="raw",
                        choices=["raw", "sphere"],
                        help="GMM 拟合空间。raw 对齐官方 LADA；sphere 仅用于错误实现对照。")
    parser.add_argument("--gmm_sample_mode", type=str, default="sample",
                        choices=["sample", "mean"],
                        help="GMM 回放方式：sample=按协方差采样，mean=仅重复分量均值。")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="实验名称，写入结果 JSON 并用于输出文件名。")
    parser.add_argument("--output_dir", type=str, default="experiments/joint_classifier_replay",
                        help="结果 JSON 输出目录。")

    # 高斯采样正则化参数（独立于分类器构建参数）
    parser.add_argument("--sample_alpha1", type=float, default=None,
                        help="采样正则化 α1，默认沿用 rgda_alpha1。")
    parser.add_argument("--sample_alpha2", type=float, default=None,
                        help="采样正则化 α2，默认沿用 rgda_alpha2。")
    parser.add_argument("--sample_alpha3", type=float, default=None,
                        help="采样正则化 α3，默认沿用 rgda_alpha3。")

    # 文本编码器 LoRA 参数（联合训练默认关闭，1100 类全量编码显存不足）
    parser.add_argument("--tune_text_encoder", type=lambda x: x.lower() == 'true', default=False,
                        help="是否同时微调文本编码器（默认 False）。联合训练类多，显存压力大。")
    parser.add_argument("--text_lora_rank", type=int, default=4,
                        help="文本编码器 LoRA rank（默认 4）。")
    parser.add_argument("--tune_vision_encoder", type=lambda x: x.lower() == 'true', default=False,
                        help="是否微调视觉编码器（默认 False）。设为 True 则微调视觉编码器。")

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
        all_raw_feats = []
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
            raw_features, labels = extract_features(
                eval_model, tr_loader, args.device, normalize=False)
            features = torch.nn.functional.normalize(raw_features, dim=-1)
            all_raw_feats.append(raw_features)
            all_feats.append(features)
            all_lbls.append(labels + feat_offset)
            feat_offset += len(c_names)

        all_features = torch.cat(all_feats)
        all_raw_features = torch.cat(all_raw_feats)  # 用于 GMM 拟合
        all_labels = torch.cat(all_lbls)

        # 原始空间 stats_dict（用于高斯采样，协方差有意义的量级）
        raw_stats_dict, _ = build_multi_center_stats_dict(
            all_raw_features, all_labels, M=1
        )

        # 原始空间全局协方差
        feat_offset = 0
        raw_per_dataset_covs = []
        for d_name in args.id_datasets:
            c_names = get_xtail_classnames(args.root, d_name, args.num_shots)
            n_classes = len(c_names)
            ds_cov = sum(raw_stats_dict[cid].cov for cid in range(feat_offset, feat_offset + n_classes)) / n_classes
            raw_per_dataset_covs.append(ds_cov)
            feat_offset += n_classes
        raw_global_cov = sum(raw_per_dataset_covs) / len(raw_per_dataset_covs)

        # 可选：用高斯分布采样伪特征替代真实特征
        if args.use_gaussian_features:
            if args.gmm_k > 0:
                # ---- Spherical GMM 模式 ----
                logging.info(f"Sampling ~{args.gaussian_samples_per_class} features per class "
                             f"from spherical GMM (k={args.gmm_k}, "
                             f"fit_space={args.gmm_fit_space})")
                from sklearn.mixture import GaussianMixture
                import numpy as np
                gmm_fit_features = (
                    all_raw_features if args.gmm_fit_space == "raw" else all_features
                )
                gmm_fit_features_np = gmm_fit_features.cpu().numpy()
                all_labels_np = all_labels.cpu().numpy()
                syn_feats = []
                syn_labels = []

                for cid in sorted(torch.unique(all_labels).tolist()):
                    mask = all_labels_np == cid
                    feats_c = gmm_fit_features_np[mask]

                    if len(feats_c) < args.gmm_k:
                        # 样本太少，直接复制或降 k
                        actual_k = max(1, len(feats_c))
                    else:
                        actual_k = args.gmm_k

                    gmm = GaussianMixture(n_components=actual_k, covariance_type='spherical',
                                          random_state=42, reg_covar=1e-6)
                    gmm.fit(feats_c)

                    # 按权重比例分配每分量采样数
                    total = args.gaussian_samples_per_class
                    n_per_comp = np.maximum(1, (gmm.weights_ * total).round().astype(int))
                    diff = total - n_per_comp.sum()
                    n_per_comp[np.argmax(gmm.weights_)] += diff

                    if args.gmm_cov_type == 'rank1':
                        # 球形基 + rank-1 主成分: σ²_base·I + λ·vvᵀ
                        # 采样 = μ + σ_base·ε + √λ·v·η
                        hard_labels = gmm.predict(feats_c)
                        sigma_base = np.sqrt(np.mean(gmm.covariances_))  # 球形基标准差
                        for comp in range(actual_k):
                            mean = torch.from_numpy(gmm.means_[comp]).float()
                            n = max(1, int(n_per_comp[comp]))
                            comp_mask = hard_labels == comp
                            n_points = comp_mask.sum()
                            # 球形基噪声
                            iso_noise = torch.randn(n, mean.shape[0]) * sigma_base
                            # rank-1 主成分
                            if n_points >= 2:
                                centered = feats_c[comp_mask] - gmm.means_[comp]
                                _, S, Vt = np.linalg.svd(centered, full_matrices=False)
                                direction = torch.from_numpy(Vt[0]).float()
                                var_pc1 = (S[0] ** 2) / n_points
                                pc1_noise = torch.randn(n, 1) * np.sqrt(max(var_pc1, 1e-8)) * direction.unsqueeze(0)
                            else:
                                pc1_noise = torch.zeros(n, mean.shape[0])
                            samples = mean.unsqueeze(0) + iso_noise + pc1_noise
                            samples = samples / samples.norm(dim=-1, keepdim=True)
                            syn_feats.append(samples.to(args.device))
                            syn_labels.append(torch.full((samples.shape[0],), cid, device=args.device))
                    else:
                        # spherical 模式
                        for comp in range(actual_k):
                            mean = torch.from_numpy(gmm.means_[comp]).float()
                            var = gmm.covariances_[comp] + args.gmm_reg
                            n = max(1, int(n_per_comp[comp]))
                            if args.gmm_sample_mode == "mean":
                                samples = mean.unsqueeze(0).repeat(n, 1)
                            else:
                                noise = torch.randn(n, mean.shape[0]) * np.sqrt(max(var, 1e-8))
                                samples = mean.unsqueeze(0) + noise
                            samples = samples / samples.norm(dim=-1, keepdim=True)
                            syn_feats.append(samples.to(args.device))
                            syn_labels.append(torch.full((samples.shape[0],), cid, device=args.device))

                syn_features = torch.cat(syn_feats)
                syn_labels_t = torch.cat(syn_labels)
                logging.info(
                    f"Generated {syn_features.shape[0]} features via {args.gmm_cov_type} "
                    f"GMM (k={args.gmm_k}, fit_space={args.gmm_fit_space}, "
                    f"sample_mode={args.gmm_sample_mode})")
            else:
                # ---- 正则化高斯模式（原始空间估计 + 正则化 + 归一化）----
                sa1 = args.sample_alpha1 if args.sample_alpha1 is not None else args.rgda_alpha1
                sa2 = args.sample_alpha2 if args.sample_alpha2 is not None else args.rgda_alpha2
                sa3 = args.sample_alpha3 if args.sample_alpha3 is not None else args.rgda_alpha3
                logging.info(f"Sampling {args.gaussian_samples_per_class} synthetic features "
                             f"per class from full Gaussian (α1={sa1}, α2={sa2}, α3={sa3}) "
                             f"[raw feature space → regularize → sample → L2 normalize]")
                identity = torch.eye(raw_global_cov.shape[0], device=args.device)
                raw_global_cov_device = raw_global_cov.to(args.device)
                syn_feats = []
                syn_labels = []
                from src.classifiers.gaussian_statistics import GaussianStatistics

                for cid in sorted(raw_stats_dict.keys()):
                    raw_cov = raw_stats_dict[cid].cov.to(args.device)
                    mean = raw_stats_dict[cid].mean.to(args.device)

                    reg_cov = (sa1 * raw_cov +
                               sa2 * raw_global_cov_device +
                               sa3 * identity)
                    reg_cov = 0.5 * (reg_cov + reg_cov.T)

                    reg_stats = GaussianStatistics(mean, reg_cov, reg=1e-4, cholesky=True)
                    samples = reg_stats.sample(n_samples=args.gaussian_samples_per_class)
                    samples = samples / samples.norm(dim=-1, keepdim=True)  # 采样后 L2 归一化
                    syn_feats.append(samples)
                    syn_labels.append(torch.full((args.gaussian_samples_per_class,), cid, device=args.device))
                syn_features = torch.cat(syn_feats)
                syn_labels_t = torch.cat(syn_labels)
                logging.info(f"Generated {syn_features.shape[0]} features (regularized covariance)")
            fit_features = syn_features
            fit_labels = syn_labels_t
        else:
            fit_features = all_features.to(args.device)
            fit_labels = all_labels.to(args.device)

        # Both classifiers must be constructed from the same feature source.
        # Under replay, this prevents LR-RGDA from retaining hidden access to
        # real-feature centers and covariances while LADA uses pseudo features.
        fit_features_cpu = fit_features.detach().cpu()
        fit_labels_cpu = fit_labels.detach().cpu()
        stats_dict, center_means = build_multi_center_stats_dict(
            fit_features_cpu, fit_labels_cpu, M=args.num_centers
        )

        feat_offset = 0
        per_dataset_covs = []
        for d_name in args.id_datasets:
            c_names = get_xtail_classnames(args.root, d_name, args.num_shots)
            n_classes = len(c_names)
            ds_cov = sum(
                stats_dict[cid].cov
                for cid in range(feat_offset, feat_offset + n_classes)
            ) / n_classes
            per_dataset_covs.append(ds_cov)
            feat_offset += n_classes
        dataset_balanced_global_cov = sum(per_dataset_covs) / len(per_dataset_covs)

        source_name = (
            f"gmm-{args.gmm_fit_space}" if args.use_gaussian_features else "real"
        )
        logging.info(
            "[Classifier Source] %s features: n=%d, mean_norm=%.4f",
            source_name, fit_features.shape[0],
            fit_features.norm(dim=-1).mean().item())

        lr_rgda_classifier = LRRGDAClassifier(
            stats_dict=stats_dict, device=args.device,
            rank=args.rgda_rank,
            qda_reg_alpha1=args.rgda_alpha1,
            qda_reg_alpha2=args.rgda_alpha2,
            qda_reg_alpha3=args.rgda_alpha3,
            temperature=1.0,
            M=args.num_centers, center_means=center_means,
            global_cov=dataset_balanced_global_cov,
        )

        if args.rgda_train_iter > 0:
            logging.info(f"LR-RGDA fine-tuning: {args.rgda_train_iter} iters, lr={args.rgda_train_lr}")
            lr_rgda_classifier.fit(fit_features, fit_labels,
                                   iterations=args.rgda_train_iter, lr=args.rgda_train_lr)

        if args.enable_lada:
            lada_classifier = LADAClassifier(feature_dim=fit_features.shape[1], beta=args.lada_beta)
            lada_classifier.build_from_data(fit_features, fit_labels, k=args.lada_k)
            lada_classifier.to(args.device)
            logging.info(f"LADA built: {lada_classifier.get_total_classes()} classes, "
                         f"k={args.lada_k}, beta={args.lada_beta}")
            if args.lada_train_iter > 0:
                logging.info(f"LADA fine-tuning: {args.lada_train_iter} iters, lr={args.lada_train_lr}")
                lada_classifier.fit(fit_features, fit_labels,
                                    iterations=args.lada_train_iter, lr=args.lada_train_lr)
        else:
            lada_classifier = None

        num_id_classes = len(all_class_names)
        zs_classifier = get_zeroshot_classifier(eval_model, processor,
                                                 all_class_names, args.device)

        logging.info(f"\n=== Evaluating ID Datasets {tag}===")
        id_zs, id_rgda, id_ens = [], [], []
        id_lada, id_lada_zs = [], []

        id_dataset_offset_map = {}
        eval_offset = 0
        for d_name in args.id_datasets:
            c_names = get_xtail_classnames(args.root, d_name, args.num_shots)
            id_dataset_offset_map[d_name] = eval_offset
            eval_offset += len(c_names)

        eval_offset = 0
        for d_name in args.id_datasets:
            zs_acc, rgda_acc, ens_acc, lada_acc, lada_zs_acc, c_len, _ = evaluate_dataset(
                args, d_name, eval_model, zs_classifier, lr_rgda_classifier,
                num_id_classes, eval_offset,
                lada_classifier=lada_classifier, lada_alpha=args.lada_alpha
            )
            id_zs.append(zs_acc)
            id_rgda.append(rgda_acc)
            id_ens.append(ens_acc)
            if lada_classifier is not None:
                id_lada.append(lada_acc)
                id_lada_zs.append(lada_zs_acc)
            eval_offset += c_len
            log_str = f"[{tag} ID] {d_name:<12s} | ZS: {zs_acc:5.1f}% | " \
                      f"RGDA: {rgda_acc:5.1f}% | Ensemble: {ens_acc:5.1f}%"
            if lada_classifier is not None:
                log_str += f" | LADA: {lada_acc:5.1f}% | LADA+ZS: {lada_zs_acc:5.1f}%"
            logging.info(log_str)

        id_zs_avg = sum(id_zs) / len(id_zs)
        id_rgda_avg = sum(id_rgda) / len(id_rgda)
        id_ens_avg = sum(id_ens) / len(id_ens)
        id_lada_avg = sum(id_lada) / len(id_lada) if id_lada else None
        id_lada_zs_avg = sum(id_lada_zs) / len(id_lada_zs) if id_lada_zs else None

        # 打印格式化总表（中间评估 + 最终评估共用）
        header = f"{'Dataset':<15s} | {'Zero-shot':>9s} | {'LR-RGDA':>9s} | {'Ensemble':>9s}"
        sep = "-" * 55
        if lada_classifier is not None:
            header += f" | {'LADA':>9s} | {'LADA+ZS':>9s}"
            sep = "-" * 79
        print(f"\n[{tag} ID Summary]")
        print(header)
        print(sep)
        for i, d_name in enumerate(args.id_datasets):
            row = f"{d_name:<15s} | {id_zs[i]:>7.1f}%  | {id_rgda[i]:>7.1f}%  | {id_ens[i]:>7.1f}%"
            if lada_classifier is not None:
                row += f" | {id_lada[i]:>7.1f}%  | {id_lada_zs[i]:>7.1f}%"
            print(row)
        print(sep)
        avg_row = f"{'Average':<15s} | {id_zs_avg:>7.1f}%  | {id_rgda_avg:>7.1f}%  | {id_ens_avg:>7.1f}%"
        if lada_classifier is not None:
            avg_row += f" | {id_lada_avg:>7.1f}%  | {id_lada_zs_avg:>7.1f}%"
        print(avg_row)

        return (id_zs, id_rgda, id_ens, id_zs_avg, id_rgda_avg, id_ens_avg,
                id_lada, id_lada_zs, id_lada_avg, id_lada_zs_avg,
                id_dataset_offset_map, num_id_classes, zs_classifier,
                lr_rgda_classifier, lada_classifier)

    all_class_names = []
    if tune_student and args.iterations > 0:
        # 只有在微调且蒸馏损失权重 > 0 时才加载参考数据集
        use_distillation = args.fd_weight > 0 or args.cd_weight > 0
        if use_distillation:
            reference_loader = load_reference_dataset(args, trainer.model_pretrain,
                                                      processor, args.device)
        else:
            reference_loader = None

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

        # 均衡采样：每个数据集在每个 batch 中等比例出现
        num_datasets = len(all_shifted_datasets)
        weights = []
        for ds in all_shifted_datasets:
            n = len(ds)
            weights.extend([1.0 / (n * num_datasets)] * n)
        sampler = WeightedRandomSampler(weights, num_samples=len(merged_dataset), replacement=True)
        merged_loader = DataLoader(merged_dataset, batch_size=args.batch_size,
                                   sampler=sampler, num_workers=6)

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
            c_names = get_xtail_classnames(args.root, d_name, args.num_shots)
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
         id_lada_accs, id_lada_zs_accs, id_lada_avg, id_lada_zs_avg,
         id_dataset_offset, num_id_classes,
         zeroshot_classifier, lr_rgda_classifier, lada_classifier) = run_full_evaluation(model, tag="Final")

    else:
        logging.info(f"\n=== Skipping Fine-tuning (tune_student={args.tune_student}, iterations={args.iterations}) ===")
        for d_name in args.id_datasets:
            c_names = get_xtail_classnames(args.root, d_name, args.num_shots)
            all_class_names.extend(c_names)

        (id_zs_accs, id_rgda_accs, id_ens_accs,
         id_zs_avg, id_rgda_avg, id_ens_avg,
         id_lada_accs, id_lada_zs_accs, id_lada_avg, id_lada_zs_avg,
         id_dataset_offset, num_id_classes,
         zeroshot_classifier, lr_rgda_classifier, lada_classifier) = run_full_evaluation(model, tag="No-tune")

    # ========== 4. 评估 OOD 数据集 ==========
    logging.info("\n=== Evaluating OOD Datasets (Ensemble with alpha=%.1f) ===" % args.alpha)
    ood_zs_accs = []
    ood_rgda_accs = []
    ood_ens_accs = []
    ood_lada_accs = []
    ood_lada_zs_accs = []

    for d_name in args.ood_datasets:
        if d_name in id_dataset_offset:
            zs_acc, rgda_acc, ens_acc, lada_acc, lada_zs_acc, _, _ = evaluate_dataset(
                args, d_name, model, zeroshot_classifier, lr_rgda_classifier,
                num_id_classes, id_dataset_offset[d_name],
                lada_classifier=lada_classifier, lada_alpha=args.lada_alpha
            )
        else:
            _, test_transform = get_transforms(d_name)
            _, te_loader, _, ood_c_names = get_xtail_trainloader(
                root=args.root, dataset_name=d_name,
                transform_train=None, transform_test=test_transform,
                num_shots=args.num_shots, batch_size=args.batch_size
            )
            combined_class_names = all_class_names + ood_c_names
            combined_zeroshot = get_zeroshot_classifier(model, processor, combined_class_names, args.device)
            total_classes = len(combined_class_names)

            zs_acc, rgda_acc, ens_acc, lada_acc, lada_zs_acc, _, _ = evaluate_dataset(
                args, d_name, model, combined_zeroshot, lr_rgda_classifier,
                num_id_classes, num_id_classes,
                lada_classifier=lada_classifier, lada_alpha=args.lada_alpha
            )

        ood_zs_accs.append(zs_acc)
        ood_rgda_accs.append(rgda_acc)
        ood_ens_accs.append(ens_acc)
        if lada_classifier is not None:
            ood_lada_accs.append(lada_acc)
            ood_lada_zs_accs.append(lada_zs_acc)
        log_str = f"[OOD] {d_name:<12s} | ZS: {zs_acc:5.1f}% | " \
                  f"RGDA: {rgda_acc:5.1f}% | Ensemble: {ens_acc:5.1f}%"
        if lada_classifier is not None:
            log_str += f" | LADA: {lada_acc:5.1f}% | LADA+ZS: {lada_zs_acc:5.1f}%"
        logging.info(log_str)

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

    has_lada = lada_classifier is not None
    if has_lada:
        id_lada_avg = sum(id_lada_accs) / len(id_lada_accs) if id_lada_accs else 0.0
        id_lada_zs_avg = sum(id_lada_zs_accs) / len(id_lada_zs_accs) if id_lada_zs_accs else 0.0
        if len(ood_lada_accs) > 0:
            ood_lada_avg = sum(ood_lada_accs) / len(ood_lada_accs)
            ood_lada_zs_avg = sum(ood_lada_zs_accs) / len(ood_lada_zs_accs)
        else:
            ood_lada_avg = ood_lada_zs_avg = 0.0
        all_lada_accs = id_lada_accs + ood_lada_accs
        all_lada_zs_accs = id_lada_zs_accs + ood_lada_zs_accs
        total_lada_avg = sum(all_lada_accs) / len(all_lada_accs)
        total_lada_zs_avg = sum(all_lada_zs_accs) / len(all_lada_zs_accs)

    print("\n" + "=" * 110)
    print("JOINT FINE-TUNING RESULTS")
    print("=" * 110)

    # 构建表格头
    id_header = f"{'Dataset':<15s} | {'Zero-shot':>9s} | {'LR-RGDA':>9s} | {'Ensemble':>9s}"
    id_sep = "-" * 55
    if has_lada:
        id_header += f" | {'LADA':>9s} | {'LADA+ZS':>9s}"
        id_sep = "-" * 79

    # ID 数据集详细结果
    print("\n[ID Datasets — Fine-tuned]")
    print(id_header)
    print(id_sep)
    for i, d_name in enumerate(args.id_datasets):
        row = f"{d_name:<15s} | {id_zs_accs[i]:>7.1f}%  | {id_rgda_accs[i]:>7.1f}%  | {id_ens_accs[i]:>7.1f}%"
        if has_lada:
            row += f" | {id_lada_accs[i]:>7.1f}%  | {id_lada_zs_accs[i]:>7.1f}%"
        print(row)
    print(id_sep)
    avg_row = f"{'ID Average':<15s} | {id_zs_avg:>7.1f}%  | {id_rgda_avg:>7.1f}%  | {id_ens_avg:>7.1f}%"
    if has_lada:
        avg_row += f" | {id_lada_avg:>7.1f}%  | {id_lada_zs_avg:>7.1f}%"
    print(avg_row)

    # OOD 数据集详细结果
    if len(args.ood_datasets) > 0:
        print(f"\n[OOD Datasets — Non-fine-tuned]")
        print(id_header)
        print(id_sep)
        for i, d_name in enumerate(args.ood_datasets):
            row = f"{d_name:<15s} | {ood_zs_accs[i]:>7.1f}%  | {ood_rgda_accs[i]:>7.1f}%  | {ood_ens_accs[i]:>7.1f}%"
            if has_lada:
                row += f" | {ood_lada_accs[i]:>7.1f}%  | {ood_lada_zs_accs[i]:>7.1f}%"
            print(row)
        print(id_sep)
        avg_row = f"{'OOD Average':<15s} | {ood_zs_avg:>7.1f}%  | {ood_rgda_avg:>7.1f}%  | {ood_ens_avg:>7.1f}%"
        if has_lada:
            avg_row += f" | {ood_lada_avg:>7.1f}%  | {ood_lada_zs_avg:>7.1f}%"
        print(avg_row)
        print(id_sep)

    # 全部数据集总平均
    print(f"\n[All Datasets — Total Average]")
    total_row = f"{'All Average':<15s} | {total_zs_avg:>7.1f}%  | {total_rgda_avg:>7.1f}%  | {total_ens_avg:>7.1f}%"
    if has_lada:
        total_row += f" | {total_lada_avg:>7.1f}%  | {total_lada_zs_avg:>7.1f}%"
    print(total_row)
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
        "experiment_name": args.experiment_name,
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

    # 追加 LADA 指标（如果启用）
    if has_lada:
        save_results["metrics"]["id"]["per_dataset"]["lada"] = {d: v for d, v in zip(args.id_datasets, id_lada_accs)}
        save_results["metrics"]["id"]["per_dataset"]["lada_zs"] = {d: v for d, v in zip(args.id_datasets, id_lada_zs_accs)}
        save_results["metrics"]["id"]["average"]["lada"] = id_lada_avg
        save_results["metrics"]["id"]["average"]["lada_zs"] = id_lada_zs_avg
        save_results["metrics"]["ood"]["per_dataset"]["lada"] = {d: v for d, v in zip(args.ood_datasets, ood_lada_accs)}
        save_results["metrics"]["ood"]["per_dataset"]["lada_zs"] = {d: v for d, v in zip(args.ood_datasets, ood_lada_zs_accs)}
        save_results["metrics"]["ood"]["average"]["lada"] = ood_lada_avg
        save_results["metrics"]["ood"]["average"]["lada_zs"] = ood_lada_zs_avg
        save_results["metrics"]["total"]["per_dataset"]["lada"] = {d: v for d, v in zip(all_dataset_names, all_lada_accs)}
        save_results["metrics"]["total"]["per_dataset"]["lada_zs"] = {d: v for d, v in zip(all_dataset_names, all_lada_zs_accs)}
        save_results["metrics"]["total"]["average"]["lada"] = total_lada_avg
        save_results["metrics"]["total"]["average"]["lada_zs"] = total_lada_zs_avg

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

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_stem = args.experiment_name or f"joint_results_{timestamp}"
    save_path = os.path.join(args.output_dir, f"{result_stem}_seed{args.seed}.json")
    with open(save_path, 'w') as f:
        json.dump(save_results, f, indent=4)
    logging.info(f"\n联合微调结果已保存至: {save_path}")


if __name__ == "__main__":
    command_line_args = parse_args()
    main(command_line_args)
