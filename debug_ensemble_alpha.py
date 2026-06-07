"""
集成分类器 Alpha 失调诊断脚本

实验目的：研究微调后 LR-RGDA 分类器与零样本分类器的集成行为

实验 1：4 种 RGDA 配置的 alpha sweep（加权混合）
实验 2：Logit 分布统计（margin, entropy, std）
实验 3：两种集成方式对比（加权混合 vs 加法+指数变换）

用法：
    python debug_ensemble_alpha.py --id_datasets ALL --gpu 0
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import torch.nn.functional as F
import argparse
import logging
import numpy as np

from src.models.clip import get_clip_model
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.classifiers.gaussian_statistics import build_multi_center_stats_dict
from src.utils.feature_extractor import extract_features
from src.utils.main_utils import fix_random_seed, get_zeroshot_classifier
from utils_data import get_xtail_trainloader, get_xtail_classnames, get_transforms

ALL_XTAIL_DATASETS = [
    "aircraft", "caltech101", "dtd", "eurosat", "flowers",
    "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Alpha Diagnosis")
    parser.add_argument("--id_datasets", type=str, nargs='+', default=ALL_XTAIL_DATASETS)
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL")
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rgda_rank", type=int, default=32)
    parser.add_argument("--rgda_alpha1", type=float, default=0.2)
    parser.add_argument("--rgda_alpha2", type=float, default=2.0)
    parser.add_argument("--rgda_alpha3", type=float, default=0.5)
    parser.add_argument("--rgda_train_iter", type=int, default=200)
    parser.add_argument("--rgda_train_lr", type=float, default=0.01)
    parser.add_argument("--n_alpha", type=int, default=41,
                        help="alpha 扫描点数 (0 到 1 之间等分)")
    parser.add_argument("--beta_values", type=float, nargs='+', default=[0.5, 1.0, 2.0, 5.0],
                        help="指数变换的 β 值列表")
    args = parser.parse_args()

    if len(args.id_datasets) == 1 and args.id_datasets[0].upper() == 'ALL':
        args.id_datasets = ALL_XTAIL_DATASETS

    if args.device is None:
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    return args


def build_rgda_classifier(train_feats, train_labels, args, num_centers, do_fit):
    stats_dict, center_means = build_multi_center_stats_dict(
        train_feats, train_labels, M=num_centers
    )

    per_dataset_covs = []
    offset = 0
    for d_name in args.id_datasets:
        c_names = get_xtail_classnames(args.root, d_name, args.num_shots)
        n_classes = len(c_names)
        ds_cov = sum(stats_dict[cid].cov for cid in range(offset, offset + n_classes)) / n_classes
        per_dataset_covs.append(ds_cov)
        offset += n_classes
    global_cov = sum(per_dataset_covs) / len(per_dataset_covs)

    classifier = LRRGDAClassifier(
        stats_dict=stats_dict, device=args.device,
        rank=args.rgda_rank,
        qda_reg_alpha1=args.rgda_alpha1,
        qda_reg_alpha2=args.rgda_alpha2,
        qda_reg_alpha3=args.rgda_alpha3,
        temperature=1.0,
        M=num_centers, center_means=center_means,
        global_cov=global_cov,
    )

    if do_fit:
        classifier.fit(train_feats.to(args.device), train_labels.to(args.device),
                       iterations=args.rgda_train_iter, lr=args.rgda_train_lr,
                       verbose=False)

    return classifier


def compute_logit_stats(logits, name):
    logits_norm = logits - logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(logits_norm, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()

    top2, _ = logits_norm.topk(2, dim=-1)
    margin = (top2[:, 0] - top2[:, 1]).mean().item()

    std = logits_norm.std(dim=-1).mean().item()
    max_val = logits.max(dim=-1).values.mean().item()
    min_val = logits.min(dim=-1).values.mean().item()
    spread = max_val - min_val

    return {
        "name": name,
        "entropy": entropy,
        "margin": margin,
        "std": std,
        "spread": spread,
    }


def evaluate_ensemble_weighted(zs_logits, rgda_logits, labels, num_id_classes, alpha):
    zs_norm = zs_logits - zs_logits.max(dim=-1, keepdim=True).values
    rgda_norm = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values
    ens = zs_norm * (1 - alpha)
    ens[:, :num_id_classes] += alpha * rgda_norm
    preds = ens.argmax(dim=1)
    return preds.eq(labels).float().mean().item() * 100


def evaluate_ensemble_additive_exp(zs_logits, rgda_logits, labels, num_id_classes, alpha, beta):
    zs_norm = zs_logits - zs_logits.max(dim=-1, keepdim=True).values
    rgda_norm = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values
    rgda_transformed = torch.exp(-beta * (1 - rgda_norm))
    ens = zs_norm + alpha * rgda_transformed
    preds = ens.argmax(dim=1)
    return preds.eq(labels).float().mean().item() * 100


def alpha_sweep_weighted(zs_logits, rgda_logits, labels, num_id_classes, alphas):
    results = []
    for a in alphas:
        acc = evaluate_ensemble_weighted(zs_logits, rgda_logits, labels, num_id_classes, a)
        results.append((round(a, 4), round(acc, 2)))
    return results


def alpha_sweep_additive_exp(zs_logits, rgda_logits, labels, num_id_classes, alphas, beta):
    results = []
    for a in alphas:
        acc = evaluate_ensemble_additive_exp(zs_logits, rgda_logits, labels, num_id_classes, a, beta)
        results.append((round(a, 4), round(acc, 2)))
    return results


def main(args):
    fix_random_seed(args.seed)
    logging.info(f"Device: {args.device}")

    logging.info("Loading CLIP model (frozen)...")
    dummy_args = argparse.Namespace(
        lora_type='lora_vanilla', lora_rank=4,
        tune_vision_encoder=False, tune_text_encoder=False
    )
    model, processor = get_clip_model(dummy_args, train_mode='frozen')
    model = model.to(args.device)
    model.eval()

    logging.info("Extracting features (once, cached)...")
    all_train_feats, all_train_labels = [], []
    all_test_feats, all_test_labels = [], []
    all_class_names = []
    offset = 0

    for d_name in args.id_datasets:
        train_transform, test_transform = get_transforms(d_name)
        tr_loader, te_loader, _, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name,
            transform_train=train_transform, transform_test=test_transform,
            num_shots=args.num_shots, batch_size=args.batch_size
        )

        tr_feats, tr_lbls = extract_features(model, tr_loader, args.device)
        te_feats, te_lbls = extract_features(model, te_loader, args.device)

        all_train_feats.append(tr_feats)
        all_train_labels.append(tr_lbls + offset)
        all_test_feats.append(te_feats)
        all_test_labels.append(te_lbls + offset)
        all_class_names.extend(c_names)
        offset += len(c_names)

        logging.info(f"  {d_name}: train={tr_feats.shape[0]}, test={te_feats.shape[0]}, classes={len(c_names)}")

    train_feats = torch.cat(all_train_feats)
    train_labels = torch.cat(all_train_labels)
    test_feats = torch.cat(all_test_feats)
    test_labels = torch.cat(all_test_labels)
    num_id_classes = len(all_class_names)

    train_feats_norm = train_feats / train_feats.norm(dim=-1, keepdim=True)
    test_feats_norm = test_feats / test_feats.norm(dim=-1, keepdim=True)
    train_feats_norm = train_feats_norm.to(args.device)
    test_feats_norm = test_feats_norm.to(args.device)
    train_labels = train_labels.to(args.device)
    test_labels = test_labels.to(args.device)

    logging.info("Building zero-shot classifier...")
    zs_classifier = get_zeroshot_classifier(model, processor, all_class_names, args.device)

    with torch.no_grad():
        zs_test_logits = test_feats_norm @ zs_classifier

    zs_preds = (zs_test_logits - zs_test_logits.max(dim=-1, keepdim=True).values).argmax(dim=1)
    zs_acc = zs_preds.eq(test_labels).float().mean().item() * 100
    logging.info(f"Zero-shot accuracy: {zs_acc:.1f}%")

    configs = [
        ("A: 1-center analytical", 1, False),
        ("B: 4-center analytical", 4, False),
        ("C: 1-center fine-tuned", 1, True),
        ("D: 4-center fine-tuned", 4, True),
    ]

    alphas = torch.linspace(0, 1.0, args.n_alpha).tolist()

    with torch.no_grad():
        rgda_results = {}

        for cfg_name, n_centers, do_fit in configs:
            logging.info(f"\nBuilding {cfg_name}...")
            classifier = build_rgda_classifier(
                train_feats_norm, train_labels, args, n_centers, do_fit
            )
            rgda_test_logits = classifier.forward(test_feats_norm)

            rgda_preds = (rgda_test_logits - rgda_test_logits.max(dim=-1, keepdim=True).values).argmax(dim=1)
            rgda_acc = rgda_preds.eq(test_labels).float().mean().item() * 100

            sweep_weighted = alpha_sweep_weighted(
                zs_test_logits, rgda_test_logits, test_labels,
                num_id_classes, alphas
            )
            best_weighted = max(sweep_weighted, key=lambda x: x[1])

            best_additive_exp = {}
            sweep_additive_exp = {}
            for beta in args.beta_values:
                sweep = alpha_sweep_additive_exp(
                    zs_test_logits, rgda_test_logits, test_labels,
                    num_id_classes, alphas, beta
                )
                best = max(sweep, key=lambda x: x[1])
                sweep_additive_exp[beta] = sweep
                best_additive_exp[beta] = best

            zs_stats = compute_logit_stats(zs_test_logits, "ZS")
            rgda_stats = compute_logit_stats(rgda_test_logits, "RGDA")

            rgda_results[cfg_name] = {
                "rgda_acc": rgda_acc,
                "weighted": {"sweep": sweep_weighted, "best": best_weighted},
                "additive_exp": {"sweep": sweep_additive_exp, "best": best_additive_exp},
                "zs_stats": zs_stats,
                "rgda_stats": rgda_stats,
            }

    print("\n" + "=" * 110)
    print("实验 1：Alpha Sweep — 4 种 RGDA 配置（加权混合: (1-α)*ZS + α*RGDA）")
    print("=" * 110)
    print(f"{'Config':<30s} | {'RGDA':>7s} | {'Best α':>8s} | {'Best Ens':>8s} | {'α=0.5':>8s} | {'ZS':>7s}")
    print("-" * 110)
    for cfg_name, _, _ in configs:
        r = rgda_results[cfg_name]
        best_a, best_acc = r["weighted"]["best"]
        acc_05 = dict(r["weighted"]["sweep"]).get(0.5, "N/A")
        if isinstance(acc_05, float):
            acc_05 = f"{acc_05:.1f}"
        print(f"{cfg_name:<30s} | {r['rgda_acc']:>5.1f}%  | {best_a:>7.3f}  | {best_acc:>6.1f}%  | {acc_05:>6s}%  | {zs_acc:>5.1f}%")

    print("\n" + "=" * 110)
    print("实验 2：Logit 分布统计")
    print("=" * 110)
    print(f"{'Config':<30s} | {'Classifier':>10s} | {'Entropy':>8s} | {'Margin':>8s} | {'Std':>8s} | {'Spread':>8s}")
    print("-" * 110)
    for cfg_name, _, _ in configs:
        r = rgda_results[cfg_name]
        for stats_key in ["zs_stats", "rgda_stats"]:
            s = r[stats_key]
            print(f"{cfg_name:<30s} | {s['name']:>10s} | {s['entropy']:>8.3f} | {s['margin']:>8.4f} | {s['std']:>8.4f} | {s['spread']:>8.2f}")
        print("-" * 110)

    print("\n" + "=" * 110)
    print("实验 3：两种集成方式对比")
    print("=" * 110)
    print(f"{'Config':<30s} | {'Method':<25s} | {'Best α':>8s} | {'Best Acc':>8s} | {'α=0.5':>8s}")
    print("-" * 110)
    for cfg_name, _, _ in configs:
        r = rgda_results[cfg_name]
        best_a, best_acc = r["weighted"]["best"]
        acc_05 = dict(r["weighted"]["sweep"]).get(0.5, "N/A")
        if isinstance(acc_05, float):
            acc_05 = f"{acc_05:.1f}"
        print(f"{cfg_name:<30s} | {'加权混合':<25s} | {best_a:>7.3f}  | {best_acc:>6.1f}%  | {acc_05:>6s}%")

        for beta in args.beta_values:
            best_a, best_acc = r["additive_exp"]["best"][beta]
            acc_05 = dict(r["additive_exp"]["sweep"][beta]).get(0.5, "N/A")
            if isinstance(acc_05, float):
                acc_05 = f"{acc_05:.1f}"
            method_str = f"加法+exp(β={beta})"
            print(f"{cfg_name:<30s} | {method_str:<25s} | {best_a:>7.3f}  | {best_acc:>6.1f}%  | {acc_05:>6s}%")
        print("-" * 110)

    print("\n" + "=" * 110)
    print("详细 Alpha Sweep（加权混合）")
    print("=" * 110)
    header = f"{'Alpha':>8s}"
    for cfg_name, _, _ in configs:
        short = cfg_name.split(":")[1].strip()
        header += f" | {short:>20s}"
    print(header)
    print("-" * 110)

    key_alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for a in key_alphas:
        row = f"{a:>8.2f}"
        for cfg_name, _, _ in configs:
            sweep_dict = dict(rgda_results[cfg_name]["weighted"]["sweep"])
            closest_a = min(sweep_dict.keys(), key=lambda x: abs(x - a))
            acc = sweep_dict[closest_a]
            row += f" | {acc:>18.1f}%"
        print(row)

    print("\n" + "=" * 110)
    print("最佳加法+指数变换配置")
    print("=" * 110)
    for cfg_name, _, _ in configs:
        r = rgda_results[cfg_name]
        best_beta = None
        best_overall = -1
        for beta in args.beta_values:
            _, acc = r["additive_exp"]["best"][beta]
            if acc > best_overall:
                best_overall = acc
                best_beta = beta
        best_a, best_acc = r["additive_exp"]["best"][best_beta]
        print(f"{cfg_name:<30s} | β={best_beta:.1f}, α={best_a:.3f}, Acc={best_acc:.1f}%")


if __name__ == "__main__":
    args = parse_args()
    main(args)
