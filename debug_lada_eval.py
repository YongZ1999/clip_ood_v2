"""
LADA 分类器评估脚本 v1

实验 1：6 种 LADA 配置的 alpha sweep（加权混合 ZS+LADA）
  A/B/C: 分析版（k=1/4/16 centers per class）
  D/E/F: 微调版（k=1/4/16 + gradient fit 200 iter）
实验 2：Logit 分布统计
实验 3：两种集成方式对比（加权混合 vs 加法+指数变换）
实验 4：train_transform vs test_transform 双分支对比

指标：per-dataset 平均准确率（10 个数据集等权平均）

用法：
    python debug_lada_eval.py --id_datasets ALL --gpu 0
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import torch.nn.functional as F
import argparse
import logging

from src.models.clip import get_clip_model
from src.lada import LADAClassifier
from src.utils.feature_extractor import extract_features
from src.utils.main_utils import fix_random_seed, get_zeroshot_classifier
from utils_data import get_xtail_trainloader, get_transforms

ALL_XTAIL_DATASETS = [
    "aircraft", "caltech101", "dtd", "eurosat", "flowers",
    "food101", "mnist", "oxford_pets", "stanford_cars", "sun397"
]


def parse_args():
    parser = argparse.ArgumentParser(description="LADA Evaluation")
    parser.add_argument("--id_datasets", type=str, nargs='+', default=ALL_XTAIL_DATASETS)
    parser.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL")
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lada_beta", type=float, default=1.0)
    parser.add_argument("--lada_train_iter", type=int, default=200)
    parser.add_argument("--lada_train_lr", type=float, default=0.01)
    parser.add_argument("--n_alpha", type=int, default=41)
    parser.add_argument("--beta_values", type=float, nargs='+', default=[0.5, 1.0, 2.0, 5.0])
    args = parser.parse_args()

    if len(args.id_datasets) == 1 and args.id_datasets[0].upper() == 'ALL':
        args.id_datasets = ALL_XTAIL_DATASETS

    if args.device is None:
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    return args


def build_lada_classifier(features, labels, args, k, do_fit):
    """构建 LADA 分类器（k-means 聚类中心 + 指数亲和变换），可选梯度微调"""
    feature_dim = features.shape[1]

    classifier = LADAClassifier(feature_dim=feature_dim, beta=args.lada_beta)
    classifier.build_from_data(features.to(args.device), labels.to(args.device), k=k, label_offset=0)
    classifier.to(args.device)

    if do_fit and args.lada_train_iter > 0:
        classifier.fit(features.to(args.device), labels.to(args.device),
                       iterations=args.lada_train_iter, lr=args.lada_train_lr,
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


def per_dataset_acc(logits, labels, dataset_slices, num_id_classes, alpha, method="weighted", beta=1.0):
    zs_norm = logits[:, :num_id_classes] - logits[:, :num_id_classes].max(dim=-1, keepdim=True).values
    lada_norm = logits[:, num_id_classes:] - logits[:, num_id_classes:].max(dim=-1, keepdim=True).values

    accs = []
    for start, end in dataset_slices:
        zs = zs_norm[start:end]
        lada = lada_norm[start:end]
        lbl = labels[start:end]

        if method == "weighted":
            ens = zs * (1 - alpha) + alpha * lada
        elif method == "additive_exp":
            lada_t = torch.exp(-beta * (1 - lada))
            ens = zs + alpha * lada_t
        else:
            raise ValueError(f"Unknown method: {method}")

        acc = ens.argmax(dim=1).eq(lbl).float().mean().item() * 100
        accs.append(acc)
    return sum(accs) / len(accs)


def alpha_sweep_per_dataset(combined_logits, test_labels, dataset_slices,
                            num_id_classes, alphas, method="weighted", beta=1.0):
    results = []
    for a in alphas:
        acc = per_dataset_acc(combined_logits, test_labels, dataset_slices,
                              num_id_classes, a, method, beta)
        results.append((round(a, 4), round(acc, 2)))
    return results


def _extract_features_for_branch(model, args, all_class_names, dataset_slices, use_test_transform):
    all_train_feats, all_train_labels = [], []
    all_test_feats, all_test_labels = [], []
    all_cn = []
    ds_slices = []
    offset = 0
    test_offset = 0

    for d_name in args.id_datasets:
        train_transform, test_transform = get_transforms(d_name)
        tr_transform = test_transform if use_test_transform else train_transform
        tr_loader, _, te_loader, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name,
            transform_train=tr_transform, transform_test=test_transform,
            num_shots=args.num_shots, batch_size=args.batch_size
        )
        tr_feats, tr_lbls = extract_features(model, tr_loader, args.device)
        te_feats, te_lbls = extract_features(model, te_loader, args.device)
        all_train_feats.append(tr_feats)
        all_train_labels.append(tr_lbls + offset)
        all_test_feats.append(te_feats)
        all_test_labels.append(te_lbls + offset)
        ds_slices.append((test_offset, test_offset + te_feats.shape[0]))
        all_cn.extend(c_names)
        offset += len(c_names)
        test_offset += te_feats.shape[0]
        logging.info(f"  {d_name}: train={tr_feats.shape[0]}, test={te_feats.shape[0]}, classes={len(c_names)}")

    train_feats_raw = torch.cat(all_train_feats)
    train_labels_gpu = torch.cat(all_train_labels).to(args.device)
    test_feats = torch.cat(all_test_feats)
    test_labels = torch.cat(all_test_labels)

    train_feats_norm = (train_feats_raw / train_feats_raw.norm(dim=-1, keepdim=True)).to(args.device)
    test_feats_norm = (test_feats / test_feats.norm(dim=-1, keepdim=True)).to(args.device)
    test_labels_gpu = test_labels.to(args.device)
    num_id_classes = len(all_cn)

    all_class_names[:] = all_cn
    dataset_slices[:] = ds_slices

    return (train_feats_norm, train_labels_gpu,
            test_feats_norm, test_labels_gpu, num_id_classes)


def _build_and_eval_lada(args, branch_label, train_feats_norm, train_labels_gpu,
                          test_feats_norm, test_labels_gpu,
                          zs_test_logits, num_id_classes,
                          dataset_slices, zs_per_ds, zs_avg, configs, alphas):
    results = {}

    for cfg_name, k, do_fit in configs:
        logging.info(f"\n{'='*60}")
        logging.info(f"[{branch_label}] Config: {cfg_name}")
        logging.info(f"  k={k}, fit={do_fit}")

        classifier = build_lada_classifier(
            train_feats_norm, train_labels_gpu, args, k=k, do_fit=do_fit
        )

        lada_logits_list = []
        lada_per_ds = []
        with torch.no_grad():
            for start, end in dataset_slices:
                batch = test_feats_norm[start:end]
                lada_l = classifier(batch)
                lada_l_norm = lada_l - lada_l.max(dim=-1, keepdim=True).values
                acc = lada_l_norm.argmax(dim=1).eq(test_labels_gpu[start:end]).float().mean().item() * 100
                lada_per_ds.append(acc)
                lada_logits_list.append(lada_l.cpu())
                del lada_l, lada_l_norm
                torch.cuda.empty_cache()

        lada_test_logits = torch.cat(lada_logits_list).to(args.device)
        lada_avg = sum(lada_per_ds) / len(lada_per_ds)

        combined_logits = torch.cat([zs_test_logits, lada_test_logits], dim=1)

        sweep_w = alpha_sweep_per_dataset(
            combined_logits, test_labels_gpu, dataset_slices,
            num_id_classes, alphas, method="weighted"
        )
        best_w = max(sweep_w, key=lambda x: x[1])

        sweep_ae = {}
        best_ae = {}
        for beta in args.beta_values:
            sw = alpha_sweep_per_dataset(
                combined_logits, test_labels_gpu, dataset_slices,
                num_id_classes, alphas, method="additive_exp", beta=beta
            )
            sweep_ae[beta] = sw
            best_ae[beta] = max(sw, key=lambda x: x[1])

        lada_stats = compute_logit_stats(lada_test_logits, "LADA")

        results[cfg_name] = {
            "lada_avg": lada_avg,
            "lada_per_ds": lada_per_ds,
            "weighted": {"sweep": sweep_w, "best": best_w},
            "additive_exp": {"sweep": sweep_ae, "best": best_ae},
            "lada_stats": lada_stats,
        }

        del lada_logits_list, lada_test_logits, combined_logits, classifier
        torch.cuda.empty_cache()

    return results


def _print_all_tables(branch_label, configs, results, zs_avg, zs_per_ds, zs_test_logits, args):
    def print_section(tag):
        print(f"\n{'=' * 120}")
        print(f"=== {branch_label} — {tag} ===")
        print(f"{'=' * 120}")

    # --- 实验 1 ---
    print_section("实验 1")
    print(f"\n{'Config':<25s} | {'LADA':>7s} | {'Best α':>8s} | {'Best Ens':>8s} | {'α=0.5':>8s} | {'ZS avg':>7s}")
    print("-" * 120)
    for cfg_name, _, _ in configs:
        r = results[cfg_name]
        best_a, best_acc = r["weighted"]["best"]
        sweep_dict = dict(r["weighted"]["sweep"])
        closest_05 = min(sweep_dict.keys(), key=lambda x: abs(x - 0.5))
        acc_05 = sweep_dict[closest_05]
        print(f"{cfg_name:<25s} | {r['lada_avg']:>5.1f}%  | {best_a:>7.3f}  | {best_acc:>6.1f}%  | {acc_05:>6.1f}%  | {zs_avg:>5.1f}%")

    # --- 实验 2 ---
    print_section("实验 2")
    print(f"\n{'Config':<25s} | {'Classifier':>10s} | {'Entropy':>8s} | {'Margin':>8s} | {'Std':>8s} | {'Spread':>8s}")
    print("-" * 120)
    print(f"{'ZS':<25s} | {'ZS':>10s} | {compute_logit_stats(zs_test_logits, 'ZS')['entropy']:>8.3f} | "
          f"{compute_logit_stats(zs_test_logits, 'ZS')['margin']:>8.4f} | "
          f"{compute_logit_stats(zs_test_logits, 'ZS')['std']:>8.4f} | "
          f"{compute_logit_stats(zs_test_logits, 'ZS')['spread']:>8.2f}")
    for cfg_name, _, _ in configs:
        s = results[cfg_name]["lada_stats"]
        print(f"{cfg_name:<25s} | {s['name']:>10s} | {s['entropy']:>8.3f} | {s['margin']:>8.4f} | {s['std']:>8.4f} | {s['spread']:>8.2f}")

    # --- 实验 3 ---
    print_section("实验 3")
    print(f"\n{'Config':<25s} | {'Method':<25s} | {'Best α':>8s} | {'Best Acc':>8s} | {'α=0.5':>8s}")
    print("-" * 120)
    for cfg_name, _, _ in configs:
        r = results[cfg_name]
        best_a, best_acc = r["weighted"]["best"]
        sweep_dict = dict(r["weighted"]["sweep"])
        closest_05 = min(sweep_dict.keys(), key=lambda x: abs(x - 0.5))
        acc_05 = sweep_dict[closest_05]
        print(f"{cfg_name:<25s} | {'加权混合':<25s} | {best_a:>7.3f}  | {best_acc:>6.1f}%  | {acc_05:>6.1f}%")

        for beta in args.beta_values:
            ba, bac = r["additive_exp"]["best"][beta]
            sd = dict(r["additive_exp"]["sweep"][beta])
            c05 = min(sd.keys(), key=lambda x: abs(x - 0.5))
            a05 = sd[c05]
            print(f"{cfg_name:<25s} | {'加法+exp(β='+str(beta)+')':<25s} | {ba:>7.3f}  | {bac:>6.1f}%  | {a05:>6.1f}%")
        print("-" * 120)

    # --- 详细 Alpha Sweep ---
    print_section("详细 Alpha Sweep")
    header = f"\n{'Alpha':>8s}"
    for cfg_name, _, _ in configs:
        header += f" | {cfg_name:>18s}"
    print(header)
    print("-" * 120)

    key_alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for a in key_alphas:
        row = f"{a:>8.2f}"
        for cfg_name, _, _ in configs:
            sweep_dict = dict(results[cfg_name]["weighted"]["sweep"])
            closest_a = min(sweep_dict.keys(), key=lambda x: abs(x - a))
            acc = sweep_dict[closest_a]
            row += f" | {acc:>16.1f}%"
        print(row)

    # --- Per-dataset 详细表 ---
    print_section("Per-dataset 详细准确率")
    header2 = f"\n{'Dataset':<15s} | {'ZS':>7s}"
    for cfg_name, _, _ in configs:
        header2 += f" | {cfg_name:>14s}"
    print(header2)
    print("-" * 120)
    for i, d_name in enumerate(args.id_datasets):
        row = f"{d_name:<15s} | {zs_per_ds[i]:>5.1f}%"
        for cfg_name, _, _ in configs:
            ld = results[cfg_name]["lada_per_ds"][i]
            row += f" | {ld:>12.1f}%"
        print(row)
    print("-" * 120)
    row_avg = f"{'Average':<15s} | {zs_avg:>5.1f}%"
    for cfg_name, _, _ in configs:
        row_avg += f" | {results[cfg_name]['lada_avg']:>12.1f}%"
    print(row_avg)

    # --- 最佳加法+指数变换 ---
    print_section("最佳加法+指数变换配置")
    for cfg_name, _, _ in configs:
        r = results[cfg_name]
        best_beta = None
        best_overall = -1
        for beta in args.beta_values:
            _, acc = r["additive_exp"]["best"][beta]
            if acc > best_overall:
                best_overall = acc
                best_beta = beta
        ba, bac = r["additive_exp"]["best"][best_beta]
        print(f"{cfg_name:<25s} | β={best_beta:.1f}, α={ba:.3f}, Acc={bac:.1f}%")


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

    all_class_names = []
    dataset_slices = []

    configs = [
        ("A: k=1 analytical",       1,  False),
        ("B: k=4 analytical",       4,  False),
        ("C: k=16 analytical",      16, False),
        ("D: k=1 fit",              1,  True),
        ("E: k=4 fit",              4,  True),
        ("F: k=16 fit",             16, True),
    ]
    alphas = torch.linspace(0, 1.0, args.n_alpha).tolist()

    branches = [
        ("train_transform", False),
        ("test_transform", True),
    ]

    zs_classifier = None
    zs_test_logits = None
    zs_per_ds = None
    zs_avg = None
    all_branch_results = {}

    for branch_label, use_test_transform in branches:
        logging.info(f"\n{'#'*60}")
        logging.info(f"Branch: {branch_label}")
        logging.info(f"{'#'*60}")

        (train_feats_norm, train_labels_gpu,
         test_feats_norm, test_labels_gpu, ncid) = _extract_features_for_branch(
            model, args, all_class_names, dataset_slices, use_test_transform
        )
        num_id_classes = ncid

        if zs_classifier is None:
            zs_classifier = get_zeroshot_classifier(model, processor, all_class_names, args.device)
            with torch.no_grad():
                zs_test_logits = test_feats_norm @ zs_classifier

            zs_per_ds = []
            for start, end in dataset_slices:
                zs_l = zs_test_logits[start:end]
                zs_l_norm = zs_l - zs_l.max(dim=-1, keepdim=True).values
                acc = zs_l_norm.argmax(dim=1).eq(test_labels_gpu[start:end]).float().mean().item() * 100
                zs_per_ds.append(acc)
            zs_avg = sum(zs_per_ds) / len(zs_per_ds)
            logging.info(f"Zero-shot per-dataset avg: {zs_avg:.1f}%")
            test_feats_shared = test_feats_norm
            test_labels_shared = test_labels_gpu
        else:
            test_feats_norm = test_feats_shared
            test_labels_gpu = test_labels_shared

        branch_results = _build_and_eval_lada(
            args, branch_label, train_feats_norm, train_labels_gpu,
            test_feats_norm, test_labels_gpu, zs_test_logits, num_id_classes,
            dataset_slices, zs_per_ds, zs_avg, configs, alphas
        )
        all_branch_results[branch_label] = branch_results

        _print_all_tables(branch_label, configs, branch_results, zs_avg, zs_per_ds, zs_test_logits, args)

        del train_feats_norm, train_labels_gpu
        torch.cuda.empty_cache()

    # ========== 最终对比总结 ==========
    print("\n\n" + "=" * 120)
    print("=== 双分支对比总结 ===")
    print("=" * 120)
    print(f"\n{'Config':<25s} | {'LADA(train_t)':>13s} | {'LADA(test_t)':>13s} | {'Diff':>8s}")
    print("-" * 75)
    train_r = all_branch_results.get("train_transform", {})
    test_r = all_branch_results.get("test_transform", {})
    for cfg_name, _, _ in configs:
        tv = train_r.get(cfg_name, {}).get("lada_avg", 0)
        sv = test_r.get(cfg_name, {}).get("lada_avg", 0)
        print(f"{cfg_name:<25s} | {tv:>11.1f}%  | {sv:>12.1f}%  | {sv-tv:>+7.1f}%")


if __name__ == "__main__":
    args = parse_args()
    main(args)
