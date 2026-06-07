"""
临时诊断脚本：对比 main_joint.py 和 debug_ensemble_alpha.py 的 RGDA 评估差异

测试项：
  1. Per-class avg global_cov vs per-dataset avg global_cov
  2. main_joint.py evaluate_dataset 逐数据集评估 vs 全量拼接评估

用法：
    python debug_rgda_eval_diff.py --id_datasets ALL --gpu 0
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--id_datasets", type=str, nargs='+', default=ALL_XTAIL_DATASETS)
    p.add_argument("--root", type=str, default="/data1/open_datasets/X-TAIL")
    p.add_argument("--num_shots", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    if len(args.id_datasets) == 1 and args.id_datasets[0].upper() == 'ALL':
        args.id_datasets = ALL_XTAIL_DATASETS
    if args.device is None:
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    return args


def build_classifier(train_feats, train_labels, args, num_centers, global_cov_mode):
    """global_cov_mode: 'per_class' | 'per_dataset'"""
    stats_dict, center_means = build_multi_center_stats_dict(
        train_feats.cpu(), train_labels.cpu(), M=num_centers
    )

    if global_cov_mode == 'per_class':
        global_cov = None  # LRRGDA will compute per-class average internally
    elif global_cov_mode == 'per_dataset':
        per_dataset_covs = []
        offset = 0
        for d_name in args.id_datasets:
            c_names = get_xtail_classnames(args.root, d_name, args.num_shots)
            n_classes = len(c_names)
            ds_cov = sum(stats_dict[cid].cov for cid in range(offset, offset + n_classes)) / n_classes
            per_dataset_covs.append(ds_cov)
            offset += n_classes
        global_cov = sum(per_dataset_covs) / len(per_dataset_covs)
    else:
        raise ValueError(f"Unknown mode: {global_cov_mode}")

    classifier = LRRGDAClassifier(
        stats_dict=stats_dict, device=args.device,
        rank=32, qda_reg_alpha1=0.2, qda_reg_alpha2=2.0, qda_reg_alpha3=0.5,
        temperature=1.0,
        M=num_centers, center_means=center_means,
        global_cov=global_cov,
    )
    return classifier


def eval_per_dataset(args, model, zs_classifier, lr_rgda_classifier, num_id_classes, dataset_slices):
    """模拟 main_joint.py 的 evaluate_dataset 逐数据集评估"""
    all_zs, all_rgda = [], []
    for i, d_name in enumerate(args.id_datasets):
        _, test_transform = get_transforms(d_name)
        _, _, te_loader, c_names = get_xtail_trainloader(
            root=args.root, dataset_name=d_name,
            transform_train=None, transform_test=test_transform,
            num_shots=args.num_shots, batch_size=args.batch_size
        )
        features, labels = extract_features(model, te_loader, args.device)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.to(args.device)

        start, end = dataset_slices[i]
        labels = (labels + start).to(args.device)

        with torch.no_grad():
            zs_logits = features @ zs_classifier
            zs_norm = zs_logits - zs_logits.max(dim=-1, keepdim=True).values
            zs_acc = zs_norm.argmax(dim=1).eq(labels).float().mean().item() * 100

            rgda_logits = lr_rgda_classifier.forward(features)
            rgda_norm = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values
            rgda_acc = rgda_norm.argmax(dim=1).eq(labels).float().mean().item() * 100

        all_zs.append(zs_acc)
        all_rgda.append(rgda_acc)

    return sum(all_zs) / len(all_zs), sum(all_rgda) / len(all_rgda), all_zs, all_rgda


def eval_merged(args, test_feats, test_labels, zs_classifier, lr_rgda_classifier, dataset_slices):
    """模拟 debug_ensemble_alpha.py 的全量拼接评估"""
    with torch.no_grad():
        zs_logits = test_feats @ zs_classifier
        rgda_logits = lr_rgda_classifier.forward(test_feats)

    zs_per_ds, rgda_per_ds = [], []
    for start, end in dataset_slices:
        zs_norm = zs_logits[start:end] - zs_logits[start:end].max(dim=-1, keepdim=True).values
        rgda_norm = rgda_logits[start:end] - rgda_logits[start:end].max(dim=-1, keepdim=True).values
        lbl = test_labels[start:end]
        zs_acc = zs_norm.argmax(dim=1).eq(lbl).float().mean().item() * 100
        rgda_acc = rgda_norm.argmax(dim=1).eq(lbl).float().mean().item() * 100
        zs_per_ds.append(zs_acc)
        rgda_per_ds.append(rgda_acc)

    return sum(zs_per_ds) / len(zs_per_ds), sum(rgda_per_ds) / len(rgda_per_ds), zs_per_ds, rgda_per_ds


def main(args):
    fix_random_seed(args.seed)
    logging.info(f"Device: {args.device}")

    logging.info("Loading frozen CLIP...")
    dummy_args = argparse.Namespace(
        lora_type='lora_vanilla', lora_rank=4,
        tune_vision_encoder=False, tune_text_encoder=False
    )
    model, processor = get_clip_model(dummy_args, train_mode='frozen')
    model = model.to(args.device)
    model.eval()

    # 提取所有特征（仅一次）
    logging.info("Extracting features...")
    all_train_feats, all_train_labels = [], []
    all_test_feats, all_test_labels = [], []
    all_class_names = []
    dataset_slices = []
    offset, test_offset = 0, 0

    for d_name in args.id_datasets:
        train_transform, test_transform = get_transforms(d_name)
        tr_loader, _, te_loader, c_names = get_xtail_trainloader(
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
        dataset_slices.append((test_offset, test_offset + te_feats.shape[0]))
        all_class_names.extend(c_names)
        offset += len(c_names)
        test_offset += te_feats.shape[0]

    train_feats = torch.cat(all_train_feats)
    train_labels = torch.cat(all_train_labels)
    test_feats = torch.cat(all_test_feats)
    test_labels = torch.cat(all_test_labels)
    num_id_classes = len(all_class_names)

    train_feats_norm = (train_feats / train_feats.norm(dim=-1, keepdim=True)).to(args.device)
    test_feats_norm = (test_feats / test_feats.norm(dim=-1, keepdim=True)).to(args.device)
    train_labels = train_labels.to(args.device)
    test_labels = test_labels.to(args.device)

    logging.info(f"Total: {num_id_classes} classes, train={train_feats.shape[0]}, test={test_feats.shape[0]}")

    zs_classifier = get_zeroshot_classifier(model, processor, all_class_names, args.device)

    print("\n" + "=" * 100)
    print("对比实验：main_joint.py vs debug_ensemble_alpha.py 的 RGDA 评估")
    print("=" * 100)

    print(f"\n{'Config':<35s} | {'ZS':>7s} | {'RGDA':>7s}")
    print("-" * 55)

    for n_centers in [1, 4]:
        for cov_mode in ['per_class', 'per_dataset']:
            label = f"M={n_centers}, global_cov={cov_mode}"
            classifier = build_classifier(
                train_feats_norm, train_labels, args, num_centers=n_centers, global_cov_mode=cov_mode
            )

            # 方法 A: 逐数据集评估（模拟 main_joint.py）
            zs_a, rgda_a, zs_a_list, rgda_a_list = eval_per_dataset(
                args, model, zs_classifier, classifier, num_id_classes, dataset_slices
            )

            # 方法 B: 全量拼接评估（模拟 debug_ensemble_alpha.py）
            zs_b, rgda_b, zs_b_list, rgda_b_list = eval_merged(
                args, test_feats_norm, test_labels, zs_classifier, classifier, dataset_slices
            )

            print(f"{label:<35s} (per-ds) | {zs_a:>5.1f}% | {rgda_a:>5.1f}%")
            print(f"{label:<35s} (merged) | {zs_b:>5.1f}% | {rgda_b:>5.1f}%")
            print("-" * 55)

    # Per-dataset 详细对比
    print("\n" + "=" * 100)
    print("Per-dataset 详细对比（M=1, per-dataset global_cov）")
    print("=" * 100)

    classifier_ds = build_classifier(
        train_feats_norm, train_labels, args, num_centers=1, global_cov_mode='per_dataset'
    )
    classifier_cls = build_classifier(
        train_feats_norm, train_labels, args, num_centers=1, global_cov_mode='per_class'
    )

    _, _, _, rgda_ds_list = eval_merged(args, test_feats_norm, test_labels, zs_classifier, classifier_ds, dataset_slices)
    _, _, _, rgda_cls_list = eval_merged(args, test_feats_norm, test_labels, zs_classifier, classifier_cls, dataset_slices)

    print(f"{'Dataset':<15s} | {'RGDA per_class':>12s} | {'RGDA per_dataset':>12s} | {'Diff':>8s}")
    print("-" * 55)
    for i, d_name in enumerate(args.id_datasets):
        diff = rgda_ds_list[i] - rgda_cls_list[i]
        print(f"{d_name:<15s} | {rgda_cls_list[i]:>10.1f}%  | {rgda_ds_list[i]:>11.1f}%  | {diff:>+7.1f}%")
    print("-" * 55)
    print(f"{'Average':<15s} | {sum(rgda_cls_list)/len(rgda_cls_list):>10.1f}%  | "
          f"{sum(rgda_ds_list)/len(rgda_ds_list):>11.1f}%  | "
          f"{sum(rgda_ds_list)/len(rgda_ds_list) - sum(rgda_cls_list)/len(rgda_cls_list):>+7.1f}%")


if __name__ == "__main__":
    args = parse_args()
    main(args)
