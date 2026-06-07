import os, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import torch, argparse, logging, numpy as np
from src.models.clip import get_clip_model
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.classifiers.gaussian_statistics import build_multi_center_stats_dict
from src.utils.feature_extractor import extract_features
from src.utils.main_utils import fix_random_seed, get_zeroshot_classifier
from utils_data import get_xtail_trainloader, get_xtail_classnames, get_transforms

ALL_DS = ["aircraft","caltech101","dtd","eurosat","flowers","food101","mnist","oxford_pets","stanford_cars","sun397"]
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

fix_random_seed(42)
device = "cuda:5"

dummy_args = argparse.Namespace(lora_type="lora_vanilla", lora_rank=4, tune_vision_encoder=False, tune_text_encoder=False)
model, processor = get_clip_model(dummy_args, train_mode="frozen")
model = model.to(device)
model.eval()

all_train_feats, all_train_labels, all_test_feats, all_test_labels = [], [], [], []
all_class_names, dataset_slices = [], []
offset, test_offset = 0, 0

for d_name in ALL_DS:
    train_transform, test_transform = get_transforms(d_name)
    tr_loader, _, te_loader, c_names = get_xtail_trainloader(root="/data1/open_datasets/X-TAIL", dataset_name=d_name, transform_train=train_transform, transform_test=test_transform, num_shots=16, batch_size=32)
    tr_feats, tr_lbls = extract_features(model, tr_loader, device)
    te_feats, te_lbls = extract_features(model, te_loader, device)
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
num_classes = len(all_class_names)

train_feats_n = (train_feats / train_feats.norm(dim=-1, keepdim=True)).to(device)
test_feats_n = (test_feats / test_feats.norm(dim=-1, keepdim=True)).to(device)
train_labels = train_labels.to(device)
test_labels = test_labels.to(device)

zs_cls = get_zeroshot_classifier(model, processor, all_class_names, device)

stats_cls, _ = build_multi_center_stats_dict(train_feats_n.cpu(), train_labels.cpu(), M=1)

# per-class avg global_cov (None = computed internally)
clf_per_class = LRRGDAClassifier(stats_dict=stats_cls, device=device, rank=32,
    qda_reg_alpha1=0.2, qda_reg_alpha2=2.0, qda_reg_alpha3=0.5,
    temperature=1.0, M=1, global_cov=None)

# per-dataset avg global_cov
per_ds = []
off = 0
for d_name in ALL_DS:
    c_names = get_xtail_classnames("/data1/open_datasets/X-TAIL", d_name, 16)
    n_c = len(c_names)
    ds_cov = sum(stats_cls[cid].cov for cid in range(off, off+n_c)) / n_c
    per_ds.append(ds_cov)
    off += n_c
balanced_cov = sum(per_ds) / len(per_ds)
logging.info("Per-class global_cov trace: %.4f, Per-dataset: %.4f",
    stats_cls[0].cov.trace().item(), balanced_cov.trace().item())

clf_per_dataset = LRRGDAClassifier(stats_dict=stats_cls, device=device, rank=32,
    qda_reg_alpha1=0.2, qda_reg_alpha2=2.0, qda_reg_alpha3=0.5,
    temperature=1.0, M=1, global_cov=balanced_cov)

with torch.no_grad():
    zs_logits = test_feats_n @ zs_cls
    rgda_per_class_logits = clf_per_class.forward(test_feats_n)
    rgda_per_dataset_logits = clf_per_dataset.forward(test_feats_n)

zs_pd, pc_pd, pd_pd = [], [], []
for start, end in dataset_slices:
    zs_n = zs_logits[start:end] - zs_logits[start:end].max(dim=-1, keepdim=True).values
    pc_n = rgda_per_class_logits[start:end] - rgda_per_class_logits[start:end].max(dim=-1, keepdim=True).values
    pd_n = rgda_per_dataset_logits[start:end] - rgda_per_dataset_logits[start:end].max(dim=-1, keepdim=True).values
    lbl = test_labels[start:end]
    zs_pd.append(zs_n.argmax(dim=1).eq(lbl).float().mean().item()*100)
    pc_pd.append(pc_n.argmax(dim=1).eq(lbl).float().mean().item()*100)
    pd_pd.append(pd_n.argmax(dim=1).eq(lbl).float().mean().item()*100)

header = f"{'Dataset':<15s} | {'ZS':>7s} | {'RGDA per_class':>13s} | {'RGDA per_dataset':>14s} | {'Diff':>8s}"
sep = "-" * 65
print("\n" + "=" * 80)
print("global_cov 对比: per-class average vs per-dataset average (M=1, 分析版)")
print("=" * 80)
print(header)
print(sep)
for i, d_name in enumerate(ALL_DS):
    diff = pd_pd[i] - pc_pd[i]
    print(f"{d_name:<15s} | {zs_pd[i]:>5.1f}%  | {pc_pd[i]:>12.1f}%  | {pd_pd[i]:>13.1f}%  | {diff:>+7.1f}%")
print(sep)
zs_avg = sum(zs_pd)/len(zs_pd)
pc_avg = sum(pc_pd)/len(pc_pd)
pd_avg = sum(pd_pd)/len(pd_pd)
print(f"{'Average':<15s} | {zs_avg:>5.1f}%  | {pc_avg:>12.1f}%  | {pd_avg:>13.1f}%  | {pd_avg-pc_avg:>+7.1f}%")
