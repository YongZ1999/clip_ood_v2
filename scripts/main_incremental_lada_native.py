#!/usr/bin/env python3
"""Native-LADA continual learning port for Hugging Face SigLIP2.

This is intentionally separate from ``main_incremental.py``.  It preserves
LADA's method-specific recipe (DPT, task-local AdaptFormer, label-specific
memory, OneCycle and per-dataset epochs) rather than forcing LADA into the
LoRA-NF 800-update protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lada.lada_trainer import LADATrainer
from src.lada.native_protocol import LADA_16SHOT_EPOCHS, LADA_NATIVE_DEFAULTS
from src.utils.continual_metrics import ContinualLearningMetrics
from src.utils.data import get_transforms, get_xtail_classnames, get_xtail_trainloader
from src.utils.main_utils import fix_random_seed, get_zeroshot_classifier


DEFAULT_SEQUENCE = [
    "aircraft", "caltech101", "dtd", "eurosat", "flowers", "food101",
    "mnist", "oxford_pets", "stanford_cars", "sun397",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Native LADA on SigLIP2 X-TAIL")
    parser.add_argument("--root", default="/data1/open_datasets/X-TAIL")
    parser.add_argument("--model_name", default="google/siglip2-base-patch16-224")
    parser.add_argument("--dataset_sequence", nargs="+", default=DEFAULT_SEQUENCE)
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=LADA_NATIVE_DEFAULTS["batch_size"])
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="experiments/paper_formal/E7_siglip2")
    parser.add_argument("--experiment_name", required=True)

    # Public LADA model/training configuration.
    parser.add_argument("--lr", type=float, default=LADA_NATIVE_DEFAULTS["lr"])
    parser.add_argument("--weight_decay", type=float, default=LADA_NATIVE_DEFAULTS["weight_decay"])
    parser.add_argument("--lada_k", type=int, default=LADA_NATIVE_DEFAULTS["lada_k"])
    parser.add_argument("--prototype_k", type=int, default=LADA_NATIVE_DEFAULTS["prototype_k"])
    parser.add_argument("--lada_alpha", type=float, default=1.0)
    parser.add_argument("--lada_beta", type=float, default=1.0)
    parser.add_argument("--image_prototypes_weight_coef", type=float,
                        default=LADA_NATIVE_DEFAULTS["image_prototypes_weight_coef"])
    parser.add_argument("--text_adapter_dim", type=int,
                        default=LADA_NATIVE_DEFAULTS["text_adapter_dim"])
    parser.add_argument("--text_adapter_scale", type=float, default=0.1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    # Arguments consumed by get_clip_model / the shared trainer.  Native LADA
    # freezes the visual encoder and uses only the LADA AdaptFormer text path.
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--text_lora_rank", type=int, default=4)
    parser.add_argument("--lora_type", default="lora_nsp")
    parser.add_argument("--lora_alpha", type=float, default=4.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", nargs="*", default=None)
    parser.add_argument("--use_dora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_soft_projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--projection_param_mode", default="full")
    parser.add_argument("--basis_rank", type=int, default=None)
    parser.add_argument("--nsp_eps", type=float, default=0.20)
    parser.add_argument("--nsp_weight", type=float, default=0.02)
    parser.add_argument("--weight_temp", type=float, default=1.0)
    parser.add_argument("--weight_kind", default="log1p")
    parser.add_argument("--weight_p", type=float, default=1.0)
    parser.add_argument("--fused_qkv", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    if not args.model_name.lower().startswith("google/siglip2-"):
        parser.error("Native E7 LADA is intentionally restricted to a SigLIP2 checkpoint")
    if args.num_shots != 16:
        parser.error("This native protocol is defined for the public LADA 16-shot setting")
    args.device = args.device or (f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.tune_vision_encoder = False
    args.tune_text_encoder = True
    args.text_adapter_type = "lada_adaptformer"
    args.lada_native_protocol = True
    args.lada_official_mode = True
    args.lada_replay_mode = "dpt"
    args.dpt_feature_normalize = False
    return args


@torch.no_grad()
def evaluate_task(model, trainer, loader, frozen_weights, seen_classes, label_offset):
    features, labels = trainer._extract_features_manual(loader, normalize=True)
    features = features.to(trainer.device)
    labels = (labels + label_offset).to(trainer.device)
    text_weights = frozen_weights.clone()
    if seen_classes:
        text_weights[:, :seen_classes] = trainer.dpt.text_prototypes.to(trainer.device).t()
    text_weights = F.normalize(text_weights, dim=0)
    text_logits = model.logit_scale.detach().exp() * (features @ text_weights)
    zs_accuracy = text_logits.argmax(dim=1).eq(labels).float().mean().item()

    lada_logits = trainer.lada_classifier(features)
    if lada_logits.shape[1] != seen_classes:
        raise RuntimeError(
            f"LADA memory has {lada_logits.shape[1]} classes; expected {seen_classes}")
    padded = F.pad(lada_logits, (0, text_logits.shape[1] - seen_classes))
    selector = (text_logits.argmax(dim=1) < seen_classes).float().unsqueeze(1)
    native_logits = text_logits + selector * trainer.lada_alpha * padded
    native_accuracy = native_logits.argmax(dim=1).eq(labels).float().mean().item()
    return zs_accuracy, native_accuracy


def write_result(path, *, args, metric_name, tracker, protocol):
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": metric_name,
        "dataset_order": args.dataset_sequence,
        "metrics": tracker.get_summary(),
        "accuracy_matrix": tracker.get_accuracy_matrix().tolist(),
        "protocol": protocol,
        "arguments": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Wrote %s", path)


def main(args):
    fix_random_seed(args.seed)
    trainer = LADATrainer(args)
    model, processor = trainer.model, trainer.processor
    task_names = list(args.dataset_sequence)
    class_names_by_task = [get_xtail_classnames(args.root, name, args.num_shots)
                           for name in task_names]
    all_class_names = [name for task in class_names_by_task for name in task]
    frozen_weights = get_zeroshot_classifier(
        trainer.model_pretrain, processor, all_class_names, args.device)

    test_loaders = {}
    for dataset_name in task_names:
        _, transform = get_transforms(dataset_name, model_name=args.model_name)
        _, _, test_loader, _ = get_xtail_trainloader(
            root=args.root, dataset_name=dataset_name, transform_train=None,
            transform_test=transform, num_shots=args.num_shots,
            batch_size=args.eval_batch_size, num_workers=args.num_workers)
        test_loaders[dataset_name] = test_loader

    zs_metrics = ContinualLearningMetrics(task_names)
    lada_zs_metrics = ContinualLearningMetrics(task_names)
    label_offset = 0
    for step, (dataset_name, class_names) in enumerate(zip(task_names, class_names_by_task)):
        train_transform, test_transform = get_transforms(dataset_name, model_name=args.model_name)
        train_loader, update_loader, _, _ = get_xtail_trainloader(
            root=args.root, dataset_name=dataset_name, transform_train=train_transform,
            transform_test=test_transform, num_shots=args.num_shots,
            batch_size=args.batch_size, num_workers=args.num_workers)
        prototype_loader = DataLoader(
            update_loader.dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0)
        epochs = LADA_16SHOT_EPOCHS[dataset_name]
        args.iterations = epochs * len(train_loader)
        logging.info("Task %d/%d %s: native LADA %d epochs x %d steps = %d updates",
                     step + 1, len(task_names), dataset_name, epochs, len(train_loader), args.iterations)

        trainer.train(train_loader, class_names, label_offset=label_offset,
                      prototype_loader=prototype_loader)
        trainer.finalize_lada_task(prototype_loader, class_names, label_offset)
        seen_classes = label_offset + len(class_names)

        zs_step, lada_step = {}, {}
        eval_offset = 0
        for eval_name, eval_classes in zip(task_names, class_names_by_task):
            zs_acc, lada_acc = evaluate_task(
                model, trainer, test_loaders[eval_name], frozen_weights, seen_classes, eval_offset)
            zs_step[eval_name], lada_step[eval_name] = zs_acc, lada_acc
            logging.info("Task %d eval %s: ZS %.2f%% | native LADA+ZS %.2f%%",
                         step + 1, eval_name, 100 * zs_acc, 100 * lada_acc)
            eval_offset += len(eval_classes)
        zs_metrics.update(step, zs_step)
        lada_zs_metrics.update(step, lada_step)
        trainer.reset_task_text_tuner()
        label_offset = seen_classes

    output = Path(args.output_dir)
    protocol = {
        "name": "native_lada_siglip2_port",
        "backbone": args.model_name,
        "vision_encoder": "frozen",
        "text_tuner": "task-local AdaptFormer (dim=16)",
        "memory": "label-specific memory + DPT spherical GMM replay",
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "epoch_schedule": LADA_16SHOT_EPOCHS,
        "source": "LADA/configs/model/clip_vit_b16.yaml and LADA/trainer.py",
    }
    stem = output / args.experiment_name
    write_result(stem.with_name(stem.name + "_lada_zs_results.json"), args=args,
                 metric_name="native_lada_zs", tracker=lada_zs_metrics, protocol=protocol)
    write_result(stem.with_name(stem.name + "_zs_results.json"), args=args,
                 metric_name="zero_shot", tracker=zs_metrics, protocol=protocol)
    logging.info("Native LADA+ZS summary: %s", lada_zs_metrics.get_summary())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main(parse_args())
