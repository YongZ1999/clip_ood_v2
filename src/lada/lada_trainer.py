"""LADA training state for the in-repository continual-learning runner.

This module mirrors the public LADA update rule: frozen image encoder,
task-local text AdaptFormer, label-specific memory, and DPT replay.  It is
also used by the SigLIP2 port, where only the dual-encoder API differs.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

from src.lada.dpt import DPTManager
from src.lada.lada_classifier import LADAClassifier
from src.models.backbone_utils import embedding_dim, encode_image_features
from src.trainers.lora_nsp_trainer import LoRANSPTrainer


def weighted_cross_entropy(logits, targets, weights=None):
    per_sample = F.cross_entropy(logits, targets, reduction="none")
    return (per_sample * weights).mean() if weights is not None else per_sample.mean()


class LADATrainer(LoRANSPTrainer):
    """Native LADA trainer with a backbone-agnostic image/text interface."""

    def __init__(self, args, covariance_history=None, text_covariance_history=None,
                 lada_state=None, dpt_state=None):
        super().__init__(args, covariance_history, text_covariance_history)
        feature_dim = embedding_dim(self.model)
        self.lada_classifier = LADAClassifier(
            feature_dim, beta=getattr(args, "lada_beta", 1.0)).to(self.device)
        self.dpt = DPTManager(feature_dim, prototype_k=getattr(args, "prototype_k", 4))

        if lada_state is not None:
            self.lada_classifier.load_state_dict(lada_state["classifier"])
            self.lada_classifier.prev_lada_features = lada_state["prev_features"].to(self.device)
            self.lada_classifier.joint_classifier = lada_state["joint_classifier"].to(self.device)
            self.lada_classifier.num_prev_classes = lada_state["num_prev_classes"]
        if dpt_state is not None:
            for name in (
                "image_prototypes", "image_prototypes_covs", "image_prototypes_weights",
                "prototype_labels", "text_prototypes",
            ):
                setattr(self.dpt, name, dpt_state[name].to(self.device))

        self.lada_alpha = getattr(args, "lada_alpha", 1.0)
        self.lada_k = getattr(args, "lada_k", 16)
        self.replay_mode = getattr(args, "lada_replay_mode", "dpt")
        self.enable_dpt = self.replay_mode != "none"
        self.official_mode = bool(getattr(args, "lada_official_mode", False))
        self.dpt_feature_normalize = bool(getattr(args, "dpt_feature_normalize", False))
        if self.official_mode and self.dpt_feature_normalize:
            raise ValueError("Native LADA fits DPT GMMs in unnormalized embedding space")
        self.image_prototypes_weight_coef = getattr(
            args, "image_prototypes_weight_coef", 64.0)

    @torch.no_grad()
    def _extract_features_manual(self, dataloader, normalize=True):
        self.model.eval()
        features, labels = [], []
        for images, batch_labels in tqdm(dataloader, desc="Extracting LADA features"):
            feats = encode_image_features(self.model, images.to(self.device))
            if normalize:
                feats = F.normalize(feats, dim=-1)
            features.append(feats.cpu())
            labels.append(batch_labels.cpu())
        return torch.cat(features), torch.cat(labels)

    def build_lada_from_loader(self, prototype_loader):
        logging.info("=== Building label-specific LADA memory ===")
        features, labels = self._extract_features_manual(prototype_loader, normalize=True)
        self.lada_classifier.build_from_data(
            features.to(self.device), labels.to(self.device), k=self.lada_k)

    def _native_text_classifier(self, class_names, templates):
        """Current prompts plus frozen DPT prompts, in chronological class order."""
        current = self.zeroshot_classifier(class_names, templates, use_grad=True)
        if self.dpt.get_num_text_prototypes() == 0:
            return current
        return torch.cat([self.dpt.text_prototypes.detach().t(), current], dim=1)

    def train(self, train_loader, class_names, reference_loader=None, aux_weight=0.0,
              label_offset=0, all_class_names=None, prototype_loader=None):
        """Train one task with the public LADA objective and DPT replay."""
        del reference_loader, aux_weight, all_class_names
        if not self.has_text_adapter:
            raise RuntimeError("Native LADA requires the text AdaptFormer")
        if prototype_loader is None:
            prototype_loader = train_loader
        self.build_lada_from_loader(prototype_loader)

        params = list(self.model.text_model.get_params())
        if self.has_vision_lora:
            params += list(self.model.vision_model.get_params())
        param_groups = [{"params": params, "lr": self.args.lr}]
        param_groups.append({"params": [self.lada_classifier.curr_lada_features], "lr": self.args.lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        if getattr(self.args, "lada_native_protocol", False):
            scheduler = OneCycleLR(
                optimizer, max_lr=[group["lr"] for group in optimizer.param_groups],
                total_steps=self.args.iterations)
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.args.iterations, eta_min=self.args.lr / 3)

        use_amp = bool(getattr(self.args, "amp", True))
        scaler = GradScaler("cuda", enabled=use_amp)
        logit_scale = self.model.logit_scale.detach()
        templates = [lambda name: f"a photo of a {name}."]
        self.model.train()
        self.lada_classifier.train()
        train_iter = iter(train_loader)

        for step in tqdm(range(self.args.iterations), desc="Training (native LADA)"):
            try:
                images, local_labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, local_labels = next(train_iter)
            images, local_labels = images.to(self.device), local_labels.to(self.device)

            with autocast("cuda", enabled=use_amp):
                vision_ctx = torch.enable_grad() if self.has_vision_lora else torch.no_grad()
                with vision_ctx:
                    current_features = encode_image_features(self.model, images)
                global_labels = local_labels + label_offset

                if self.enable_dpt and self.dpt.has_prototypes():
                    replay_feats, replay_labels, replay_weights = self.dpt.sample_prototypes(
                        self.device, add_noise=self.replay_mode == "dpt")
                    features = torch.cat([replay_feats.detach(), current_features], dim=0)
                    labels = torch.cat([replay_labels, global_labels], dim=0)
                    weights = torch.cat([
                        replay_weights * self.image_prototypes_weight_coef,
                        torch.ones(images.shape[0], device=self.device),
                    ])
                else:
                    features, labels, weights = current_features, global_labels, None
                features = F.normalize(features, dim=-1)

                text_classifier = self._native_text_classifier(class_names, templates)
                text_logits = logit_scale.exp() * (features @ text_classifier)
                lada_logits = self.lada_classifier(features)
                if lada_logits.shape[1] != text_logits.shape[1]:
                    raise RuntimeError(
                        "LADA/text class order mismatch: "
                        f"{lada_logits.shape[1]} vs {text_logits.shape[1]}")
                logits = text_logits + self.lada_alpha * lada_logits
                loss = weighted_cross_entropy(logits, labels, weights)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if (step + 1) % 50 == 0 or step + 1 == self.args.iterations:
                accuracy = logits.argmax(dim=-1).eq(labels).float().mean().mul(100).item()
                logging.info("LADA step %d/%d: loss=%.4f acc=%.1f%%", step + 1,
                             self.args.iterations, loss.item(), accuracy)

        self.model.eval()
        self.lada_classifier.eval()
        return self.model

    def finalize_lada_task(self, prototype_loader, class_names, label_offset):
        """Commit current LADA/DPT memory, then leave a fresh tuner for next task."""
        templates = [lambda name: f"a photo of a {name}."]
        # Capture the adapted current-task prompts before resetting the tuner.
        current_text = self.zeroshot_classifier(class_names, templates).t()
        if self.enable_dpt:
            raw_features, local_labels = self._extract_features_manual(
                prototype_loader, normalize=self.dpt_feature_normalize)
            self.dpt.update_image_prototypes(
                raw_features.to(self.device), local_labels.to(self.device), label_offset,
                k=self.dpt.prototype_k)
        self.dpt.update_text_prototypes(current_text)
        self.lada_classifier.finalize_task()

    def reset_task_text_tuner(self):
        """Official LADA reinitializes its task-local text tuner after each task."""
        if hasattr(self.model.text_model, "reset_adapters"):
            self.model.text_model.reset_adapters()
            logging.info("Reset task-local LADA AdaptFormer.")

    def get_lada_state(self):
        return {
            "classifier": self.lada_classifier.state_dict(),
            "prev_features": self.lada_classifier.prev_lada_features.cpu(),
            "joint_classifier": self.lada_classifier.joint_classifier.cpu(),
            "num_prev_classes": self.lada_classifier.num_prev_classes,
        }

    def get_dpt_state(self):
        return {name: getattr(self.dpt, name).cpu() for name in (
            "image_prototypes", "image_prototypes_covs", "image_prototypes_weights",
            "prototype_labels", "text_prototypes",
        )}
