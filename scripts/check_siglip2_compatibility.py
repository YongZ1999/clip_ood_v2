#!/usr/bin/env python3
"""One-minute runtime gate before launching the SigLIP 2 formal experiments."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from src.models.backbone_utils import encode_image_features, encode_text_features, tokenize_texts
from src.models.clip import get_clip_model


MODEL_NAME = "google/siglip2-base-patch16-224"


def model_args(model_name, tune_vision, text_adapter_type):
    return SimpleNamespace(
        model_name=model_name,
        lora_rank=4,
        text_lora_rank=4,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_type="lora_nsp",
        tune_vision_encoder=tune_vision,
        tune_text_encoder=True,
        text_adapter_type=text_adapter_type,
        text_adapter_dim=16,
        text_adapter_scale=0.1,
        use_soft_projection=False,
        projection_param_mode="full",
        basis_rank=None,
        use_dora=False,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        fused_qkv=False,
        nsp_eps=0.20,
        nsp_weight=0.02,
        weight_temp=1.0,
        weight_kind="log1p",
        weight_p=1.0,
    )


def check_model(args, label, device):
    model, processor = get_clip_model(args, train_mode="lora")
    model.to(device).eval()
    images = torch.zeros(2, 3, 224, 224, device=device)
    tokens = tokenize_texts(processor, ["a photo of an aircraft", "a photo of a flower"], model=model)
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.inference_mode():
        image_features = encode_image_features(model, images)
        text_features = encode_text_features(model, tokens)
    assert image_features.shape == text_features.shape == (2, image_features.shape[-1])
    assert torch.isfinite(image_features).all() and torch.isfinite(text_features).all()
    print(f"{label}: embedding_dim={image_features.shape[-1]}, "
          f"vision_lora={hasattr(model.vision_model, 'lora_modules')}, "
          f"text_adapter={hasattr(model.text_model, 'adaptformer_modules') or hasattr(model.text_model, 'lora_modules')}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = parser.parse_args()
    if not cli.model_name.startswith("google/siglip2-"):
        raise ValueError("This gate is intentionally limited to a SigLIP 2 checkpoint")
    device = torch.device(cli.device)
    check_model(model_args(cli.model_name, True, "matched"), "LoRA-NF path", device)
    check_model(model_args(cli.model_name, False, "lada_adaptformer"), "LADA-style path", device)
    print("SigLIP 2 compatibility gate: passed")


if __name__ == "__main__":
    main()
