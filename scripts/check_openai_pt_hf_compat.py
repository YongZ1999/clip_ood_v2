#!/usr/bin/env python3
"""Verify that the offline OpenAI CLIP fallback preserves CLIP features.

Run this on the GPU server before launching transfer-aware E1 when the
Hugging Face cache is unavailable:

    python scripts/check_openai_pt_hf_compat.py --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from LADA.clip import clip as native_clip
from src.models.openai_clip_compat import (
    default_openai_clip_checkpoint,
    load_openai_clip_as_hf,
)


def _feature_report(name: str, native: torch.Tensor, converted: torch.Tensor) -> None:
    native = F.normalize(native.float(), dim=-1)
    converted = F.normalize(converted.float(), dim=-1)
    cosine = F.cosine_similarity(native, converted, dim=-1)
    max_abs = (native - converted).abs().max().item()
    print(f"{name}: min_cosine={cosine.min().item():.8f} mean_cosine={cosine.mean().item():.8f} "
          f"max_abs={max_abs:.8g}")
    if cosine.min().item() < 0.9999:
        raise AssertionError(f"{name} conversion mismatch: minimum cosine {cosine.min().item():.8f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=default_openai_clip_checkpoint())
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    torch.manual_seed(1234)
    native_model, _ = native_clip.load(args.checkpoint, device=args.device, jit=False)
    converted_model, processor = load_openai_clip_as_hf(args.checkpoint)
    converted_model = converted_model.to(args.device).eval()
    resolution = int(converted_model.config.vision_config.image_size)
    images = torch.randn(2, 3, resolution, resolution, device=args.device)
    prompts = ["a photo of a red car", "a photo of an aircraft"]
    native_tokens = native_clip.tokenize(prompts).to(args.device)
    converted_tokens = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    converted_tokens = {key: value.to(args.device) for key, value in converted_tokens.items()}
    if not torch.equal(native_tokens, converted_tokens["input_ids"]):
        raise AssertionError("Vendored OpenAI tokenizer does not reproduce native CLIP tokens")

    with torch.no_grad():
        native_image = native_model.encode_image(images)
        native_text = native_model.encode_text(native_tokens)
        converted_image = converted_model.get_image_features(pixel_values=images)
        converted_text = converted_model.get_text_features(**converted_tokens)

    _feature_report("image", native_image, converted_image)
    _feature_report("text", native_text, converted_text)
    print("PASS: local OpenAI .pt is compatible with the Hugging Face LoRA-NF model path.")


if __name__ == "__main__":
    main()
