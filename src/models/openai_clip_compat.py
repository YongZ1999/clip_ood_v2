"""Offline compatibility loader for OpenAI CLIP ``.pt`` checkpoints.

The continual-learning adapters in this repository are written against the
Hugging Face ``CLIPModel`` module layout (separate Q/K/V projections and
``vision_model.encoder.layers``).  OpenAI's public ``ViT-B-16.pt`` is a
different, fused-QKV implementation, so returning ``clip.load(...)`` directly
would silently bypass those adapters.

This module instead maps the *pretrained weights* from the OpenAI checkpoint
into an equivalent Hugging Face ``CLIPModel``.  It is deliberately limited to
OpenAI ViT CLIP checkpoints; ResNet CLIP checkpoints have a different visual
backbone and are not compatible with the project's LoRA wrappers.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import BatchEncoding, CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig


OPENAI_VIT_B16_DEFAULT_PATH = "~/.cache/clip/ViT-B-16.pt"


class OpenAIClipProcessorAdapter:
    """Small ``CLIPProcessor``-compatible text tokenizer for an OpenAI `.pt`.

    Image preprocessing is handled by the project's dataset transforms.  The
    training/evaluation code only needs the processor for CLIP BPE text
    tokenization, so using the vendored OpenAI tokenizer avoids any Hugging
    Face tokenizer-cache or network dependency.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, context_length: int = 77):
        self.context_length = int(context_length)
        self.model_max_length = self.context_length

    def __call__(self, *, text: str | Iterable[str] | None = None,
                 return_tensors: str | None = "pt", padding: Any = True,
                 truncation: bool = True, max_length: int | None = None,
                 **_: Any) -> BatchEncoding:
        if text is None:
            raise ValueError("OpenAIClipProcessorAdapter requires text=...")
        if return_tensors not in (None, "pt"):
            raise ValueError("Only return_tensors='pt' is supported for OpenAI CLIP fallback")

        # Import lazily: normal Hugging Face loading must not depend on the
        # vendored OpenAI tokenizer or its optional text-normalization package.
        from LADA.clip.clip import tokenize

        texts = [text] if isinstance(text, str) else list(text)
        length = int(max_length or self.context_length)
        input_ids = tokenize(texts, context_length=length, truncate=bool(truncation))
        # CLIP's causal mask already makes post-EOT pads irrelevant.  Supplying
        # the usual padding mask mirrors CLIPProcessor's normal output shape.
        attention_mask = input_ids.ne(0).long()
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})


def default_openai_clip_checkpoint() -> str:
    """Return the conventional local OpenAI CLIP checkpoint path."""
    return os.path.expanduser(os.environ.get(
        "CLIP_OPENAI_PT_PATH", OPENAI_VIT_B16_DEFAULT_PATH))


def has_openai_clip_checkpoint(path: str | os.PathLike[str] | None = None) -> bool:
    return Path(path or default_openai_clip_checkpoint()).is_file()


def _load_openai_state_dict(checkpoint_path: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    checkpoint_path = str(checkpoint_path)
    try:
        state_dict = torch.jit.load(checkpoint_path, map_location="cpu").eval().state_dict()
    except RuntimeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported OpenAI CLIP checkpoint payload: {type(payload)!r}")
        state_dict = payload

    # The JIT state dict includes a few integer metadata entries.  Conversion
    # only needs tensor parameters/buffers.
    return {
        key: value.detach().cpu()
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }


def _layer_count(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    indices = set()
    marker = f"{prefix}.resblocks."
    for key in state_dict:
        if key.startswith(marker):
            remainder = key[len(marker):]
            indices.add(int(remainder.split(".", 1)[0]))
    if not indices:
        raise ValueError(f"No transformer layers found under {marker!r}")
    if indices != set(range(max(indices) + 1)):
        raise ValueError(f"Non-contiguous transformer layer indices under {marker!r}: {sorted(indices)}")
    return len(indices)


def _build_hf_clip_config(state_dict: dict[str, torch.Tensor]) -> CLIPConfig:
    if "visual.proj" not in state_dict or "visual.conv1.weight" not in state_dict:
        raise ValueError(
            "Only OpenAI ViT CLIP checkpoints are supported by the offline fallback; "
            "this checkpoint does not look like ViT-B/16."
        )

    vision_width = int(state_dict["visual.conv1.weight"].shape[0])
    vision_patch_size = int(state_dict["visual.conv1.weight"].shape[-1])
    vision_positions = int(state_dict["visual.positional_embedding"].shape[0])
    grid_size = math.isqrt(vision_positions - 1)
    if grid_size * grid_size + 1 != vision_positions:
        raise ValueError("OpenAI visual positional embedding does not describe a square ViT grid")

    text_width = int(state_dict["ln_final.weight"].shape[0])
    projection_dim = int(state_dict["text_projection"].shape[1])
    text_config = CLIPTextConfig(
        vocab_size=int(state_dict["token_embedding.weight"].shape[0]),
        hidden_size=text_width,
        intermediate_size=int(state_dict["transformer.resblocks.0.mlp.c_fc.weight"].shape[0]),
        projection_dim=projection_dim,
        num_hidden_layers=_layer_count(state_dict, "transformer"),
        num_attention_heads=text_width // 64,
        max_position_embeddings=int(state_dict["positional_embedding"].shape[0]),
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
    )
    vision_config = CLIPVisionConfig(
        hidden_size=vision_width,
        intermediate_size=int(state_dict["visual.transformer.resblocks.0.mlp.c_fc.weight"].shape[0]),
        projection_dim=projection_dim,
        num_hidden_layers=_layer_count(state_dict, "visual.transformer"),
        num_attention_heads=vision_width // 64,
        image_size=grid_size * vision_patch_size,
        patch_size=vision_patch_size,
        num_channels=int(state_dict["visual.conv1.weight"].shape[1]),
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
    )
    return CLIPConfig.from_text_vision_configs(
        text_config, vision_config, projection_dim=projection_dim,
    )


def _convert_openai_state_dict(state_dict: dict[str, torch.Tensor],
                               model: CLIPModel) -> dict[str, torch.Tensor]:
    """Map OpenAI CLIP parameter names/layout into Hugging Face CLIP names."""
    target = model.state_dict()
    converted: dict[str, torch.Tensor] = {}

    def copy(source: str, destination: str, transpose: bool = False) -> None:
        if source not in state_dict:
            raise KeyError(f"OpenAI checkpoint is missing {source!r}")
        if destination not in target:
            raise KeyError(f"Installed transformers CLIPModel lacks expected key {destination!r}")
        value = state_dict[source].t() if transpose else state_dict[source]
        if tuple(value.shape) != tuple(target[destination].shape):
            raise ValueError(
                f"Shape mismatch {source} {tuple(value.shape)} -> {destination} "
                f"{tuple(target[destination].shape)}"
            )
        converted[destination] = value

    copy("visual.class_embedding", "vision_model.embeddings.class_embedding")
    copy("visual.conv1.weight", "vision_model.embeddings.patch_embedding.weight")
    copy("visual.positional_embedding", "vision_model.embeddings.position_embedding.weight")
    for suffix in ("weight", "bias"):
        copy(f"visual.ln_pre.{suffix}", f"vision_model.pre_layrnorm.{suffix}")
        copy(f"visual.ln_post.{suffix}", f"vision_model.post_layernorm.{suffix}")

    for index in range(_layer_count(state_dict, "visual.transformer")):
        source_base = f"visual.transformer.resblocks.{index}"
        target_base = f"vision_model.encoder.layers.{index}"
        for suffix in ("weight", "bias"):
            copy(f"{source_base}.ln_1.{suffix}", f"{target_base}.layer_norm1.{suffix}")
            copy(f"{source_base}.ln_2.{suffix}", f"{target_base}.layer_norm2.{suffix}")
            copy(f"{source_base}.attn.out_proj.{suffix}", f"{target_base}.self_attn.out_proj.{suffix}")
            copy(f"{source_base}.mlp.c_fc.{suffix}", f"{target_base}.mlp.fc1.{suffix}")
            copy(f"{source_base}.mlp.c_proj.{suffix}", f"{target_base}.mlp.fc2.{suffix}")

        for suffix in ("weight", "bias"):
            fused = state_dict[f"{source_base}.attn.in_proj_{suffix}"]
            q, k, v = fused.chunk(3, dim=0)
            for name, value in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
                destination = f"{target_base}.self_attn.{name}.{suffix}"
                if tuple(value.shape) != tuple(target[destination].shape):
                    raise ValueError(f"Shape mismatch for {destination!r}")
                converted[destination] = value

    copy("visual.proj", "visual_projection.weight", transpose=True)

    copy("token_embedding.weight", "text_model.embeddings.token_embedding.weight")
    copy("positional_embedding", "text_model.embeddings.position_embedding.weight")
    for suffix in ("weight", "bias"):
        copy(f"ln_final.{suffix}", f"text_model.final_layer_norm.{suffix}")

    for index in range(_layer_count(state_dict, "transformer")):
        source_base = f"transformer.resblocks.{index}"
        target_base = f"text_model.encoder.layers.{index}"
        for suffix in ("weight", "bias"):
            copy(f"{source_base}.ln_1.{suffix}", f"{target_base}.layer_norm1.{suffix}")
            copy(f"{source_base}.ln_2.{suffix}", f"{target_base}.layer_norm2.{suffix}")
            copy(f"{source_base}.attn.out_proj.{suffix}", f"{target_base}.self_attn.out_proj.{suffix}")
            copy(f"{source_base}.mlp.c_fc.{suffix}", f"{target_base}.mlp.fc1.{suffix}")
            copy(f"{source_base}.mlp.c_proj.{suffix}", f"{target_base}.mlp.fc2.{suffix}")

        for suffix in ("weight", "bias"):
            fused = state_dict[f"{source_base}.attn.in_proj_{suffix}"]
            q, k, v = fused.chunk(3, dim=0)
            for name, value in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
                destination = f"{target_base}.self_attn.{name}.{suffix}"
                if tuple(value.shape) != tuple(target[destination].shape):
                    raise ValueError(f"Shape mismatch for {destination!r}")
                converted[destination] = value

    copy("text_projection", "text_projection.weight", transpose=True)
    copy("logit_scale", "logit_scale")

    parameter_names = set(dict(model.named_parameters()))
    missing_parameters = sorted(parameter_names.difference(converted))
    if missing_parameters:
        raise ValueError(
            "OpenAI-to-HF conversion did not initialize all CLIP parameters: "
            + ", ".join(missing_parameters)
        )
    return converted


def load_openai_clip_as_hf(checkpoint_path: str | os.PathLike[str] | None = None):
    """Load local OpenAI ViT CLIP weights into a Hugging Face ``CLIPModel``.

    Returns a normal ``(CLIPModel, processor)`` pair, so all existing LoRA
    wrappers, feature extraction, distillation and classifiers keep exactly the
    same model interface as the online Hugging Face path.
    """
    checkpoint = Path(checkpoint_path or default_openai_clip_checkpoint()).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"OpenAI CLIP checkpoint not found: {checkpoint}. Set CLIP_OPENAI_PT_PATH "
            "or repair the Hugging Face checkpoint cache/network."
        )

    state_dict = _load_openai_state_dict(checkpoint)
    config = _build_hf_clip_config(state_dict)
    model = CLIPModel(config)
    converted = _convert_openai_state_dict(state_dict, model)
    incompatible = model.load_state_dict(converted, strict=False)
    missing = sorted(set(incompatible.missing_keys).intersection(dict(model.named_parameters())))
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected incomplete OpenAI-to-HF conversion: "
            f"missing_parameters={missing}, unexpected={incompatible.unexpected_keys}"
        )
    model.config._name_or_path = f"openai-pt:{checkpoint}"
    model.eval()
    return model, OpenAIClipProcessorAdapter(context_length=config.text_config.max_position_embeddings)
