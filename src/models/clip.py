# In[]
from torch import nn
from src.models.lora_sgp import (
    LoRACLIPVisionTransformer, LoRACLIPTextTransformer,
    SGPBaseLoRA, SGPBaseDoRA,
)
from src.models.lora_baseline import VanillaLoRACLIPVisionTransformer, VanillaLoRACLIPTextTransformer
from src.models.lada_text_adapter import LADAAdaptFormerCLIPTextTransformer
from src.models.backbone_utils import (
    embedding_dim,
    encode_image_features,
    encode_text_features,
    is_siglip2_model_name,
    tokenize_texts,
)
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _can_use_openai_pt_fallback(model_name: str) -> bool:
    """Restrict the offline converter to the exact supported CLIP family."""
    normalized = str(model_name).lower().strip()
    return normalized in {"openai/clip-vit-base-patch16", "clip-vit-base-patch16"}


def _load_openai_pt_fallback(model_name: str, original_error: Exception):
    """Convert the local OpenAI checkpoint without changing the LoRA API."""
    fallback_enabled = _env_flag("CLIP_OPENAI_PT_FALLBACK", True)
    if not (fallback_enabled and _can_use_openai_pt_fallback(model_name)):
        raise original_error
    from src.models.openai_clip_compat import (
        default_openai_clip_checkpoint,
        has_openai_clip_checkpoint,
        load_openai_clip_as_hf,
    )
    checkpoint = default_openai_clip_checkpoint()
    if not has_openai_clip_checkpoint(checkpoint):
        raise OSError(
            f"Could not load {model_name!r} from Hugging Face and no usable local "
            f"OpenAI CLIP checkpoint was found at {checkpoint!r}. Original error: {original_error}"
        ) from original_error
    warnings.warn(
        f"Hugging Face CLIP load failed ({original_error}). Falling back to offline conversion "
        f"of OpenAI checkpoint {checkpoint!r} into Hugging Face CLIPModel.",
        RuntimeWarning,
    )
    return load_openai_clip_as_hf(checkpoint)


def get_clip_model(args, train_mode="lora"):
    model_name = getattr(args, "model_name", None) or os.environ.get(
        "CLIP_MODEL_NAME", "openai/clip-vit-base-patch16")
    # The project's cached OpenAI CLIP checkpoint is stored as the legacy
    # PyTorch weights, whereas the supported SigLIP2 checkpoint is safetensors
    # based.  Keep the environment variable as an explicit override, but make
    # the no-env default compatible with the actual checkpoint family.
    use_safetensors = _env_flag(
        "CLIP_USE_SAFETENSORS", is_siglip2_model_name(model_name))
    local_files_only = _env_flag("CLIP_LOCAL_FILES_ONLY", False)
    model_cls = AutoModel if is_siglip2_model_name(model_name) else CLIPModel
    processor_cls = AutoProcessor if is_siglip2_model_name(model_name) else CLIPProcessor
    try:
        model = model_cls.from_pretrained(
            model_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            attn_implementation="sdpa",
        )
    except (OSError, AttributeError) as exc:
        # ``clip.load`` itself is not compatible with this project's LoRA
        # wrappers (OpenAI CLIP fuses QKV).  Convert the local OpenAI ``.pt``
        # weights to the existing HF module layout instead, but only for the
        # known ViT-B/16 checkpoint and only when explicitly enabled.
        model, processor = _load_openai_pt_fallback(model_name, exc)
    else:
        try:
            processor = processor_cls.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                use_fast=True,
            )
        except (OSError, AttributeError) as exc:
            # A complete model cache paired with an evicted tokenizer cache is
            # still usable: retain the HF model and obtain the exact OpenAI
            # BPE tokenizer from the vendored project files.
            if not (_env_flag("CLIP_OPENAI_PT_FALLBACK", True)
                    and _can_use_openai_pt_fallback(model_name)):
                raise
            from src.models.openai_clip_compat import (
                default_openai_clip_checkpoint,
                has_openai_clip_checkpoint,
                OpenAIClipProcessorAdapter,
            )
            checkpoint = default_openai_clip_checkpoint()
            if not has_openai_clip_checkpoint(checkpoint):
                raise
            warnings.warn(
                f"Hugging Face CLIP tokenizer load failed ({exc}). Keeping the loaded HF model "
                "and using the local OpenAI CLIP tokenizer.",
                RuntimeWarning,
            )
            processor = OpenAIClipProcessorAdapter(
                context_length=model.config.text_config.max_position_embeddings)
    # 关闭 attention 输出，避免 SDPA 回退到 eager / 产生警告，同时减少前向开销
    for cfg in (model.config,
                getattr(model, "vision_model", None),
                getattr(model, "text_model", None)):
        if cfg is not None:
            cfg.output_attentions = False
            cfg.output_hidden_states = False
    if train_mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        return model, processor

    elif train_mode == "full":
        for n, p in model.named_parameters():
            if "vision_model.encoder.layers" in n and ("self_attn" in n or "mlp" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False
        return model, processor

    elif train_mode == "lora":
        for p in model.parameters():
            p.requires_grad = False

        rank = args.lora_rank
        lora_type = getattr(args, 'lora_type', 'lora_sgp')
        text_adapter_type = getattr(args, 'text_adapter_type', 'matched')

        def maybe_wrap_lada_text_adapter():
            if not getattr(args, 'tune_text_encoder', True):
                return
            if text_adapter_type == "lada_adaptformer":
                model.text_model = LADAAdaptFormerCLIPTextTransformer(
                    model.text_model,
                    adapter_dim=getattr(args, 'text_adapter_dim', 16),
                    adapter_scale=getattr(args, 'text_adapter_scale', 0.1),
                )
        
        if lora_type == 'lora_vanilla':
            # 普通 LoRA 基线（无 SGP/NSP 投影）
            alpha = getattr(args, 'lora_alpha', None)
            if alpha is None:
                alpha = rank
            dropout = getattr(args, 'lora_dropout', 0.0)
            if getattr(args, 'tune_vision_encoder', True):
                model.vision_model = VanillaLoRACLIPVisionTransformer(
                    model.vision_model,
                    r=rank,
                    lora_alpha=alpha,
                    lora_dropout=dropout)
            if getattr(args, 'tune_text_encoder', True):
                if text_adapter_type == "lada_adaptformer":
                    maybe_wrap_lada_text_adapter()
                else:
                    model.text_model = VanillaLoRACLIPTextTransformer(
                        model.text_model,
                        r=getattr(args, 'text_lora_rank', 4),
                        lora_alpha=alpha,
                        lora_dropout=dropout)
        
        elif lora_type == 'lora_nsp':
            use_soft_projection = getattr(args, "use_soft_projection", False)
            projection_param_mode = getattr(args, 'projection_param_mode', 'full')
            basis_rank = getattr(args, 'basis_rank', None)
            lora_class = SGPBaseDoRA if getattr(args, 'use_dora', True) else SGPBaseLoRA
            if getattr(args, 'tune_vision_encoder', True):
                model.vision_model = LoRACLIPVisionTransformer(
                    model.vision_model,
                    r=rank,
                    use_soft_projection=use_soft_projection,
                    weight_temp=getattr(args, 'weight_temp', 1.0),
                    weight_kind=getattr(args, 'weight_kind', 'log1p'),
                    weight_p=getattr(args, 'weight_p', 1.0),
                    nsp_eps=getattr(args, 'nsp_eps', 0.05),
                    nsp_weight=getattr(args, 'nsp_weight', 0.02),
                    projection_param_mode=projection_param_mode,
                    basis_rank=basis_rank,
                    lora_class=lora_class,
                    target_modules=getattr(args, 'lora_target_modules', None),
                    fused_qkv=getattr(args, 'fused_qkv', False))
            if getattr(args, 'tune_text_encoder', True):
                if text_adapter_type == "lada_adaptformer":
                    maybe_wrap_lada_text_adapter()
                else:
                    model.text_model = LoRACLIPTextTransformer(
                        model.text_model,
                        r=getattr(args, 'text_lora_rank', 4),
                        use_soft_projection=use_soft_projection,
                        nsp_eps=getattr(args, 'nsp_eps', 0.05),
                        weight_temp=getattr(args, 'weight_temp', 1.0),
                        weight_kind=getattr(args, 'weight_kind', 'log1p'),
                        weight_p=getattr(args, 'weight_p', 1.0),
                        nsp_weight=getattr(args, 'nsp_weight', 0.02),
                        projection_param_mode=projection_param_mode,
                        basis_rank=basis_rank,
                        lora_class=lora_class,
                        target_modules=getattr(args, 'lora_target_modules', None),
                        fused_qkv=getattr(args, 'fused_qkv', False))

        elif lora_type == "lora_sgp":
            use_soft_projection = True
            projection_param_mode = getattr(args, 'projection_param_mode', 'full')
            basis_rank = getattr(args, 'basis_rank', None)
            lora_class = SGPBaseDoRA if getattr(args, 'use_dora', True) else SGPBaseLoRA
            if getattr(args, 'tune_vision_encoder', True):
                model.vision_model = LoRACLIPVisionTransformer(
                    model.vision_model,
                    r=rank,
                    weight_temp=getattr(args, 'weight_temp', 1.0),
                    weight_kind=getattr(args, 'weight_kind', 'log1p'),
                    weight_p=getattr(args, 'weight_p', 1.0),
                    use_soft_projection=use_soft_projection,
                    projection_param_mode=projection_param_mode,
                    basis_rank=basis_rank,
                    lora_class=lora_class,
                    target_modules=getattr(args, 'lora_target_modules', None),
                    fused_qkv=getattr(args, 'fused_qkv', False))
            if getattr(args, 'tune_text_encoder', True):
                if text_adapter_type == "lada_adaptformer":
                    maybe_wrap_lada_text_adapter()
                else:
                    model.text_model = LoRACLIPTextTransformer(
                        model.text_model,
                        r=getattr(args, 'text_lora_rank', 4),
                        weight_temp=getattr(args, 'weight_temp', 1.0),
                        weight_kind=getattr(args, 'weight_kind', 'log1p'),
                        weight_p=getattr(args, 'weight_p', 1.0),
                        use_soft_projection=use_soft_projection,
                        projection_param_mode=projection_param_mode,
                        basis_rank=basis_rank,
                        lora_class=lora_class,
                        target_modules=getattr(args, 'lora_target_modules', None),
                        fused_qkv=getattr(args, 'fused_qkv', False))
        else:
            raise ValueError(f"Unsupported lora_type: {lora_type}")

        return model, processor

    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")


class CLIP_BaseNet(nn.Module):
    def __init__(self, args, train_mode="lora"):
        super(CLIP_BaseNet, self).__init__()
        self.train_mode = train_mode
        self.model, self.processor = get_clip_model(args, train_mode=train_mode)

    def forward(self, img, text):
        x = encode_image_features(self.model, img)
        if isinstance(text, dict):
            text_inputs = text
        else:
            text_inputs = tokenize_texts(self.processor, text, model=self.model)
            text_inputs = {k: v.to(img.device) for k, v in text_inputs.items()}
        y = encode_text_features(self.model, text_inputs)
        return x, y

    def encode_image(self, img):
        return encode_image_features(self.model, img)

    def encode_text(self, text):
        text_inputs = tokenize_texts(self.processor, text, model=self.model)
        text_inputs = {k: v.to(self.model.device) for k, v in text_inputs.items()}
        return encode_text_features(self.model, text_inputs)

    @property
    def feature_dim(self):
        return embedding_dim(self.model)

# In[]
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", type=str, default="lora", choices=["frozen", "full", "lora"])
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_type", type=str, default="lora_sgp", choices=["lora_sgp", "lora_nsp"])
    parser.add_argument("--nsp_eps", type=float, default=0.1)
    parser.add_argument("--nsp_weight", type=float, default=1.0)
    parser.add_argument("--weight_temp", type=float, default=0.1)
    parser.add_argument("--weight_kind", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--weight_p", type=float, default=2.0)
    args = parser.parse_args()

    model = CLIP_BaseNet(args, train_mode=args.train_mode)
    print(model)
