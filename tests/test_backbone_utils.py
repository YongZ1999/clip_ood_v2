"""CPU-only checks for the CLIP/SigLIP feature compatibility helpers."""

from types import SimpleNamespace

import torch

from src.models.backbone_utils import (
    embedding_dim,
    encode_image_features,
    encode_text_features,
    text_tokenize_kwargs,
)


class _Features:
    def __init__(self, value):
        self.pooler_output = value


class _FakeSiglip:
    config = SimpleNamespace(model_type="siglip", text_config=SimpleNamespace(projection_size=7))

    def get_image_features(self, pixel_values):
        return _Features(pixel_values + 1)

    def get_text_features(self, input_ids, attention_mask=None):
        return _Features(input_ids.float() + 2)


def main():
    model = _FakeSiglip()
    x = torch.ones(2, 7)
    torch.testing.assert_close(encode_image_features(model, x), x + 1)
    torch.testing.assert_close(
        encode_text_features(model, {"input_ids": x.long()}), x + 2)
    assert embedding_dim(model) == 7
    assert text_tokenize_kwargs(model) == {
        "padding": "max_length", "truncation": True, "max_length": 64}
    print("backbone utils: passed")


if __name__ == "__main__":
    main()
