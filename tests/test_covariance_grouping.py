"""Regression test for per-hook-group covariance normalization.

Run with: python tests/test_covariance_grouping.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trainers.lora_nsp_trainer import LoRANSPTrainer


def main():
    # q/k/v share one input hook; out_proj is a separate group.  Both groups
    # see the same two samples, so both estimates must equal X^T X / N.
    q_proj = nn.Identity()
    k_proj = nn.Identity()
    v_proj = nn.Identity()
    out_proj = nn.Identity()
    lora_modules = {
        "layer_0_attn_q_proj": q_proj,
        "layer_0_attn_k_proj": k_proj,
        "layer_0_attn_v_proj": v_proj,
        "layer_0_attn_out_proj": out_proj,
    }

    holder = nn.Module()
    holder.add_module("q_proj", q_proj)
    holder.add_module("k_proj", k_proj)
    holder.add_module("v_proj", v_proj)
    holder.add_module("out_proj", out_proj)
    trainer = object.__new__(LoRANSPTrainer)
    trainer.device = "cpu"
    trainer.model = holder

    batches = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]

    def forward_fn(x):
        y = q_proj(x)
        k_proj(x)
        v_proj(x)
        out_proj(y)

    covariances = trainer._extract_covariances_from_modules(
        lora_modules, forward_fn, batches, desc="covariance regression"
    )
    expected = torch.tensor([[5.0, 7.0], [7.0, 10.0]]) + torch.eye(2) * 1e-6

    for name, covariance in covariances.items():
        torch.testing.assert_close(covariance, expected)
        print(f"{name}: passed")


if __name__ == "__main__":
    main()
