# Phase D2 — RGDA Rank Sweep Results

**Commit**: `1578060` | **Seed**: 42 | **Eval Seed**: 42
**D1 Matching Baseline**: M=4, gmm_mean, fit=200, rank=32(default), alpha=0.02
→ T=61.64, A=72.75, L=83.93

## Rank Sweep (M=4, gmm_mean, fit=200, eval_seed=42)

| Rank | Alpha | Transfer | Average | Last | ΔA vs D1 baseline |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.01 | 61.57 | 72.53 | 83.45 | -0.22 |
| 8 | 0.02 | 61.64 | 72.75 | 83.95 | +0.00 |
| 8 | 0.05 | 61.68 | 72.83 | 84.07 | +0.08 |
| 15 | 0.05 | 61.68 | 72.83 | 84.05 | +0.08 |
| 24 | 0.05 | 61.68 | 72.83 | 84.05 | +0.08 |
| 32 | 0.05 | 61.68 | 72.83 | 84.05 | +0.08 |

## Decision

**No candidate meets the +0.20 Average threshold.**

| Criterion | Best D2 value | Threshold | Met? |
|---|---:|---:|:---:|
| Average gain | +0.08 (rank=8, alpha=0.05) | ≥ +0.20 | **NO** |
| Transfer protection | 61.68 (within 0.2 of 61.67) | ≥ 61.47 | YES |
| Last protection | 84.07 (≥ 83.78) | ≥ 83.78 | YES |

## Key Insight

Rank sweep reveals minimal sensitivity in the 16-shot regime: even rank=8 preserves full performance, consistent with the limited effective rank of data with only 16 examples per class. Further classifier-only optimization (alpha, M, fit, source, rank) appears to have reached diminishing returns.

Phase D3 (training-side changes: CD weight, NSP epsilon) remains the logical next step if improvement beyond +0.20 is required before multi-seed confirmation.