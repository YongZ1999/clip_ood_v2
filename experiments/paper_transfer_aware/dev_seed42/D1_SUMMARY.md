# Phase D1 — Single-Seed Transfer-Aware Optimization Results

**Commit**: `22ab1c0` | **Seed**: 42 | **Method**: LoRA-NF 16-shot
**Baseline** (current config): M=4, gmm_sample, 200 iter, alpha=0.05 → Ens T=61.67, A=72.67, L=83.78

## 1. Alpha Sweep (M=4, gmm_sample, 200 iter)

| Alpha | Transfer | Average | Last | ΔA vs baseline |
|---:|---:|---:|---:|---:|
| 0.00 | 61.14 | 71.22 | 80.61 | -1.45 |
| 0.01 | 61.57 | 72.49 | 83.39 | -0.18 |
| 0.02 | 61.64 | 72.70 | 83.84 | +0.03 |
| 0.05 | 61.68 | 72.69 | 83.83 | +0.02 |
| 0.08 | 61.70 | 72.60 | 83.56 | -0.07 |
| 0.10 | 61.70 | 72.54 | 83.38 | -0.13 |
| 0.15 | 61.70 | 72.44 | 83.12 | -0.23 |
| 0.20 | 61.71 | 72.37 | 82.89 | -0.30 |

**Best alpha**: 0.02 (A=72.70, L=83.84)

## 2. M × Fit × Source Sweep (alpha=0.02)

| M | Fit | Source | Transfer | Average | Last | ΔA vs baseline |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 200 | gmm_mean | 61.67 | 72.75 | 83.91 | +0.08 |
| 4 | 200 | gmm_mean | 61.64 | 72.75 | 83.93 | +0.08 |
| 1 | 200 | gmm_sample | 61.65 | 72.69 | 83.75 | +0.02 |
| 4 | 200 | gmm_sample | 61.64 | 72.70 | 83.79 | +0.03 |
| 1 | 100 | gmm_mean | 61.63 | 72.54 | 83.47 | -0.13 |
| 2 | 200 | gmm_mean | 61.64 | 72.72 | 83.87 | +0.05 |
| 4 | 100 | gmm_mean | 61.60 | 72.53 | 83.49 | -0.14 |

## 3. Recommended Candidate

**M=1, gmm_mean, fit=200, alpha=0.02**

| Metric | Value | vs Baseline |
|---|---:|:---:|
| Transfer | 61.67 | +0.00 |
| **Average** | **72.75** | **+0.08** |
| **Last** | **83.91** | **+0.13** |

**Selection rationale**: Highest Average among all candidates meeting Transfer (≥61.47) and Last (≥83.78) constraints. Uses simpler M=1 and deterministic gmm_mean.

## 4. Key Insight
M=1 with gmm_mean outperforms M=4 with gmm_sample in the 16-shot regime, likely because:
- With only 16 samples/class, a single well-estimated center (gmm_mean) is more stable than sampling from GMM components
- Lower alpha (0.02 vs 0.05) gives slightly less weight to RGDA, preserving more ZS transfer while still improving Last