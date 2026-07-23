# E0-PA — Isolated Preserve-Aspect Ablation

**Commit**: `11216d4` | **Config**: old E1 LoRA-NF + classwise + preserve_aspect
**Comparison**: vs old `experiments/paper_formal/E1_main/` LoRA-NF 16-shot (classwise + legacy_square)

## 1. Per-Seed Ensemble Comparison

| Seed | E0 Transfer | E0 Average | E0 Last | Old Transfer | Old Average | Old Last | ΔTransfer | ΔAverage | ΔLast |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 61.82 | 72.69 | 83.61 | 60.00 | 71.65 | 83.70 | +1.83 | +1.04 | -0.09 |
| 43 | 61.83 | 72.50 | 83.86 | 60.05 | 71.84 | 83.74 | +1.78 | +0.66 | +0.12 |
| 44 | 61.98 | 72.53 | 83.60 | 60.37 | 71.77 | 83.78 | +1.61 | +0.76 | -0.18 |

## 2. 3-Seed Summary

| Config | Transfer | Average | Last |
|---|---:|---:|---:|
| **E0 (preserve_aspect)** | 61.88 ± 0.09 | 72.57 ± 0.10 | 83.69 ± 0.15 |
| **Old (legacy_square)** | 60.14 ± 0.20 | 71.75 ± 0.10 | 83.74 ± 0.04 |
| **Δ** | +1.74 ± 0.11 | +0.82 ± 0.20 | -0.05 ± 0.15 |

## 3. Key Finding

preserve_aspect vs legacy_square resize: the isolated effect shows the contribution of standard CLIP aspect-ratio-preserving preprocessing.