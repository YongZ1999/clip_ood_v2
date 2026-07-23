# E3-TA — Adapter Comparison (3-seed)

**Commit**: `11216d4` | **Protocol**: transfer-aware (preserve_aspect + zs_predicted_seen)

## 1. Classification (Ensemble)

| Adapter | Transfer | Average | Last |
|---|---:|---:|---:|
| **Standard LoRA** | 61.59 ± 0.15 | 71.75 ± 0.34 | 82.04 ± 0.29 |
| **LoRA-Null** | 61.74 ± 0.12 | 71.49 ± 0.19 | 82.14 ± 0.16 |
| **GradProj** | 61.78 ± 0.14 | 72.13 ± 0.23 | 82.96 ± 0.07 |
| **LoRA-NF** | 61.91 ± 0.22 | 72.63 ± 0.05 | 83.72 ± 0.10 |

## 2. Retrieval — MSCOCO 5K

| Adapter | Avg I2T R@1 | Avg T2I R@1 | Last I2T R@1 | Last T2I R@1 |
|---|---:|---:|---:|---:|
| **Standard LoRA** | 51.83 ± 0.04 | 33.28 ± 0.07 | 51.71 ± 0.18 | 33.83 ± 0.13 |
| **LoRA-Null** | 51.83 ± 0.03 | 33.37 ± 0.14 | 51.65 ± 0.44 | 33.97 ± 0.29 |
| **GradProj** | 51.86 ± 0.03 | 33.33 ± 0.06 | 51.72 ± 0.04 | 33.84 ± 0.03 |
| **LoRA-NF** | 51.97 ± 0.03 | 33.30 ± 0.08 | 51.88 ± 0.27 | 33.68 ± 0.15 |

## 3. Ranking (Ensemble Last)

1. **LoRA-NF**: 83.72
2. **GradProj**: 82.96
3. **LoRA-Null**: 82.14
4. **Standard LoRA**: 82.04