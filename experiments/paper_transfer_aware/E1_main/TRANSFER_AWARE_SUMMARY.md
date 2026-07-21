# Transfer-Aware E1 Main Table — Full Summary

**Commit**: `6db6551`
**Protocol**: `preserve_aspect + zs_predicted_seen`, 3 seeds each

## 1. Classification (ZS + Ensemble)

| Method | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| **LoRA-NF 16-shot** | 61.54 ± 0.38 | 71.28 ± 0.07 | 80.88 ± 0.24 | 61.91 ± 0.22 | 72.63 ± 0.05 | 83.72 ± 0.10 |
| **Standard LoRA 16-shot** | 61.16 ± 0.32 | 70.36 ± 0.28 | 79.17 ± 0.51 | 61.59 ± 0.15 | 71.75 ± 0.34 | 82.04 ± 0.29 |
| **LoRA-NF Full-shot** | 61.48 ± 0.28 | 73.41 ± 0.08 | 83.70 ± 0.09 | 61.89 ± 0.15 | 74.98 ± 0.01 | 86.19 ± 0.02 |
| **Standard LoRA Full-shot** | 61.13 ± 0.45 | 72.86 ± 0.22 | 82.86 ± 0.07 | 61.56 ± 0.33 | 74.33 ± 0.11 | 84.90 ± 0.06 |

## 2. Retrieval — MSCOCO 5K (per-task, 10 tasks)

| Method | Avg I2T R@1 | Avg T2I R@1 | Last I2T R@1 | Last T2I R@1 |
|---|---:|---:|---:|---:|
| **LoRA-NF 16-shot** | 51.97 ± 0.03 | 33.30 ± 0.08 | 51.88 ± 0.27 | 33.68 ± 0.15 |
| **Standard LoRA 16-shot** | 51.83 ± 0.04 | 33.28 ± 0.07 | 51.71 ± 0.18 | 33.83 ± 0.13 |
| **LoRA-NF Full-shot** | 52.15 ± 0.16 | 33.53 ± 0.02 | 51.92 ± 0.37 | 33.97 ± 0.07 |
| **Standard LoRA Full-shot** | 52.00 ± 0.09 | 33.43 ± 0.02 | 52.05 ± 0.24 | 33.93 ± 0.06 |

## 3. Per-Seed Details

| Method | Shot | Seed | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA-NF | 16-shot | 42 | 61.14 | 71.22 | 80.61 | 61.67 | 72.67 | 83.78 |
| LoRA-NF | 16-shot | 43 | 61.61 | 71.27 | 81.08 | 61.96 | 72.58 | 83.78 |
| LoRA-NF | 16-shot | 44 | 61.89 | 71.36 | 80.94 | 62.11 | 72.65 | 83.60 |
| Standard LoRA | 16-shot | 42 | 60.99 | 70.67 | 79.73 | 61.67 | 72.09 | 82.22 |
| Standard LoRA | 16-shot | 43 | 60.95 | 70.25 | 79.04 | 61.42 | 71.73 | 82.20 |
| Standard LoRA | 16-shot | 44 | 61.53 | 70.15 | 78.74 | 61.68 | 71.41 | 81.70 |
| LoRA-NF | Full-shot | 42 | 61.18 | 73.35 | 83.79 | 61.72 | 74.97 | 86.20 |
| LoRA-NF | Full-shot | 43 | 61.73 | 73.50 | 83.70 | 61.99 | 74.99 | 86.19 |
| LoRA-NF | Full-shot | 44 | 61.52 | 73.39 | 83.61 | 61.97 | 74.99 | 86.17 |
| Standard LoRA | Full-shot | 42 | 60.62 | 72.61 | 82.78 | 61.24 | 74.21 | 84.93 |
| Standard LoRA | Full-shot | 43 | 61.27 | 72.93 | 82.91 | 61.54 | 74.35 | 84.94 |
| Standard LoRA | Full-shot | 44 | 61.49 | 73.04 | 82.89 | 61.90 | 74.43 | 84.83 |

## 4. Anomalies
- Standard LoRA 16-shot seed 44: OOM on GPU 2 (raoxuan processes); successfully retried on GPU 0.
- All other 11 jobs completed on first attempt.
- 12 ens + 12 zs + 12 retrieval JSONs verified.