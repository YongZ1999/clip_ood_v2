# E2-TA — Distillation Ablation (Seed 42)

**Commit**: `d018b43` | **Protocol**: transfer-aware (preserve_aspect + zs_predicted_seen)

## 1. Classification

| Config | ZS Transfer | ZS Average | ZS Last | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|---:|---:|---:|
| **C0 (fd=0,cd=0)** | 61.77 | 70.78 | 80.17 | 61.95 | 71.84 | 83.11 |
| **C1 (fd=1,cd=0)** | 61.78 | 70.86 | 80.29 | 61.99 | 72.08 | 83.35 |
| **C2 (fd=0,cd=2)** | 61.11 | 71.21 | 80.60 | 61.68 | 72.71 | 83.89 |
| **C3 (fd=1,cd=2)** | 61.14 | 71.22 | 80.61 | 61.68 | 72.72 | 83.80 |

## 2. Δ vs C3 (full config)

| Config | ΔZS Transfer | ΔZS Average | ΔZS Last | ΔEns Transfer | ΔEns Average | ΔEns Last |
|---|---:|---:|---:|---:|---:|---:|
| **C0 (fd=0,cd=0)** | +0.64 | -0.44 | -0.44 | +0.28 | -0.89 | -0.69 |
| **C1 (fd=1,cd=0)** | +0.65 | -0.37 | -0.32 | +0.31 | -0.64 | -0.45 |
| **C2 (fd=0,cd=2)** | -0.02 | -0.01 | -0.01 | +0.00 | -0.01 | +0.09 |

## 3. Retrieval — MSCOCO 5K

| Config | Avg I2T R@1 | Avg T2I R@1 | Last I2T R@1 | Last T2I R@1 |
|---|---:|---:|---:|---:|
| **C0 (fd=0,cd=0)** | 51.17 | 33.49 | 52.62 | 34.35 |
| **C1 (fd=1,cd=0)** | 50.15 | 32.45 | 51.92 | 33.83 |
| **C2 (fd=0,cd=2)** | 51.93 | 33.20 | 52.12 | 33.74 |
| **C3 (fd=1,cd=2)** | 51.94 | 33.20 | 52.16 | 33.71 |

## 4. R1 — Routing Comparison (E6-TA)

| classwise | 61.68 | 72.74 | 83.73 |
| zs_predicted_seen | 61.68 | 72.74 | 83.73 |

*No difference between routing modes for this seed.*

## 5. Key Findings

- **FD alone (C1 vs C0)**: Ens Average +0.25, Ens Last +0.24
- **CD alone (C2 vs C0)**: Ens Average +0.88, Ens Last +0.78
- **CD dominant**: CD contributes more to Average (+0.64 vs FD alone)
- **Routing**: classwise and zs_predicted_seen produce identical results for this seed (ZS well-calibrated)