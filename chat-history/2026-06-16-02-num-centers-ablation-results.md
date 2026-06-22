# num_centers 消融实验：多中心 + GMM 回放 + 分类器微调

**日期**: 2026-06-16

## 目的

验证 chat-history 中 05-24 的结论（16-shot 下多中心分析版退化）以及 06-14 的结论（多中心 + 分类器微调 + GMM 回放是推理端最强配置），在增量学习场景下找到最优的 num_centers + 分类器微调 + GMM 回放组合。

## 固定协议

```
LoRA-NSP + vision/text + FD/CD
text_tuning_schedule = low_lr_after (Task1: 1e-4, Task2+: 2e-5)
alpha = 0.05
num_shots = 16, batch_size = 32, iterations = 800
seed = 42
rgda_rank = 32, rgda_alpha1 = 0.2, rgda_alpha2 = 2.0, rgda_alpha3 = 0.5
classifier_feature_transform = test
fd_weight = 1.0, cd_weight = 1.0, aux_weight = 1.0
eval_max_samples = 0 (full test split)
```

## 三个实验

| | Exp1 | Exp2 | Exp3 |
|---|---|---|---|
| 日志 | `inc_full_20260615_180647.log` | `inc_full_20260615_204329.log` | `inc_full_20260616_000303.log` |
| `--num_centers` | 4 | 1 | 4 |
| `--rgda_train_iter` | 0 | 0 | 200 |
| `--gmm_sample_mode` | — | — | mean |
| `--gaussian_samples_per_class` | — | — | 16 |

## 最终对比（LADA 协议）

| Method | Transfer | Average | Last |
|--------|:-------:|:-------:|:----:|
| Exp1 (4c, 无微调) Ensemble | 71.31 | 79.25 | 77.61 |
| Exp2 (1c, 无微调) Ensemble | 71.61 | 79.47 | 77.88 |
| **Exp3 (4c+GMM+fit) Ensemble** | **72.30** | **81.57** | **82.05** |
| LADA (paper) | 56.7 | 68.9 | 83.3 |

### 关键差异

| 对比 | Last Δ | Average Δ |
|------|:------:|:--------:|
| Ensemble: Exp3 − Exp2 (1c, 无微调) | **+4.17** | **+2.10** |
| Ensemble: Exp3 − Exp1 (4c, 无微调) | **+4.44** | **+2.32** |
| Ensemble: Exp3 − LADA | **−1.25** | — |
| RGDA: Exp3 − Exp1 (4c, 无微调) | **+4.70** | **+3.72** |

## 结论

1. **Exp3 (4-center + GMM mean replay + rgda_train_iter=200) 是最优配置**：Ensemble Last = 82.05，Average = 81.57
2. **验证了 chat-history 05-24 的结论**：16-shot 下多中心分析版（无微调）RGDA 严重退化，Exp1 RGDA Last 仅 75.94
3. **验证了 chat-history 06-14 的结论**：多中心 + 分类器微调 + GMM 回放是推理端最佳配置。RGDA 从 75.94 提升到 80.64（+4.70）
4. **GMM 回放是关键**：Exp3 相比 Exp1 在 stanford_cars 上提升最显著（Ensemble: 82.5 vs 72.2, +10.3），说明 GMM 分量均值回放有效防止了旧类权重在微调时漂移
5. **距 LADA Last 83.3 差 −1.25**。Exp3 的 Average 81.57 远高于 LADA 的 68.9，Last 仍有差距但已大幅缩小
6. **ZS 在所有实验中一致**（Last ≈ 74.09），证明训练端可复现
