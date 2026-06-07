# 集成分类器 Alpha 失调诊断实验方案

**日期**: 2026-06-07
**会话概况**: 发现微调后 LR-RGDA 集成分类器性能未达预期，设计诊断实验研究原因并探索替代集成方式。

---

## 1. 问题背景

上次实验发现：多中心 + 微调的 LR-RGDA 分类器性能显著优于分析版（74.8% vs 69.2%），但集成分类器（ZS + LR-RGDA）在 α=0.5 时并未享受到这种增幅。

**假设**：微调后 LR-RGDA 的 logit 分布更尖锐（置信度更高），导致 α=0.5 时 RGDA 在集成中过度主导，需要调低 α。

---

## 2. 实验设计

### 实验 1：Alpha Sweep — 4 种 RGDA 配置

验证上述假设，找到每种配置的最优 α。

| 配置 | 中心数 | 微调 | 说明 |
|------|--------|------|------|
| A | 1 | 否 | 分析版基线 |
| B | 4 | 否 | 多中心不微调 |
| C | 1 | 是 | 微调不多中心 |
| D | 4 | 是 | 多中心+微调（性能最强） |

集成方式：`(1-α) * zs_norm + α * rgda_norm`（当前方式）
扫描：α ∈ {0, 0.025, 0.05, ..., 1.0}（41 个点）

### 实验 2：Logit 分布统计

对 4 种配置的 RGDA logits 和 ZS logits 分别统计：

| 指标 | 含义 |
|------|------|
| entropy | softmax 后的熵，衡量分布锐度 |
| margin | max - 2nd_max 均值，衡量决策确定性 |
| std | max-normalize 后的标准差 |
| spread | max - min，logit 动态范围 |

预期：微调后 RGDA 的 entropy 更低、margin 更大、std 更大。

### 实验 3：两种集成方式对比

| 方式 | 公式 | 说明 |
|------|------|------|
| 加权混合 | `(1-α)*zs_norm + α*rgda_norm` | 当前方式，凸组合 |
| 加法+指数变换 | `zs_norm + α * exp(-β*(1-rgda_norm))` | 借鉴 LADA 的亲和变换 |

**加法+指数变换细节**：
```python
rgda_norm = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values  # (-∞, 0]
rgda_transformed = exp(-β * (1 - rgda_norm))  # (0, exp(-β)]
ensemble = zs_norm + α * rgda_transformed
```

- `rgda_norm` 的 max=0 处 → `exp(-β)`
- `rgda_norm` 的负值处 → 指数衰减至 0
- 与 LADA 的 `exp(-β*(1-affinity))` 公式一致，β 控制锐度
- β 扫描：{0.5, 1.0, 2.0, 5.0}

每种方式都扫描 α 找最优值。

### 不做的方案

- ~~加法+mask~~：`main_joint.py` 下 ZS 预测一定 < C，mask 恒为 1，无意义
- ~~温度缩放~~：等实验 1-3 完成后再决定

---

## 3. 脚本设计

新建 `debug_ensemble_alpha.py`：
- 使用 `--iterations 0` 语义（纯推理端分析，不微调 CLIP 主干）
- 复用现有模块：`get_clip_model(frozen)`, `extract_features`, `get_zeroshot_classifier`, `build_multi_center_stats_dict`, `LRRGDAClassifier`
- 特征只提取一次，logits 只算一次，后续 alpha sweep 在缓存 logits 上做插值
- 输出格式化表格

### 运行命令

```bash
python debug_ensemble_alpha.py --id_datasets ALL --gpu 0
```

---

## 4. 预期产出

1. 确认/否定"微调后最优 α 更低"的假设
2. 每种配置的最优 α 和对应的集成精度
3. 加法+指数变换 vs 加权混合的优劣对比
4. 为后续训练端实验（B4 基线）提供正确的集成参数和方式

---

## 5. 相关文件

- `debug_ensemble_alpha.py`: 新建的诊断脚本
- `src/utils/main_utils.py`: 当前集成方式实现
- `src/lada/lada_classifier.py`: LADA 的指数亲和变换参考
- `src/classifiers/lr_rgda_classifier.py`: LR-RGDA 分类器
- `src/classifiers/gaussian_classifier.py`: LRRGDA 底层实现
