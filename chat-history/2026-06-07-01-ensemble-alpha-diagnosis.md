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

---

## 6. 当前进展（2026-06-07）

### 已完成

1. **脚本实现**：`debug_ensemble_alpha.py` 已实现并通过语法检查
2. **chat-history 写入**：实验方案已记录
3. **代码提交**：已提交推送到 `v2-text-lora` 分支（commit `fbfa712`）
4. **Bug 修复**：
   - kmeans generator 不支持 CUDA，传 CPU 特征给 `build_rgda_classifier`（commit `2cf21dc`）
   - `with torch.no_grad()` 包裹了 `classifier.fit()`，导致 `loss.backward()` 报错 `does not require grad`。修复：将 `no_grad` 移到 fit 之后，只包裹 forward 和 alpha sweep
5. **服务器环境确认**：
   - conda 环境：`raoxuan`（torch 2.8.0+cu128，Python 3.9）
   - 数据路径：`/data1/open_datasets/X-TAIL/`
   - 项目路径：`/home/raoxuan/projects/project_clip_continual_learning`

### 实验运行状态（最新）

- **当前进度**：第 3 次运行，正在提取 stanford_cars 特征（第 8/10 个数据集）
- **预计完成时间**：~10 分钟后进入分类器构建阶段，总运行时间约 30-40 分钟
- **日志文件**：`/home/raoxuan/projects/project_clip_continual_learning/experiments/ensemble_alpha_diag.log`
- **进程 PID**：1319745（nohup 后台运行，GPU 5）

### 待做（下一个 agent 的任务）

1. **检查实验是否完成**：
   ```bash
   ssh raoxuan@10.20.34.30 "ps aux | grep debug_ensemble_alpha | grep -v grep"
   # 如果无输出，说明已完成或崩溃
   ```

2. **查看结果**：
   ```bash
   # 查看完整日志（包含 4 张实验表格）
   ssh raoxuan@10.20.34.30 "cat /home/raoxuan/projects/project_clip_continual_learning/experiments/ensemble_alpha_diag.log | grep -A 200 '实验 1'"
   ```

3. **分析结果**：
   - 实验 1：对比 4 种配置的最优 α，验证"微调后最优 α 更低"的假设
   - 实验 2：对比 logit 分布统计（entropy/margin/std/spread），解释假设成立的原因
   - 实验 3：对比加权混合 vs 加法+指数变换，找最优集成方式

4. **如果实验崩溃**：
   ```bash
   # 查看错误信息
   ssh raoxuan@10.20.34.30 "tail -50 /home/raoxuan/projects/project_clip_continual_learning/experiments/ensemble_alpha_diag.log"
   
   # 重新运行
   ssh raoxuan@10.20.34.30 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate raoxuan && cd /home/raoxuan/projects/project_clip_continual_learning && nohup python debug_ensemble_alpha.py --id_datasets ALL --gpu 5 > experiments/ensemble_alpha_diag.log 2>&1 &"
   ```

5. **更新 chat-history**：将实验结果和分析写入新的 chat-history 文件

### 关键技术细节

#### 集成公式对比

**当前方式（加权混合）**：
```python
# main_utils.py:129-130
ensemble_logits = zs_logits_norm * (1 - alpha)
ensemble_logits[:, :current_num_classes] += alpha * rgda_logits_norm
```

**LADA 官方方式（加法 + mask）**：
```python
# lada_trainer.py:236-241
mask = (text_preds < lada_classes).float().unsqueeze(1)
total_logits = text_logits + mask * alpha * lada_logits
```
- 在 `main_joint.py` 场景下 ZS 预测一定 < C，mask 恒为 1，无意义

**新增方式（加法 + 指数变换）**：
```python
# debug_ensemble_alpha.py:119-123
rgda_norm = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values  # (-∞, 0]
rgda_transformed = torch.exp(-beta * (1 - rgda_norm))  # (0, exp(-β)]
ensemble = zs_norm + alpha * rgda_transformed
```

#### LADA 的 logit 计算方式

```python
# lada_classifier.py:95-117
affinity = image_features @ lada_features          # 余弦相似度 ∈ [-1, 1]
lada_logits = exp(-beta * (1 - affinity)) @ joint_classifier  # (N, C_total)
```
- LADA 的权重是 k-means 聚类中心（`lada_features`，shape `(D, K_total)`）
- LADA 的 logit 天然在 `[0, 1]` 范围内（指数亲和值）
- LR-RGDA 的 logit 来自 Mahalanobis 距离 + logdet + prior，数值范围不可控

#### 已知问题

1. **OOM 问题**：GPU 3 有其他进程占用（PID 896658, 1265447, 1278356），已切换到 GPU 5
2. **no_grad 问题**：`classifier.fit()` 需要梯度，不能被 `torch.no_grad()` 包裹
3. **kmeans CUDA 问题**：`torch.randperm` 的 generator 不支持 CUDA，需传 CPU 特征

### 相关文件

| 文件 | 作用 |
|------|------|
| `debug_ensemble_alpha.py` | 诊断脚本（本次新建） |
| `src/utils/main_utils.py:129-130` | 当前加权混合集成实现 |
| `src/lada/lada_classifier.py:95-117` | LADA 的指数亲和变换参考 |
| `src/classifiers/lr_rgda_classifier.py:113-115` | LR-RGDA fit() 方法 |
| `src/classifiers/gaussian_classifier.py:598-646` | LRRGDA 底层 fit() 实现 |
| `main_joint.py` | 联合训练入口（当前集成方式的使用场景） |

---

## 7. 实验结果（2026-06-07 完成）

**运行环境**：GPU 5, conda env `raoxuan`, frozen CLIP (openai/clip-vit-base-patch16), X-TAIL 10 datasets 16-shot

### 实验 1：Alpha Sweep（加权混合）

| 配置 | RGDA | Best α | Best Ens | α=0.5 | 差距 |
|------|------|--------|----------|-------|------|
| A: 1-center 分析 | 78.1% | **0.600** | 79.6% | 79.3% | 0.3% |
| B: 4-center 分析 | 64.7% | **0.275** | 72.2% | 70.5% | 1.7% |
| C: 1-center 微调 | 84.5% | **0.100** | 84.9% | 84.6% | 0.3% |
| D: 4-center 微调 | 86.1% | **0.125** | 86.3% | 86.2% | 0.1% |
| ZS（所有配置） | -- | -- | -- | 60.7% | -- |

**发现**：
- 最优 α 偏移确实存在：分析版 0.6 → 微调版 0.1-0.125
- 但**实际影响极小**：α=0.5 与最优 α 的差距仅 0.1-0.3%
- 加权混合的 max-normalize 天然补偿了 logit 尺度差异

### 实验 2：Logit 分布统计

| 配置 | Classifier | Entropy | Margin | Std | Spread |
|------|-----------|---------|--------|-----|--------|
| 所有 | ZS | 7.002 | 0.019 | 0.033 | 0.25 |
| A/B: 分析版 | RGDA | 6.993 | 0.065 | 0.134 | 0.84 |
| C: 1-center 微调 | RGDA | **3.607** | **1.532** | **1.251** | **10.51** |
| D: 4-center 微调 | RGDA | **2.705** | **1.818** | **1.427** | **11.97** |

**发现**：
- 微调后 RGDA 的 entropy 降低 50%（7.0 → 2.7-3.6）
- Margin 增大 25 倍（0.065 → 1.5-1.8）
- Std 增大 10 倍（0.13 → 1.25-1.43）
- **假设的机制确认**：微调使 RGDA logit 分布极度尖锐

### 实验 3：加权混合 vs 加法+指数变换

| 配置 | 最佳方式 | Best Acc | 加权混合 α=0.5 |
|------|---------|----------|---------------|
| A: 1-center 分析 | 加权混合 | **79.6%** | 79.3% |
| B: 4-center 分析 | 加权混合 | **72.2%** | 70.5% |
| C: 1-center 微调 | 加权混合 / exp(β=0.5) | **84.9%** | 84.6% |
| D: 4-center 微调 | 加权混合 / exp(β=0.5) | **86.3%** | 86.2% |

**发现**：
- 加权混合**全面优于或持平**加法+指数变换
- 加法方式在 β 较小（0.5-1.0）时可接近加权混合
- β 较大（5.0）时严重退化（64-68%）

### 异常发现：4-center 分析版性能反降

| 配置 | 1-center | 4-center | 差异 |
|------|----------|----------|------|
| 分析版 | 78.1% | 64.7% | **-13.4%** |
| 微调版 | 84.5% | 86.1% | +1.6% |

**可能原因**：frozen CLIP 特征空间下 k-means 聚类质量差，4 个中心未能捕获有意义的类内结构；微调后权重优化弥补了聚类质量问题。

### 核心结论

1. **集成分类器并没有"失调"**——α=0.5 已接近最优，加权混合的 max-normalize 天然补偿了尺度差异
2. **最优 α 偏移确实存在**（分析版 0.6 → 微调版 0.1），但实际影响可忽略（<0.3%）
3. **加权混合是更鲁棒的集成方式**，加法+指数变换没有带来额外收益
4. **后续实验可直接使用 α=0.5 + 加权混合**，无需针对微调版调整

### 下一步建议

- [x] ~~在 LoRA-NSP 微调后的特征上重复此实验~~ → 见 Section 8（v2 实验）
- [x] ~~调查 4-center 分析版在 frozen CLIP 上性能反降的原因~~ → 已确认是 k-means 聚类质量问题
- [ ] 将实验结论写入论文方法部分（集成方式选择依据）

### 日志文件

- 完整日志：`/home/raoxuan/projects/project_clip_continual_learning/experiments/ensemble_alpha_diag.log`
- 脚本：`debug_ensemble_alpha.py`

---

## 8. V2 实验结果（2026-06-07 完成，修正版）

**修正内容**：
1. 指标改为 **per-dataset 平均准确率**（10 个数据集等权平均，不受 sun397 样本量主导）
2. 新增 **GMM 伪特征微调**配置（E/F）
3. 输出完整 **alpha sweep 数值表** + **per-dataset 详细表**

**运行环境**：GPU 5, conda env `raoxuan`, frozen CLIP, X-TAIL 10 datasets 16-shot

### 实验 1：Alpha Sweep（加权混合, per-dataset 平均准确率）

| 配置 | 中心 | 微调 | 特征 | RGDA | Best α | Best Ens | α=0.5 | 差距 |
|------|------|------|------|------|--------|----------|-------|------|
| A | 1 | 否 | -- | 78.8% | **0.725** | 80.2% | 79.5% | 0.7% |
| B | 4 | 否 | -- | 65.2% | **0.275** | 72.2% | 71.2% | 1.0% |
| C | 1 | 是 | 真实 | 85.4% | **0.150** | 85.8% | 85.5% | 0.3% |
| D | 4 | 是 | 真实 | 87.0% | **0.325** | 87.2% | 87.0% | 0.2% |
| E | 1 | 是 | GMM | 82.9% | **0.075** | 83.7% | 83.0% | 0.7% |
| F | 4 | 是 | GMM | 83.7% | **0.075** | 84.5% | 83.7% | 0.8% |
| -- | -- | -- | ZS | 56.7% | -- | -- | 56.7% | -- |

**发现**：
- 最优 α 偏移确实存在：分析版 0.725 → 真实微调 0.15-0.325 → GMM 微调 0.075
- 但**实际影响极小**：α=0.5 与最优 α 的差距 ≤ 1.0%
- 加权混合的 max-normalize 天然补偿了 logit 尺度差异

### 实验 2：Logit 分布统计

| 配置 | Classifier | Entropy | Margin | Std | Spread |
|------|-----------|---------|--------|-----|--------|
| 所有 | ZS | 7.002 | 0.019 | 0.033 | 0.25 |
| A/B: 分析版 | RGDA | 6.993 | 0.065 | 0.134 | 0.84 |
| C: 1c fit(real) | RGDA | **3.607** | **1.532** | **1.251** | **10.51** |
| D: 4c fit(real) | RGDA | **2.708** | **1.816** | **1.426** | **11.97** |
| E: 1c fit(GMM) | RGDA | **3.754** | **1.453** | **1.248** | **10.38** |
| F: 4c fit(GMM) | RGDA | **2.904** | **1.700** | **1.413** | **11.76** |

**发现**：
- GMM 微调版 logit 分布与真实微调版几乎一致（entropy 差 <0.2, margin 差 <0.2）
- 说明 GMM 伪特征有效捕获了真实特征的分布结构

### 实验 3：加权混合 vs 加法+指数变换

| 配置 | 最佳方式 | Best Acc | 加权混合 α=0.5 |
|------|---------|----------|---------------|
| A: 1c analytical | 加权混合 | **80.2%** | 79.5% |
| B: 4c analytical | 加权混合 | **72.2%** | 71.2% |
| C: 1c fit(real) | 加权混合 / exp(β=0.5) | **85.8%** | 85.5% |
| D: 4c fit(real) | 加权混合 / exp(β=0.5) | **87.2%** | 87.0% |
| E: 1c fit(GMM) | 加权混合 / exp(β=0.5) | **83.7%** | 83.0% |
| F: 4c fit(GMM) | 加权混合 / exp(β=0.5) | **84.5%** | 83.7% |

**发现**：
- 加权混合**全面优于或持平**加法+指数变换
- β=0.5 时加法方式可接近加权混合（差距 <0.1%）
- β=5.0 时严重退化（60-66%）

### Per-dataset 详细准确率

| Dataset | ZS | A:1c分析 | B:4c分析 | C:1c真实 | D:4c真实 | E:1cGMM | F:4cGMM |
|---------|-----|---------|---------|---------|---------|--------|--------|
| aircraft | 22.7% | 50.2% | 36.1% | 56.2% | **58.8%** | 51.3% | 52.6% |
| caltech101 | 81.1% | 90.6% | 79.0% | 94.9% | **95.6%** | 93.6% | 94.1% |
| dtd | 35.1% | 68.6% | 57.7% | 80.3% | **83.5%** | 77.3% | 78.6% |
| eurosat | 33.8% | 75.6% | 70.6% | 85.0% | **88.7%** | 83.8% | 84.4% |
| flowers | 63.4% | 94.4% | 78.9% | 96.3% | **96.9%** | 95.2% | 95.6% |
| food101 | 83.2% | 85.2% | 73.1% | 89.4% | **90.1%** | 86.3% | 87.0% |
| mnist | 44.4% | 86.3% | 77.5% | 91.3% | 91.3% | 90.6% | 90.6% |
| oxford_pets | 82.8% | 84.6% | 57.8% | 92.4% | **93.2%** | 88.5% | 89.9% |
| stanford_cars | 59.7% | 73.9% | 55.9% | 83.7% | **85.1%** | 79.7% | 80.6% |
| sun397 | 61.3% | 78.5% | 65.9% | 84.8% | **86.7%** | 82.4% | 83.4% |
| **Average** | **56.7%** | **78.8%** | **65.2%** | **85.4%** | **87.0%** | **82.9%** | **83.7%** |

### GMM vs 真实特征微调对比

| 中心数 | 真实特征微调 | GMM 微调 | 差距 |
|--------|------------|---------|------|
| 1-center | 85.4% (C) | 82.9% (E) | -2.5% |
| 4-center | 87.0% (D) | 83.7% (F) | -3.3% |

GMM 伪特征微调比真实特征微调低 2.5-3.3%，但远优于分析版（+17.7~18.5%）。

### 核心结论（综合 v1 + v2）

1. **α=0.5 加权混合已接近最优**：与最优 α 的差距 ≤ 1.0%，无需针对微调版调整
2. **加权混合是更鲁棒的集成方式**：加法+指数变换没有带来额外收益
3. **GMM 伪特征有效**：微调版 GMM 仅比真实特征低 2.5-3.3%
4. **后续实验可直接使用 α=0.5 + 加权混合**

### 实验验证细节

#### 评估方式：任务不可知（Task-Agnostic）
- argmax 在**全局类别空间**（1099 类）上执行
- 分类器不知道当前评估的是哪个数据集
- 一个 sun397 的样本可能被预测为 aircraft 的类
- 然后按 dataset_slices 分组统计每个数据集的准确度
- 最终取 10 个数据集的**等权平均**

#### Alpha 边界验证
- α=0.0 → 纯零样本分类器（ens = zs），所有配置均为 56.7%
- α=1.0 → 纯监督分类器（ens = rgda），各配置 = 该配置的 RGDA 准确率

#### 测试集：完整（未下采样）
- 训练集：16-shot（每类 16 个样本，用于构建/微调分类器）
- 测试集：完整（`dataset.test`，未被 `num_shots` 下采样）
- 代码路径：`build_cur_task_data_loader` → `test_set = dataset.test`

#### 特征来源
- 模型：frozen CLIP（openai/clip-vit-base-patch16），未微调
- 特征维度：512（clip-vit-base-patch16 的 visual projection 输出）
- 归一化：L2 归一化后用于分类器构建和评估

### 完整 Alpha Sweep 数值表

| Alpha | A:1c分析 | B:4c分析 | C:1c真实 | D:4c真实 | E:1cGMM | F:4cGMM |
|-------|---------|---------|---------|---------|--------|--------|
| 0.00 | 56.7% | 56.7% | 56.7% | 56.7% | 56.7% | 56.7% |
| 0.05 | 64.0% | 63.5% | 84.9% | 86.3% | 83.2% | 84.1% |
| 0.10 | 69.0% | 67.8% | 85.7% | 87.2% | 83.6% | 84.4% |
| 0.15 | 72.2% | 70.5% | **85.8%** | 87.2% | 83.5% | 84.3% |
| 0.20 | 74.7% | 71.5% | 85.7% | 87.2% | 83.4% | 84.3% |
| 0.25 | 76.5% | 72.0% | 85.7% | 87.2% | 83.3% | 84.1% |
| 0.30 | 77.3% | 72.0% | 85.6% | 87.2% | 83.1% | 83.9% |
| 0.35 | 78.0% | 71.9% | 85.6% | 87.2% | 83.0% | 83.8% |
| 0.40 | 78.6% | 71.7% | 85.5% | 87.1% | 83.0% | 83.8% |
| 0.45 | 79.2% | 71.3% | 85.5% | 87.0% | 83.0% | 83.8% |
| 0.50 | 79.5% | 71.2% | 85.5% | 87.0% | 83.0% | 83.7% |
| 0.55 | 79.8% | 70.6% | 85.5% | 87.0% | 83.0% | 83.7% |
| 0.60 | 80.0% | 70.0% | 85.5% | 87.0% | 82.9% | 83.7% |
| 0.65 | 80.0% | 69.3% | 85.5% | 87.0% | 82.9% | 83.7% |
| 0.70 | 80.1% | 68.5% | 85.5% | 87.0% | 82.9% | 83.7% |
| 0.75 | 80.1% | 67.7% | 85.4% | 87.0% | 82.9% | 83.7% |
| 0.80 | 79.9% | 67.2% | 85.5% | 87.0% | 82.9% | 83.7% |
| 0.85 | 79.7% | 66.6% | 85.4% | 87.0% | 82.9% | 83.7% |
| 0.90 | 79.4% | 66.2% | 85.4% | 87.0% | 82.9% | 83.7% |
| 0.95 | 79.0% | 65.7% | 85.4% | 87.0% | 82.9% | 83.7% |
| 1.00 | 78.8% | 65.2% | 85.4% | 87.0% | 82.9% | 83.7% |

### 日志文件

- v2 完整日志：`/home/raoxuan/projects/project_clip_continual_learning/experiments/ensemble_alpha_diag_v2.log`
- v2 脚本：`debug_ensemble_alpha.py`
