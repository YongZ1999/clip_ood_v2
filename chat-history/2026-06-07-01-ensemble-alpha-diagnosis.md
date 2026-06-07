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
