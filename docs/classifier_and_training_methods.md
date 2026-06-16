# LR-RGDA 分类器与 LoRA-NSP 训练方法

> 本文档描述 `main_incremental.py` 中使用的核心分类器和增量训练方法，基于师兄的实验代码集成。

---

## 1. LR-RGDA 分类器（低秩正则化高斯判别分析）

### 1.1 动机

标准 QDA（二次判别分析）需要为每个类 $c$ 计算协方差矩阵的逆 $\Sigma_c^{-1}$，当特征维度 $D=512$、类别数 $C$ 增长时：
- **存储开销**：$O(C \cdot D^2)$，每个类 $512 \times 512 \approx 262K$ 个浮点数
- **计算开销**：求逆 $O(D^3)$，增量场景中每加一个类都要重算

LR-RGDA 通过**低秩分解 + Woodbury 恒等式**将协方差逆的计算从 $O(D^3)$ 降到 $O(r^3)$，其中 $r=32 \ll D$。

### 1.2 数学公式

每类协方差的低秩近似（SVD）：

$$\Sigma_c \approx U_{SVD} \cdot \text{diag}(s_1, \ldots, s_r) \cdot U_{SVD}^\top$$

正则化协方差：

$$\tilde{\Sigma}_c = \alpha_1 \Sigma_c + \alpha_2 \Sigma_{global} + \alpha_3 I$$

其中：
- $\alpha_1$ ($`--rgda\_alpha1`$, default=0.2)：类内协方差的权重
- $\alpha_2$ ($`--rgda\_alpha2`$, default=2.0)：全局协方差的权重
- $\alpha_3$ ($`--rgda\_alpha3`$, default=0.5)：恒等正则化（保证正定性）

记共享基底 $A = \alpha_2 \Sigma_{global} + \alpha_3 I$，有效低秩因子 $U_{eff} = \sqrt{\alpha_1 \cdot s} \cdot U_{SVD}$。

Woodbury 求逆：

$$\tilde{\Sigma}_c^{-1} = A^{-1} - A^{-1} U_{eff} \left( I + U_{eff}^\top A^{-1} U_{eff} \right)^{-1} U_{eff}^\top A^{-1}$$

关键优势：$I + U_{eff}^\top A^{-1} U_{eff}$ 的尺寸仅为 $r \times r = 32 \times 32$，求逆几乎免费。

### 1.3 分类得分

对特征 $\mathbf{x}$，类 $c$ 的得分：

$$\text{logit}_c = \mathbf{w}_c^\top \mathbf{x} + b_c + \frac{1}{2} \mathbf{u}^\top M^{-1} \mathbf{u}$$

其中：
- $\mathbf{w}_c = A^{-1} \boldsymbol{\mu}_c$（线性项，预计算）
- $b_c = -\frac{1}{2} \boldsymbol{\mu}_c^\top A^{-1} \boldsymbol{\mu}_c - \frac{1}{2} \log|\tilde{\Sigma}_c| + \log p(c)$（偏置，预计算）
- $\mathbf{u} = U_{eff}^\top A^{-1} (\mathbf{x} - \boldsymbol{\mu}_c)$（Woodbury 修正项）
- $M = I + U_{eff}^\top A^{-1} U_{eff}$（$32 \times 32$ 矩阵，预计算）

### 1.4 多中心扩展（Multi-Center LR-RGDA）

当 `--num_centers M > 1` 时，每类使用 k-means 聚成 $M$ 个中心：

$$\text{logit}_c = \log\sum_{m=1}^{M} \exp(\text{logit}_{c,m})$$

即 log-sum-exp 聚合多个中心的得分。共享协方差在每个中心的 Woodbury 修正中保持独立。

### 1.5 全局协方差的增量更新

增量学习中，全局协方差采用**等权平均**（非 EMA）：

$$\Sigma_{global}^{(t)} = \frac{\Sigma_{global}^{(t-1)} \cdot (t-1) + \Sigma_{new}}{t}$$

并按数据集做均衡（防止类数多的数据集主导协方差估计）：

$$\Sigma_{dataset\_balanced} = \frac{1}{|\text{datasets}|} \sum_{d} \frac{1}{|C_d|} \sum_{c \in C_d} \Sigma_c$$

---

## 2. LoRA-NSP 训练方法

### 2.1 双塔 LoRA 微调

同时对 CLIP 的图像编码器（ViT）和文本编码器（Transformer）插入 LoRA 模块：

- **图像端**：ViT 每层的 q/k/v/out_proj + MLP fc1/fc2，共 72 个 LoRA 模块
- **文本端**：Transformer 每层的 q/k/v/out_proj + MLP fc1/fc2，共 72 个 LoRA 模块
- **秩**：$r=4$（默认），$\text{lora\_alpha}=r$

LoRA 前向传播：$W' = W + B \cdot A \cdot P$

其中 $P$ 是零空间投影矩阵（NSP 特有）。

### 2.2 零空间投影（NSP）

**目标**：增量学习中，限制新任务的 LoRA 更新在"不重要"的子空间内，保护旧任务的知识。

**步骤**：
1. **提取协方差**：将当前任务数据通过各层，计算中间特征的二阶矩 $X^\top X / N$
2. **特征分解**：$\text{eigh}(\Sigma)$ 返回升序特征值，最小的 $m$ 个特征向量 = 噪声方向
3. **构建投影**：$P = V_{keep} \cdot V_{keep}^\top$，其中 $V_{keep}$ 是保留的特征向量

**关键参数**：
- `--nsp_eps` (default=0.05)：累计能量比阈值，控制投影保留多少"不重要"方向
- `--nsp_weight` (default=0.02)：投影权重

### 2.3 增量训练流程

每学完一个任务：

```
1. merge_lora_weights():  B·A·P → 合入基础权重 W
2. extract_covariances():  从当前任务提取特征协方差
3. update_covariance_history():  等权累加历史协方差
4. update_projection_matrices():  从累积协方差重建投影 P
5. reset_lora():  重置 A/B 为随机初始化
```

### 2.4 文本编码器调度（Text Tuning Schedule）

通过 `--text_tuning_schedule` 控制文本 LoRA 在不同任务的学习率：

| 模式 | Task 1 | Task 2+ | 说明 |
|------|--------|---------|------|
| `always` | lr=1e-4 | lr=1e-4 | 所有任务全量微调 |
| `never` | 不训练 | 不训练 | 仅图像 LoRA |
| `freeze_after` | lr=1e-4 | **冻结** | Task 1 后完全冻结文本 |
| `low_lr_after` | lr=1e-4 | **lr=2e-5** | Task 1 后降学习率（$\times 0.2$） |

参数：
- `--text_schedule_switch_task` (default=1)：第几个任务后切换（1-indexed）
- `--text_lr_scale_after_task` (default=0.2)：缩放因子

### 2.5 损失函数

总损失由多个分量组成：

$$\mathcal{L} = \mathcal{L}_{SCE} + w_{aux} \cdot \mathcal{L}_{aux} + w_{FD} \cdot \mathcal{L}_{FD} + w_{CD} \cdot \mathcal{L}_{CD}$$

| 损失 | 说明 | 默认权重 |
|------|------|---------|
| $\mathcal{L}_{SCE}$ | 对称交叉熵（CE + RCE），标签噪声鲁棒 | 1.0 |
| $\mathcal{L}_{aux}$ | 辅助线性分类头 SCE 损失 | 1.0 (`--aux_weight`) |
| $\mathcal{L}_{FD}$ | 特征蒸馏：$1 - \cos(\text{feat}_{student}, \text{feat}_{teacher})$ | 1.0 (`--fd_weight`) |
| $\mathcal{L}_{CD}$ | 跨模态蒸馏：KL(teacher_logits $\|$ student_logits) | 1.0 (`--cd_weight`) |

---

## 3. 集成分类器（Ensemble Classifier）

### 3.1 固定权重集成（Fixed-$\alpha$）

将 Zero-shot 和 LR-RGDA 的 logits 线性组合：

$$\text{logit}_{ens} = (1 - \alpha) \cdot \text{logit}_{ZS} + \alpha \cdot \text{logit}_{RGDA}$$

其中 $\alpha$ 由 `--alpha` 控制（default=0.05），仅在 ID 类别区域叠加 RGDA。

### 3.2 自适应集成（Adaptive Ensemble）

通过 `--adaptive_ensemble` 启用，对每个样本根据两分类器置信度动态调整 $\alpha$：

$$\alpha_{sample} = \sigma(\text{conf}_{RGDA} - \text{conf}_{ZS})$$

其中 $\sigma$ 是 sigmoid 函数，$\text{conf}$ = softmax 后的最大概率。

---

## 4. 评估协议（LADA Paper Metrics）

### 4.1 准确率矩阵

增量学习维护三角矩阵 $M$，其中 $M_{i,j}$ 表示学完第 $i+1$ 个任务后在第 $j+1$ 个任务上的准确率（$j \le i$）：

```
Step 1:  [acc₁]
Step 2:  [acc₁, acc₂]
...
Step 10: [acc₁, acc₂, ..., acc₁₀]
```

### 4.2 三项指标

| 指标 | 定义 | 计算公式 |
|------|------|---------|
| **Transfer** | 旧任务在学习新任务后的平均保持率 | $\text{Transfer}_k = \frac{1}{k}\sum_{j<k} M_{k,j}$（$k \ge 2$） |
| **Average** | 每列从首次可测到最后一步的平均 | $\text{Avg}_k = \frac{1}{K-k}\sum_{j=k}^{K-1} M_{j,k}$ |
| **Last** | 所有任务学完后每个任务最终准确率 | $\text{Last}_k = M_{K-1,k}$ |

---

## 5. 命令行参数速查

### 核心训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--alpha` | 0.05 | Ensemble 中 RGDA 的权重 |
| `--num_shots` | 16 | 每类训练样本数 |
| `--full_shot` | False | 使用全量数据（覆盖 num_shots） |
| `--batch_size` | 32 | 训练/测试 batch size |
| `--iterations` | 800 | 每任务训练迭代次数 |
| `--lr` | 1e-4 | 基础学习率 |
| `--num_centers` | 1 | 每类 k-means 中心数（>1 启用多中心 LR-RGDA） |

### 文本 LoRA 调度

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tune_text_encoder` | True | 启用文本 LoRA |
| `--text_lora_rank` | 4 | 文本 LoRA 秩 |
| `--text_tuning_schedule` | None→"always" | 调度模式 |
| `--text_schedule_switch_task` | 1 | 切换 LR 的任务编号 |
| `--text_lr_scale_after_task` | 0.2 | 降 LR 缩放因子 |

### 蒸馏与损失

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fd_weight` | 1.0 | 特征蒸馏权重 |
| `--cd_weight` | 1.0 | 跨模态蒸馏权重 |
| `--aux_weight` | 1.0 | 辅助分类头权重 |

### 分类器构建

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rgda_rank` | 32 | LR-RGDA 低秩分解秩 |
| `--rgda_alpha1` | 0.2 | 类内协方差权重 |
| `--rgda_alpha2` | 2.0 | 全局协方差权重 |
| `--rgda_alpha3` | 0.5 | 恒等正则化权重 |
| `--classifier_feature_transform` | "test" | 分类器特征提取的 transform |
| `--adaptive_ensemble` | False | 启用自适应集成 |

---

## 6. 推荐实验配置

### 增量学习（Incremental）

```bash
python main_incremental.py \
  --alpha 0.05 \
  --num_shots 16 \
  --batch_size 32 \
  --iterations 800 \
  --num_centers 4 \
  --text_tuning_schedule low_lr_after \
  --gpu 0
```

### 联合学习（Joint）

```bash
python main_joint.py \
  --alpha 0.05 \
  --num_shots 16 \
  --batch_size 32 \
  --iterations 6400 \
  --num_centers 4 \
  --tune_text_encoder True \
  --gpu 0
```
