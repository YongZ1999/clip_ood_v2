# LADA: Scalable Label-Specific CLIP Adapter for Continual Learning 内容总结

- **作者**: Mao-Lin Luo, Zi-Hao Zhou, Tong Wei, Min-Ling Zhang | **年份**: 2025 | **会议/期刊**: ICML 2025

---

## 问题定义

### 论文试图解决什么问题？
CLIP 在持续学习中的灾难性遗忘问题，特别是：
1. **前向遗忘（forward forgetting）**：微调破坏预训练知识，导致零样本 OOD 泛化能力下降
2. **后向遗忘（backward forgetting）**：学习新任务后，已学任务性能下降
3. **参数选择问题**：现有方法（如 prompt-based、MoE-Adapters）需要在推理时选择对应的参数集，容易出错

### 为什么这个问题重要？
- CLIP 作为预训练视觉-语言模型，具有强大的可迁移性，是持续学习的理想候选
- 现实应用中需要模型能增量学习多个任务，同时保持预训练的零样本能力
- 稳定性（记忆旧知识）与可塑性（学习新知识）的权衡是持续学习的核心挑战

### 在此之前最好的方法是什么？它有什么局限？
| 方法 | 核心思路 | 局限 |
|------|---------|------|
| **ZSCL** (Zheng et al., 2023) | 从 vanilla CLIP 蒸馏知识 | 仍可能更新预训练参数，无法完全保留预训练知识 |
| **MoE-Adapters** (Yu et al., 2024) | 在图像编码器中插入混合适配器 | 需要预定义适配器数量，推理时需激活部分参数，易导致错误分配 |
| **RAIL** (Xu et al., 2024) | 扩展分类器维度，保持特征表示固定 | 依赖 vanilla CLIP 作为选择器区分已学/未学任务，易导致误差传播 |

---

## 核心方法

### 核心思路（一句话，用直觉说清楚）
LADA 在冻结的 CLIP 图像编码器后追加轻量级的**标签特定记忆单元**，将所有任务信息压缩到统一表示空间，通过特征蒸馏防止灾难性遗忘，无需推理时的参数选择。

### 核心公式

#### 1. 标签特定特征构建
对于任务 $\mathcal{T}^k$ 的类别 $j$，用 $k$-means 聚类得到 $\lambda_1$ 个聚类中心作为记忆单元：
$$
\boldsymbol{W}_{j}^{k} \in \mathbb{R}^{\lambda_{1} \times d} = \left[\boldsymbol{w}_{j}^{k}(1), \dots , \boldsymbol{w}_{j}^{k} (\lambda_1)\right]
$$

标签特定特征映射：
$$
\varphi^{k}(\boldsymbol{i}) = \left[\boldsymbol{W}_{1}^{k} \boldsymbol{i},  \dots , \boldsymbol{W}_{M^{k}}^{k} \boldsymbol{i} \right]
$$

最终特征表示（聚合所有任务）：
$$
\varphi(\boldsymbol{i}) = [\varphi^1(\boldsymbol{i}), \cdots , \varphi^k (\boldsymbol{i})]
$$

#### 2. 固定分类器
$$
\left(h \circ \varphi\right)(\boldsymbol{i})_{j}^{i} =  \phi ( \boldsymbol{W}_{j}^{i} \boldsymbol{i}) \boldsymbol{1}
$$
其中 $\phi = \exp(-\beta(1-x))$ 将内积转换为非负值。

#### 3. 训练损失
当前任务损失（Eq. 7）：
$$
\frac{1}{|\mathcal{D}^{k}_{j}|} \sum_{\boldsymbol{v} \in \mathcal{D}^{k}_{j}} - \log \frac{e ^{ \left(h \circ \varphi\right)(f_{I} (\boldsymbol{v} ))^{k}_{j}}}{\sum\limits_{n \in [k]} \sum\limits_{m \in [M^{n}]} e^ { \left(h \circ \varphi\right)(f_{I} (\boldsymbol{v} ))_{m}^{n}}}
$$

旧任务蒸馏损失（Eq. 8）：
$$
\frac{1}{\lambda_{2}} \sum_{l=1}^{\lambda_{2}}  - \log \frac{e ^{ \left(h \circ \varphi\right)(\boldsymbol{p}^{i}_{j}(l))_{j}^{i}}  }{\sum\limits_{n \in [k]} \sum\limits_{m \in [M^{n}]} e^ { \left(h \circ \varphi\right)(\boldsymbol{p}^{i}_{j} (l))_{m}^{n}}}
$$

#### 4. 分布保持训练（DPT）
用高斯混合模型拟合旧任务分布：
$$
\{\pi_{j}^{i}(l), \boldsymbol{p}_{j}^{i} (l), \boldsymbol{\Sigma}_{i}^{j}(l) \}_{l=1}^{\lambda_{2}} = \mathrm{GMM} (\mathcal{D}_{j}^{i})
$$

增强原型：
$$
\widetilde{\boldsymbol{p}}_{j}^{i}(l) = \boldsymbol{p}_{j}^{i}(l) + \boldsymbol{e} \cdot\sqrt{\frac{\operatorname{Tr}\left(\boldsymbol{\Sigma} ^{i}_{j}(l)\right)}{d}}
$$

### 公式推导路径
1. **问题分析**：现有方法需要推理时参数选择 → 错误分配风险
2. **核心洞察**：将所有任务信息压缩到统一空间，无需选择
3. **标签特定记忆**：用聚类中心作为每个类别的记忆单元
4. **特征蒸馏**：冻结旧任务记忆单元，用蒸馏原型保持旧任务性能
5. **分布保持**：用 GMM 增强原型，更好保留分布信息

---

## 实验设计

### 用的什么数据集/benchmark？和哪些 baseline 比较？
**数据集**：X-TAIL（10 个数据集，1100 个类别）
- Aircraft, Caltech101, DTD, EuroSAT, Flowers, Food101, MNIST, OxfordPets, StanfordCars, SUN397

**设置**：
- 16-shot（每类 16 个样本）
- Full-shot（完整数据集）

**Baseline**：
- Zero-shot（CLIP 零样本）
- LwF, WiSE-FT
- ZSCL, MoE-Adapters
- Primal-RAIL, Dual-RAIL

**评估指标**（来自 Zheng et al., 2023）：
- **Transfer**：衡量前向遗忘（训练任务 $j$ 后在未来任务 $k+1,\dots,K$ 上的平均准确率）
- **Average**：所有时间步的平均准确率
- **Last**：最终性能（衡量后向遗忘）

### 核心结果是什么？（用具体数字）

#### X-TAIL 16-shot
| 方法 | Transfer | Average | Last |
|------|----------|---------|------|
| Zero-shot | 57.7 | 57.7 | 57.7 |
| ZSCL | 59.0 | 60.0 | 63.4 |
| Dual-RAIL | - | 71.3 | 82.3 |
| **LADA (Ours)** | **61.5** | **72.7** | **83.1** |

**改进**：Transfer +2.5%, Average +1.4%, Last +0.8%（相比 Dual-RAIL）

#### X-TAIL Full-shot
| 方法 | Transfer | Average | Last |
|------|----------|---------|------|
| Zero-shot | 57.7 | 57.7 | 57.7 |
| ZSCL | 59.0 | 64.5 | 72.1 |
| Primal-RAIL | - | 72.8 | 84.0 |
| **LADA (Ours)** | **61.9** | **75.2** | **86.9** |

**改进**：Transfer +2.9%, Average +2.4%, Last +2.9%（相比 Primal-RAIL）

**亮点**：
- 在 Flowers、Food、SUN397 等数据集上，Transfer 准确率超过 vanilla zero-shot CLIP
- 无需像 RAIL 那样用 vanilla CLIP 作为选择器，减少误差传播

---

## 作者自己说的局限

论文结论部分明确指出的不足/未来工作：
1. （论文未在结论中明确列出主要局限，但从实验可推断）
2. 虽然 LADA 比 RAIL 更高效，但仍需要存储每个类别的记忆单元
3. Full-shot 下 Dual-RAIL 因内存限制无法完成训练，LADA 在此场景下更具优势
