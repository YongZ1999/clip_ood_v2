# Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models (ZSCL) 内容总结

- **作者**: Zheng et al. | **年份**: 2023 | **会议/期刊**: ICCV 2023

---

## 问题定义

### 论文试图解决什么问题？
CLIP 等视觉-语言模型（VLM）在持续学习中的**零样本转移能力退化（Zero-Shot Transfer Degradation）**问题。
- **现象**：在下游任务上微调 CLIP 会显著破坏其预训练的特征空间结构，导致在未见过的分布外（OOD）任务上性能急剧下降。
- **权衡**：如何在提升新任务性能的同时，保持原有的零样本泛化能力？

### 为什么这个问题重要？
- VLM 的核心优势在于强大的零样本迁移能力。
- 传统的持续学习方法（如 LwF, iCaRL）主要关注防止已学任务的遗忘（后向遗忘），而忽视了预训练知识的保持（前向遗忘/转移退化）。
- 现实场景需要模型既能学习新领域，又不丧失作为通用模型的能力。

### 在此之前最好的方法是什么？它有什么局限？
- **LwF / iCaRL**：使用当前任务数据或少量旧样本进行蒸馏。
  - **局限**：下游任务数据分布狭窄（subspace），不足以约束整个庞大的预训练特征空间。
- **WiSE-FT**：在预训练权重和微调权重之间做线性插值。
  - **局限**：对超参数 $\alpha$ 极其敏感，且在持续学习长序列中难以平衡多个阶段。

---

## 核心方法

### 核心思路（一句话，用直觉说清楚）
ZSCL 通过在特征空间引入**参考数据集（Reference Dataset）**进行双向蒸馏（图像到文本、文本到图像），并配合参数空间的**权重集成（Weight Ensemble）**，在不使用私有预训练数据的情况下锁定预训练特征空间的拓扑结构。

### 核心公式

#### 1. 特征空间蒸馏 (ZSCL-Distill)
使用一个冻结的初始 CLIP 作为教师模型 $\overline{f}$，在参考数据集 $\mathcal{D}_{ref}$ 上进行蒸馏：
- **图像蒸馏**：保持图像特征相对于文本特征的相对分布。
$$ \mathcal{L}_{\text{dist\_img}} = -\sum_{j=1}^m \bm{p}_j \cdot \log \overline{\bm{p}}_j $$
- **文本蒸馏**：保持文本特征相对于图像特征的相对分布（反向约束）。
$$ \mathcal{L}_{\text{dist\_txt}} = \text{对称的文本端损失} $$

#### 2. 权重集成 (Weight Ensemble, WE)
受 SWA 启发，在训练过程中对模型参数进行持续平均，而不是仅在任务结束时插值：
$$ \hat{\theta}_t = \frac{1}{t+1} \theta_t + \frac{t}{t+1} \hat{\theta}_{t-1} $$
这提供了一个比 WiSE-FT 更鲁棒的“学习-遗忘”折中。

#### 3. 总损失函数
$$ \mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda (\mathcal{L}_{\text{dist\_img}} + \mathcal{L}_{\text{dist\_txt}}) + \gamma \mathcal{L}_{\text{WC}} $$
其中 $\mathcal{L}_{\text{WC}}$ 是权重巩固正则项（可选）。

---

## 实验设计

### 用的什么数据集/benchmark？和哪些 baseline 比较？
**数据集**：
- **MTIL (Multi-domain Task Incremental Learning)**：11 个跨域数据集（Aircraft, Caltech101, EuroSAT 等），共 1201 类。
- **传统 CIL**：CIFAR100, TinyImageNet.
- **参考数据集**：ImageNet-100k, Conceptual Captions (CC).

**Baseline**：
- Zero-shot CLIP, Fine-tune
- LwF, iCaRL, LwF-VR
- WiSE-FT

**指标**：
- **Transfer**：衡量零样本能力保持（上三角区域均值）。
- **Avg**：所有步所有任务均值。
- **Last**：最终时刻所有任务均值。

### 核心结果是什么？
- **MTIL 性能**：ZSCL 在 Transfer 指标上仅比原始 CLIP 下降 1.3%（从 69.4% 到 68.1%），而普通微调下降了 24.8%。
- **Avg 指标**：比原始 CLIP 提升 10.1%（从 65.3% 到 75.4%）。
- **鲁棒性**：WE 相比 WiSE-FT 对采样间隔不敏感，且在不同任务顺序（Order I/II）下表现一致。

---

## 作者自己说的局限
1. **参考数据集依赖**：虽然不需要原预训练数据，但仍需一个规模足够（如 100k）且语义丰富的参考集。
2. **计算开销**：双向蒸馏增加了训练时的推理前向传播次数。
3. **文本模板敏感性**：文本端蒸馏仍依赖于 prompt 模板的质量。
