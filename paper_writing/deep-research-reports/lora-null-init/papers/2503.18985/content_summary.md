# LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning 内容总结
- **作者**: Xuan Liu, Xiaobin Chang | **年份**: 2025 | **会议/期刊**: arXiv:2503.18985v2 (2025.03)

## 问题定义

- **论文试图解决什么问题？**
  在 Exemplar-Free Continual Learning (EFCL) 设定下，模型在学习新任务时，由于无法保留旧任务的样本，旧任务的特征表征会随时间发生 **feature drift（特征漂移）**，导致灾难性遗忘。现有方法依赖从旧任务存储的静态特征或过时统计量来减轻漂移，但无法捕获 CL 中特征空间的动态演化，随着任务数增加性能显著下降。

- **为什么这个问题重要？**
  EFCL 在许多实际场景中不可避免——隐私法规（如 GDPR）禁止存储用户数据，或设备存储容量有限无法保存历史样本。在无样本回放的情况下，特征漂移是灾难性遗忘的核心原因。如果能够有效控制特征漂移，就能在完全不依赖旧样本的前提下实现长序列的持续学习，具有重大的实际应用价值。

- **在此之前最好的方法是什么？它有什么局限？**
  此前有三类代表性方法用于减轻特征漂移：

  1. **InfLoRA** (Liang & Li, 2024)：通过旧任务的梯度信息设计 LoRA 降维矩阵的子空间，减少新任务对旧任务的干扰。局限：依赖存储的旧任务统计量，随任务数增加这些统计量变得过时，仍然表现出持续的特征漂移。

  2. **Adam-NSCL** (Wang et al., 2021)：使用旧任务输入特征的协方差矩阵的近似的零空间（null space）来约束参数更新。局限：同样依赖存储的静态旧任务统计数据，且需要额外的内存来存储这些特征。

  3. **EASE** (Zhou et al., 2024)：设计语义引导的原型补充策略，在新特征空间中重新计算原型。局限：其子空间创建依赖于旧任务特征，随着 CL 的进行无法准确反映动态特征空间的演变。

  这些方法的**共同根本局限**：它们创建的子空间基于旧任务的**静态**信息（旧特征或旧统计量），但 CL 中特征空间是**动态**演化的。旧统计量随着新任务的加入变得过时，导致性能随着任务数增加明显下降（如 Fig.1 所示的上行漂移趋势和 Tab.1-3 中长任务序列的显著性能衰减）。

## 核心方法

- **核心思路（一句话）**
  **不去显式建模或存储旧任务特征，而是在参数空间中通过"减去旧任务的 LoRA 权重"来构造一个 Drift-Resistant Space（DRS），让新任务的学习在这个空间中进行，从而在不依赖旧样本的前提下有效控制特征漂移。**

- **核心公式**

  **Stage 1: LoRA Subtraction 构造 DRS**
  
  $$
  \underbrace{\mathbf{V}_{t-1}}_{\text{旧任务向量}} = \mathbf{W}_{t-1} - \mathbf{W}_0 = \sum_{j=1}^{t-1} \mathbf{B}_j \mathbf{A}_j \quad \text{(累积 LoRA 权重)}
  $$
  
  $$
  \underbrace{\widetilde{\mathbf{W}}_t}_{\text{减法后的权重}} = \underbrace{\mathbf{W}_0}_{\text{预训练权重}} - \underbrace{\mathbf{V}_{t-1}}_{\text{旧任务向量}} = \mathbf{W}_0 - \sum_{j=1}^{t-1} \mathbf{B}_j \mathbf{A}_j \quad \text{(LoRA Subtraction)}
  $$
  
  $$
  \underbrace{\mathbf{\Sigma}^l}_{\text{输入协方差矩阵}} = \frac{1}{n_t} (\widetilde{\mathbf{X}}_t^l)^T \widetilde{\mathbf{X}}_t^l \quad \text{(在减法后的模型中计算新任务输入特征)}
  $$

  **Stage 2: 在 DRS 中训练新任务**
  
  $$
  \underbrace{\mathbf{U}^l \mathbf{\Lambda}^l (\mathbf{U}^l)^T}_{\text{SVD 分解}} = \text{SVD}(\mathbf{\Sigma}^l)
  $$
  
  $$
  \underbrace{\mathbf{P}^l}_{\text{DRS 投影矩阵}} = (\mathbf{U}^l)_k \quad \text{(取前 } k \text{ 个主成分，对应最大奇异值)}
  $$
  
  $$
  \underbrace{\Delta \mathbf{w}_{t,s}^l}_{\text{投影后的梯度更新}} = \mathbf{P}^l (\mathbf{P}^l)^T \mathbf{g}_{t,s}^l \quad \text{(将梯度投影到 DRS)}
  $$

- **Augmented Triplet Loss (ATL)**
  
  $$
  \mathcal{L}_{TL} = \max(0, \epsilon_{ap} - \epsilon_{an} + \epsilon)
  $$
  $$
  \epsilon_{an} = \min\Big( \min_{y_{i,z} \ne y_{i,j}} \|M_{\theta_t}(x_{i,j}) - M_{\theta_t}(x_{i,z})\|_2, \min_{p \in \mathcal{P}_{t-1}} \|M_{\theta_t}(x_{i,j}) - p\|_2 \Big)
  $$
  $$
  \mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{TL}
  $$

- **公式推导路径（从问题定义到核心公式的逻辑链）**

  1. **问题定义**：EFCL 中无旧样本，新任务训练会导致旧任务特征漂移 → 灾难性遗忘。
  
  2. **现有方法的局限**：试图用旧任务的静态特征/统计量构建干扰最小化空间 → 特征空间动态变化，静态信息很快过时 → 漂移控制效果差。
  
  3. **关键洞察**：与其建模旧任务特征，不如在 **参数空间** 层面直接消除旧任务的影响。受 Task Arithmetic (Ilharco et al., 2023) 启发——"任务向量"（fine-tuned 权重 - 预训练权重）可以被 negate 来"遗忘"特定任务的知识。
  
  4. **Stage 1 — LoRA Subtraction**：将旧任务的 LoRA 权重（即任务向量）从预训练权重中减去，使模型在新任务数据到来前就"遗忘"了旧任务的知识。然后在此修改后的模型上计算新任务输入特征 → 得到 DRS。
  
  5. **Stage 2 — DRS 投影训练**：对新任务输入特征矩阵做 SVD，取主成分方向构建 DRS 投影矩阵。每个训练步的梯度先投影到 DRS 再更新权重，从而确保新任务的学习不干扰旧任务的知识。
  
  6. **ATL 增强塑性**：纯投影训练会限制模型的塑性（plasticity），引入 triplet loss 并利用旧任务的原型作为负样本来类别分离，在不牺牲稳定性的前提下增强新类别的可分性。

## 实验设计

- **使用的数据集/benchmark 和 baseline 比较**

  **数据集**：
  - **ImageNet-R**：200 类，应用艺术变换的鲁棒性基准（CL 中 PEFT 方法的标准 benchmark）
  - **CIFAR-100**：100 类小规模图像

  **任务序列长度**：10 / 20 / 25 / 50 个增量任务

  **Backbone**：预训练 ViT-B/16-IN21K

  **Baselines**：
  - **LoRA-FT**：依次用 LoRA 微调所有任务（无特殊漂移控制）
  - **L2P** (Wang et al., 2022), **DualPrompt** (Wang et al., 2022), **CODA-Prompt** (Smith et al., 2023)：基于 Prompt 的 EFCIL 方法
  - **LAE** (Gao et al., 2023)：统一 PEFT 框架
  - **InfLoRA** (Liang & Li, 2024)：基于梯度子空间的方法
  - **Adam-NSCL** (Wang et al., 2021)：基于零空间的方法
  - **EASE** (Zhou et al., 2024)：可扩展子空间集成方法

- **核心结果（具体数字）**

  **ImageNet-R (Tab.2)**：
  | 任务数 | 指标 | Ours | EASE (亚军) | 领先差距 |
  |-------|------|------|------------|---------|
  | 10 | ACC₁₀ | 81.16 | 81.67 | -0.51 (相当) |
  | 10 | ACC 平均 | 74.74 | 75.94 | -1.20 |
  | 25 | ACC₂₅ | 74.19 | 72.69 | **+1.50** |
  | 25 | ACC 平均 | 80.06 | 79.65 | **+0.41** |
  | 50 | ACC₅₀ | **72.12** | 68.54 | **+3.58** |
  | 50 | ACC 平均 | **77.94** | 75.77 | **+2.17** |

  **CIFAR-100 (Tab.3)**：
  | 任务数 | 指标 | Ours | EASE (亚军) | 领先差距 |
  |-------|------|------|------------|---------|
  | 10 | ACC₁₀ | 89.14 | 88.34 | +0.80 |
  | 20 | ACC₂₀ | 88.69 | 82.21 | **+6.48** |
  | 25 | ACC₂₅ | 88.39 | 85.01 | **+3.38** |
  | 25 | ACC 平均 | 92.02 | 89.98 | +2.04 |
  | 50 | ACC₅₀ | **87.29** | 82.10 | **+5.19** |
  | 50 | ACC 平均 | **91.29** | 87.65 | **+3.64** |

  **Backward Transfer (BWT, Tab.4)**：
  - CIFAR-100：Ours = **-3.38**（LoRA-FT = -13.08, EASE = -5.54, InfLoRA = -3.70）
  - ImageNet-R：Ours = **-3.90**（LoRA-FT = -27.52, EASE = -4.53, Adam-NSCL = -3.91）
  - ★ Ours 在 CIFAR-100 上 BWT 最优，在 ImageNet-R 上 BWT 第二（仅略低于 Adam-NSCL 的 -3.91），但结合 ACC 看总体最优。

  **Ablation (Tab.5)**：
  - 无 DRS + 无 ATL（纯 LoRA-FT）：50-task ACC = 58.56
  - +DRS 但无 ATL：50-task ACC = 76.03（+17.47）
  - +DRS + ATL（完整方法）：50-task ACC = **77.93**（+1.90）

  **DRS vs InfLoRA vs Adam-NSCL (Fig.3)**：在 50-task ImageNet-R 上，DRS 的 ACC 曲线保持稳定较高水平，InfLoRA 和 Adam-NSCL 初始略高但随任务增加显著下降。

  **ATL 对漂移控制的依赖性 (Tab.7)**：
  - InfLoRA + ATL：50-task ACC 从 66.27 下降到 **61.46**（-4.81）
  - 说明：没有有效的漂移控制，ATL 反而损害性能。

## 作者自己说的局限

1. **LoRA 模块的开销**：虽然比全参数微调参数少得多，但适配器组件仍然引入了增量模型大小开销。未来工作可以探索将适配机制无缝集成到架构中（而非作为附加模块）。
2. **论文未明确提及的其他注意事项**：
   - 方法建立在 **PEFT on PTM** 范式之上，依赖一个强预训练模型（ViT-B/16-IN21K）；如果 PTM 的质量不足，方法的有效性可能受影响。
   - DRS 的构造依赖新任务数据 D_t 在 LoRA Subtraction 后的模型上计算输入特征，这意味着 DRS 本身是**任务相关**的（每个新任务都要重新计算），计算开销随任务数线性增长。
   - 累积 LoRA 减法 $\mathbf{W}_0 - \sum \mathbf{B}_j \mathbf{A}_j$ 在长任务序列中可能导致数值漂移或权重偏离预训练分布过远。
   - 仅在大规模图像分类 benchmark 上验证（ImageNet-R, CIFAR-100），未在 NLP、视觉-language 或多模态 CL 设定上验证。
