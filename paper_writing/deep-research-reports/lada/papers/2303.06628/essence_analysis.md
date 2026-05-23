# Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models (ZSCL) 本质分析

---

## 核心公式推导（严谨推导）

### 从特征空间扭曲到分布保持

#### Step 1: 问题的数学本质
微调 CLIP 时，损失函数 $\mathcal{L}_{\text{CE}}$ 仅作用于当前任务的类别集合 $C^i$。这导致图像特征 $\bm{f}_i(\bm{x})$ 朝着当前类别 $\bm{t}_j$ 剧烈移动。由于预训练模型是对比学习训练的，特征空间是高度耦合的拓扑结构。
**推论**：局部区域的特征漂移会导致全局拓扑（Relative Distance）的坍塌，这就是 Transfer Degradation 的根源。

#### Step 2: 为什么 LwF 失效？
LwF 使用当前任务数据 $\mathcal{D}^i$ 进行蒸馏。
$$ \mathcal{L}_{\text{LwF}} = \text{KL}(\text{Softmax}(\bm{f}_i(\bm{x}_{\text{curr}})), \text{Softmax}(\overline{\bm{f}}_i(\bm{x}_{\text{curr}}))) $$
**缺陷**：$\bm{x}_{\text{curr}}$ 仅占据特征空间的一个极小超平面（Subspace）。在 Subspace 之外的区域，模型完全没有约束，处于“野蛮生长”状态。

#### Step 3: 参考数据集的引入
为了覆盖整个特征空间，需要满足：
$$ \text{Span}(\mathcal{D}_{\text{ref}}) \approx \text{Span}(\mathcal{D}_{\text{pre-train}}) $$
ZSCL 证明了：不需要 *匹配* 的图文对，只需要 *足够语义* 的独立图、文集合。
**逻辑**：通过计算图像相对于 *大量随机/丰富文本* 的相似度分布 $\bm{p}$，可以刻画该图像在原始特征空间中的“定位（Location）”。

#### Step 4: 权重集成 (WE) 的优越性
WiSE-FT 是后验插值：$\theta = (1-\alpha)\theta_0 + \alpha\theta_1$。
WE 是过程均值：$\hat{\theta} = \frac{1}{T}\sum_{t=1}^T \theta_t$。
**数学直觉**：在非凸优化平面上，WiSE-FT 是两点间的线段，而 WE 是训练轨迹的质心。WE 往往能落在更平滑、泛化性更好的极小值区域。

---

## 核心假设

### 显式假设
1. **参考数据集可用性**：假设可以获得一个通用的、语义丰富的公开数据集（如 ImageNet/CC）作为代理。
2. **教师模型恒定**：假设 *初始* CLIP 是最佳教师，而不是 *上一任务* 的模型。
   - **理由**：上一任务的模型已经产生了偏差，连续蒸馏会产生“漂移累积”。

### 隐式假设
1. **线性连通性 (Linear Mode Connectivity)**：WE 的有效性隐含假设了微调前后的模型位于同一个损失盆地（Loss Basin），否则权重平均会失效。
2. **文本编码器稳定性**：ZSCL 主要微调图像编码器，隐式假设文本编码器不需要大规模调整，或者文本空间的结构比图像空间更稳定。
3. **特征空间对齐度**：假设余弦相似度分布足以完全刻画特征空间的结构。

---

## 与前驱工作的逻辑关系

### 继承与挑战
- **继承**：继承了 LwF 的蒸馏框架和 WiSE-FT 的参数平均思想。
- **挑战**：挑战了“持续学习需要旧样本（Replay）”的传统认知。ZSCL 证明了 VLM 场景下，**无标签的第三方参考数据** 比 **少量的任务相关旧样本** 更能保持模型能力。

### 解决了什么瓶颈？
- 解决了微调 CLIP 带来的“降维打击”——即模型从通用 VLM 退化为特定领域的分类器。

---

## 留下的开放问题

1. **参考数据集的最小规模**：到底需要多少数据才能代表“通用空间”？消融实验显示 10k 到 100k 有提升，但更往上呢？
2. **计算负担**：每次迭代都要在参考集上做推理，对于大规模持续学习任务（如 100 步）压力很大。
3. **任务内遗忘 vs 领域间转移**：ZSCL 强于 Transfer，但在极致的 Class-incremental 任务中，其防止旧任务遗忘的能力（Last）是否会被专门的 Replay 方法超越？
