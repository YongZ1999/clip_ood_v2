# PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models 本质分析

## 核心公式推导（严谨推导）

### 起点：LoRA 的前向与初始化

给定预训练权重矩阵 $\mathbf{W} \in \mathbb{R}^{m \times n}$，LoRA 的前向为：

$$
\mathbf{Y} = \mathbf{X}(\mathbf{W} + \Delta \mathbf{W}) = \mathbf{X}(\mathbf{W} + \mathbf{A}\mathbf{B}) \tag{1}
$$

其中 $\mathbf{A} \in \mathbb{R}^{m \times r}$, $\mathbf{B} \in \mathbb{R}^{r \times n}$，$r \ll \min(m, n)$。

LoRA 初始化：
$$
\mathbf{A} \sim \mathcal{N}(0, \sigma^2), \quad \mathbf{B} = \mathbf{0} \tag{2}
$$

**动机**：使初始时 $\mathbf{A}\mathbf{B} = \mathbf{0}$，不改变模型输出。

**梯度分析**：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{A}} = \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \mathbf{B}^T, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{B}} = \mathbf{A}^T \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \tag{3}
$$

由于 $\mathbf{B} = \mathbf{0}$，$\frac{\partial \mathcal{L}}{\partial \mathbf{A}} = \mathbf{0}$ — A 的梯度初始为零。
$\frac{\partial \mathcal{L}}{\partial \mathbf{B}}$ 仅通过"噪声器" A 传播，方向随机。

> **这就是 LoRA 初始化的核心缺陷**：A 不更新，B 方向随机，前几步实际上是在"醒过来"而不是在"学东西"。

---

### PiSSA 的推导

**步骤 1：对 $\mathbf{W}$ 做经济 SVD**

$$
\mathbf{W} = \mathbf{U} \mathbf{S} \mathbf{V}^T \tag{4}
$$

其中：
- $\mathbf{U} \in \mathbb{R}^{m \times \min(m,n)}$，列正交（左奇异向量）
- $\mathbf{V} \in \mathbb{R}^{n \times \min(m,n)}$，列正交（右奇异向量）
- $\mathbf{S} = \operatorname{diag}(s_1, s_2, \ldots, s_{\min(m,n)})$，$s_1 \geq s_2 \geq \cdots \geq s_{\min(m,n)} \geq 0$

**动机**：SVD 将 $\mathbf{W}$ 分解为相互正交的方向（奇异向量）及其强度（奇异值），使得我们可以按"重要性"（即奇异值大小）拆分权重。

**步骤 2：按奇异值大小拆分为主成分和残差**

取前 $r$ 个最大的奇异值作为"主成分"：

$$
\mathbf{W} = \underbrace{\mathbf{U}_{:,:r} \mathbf{S}_{:r,:r} \mathbf{V}_{:r,:}^T}_{\text{主成分 } \mathbf{W}^{\text{prin}}} + \underbrace{\mathbf{U}_{:,r:} \mathbf{S}_{r:,:} \mathbf{V}_{:,r:}^T}_{\text{残差 } \mathbf{W}^{\text{res}}} \tag{5}
$$

> **为什么这一步要这样做？**
> 大奇异值对应 $\mathbf{W}$ 中对输出影响最大的方向（即 $\mathbf{W}$ 在哪些方向上有最强的"拉伸"效应）。把这些方向提取出来用全精度适配器保留，剩下的"弱方向"可以量化或冻结。

**步骤 3：用主成分初始化适配器**

关键是巧妙地将 $\mathbf{S}$ 的平方根分配到 $\mathbf{A}$ 和 $\mathbf{B}$ 中：

$$
\mathbf{A} = \mathbf{U}_{:,:r} \mathbf{S}_{:r,:r}^{1/2} \in \mathbb{R}^{m \times r} \tag{6}
$$

$$
\mathbf{B} = \mathbf{S}_{:r,:r}^{1/2} \mathbf{V}_{:r,:}^T \in \mathbb{R}^{r \times n} \tag{7}
$$

这样 $\mathbf{A}\mathbf{B} = \mathbf{U}_{:,:r} \mathbf{S}_{:r,:r}^{1/2} \cdot \mathbf{S}_{:r,:r}^{1/2} \mathbf{V}_{:r,:}^T = \mathbf{U}_{:,:r} \mathbf{S}_{:r,:r} \mathbf{V}_{:r,:}^T = \mathbf{W}^{\text{prin}}$。

> **为什么把 $\mathbf{S}^{1/2}$ 分开放而不直接放 $\mathbf{S}$？**
> 因为 $\mathbf{A}$ 和 $\mathbf{B}$ 是秩 $r$ 矩阵，乘积 $\mathbf{A}\mathbf{B}$ 的秩最多为 $r$。将 $\mathbf{S}$ 矩阵的平方根分别乘入 $\mathbf{A}$ 和 $\mathbf{B}$，可使 $\mathbf{A}\mathbf{B}$ 精确等于 $\mathbf{U}_{:,:r}\mathbf{S}_{:r,:r}\mathbf{V}_{:r,:}^T$（秩 $r$ 矩阵）。如果直接放 $\mathbf{S}$ 到其中一个矩阵会破坏维度兼容性或丢失秩约束。这是 SVD 分解与 LoRA 架构精确对齐的设计选择。

**步骤 4：构建冻结的残差矩阵**

$$
\mathbf{W}^{\text{res}} = \mathbf{U}_{:,r:} \mathbf{S}_{r:,:} \mathbf{V}_{:,r:}^T \in \mathbb{R}^{m \times n} \tag{8}
$$

初始时精确满足：
$$
\mathbf{W} = \mathbf{W}^{\text{res}} + \mathbf{A}\mathbf{B} \tag{9}
$$

因此前向完全等价于原始模型：
$$
\mathbf{Y} = \mathbf{X}(\mathbf{W}^{\text{res}} + \mathbf{A}\mathbf{B}) = \mathbf{X}\mathbf{W} \tag{10}
$$

**步骤 5：训练过程中的梯度**

PiSSA 的梯度形式与 LoRA 相同（因为前向结构相同）：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{A}} = \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \mathbf{B}^T, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{B}} = \mathbf{A}^T \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \tag{11}
$$

但关键差异在**数值**层面：PiSSA 的 $\mathbf{A}, \mathbf{B}$ 初始值即包含 $\mathbf{W}$ 的主成分信息（$s_{:r} \gg s_{r:}$），因此：
- $\frac{\partial \mathcal{L}}{\partial \mathbf{A}}$ 不为零（因 $\mathbf{B} \neq \mathbf{0}$）
- $\frac{\partial \mathcal{L}}{\partial \mathbf{B}}$ 沿主成分方向传播（因 $\mathbf{A}$ 包含主奇异向量）

> **与 LoRA 的本质差异**：相同的公式，但不同的初始值导致优化路径完全不同。PiSSA 第一歩的梯度方向就是有信息量的，而 LoRA 需要多步热身。

---

### QPiSSA 量化误差推导

**QLoRA 的量化误差**：
$$
\text{QuantErr}_{\text{QLoRA}} = \|\mathbf{W} - (\operatorname{nf4}(\mathbf{W}) + \mathbf{A}\mathbf{B})\|_* = \|\mathbf{W} - \operatorname{nf4}(\mathbf{W})\|_* \tag{12}
$$

其中 $\|\cdot\|_*$ 为核范数（奇异值之和），$\operatorname{nf4}(\cdot)$ 为 4-bit NormalFloat 量化。

**QPiSSA 的量化误差**：
$$
\text{QuantErr}_{\text{QPiSSA}} = \|\mathbf{W} - (\operatorname{nf4}(\mathbf{W}^{\text{res}}) + \mathbf{A}\mathbf{B})\|_* = \|\mathbf{W}^{\text{res}} - \operatorname{nf4}(\mathbf{W}^{\text{res}})\|_* \tag{13}
$$

> **为什么 QPiSSA 误差更小？**
> $\mathbf{W}^{\text{res}}$ 去除了大奇异值的主分量，其值的分布更窄且更接近高斯分布（均值和标准差均更小），而 NF4 量化是为高斯分布优化的。量化误差与数据分布范围正相关 — 分布越窄，量化误差越小。

---

### 训练后等价转换为 LoRA 格式

训练后，PiSSA 适配器变为 $\mathbf{A}'\mathbf{B}'$，权重变化为：
$$
\Delta \mathbf{W} = \mathbf{A}'\mathbf{B}' - \mathbf{A}\mathbf{B} = [\mathbf{A}', \mathbf{A}] \begin{bmatrix} \mathbf{B}' \\ -\mathbf{B} \end{bmatrix} = \mathbf{A} \mathbf{\Delta} \mathbf{B} \tag{14}
$$

其中 $\mathbf{A} \Delta \in \mathbb{R}^{m \times 2r}$, $\Delta \mathbf{B} \in \mathbb{R}^{2r \times n}$。这样即可将 PiSSA 适配器保存为与 LoRA 完全兼容的格式，无需在部署时做 SVD。

> 这里存在一个存储开销：从 $mr + nr$ 变为 $2mr + 2nr$，但由于 $r$ 很小，仍是可接受的。

---

## 核心假设（显式 + 隐式）

### 显式假设

1. **低秩假设**：$\Delta \mathbf{W}$（微调导致的权重变化）是低秩的。这是 LoRA 的原始假设，PiSSA 继承了这个假设的框架形式，但改变了语义。

2. **奇异值长尾分布假设**：$\mathbf{W}$ 的大奇异值数量远小于小奇异值，"前 $r$ 个奇异值远大于其余"，即 $s_{:r} \gg s_{r:}$。论文中 Figure 3 和 Appendix G 展示了这一现象。

3. **主成分方向与微调方向一致假设**：$\mathbf{W}$ 的主奇异向量方向是微调时最有用的方向，"fine-tuning the principal components matches the behavior of fine-tuning the full matrix"。

### 隐式假设

> **假设 A — 对"主方向 = 好方向"的隐含假定**
>
> **在哪里引入**：Section 3（公式 4-5 的分解逻辑）和 Section 5.2（"PiSSA is a denoised version of full fine-tuning"）。
>
> PiSSA 的核心思路是"分解 $\mathbf{W}$，保留主方向训练，冻结小方向"。这隐含假设了：**预训练权重中的主成分方向（即对原始模型输出影响最大的方向）也是微调时最有信息量的梯度方向。**
>
> 但这个假设是否一定成立？考虑以下情形：
> - 微调任务与预训练任务分布差异很大（例如从通用语料到高度专业化的医学领域），此时预训练权重最重要的方向可能根本不是下游任务需要调整的方向；
> - 预训练权重的"主成分"可能编码了高频语法模式或常见语义模板，而下游任务需要的是"微调"权重的尾部精细调整。
>
> **在什么条件下成立**：当预训练分布与下游分布高度重叠，微调是"在已有能力上精调"而非"学习全新能力"时。
>
> **可能不成立的条件**：领域迁移幅度很大、下游任务需要学习的表征与预训练主成分正交时。

> **假设 B — 奇异值截断不会丢失关键信息**
>
> **在哪里引入**：公式 5 将 $\mathbf{W}$ 截断为 $\mathbf{W}^{\text{prin}}$ 和 $\mathbf{W}^{\text{res}}$。
>
> PiSSA 将 $\mathbf{W}_{:,r:}\mathbf{S}_{r:,:}\mathbf{V}_{:,r:}^T$ 这部分冻结（即不训练）。这隐含了：**残差奇异值对应的方向对微调不重要，可以在训练中完全冻结。**
>
> 实际上 $\mathbf{W}^{\text{res}}$ 包含的是非零但较小的奇异值方向。虽然单个方向强度弱，但数量远多于主成分方向。是否存在累积效应？即：残差方向整体在微调中也可能贡献可观的表达力？
>
> 论文通过实验（Appendix F，三种奇异值初始化的对比）验证了主成分初始化的优越性，但**没有直接证明冻结残差**不带来损失 — 即没有对比"PiSSA"和"用主成分初始化但允许所有方向更新"。
>
> **可能不成立的条件**：当 $r$ 选择过小，截断了实际上对下游任务有信息量的方向时。

> **假设 C — 梯度传播的向量方向假设**
>
> **在哪里引入**：Section 3，公式（11）的梯度分析。
>
> 论文论证 PiSSA 的梯度方向比 LoRA 更稳定、更一致。他们用实验（Table 11）展示了不同随机种子下 PiSSA 的梯度方向始终为 $[0,1]$ 或 $[1,0]$（二维投影）。但这隐含假设了：**梯度的方向一致性与梯度的质量（即能否引导到好的局部极小点）正相关。**
>
> 方向一致不一定意味着方向正确 — 一个方法可以一致地沿着某个方向走，但那个方向可能一致地不那么好。论文实际上用最终的 loss 和 accuracy 间接验证了这个假设，但缺乏更严格的收敛性分析。

> **假设 D — SVD 分解的精度无关性**
>
> **在哪里引入**：Appendix B，Fast SVD。
>
> 论文使用 Fast SVD（随机 SVD）替代精确 SVD，声称"niter=2 时训练 loss 接近精确 SVD"。但这隐含假设了：**初始化精度的小误差在训练中可以被纠正**。
>
> 从 Table 6 可以看到，Fast SVD 的初始化误差在 $10^{-4} \sim 10^{-3}$ 量级，而训练 loss 与精确 SVD 的差异在 $10^{-3} \sim 10^{-4}$ 量级。这确实表明误差很小，但论文没有讨论这种近似初始化在不同 rank、不同模型上的鲁棒性边界。

> **假设 E — 可叠加性假设（未经证明）**
>
> **在哪里引入**：Section 3 末尾，"one pre-trained model can accommodate multiple $\Delta\mathbf{A}$, $\Delta\mathbf{B}$, fine-tuned by diverse PiSSA or LoRA procedures"。
>
> 这继承了 LoRA 的多适配器叠加能力。但当同一个模型叠加多个 PiSSA 适配器时，由于每个适配器的 $\mathbf{W}^{\text{res}}$ 来自同一个 $\mathbf{W}$ 的不同分解结果，它们共享相同的残差模型。不同任务对主成分的调整是否会相互干扰？论文没有分析。

---

## 与前驱工作的逻辑关系

### 继承了谁的思路/方法？

1. **LoRA (Hu et al., 2021)** — 直接继承其低秩适配器架构（前向 $\mathbf{Y} = \mathbf{X}(\mathbf{W}^{\text{frozen}} + \mathbf{A}\mathbf{B})$）。PiSSA 实际上只是在 LoRA 的初始化步骤做了替换，其他完全兼容。

2. **SVD 在 PEFT 中的应用思想** — 借鉴了 AdaLoRA 中使用 SVD 术语的思想（$\mathbf{A} \operatorname{diag}(\mathbf{E}) \mathbf{B}$），但根本不同在于 AdaLoRA 是对 $\Delta \mathbf{W}$ 做 SVD 风格的分解，而 PiSSA 是真的对 $\mathbf{W}$ 做 SVD。

3. **LoftQ (Li et al., 2023)** — LoftQ 也是用 SVD 改进初始化，但它是对**量化误差矩阵** $\mathbf{W} - \operatorname{nf4}(\mathbf{W})$ 做 SVD 并提取主分量到适配器。PiSSA 是对 $\mathbf{W}$ 本身做 SVD。

### 解决了前人的什么根本局限？

**Limitation 1：LoRA 的"噪声+零"初始化缺陷**

这是最核心的局限。LoRA 从零信号开始，梯度初始无效 → 需要热身步 → 收敛慢。PiSSA 的解决方式是：不去近似 $\Delta \mathbf{W}$，而去近似 $\mathbf{W}$ 本身，从而让适配器从有信息量的初始状态开始。

**数学上的根本区别**：

LoRA 的哲学：
$$
\Delta \mathbf{W} \approx \mathbf{A}\mathbf{B},\quad \mathbf{A} \sim \mathcal{N}, \mathbf{B} = \mathbf{0} \quad \Rightarrow \quad \text{初始输出不变，但梯度为零} 
$$

PiSSA 的哲学：
$$
\mathbf{W} = \underbrace{\mathbf{W}^{\text{prin}}}_{\text{可训练的 } \mathbf{A}\mathbf{B}} + \underbrace{\mathbf{W}^{\text{res}}}_{\text{冻结}} \quad \Rightarrow \quad \text{初始输出不变，且梯度有信息量}
$$

**Limitation 2：QLoRA 无法降低量化误差**

QLoRA 直接量化 $\mathbf{W}$，量化误差为 $\|\mathbf{W} - \operatorname{nf4}(\mathbf{W})\|_*$。PiSSA 先提取主成分到全精度适配器，只量化残差 $\mathbf{W}^{\text{res}}$，量化误差为 $\|\mathbf{W}^{\text{res}} - \operatorname{nf4}(\mathbf{W}^{\text{res}})\|_*$。由于 $\mathbf{W}^{\text{res}}$ 的分布更窄更高斯，误差自然更小。

**与 LoftQ 的对比**：LoftQ 是在量化**之后**对量化误差做 SVD 补救，而 PiSSA 是在量化**之前**就移除了大值分量。如 Figure 10e vs 10f 所示，LoftQ 只消除了 QLoRA 误差矩阵中最大的 $r$ 个奇异值，而 PiSSA 的误差矩阵本身就小得多。

### 引入/改变了什么约束/假设？

PiSSA 将 LoRA 的假设从"$\Delta \mathbf{W}$ 是低秩的"转向了"$\mathbf{W}$ 的低秩主成分包含了微调需要调整的最重要方向"。这不是对 LoRA 假设的加强或放松，而是**完全改变了假设的语义**。

这带来了一个有意思的悖论：如果 $\mathbf{W}$ 本身不是低秩的（即奇异值分布平坦，没有明显的"长尾"），PiSSA 会面临困难。而 LoRA 的假设（$\Delta \mathbf{W}$ 低秩）与 $\mathbf{W}$ 是否低秩无关。$$ \text{不过，实验中几乎所有 LLM 的权重矩阵都展现出了明显的长尾奇异值分布。} $$

---

## 留下的开放问题

### 悬而未决的问题

1. **理论分析的缺失**：论文没有给出 PiSSA 收敛速度的严格理论证明。为什么用主奇异向量初始化会加速收敛？能否从优化理论（如 Polyak-Łojasiewicz 条件、初始化与 Hessian 矩阵特征向量的对齐）给出定量分析？

2. **为什么在 Gemma-7B 上 rank 增大时 PiSSA 反而先于 LoRA 过参数化？**（Appendix K.2，Figure 15）这是否意味着 PiSSA 的有效秩比 LoRA 更小？或者与 Gemma 特殊的预训练策略有关？

3. **SVD 计算成本**：虽然 Fast SVD 只需几秒，但对超大模型（7B+）的所有线性层（数百个）做 SVD，累积开销是否仍然可观？论文没有报告总初始化时间。

4. **PiSSA 与 LoRA 的混合使用**：如果一个模型的部分层用 PiSSA、部分层用 LoRA 初始化，是否能进一步提升性能？论文没有探索。

5. **精确 SVD vs Fast SVD 的选择准则**：什么情况下选择精确 SVD 而不是 Fast SVD 是必要的？初始化误差边界与训练效果的量化关系未明确。

### 明显但论文没做的实验/分析

1. **消融实验：冻结残差的代价**。PiSSA = 主成分训练 + 残差冻结。应该对比"主成分初始化后允许所有参数更新"（即不冻结 $\mathbf{W}^{\text{res}}$）。这会直接把 PiSSA 与非冻结版本对比，量化冻结残差的损失。

2. **任务相关性分析**。微调任务与预训练任务的相似度与 PiSSA 提升幅度之间的关系。预期：任务越接近预训练分布，PiSSA 优势越大？论文没有做这个分析。

3. **跨任务叠加适配器的干扰分析**。多个 PiSSA 适配器叠加时，共享 $\mathbf{W}^{\text{res}}$ 是否会引入冲突？

4. **与 LoRA 的并置对比：相同 seed 下的优化轨迹可视化**。从优化轨迹的希尔伯特空间投影分析两个方法的路径差异（论文 Table 11 只做了二维投影，且只对比了梯度方向的稳定性）。

5. **奇异值分布与性能的相关性分析**。不同层、不同模型的奇异值分布形状是否影响 PiSSA 的提升幅度？预期：$\frac{s_{:r}}{s_{r:}}$ 比值（即"主成分占比"）越大，PiSSA 提升越显著。论文没有给出这个定量关系。

### 如果我来改进这篇工作，我会从哪里入手？

1. **理论深化**：证明在一定的假设下（如 $\Delta \mathbf{W}$ 与 $\mathbf{W}$ 的主分量方向对齐），PiSSA 的收敛速度至少比 LoRA 快 $O(1/\sigma_{\min}^2(\mathbf{B}_0))$ 倍，其中 $\mathbf{B}_0$ 是 LoRA 的初始 B 矩阵的方差。

2. **自适应 rank 选择**：根据每层奇异值 $\frac{s_r}{s_{r+1}}$ 的比例动态选择不同层的 $r$，而非所有层用统一 rank。这在 AdaLoRA 中已有雏形，但与 PiSSA 的 SVD 初始化天然契合。

3. **多步 SVD 的替代方案**：QPiSSA 的多步 SVD（Algorithm 1）需要交替量化和分解，计算开销随 T 线性增长。是否可以设计一种端到端的量化感知初始化方法，不需要迭代？

4. **灵活的残差处理**：不是简单冻结 $\mathbf{W}^{\text{res}}$，而是对其实施一种"更粗粒度"的微调（如加一个 rank 更小的适配器），以弥补截断损失。

5. **跨模态扩展**：在视觉 Transformer（ViT）上实验 PiSSA，验证其泛化能力。论文自己也指出了这个方向。

---

## 在领域中的定位

### 所属范式/技术路线

PiSSA 属于 **Parameter-Efficient Fine-Tuning (PEFT)** 下的 **Low-Rank Adapter** 路线，具体是 LoRA 的**初始化改进**方法。

与 LoRA 及其变体的关系：

```
PEFT
├── Prompt-based (Prefix Tuning, Prompt Tuning, P-Tuning)
├── Adapter-based (HAdapter, PAdapter, AdapterFusion)
└── Low-Rank Adapter (LoRA family)
    ├── LoRA (2021) — 基础架构
    ├── PiSSA (2024) — 初始化改进（本文）
    ├── AdaLoRA (2023) — 动态 rank 分配 + SVD 式分解
    ├── DoRA (2024) — 权重解耦（方向+幅度）
    ├── LoftQ (2023) — 量化误差 SVD 补偿
    ├── DeltaLoRA (2023) — 用适配器参数更新原权重
    └── ...
```

### 与同一路线中其他工作的关系

- **相对于 LoRA**：PiSSA 是一个简单的"drop-in replacement"（即插即用替换），只改初始化不改架构。最易于落地。
- **相对于 AdaLoRA**：PiSSA 与 AdaLoRA 的关系是正交的且可叠加的。AdaLoRA 在训练中动态调整 rank 和保持正交约束，PiSSA 在初始化时提供更好的起点。论文 Appendix A 验证了 PiSSA + AdaLoRA 的组合效果（78.59% on GSM8K，超过各自单独使用）。
- **相对于 LoftQ**：两者都使用 SVD 改善初始化，但哲学相反 — LoftQ 对量化误差做 SVD，PiSSA 对原权重做 SVD。这使得 PiSSA 在量化场景下天然优于 LoftQ。实际上从 Table 8 看，即使在 1 次迭代时 PiSSA 的量化误差降低就远超 LoftQ 的 5 次迭代。
- **相对于 DoRA**：DoRA 通过解耦方向/幅度引入额外的计算开销（每次前向需 normalize $\mathbf{W} + \mathbf{A}\mathbf{B}$），而 PiSSA 零额外训练开销。二者也可叠加使用。

**PiSSA 在 PEFT 发展中的独特位置**：它提出了一种"不从噪声开始学习，而是从结构开始学习"的范式转换。这一思想后来可能被扩展到其他初始化场景（如权重衰减策略、学习率预热策略）。其"用 SVD 分解预训练权重来获得更好的初始化"的思路，可以看作是 LoRA 从"实习期"（热身慢）到"直接上岗"（从主成分出发）的进化。
