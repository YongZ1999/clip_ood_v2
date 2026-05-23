# MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning 本质分析

## 核心公式推导（严谨推导）

### 起点：SVD 分解与权重拆分

给定预训练权重矩阵 $\mathbf{W} \in \mathbb{R}^{m \times n}$（不失一般性，设 $m \leq n$），其奇异值分解为：

$$
\mathbf{W} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^{\top} = \sum_{i=1}^{m} \sigma_i \mathbf{u}_i \mathbf{v}_i^{\top} \tag{1}
$$

其中 $\mathbf{U} = [\mathbf{u}_1, \ldots, \mathbf{u}_m] \in \mathbb{R}^{m \times m}$，$\mathbf{V} = [\mathbf{v}_1, \ldots, \mathbf{v}_n] \in \mathbb{R}^{n \times n}$，$\mathbf{\Sigma} = \operatorname{diag}(\sigma_1, \ldots, \sigma_m)$ 且 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_m \geq 0$。

**动机**：SVD 给出了矩阵 $\mathbf{W}$ 在标准正交基下的"能量分布"——每个方向 $\mathbf{u}_i \mathbf{v}_i^{\top}$ 对 $\mathbf{W}$ 的贡献由其奇异值 $\sigma_i$ 量化。这使我们能够按"信息量"将权重分成两部分。

---

### 步骤 1：按奇异值大小分为主成分和次成分

MiLoRA 选择一个阈值 $r$（即 LoRA 的 rank 超参数），将奇异值划分为两组：前 $m-r$ 个大的（主成分）和后 $r$ 个小的（次成分）：

$$
\mathbf{W} = \underbrace{\sum_{i=1}^{m-r} \sigma_i \mathbf{u}_i \mathbf{v}_i^{\top}}_{\mathbf{W}_p \; \text{(主成分)}} + \underbrace{\sum_{i=m-r+1}^{m} \sigma_i \mathbf{u}_i \mathbf{v}_i^{\top}}_{\mathbf{W}_m \; \text{(次成分)}} \tag{2}
$$

**动机**：论文引用 LASER (Sharma et al., 2024) 的发现——主奇异分量（大奇异值）编码跨任务的重要特征，而次奇异分量（小奇异值）包含噪声或长尾信息。因此 $\mathbf{W}_p$ 应被冻结以保留预训练知识，$\mathbf{W}_m$ 是微调时应该修改的部分。

用矩阵形式重写：

$$
\mathbf{U} = [\underbrace{\mathbf{U}_p}_{m-r \text{ 列}}, \underbrace{\mathbf{U}_m}_{r \text{ 列}}], \quad \mathbf{V} = [\underbrace{\mathbf{V}_p}_{m-r \text{ 列}}, \underbrace{\mathbf{V}_m}_{r \text{ 列}}], \quad \mathbf{\Sigma} = \begin{bmatrix} \mathbf{\Sigma}_p & \mathbf{0} \\ \mathbf{0} & \mathbf{\Sigma}_m \end{bmatrix}
$$

则：

$$
\mathbf{W} = \underbrace{\mathbf{U}_p \mathbf{\Sigma}_p \mathbf{V}_p^{\top}}_{\mathbf{W}_p} + \underbrace{\mathbf{U}_m \mathbf{\Sigma}_m \mathbf{V}_m^{\top}}_{\mathbf{W}_m} \tag{3}
$$

---

### 步骤 2：用次成分初始化 LoRA 适配器

与 PiSSA 一样，MiLoRA 巧妙地利用平方根分配来初始化低秩矩阵 $\mathbf{A}_m$ 和 $\mathbf{B}_m$：

$$
\mathbf{W}_m = \mathbf{U}_m \mathbf{\Sigma}_m \mathbf{V}_m^{\top} = \underbrace{(\mathbf{U}_m \sqrt{\mathbf{\Sigma}_m})}_{\mathbf{B}_m \in \mathbb{R}^{m \times r}} \underbrace{(\sqrt{\mathbf{\Sigma}_m} \mathbf{V}_m^{\top})}_{\mathbf{A}_m \in \mathbb{R}^{r \times n}} \tag{4}
$$

**动机**：将 $\mathbf{\Sigma}_m^{1/2}$ 分别分配到 $\mathbf{B}_m$ 和 $\mathbf{A}_m$ 中，使得 $\mathbf{B}_m \mathbf{A}_m$ 精确等于 $\mathbf{W}_m$（秩 $r$ 矩阵）。这保证了：

1. 初始时 $\mathbf{W}_p + \mathbf{B}_m \mathbf{A}_m = \mathbf{W}$，模型输出不变（与 LoRA 的零初始化效果一致）
2. $\mathbf{B}_m \neq \mathbf{0}$，因此 $\mathbf{A}_m$ 和 $\mathbf{B}_m$ 的梯度**初始即非零**，避免了 LoRA 的"热身延迟"
3. 不需要像 LoRA 那样额外调优 scaling factor $\alpha$（因为初始分解已经精确）

前向计算为：

$$
\mathbf{Y} = \mathbf{X} \mathbf{W} = \mathbf{X}(\mathbf{W}_p + \mathbf{B}_m \mathbf{A}_m) \tag{5}
$$

训练时只更新 $\mathbf{B}_m$ 和 $\mathbf{A}_m$，$\mathbf{W}_p$ 冻结。

---

### 步骤 3：梯度分析

MiLoRA 的前向结构与 LoRA 相同，因此梯度形式也相同：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{A}_m} = \mathbf{X}^{\top} \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \mathbf{B}_m^{\top}, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{B}_m} = \mathbf{X}^{\top} \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \mathbf{A}_m^{\top} \tag{6}
$$

但与 LoRA 的关键差异在于：$\mathbf{B}_m \neq \mathbf{0}$（由次奇异向量初始化），因此 $\frac{\partial \mathcal{L}}{\partial \mathbf{A}_m} \neq \mathbf{0}$ 从第一步起就成立。同时，$\mathbf{A}_m$ 和 $\mathbf{B}_m$ 初始处于**与 $\mathbf{W}_p$ 正交的子空间**中（见下节），因此梯度的更新方向天然避开预训练主成分的"领地"。

> **与 PiSSA 的梯度差异**：
> - PiSSA 的 $\mathbf{A}, \mathbf{B}$ 包含主奇异向量，梯度沿 $\mathbf{W}$ 的主方向传播 → 快速改变权重的核心部分
> - MiLoRA 的 $\mathbf{A}_m, \mathbf{B}_m$ 包含次奇异向量，梯度沿 $\mathbf{W}$ 的次要方向传播 → 在保留核心的前提下做精细调整

---

### 步骤 4：训练效率

训练完成后，MiLoRA 的适配器可以像 LoRA 一样合并回权重矩阵：

$$
\mathbf{W}' = \mathbf{W}_p + \mathbf{B}_m' \mathbf{A}_m' \tag{7}
$$

其中 $\mathbf{B}_m', \mathbf{A}_m'$ 是训练后的适配器参数。推理时无额外计算开销。SVD 分解的额外时间成本不到 6 分钟，相比训练时间可以忽略。

---

## 核心假设（显式 + 隐式）

### 显式假设

1. **低秩假设**（继承自 LoRA）：权重变化 $\Delta \mathbf{W}$ 是低秩的。MiLoRA 进一步假设**次奇异分量本身**就是低秩的（秩为 $r$），这是天然的，因为 W_m 的秩就是 $r$。

2. **奇异值分层假设**：$\mathbf{W}$ 的奇异值呈长尾分布，前 $m-r$ 个远大于后 $r$ 个，即 $\sigma_{m-r} \gg \sigma_{m-r+1}$。这保证了"主成分 vs 次成分"的划分有意义。

3. **知识编码假设**：大奇异值方向编码"重要知识"，小奇异值方向编码"噪声/长尾信息"。论文引用 LASER (Sharma et al., 2024) 和低秩压缩文献来支持这一假设。

4. **正交初始化假设**：训练的次适配器 $\mathbf{B}_m \mathbf{A}_m$ 初始化在 $\mathbf{W}_p$ 的正交空间中，从而减少对预训练知识的干扰。

### 隐式假设

> **假设 A — "次成分 = 可修改的知识" 假设**
>
> **在哪里引入**：Section 3（公式 2-3 的拆分逻辑），以及 LASER 引用。
>
> MiLoRA 将 $\mathbf{W}_m$ 视为"可安全修改的部分"，声称其包含"噪声或长尾信息"。这隐含假设了：**小奇异值方向对预训练模型的核心能力贡献最小，因此可以安全地修改而不损害模型性能。**
>
> 这个假设在以下条件可能不成立：
> - **长尾知识贡献累积效应**：单个次奇异方向虽然弱（$\sigma_i$ 小），但 $r$ 个次方向的**联合表达力**可能不可忽略。如果 $r$ 选择过大（例如 $r > m/2$），次成分可能实际上包含了大量可用的预训练信息，修改它们会导致知识丢失。
> - **任务-知识对齐问题**：微调任务所需的"新知识"可能与某些次奇异方向编码的"预训练知识"冲突。如果次方向确实包含有用的长尾预训练知识（例如罕见实体、低频语法模式），覆盖它们可能导致在这些方面的能力下降。
>
> 论文通过消融实验（Table 6）部分验证了这一点：用次成分初始化优于随机初始化和主成分初始化。**但这验证的是"起始点"的好坏，而非"冻结主成分不造成损失"**——后者需要对比"MiLoRA vs 同时更新 $\mathbf{W}_p$ 和 $\mathbf{W}_m$"。

> **假设 B — 正交性在训练中保留的假设（"soft" 版本）**
>
> **在哪里引入**：Section 3 末尾，"we adopt a softer approach by guiding the optimization direction solely through initialization"。
>
> MiLoRA 明确承认它**不强制**训练中的正交性约束。它只在初始化时将 $\mathbf{B}_m \mathbf{A}_m$ 置于与 $\mathbf{W}_p$ 正交的子空间中，训练过程中 $\mathbf{B}_m$ 和 $\mathbf{A}_m$ 可以自由漂移出这个子空间。
>
> **这是论文最微妙的隐性假设**：初始化引导足以使训练过程中适配器的参数不会显著进入 $\mathbf{W}_p$ 的子空间。
>
> 这个假设在什么条件下成立？
> - 当微调数据集的规模适中，优化步数不太多
> - 当学习率适当，不会导致剧烈的权重跳跃
> - 当 $\mathbf{W}_p$ 和 $\mathbf{W}_m$ 的奇异值差距足够大（即 $\sigma_{m-r} \gg \sigma_{m-r+1}$）
>
> 在什么条件下可能不成立？
> - 如果微调数据集很大（如 MetaMathQA 395K），训练步数多，适配器参数可能逐渐漂移进入主成分子空间
> - 如果 $\sigma_{m-r}$ 和 $\sigma_{m-r+1}$ 差异不显著（奇异值分布过渡平缓），"正交"的初始化边界本身就模糊

> **假设 C — SVD 截断的最优性假设**
>
> **在哪里引入**：公式 2-3，以及 Section 5.2（三种初始化方式的对比实验）。
>
> MiLoRA 将奇异值简单地按**大小排序**后截断：前 $m-r$ 个冻结，后 $r$ 个训练。这隐含假设了：**奇异值大小是衡量"可修改性"的最佳指标。**
>
> 但这并非显而易见：
> - 是否存在某些小奇异值方向编码了对某些下游任务至关重要的信息？
> - 是否存在某些大奇异值方向实际上编码了噪声（如过拟合到预训练数据中的偏差）？
> - 对于不同层（embedding, attention, MLP），最优的"主/次"划分比例是否应该不同？
>
> Table 6 的消融实验验证了在数学推理上"次成分 > 随机 > 主成分"，但在不同任务/模型上这个排序是否一致？论文只在一个设置（LLaMA2-7B, Math）上做这个消融，结论的泛化性有待验证。

> **假设 D — 遗忘损失衡量的充分性假设**
>
> **在哪里引入**：Section 5.4（Table 8, Forgetting loss）。
>
> 论文用 WikiText-103 上的交叉熵（衡量模型输出偏离预训练分布的程度）作为"遗忘"的度量。这隐含假设了：**WikiText-103 上的 next-token 预测分布偏移可以代表所有任务上的预训练知识保留情况。**
>
> 这可能有偏差：
> - WikiText-103 是 Wikipedia 文本，与常识推理、数学推理等下游任务的数据分布不同
> - 模型可能在 WikiText-103 上保留得很好，但某些特定领域的知识（如数学公式、罕见词汇）已经丢失
> - 更全面的遗忘评估应使用多个领域的数据集

---

## 与前驱工作的逻辑关系

### 继承了谁的思路/方法？

1. **LoRA (Hu et al., 2021)** — 继承了完整的框架：前向 $\mathbf{Y} = \mathbf{X}(\mathbf{W}_{\text{frozen}} + \frac{\alpha}{r}\mathbf{B}\mathbf{A})$，适配器合并方式。MiLoRA 只替换了初始化策略和冻结策略。

2. **LASER (Sharma et al., 2024)** — 提供了关键的理论支撑：层选择性秩缩减的实验表明，大奇异值方向编码跨任务的重要特征，小奇异值方向包含噪声。MiLoRA 将此发现从"推理时降噪"应用到了"训练时适配"。

3. **PiSSA (Meng et al., 2024)** — 提供了与 MiLoRA 完全相同的"SVD 初始化 LoRA 适配器"的技术框架（包括 $\mathbf{\Sigma}^{1/2}$ 的平方根分配策略）。核心区别仅在于选择哪一部分来初始化适配器。

### 解决了前人的什么根本局限？

**Limitation 1：PiSSA 的高遗忘损失问题**

PiSSA 训练主成分、冻结次成分。由于主成分编码了权重的"核心知识"，直接修改它们会导致更高的遗忘损失（PiSSA 的遗忘损失 6.07 ≈ MiLoRA 2.54 的 2.4 倍）。MiLoRA 的策略正好相反——保留核心（主成分冻结）、修改次要部分（次成分训练）。

**数学上的根本对比**：

PiSSA 的哲学：
$$
\mathbf{W} = \underbrace{\mathbf{W}^{\text{prin}}}_{\text{可训练 } \mathbf{A}\mathbf{B}} + \underbrace{\mathbf{W}^{\text{res}}}_{\text{冻结}} \quad \Rightarrow \quad \text{近似全参数微调，但忘得快}
$$

MiLoRA 的哲学：
$$
\mathbf{W} = \underbrace{\mathbf{W}_p}_{\text{冻结}} + \underbrace{\mathbf{W}_m}_{\text{可训练 } \mathbf{B}_m\mathbf{A}_m} \quad \Rightarrow \quad \text{保留知识，精细适配，但可能偏离全参数微调路径}
$$

**Limitation 2：LoRA 的"无引导"随机初始化**

LoRA 随机初始化适配器在无引导的子空间中搜索，可能无意中覆盖预训练特征。MiLoRA 通过 SVD 将搜索空间明确限定在与 $\mathbf{W}_p$ 正交的次成分子空间中，从而从根本上避免了这个问题。

**与 PiSSA 的本质差异**：
- PiSSA 选择"学得快、忘得快"——从主成分出发，快速逼近全参数微调的路径
- MiLoRA 选择"学得稳、忘得慢"——从次成分出发，保留核心知识做精细调整

### 引入/改变了什么约束/假设？

MiLoRA 没有改变 LoRA 的基本约束（仍然是 rank-r $\mathbf{B}\mathbf{A}$ 分解 + 冻结原权重），但它**重新定义了**"什么应该被冻结"的语义：

- LoRA：冻结 $\mathbf{W}$（原权重），可训练 $\mathbf{B}\mathbf{A} \approx \Delta \mathbf{W}$
- PiSSA：冻结 $\mathbf{W}^{\text{res}}$（次成分），可训练 $\mathbf{B}\mathbf{A} \approx \mathbf{W}^{\text{prin}}$
- MiLoRA：冻结 $\mathbf{W}_p$（主成分），可训练 $\mathbf{B}_m\mathbf{A}_m \approx \mathbf{W}_m$

值得注意的是，MiLoRA 和 PiSSA 共享完全相同的数学形式（都是 $\mathbf{W} = \mathbf{W}_{\text{frozen}} + \mathbf{B}\mathbf{A}$），但参数数量和初始化来源不同。如果设 PiSSA 的 rank 为 $r$，MiLoRA 的 rank 也为 $r$，则：
- PiSSA：$\mathbf{W}_{\text{frozen}} = \mathbf{W}_{:,r:}\mathbf{\Sigma}_{r:,:}\mathbf{V}_{:,r:}^{\top}$（秩为 $m-r$ 的矩阵），$\mathbf{B}\mathbf{A} = \mathbf{W}_{:,:r}\mathbf{\Sigma}_{:r,:r}\mathbf{V}_{:r,:}^{\top}$（秩为 $r$）
- MiLoRA：$\mathbf{W}_p = \mathbf{U}_p\mathbf{\Sigma}_p\mathbf{V}_p^{\top}$（秩为 $m-r$），$\mathbf{B}_m\mathbf{A}_m = \mathbf{U}_m\mathbf{\Sigma}_m\mathbf{V}_m^{\top}$（秩为 $r$）

**它们在数学上是完全对称的**——PiSSA 训练前 $r$ 个奇异分量，MiLoRA 训练后 $r$ 个奇异分量。两者的 $\mathbf{W}_{\text{frozen}}$ 都是秩 $m-r$ 的矩阵，参数数量完全相同（$mr + nr$）。唯一的区别是哪一个奇异值范围被分配到可训练部分。

---

## MiLoRA 的"正交空间"构造与"Null Space"的关系

### 正交空间构造细节

给定 $\mathbf{W} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top}$，MiLoRA 将 $\mathbf{U}$ 和 $\mathbf{V}$ 分块：

$$
\mathbf{U} = [\mathbf{U}_p, \mathbf{U}_m], \quad \mathbf{U}_p \in \mathbb{R}^{m \times (m-r)}, \; \mathbf{U}_m \in \mathbb{R}^{m \times r}
$$

$$
\mathbf{V} = [\mathbf{V}_p, \mathbf{V}_m], \quad \mathbf{V}_p \in \mathbb{R}^{n \times (m-r)}, \; \mathbf{V}_m \in \mathbb{R}^{n \times r}
$$

由于 $\mathbf{U}$ 是酉矩阵，$\mathbf{U}_p^{\top} \mathbf{U}_m = \mathbf{0}$。同理 $\mathbf{V}_p^{\top} \mathbf{V}_m = \mathbf{0}$。这意味着：

- **列空间正交**：$\operatorname{Col}(\mathbf{W}_p)$（张成于 $\mathbf{U}_p$）与 $\operatorname{Col}(\mathbf{W}_m)$（张成于 $\mathbf{U}_m$）正交
- **行空间正交**：$\operatorname{Row}(\mathbf{W}_p)$（张成于 $\mathbf{V}_p$）与 $\operatorname{Row}(\mathbf{W}_m)$（张成于 $\mathbf{V}_m$）正交

因此，$\mathbf{W}_m$ 的作用空间（即输入经 $\mathbf{W}_m$ 变换后能到达的输出子空间）与 $\mathbf{W}_p$ 的作用空间是正交的。

### 与 "Null Space" 的关系

"Null space"（零空间）通常指 $\mathcal{N}(\mathbf{W}) = \{\mathbf{x} \mid \mathbf{W}\mathbf{x} = \mathbf{0}\}$。MiLoRA 中 $\mathbf{W}_m$ 并不对应 $\mathbf{W}$ 的零空间（因为 $\mathbf{W}_m$ 的奇异值非零，所以 $\mathbf{W}_m \mathbf{x} \neq \mathbf{0}$ 一般成立）。

更准确地说，$\mathbf{W}_m$ 操作在 **$\mathbf{W}_p$ 的零空间的补空间的正交补**中。具体来说：

- $\mathcal{N}(\mathbf{W}_p) = \operatorname{span}(\mathbf{V}_m)$（因为 $\mathbf{W}_p \mathbf{v} = \mathbf{0}$ 对所有 $\mathbf{v} \in \operatorname{span}(\mathbf{V}_m)$ 成立，$\mathbf{W}_p$ 的后 $r$ 个右奇异向量对应零奇异值）
- $\operatorname{Row}(\mathbf{W}_m) = \operatorname{span}(\mathbf{V}_m)$
- 所以：$\operatorname{Row}(\mathbf{W}_m) = \mathcal{N}(\mathbf{W}_p)$

即 **$\mathbf{W}_m$ 的行空间正好是 $\mathbf{W}_p$ 的零空间**。这意味着：
- 任何 $\mathbf{W}_p$ 无法"感知"的输入方向（即在 $\mathbf{V}_m$ 张成的空间中），恰好是 $\mathbf{W}_m$ 可以独立处理的方向。
- 反过来，任何 $\mathbf{W}_p$ 有响应的方向（$\mathbf{V}_p$ 张成的空间），$\mathbf{W}_m$ 不会有贡献。

**这解释了为什么 MiLoRA 能够"保留预训练知识"**：$\mathbf{W}_p$ 冻结后，它在 $\mathbf{V}_p$ 方向上的响应完全不受训练影响；$\mathbf{B}_m\mathbf{A}_m$ 在训练中调整 $\mathbf{V}_m$ 方向上的响应，这些方向是 $\mathbf{W}_p$ 的零空间方向，因此不会与 $\mathbf{W}_p$ 的输出干扰。

### 训练中的正交性漂移

一个重要但容易被忽略的细节：**这种正交性只在初始化时成立**。训练过程中，$\mathbf{B}_m$ 和 $\mathbf{A}_m$ 的参数更新会使它们偏离初始值，从而可能不再精确保持 $\operatorname{Col}(\mathbf{B}_m\mathbf{A}_m) \perp \operatorname{Col}(\mathbf{W}_p)$。

论文明确承认这一点（Section 3 末尾），选择了"softer approach"——仅通过初始化引导，不强制正交约束。这带来了讨论空间：如果在训练中显式施加正交约束（例如通过正则化项 $\|\mathbf{U}_p^{\top} \mathbf{B}_m\|_F^2$），是否能进一步提升性能？论文的实验没有回答这个问题。

---

## 预训练知识保留的实验证据

### 证据 1：遗忘损失（Table 8）

| 方法 | 遗忘损失 |
|------|---------|
| MiLoRA | **2.54** |
| LoRA | 3.24 |
| PiSSA | 6.07 |

MiLoRA 的遗忘损失最低（2.54），PiSSA 最高（6.07 ≈ 2.4× MiLoRA）。这说明在 WikiText-103 上，MiLoRA 微调后的模型输出分布最接近原始预训练模型。

### 证据 2：子空间相似性分析（Figure 2）

Figure 2（左图）展示了三种方法 $\Delta \mathbf{W}$ 的 Top-r 奇异向量与 $\mathbf{W}$ 的 Top/Bottom/Random r 奇异向量之间的子空间相似度：

- **PiSSA** 的 $\Delta \mathbf{W}$ 与 $\mathbf{W}$ 的 Top-r 奇异向量高度对齐（相似度 ~0.3）：它确实在修改主成分
- **LoRA** 的 $\Delta \mathbf{W}$ 与三者相似度都低（~0.05-0.08）：在无引导的子空间中搜索
- **MiLoRA** 的 $\Delta \mathbf{W}$ 与 $\mathbf{W}$ 的 Bottom-r 奇异向量对齐（~0.015）：它在修改次成分方向

特别值得注意的是 Figure 2 的左、右下两张图。MiLoRA 对 MLP 模块的 Bottom 方向对齐度远高于 self-attn模块的同指标。这暗示 MLP 层的次奇异分量可能携带了更多的可迁移/可修改信息。

### 证据 3：Frobenius 范数分析（Table 7）

MiLoRA 对 $\mathbf{W}$ 的 top-r 子空间的影响最小（$\|\mathbf{U}^{\top} \Delta \mathbf{W} \mathbf{V}\|_F = 44.95$，$\mathbf{U},\mathbf{V}$ 为 $\Delta\mathbf{W}$ 的 top-r 奇异向量），放大因子 24.17 远低于 LoRA 的 37.46。

> **注意**：PiSSA 的放大因子只有 3.18 看起来更小，但这是因为 PiSSA 的 $\Delta \mathbf{W}$ 与 $\mathbf{W}$ 的 top 方向高度对齐，投影值大，所以比值显得小。这不意味着 PiSSA 对预训练知识的影响小——实际上遗忘损失表明它的影响最大。

### 综合判断

三个证据一致支持 MiLoRA 保留了更多预训练知识：
1. 遗忘损失最低 ✓
2. $\Delta \mathbf{W}$ 对齐 Bottom 方向而非 Top 方向 ✓
3. 对 Top-r 子空间的投影/放大最小（相对 LoRA） ✓

但需要注意：这些证据主要集中在数学推理任务（LLaMA2-7B, MetaMathQA）上，缺乏跨任务/跨模型的泛化验证。不同任务类型（如指令遵循 vs 代码生成）可能对"知识保留"有不同的需求。

---

## 留下的开放问题

### 悬而未决的问题

1. **"正交漂移"的影响有多大？** MiLoRA 不强制训练中的正交性——训练过程中 $\mathbf{B}_m\mathbf{A}_m$ 逐渐漂离 $\mathbf{W}_p$ 的零空间。这种漂移的程度和对性能的影响没有被量化。如果漂移很大，"次成分初始化"的唯一优势就只剩下初始梯度不为零（这 LoRA-GA 也能实现），而不再是"保留知识"。

2. **为什么 PiSSA 在 PiSSA 的超参数设置下优于 MiLoRA，但在 LLM-Adapters 设置下相反？**（Table 9）这揭示了一个更深层的问题：不同超参数配置下，"训练主成分"和"训练次成分"策略的相对优势可能完全逆转。能否给出一个理论准则来判断"什么时候该训练哪个部分"？

3. **MiLoRA 与 LoRA 结合的可叠加性**：如果多个 LoRA 适配器叠加使用，MiLoRA 适配器之间是否兼容？每个 MiLoRA 适配器的 $\mathbf{W}_p$ 是相同的（来自同一个预训练模型），但不同任务对次成分的修改是否会相互干扰？

4. **动态 rank 分配的可行性**：不同层的奇异值分布不同（attention vs MLP 的奇异值衰减速度不同），用统一 $r$ 是否最优？能否像 AdaLoRA 一样为每层动态分配不同的 $r$？

### 明显但论文没做的实验/分析

1. **训练过程中的正交性追踪**：论文只在初始化时保证了正交性。如果在训练过程中持续监控 $\|\mathbf{U}_p^{\top} \mathbf{B}_m\|_F$（正交漂移度量），可以定量回答"初始化引导到底能维持多久"。

2. **奇异值衰减速度与性能的关系**：不同层、不同模型的奇异值分布形状（$\sigma_{m-r}/\sigma_{m-r+1}$ 的比值）与 MiLoRA 提升幅度的相关性分析。预期：$\sigma_{m-r} / \sigma_{m-r+1}$ 越大（即主/次分割更清晰），MiLoRA 优势越明显。

3. **全参数微调路径的投影分析**：将 Full FT 的 $\Delta \mathbf{W}$ 投影到 $\mathbf{W}$ 的主成分子空间和次成分子空间，直接验证"全参数微调主要在修改哪个子空间？"如果 Full FT 主要修改主成分子空间，则 MiLoRA 的策略与 Full FT 的路径有偏差；如果反之，则 MiLoRA 的路径更接近 Full FT。

4. **更大的 rank 实验**：当 $r$ 接近 $m$（即次成分包含了几乎所有奇异值）时，MiLoRA 退化为"几乎训练整个权重"，此时与 LoRA 的区别消失了。论文没有系统研究 $r$ 从 1 到 $m/2$ 的变化趋势。

5. **与更多 "freeze-principal" 变体的对比**：除了训练次成分，还可以设计"冻结主成分 + 训练全秩残差"（即不限制为低秩）的变体，作为上限比较。

### 如果我来改进这篇工作，我会从哪里入手？

1. **正交正则化**：在训练中添加轻量的正交正则化项 $\lambda \|\mathbf{U}_p^{\top} \mathbf{B}_m\|_F^2$，防止适配器参数漂移出次成分子空间。这可能会进一步提升遗忘损失指标。

2. **混合 PiSSA + MiLoRA 策略**：不同层采用不同策略——底层（embedding、near-embedding attention）用 MiLoRA（保留知识），顶层（靠近输出的层）用 PiSSA（快速适应任务特定模式）。类似 LoRA 的分层重要性采样思想。

3. **自适应 r 选择**：利用奇异值的"gap"（$\sigma_{m-r} - \sigma_{m-r+1}$）为每层自动确定最优的 $r$。AdaLoRA 的思路加上 SVD 初始化，天然契合。

4. **量化版本的深入分析**：论文只在全精度下实验。在 QLoRA 场景下，MiLoRA 的量化误差是否会比 QPiSSA 更小？由于 MiLoRA 冻结了主成分（大值），只量化次成分（小值），量化误差应该更小——但论文没有验证这个明显的研究方向。

5. **理论泛化界分析**：能否从 PAC-Bayes 或 NTK 角度给出 MiLoRA 泛化性能的理论保证？直觉上，限制优化子空间（次成分）会减小模型的"有效容量"，从而可能带来更好的泛化界。

---

## 在领域中的定位

### 所属范式/技术路线

MiLoRA 属于 **Parameter-Efficient Fine-Tuning (PEFT)** 下的 **Low-Rank Adapter** 路线，具体是 LoRA 的**初始化改进 + 参数冻结策略**方法。

```
PEFT
├── Prompt-based (Prefix Tuning, Prompt Tuning, P-Tuning)
├── Adapter-based (HAdapter, PAdapter, AdapterFusion)
└── Low-Rank Adapter (LoRA family)
    ├── LoRA (2021) — 基础架构，随机初始化
    ├── PiSSA (2024) — 训练主成分，冻结残差（"从骨架开始学"）
    ├── MiLoRA (2024, NAACL) — 冻结主成分，训练次成分（"从噪声开始调"）← 本文
    ├── LoRA-GA (2024) — 梯度对齐初始化
    ├── AdaLoRA (2023) — 动态 rank 分配 + SVD 式分解
    ├── DoRA (2024) — 权重解耦（方向+幅度）
    ├── LoftQ (2023) — 量化误差 SVD 补偿
    └── ...
```

### 与同一路线中其他工作的关系

- **相对于 LoRA**：MiLoRA 是一个即插即用替换，只改初始化和冻结策略，不改架构。训练/推理效率与 LoRA 完全相同（SVD 的 6 分钟开销可忽略）。

- **相对于 PiSSA**：这是最核心的关系对比。两条路线的哲学完全相反：
  - **PiSSA：训练主成分 + 冻结次成分** ≈ 近似全参数微调（快速但遗忘多）
  - **MiLoRA：冻结主成分 + 训练次成分** ≈ 保留预训练知识（稳但可能偏离全参数微调路径）
  
  从公式层面看，两者是数学对称的：PiSSA 选择前 $r$ 个奇异值初始化适配器，MiLoRA 选择后 $r$ 个。没有哪个"绝对更好"——Table 9 显示在 PiSSA 的超参数设置下 PiSSA 更好，在 LLM-Adapters 设置下 MiLoRA 更好。这暗示**最优策略依赖于超参数配置**，而超参数配置本身可能与模型大小、任务类型存在交互。

- **相对于 LoRA-GA**：LoRA-GA 通过使 $\mathbf{BA}$ 的梯度与 Full FT 的梯度对齐来改进初始化。MiLoRA 通过 SVD 来引导优化子空间。两者思路不同但目标相似（更好的初始化方向）。在 Table 5 的对比中，MiLoRA (54.7/24.4) 超过了 LoRA-GA (53.6/19.8)。

- **相对于 DoRA**：DoRA 解耦权重为"方向 × 幅度"，而 MiLoRA 解耦权重为"主成分 × 次成分"。两者可以叠加使用（DoRA 的"方向"部分可以使用 MiLoRA 的初始化策略）。

- **相对于 LASER**：MiLoRA 的核心直觉（主成分编码重要知识、次成分编码噪声）直接来自 LASER。但 LASER 是在推理时通过秩缩减来降噪（去掉次成分让推理更准确），而 MiLoRA 是在训练时反其道而行之——专门训练这些次成分来适配新任务。这种"推理时丢弃、训练时利用"的对称性是一个有趣的设计空间。

### MiLoRA 在 PEFT 发展中的独特位置

MiLoRA 代表了 PEFT 中 **"知识保留优先"** 路线的典型方法。它提出了一个反直觉但有效的策略：不要训练权重的"精华"部分（主成分），而是训练权重的"糟粕"部分（次成分），因为修改"精华"会遗忘，修改"糟粕"才能保留精华的同时学习新东西。这种思路可以被扩展到其他 PEFT 场景（如 adapter、prompt tuning 中的知识保留策略）。

与 PiSSA 一起，这两篇工作共同揭示了 SVD 初始化空间中一个**完全对称的设计空间**：
- 主成分训练（PiSSA）vs 次成分训练（MiLoRA）
- 哪个更好取决于任务性质、超参数、模型架构

这为后续研究提出了一个根本问题：**是否存在一个理论准则，可以预判给定条件下应该训练哪个奇异子空间？**
