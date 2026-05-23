# Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge 本质分析

## 核心公式推导（严谨推导）

### 预备：标准 LoRA

给定预训练权重矩阵 $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，LoRA 引入低秩分解：

$$
\mathbf{y} = \mathbf{W}^* \mathbf{x} = \mathbf{W}_0 \mathbf{x} + \Delta \mathbf{W} \mathbf{x} = \mathbf{W}_0 \mathbf{x} + \mathbf{B} \mathbf{A} \mathbf{x} \tag{1}
$$

其中 $\mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$, $\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}$, $r \ll \min(d_{\text{out}}, d_{\text{in}})$。

### Step 1: 获取预训练知识激活值 $\mathbf{X}_{\text{pre}}$

**动机**：需要构造一个能代表预训练知识的矩阵，以便找到与之正交的更新方向。

从校准集（如 NQ Open）中采样数据，经过模型前向传播后收集某一线性层的输入激活值。设采样了 $B$ 个样本，最大序列长度为 $L$，则 $\mathbf{X}_{\text{pre}} \in \mathbb{R}^{d_{\text{in}} \times (B \times L)}$，每一列是一个 $d_{\text{in}}$ 维的输入激活向量。

> 注意：这里论文定义 $\mathbf{X}_{\text{pre}}$ 维度为 $\mathbb{R}^{d_{\text{in}} \times (B \times L)}$（见 Preliminaries 部分），但在 SVD 时 Eq.(6) 又将其写为 $\mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$（因为当 $B \times L > d_{\text{in}}$ 时，rank 受 $d_{\text{in}}$ 限制，$R = \min(d_{\text{in}}, B \times L) = d_{\text{in}}$）。这里存在一个维度上的微妙转换：$\mathbf{X}_{\text{pre}}$ 实际有效维度为 $d_{\text{in}} \times d_{\text{in}}$（当样本足够多时）。

### Step 2: 对 $\mathbf{X}_{\text{pre}}$ 做 SVD 得到零空间

**动机**：SVD 提供了一种自然的"主成分-次要成分"分解方式，最小的奇异值对应的奇异方向构成零空间近似。

对 $\mathbf{X}_{\text{pre}}$ 进行 SVD：

$$
\mathbf{X}_{\text{pre}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^{\top} = \sum_{w=1}^{R} \sigma_w \mathbf{u}_w \mathbf{v}_w^{\top} \tag{6}
$$

其中 $\mathbf{U} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$, $\mathbf{V} \in \mathbb{R}^{(B \times L) \times d_{\text{in}}}$, $\mathbf{\Sigma} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$，奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_R \geq 0$。

**这里的动机链**：$\mathbf{U}$ 的列是 $\mathbf{X}_{\text{pre}}$ 的左奇异向量，每个 $\mathbf{u}_i$ 对应 $\mathbf{X}_{\text{pre}}$ 的一个"激活模式"方向。$\sigma_i$ 越大，该方向在激活数据中出现的强度越高。我们希望忽略最强的那些方向（因为它们携带了最多预训练知识），只保留最弱的方向（零空间方向）。

将 $\mathbf{U}$ 拆分为：

$$
\mathbf{U} = [\mathbf{U}_1 \;|\; \mathbf{U}_2], \quad \mathbf{U}_2 \in \mathbb{R}^{d_{\text{in}} \times r}
$$

$\mathbf{U}_2$ 由最小的 $r$ 个奇异值对应的左奇异向量构成，即 $\mathbf{U}_{\text{null}} = \mathbf{U}_2$。

**关键近似**：论文假设 $\{\sigma_{R-r+1}, \dots, \sigma_R\}$ 可忽略不计，从而有：

$$
\mathbf{U}_{\text{null}}^{\top} \mathbf{X}_{\text{pre}} \approx \mathbf{0} \tag{7}
$$

这是整个方法的核心——这一近似成立的程度决定了 LoRA 初始化与预训练知识正交的程度。

### Step 3: 将 $\mathbf{W}_0$ 投影到 $\mathbf{X}_{\text{pre}}$ 的零空间

**动机**：我们需要的是 LoRA 适配器 $\mathbf{BA}$ 本身（一个在 $\mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 中的矩阵），而 $\mathbf{U}_{\text{null}}$ 是 $d_{\text{in}}$ 维空间中的方向。为了使 $\mathbf{BA}$ 作用于激活时落在 $\mathbf{X}_{\text{pre}}$ 的零空间，我们需要让 $\mathbf{BA}$ 的行空间（或更确切地说，$\mathbf{BA}$ 对输入的作用）被 $\mathbf{U}_{\text{null}}$ 所约束。

论文选择令 $\mathbf{BA}$ 为 $\mathbf{W}_0$ 在 $\mathbf{U}_{\text{null}}$ 上的投影：

$$
\mathbf{BA} = \mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top} \tag{8}
$$

**为什么是这样？** 对于任意输入 $\mathbf{x}$，$\mathbf{U}_{\text{null}}^{\top} \mathbf{x}$ 是 $\mathbf{x}$ 在 $\mathbf{U}_{\text{null}}$ 方向上的分量。由于 $\mathbf{U}_{\text{null}}^{\top} \mathbf{X}_{\text{pre}} \approx \mathbf{0}$，对于 $\mathbf{X}_{\text{pre}}$ 列空间中的向量 $\mathbf{x}$，$\mathbf{U}_{\text{null}}^{\top} \mathbf{x} \approx \mathbf{0}$。因此 $\mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top} \mathbf{x} \approx \mathbf{0}$ 对于预训练知识激活 $\mathbf{x}$ 成立——即 $\mathbf{BA}$ 对预训练知识"不可见"。

注意：这使得 $\mathbf{BA}$ 是 $\mathbf{W}_0$ 的一部分（从 $\mathbf{W}_0$ 中"切出"的一块），而残差权重 $\mathbf{W}_0' = \mathbf{W}_0 - \mathbf{BA}$。

### Step 4: 对投影矩阵做第二次 SVD 以得到 $\mathbf{A}, \mathbf{B}$

**动机**：我们需要将 $\mathbf{BA}$ 分解为标准的 LoRA 形式（$\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$）。

对 $\mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top}$ 做 SVD：

$$
\mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top} = \mathbf{U}' \mathbf{D}' (\mathbf{V}')^{\top} \tag{9}
$$

取前 $r$ 个分量初始化 LoRA：

$$
\mathbf{B} = \mathbf{U}'_{[:, :r]} \sqrt{\mathbf{D}'_{[:r, :r]}}, \quad
\mathbf{A} = \sqrt{\mathbf{D}'_{[:r, :r]}} (\mathbf{V}'_{[:, :r]})^{\top} \tag{10}
$$

### Step 5: 残差权重

$$
\mathbf{W}_0' = \mathbf{W}_0 - \mathbf{BA} \tag{11}
$$

### 完整的微调过程

- 冻结 $\mathbf{W}_0'$
- 仅更新 $\mathbf{A}$ 和 $\mathbf{B}$
- 前向传播：$\mathbf{y} = \mathbf{W}_0' \mathbf{x} + \mathbf{B} \mathbf{A} \mathbf{x}$

### 与 MiLoRA 和 CorDA 的公式对比

**MiLoRA**：$\text{SVD}(\mathbf{W}_0) = \mathbf{U} \mathbf{S} \mathbf{V}^{\top}$，取最小的 $r$ 个奇异值/向量对。即 $\mathbf{W}_0' = \mathbf{U}_{[:, :R-r]} \mathbf{S}_{[:R-r, :R-r]} \mathbf{V}_{[:, :R-r]}^{\top}$，$\mathbf{BA} = \mathbf{U}_{[:, R-r+1:R]} \mathbf{S}_{[R-r+1:R, R-r+1:R]} \mathbf{V}_{[:, R-r+1:R]}^{\top}$。

**CorDA（知识保留模式）**：计算 $\mathbf{C} = \mathbf{X}_{\text{pre}} \mathbf{X}_{\text{pre}}^{\top}$，对 $\mathbf{W}_0 \mathbf{C}$ 做 SVD，取最大的 $r$ 个分量作为适配器。

**LoRA-Null**：对 $\mathbf{X}_{\text{pre}}$ 做 SVD 得到 $\mathbf{U}$，取最小的 $r$ 个左奇异向量作为 $\mathbf{U}_{\text{null}}$，然后 $\mathbf{BA} = \mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top}$。

**Theorem 3**（关键理论结果）：LoRA-Null 的初始化**不是** Eq.(4)（MiLoRA 的目标函数）和 Eq.(5)（CorDA 的目标函数）的解。这证实了 LoRA-Null 完全不关心残差权重是否接近 $\mathbf{W}_0$，只关心 LoRA 初始化空间是否正交于预训练知识。

> **存在推导间隙**：论文从 Eq.(6) 到 Eq.(7) 的跳跃没有严格证明——"$\mathbf{U}_2^{\top} \mathbf{X}_{\text{pre}} \approx \mathbf{0}$"的近似质量取决于 $\{\sigma_{R-r+1}, \dots, \sigma_R\}$ 的实际大小。在什么阈值下 $\sigma_i$ 可被视为"足够小以至于可以忽略"？论文说"as evidenced in Figure 2"，但这是一个实验观测而非理论保证。当选择了不合适的 $r$ 时，零空间近似可能严重不准确。

## 核心假设（显式 + 隐式）

### 显式假设

1. **低秩假设**：LoRA 微调中权重的变化具有低秩结构（$r \ll \min(d_{\text{in}}, d_{\text{out}})$），这是 LoRA 本身的基本假设，不是本文特有的。

2. **校准集代表预训练知识**：从 NQ Open 等数据集中采样的 256 条数据可以代表模型预训练阶段的整体知识分布（论文跟随 Yang et al. 的做法）。

3. **正交性假设**：让 LoRA 初始化与预训练知识正交可以防止灾难性遗忘。

### **隐式假设**

1. **$\mathbf{U}_{\text{null}}^{\top} \mathbf{X}_{\text{pre}} \approx \mathbf{0}$ 的有效性假设 (Eq.7)**
   - **在哪里偷偷引入的？** Section "Our Method" 中的 Eq.(7)。论文将 $\mathbf{U}_2$（最小的 $r$ 个左奇异向量）直接当作 $\mathbf{X}_{\text{pre}}$ 的"近似零空间"，假设 $\{\sigma_{R-r+1}, \dots, \sigma_R\}$ 小到可以忽略。
   - **什么条件下成立**：当 $\mathbf{X}_{\text{pre}}$ 的奇异值谱呈现急剧衰减时成立（即前几个奇异值占据绝大多数能量，尾部快速趋近于零）。
   - **什么条件下可能不成立**：如果 $\mathbf{X}_{\text{pre}}$ 的奇异值谱较为平坦（有效秩较高），那么尾部奇异值并不接近零，$\mathbf{U}_2^{\top} \mathbf{X}_{\text{pre}}$ 不再接近零矩阵，零空间近似的假设就崩溃了。有趣的是，论文在 Table 2 中恰恰论证了 $\mathbf{X}_{\text{pre}}$ 的有效秩远小于 $\mathbf{W}_0$（即 $\mathbf{X}_{\text{pre}}$ 的谱更陡峭），所以这个假设在本文的设定下是自洽的——但它仍然是一个近似，没有理论界的误差保证。

2. **线性可分离性假设**
   - **在哪里偷偷引入的？** 论文假设 $\mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top}$（即 $\mathbf{BA}$）与预训练知识的正交关系在微调**过程中**能持续保持。但实际上，$\mathbf{A}$ 和 $\mathbf{B}$ 在微调中会被更新，这意味着 $\mathbf{BA}$ 会偏离它最初的投影形式 $\mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top}$。论文在 Figure 1 中只能证明**初始状态到微调后**的相对变化很小，但并未证明这一变化不会使 $\mathbf{BA}$ 旋转到包含预训练知识的方向。
   - **什么条件下成立**：如果 LoRA 适配器的更新范数很小（Figure 1 显示相对变化确实很小，对于 LoRA-Null 在 700 步后约 10-15x 基准大小），且优化器的隐式偏差倾向于最小化参数变化（这在高维 LoRA 中通常成立）。
   - **什么条件下可能不成立**：当下游任务与预训练分布差异极大时，适配器可能被迫发生大的更新，超出正交空间。

3. **激活空间的线性结构假设**
   - **在哪里偷偷引入的？** 论文对 $\mathbf{X}_{\text{pre}}$ 做线性 SVD 分解，隐含假设了神经网络的激活空间可以通过线性子空间来有效刻画。但实际的激活经过非线性激活函数（如 ReLU, GELU），其几何结构是非线性的流形。
   - **什么条件下成立**：如果考虑的是激活函数之前的线性变换（$\mathbf{Wx}$）而不是之后的非线性输出（$\phi(\mathbf{Wx})$），这是合理的。论文在第 2 页明确提到 "consider the null space of $\mathbf{X}_{\text{pre}}$" 并引用 "$\phi(\mathbf{Wx})$"——但他们实际在 $\mathbf{X}_{\text{pre}}$（被定义为"input activations"）上操作，具体是线性变换的输入还是输出？论文的 $\mathbf{X}_{\text{pre}}$ 定义（第 3 页）是"the input activations of a linear layer"，即进入线性层的输入，也就是上一层的输出（可能是非线性后的）。这导致了概念上的模糊性：论文一会说考虑 $\phi(\mathbf{Wx})$，一会又在线性激活前的输入上做 SVD。

4. **跨层信息传递的线性假设**
   - **在哪里偷偷引入的？** 论文论证 $\mathbf{X}_{\text{pre}}$ 比 $\mathbf{W}_0$ 包含更多信息时说 "$\mathbf{X}_{\text{pre}}$ takes into account the parameters from all previous layers as well as the input data"——但 $\mathbf{X}_{\text{pre}}$ 虽然在计算上游包括了前面的层，但它是一个确定性的数值矩阵，对这些上游信息的"编码"方式是非线性的，而 SVD 是一个线性操作。SVD 能否有效解耦这种非线性编码的信息，并没有被讨论。

## 与前驱工作的逻辑关系

### 继承了谁的思路/方法？

1. **LoRA (Hu et al., 2022)**：低秩分解 $\Delta \mathbf{W} = \mathbf{BA}$ 的框架是一切的基础。
2. **MiLoRA (Wang et al., 2025a)**：直接继承了"在零空间中初始化 LoRA"的思路，但将零空间从 $\mathbf{W}_0$ 切换到了 $\mathbf{X}_{\text{pre}}$。
3. **CorDA (Yang et al., 2024)**：继承了使用校准集和激活协方差矩阵 $\mathbf{C} = \mathbf{X}_{\text{pre}} \mathbf{X}_{\text{pre}}^{\top}$ 来引导分解的思想，以及"校准集代表预训练知识"的设定。
4. **PiSSA (Meng, Wang, Zhang, 2024)**：虽然 PiSSA 的目标是加速下游任务而非知识保留，但它开创性地使用了 SVD 来初始化 LoRA，这一点影响了后续的所有方法。

### 它解决了前人的什么根本局限？

**MiLoRA 的根本局限**：MiLoRA 在 $\mathbf{W}_0$ 的零空间中初始化 LoRA。但 $\mathbf{W}_0$ 只包含**当前层**的权重信息。神经网络的输出是 $\phi(\mathbf{W}_L \cdots \phi(\mathbf{W}_1 \mathbf{x}))$，预训练知识通过所有层的级联体现。$\mathbf{W}_0$ 的零空间只保证在该层权重层面与当前层正交，但不保证在**激活空间**中与预训练知识正交。论文用 Figure 2 和 Table 2 展示了 $\mathbf{W}_0$ 的有效秩远高于 $\mathbf{X}_{\text{pre}}$（LLaMA-3.2-3B layer 0 k_proj：$\mathbf{W}_0$ 有效秩 548.30 vs $\mathbf{X}_{\text{pre}}$ 有效秩 101.28），这意味着 $\mathbf{W}_0$ 的奇异值分布更均匀，其零空间包含更多信息——即 $\mathbf{W}_0$ 的零空间"不够干净"，混入了更多预训练知识的成分。

**CorDA 的根本局限**：CorDA 的 $\mathbf{BA}$ 空间既不是 $\mathbf{W}_0$ 的零空间也不是 $\mathbf{X}_{\text{pre}}$ 的零空间——它提取的是与协方差引导后的**主成分**。这意味着 CorDA 的 LoRA 初始化本身就可能包含预训练知识的主要方向，在微调中更容易破坏这些方向。

### 它引入或改变了什么约束/假设？

- **改变了"两个目标同样重要"的隐含假设**：此前 MiLoRA 和 CorDA 同时追求(1)残差权重接近 $\mathbf{W}_0$ 和(2) LoRA 空间正交于预训练知识。论文证明目标(2)才是关键的，目标(1)可以被放弃。
- **引入了"激活的零空间优于权重的零空间"的假设**：用有效秩的概念量化了这种优势。
- **去掉了"$\mathbf{BA}$ 是 $\mathbf{W}_0$ 的次要分量"的约束**：在 MiLoRA 中，$\mathbf{BA}$ 必须是 $\mathbf{W}_0$ 的次要奇异分量；在 LoRA-Null 中，$\mathbf{BA}$ 是 $\mathbf{W}_0$ 在 $\mathbf{X}_{\text{pre}}$ 零空间上的投影，这**不是** $\mathbf{W}_0$ 的次要分量。

## 留下的开放问题

1. **理论上的开放问题（论文自己承认的）**：对 $\mathbf{W}_0 \mathbf{X}_{\text{pre}}$ 做 SVD（即不使用逆协方差矩阵 $\mathbf{C}^{-1}$）是否优于对 $\mathbf{W}_0 \mathbf{X}_{\text{pre}} \mathbf{X}_{\text{pre}}^{\top}$ 做 SVD？论文承认这是开放问题（Theorem 3 证明部分）。CorDA 额外乘了 $\mathbf{X}_{\text{pre}}^{\top}$ 并乘以 $\mathbf{C}^{-1}$，但这一操作是否有理论保证未被证明。

2. **LoRA-Null 的残差权重完全不可靠**：Figure 5 显示，如果去除 LoRA 适配器只保留残差权重 $\mathbf{W}_0'$，LoRA-Null 的知识保留效果远差于 MiLoRA 和 CorDA。这意味着 LoRA-Null 完全依赖于适配器的输出——一旦适配器被移除或损坏，知识就会大幅丢失。这在部署时是一个隐患：适配器的物理损坏或精度损失会导致严重后果。

3. **为什么 $\mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top}$ 的 rank 可以小于 $r$？** Eq.(9) 指出 $\text{rank}(\mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top}) < \text{rank}(\mathbf{U}_{\text{null}}) = r$，但没有讨论这个 rank 亏缺在什么条件下发生，以及它的实际影响。

4. **校准集的选择标准**：论文使用 NQ Open 作为校准集，但 NQ Open 是一个问答数据集，可能只覆盖了预训练知识的一小部分（事实型知识）。对于更广泛的预训练知识（如推理能力、语法知识、世界常识），NQ Open 是否足够代表？论文没有讨论校准集选择与知识覆盖范围的关系。

5. **为什么 LoRA-Null 在 LLaMA-3.1-8B 上略逊于 MiLoRA？** Table 12 显示 MiLoRA 的 GM=35.24 而 LoRA-Null=35.09。论文没有深入分析这个反例。

6. **明显但论文没做的实验**：
   - 没有在更大模型（LLaMA-3-70B 等）上验证
   - 没有分析 $\mathbf{U}_{\text{null}}$ 的行数选择（即 $r$）与 $\mathbf{X}_{\text{pre}}$ 奇异值谱之间的定量关系——到底尾部多大算"小"？
   - 没有测试跨领域校准集的效果（如用代码数据做校准集但微调数学任务）
   - 没有分析 $\mathbf{BA}$ 在微调过程中向预训练知识方向的旋转程度

### 如果你来改进这篇工作，你会从哪里入手？

1. **理论方面**：给出 $\|\mathbf{U}_{\text{null}}^{\top} \mathbf{X}_{\text{pre}}\|_F$ 的上界与奇异值尾部能量的定量关系，提供 $r$ 选择的自适应标准。

2. **算法方面**：在微调过程中加入对 $\mathbf{BA}$ 的正交性约束（如正则化项 $\|\mathbf{U}_{\text{null}}^{\top} (\mathbf{BA})^{\top} \|_F$），防止适配器在微调中旋转出零空间。

3. **校准集方面**：尝试多源校准集（NQ Open + Pile + Code），评估校准集质量对结果的影响。

4. **实验方面**：做 LoRA-Null + 正交性保持正则化 的消融实验；在更大的模型和更多的下游任务上验证。

## 在领域中的范式定位

### 所属范式/技术路线

**PEFT → LoRA初始化策略 → 知识保留型初始化 → 激活感知的零空间初始化**

更大的范式图谱：
- **参数高效微调（PEFT）**：Adapter、Prefix Tuning、LoRA、IA³ 等
- **LoRA改进流派**：
  - **初始化改进**：PiSSA（主分量→加速）、MiLoRA（权重零空间→知识保留）、CorDA（上下文感知）、**LoRA-Null（激活零空间→知识保留）**
  - **结构改进**：DoRA（权重分解）、LoRA+（学习率设置）
  - **量化结合**：QLoRA、GPTQ-LoRA

LoRA-Null 属于**"在 LoRA 初始化中注入先验知识"**这一子范式。它的独特之处在于：它完全放弃了"残差权重接近预训练权重"这一直觉，将注意力全部集中到"初始化空间与知识正交"上。

### "Extreme"到底是什么？

论文标题中的 "Put the Space of LoRA Initialization to the Extreme" 包含了两层的"极限"：

1. **第一层极限（空间选择的极限）**：从 $\mathbf{W}_0$ 的零空间（MiLoRA）推进到 $\mathbf{X}_{\text{pre}}$ 的零空间。$\mathbf{X}_{\text{pre}}$ 的有效秩远小于 $\mathbf{W}_0$，所以其零空间更大、更"纯粹"——这是对零空间选择的**极致化**。

2. **第二层极限（设计自由度的极限）**：完全放弃"让 $\mathbf{W}_0'$ 接近 $\mathbf{W}_0$"的目标（Theorem 3 表明 LoRA-Null 不是 Eq.4 或 Eq.5 的解），只追求"LoRA 初始化空间与预训练知识正交"。这意味着残差权重 $\mathbf{W}_0'$ 可能**严重偏离** $\mathbf{W}_0$——这是对"只关注初始化空间"这一原则的**极致化**。

### MiLoRA (weight null space) vs LoRA-Null (activation null space)

**论文的核心论证链路：**

1. **信息含量差异**：$\mathbf{X}_{\text{pre}}$ 包含了所有前面层的参数和输入数据信息，而 $\mathbf{W}_0$ 只包含当前层信息。这意味着 $\mathbf{X}_{\text{pre}}$ 对预训练知识的"表征"比 $\mathbf{W}_0$ 更全面——它不仅编码了当前层的变换，还编码了所有前面的计算历史。

2. **有效秩差异的实验证据**：Table 2（完整版 Table 6）显示在 LLaMA-3.2-3B 上，$\mathbf{W}_0$ 的 effective rank 在 548.30（k_proj, layer 0）到 2883.30（down_proj, layer 0）之间，而 $\mathbf{X}_{\text{pre}}$ 的 effective rank 在 75.60（o_proj, layer 0）到 758.06（down_proj, layer 0）之间。$\mathbf{X}_{\text{pre}}$ 的有效秩比 $\mathbf{W}_0$ 小 1-2 个数量级。这意味着：
   - $\mathbf{X}_{\text{pre}}$ 的奇异值分布更陡峭（Figure 2），信息集中在极少数方向
   - $\mathbf{X}_{\text{pre}}$ 的零空间相对更大，且包含的信息**比例**更少
   - 在 $\mathbf{X}_{\text{pre}}$ 的零空间中做 LoRA 更新，比在 $\mathbf{W}_0$ 的零空间中做更新更"安全"

3. **为什么 $\mathbf{X}_{\text{pre}}$ 有效秩更小？** 论文的解释是：$\mathbf{X}_{\text{pre}}$ 经过了前面所有层的非线性变换，这些变换不断"聚焦"信息，使得激活空间的不确定性（熵）降低，有效秩减少。这类似于多层神经网络对数据进行层级化抽象压缩：每一层都在消除冗余信息，所以越深的层，其激活的有效秩越低。

**完整的数学推导过程：**

神经网络第 $l$ 层的输入激活为：
$$\mathbf{X}_{\text{pre}}^{(l)} = \phi^{(l-1)}\big(\mathbf{W}^{(l-1)} \cdots \phi^{(1)}(\mathbf{W}^{(1)} \mathbf{X}^{(0)})\big)$$

这里 $\mathbf{X}^{(0)}$ 是原始输入（token embedding），$\phi^{(i)}$ 是非线性激活函数，$\mathbf{W}^{(i)}$ 是第 $i$ 层的权重。$\mathbf{X}_{\text{pre}}^{(l)}$ 通过层级的非线性变换级联生成，"携带"了前面所有层的信息。

MiLoRA 对 $\mathbf{W}^{(l)}$ 本身做 SVD 并取零空间：
$$\mathbf{W}^{(l)} = \mathbf{U}^{(l)} \mathbf{S}^{(l)} (\mathbf{V}^{(l)})^{\top}, \quad \mathbf{BA}_{\text{MiLoRA}} = \mathbf{U}^{(l)}_{[:, R-r+1:R]} \mathbf{S}^{(l)}_{[R-r+1:R, :]} (\mathbf{V}^{(l)})^{\top}$$

LoRA-Null 对 $\mathbf{X}_{\text{pre}}^{(l)}$ 做 SVD 并取零空间：
$$\mathbf{X}_{\text{pre}}^{(l)} = \mathbf{U}^{(l)}_{X} \mathbf{\Sigma}^{(l)}_{X} (\mathbf{V}^{(l)}_{X})^{\top}, \quad \mathbf{BA}_{\text{LoRA-Null}} = \mathbf{W}^{(l)} \mathbf{U}^{(l)}_{X, \text{null}} (\mathbf{U}^{(l)}_{X, \text{null}})^{\top}$$

关键区别：MiLoRA 的 $\mathbf{BA}$ 列空间在 $\mathbf{W}^{(l)}$ 的右奇异向量的尾部，而 LoRA-Null 的 $\mathbf{BA}$ 列空间在 $\mathbf{W}^{(l)}$ 与 $\mathbf{X}_{\text{pre}}^{(l)}$ 零空间投影的叠加。

### PiSSA vs MiLoRA vs CorDA vs LoRA-Null 对比

| 方法 | 初始化目标 | BA 的来源 | 残差权重接近 W₀？ | BA 空间正交于预训练知识？ |
|------|-----------|-----------|-----------------|------------------------|
| **PiSSA** | 加速下游任务 | $\mathbf{W}_0$ 的主奇异分量 | 是 | 否（BA本身就是预训练知识的主成分） |
| **MiLoRA** | 知识保留 | $\mathbf{W}_0$ 的次要奇异分量 | 是 | 部分（$\mathbf{W}_0$ 的零空间，不完整） |
| **CorDA(KP)** | 知识保留 + 上下文感知 | $\mathbf{W}_0\mathbf{C}$ 的主分量 × $\mathbf{C}^{-1}$ | 是 | 否（主分量方向，最"远"于零空间） |
| **LoRA-Null** | 知识保留（极致） | $\mathbf{W}_0$ 在 $\mathbf{X}_{\text{pre}}$ 零空间上的投影 | **否**（完全不考虑） | 是（最"纯"的零空间） |

Figure 4 的实验支持：将各方法的 $\mathbf{A}$ 矩阵投影到 $\mathbf{U}_{[:, i]}$（$\mathbf{X}_{\text{pre}}$ 的左奇异向量基）上：
- LoRA-Null 的 $\mathbf{A}$ 只落在 $\mathbf{U}_{\text{null}}$ 上（尾部方向）
- MiLoRA 的 $\mathbf{A}$ 主要落在尾部，但仍有部分在主成分方向上
- CorDA 的 $\mathbf{A}$ 更偏向尾部（比 MiLoRA 更集中），但仍然在尾部与主成分之间有分布

论文对 CorDA 优于 MiLoRA 的解释：$\mathbf{W}_0 \mathbf{X}_{\text{pre}} \mathbf{X}_{\text{pre}}^{\top} = \mathbf{W}_0 \mathbf{U} \mathbf{\Sigma}^2 \mathbf{U}^{\top}$，这放大了 $\mathbf{U}$ 中较大奇异值对应方向上的 $\mathbf{W}_0$ 分量，衰减了较小奇异值对应方向上的分量。然后乘以 $\mathbf{C}^{-1} = \mathbf{U} \mathbf{\Sigma}^{-2} \mathbf{U}^{\top}$ 会抵消这种放大，回到原来的子空间。因此 CorDA 的初始化空间主要由 $\mathbf{U}$ 中小奇异值对应方向主导——这解释了为什么 CorDA 比 MiLoRA 更接近 $\mathbf{X}_{\text{pre}}$ 的零空间。
