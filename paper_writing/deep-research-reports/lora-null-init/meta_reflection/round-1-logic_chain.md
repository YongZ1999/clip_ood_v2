# Round 1: 逻辑链反思 — LoRA初始化空间与预训练知识保留

## 领域要解决的核心问题
**在参数高效微调（PEFT）中，如何在用 LoRA 适配下游任务的同时，最大程度保留预训练知识（即防止灾难性遗忘）？** 核心矛盾是：微调需要修改权重以适应新任务，但修改权重会覆盖已有的有用知识。

---

## 演化链

### 起点：LoRA (Hu et al., 2021)
- **核心公式**: $\mathbf{Y} = \mathbf{X}(\mathbf{W}_0 + \mathbf{B}\mathbf{A})$, $\mathbf{A} \sim \mathcal{N}$, $\mathbf{B} = \mathbf{0}$
- **核心假设**: 权重更新 $\Delta\mathbf{W}$ 是低秩的
- **根本局限**: "噪声+零"初始化 → 梯度初始为零 → 需要热身 → 收敛慢；且适配器的搜索空间**无引导**，可能无意中覆盖预训练知识
  - 梯度分析: $\partial\mathcal{L}/\partial\mathbf{A} = \mathbf{X}^T (\partial\mathcal{L}/\partial\mathbf{Y}) \mathbf{B}^T = \mathbf{0}$（因 $\mathbf{B}=\mathbf{0}$）
  - 方向随机: $\partial\mathcal{L}/\partial\mathbf{B} = \mathbf{A}^T \mathbf{X}^T (\partial\mathcal{L}/\partial\mathbf{Y})$ 通过噪声 $\mathbf{A}$ 传播，方向随机

```
↓ LoRA 的局限：梯度初始为零 + 搜索空间无引导，导致"学得慢"且"可能忘"
```

### PiSSA — 训练主成分，冻结残差 (Meng et al., 2024)
- **核心公式**: $\mathbf{W}_0 = \underbrace{\mathbf{W}^{\text{prin}}}_{\mathbf{A}\mathbf{B} \text{ 可训练}} + \underbrace{\mathbf{W}^{\text{res}}}_{\text{冻结}}$, $\mathbf{A} = \mathbf{U}_{:,:r}\mathbf{S}^{1/2}$, $\mathbf{B} = \mathbf{S}^{1/2}\mathbf{V}_{:r,:}^T$
- **核心假设**: 预训练权重的**主成分方向**（大奇异值）也是微调时最有信息量的梯度方向
- **解决了 LoRA 的什么局限**: 梯度从第一步非零 → 收敛快；搜索空间被引导到 $\mathbf{W}_0$ 的主方向 → 近似全参数微调
- **引入的新矛盾**: **高遗忘风险**。主成分编码了 $\mathbf{W}_0$ 的核心知识，修改它们导致较高的灾难性遗忘（遗忘损失 6.07）

```
↓ PiSSA 的局限：学得快、忘得快（"修改了骨架"）
```

### MiLoRA — 冻结主成分，训练次成分 (Wang et al., 2024)
- **核心公式**: $\mathbf{W}_0 = \underbrace{\mathbf{W}_p}_{\text{冻结主成分}} + \underbrace{\mathbf{W}_m}_{\mathbf{B}_m\mathbf{A}_m \text{ 可训练}}$, $\mathbf{B}_m = \mathbf{U}_m\mathbf{S}_m^{1/2}$, $\mathbf{A}_m = \mathbf{S}_m^{1/2}\mathbf{V}_m^T$
- **核心假设**: 主成分编码重要知识 → 应冻结；次成分（小奇异值）包含噪声/长尾信息 → 可安全修改
- **解决了 PiSSA 的什么局限**: 通过冻结主成分、训练次成分，大幅降低了遗忘损失（2.54 vs PiSSA 6.07）
- **根本创新**: 提出了"**LoRA 初始化空间**"的概念——适配器应该在预训练知识的**零空间**中初始化（$\text{Row}(\mathbf{W}_m) = \mathcal{N}(\mathbf{W}_p)$）
- **引入的新矛盾**: 
  - 次成分的正交性**只在初始化时成立**，训练中可能漂移（论文选择"softer approach"，不强制约束）
  - 如果 Full FT 的 $\Delta\mathbf{W}$ 主要落在主成分子空间中，MiLoRA 的搜索空间限制可能**限制表达力**
  - **最关键的局限**: $\mathbf{W}_0$ 的零空间"不够纯"——$\mathbf{W}_0$ 的有效秩很高（~500-2900），其零空间仍包含大量预训练知识相关信息
  
$$\text{MiLoRA: } \text{Row}(\mathbf{W}_m) = \mathcal{N}(\mathbf{W}_p)$$

```
↓ MiLoRA 的局限：权重零空间不够"纯粹"——W₀ 有效秩高，零空间混入过多预训练知识
```

### LoRA-Null — 激活零空间初始化 (Tang et al., 2025)
- **核心公式**: 
  - $\mathbf{X}_{\text{pre}} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$（对激活做 SVD）
  - $\mathbf{U}_{\text{null}} = \mathbf{U}_{[:,R-r+1:R]}$（取最小的 $r$ 个左奇异向量）
  - $\mathbf{B}\mathbf{A} = \mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top}$（将 $\mathbf{W}_0$ 投影到激活零空间）
- **核心洞察**: 
  1. **空间 > 残差**: 论文证明"残差权重接近 $\mathbf{W}_0$"不是关键，关键是"初始化空间与预训练知识正交"
  2. **激活 > 权重**: $\mathbf{X}_{\text{pre}}$ 比 $\mathbf{W}_0$ 的有效秩**小1-2个数量级** → 其零空间更"纯净"
- **解决了 MiLoRA 的什么局限**: 
  - MiLoRA 用 $\mathbf{W}_0$ 的零空间，不够纯 → LoRA-Null 用 $\mathbf{X}_{\text{pre}}$ 的零空间，更纯
  - 激活经过前层非线性变换后，信息被"压缩" → 有效秩降低 → 零空间更大且更干净
  - Avg1(Per) 提升: LoRA-Null 79.21% vs MiLoRA 69.66%
- **引入的新矛盾**: 
  - **极端化的代价**: 残差权重 $\mathbf{W}_0'$ 严重偏离 $\mathbf{W}_0$（Figure 5：去除适配器后知识保留最差）
  - **校准集依赖**: NQ Open 能否代表全部预训练知识？校准集质量是关键瓶颈
  - 在特定设置下（LLaMA-3.1-8B）略逊于 MiLoRA

``` 
↓ LoRA-Null 的局限：完全依赖适配器（残差不可靠）+ 校准集依赖 + 极端化可能非最优
```

### Least but not Last — 中间分量最优 (Quercia et al., 2026)
- **核心公式（广义框架）**: $\mathbf{A} = \mathbf{U}_{s:s+r}\sqrt{\mathbf{D}_{s:s+r}}$, $\mathbf{B} = \sqrt{\mathbf{D}_{s:s+r}}\mathbf{V}_{s:s+r}^T$，统一了 PiSSA (s=0) 和 MiLoRA (s=m-r)
- **核心发现**: **U形遗忘曲线**——两端（首分量或末分量）都比中间分量造成更大的遗忘
- **为什么中间更优**: 
  - 微调**极端的奇异分量**（无论是最大还是最小）会导致奇异值方向的**旋转**（off-diagonal changes），损伤主成分方向编码的核心知识
  - 中间分量在达到相同下游任务性能的同时，引起的最小化旋转最小
- **解决了前述方法的什么局限**: 揭示了"首 vs 末"的二元选择是次优的——最优解不在端点
- **引入的新矛盾**: 
  - **挑战 LoRA-Null 的"极端化"策略**: 如果中间最优，LoRA-Null 的"极致的零空间"是否也是次优的？
  - 不过，LoRA-Null 是在**激活空间**（不是权重空间）中找零空间，不在 Least 的框架内
  - 这是一个**未解决的张力**: "极端化零空间" vs "中间分量"——两条看似矛盾的发现

``` 
↓ Least but not Last 挑起的未解决问题：中间最优 vs 激活零空间最纯——哪个对？
```

### LoRA Subtraction / LoRA- — 动态训练空间约束 (Liu & Chang, 2025)
- **核心机制**: 非 SVD 初始化，而是**训练约束**——减去旧知识的 LoRA 权重得到"遗忘模型"→ 在新模型上提取特征主成分（DRS）→ 将梯度投影到 DRS
- **哲学差异**: 不是"从哪出发"（初始化策略），而是"往哪走"（训练约束策略）
- **与前述方法的关系**: 正交的维度——**初始化策略 vs 训练约束策略**可以互补

---

## 关键转折点

1. **转折 1: PiSSA (2024)** — 首次用 SVD 分解预训练权重来初始化 LoRA，证明了"初始化质量"对 PEFT 性能有决定性影响。打开了"LoRA 初始化改进"这一子领域。

2. **转折 2: MiLoRA (2024)** — 反转 PiSSA 的选择（训练次成分而非主成分），提出了"权重零空间初始化"的概念，将焦点从"加速收敛"转向"知识保留"。

3. **转折 3: LoRA-Null (2025)** — 从权重零空间跨越到**激活零空间**，证明零空间"纯度"的重要性，并打破"残差权重需接近预训练权重"的先验直觉。

4. **转折 4: Least but not Last (2026)** — 打破"非首即末"的二元思维，揭示 U 形遗忘曲线，证明中间分量在 trade-off 上优于两端。

---

## 演化断裂（Gaps）

- **Gap 1: LoRA-Null vs Least but not Last 的张力未解决**。LoRA-Null 主张"极致的激活零空间"，Least but not Last 主张"中间分量最优"。如果激活零空间可以理解为"在激活层面上取末分量"，那它可能同样面临 U 形遗忘的问题。但 LoRA-Null 的 $\mathbf{BA} = \mathbf{W}_0\mathbf{U}_{\text{null}}\mathbf{U}_{\text{null}}^{\top}$ 无法简化为 Least 框架中的 $s=m-r$（因为经过了 $\mathbf{W}_0$ 的投影）。需要一个统一的理论框架来解释两者的关系。

- **Gap 2: 训练中的正交性保持未被研究**。所有 SVD-based 初始化方法（PiSSA/MiLoRA/LoRA-Null）都只在初始化时保证正交性，训练中适配器可以自由漂移。LoRA- 提供了训练约束的思路但尚未与初始化方法结合。正交性正则化可能是填补这一 gap 的关键。

- **Gap 3: 缺少跨领域校准集的系统研究**。LoRA-Null 和 CorDA 都使用 NQ Open 作为校准集，但没有论证为什么 NQ Open（一个事实型 QA 数据集）能代表广泛的预训练知识（包括推理能力、语法知识等）。

- **Gap 4: 缺乏"初始化 + 训练约束"联合优化的工作**。至今的工作要么只改进初始化（PiSSA/MiLoRA/LoRA-Null/Least），要么只改进训练约束（LoRA-）。两者结合的潜力未被探索。

---

## 未解决的问题

1. **零空间纯度 vs 中间分量——哪个才是真正的优化目标？** U 形遗忘曲线暗示"极端"（无论什么空间中的极端）是次优的，但如果零空间的纯度是关键，激活零空间应该最纯。这两者如何调和？

2. **校准集无关的零空间构造方法**。是否可以在不需要校准集的情况下构造激活零空间？例如，利用模型自身的生成分布或合成数据？

3. **跨模型、跨任务的泛化**。目前的工作主要在 LLaMA 族模型和 math/code/QA 任务上验证。在更大模型（70B+）、更多模型族（Mistral, Gemma, Qwen）和更多任务类型上的表现未知。

4. **LoRA rank 与奇异值谱的关系**。当 rank r 接近最大奇异值数量时，"极端 vs 中间"的边界变得模糊。不同 rank 下最优策略是否会变化？
