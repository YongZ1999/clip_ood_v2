# LoRA初始化空间与预训练知识保留 —— 深度调研报告

> **调研主题**: "Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge" 及相关研究
>
> **反思深度**: 🥇 穷极式思考（逻辑链 → 元认知 → 范式批判 → 自我反驳）
>
> **精读论文**: PiSSA (2024) → MiLoRA (2024) → LoRA-Null (2025) → Least but not Last (2026) → LoRA Subtraction (2025)

---

## 0. 执行摘要

**领域核心矛盾**: 参数高效微调（LoRA）在适应下游任务时必然修改预训练权重，但修改方向与预训练知识冲突时会导致灾难性遗忘——如何在**零额外开销**的前提下让适配器"只学新知识、不覆盖旧知识"？

**技术演进主线**: 从"随机初始化"（标准LoRA）到"SVD结构初始化"（PiSSA, 2024），从"训练主成分"到"在权重零空间中训练次成分"（MiLoRA, 2024），再到"在激活零空间中初始化"（LoRA-Null, 2025），演进的核心驱动力是不断寻找更"纯净"的零空间——一个与预训练知识彻底正交的子空间，让适配器在其中安全更新而不破坏已有知识。但2026年的新工作（Least but not Last）挑战了这一"极端化"策略，发现**中间分量**（而非两端的最优/最劣奇异分量）才能在知识保留和新任务适应间实现最佳平衡。

**最有价值的发现（自我反驳后修正）**: 这一领域存在一个**三轴设计空间**：(1) 奇异谱窗口选择（首/中/末），(2) 知识表征层次（权重 vs 激活），(3) 约束机制（初始化 vs 训练约束）。大多数现有方法只探索了一个轴，而真正的突破可能来自多个轴的联合优化——特别是"好的SVD初始化 + 轻量训练约束"的组合。

---

## 1. 知识地图

### 范式分类

```
LoRA微调知识保留方法
│
├── 权重SVD初始化路线 (Weight-level SVD)
│   ├── PiSSA (Meng et al., 2024) — 训练主成分/首分量
│   ├── MiLoRA (Wang et al., 2024) — 训练次成分/末分量（权重零空间）
│   └── Least but not Last (Quercia et al., 2026) — 训练中间分量
│
├── 激活SVD初始化路线 (Activation-level SVD)
│   ├── CorDA (Yang et al., 2024) — 上下文感知分解（知识保留模式）
│   └── LoRA-Null (Tang et al., 2025) — 激活零空间初始化 ⬅️ 目标论文
│
└── 训练约束路线 (Training-time Constraint)
    └── LoRA Subtraction / LoRA- (Liu & Chang, 2025) — DRS梯度投影
```

### 论文地图

| 论文 (年份) | 范式 | 知识保留 | 下游性能 | 计算开销 | 模型覆盖 |
|-----------|------|---------|---------|---------|---------|
| PiSSA (2024) | 权重SVD-首分量 | ★★☆☆☆ | ★★★★★ | 低 | LLaMA, Mistral, Gemma |
| MiLoRA (2024) | 权重SVD-末分量 | ★★★★☆ | ★★★★☆ | 低 | LLaMA, LLaVA |
| LoRA-Null (2025) | **激活SVD-零空间** | **★★★★★** | ★★★★☆ | 中 | LLaMA, Gemma |
| Least (2026) | 权重SVD-中间分量 | ★★★★☆ | ★★★★★ | 低 | ViT, LLaMA |
| LoRA- (2025) | 训练约束 | ★★★★☆ | ★★★☆☆ | 高 | ViT (CL) |

### 路线间关系

- **PiSSA ↔ MiLoRA 是对称互补的**: 同一SVD框架的两个端点（s=0 vs s=m-r）。实验设置不同时优势逆转（MiLoRA Table 9）。
- **MiLoRA → LoRA-Null 是递进关系**: 从权重的零空间到激活的零空间，提升了零空间"纯度"。
- **Least 挑战 PiSSA/MiLoRA 的二元选择**: 证明中间 > 两端。
- **LoRA-Null vs Least 存在未解决的张力**: "极致的激活零空间" vs "中间分量最优"——两者不在同一框架中，未有直接对比实验。
- **LoRA- 与其他方法正交**: 训练约束可与任何初始化方法叠加。

---

## 2. 核心方法演进（逻辑链）

### 起点: LoRA (Hu et al., 2021)

**问题**: 全参数微调需要海量显存（LLaMA 65B > 780GB）。LoRA 用低秩分解 $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$ 减少可训练参数。

**方案**: $\mathbf{A} \sim \mathcal{N}$, $\mathbf{B} = \mathbf{0}$，初始时 $\mathbf{A}\mathbf{B} = \mathbf{0}$，输出不变。

**本质**: 从"噪声+零"开始学习 → 梯度初始为零 → 需要热身步 → 收敛慢，且无引导。

### ↓ LoRA的局限: 梯度为零 + 方向随机

### PiSSA (Meng et al., 2024) — "从骨架开始学"

**问题**: LoRA 的初始化导致收敛慢。

**方案**: 对 $\mathbf{W}_0$ 做 SVD，用**主奇异分量**（前 $r$ 个最大的奇异值/向量）初始化适配器 $\mathbf{A}, \mathbf{B}$，残差冻结。

**公式**: 
$$\mathbf{W}_0 = \underbrace{\mathbf{U}_{:,:r}\mathbf{S}_{:r,:r}^{1/2}}_{\mathbf{A}} \underbrace{\mathbf{S}_{:r,:r}^{1/2}\mathbf{V}_{:r,:}^{\top}}_{\mathbf{B}} + \underbrace{\mathbf{U}_{:,r:}\mathbf{S}_{r:,:}\mathbf{V}_{:,r:}^{\top}}_{\text{冻结 }\mathbf{W}^{\text{res}}}$$

**本质**: 将洛RA从"近似 $\Delta\mathbf{W}$"改为"近似 $\mathbf{W}_0$ 的主成分" → 梯度从第一步非零 → 收敛快。但**主成分是权重的核心骨架，修改它们 → 高遗忘风险**（遗忘损失 6.07）。

### ↓ PiSSA的局限: 学得快、忘得快（修改核心知识）

### MiLoRA (Wang et al., 2024) — "从噪声开始调"

**问题**: PiSSA 遗忘过高。

**方案**: 反转选择——冻结主成分（前 $m-r$ 个），用**次成分**（后 $r$ 个）初始化适配器。

**公式**:
$$\mathbf{W}_0 = \underbrace{\mathbf{U}_p\mathbf{S}_p\mathbf{V}_p^{\top}}_{\text{冻结 }\mathbf{W}_p} + \underbrace{\mathbf{U}_m\sqrt{\mathbf{S}_m}}_{\mathbf{B}_m} \underbrace{\sqrt{\mathbf{S}_m}\mathbf{V}_m^{\top}}_{\mathbf{A}_m}$$

**关键洞察**: 次成分的行空间 = 主成分的零空间 → $\text{Row}(\mathbf{W}_m) = \mathcal{N}(\mathbf{W}_p)$ → 适配器在 $\mathbf{W}_p$ 的零空间中初始化 → 不与预训练主方向冲突。

**效果**: 遗忘损失 2.54（PiSSA 6.07 的 1/2.4x），但性能在部分任务略低于 PiSSA。

### ↓ MiLoRA的局限: 权重的零空间"不够纯"（W₀有效秩高）

### LoRA-Null (Tang et al., 2025) — "激活零空间" ⬅️ 目标论文

**问题**: MiLoRA 的零空间在权重空间中，但 $\mathbf{W}_0$ 的有效秩高（~500-2900），零空间混入过多预训练知识。

**方案**: 对**激活值** $\mathbf{X}_{\text{pre}}$（预训练数据通过前层后的输出）做 SVD，取最小的 $r$ 个左奇异向量构造零空间，将 $\mathbf{W}_0$ 投影到该零空间中。

**公式**:
$$\mathbf{X}_{\text{pre}} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top}, \quad \mathbf{U}_{\text{null}} = \mathbf{U}_{[:,R-r+1:R]}$$
$$\mathbf{B}\mathbf{A} = \mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top}, \quad \mathbf{W}_0' = \mathbf{W}_0 - \mathbf{B}\mathbf{A}$$

**核心洞察**: 
1. $\mathbf{X}_{\text{pre}}$ 的有效秩比 $\mathbf{W}_0$ 小 1-2 个数量级（如 k_proj: 101 vs 548） → 激活零空间更大、更"纯净"
2. 残差权重接近 $\mathbf{W}_0$ 不是关键（Theorem 3），关键是**初始化空间与预训练知识正交**

**效果**: Avg1(Per) 79.21% vs MiLoRA 69.66%（提升 ~10pp），GM 综合指标最优

### ↓ LoRA-Null的局限: 残差权重不可靠（去除适配器后知识最差）+ 校准集依赖

### Least but not Last (Quercia et al., 2026) — "中间分量最优"

**问题**: "非首即末"的二元选择是次优的。

**方案**: 用统一框架 $\mathbf{A} = \mathbf{U}_{s:s+r}\sqrt{\mathbf{D}_{s:s+r}}$, $\mathbf{B} = \sqrt{\mathbf{D}_{s:s+r}}\mathbf{V}_{s:s+r}^T$ 统一 PiSSA (s=0) 和 MiLoRA (s=m-r)，然后证明 $0 < s < m-r$（中间）最优。

**核心发现**: **U形遗忘曲线**——两端导致更大的奇异值方向旋转（off-diagonal changes），损伤主成分；中间分量旋转最小、遗忘最少。

### ↓ 挑起的未解决问题: 中间最优 vs 激活零空间最优——哪个对？

### LoRA Subtraction / LoRA- (Liu & Chang, 2025) — "训练约束"

**问题**: 持续学习中多任务的特征漂移无法用初始化解决。

**方案**: 非 SVD 初始化，而是 **DRS 梯度投影**——减去旧任务的 LoRA 权重 → 在"遗忘模型"上提取特征主成分 → 训练时梯度投影到该空间。

**本质**: 与前述方法正交——不是"从哪出发"（初始化策略），而是"往哪走"（训练约束策略）。

---

## 3. 元认知洞察

### 核心假设变迁

| 假设 | 从"默认成立"到"被挑战" | 关键论文 |
|-----|----------------------|---------|
| LoRA 零初始化不会破坏知识 | 被挑战：零 = 从噪声开始，可能收敛到次优 | PiSSA |
| 残差权重接近 W₀ 是知识保留必要条件 | 被推翻：空间正交性才是关键 | LoRA-Null |
| 主成分 = 重要知识，次成分 = 可安全修改 | 被挑战：修改次成分也通过 off-diagonal rotation 损伤主成分 | Least |
| SVD 初始化引导足够防止漂移 | 被怀疑：训练中正交性可能快速丢失 | MiLoRA（自认）, LoRA-（隐式反驳） |

### 核心矛盾演化

**矛盾 1 → 矛盾 2 → 矛盾 3 的迁移**:
1. 早期（PiSSA vs LoRA）: **收敛速度**（主成分 vs 随机）
2. 中期（MiLoRA vs PiSSA）: **知识保留 vs 下游性能**（次成分 vs 主成分）
3. 近期（LoRA-Null vs Least vs LoRA-）: **零空间纯度 vs 表达力 vs 训练稳定性**（激活零空间 vs 中间分量 vs 梯度投影）

### 反直觉发现

1. **最"安全"的末分量修改反而比中间分量造成更大遗忘**（U 形曲线）
2. **最"纯"的激活零空间的残差权重最不可靠**（去除适配器时最差）
3. **LoRA 适配器的更新范数远小于预期**（LoRA-Null Figure 1: 仅 10-15x）

### 被忽略的连接

- **分层异质性**: 不同层（attention vs MLP）的奇异值分布差异可达 10x，统一策略是次优的
- **优化器隐式偏差**: AdamW 的权重衰减与 SVD 初始化的交互未被研究
- **有效学习率效应**: SVD 初始化可能隐式改变了有效学习率，混淆了"零空间选择"的贡献

---

## 4. 框架的自省

### 曾经以为正确、后来被修正的认知

1. ~~"零空间纯度越高越好"~~ → 修正: 纯度只在"保护输出响应"层面成立，MiLoRA 的权重零空间保护的是权重方向，两者保护不同对象。纯度不是唯一标准。

2. ~~"激活零空间最优遍压倒权重零空间"~~ → 修正: LoRA-Null 在 Avg1 上确实优于 MiLoRA，但残差权重完全不可靠（Figure 5）。最优解可能在两者之间。

3. ~~"中间分量最优是普遍规律"~~ → 修正: 在视觉模型上证据充分，在 LLM 上有限。且 LoRA-Null 的激活零空间不在 Least 框架内。

### 经过批判后的高置信结论

1. **初始化空间选择是知识保留的第一效应** — 所有论文一致支持
2. **激活空间比权重空间提供更"纯净"的知识表征** — 有效秩差异 5-10x，实验事实
3. **"残差权重接近预训练权重"不是知识保留的必要条件** — LoRA-Null Theorem 3 形式化证明

### 仍无法确定的二律背反

1. **零空间纯度 vs 表达力**: 无法同时测量和优化
2. **激活零空间 vs 中间分量**: 缺乏直接对比实验
3. **校准集依赖 vs 无校准集**: 缺少跨领域校准集的消融研究

---

## 5. 论文精读详情

### Paper 1: PiSSA — Principal Singular Values and Singular Vectors Adaptation (Meng et al., 2024)

- **核心贡献**: 首次证明用 SVD 分解预训练权重来初始化 LoRA 可以显著加速收敛并提升性能
- **关键公式**: $\mathbf{W}_0 = \mathbf{W}^{\text{prin}}_{\text{trainable}} + \mathbf{W}^{\text{res}}_{\text{frozen}}$
- **隐式假设**: "主方向 = 好方向"假设 — 当领域迁移大时可能不成立
- **开放问题**: 理论分析缺失；Gemma-7B 上 rank 增大时过参数化

### Paper 2: MiLoRA — Minor Singular Components for LLM Finetuning (Wang et al., 2024)

- **核心贡献**: 反转 PiSSA 的选择，冻结主成分训练次成分，提出"权重零空间初始化"概念
- **关键公式**: $\text{Row}(\mathbf{W}_m) = \mathcal{N}(\mathbf{W}_p)$
- **隐式假设**: 正交性在训练中保留（但实际上无需强制约束）
- **开放问题**: 训练中正交性漂移程度未被量化；不同超参数下优势逆转

### Paper 3: LoRA-Null — Activation Null Space Initialization (Tang et al., 2025) ⬅️ 目标论文

- **核心贡献**: 将零空间从权重空间推进到激活空间，证明"空间正交性 > 残差接近性"
- **关键公式**: $\mathbf{BA} = \mathbf{W}_0 \mathbf{U}_{\text{null}} \mathbf{U}_{\text{null}}^{\top}$, $\mathbf{U}_{\text{null}}$ 来自 $\mathbf{X}_{\text{pre}}$ 的 SVD 尾部
- **核心结果**: Avg1(Per) 79.21% vs MiLoRA 69.66%（+10pp）；GM 综合最优
- **隐式假设**: $\mathbf{U}_{\text{null}}^{\top} \mathbf{X}_{\text{pre}} \approx \mathbf{0}$ 的有效性（依赖于奇异值尾部是否真够小）
- **开放问题**: 残差权重最不可靠（Figure 5）；与 Least 的张力未解决

### Paper 4: Least but not Last — Intermediate Components (Quercia et al., 2026)

- **核心贡献**: 统一 PiSSA/MiLoRA 框架并证明中间分量最优
- **关键发现**: U 形遗忘曲线 — off-diagonal rotation 导致两端遗忘更多
- **局限**: 主要在 ViT 上验证，LLM 验证有限；未涉及激活空间

### Paper 5: LoRA Subtraction — Drift-Resistant Space (Liu & Chang, 2025)

- **核心贡献**: 提出训练约束范式（非初始化范式）解决持续学习中的遗忘
- **关键机制**: 减去旧 LoRA 权重 → 在"遗忘模型"上提取 DRS → 梯度投影
- **连接**: 与所有初始化方法正交，可叠加使用

---

## 6. 全景对比表

| 维度 | PiSSA (2024) | MiLoRA (2024) | LoRA-Null (2025) | Least (2026) | LoRA- (2025) |
|------|-------------|--------------|-----------------|-------------|-------------|
| **SVD 对象** | $\mathbf{W}_0$ | $\mathbf{W}_0$ | $\mathbf{X}_{\text{pre}}$ | $\mathbf{W}_0$ | $\widetilde{\mathbf{X}}_t^l$ |
| **可训练部分** | 前 r 个奇异分量 | 后 r 个奇异分量 | $\mathbf{W}_0$ 在 $\mathbf{U}_{\text{null}}$ 的投影 | 中间 r 个奇异分量 | 标准 LoRA（训练约束） |
| **残差接近性** | 是 | 是 | **否**（弃用） | 是 | 不适用 |
| **零空间正交性** | 否 | 是（权重） | **是（激活，更纯）** | 部分（中间） | 不适用 |
| **训练约束** | 无 | 无（soft） | 无（soft） | 无 | 有（DRS 投影） |
| **校准集依赖** | 否 | 否 | **是**（NQ Open） | 否 | 是（当前任务数据） |
| **Avg1(Per)** | ~55-65%¹ | 69.66% | **79.21%** | 无直接可比数据 | 无直接可比数据 |
| **遗忘损失** | 6.07 | **2.54** | — | — | — |
| **下游性能** | 强（最高） | 中强 | 中强（最优 GM） | 强 | 中 |

¹ PiSSA 未直接报告 Avg1(Per)，从 LoRA-Null Table 1 中 LoRA 的 Avg1(Per) ≈ 55-65% 推断。

---

## 7. 开放问题与未来方向

### 短期可解决的问题

1. **LoRA-Null vs Least 的直接对比**: 在 LLaMA-2-7B 上，在相同设置下对比 LoRA-Null、MiLoRA、Least 和"激活中间分量"（取 X_pre 的中间而非最小奇异向量构造 U_null）——这个实验可以解决领域最大的张力。

2. **跨领域校准集的消融**: 使用 Pile、Code、混合数据作为 LoRA-Null 的校准集，评估 Avg1 的变化。如果变化大 → 校准集是瓶颈；如果变化小 → LoRA-Null 的核心 claim 增强。

3. **训练中的正交性追踪**: 在 MiLoRA/LoRA-Null 训练过程中，持续监控 $\|\mathbf{U}_p^{\top}\mathbf{B}_m\|_F$ 或 $\|\mathbf{U}_{\text{null}}^{\top}(\mathbf{BA})^{\top}\|_F$，量化正交性漂移。

### 中期方向

4. **分层异构策略**: 不同层使用不同初始化——embedding层用标准LoRA，浅层attention用activation-aware方法（LoRA-Null），深层MLP用weight-based方法（Least或MiLoRA）。

5. **初始化 + 训练约束联合方法**: LoRA-Null 初始化 + 轻量正交正则化（如 $\lambda \|\mathbf{U}_{\text{null}}^{\top}(\mathbf{BA})^{\top}\|_F^2$），结合两者优势。

6. **校准集无关的激活零空间**: 使用模型自身生成的数据（self-generated calibration）或合成数据构造激活零空间。

### 长期方向

7. **理论框架的建立**: Unified "三轴空间"理论——将奇异谱窗口、知识表征层次、约束机制统一到一个形式化框架中，预测给定条件下最优方法的选取准则。

8. **多适配器零空间冲突分析**: 在多 LoRA 适配器叠加场景中，分析不同任务的零空间是否冲突，设计冲突规避机制。

9. **跨模型族的系统性研究**: 分析 Mistral/Gemma/Qwen/DeepSeek 的奇异值分布差异，建立与初始化和识保留效果的量化关系。

---

## 8. 与本项目 LoRA-NSP 框架的结合讨论

> 本文档前面的内容来自文献调研。本节记录我们在项目中的讨论——如何将这些文献发现与我们的 LoRA-NSP 框架（$\Delta W = B \times A \times P$）进行结合。

### 8.1 我们的框架在三轴空间中的定位

我们的 LoRA-NSP 框架在文献的三轴设计空间中占据一个**独特且未被探索**的位置：

| 轴 | 我们的位置 | 与文献比较 |
|---|---|---|
| **X轴: 奇异谱窗口** | **软权重**（连续值，所有方向都有非零权重） | 文献全是硬选择（PiSSA选首、MiLoRA选末、Least选中），我们是唯一用连续权重的方法 |
| **Y轴: 知识表征层次** | **特征协方差空间**（$\Sigma = X^\top X/N$ 的特征分解） | 与 LoRA-Null（$X_{pre}$ 的 SVD，左奇异向量）同属激活层面，但具体分解对象不同；MiLoRA/Least 分解对象是 $\mathbf{W}_0$（权重空间） |
| **Z轴: 约束机制** | **运行时前向投影**（每步 forward 都乘 P） | 与 LoRA⁻ 同级（运行时约束），优于所有仅初始化引导的方法 |

**关键结论**：LoRA-NSP 是文献中唯一同时占据三个轴正面属性的方法。

### 8.2 与各方法的逐一对比

#### 8.2.1 vs PiSSA (Meng et al., 2024)

| 维度 | PiSSA | LoRA-NSP |
|---|---|---|
| 初始化 | $A=U\sqrt{D}, B=\sqrt{D}V^\top$（主分量） | $A\sim\mathcal{U}, B=0$（无结构） |
| 保护机制 | 冻结残差$\mathbf{W}^{\text{res}}$（保留次成分） | 运行时 soft projection P |
| 保护对象 | $\mathbf{W}_0$ 的次奇异方向 | 任务特征协方差的低能量方向 |
| 遗忘风险 | 高（直接修改主成分，遗忘损失6.07） | 低（P 持续约束） |

**连接点**：PiSSA 证明了 SVD 初始化对收敛的帮助。我们可以为 LoRA-NSP 的 A, B 引入类似的结构化初始化，而不改变运行时 P。

#### 8.2.2 vs MiLoRA (Wang et al., 2024)

| 维度 | MiLoRA | LoRA-NSP |
|---|---|---|
| 分解对象 | $\mathbf{W}_0$（预训练权重矩阵） | $\Sigma = X^\top X/N$（特征 Gram 矩阵） |
| 零空间定义 | 权重零空间：$\text{Row}(\mathbf{W}_m) = \mathcal{N}(\mathbf{W}_p)$ | soft projection：$\mathbf{P}_{\text{SGP}}$ 抑制大特征值方向 |
| 约束方式 | 初始化引导（不强制训练中保持） | 运行时前向投影（每步都约束） |
| 保护层面 | 权重方向（列空间+行空间均正交） | 输出响应（对预训练/任务激活的响应小） |
| 训练中漂移 | 未解决（论文自认 softer approach） | P 持续存在，漂移被即时修正 |

> ⚠️ **注意**：MiLoRA 的 SVD 对象是 $\mathbf{W}_0$（权重矩阵本身），其奇异值反映的是权重的"固有强度"。我们做的是 $\Sigma = X^\top X$ 的特征分解，其特征值反映的是**任务数据**的特征方差分布。两者是不同数学对象的谱分析，不能直接等价。

**连接点**：MiLoRA 的「权重零空间」和 LoRA-NSP 的「协方差 soft projection」保护的是不同的东西。自反驳轮次指出两者可互补。

#### 8.2.3 vs LoRA-Null (Tang et al., 2025) — 最直接的对比

| 维度 | LoRA-Null | LoRA-NSP |
|---|---|---|
| P 存在性 | 只在初始化时：$BA_{init} = W_0U_{null}U_{null}^\top$，运行时无 P | 运行时始终存在：$W = W_0 + BAP$ |
| P 构建来源 | 校准集激活矩阵的 SVD（一次性）：$X_{pre} = U\Sigma V^\top \to U_{null}$（左奇异向量） | 任务特征 Gram 矩阵的特征分解（每任务更新）：$\Sigma = \frac{1}{N}X^\top X \to V$（特征向量 = X 的右奇异向量） |
| P 构建方法 | 硬截断：取最小的 r 个左奇异向量 $U_{null}$ | 软权重：$w(\lambda)=1/(1+\beta\log(1+\lambda^p))$ 作用于特征值 |
| BA 初始化 | 结构化：$BA = W_0U_{null}U_{null}^\top$ | 无结构：$A\sim\mathcal{U}, B=0$ |
| 残差权重 | $W_0' = W_0 - BA$（被修改） | $W_0' = W_0$（不变） |
| 校准集依赖 | 是（NQ Open 等外部数据） | 否（使用任务数据自身的协方差） |
| 多任务适应 | 否（一次性初始化） | 是（P 可增量更新） |
| 知识保护类型 | 预训练知识（通过外部校准集） | 任务知识（通过特征协方差） |
| Avg1(Per) | 79.21%（最高） | 待实验 |
| 残差权重可靠性 | 最低（Figure 5: 去掉 BA 后知识最差） | 最高（$W_0'=W_0$ 天然保留完整知识） |

**核心区别**：
- **分解对象不同**：LoRA-Null 对 $X_{pre}$（激活矩阵）做 SVD 取左奇异向量；我们对 $\Sigma = X^\top X$（Gram 矩阵）做特征分解取特征向量。虽然都在激活层面，但具体操作不同。
- **P 的存在形式不同**：LoRA-Null 做**初始化吸收式**（P 算一次，吸收到 BA 初始化中，运行时无 P）；LoRA-NSP 做**运行时投影式**（P 每步前向存在，持续约束）
- **保护的知识类型不同**：LoRA-Null 保护**预训练知识**（校准集来自预训练数据）；LoRA-NSP 保护**任务知识**（协方差来自任务数据）

**两者本质上是互补的**：LoRA-Null 解决「微调时别忘掉 CLIP 的零样本能力」，LoRA-NSP 解决「持续学习时别忘掉之前学过的任务」。

#### 8.2.4 vs Least but not Last (Quercia et al., 2026) — 最重要的挑战

**核心发现**：Least 对 $\mathbf{W}_0$ 的奇异值谱做系统分析，发现 U 形遗忘曲线——微调极端奇异分量（首/末）比微调中间分量造成更大的实际遗忘，原因是极端分量导致更大的 off-diagonal rotation（主奇异方向被旋转）。

> $\mathbf{W}_0 = \underbrace{U_{:,:r}\sqrt{D_{:,:r}}}_{A} \underbrace{\sqrt{D_{:,:r}}V_{:r,:}^\top}_{B} + \underbrace{U_{:,r:}D_{r:,:}V_{:,r:}^\top}_{\text{冻结}}$（PiSSA），取 $s=0$（首）或 $s=m-r$（末）或 $0<s<m-r$（中）

**⚠️ 关键限定**：Least 的 U 形曲线是在 **$\mathbf{W}_0$（权重矩阵）的奇异值谱**上被实验验证的。我们的 LoRA-NSP 构建 P 的数学对象是 **$\Sigma_{\text{Cov}} = X^\top X/N$（特征 Gram 矩阵）的特征值谱**。两者是不同数学对象的谱分析，U 形曲线能否推广到协方差空间是一个**待实验验证的假说**，不能直接断定。

**Least 的 Figure 6-8 说明什么？** Least 分析了模型输出特征 $Y = XW$ 在原始 SVD 坐标系中的变化（特征空间分析），发现 U 形在特征空间更明显。但这个"特征空间"是指 Y 在 W₀ 的 SVD 基下的投影，不同于我们 $\Sigma_{\text{Cov}}$ 的特征空间。

**对 LoRA-NSP 的启发（作为假说）**：

当前 LoRA-NSP 的 P 是 low-pass 特性：
$$w(\lambda) = \frac{1}{1+\beta\log(1+\lambda^p)}$$
- 大 $\lambda$ → $w \approx 0$（抑制）
- 小 $\lambda$ → $w \approx 1$（允许）

这对应于**在协方差特征值谱中允许小特征值方向自由更新**。如果 U 形规律（在权重谱中成立）能类比到协方差谱，那么当前设计可能让适配器在极小特征值方向上过度自由，造成类似 off-diagonal 旋转的问题。

**但这只是一个启发性假说**——$\mathbf{W}_0$ 的奇异值和 $\Sigma_{\text{Cov}}$ 的特征值编码的是完全不同的信息（前者是预训练权重的变换强度，后者是任务数据的方差分布）。验证这个假说正是带通 P 实验（变体 A）要回答的问题。

#### 8.2.5 vs LoRA⁻ / LoRA Subtraction (Liu & Chang, 2025)

| 维度 | LoRA⁻ | LoRA-NSP |
|---|---|---|
| 约束方式 | 梯度投影：$\Delta w = P(P)^\top g$ | 前向投影：$W = W_0 + BAP$ |
| 投影空间定义 | 新任务在"遗忘模型"上的特征主成分 | 任务特征协方差的低能量方向 |
| 投影空间更新 | 每任务重新计算 | 每任务用 EMA 更新 |
| 与初始化正交性 | 是（训练约束与初始化无关） | 是（P 独立于 A, B 初始化） |

**连接点**：LoRA⁻ 与 LoRA-NSP 在 Z 轴上同级（都是运行时约束），且都与 X/Y 轴正交，可以与其他方法叠加。

### 8.3 从文献中提炼的变体方案

基于上述对比分析，我们从文献中提炼出以下可直接在 LoRA-NSP 代码中实验的变体：

#### 变体 A: 带通 P（Band-Pass P）⭐ 最推荐

**动机**：受 Least 的 U 形曲线启发（但需注意该曲线是在 $\mathbf{W}_0$ 的奇异谱上验证的，我们的 $\Sigma_{\text{Cov}}$ 特征谱不同），假设 U 形规律在协方差特征值谱中也成立。将 P 从 low-pass 改为 band-pass，在两端（极大和极小特征值）都施加抑制，只保留中间方向用于适应。

**数学**：
$$w_{\text{band}}(\lambda) = \underbrace{\frac{1}{1+\beta\ln(1+\lambda^p)}}_{\text{高通抑制（大}\lambda\text{）}} \cdot \underbrace{\frac{1}{1+\gamma\ln(1+\lambda^{-q})}}_{\text{低通抑制（小}\lambda\text{）}}$$

**改动量**：~10 行代码，在 `compute_weights` 中新增 `weight_kind="band_pass"`。

**理论价值**：完全原创，文献中无任何论文提出过。

#### 变体 B: Least 初始化 + LoRA-NSP 运行时 P

**动机**：文献指出「好的初始化 + 训练约束」是最有前景的方向。当前 LoRA-NSP 只做了后者。

**数学**：
1. Least 风格：$\mathbf{W}_0$ 的 SVD → 取中间分量 → 初始化 $A_{\text{init}}, B_{\text{init}}$
2. $W_0' = W_0 - U_{s:s+r}D_{s:s+r}V_{s:s+r}^\top$
3. 运行时：$W = W_0' + BAP_{\text{SGP}}$（保持现有 P）

**改动量**：修改 `SGPBaseLoRA.__init__`，新增 SVD 初始化逻辑。

#### 变体 C: 固定 B（你之前提的 idea）

**动机**：锚定适配器输出方向，减少一半可训练参数。

**数学**：
1. 从某种初始化策略（Least 中间分量 / LoRA-Null 激活零空间）计算 $B_{\text{init}}$
2. 冻结 $B = B_{\text{init}}$，只训练 $A$
3. 运行时：$W = W_0' + B_{\text{fixed}}AP$（或 $W = W_0' + B_{\text{fixed}}A$ 无 P 版）

**改动量**：新建 `FixedBLoRA` 类或加 `freeze_B` 参数。

#### 变体 D: 双投影 P = P_pre × P_task

**动机**：同时保护预训练知识和任务知识。

**数学**：
$$P_{\text{pre}} = U_{\text{null}}U_{\text{null}}^\top \quad (\text{从 Flickr8K 校准集，固定})$$
$$P_{\text{task}} = \sum_i w(\lambda_i) u_i u_i^\top \quad (\text{从任务协方差，每任务更新})$$
$$P_{\text{combined}} = P_{\text{task}} \cdot P_{\text{pre}}$$

**改动量**：修改 `build_projection`，新增校准集 SVD 路径。

### 8.4 三轴空间的交叉实验矩阵

以下实验矩阵可以一次性验证多个假设：

| 实验 | X轴（窗口） | Y轴（分解对象） | Z轴（约束） | 预期回答的问题 |
|---|---|---|---|---|
| ① 当前 LoRA-NSP | 软权重 low-pass | $\Sigma_{\text{Cov}} = X^\top X$（特征 Gram 矩阵） | 运行时 P | 基线 |
| ② 带通 P | 软权重 band-pass | $\Sigma_{\text{Cov}} = X^\top X$（特征 Gram 矩阵） | 运行时 P | U 形假说（从 $\mathbf{W}_0$ 奇异谱推广到 $\Sigma_{\text{Cov}}$ 特征谱）是否成立？ |
| ③ Least 初始化 + 运行时 P | $\mathbf{W}_0$ SVD 中间分量 | $\mathbf{W}_0$（权重 SVD）+ $\Sigma_{\text{Cov}}$（协方差特征分解） | 初始化+运行时 | 双重保护是否优于单一？ |
| ④ 固定 B（Least 初始化） | $\mathbf{W}_0$ SVD 中间分量 | $\mathbf{W}_0$（权重 SVD） | 初始化 | $\mathbf{W}_0$ 初始化引导是否足够？ |
| ⑤ 激活零空间初始化（无 P） | $X_{pre}$ SVD 最小 $r$ 个左奇异向量 | $X_{pre}$（激活 SVD，左奇异向量） | 初始化 | LoRA-Null 在我们的设定下表现？ |
| ⑥ 带通 P + Least 初始化 | 中间+band-pass | $\mathbf{W}_0$（权重 SVD）+ $\Sigma_{\text{Cov}}$（协方差特征分解） | 初始化+运行时 | 最强组合？ |

**实验 ② 和 ⑥ 是最有理论贡献的两个**，因为它们直接回应了 Least 的 U 形挑战，且文献中无人做过。

### 8.5 讨论结论

1. **我们的框架定位清晰**：LoRA-NSP 在文献全景中占据一个独特且未被探索的位置——唯一同时使用激活空间、运行时约束、软权重的方法。

2. **U 形假说值得实验验证**：Least 在 $\mathbf{W}_0$ 奇异谱上发现的 U 形规律能否推广到我们的 $\Sigma_{\text{Cov}}$ 特征谱，是一个开放问题。带通 P 实验正是为了回答这个假说。

3. **变体 A（带通 P）是最推荐优先实验的方向**：改动极小（~10 行），理论贡献清晰，且与现有代码完全兼容。

4. **变体 C（固定 B）是你之前提的方向**：可以在带通 P 的基础上叠加，形成参数高效 + 理论扎实的组合方案。
