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
