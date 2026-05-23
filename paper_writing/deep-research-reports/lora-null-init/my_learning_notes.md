# 我的学习笔记

## Paper 1: PiSSA (2404.02948)
- **读完这篇我的理解**: PiSSA 的核心洞察是：不是用"噪声+零"去近似 ΔW，而是用 SVD 把预训练权重 W 拆成"主成分（可训练适配器）+ 残差（冻结）"。这样初始输出不变（因为 W = W^res + AB），但梯度从第一步就有信息量（因为 A 和 B 初始不为零且包含 W 的主方向信息）。这是对 LoRA 初始化范式的根本转换：从"学什么"到"从哪开始学"。
- **与之前的连接**: 这是该系列的第一篇，没有之前的笔记。但需要注意它的两个关键隐含假设：(1) 主方向 = 好方向假设 — 预训练权重的主成分方向是否真的也是微调最重要的方向？(2) 截断不丢失关键信息假设 — 冻结的残差是否真的对微调无关紧要？
- **我的框架更新**: 初步认识到 SVD-based LoRA 初始化可以有两条正交路线：一条是 PiSSA（保留主成分训练，冻结残差），另一条应该是保留残差训练，冻结主成分（即 MiLoRA 的思路）。这两条路线的哲学本质差异是"粗调结构" vs "细调细节"。
- **不确定的地方**: 1) PiSSA 的主成分方向假设在领域迁移较大时是否仍然成立？2) r 的截断点在多大空间上影响性能？仅凭奇异值大小截断是否最优？


## Paper 2: MiLoRA (2406.09044)
- **读完这篇我的理解**: MiLoRA 和 PiSSA 共享完全相同的数学形式（W = W_frozen + BA），唯一的区别是**选择哪个奇异值范围来初始化适配器**。PiSSA 选前 r 个（主成分），MiLoRA 选后 r 个（次成分）。这一"一正一反"的选择带来了截然不同的知识保留特性：MiLoRA 的遗忘损失 2.54 远低于 PiSSA 的 6.07。
- **与 PiSSA 的连接**: 两条路线构成一个完全对称的设计空间。PiSSA ≈ "从骨架开始学"（快速但忘得多），MiLoRA ≈ "从噪声开始调"（稳但可能偏离全参数微调路径）。关键发现：Table 9 显示在不同超参数设置下两者的优势可能逆转，说明不存在"绝对更好"的初始化方向。
- **我的框架更新**: MiLoRA 的本质是**将 LoRA 的搜索空间限制在 W_p 的 null space 中**（因为 Row(W_m) = N(W_p)）。这让我意识到"LoRA 初始化空间"这个概念的关键性——LoRA-Null 论文说"the space of LoRA initialization is the key"，正是基于 MiLoRA 的这一思想。但 MiLoRA 用的是权重的 null space，LoRA-Null 会用 activation 的 null space——这两个 space 的差异是 LoRA-Null 的核心 claim。
- **不确定的地方**: 1) 训练中正交性漂移到什么程度？MiLoRA 承认不强制约束，但没量化漂移。2) 如果 Full FT 的 ΔW 主要落在主成分子空间中，MiLoRA 的搜索空间限制是否会限制其表达力？3) MiLoRA 在 LLaMA 上验证较好，但在其他模型家族（如 Mistral, Gemma）上的表现未知。


## Paper 3: LoRA-Null (2503.02659) — 目标论文
- **读完这篇我的理解**: LoRA-Null 完成了从"权重零空间"到"激活零空间"的关键跨越。它的核心洞察有两点：(1) 残差权重接近 W₀ **不是关键**——关键是 LoRA 初始化空间与预训练知识正交；(2) 权重的 null space 不够"纯粹"（W₀ 有效秩 ~500-2900，远大于 X_pre 的有效秩 ~75-760），而激活的 null space 包含了更少的预训练知识信息。这使 LoRA-Null 在知识保留上显著优于 MiLoRA（Avg1 提升约 10 个百分点）。
- **与 MiLoRA/PiSSA 的连接**: 
  - 从 PiSSA（训练主成分）→ MiLoRA（冻结主成分/训练次成分→权重 null space）→ LoRA-Null（激活 null space），这是一条"零空间纯度"不断增加的路线。
  - 关键矛盾转化：PiSSA vs MiLoRA 的矛盾是"训练哪个奇异部分"，而 LoRA-Null 引入了新矛盾——**权重的奇异方向 vs 激活的奇异方向是不同空间**。
- **我的框架更新**: 之前的框架只考虑了"SVD 拆分 W₀ 的哪个部分"。LoRA-Null 揭示了一个更深的维度：**零空间应该在激活空间（X_pre）还是在权重空间（W₀）中寻找？** 激活的零空间才是真正与预训练知识无关的方向。这个框架更新让我对 Figure 4 印象深刻——LoRA-Null 的 A 矩阵几乎全部落在 X_pre 零空间中，而 MiLoRA 和 CorDA 仍有部分落在主成分方向。
- **不确定的地方**: 1) 校准集 NQ Open 能否代表全部预训练知识？2) 如果去除 LoRA 适配器只保留残差权重，LoRA-Null 的知识保留最差——这是否意味着它对适配器有过度依赖？3) 为什么在 LLaMA-3.1-8B 上略逊于 MiLoRA？


## Paper 4: Least but not Last (2602.03493)
- **读完这篇我的理解**: 这篇论文用一个统一的广义框架（A=U_{s:s+r}√D, B=√D V_{s:s+r}^T）将 PiSSA（s=0）和 MiLoRA（s=m-r）统一为两个端点，然后证明了中间分量（0<s<m-r）在 performance-forgetting trade-off 上优于两端。核心发现是**U形遗忘曲线**：两端成分导致更大的奇异值方向旋转（off-diagonal changes），损伤了主成分方向；而中间成分在达到相同下游任务性能的同时引起的旋转最小。
- **与前三篇的连接**: 
  - 它揭示了 PiSSA vs MiLoRA 的二元选择是**过时**的——最优解不在两端。
  - **比 LoRA-Null 更深层的批判**: LoRA-Null 把"极端化零空间"作为目标，但这篇论文表明"中间"才是最优。如果 activation null space（比 weight null space 更纯）本质上也是"末分量"的广义形式，它可能同样面临 U 形遗忘的问题。
  - 不过，LoRA-Null 的 null space 是 activation 层面的而非 weight 层面的，所以不直接是"Least but not Last"框架的 s 参数变体——LoRA-Null 的 BA = W₀U_nullU_null^T 不能写成 U_{s:s+r}√D √D V_{s:s+r}^T 的形式，因为它经过了 W₀ 的投影。**这是两条不同的维度**: (a) 在 W₀ 的哪个奇异值范围初始化 vs (b) 在 W₀ 还是 X_pre 的 null space 中初始化。
- **我的框架更新**: 我现在需要从四个竞争方法中提取更本质的维度：
  1. **SVD 拆分维度**（PiSSA/MiLoRA/Least but not Last）: W₀ 的哪个奇异值范围分配给 BA？
  2. **零空间定义维度**（MiLoRA vs LoRA-Null）: 零空间在权重空间还是激活空间中寻找？
  3. **中间 vs 极端**（Least but not Last 的批判）: 无论是哪种空间，"极端"（最首/最末）是否总是次优？
- **不确定的地方**: Least but not Last (2026年) 是最新的论文，没有对比 LoRA-Null (2025) —— 两者是否有可比性？LoRA-Null 的 activation null space 是否仍然面临 U 形遗忘问题？


## Paper 5: LoRA Subtraction / LoRA- (2503.18985)
- **读完这篇我的理解**: LoRA- 从完全不同的视角切入——持续学习中的特征漂移。它的核心机制是两步：先"减去旧任务的 LoRA 权重"构建遗忘旧任务后的模型（消除旧知识影响），再在这个"遗忘模型"上提取新任务的特征主成分（DRS），然后训练新任务时将梯度投影到 DRS 中。关键洞察是：**通过消除旧任务的影响来保护旧任务的知识**（"负负得正"）。
- **与主课题的连接**: 
  - LoRA-Null 关注**初始化空间**，LoRA- 关注**训练空间**（梯度投影）。两者可以互补：先 LoRA-Null 初始化防遗忘，再加 DRS-style 梯度投影训练防漂移。
  - LoRA- 不依赖 SVD 初始化，而是用标准 LoRA 初始化 + 训练约束。这揭示了一个我之前忽略的维度：**初始化策略 vs 训练约束策略**是正交的维度。
- **我的框架更新**: 我现在有了一个四维框架来理解"保留预训练知识的 LoRA 微调"：
  1. **SVD 拆分策略**（PiSSA→首分量，MiLoRA→末分量，Least→中间，LoRA-Null→W₀在X_pre零空间的投影）
  2. **零空间定义层次**（weight level vs activation level）
  3. **极端 vs 中间**（Least but not Last 的批判）
  4. **初始化约束 vs 训练约束**（LoRA-Null vs LoRA-）
- **不确定的地方**: LoRA- 的 DRS 是为持续学习设计的（多任务），不太好直接移植到单次微调中。LoRA-Null + LoRA- 的组合是否真的可行？
