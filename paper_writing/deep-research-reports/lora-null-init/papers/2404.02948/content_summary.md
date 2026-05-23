# PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models 内容总结
- **作者**: Fanxu Meng, Zhaohui Wang, Muhan Zhang | **年份**: 2024 | **会议/期刊**: NeurIPS 2024

## 问题定义
- **论文试图解决什么问题？**
  LoRA 在初始时使用高斯噪声初始化 A 矩阵、零初始化 B 矩阵，导致微调初始阶段梯度很小且方向随机，收敛速度慢，且可能收敛到次优局部极小点。

- **为什么这个问题重要？**
  大语言模型（LLM）的全参数微调需要巨大的 GPU 显存（如 LLaMA 65B 需要超过 780GB）。LoRA 作为最流行的 PEFT 方法通过低秩分解大幅降低了可训练参数量，但其初始化方式导致的收敛缓慢和性能损失会直接影响微调效率和下游任务质量。改进 LoRA 的初始化可以在不增加训练成本的前提下直接提升模型微调效果。

- **在此之前最好的方法是什么？它有什么局限？**
  LoRA 用 AW = AB 近似参数更新，其中 A 用高斯噪声初始化、B 用零初始化，使得初始时 AB = 0 不改变模型输出。这一"噪声+零"初始化的缺陷是：
  1. 梯度 $\frac{\partial \mathcal{L}}{\partial A} = X^T \frac{\partial \mathcal{L}}{\partial Y} B^T$ — 由于 $B=0$，A 的梯度初始为 0，A 在前几步几乎不更新；
  2. 梯度 $\frac{\partial \mathcal{L}}{\partial B} = A^T X^T \frac{\partial \mathcal{L}}{\partial Y}$ — B 虽不为 0 但方向随机，不同随机种子下梯度方向差异大；
  3. 导致大量训练步浪费在从初始点"热身"的过程，收敛缓慢。

## 核心方法
- **核心思路（一句话）**
  对预训练权重矩阵 $\mathbf{W}$ 做 SVD 分解，用**主奇异值/向量**初始化 LoRA 适配器（$\mathbf{A}, \mathbf{B}$），将**残差奇异值/向量**冻结，从而让微调直接从权重的最重要方向开始。

- **核心公式**
  $$ \mathbf{W} = \mathbf{USV}^T \quad\text{(经济 SVD)} $$
  
  $$ \underbrace{\mathbf{A}}_{\text{可训练适配器}} = \underbrace{\mathbf{U}_{:,:r}}_{\text{左主奇异向量}} \underbrace{\mathbf{S}_{:r,:r}^{1/2}}_{\text{主奇异值的平方根}} \in \mathbb{R}^{m \times r} $$
  
  $$ \underbrace{\mathbf{B}}_{\text{可训练适配器}} = \underbrace{\mathbf{S}_{:r,:r}^{1/2}}_{\text{主奇异值的平方根}} \underbrace{\mathbf{V}_{:r,:}^T}_{\text{右主奇异向量}} \in \mathbb{R}^{r \times n} $$
  
  $$ \underbrace{\mathbf{W}^{\text{res}}}_{\text{冻结的残差矩阵}} = \underbrace{\mathbf{U}_{:,r:}}_{\text{残差左奇异向量}} \underbrace{\mathbf{S}_{r:,:}}_{\text{残差奇异值}} \underbrace{\mathbf{V}_{:,r:}^T}_{\text{残差右奇异向量}} \in \mathbb{R}^{m \times n} $$
  
  $$ \underbrace{\mathbf{Y}}_{\text{前向输出}} = \mathbf{X} \underbrace{\mathbf{W}}_{\text{完整权重}} = \mathbf{X}(\underbrace{\mathbf{W}^{\text{res}}}_{\text{冻结残差}} + \underbrace{\mathbf{A}\mathbf{B}}_{\text{可训练主成分}}) $$

- **公式推导路径**
  LoRA 的思路是通过低秩矩阵 $\mathbf{A}\mathbf{B}$ 近似权重更新 $\Delta \mathbf{W}$，而 PiSSA 的思路变为直接近似原始权重 $\mathbf{W}$ 本身。具体来说：
  1. 对 $\mathbf{W}$ 做 SVD 得到 $\mathbf{U}, \mathbf{S}, \mathbf{V}^T$，其中 $\mathbf{S}$ 的奇异值从大到小排列；
  2. 将 $\mathbf{S}$ 分成两组：前 $r$ 个大奇异值 $\mathbf{S}_{:r,:r}$（"主成分"）和剩余的较小奇异值 $\mathbf{S}_{r:,:}$（"残差"）；
  3. 主成分部分用 $\mathbf{A}\mathbf{B} = \mathbf{U}_{:,:r} \mathbf{S}_{:r,:r}^{1/2} \cdot \mathbf{S}_{:r,:r}^{1/2} \mathbf{V}_{:r,:}^T = \mathbf{U}_{:,:r} \mathbf{S}_{:r,:r} \mathbf{V}_{:r,:}^T$ 初始化并保持可训练；
  4. 残差部分 $\mathbf{W}^{\text{res}}$ 冻结不训练；
  5. 前向时为 $\mathbf{Y} = \mathbf{X}(\mathbf{W}^{\text{res}} + \mathbf{A}\mathbf{B})$，与原始 LoRA 形式 $\mathbf{Y} = \mathbf{X}(\mathbf{W} + \mathbf{A}\mathbf{B})$ 结构相同，因此完全兼容 LoRA 的部署方式。

## 实验设计
- **使用的数据集/benchmark 和 baseline 比较**
  - **NLG 任务**: 在 MetaMathQA/CodeFeedback/WizardLM-Evol-Instruct 数据集上微调，评估 GSM8K/MATH（数学）、HumanEval/MBPP（代码）、MT-Bench（对话）
  - **NLU 任务**: GLUE benchmark 的 8 个子任务（MNLI, SST-2, MRPC, CoLA, QNLI, QQP, RTE, STS-B）
  - **模型规模**: 184M ~ 70B，包括 LLaMA 2-7/13B, LLaMA-3-8/70B, Mistral-7B, Gemma-7B, Qwen1.5-7B, Yi-1.5-34B, DeepSeek-MoE-16B, Mixtral-8x7B（共 11 种模型）
  - **Baselines**: LoRA（高斯/Kaiming 初始化）, QLoRA, LoftQ, DoRA, AdaLoRA, 全参数微调

- **核心结果（具体数字）**
  - **GSM8K**: Gemma-7B + PiSSA = **77.78%** vs LoRA = 74.53%，提升 3.25%
  - **LLaMA-3-70B + QPiSSA**: **86.05%** vs QLoRA = 81.73%，提升约 4.3%
  - **NLU (GLUE 平均)**: PiSSA = **89.83%** vs LoRA(Gaussian) = 88.50%, LoRA(Kaiming) = 88.62%, AdaLoRA = 89.46%
  - **量化误差降低**: PiSSA r128 对 LLaMA-3-8B 各类线性层平均降低 **22.9%** 的量化误差，优于 LoftQ 的 18.1%
  - **收敛速度**: 初始 100 步内 PiSSA 损失快速下降，梯度范数显著高于 LoRA 且趋势接近全参数微调
  - **跨模型一致优势**: (Q)PiSSA 在 9 种不同大小和类型的模型上均优于 (Q)LoRA
  - **跨 rank 表现**: 从 rank=1 到 rank=128，PiSSA 始终优于 LoRA，且 rank 越低优势越显著

## 作者自己说的局限
1. **视觉任务泛化**: 除语言模型外，PiSSA 是否可扩展到卷积层并提升视觉任务性能？未验证。
2. **与动态 rank 方法的兼容性**: PiSSA 是否能受益于 AdaLoRA/DyLoRA 等自适应调整 rank 的改进方法？未充分验证。
3. **理论解释不足**: 对于 PiSSA 相比 LoRA 的优势，缺乏更深入的理论证明（论文主要依赖实验证据）。
4. **过参数化现象**: 在 Gemma-7B 上 rank 增加到 128 时 PiSSA 性能开始下降（比 LoRA 更早出现过参数化）。
5. **精度敏感**: FP32 和 BF16 精度对实验结果影响显著且不一致（如 LLaMA-2-7B 上 FP32 比 BF16 高 5.16%，但 Mistral-7B 上反而低 7.21%），论文未明确推荐精度选择。
