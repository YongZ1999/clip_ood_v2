# MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning 内容总结
- **作者**: Hanqing Wang, Yixia Li, Shuo Wang, Guanhua Chen, Yun Chen | **年份**: 2024 | **会议/期刊**: NAACL 2024

## 问题定义
- **论文试图解决什么问题？**
  现有 LoRA 及其变体方法使用高斯分布初始化低秩矩阵 A、零初始化 B，在**无引导的子空间**中优化可训练参数。这种"无引导"的初始化策略可能干扰预训练权重矩阵中已充分学习的子空间（即覆盖重要的预训练特征），从而降低低秩适配方法的性能。

- **为什么这个问题重要？**
  PEFT 的核心目标是在大幅降低计算和内存开销的同时尽量保留预训练模型的能力。如果适配器的随机初始化方向与预训练知识发生冲突——即训练过程中不经意地"覆盖"（override）了重要的预训练特征——则微调效果会受到损害。改进初始化策略可以在不增加训练/推理成本的前提下直接提升 PEFT 的下游任务性能。

- **在此之前最好的方法是什么？它有什么局限？**
  1. **LoRA**：用 $\mathbf{A} \sim \mathcal{N}(0,\sigma^2)$, $\mathbf{B} = \mathbf{0}$ 初始化，使初始 $\Delta\mathbf{W}=0$。局限：随机初始化导致优化子空间无引导，可能干扰预训练知识。
  2. **PiSSA**：用**主奇异值/向量**初始化适配器（即训练主成分、冻结残差）。局限：PiSSA 直接修改权重的"骨架"（主成分），虽然能更好地近似全参数微调，但更容易改变/覆盖预训练知识中的核心部分，导致更高的**遗忘损失**（forgetting loss）。

## 核心方法
- **核心思路（一句话）**
  对预训练权重矩阵 $\mathbf{W}$ 做 SVD 分解，将"主成分"（大奇异值部分）冻结以保留预训练知识，只用"次成分"（小奇异值部分）初始化 LoRA 适配器进行微调——因为次成分包含的是噪声或长尾信息，更适合为新任务调整而不会破坏已有知识。

- **核心公式**
  $$ \mathbf{W} = \underbrace{\sum_{i=1}^{m-r} \sigma_i \mathbf{u}_i \mathbf{v}_i^{\top}}_{\text{冻结的主成分 } \mathbf{W}_p} + \underbrace{\sum_{i=m-r+1}^{m} \sigma_i \mathbf{u}_i \mathbf{v}_i^{\top}}_{\text{可训练的次成分 } \mathbf{W}_m} \tag{3} $$
  
  $$ \mathbf{W}_m = \underbrace{\mathbf{U}_m}_{\text{次奇异向量}} \underbrace{\mathbf{\Sigma}_m}_{\text{次奇异值}} \underbrace{\mathbf{V}_m^{\top}}_{\text{次奇异向量}} = \underbrace{(\mathbf{U}_m \sqrt{\mathbf{\Sigma}_m})}_{\mathbf{B}_m} \underbrace{(\sqrt{\mathbf{\Sigma}_m} \mathbf{V}_m^{\top})}_{\mathbf{A}_m} \tag{5} $$
  
  $$ \text{前向: } \mathbf{Y} = \mathbf{X}(\underbrace{\mathbf{W}_p}_{\text{冻结主成分}} + \underbrace{\mathbf{B}_m \mathbf{A}_m}_{\text{可训练次成分}}) $$

- **公式推导路径**
  1. 对每个线性权重矩阵 $\mathbf{W}$ 做 SVD：$\mathbf{W} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top}$
  2. 按奇异值大小将 $\mathbf{W}$ 拆分为两部分：主成分 $\mathbf{W}_p$（最大的 $m-r$ 个奇异值）和次成分 $\mathbf{W}_m$（最小的 $r$ 个奇异值）
  3. 将 $\mathbf{W}_p$ 冻结不变，保留预训练知识
  4. 将 $\mathbf{W}_m$ 分解为 $\mathbf{B}_m \mathbf{A}_m$ 作为 LoRA 适配器的初始化（与 PiSSA 一样用平方根分配）
  5. 微调时只更新 $\mathbf{A}_m$ 和 $\mathbf{B}_m$，初始时 $\mathbf{W}_p + \mathbf{B}_m\mathbf{A}_m = \mathbf{W}$，不改变模型输出

## 实验设计
- **使用的数据集/benchmark 和 baseline 比较**
  - **常识推理（Commonsense Reasoning）**：在 Commonsense170K 上微调 LLaMA2-7B / LLaMA3-8B，使用 8 个数据集评估（BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA）
  - **数学推理（Math Reasoning）**：在 MetaMathQA 395K 上微调 LLaMA2-7B，评估 GSM8K 和 MATH
  - **指令遵循（Instruction Following）**：在 Ultrafeedback 上微调 LLaMA2-7B，评估 AlpacaEval 2.0、FollowBench、IFEval
  - **视觉指令遵循（Visual Instruction Following）**：在 LLaVA1.5-7B 上微调，评估 VQAv2, GQA, VizWiz, SQA, VQAT, POPE, MMBench
  - **Baselines**: LoRA, PiSSA
  - **额外对比**: rsLoRA, LoRA+, DoRA, AdaLoRA, LoRA-GA

- **核心结果（具体数字）**
  - **常识推理 (LLaMA2-7B)**: MiLoRA **79.2%** avg vs LoRA 77.6% (+1.6) vs PiSSA 73.8% (+5.4)
  - **常识推理 (LLaMA3-8B)**: MiLoRA **81.9%** avg vs LoRA 80.8% (+1.1) vs PiSSA 75.4% (+6.5)
  - **数学推理 (LLaMA2-7B)**: MiLoRA **40.7%** avg vs LoRA 38.7% (+2.0) vs PiSSA 37.0% (+3.7) —— 但仍低于 Full FT 的 43.2%
  - **指令遵循 (LLaMA2-7B)**: MiLoRA avg **31.0%** vs LoRA 28.1% (+2.9) vs PiSSA 28.3% (+2.7)
  - **视觉指令遵循 (LLaVA1.5-7B)**: MiLoRA avg **68.3%** vs LoRA 66.9% (+1.4) vs PiSSA 63.9% (+4.4)
  - **额外对比（数学+代码）**: MiLoRA GSM8K **54.7%** / Human-eval **24.4%**，超过 LoRA-GA (53.6%/19.8%)
  - **遗忘损失**: MiLoRA **2.54** vs LoRA 3.24 vs PiSSA 6.07（越低越好，表明预训练知识保留更完整）

## 作者自己说的局限
1. **模型覆盖范围有限**：受计算资源限制，主要评估了 LLaMA 系列模型和 LLaVA-1.5，未在 Mistral、Gemma 等其他主流 LLM 上验证。
2. **任务覆盖范围有限**：主要关注常识推理、数学推理、指令遵循，未探索更多类型的任务。
3. **超参数搜索不足**：使用了前人工作中的默认超参数配置（LLM-Adapters / LLaVA），而不是为每个任务做充分的超参数搜索。
4. **未来工作**：将 MiLoRA 扩展到其他任务和其他 LLM（如 Mistral, Gemma）作为未来方向。
