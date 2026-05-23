# Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge 内容总结
- **作者**: Pengwei Tang, Xiaolin Hu, Yong Liu, Lizhong Ding, Dongjie Zhang, Xing Wu, Debing Zhang  |  **年份**: 2025 (AAAI 2026 proceedings)  |  **会议/期刊**: AAAI 2025 (published 2026)

## 问题定义
- **论文试图解决什么问题？**  
  LoRA 微调大语言模型时存在的灾难性遗忘（catastrophic forgetting）问题——模型在适应下游任务时丢失预训练阶段获得的世界知识。

- **为什么这个问题重要？**  
  LoRA 是当前最主流的参数高效微调（PEFT）方法，被广泛用于 LLM 的 downstream adaptation。然而即便 LoRA 相比全参数微调遗忘更少，它仍然会出现显著的灾难性遗忘。如何在保持下游任务性能的同时最大程度保留预训练知识，是 LoRA 实际部署中的核心挑战。

- **在此之前最好的方法是什么？它有什么局限？**  
  此前有两类专门设计的 LoRA 初始化方法用于缓解遗忘：
  1. **MiLoRA** (Wang et al., 2025a)：在预训练权重 **W₀** 的零空间（null space）中初始化 LoRA，即只更新 W₀ 的次要奇异分量，冻结主奇异分量。局限：只考虑了当前层的权重信息，忽略了来自前面所有层的参数和输入数据的信息。
  2. **CorDA** (Yang et al., 2024)：通过对 **W₀C**（C = XₚᵣₑXₚᵣₑᵀ，输入激活协方差矩阵）做 SVD，提取与预训练知识最相关的方向，但它的 LoRA 初始化空间既不是 W₀ 的零空间，也不是 Xₚᵣₑ 的零空间。

  两类方法的共同局限：它们同时追求两个目标——(1) 让残差权重 **W₀'** 接近原始预训练权重 **W₀**，以及 (2) 让 LoRA 初始化空间正交于预训练知识。论文发现目标 (2) 才是关键，目标 (1) 并非必要。

## 核心方法
- **核心思路（一句话，用直觉说清楚）**  
  与其在预训练权重的零空间中初始化 LoRA（MiLoRA 的做法），不如在**预训练知识激活值（input activations）**的零空间中初始化 LoRA——因为激活值包含了前面所有层的参数信息和输入数据信息，它的零空间比权重的零空间更"干净"（包含更少的预训练知识信息）。

- **核心公式（LaTeX，每项用 \underbrace 标注含义）**

  **Step 1: 获取激活零空间**  
  \[
  X_{\text{pre}} = U \Sigma V^{\top} = \sum_{w=1}^{R} \sigma_w \mathbf{u}_w \mathbf{v}_w^{\top} \tag{6}
  \]
  \[
  \underbrace{U_{\text{null}}}_{\text{左零空间近似}} = U_{[:, R-r+1:R]} \quad \text{(最小的 } r \text{ 个左奇异向量)}
  \]

  **Step 2: 将 W₀ 投影到激活零空间**  
  \[
  \underbrace{BA}_{\text{LoRA 初始化}} = \underbrace{W_0}_{\text{预训练权重}} \underbrace{U_{\text{null}} U_{\text{null}}^{\top}}_{\text{投影到 } X_{\text{pre}} \text{ 的零空间}} \tag{7-8}
  \]

  **Step 3: SVD 分解投影矩阵以得到 A 和 B**  
  \[
  W_0 U_{\text{null}} U_{\text{null}}^{\top} = \underbrace{U'}_{\text{左奇异矩阵}} \underbrace{D'}_{\text{奇异值对角阵}} \underbrace{(V')^{\top}}_{\text{右奇异矩阵}} \tag{9}
  \]
  \[
  B = \underbrace{U'_{[:, :r]}}_{\text{top-} r \text{ 左奇异向量}} \underbrace{\sqrt{D'_{[:r, :r]}}}_{\text{放缩}} , \quad
  A = \underbrace{\sqrt{D'_{[:r, :r]}}}_{\text{放缩}} \underbrace{(V'_{[:, :r]})^{\top}}_{\text{top-} r \text{ 右奇异向量}} \tag{10}
  \]

  **Step 4: 残差权重**  
  \[
  W_0' = \underbrace{W_0}_{\text{原始权重}} - \underbrace{BA}_{\text{LoRA 初始化}} \tag{11}
  \]

- **公式推导路径（从问题定义到核心公式的逻辑链）**

  1. **问题定义**：LoRA 微调时，权重变化 ΔW = BA，初始时 BA = 0（标准 LoRA）。为了保留预训练知识，需要让 LoRA 适配器的更新不破坏预训练知识。
  2. **前人思路**：MiLoRA 让 BA 在 W₀ 的零空间中初始化（BA ⊥ W₀），即 BA 更新的是 W₀ 的次要奇异方向。CorDA 则通过 W₀C 的 SVD 提取任务相关方向。
  3. **关键洞察**：论文发现与"让残差权重 W₀' 接近 W₀"相比，"让 LoRA 初始化空间正交于预训练知识"才是保留知识的关键（Figure 1 显示 LoRA 适配器微调后变化很小，说明初始化空间比残差权重更重要）。
  4. **从权重到激活**：神经网络的输出由 φ(Wx) 决定，所以零空间的选择可以考虑 W₀ 或 Xₚᵣₑ。论文论证 Xₚᵣₑ 包含了前面所有层的信息，且有效秩远小于 W₀，因此 Xₚᵣₑ 的零空间包含更少的预训练知识。
  5. **具体操作**：对 Xₚᵣₑ 做 SVD，取最小的 r 个左奇异向量作为 U_null，然后将 W₀ 投影到 U_null 上得到 BA，最后做第二次 SVD 以得到 A、B 的标准形式。

## 实验设计
- **用的什么数据集/benchmark？和哪些 baseline 比较？**

  **预训练知识评估**：TriviaQA、NQ Open、WebQS（Exact Match 分数）
  **下游任务**：
  - Math：在 MetaMathQA 上训练，在 GSM8k 和 Math 验证集上测试
  - Code：在 CodeFeedback 上训练，在 HumanEval 和 MBPP 上测试
  - Instruction Following：在 WizardLM-Evol-Instruct 上训练，在 MT-Bench 上测试

  **Baselines**：LoRA（标准）、PiSSA（主奇异分量初始化）、MiLoRA（权重次要分量初始化）、CorDA（上下文感知分解）

  **模型**：LLaMA-2-7B、LLaMA-3.2-3B、Gemma-3-1B、LLaMA-3-8B、LLaMA-3.1-8B

  **配置**：LoRA rank=128，优化器 AdamW，学习率 2e-5，batch size 128，cosine annealing + warmup 0.03，训练 1 epoch（前 100k 条数据），校准集为 256 条 NQ Open 数据（max length=1024）。

- **核心结果是什么？（用具体数字）**

  1. **知识保留（Avg1）**：LoRA-Null 在所有设置下均取得最高的 Avg1（知识保留平均分）和 Avg1(Per)（相对于预训练模型保留百分比）。在 LLaMA-2-7B Math 上，LoRA-Null Avg1 = 21.52（保留 79.21%），CorDA = 20.63（76.24%），MiLoRA = 18.93（69.66%）。平均比 CorDA 高 3.35% 的保留百分比。

  2. **下游任务（Avg2）**：LoRA-Null 在数学任务上取得最高结果（LLaMA-2-7B Math 上 Avg2=26.62），总体仅次于 PiSSA。

  3. **综合指标（GM = geometric mean of Avg1 & Avg2）**：LoRA-Null 取得最高 GM 值（LLaMA-2-7B Math 上 GM=23.93，CorDA=22.64，MiLoRA=21.24，PiSSA=22.79）。

  4. **校准集规模鲁棒性（Table 4a/Table 7）**：校准集从 64 到 1024 变化时，LoRA-Null 的 Avg1(Per) 保持在 90.74%~94.95%，而 CorDA 从 68.97% 到 91.39%，波动更大。

  5. **Rank 鲁棒性（Table 4b）**：rank 从 64 到 256 变化时，CorDA 的 Avg1(Per) 从 91.93% 下降到 73.01%，而 LoRA-Null 仅从 91.79% 下降到 84.71%。

  6. **Gemma-3-1B（Table 5）**：LoRA-Null（无缩放）的 Avg1(Per)=80.78%，远高于 CorDA 的 61.38% 和 MiLoRA 的 60.76%。

## 作者自己说的局限
1. **论文结论部分未明确列出局限**，结论（Conclusion）仅总结了贡献，未专门列出不足。
2. **从 Appendix 的 Theorem 3 证明中可推断的局限**：
   - 论文承认关于 "使用 W₀Xₚᵣₑ 做 SVD 是否优于使用 W₀XₚᵣₑXₚᵣₑᵀ" 的问题"仍是一个开放问题，值得进一步研究"——即 CorDA 的做法（左乘协方差矩阵）是否有理论优势尚未清楚。
   - LoRA-Null 的残差权重 W₀'（去掉 BA 后）在知识保留上表现较差（Figure 5 显示去除 LoRA 适配器后 LoRA-Null 低于 MiLoRA 和 CorDA），说明如果只保留残差权重，LoRA-Null 的知识保留效果最差。只是由于微调过程中 BA 对知识的贡献，最终表现才最好——这意味着该方法完全依赖于 LoRA 适配器本身的输出。
3. **隐含的局限性**：
   - 需要额外采集校准集（calibration set）来构造激活值，且校准集的质量直接影响零空间的质量。
   - 需要对 Xₚᵣₑ 做 SVD 并对 W₀U_nullU_nullᵀ 做第二次 SVD，计算开销略高于标准 LoRA。
   - 在下游任务性能上，LoRA-Null 并非总是最优（例如 LLaMA-3.2-3B 上略逊于 CorDA，LLaMA-3.1-8B 上略逊于 MiLoRA）。
