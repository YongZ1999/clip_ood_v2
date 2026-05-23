# Related Work 更新建议报告

> 生成时间：2026-04-20
> 操作类型：Evidence Update + Novelty Repositioning

---

## 一、当前 Related Work 的诊断

### 现有结构
1. **Continual Learning Settings** — 标准但偏基础
2. **Continual Learning with VLMs** — 核心段落（ZSCL / Mod-X / MG-CLIP）
3. **Parameter-Efficient Fine-Tuning for CLIP** — 包含 Prompt-based 方法（L2P / DualPrompt / S-Prompts），但这些**并非 CLIP 持续学习的核心工作**
4. **Inference-Time Classifier Design** — 仅有 RAIL，过于单薄
5. **Relation to Our Work** — 总结段

### 核心问题
| # | 问题 | 风险 |
|---|------|------|
| 1 | **缺少 Null Space Projection 专题段落** | LoRA-NSP 的核心理论贡献（NSP ↔ LoRA 统一）缺乏文献支撑，reviewer 会质疑 originality |
| 2 | **缺少 LoRA + Continual Learning 的讨论** | LoRA-CL (ICLR 2024)、CoDyRA (ICLR 2026) 等已发表/将发表的工作未提及，novelty 边界模糊 |
| 3 | **Prompt-based 方法定位不当** | L2P / DualPrompt / S-Prompts 是针对普通 ViT 的，放在 CLIP-CL 语境中显得牵强 |
| 4 | **Inference-Time 段落单薄** | 仅有 RAIL，未覆盖 Gaussian classifier、test-time adaptation 等相关方向 |
| 5 | **未回应 2025-2026 新出现的直接竞争工作** | DMNSP (ICCV 2025)、LoRA-Null (AAAI 2026) 等与 LoRA-NSP 高度相关，必须在 RW 中明确区分 |

---

## 二、2025-2026 新出现的关键竞争论文

以下论文与 LoRA-NSP 存在**直接技术关联**，必须在 Related Work 中处理：

### 🔴 高优先级（必须在 RW 中讨论并明确区分）

#### 1. DMNSP — Dynamic Multi-Layer Null Space Projection for Vision-Language Continual Learning
- **作者**：Borui Kang, Lei Wang, Zhiping Wu, Tao Feng, Yawen Li, Yang Gao, Wenbin Li
- **会议**：ICCV 2025
- **核心方法**：
  - 发现视觉模态在类别增量中比文本模态有**更大的参数分布和方差**，因此对视觉分支应用 NSP，语言分支正常优化
  - 将视觉参数更新限制在**多个零空间的公共子空间**内
  - 引入**动态投影系数**精确控制梯度投影到零空间的幅度
- **与我们的区别**（必须在 RW 中明确说明）：
  - DMNSP 做的是**硬梯度投影**（backward pass，直接修改梯度方向）
  - LoRA-NSP 做的是**软投影**（forward pass，通过投影矩阵调制 LoRA 更新方向）
  - LoRA-NSP 有**理论统一框架**（证明 NSP 是 LoRA 的退化形式），DMNSP 是经验性方法
  - LoRA-NSP 与 LoRA 参数化天然兼容，DMNSP 需额外维护多层投影矩阵

#### 2. LoRA-Null — Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge
- **作者**：Pengwei Tang, Xiaolin Hu, Yong Liu, Lizhong Ding, Dongjie Zhang, Xing Wu, Debing Zhang
- **会议**：AAAI 2026
- **核心方法**：
  - 发现**LoRA 初始化空间**（而非残差权重）才是保留预训练知识的关键
  - 在**输入激活的零空间**（而非权重的零空间）中初始化 LoRA
  - 实验在 LLM 上进行，非 VLM
- **与我们的区别**（必须在 RW 中明确说明）：
  - LoRA-Null 只关注**初始化阶段**，将 LoRA A/B 初始化为零空间方向
  - LoRA-NSP 关注**整个训练过程**，通过 soft projection 矩阵持续调制更新方向
  - LoRA-Null 未在 VLM 持续学习场景验证，也未建立 NSP 与 LoRA 的理论联系

#### 3. CoDyRA — Adaptive Rank, Reduced Forgetting: Continual Learning with Dynamic Rank-Selective LoRA
- **作者**：Haodong Lu, Chongyang Zhao, Jason Xue, Lina Yao, Kristen Moore, Dong Gong
- **会议**：ICLR 2026
- **核心方法**：
  - 分析 LoRA 秩对可塑性-稳定性的影响：高秩提升可塑性但加剧遗忘，低秩反之
  - 提出**自适应秩选择**：通过稀疏正则化动态最小化每个参数、每个任务的 LoRA 秩
  - 所有参数参与更新，但最小化秩使模型接近先前状态
- **与我们的区别**（可选在 RW 中简要提及）：
  - CoDyRA 通过调整 **rank 大小**来平衡稳定性-可塑性
  - LoRA-NSP 通过 **soft projection 矩阵**改变更新方向，rank 固定
  - 两者正交：CoDyRA 优化"更新多少"，LoRA-NSP 优化"朝哪个方向更新"

### 🟡 中优先级（可在 RW 中简要提及）

#### 4. LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning
- **会议**：CVPR 2025
- **简述**：通过 LoRA 减法构造漂移抵抗子空间

#### 5. KeepLoRA (ICLR 2026)
- **来源**：网易报道提及，ICLR 2026 录用
- **简述**：从参数子空间入手抗遗忘的 VLM 持续学习方法

#### 6. Orthogonal Subspace Projection for Continual Machine Unlearning via SVD-Based LoRA
- **时间**：2026 年 4 月（arXiv 预印本）
- **简述**：基于 SVD 的 LoRA 正交子空间投影

---

## 三、建议的新结构

```latex
\section{Related Work}
\label{sec:related}

\textbf{Continual Learning Settings.}
[精简到 3-4 句，快速带过 TIL→CIL→MTIL→X-TAIL 的演进]

\textbf{Vision-Language Model Continual Learning.}
[合并原第 2+3 段，聚焦 VLM-specific 方法，去掉 Prompt-based]
- Regularization-based: ZSCL
- Architecture-based: MoE-Adapters, LADA, C-CLIP
- Geometric analysis: Mod-X, MG-CLIP
- LoRA-based adaptation: LoRA-CL, CoDyRA [新增]

\textbf{Null Space Projection for Catastrophic Forgetting.}
[★ 新增段落，支撑 LoRA-NSP 的理论贡献]
- GPM (Saha et al., ICLR 2021): 经典 NSP 工作，梯度投影到历史特征零空间
- DMNSP (Kang et al., ICCV 2025): 多层 NSP for VLM，动态投影系数
- LoRA-Null (Tang et al., AAAI 2026): 在激活零空间中初始化 LoRA
- 关键限制: 硬投影/初始化限制可塑性，缺乏 NSP 与 LoRA 的理论统一

\textbf{Inference-Time Classifier Design.}
[扩充]
- RAIL: 扩展分类器维度，固定特征
- [可选补充] Gaussian/Linear classifiers on deep features
- [可选补充] Zero-shot + supervised ensemble 的相关思路

\textbf{Relation to Our Work.}
[更新，明确区分与 DMNSP / LoRA-Null / CoDyRA 的区别]
```

---

## 四、逐段修改建议

### 第 1 段：Continual Learning Settings（精简）

**当前**：约 5 行，详细描述了 TIL→CIL→MTIL→X-TAIL。

**建议**：压缩到 3-4 句，只保留 X-TAIL 的定义，因为其他 setting 不是本文重点。

```latex
\textbf{Continual Learning Settings.}
Early continual learning research considers Task-Incremental Learning (TIL)~\cite{hsu2018re} and Class-Incremental Learning (CIL)~\cite{rebuffi2017icarl}, both assuming tasks from a single domain. Multi-domain Task-Incremental Learning (MTIL)~\cite{zheng2023zscl} relaxes this by introducing diverse domain-specific tasks. Cross-domain Task-Agnostic Incremental Learning (X-TAIL)~\cite{xu2024rail} removes task or domain identities entirely, representing the most challenging and practical setting. We conduct experiments under the X-TAIL protocol.
```

---

### 第 2 段：Vision-Language Model Continual Learning（重组 + 补充）

**当前问题**：
- 原第 2 段（Continual Learning with VLMs）和原第 3 段（PEFT for CLIP）有重叠
- Prompt-based 方法（L2P / DualPrompt / S-Prompts）不是 CLIP-CL 的核心工作

**建议**：合并为一段，聚焦 VLM-specific 方法，按技术路线组织。

```latex
\textbf{Vision-Language Model Continual Learning.}
Existing CL methods for VLMs fall into three categories. \textit{Regularization-based} methods penalize parameter shifts to preserve pre-trained knowledge: ZSCL~\cite{zheng2023zscl} constrains the model in parameter space to prevent zero-shot transfer degradation. \textit{Architecture-based} methods introduce task-specific parameters: MoE-Adapters~\cite{yu2024moe} employ mixture-of-experts adapters with task-dependent routing, while LADA~\cite{luo2025lada} appends label-specific memory units to the frozen encoder. C-CLIP~\cite{liu2025cclip} establishes a multimodal benchmark and proposes an end-to-end framework balancing old and new task learning. \textit{Geometric analysis} approaches study representation structure: Mod-X~\cite{ni2023modx} identifies Spatial Disorder caused by intra-modal rotation and inter-modal deviation, proposing off-diagonal information preservation. MG-CLIP~\cite{huang2025mgclip} reveals the modality gap as a key factor and proposes gap preservation and compensation. Recently, \textit{parameter-efficient fine-tuning} has been explored for VLM continual learning: LoRA-CL~\cite{wang2024lora} studies low-rank adaptation in CL scenarios, and CoDyRA~\cite{lu2026codyra} dynamically optimizes LoRA ranks per parameter and task via sparsity-promoting regularization. Despite these advances, existing methods lack explicit theoretical grounding connecting parameter-efficiency constraints with knowledge preservation mechanisms.
```

---

### 第 3 段：Null Space Projection for Catastrophic Forgetting（★ 新增）

**必要性**：这是 LoRA-NSP 的核心理论基础。当前 RW 完全没有提到 NSP 相关的工作，reviewer 会质疑："为什么你们不 cite GPM / DMNSP？"

```latex
\textbf{Null Space Projection for Catastrophic Forgetting.}
Null space projection (NSP) mitigates forgetting by constraining parameter updates to directions minimally affecting historical representations. GPM~\cite{saha2021gradient} projects gradients onto the approximate null space of previous task features, demonstrating strong stability at the cost of reduced plasticity. More recently, DMNSP~\cite{kang2025dynamic} extends this to vision-language continual learning by applying multi-layer null space projection exclusively to the visual branch with a dynamic projection coefficient, achieving finer stability-plasticity control. Concurrently, LoRA-Null~\cite{tang2026lora} initializes LoRA adapters in the null space of input activations to preserve pre-trained knowledge. However, these approaches treat NSP and LoRA as independent mechanisms: GPM and DMNSP perform hard gradient projection without leveraging LoRA's parameter-efficient structure, while LoRA-Null only affects initialization. Furthermore, all existing NSP methods employ hard projection that zeroes out signal subspaces, inherently limiting model plasticity.
```

**需要新增的 bib 条目**：
```bibtex
\bibitem{saha2021gradient}
G. Saha et al.
\newblock Gradient projection memory for continual learning.
\newblock In {\em ICLR}, 2021.

\bibitem{kang2025dynamic}
B. Kang et al.
\newblock Dynamic multi-layer null space projection for vision-language continual learning.
\newblock In {\em ICCV}, 2025.

\bibitem{tang2026lora}
P. Tang et al.
\newblock Put the space of LoRA initialization to the extreme to preserve pre-trained knowledge.
\newblock In {\em AAAI}, 2026.

\bibitem{lu2026codyra}
H. Lu et al.
\newblock Adaptive rank, reduced forgetting: Knowledge retention in continual learning vision-language models with dynamic rank-selective LoRA.
\newblock In {\em ICLR}, 2026.
```

---

### 第 4 段：Inference-Time Classifier Design（扩充）

**当前**：仅有 RAIL，2-3 句。

**建议**：增加对 Gaussian-based classifier 和 ensemble 思路的铺垫，为 LR-RGDA 做铺垫。

```latex
\textbf{Inference-Time Classifier Design.}
Beyond training-phase adaptation, inference-time strategies are critical for balancing in-distribution (ID) and out-of-distribution (OOD) performance. RAIL~\cite{xu2024rail} extends the classifier dimension while freezing feature representations, relying on the vanilla zero-shot CLIP classifier for OOD detection. Gaussian discriminant analysis on deep features has been explored for OOD detection~\cite{lee2018simple}, but its direct application as a classifier in continual learning remains underexplored due to computational and sample-efficiency challenges. Our approach differs by integrating a supervised LR-RGDA classifier with the zero-shot classifier: LR-RGDA's principled uncertainty estimation naturally discriminates ID from OOD samples without additional training, enabling reliable ensemble routing.
```

---

### 第 5 段：Relation to Our Work（更新）

**当前**：较笼统，未区分与具体竞争工作的差异。

**建议**：逐点明确区分。

```latex
\textbf{Relation to Our Work.}
Unlike existing methods that focus on either training or inference optimization, our approach co-optimizes both stages through theoretically grounded mechanisms. \textit{Compared to NSP-based methods} (GPM, DMNSP), LoRA-NSP establishes a novel theoretical connection proving that NSP is a degenerate form of LoRA, and proposes soft projection that preserves all eigendirections rather than hard zeroing. \textit{Compared to LoRA-Null}, LoRA-NSP operates throughout training via forward-pass projection rather than initialization only, and is validated on VLM continual learning. \textit{Compared to CoDyRA}, LoRA-NSP controls update directions via projection matrices rather than varying rank sizes, offering orthogonal and complementary benefits. \textit{Compared to RAIL}, our ensemble classifier leverages supervised Gaussian discriminant analysis rather than zero-shot confidence alone, providing more principled ID/OOD discrimination. Importantly, our training innovations (LoRA-NSP + FD + CD) and inference innovations (LR-RGDA ensemble) are orthogonal and can be combined for maximum performance.
```

---

## 五、Novelty 影响评估

### 需要弱化的表述

当前 Introduction 中关于 LoRA-NSP novelty 的表述需要微调：

| 当前表述 | 问题 | 建议修改 |
|---------|------|---------|
| "We establish a novel theoretical connection between NSP and LoRA, proving that NSP represents a degenerate form of LoRA" | 需要确认该理论是否已被 LoRA-Null / DMNSP 触及 | 保留，但强调是 "in the context of parameter-efficient continual learning" |
| "proposing soft projection that generalizes both approaches" | 需要确认 soft projection 是否已被 DMNSP 的动态系数覆盖 | 强调我们的 soft projection 是 **forward-pass** 且 **与 LoRA 参数化统一**，而非 DMNSP 的 backward-pass 硬投影 |

### 可以强化的表述

- LoRA-NSP 是**首个**将 NSP 和 LoRA 在理论层面统一，并在 forward pass 中实现 soft projection 的工作
- LR-RGDA ensemble 是**首个**将低秩正则化高斯判别分析与零 shot 分类器进行 principled ensemble 的工作

---

## 六、新增参考文献清单

| 论文 | 会议/时间 | 需添加位置 | 重要性 |
|------|----------|-----------|--------|
| Saha et al., GPM | ICLR 2021 | Null Space 段落 | 核心（经典NSP） |
| Kang et al., DMNSP | ICCV 2025 | Null Space 段落 | 核心（直接竞争） |
| Tang et al., LoRA-Null | AAAI 2026 | Null Space 段落 | 核心（直接竞争） |
| Lu et al., CoDyRA | ICLR 2026 | VLM CL 段落 | 重要（LoRA+CL） |
| Liu et al., LoRA Subtraction | CVPR 2025 | Null Space 段落（可选） | 相关 |
| Rahulamathavan et al., OSP | arXiv 2026 | Null Space 段落（可选） | 相关 |

---

## 七、修改工作量估计

| 任务 | 工作量 | 优先级 |
|------|--------|--------|
| 精简 Settings 段落 | 5 min | 低 |
| 重组 VLM CL 段落（去掉 Prompt-based，补充 LoRA-CL / CoDyRA） | 15 min | 高 |
| **新增 Null Space Projection 段落** | 20 min | **最高** |
| 扩充 Inference-Time 段落 | 10 min | 中 |
| 更新 Relation to Our Work | 10 min | 高 |
| 补充 bib 条目（6篇） | 10 min | 高 |
| **检查 Introduction novelty 表述** | 15 min | **最高** |

**总计：约 1.5 小时**

---

## 八、下一步行动

1. **确认**：你是否同意上述结构重组？特别是新增 "Null Space Projection" 段落。
2. **补充文献**：是否需要我帮你获取 GPM / DMNSP / LoRA-Null 的 PDF 并生成摘要，放入 `paper-summaries/`？
3. **执行修改**：确认后我可以直接修改 `paper_draft.tex` 并更新 bib。
4. **Introduction 联动检查**：修改 RW 后需要检查 Introduction 的贡献声明是否需要微调 novelty 表述。
