# Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters (MoE-Adapters) 内容总结

- **作者**: Jiazuo Yu et al. | **年份**: 2024 | **会议/期刊**: CVPR 2024

---

## 问题定义

### 论文试图解决什么问题？
CLIP 在持续学习中的两个核心痛点：
1. **灾难性遗忘**：微调后旧知识丢失。
2. **推理时的任务身份（Task Identity）依赖**：大多数高效微调方法（如 Prompt/Adapter）需要知道当前属于哪个任务才能调用正确的参数，这在现实的 CIL 或无身份 TIL 场景中不可行。
3. **计算与存储平衡**：如何在保持可塑性的同时，降低全参数微调的巨大开销。

### 为什么这个问题重要？
- 传统的 CLIP 微调要么破坏零样本能力（前向遗忘），要么需要手动指定任务标签进行推理，这限制了其作为自动化通用模型的使用。

### 在此之前最好的方法是什么？它有什么局限？
- **ZSCL**：通过参考集蒸馏保持特征空间。
  - **局限**：训练开销大（双向蒸馏耗时），且本质上还是共享一套权重，容量有限。
- **L2P / DualPrompt**：使用 Prompt Pool。
  - **局限**：Prompt 作用于输入端，对深层语义的修改能力弱于 Adapter；且推理时依赖 Query-Key 匹配任务，容易出错。

---

## 核心方法

### 核心思路（一句话，用直觉说清楚）
MoE-Adapters 将 **LoRA 适配器** 作为专家，通过 **任务特定路由（Task-specific Routers）** 进行动态激活，并利用一个**基于自编码器的自动选择器（DDAS）**在推理时自动识别任务身份或判定为 OOD 数据。

### 核心公式

#### 1. MoE 结构
对于任务 $t$，输出 $\bm{y}^t$ 是专家 $\mathcal{E}_i$ 的加权和：
$$ \bm{y}^t = \sum_{i=1}^{N_E} W_i^t \mathcal{E}_i(\bm{x}^t) $$
其中 $W^t$ 是路由权重，由 `[CLS]` token 经过路由层 $\mathcal{R}^t$ 计算得到：
$$ W^t = \text{Softmax}(\text{Topk}(\mathcal{R}^t(\bm{c}^t))) $$

#### 2. 增量激活-冻结策略 (Activate-Freeze Strategy)
- 训练新任务时，之前任务中最常激活的 $Top$-$k$ 专家被冻结以保护旧知识。
- 允许新路由调用旧专家（知识迁移）或训练新专家（学习新知识）。

#### 3. 自动选择器 (DDAS)
利用一组自编码器 $\{\mathcal{F}_A^t\}$ 捕获任务分布。推理时，计算重构误差 $d^t$：
$$ d^t = ||\bm{f}_i^t - \mathcal{F}_A^t(\bm{f}_i^t)||^2 $$
- 若 $\min(d^t) > \text{Threshold}$，判定为未见过的 OOD 数据，重定向回原始 CLIP（零样本）。
- 否则，选择 $d^t$ 最小的任务路由。

---

## 实验设计

### 用的什么数据集/benchmark？和哪些 baseline 比较？
**数据集**：
- MTIL (11 个跨域数据集，如 LADA 所用)。
- CIL (CIFAR100, TinyImageNet)。

**Baseline**：
- ZSCL (ICCV 2023)
- LwF-VR, WiSE-FT, iCaRL
- Continual-FT

### 核心结果是什么？
- **性能 SOTA**：在 MTIL 上，相比 ZSCL，Average 提升 1.3%-1.8%，Last 提升 1.4%-3.8%。
- **效率提升**：训练参数减少 60%，显存占用减少 15%，迭代时间缩短 60%（相比 ZSCL）。
- **零样本保持**：DDAS 有效区分了已学任务和 OOD 数据，使得 Transfer 指标保持在 68.9% 左右（仅比原始 CLIP 下降 0.5%）。

---

## 作者自己说的局限
1. **专家池大小**：虽然方法鲁棒，但专家总数 $N_E$ 仍需预设。
2. **任务识别误差**：DDAS 在某些相似任务（如 Task 9 和 11）间存在重构误差重叠，导致误分类。
3. **架构复杂性**：引入了 MoE 和自编码器组，虽然单步快，但系统整体维护较复杂。
