# 可执行 Goal Contract：LoRA-NSP + 集成分类器发表目标

**日期**: 2026-06-14

## 1. Goal 模式目标

将 LoRA-NSP + LR-RGDA/零样本集成推进到可发表研究标准：在严格对齐 LADA/X-TAIL 协议、无数据泄漏、完整测试集和多随机种子评估下，证明该方法在一个清晰且有意义的约束场景中相对 LADA 具有稳定优势，并用训练端、推理端、存储预算和消融证据解释优势来源。

当前目标不应写成“全面超过 LADA”。更准确的目标是：

> 在真实历史特征可保存时，诚实报告 LADA 的强势边界；在真实历史特征不可保存、只能保存紧凑统计量的 replay 场景中，证明 LR-RGDA/零样本集成比 LADA-style label-memory classifier 更能利用压缩统计信息；同时验证 LoRA-NSP 是正交的训练端贡献，可以改善持续学习表征并与推理端分类器组合。

## 2. 当前证据边界

### 支持主 claim 的证据

`test_transform + raw GMM component-mean replay + fixed mean center protocol` 三种子结果：

| Replay | CLIP-ZS | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS |
|---|---:|---:|---:|---:|---:|
| test_gmm_raw_mean_fixed | 56.40 +/- 0.00 | 76.93 +/- 0.16 | 76.08 +/- 0.35 | 77.12 +/- 0.13 | 76.08 +/- 0.35 |

结论：

- LR-RGDA 三个 seed 均超过 LADA，平均约 `+0.85 pp`。
- LR-RGDA+ZS 三个 seed 均超过 LADA，平均约 `+1.04 pp`。
- 这是当前最干净的推理端优势证据。

### 必须诚实报告的反例

同样 `test_transform` 和真实 16-shot 特征下：

| Replay | CLIP-ZS | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS |
|---|---:|---:|---:|---:|---:|
| test_real | 56.40 +/- 0.00 | 77.87 +/- 0.16 | 78.92 +/- 0.16 | 78.05 +/- 0.14 | 78.92 +/- 0.16 |

结论：

- 真实特征可保存时，LADA 领先 LR-RGDA 约 `+1.05 pp`。
- 论文不能声称 LR-RGDA 在真实特征或所有 replay 设置中全面优于 LADA。

## 3. 主论文 Claim 草案

建议主 claim：

> We study a two-stage CLIP continual learning framework that separates representation adaptation from classifier construction. LoRA-NSP provides a training-stage mechanism for soft null-space constrained low-rank adaptation, while LR-RGDA provides an inference-stage classifier for compact statistical replay. Against LADA, our current strongest evidence is not a universal win under unrestricted feature storage; rather, LR-RGDA/zero-shot ensemble consistently improves over LADA-style label memory when historical features are compressed into GMM component statistics, revealing a memory-performance regime not covered by standard real-feature replay comparisons.

中文等价表述：

> 本文的可发表贡献不是声称 LR-RGDA 全面替代 LADA，而是指出一个更细的存储约束场景：当历史真实特征不能保存、只能保存紧凑 GMM 统计量时，LR-RGDA/零样本集成比 LADA 的标签记忆分类器更稳定地利用压缩统计信息；同时 LoRA-NSP 作为训练端贡献改善 CLIP 表征稳定性，并与该推理端机制正交组合。

## 4. 硬性验收标准

发表级主结论必须同时满足：

- 使用 X-TAIL 10 数据集、16-shot、完整测试集。
- 每个主结论至少 `seed=42,43,44` 三种子，报告 `mean +/- std`。
- LADA 与 LR-RGDA 使用同一 feature source、同一 classifier transform、同一 train/test split、同一评估标签空间。
- 默认使用 `classifier_feature_transform=test` 作为公平主口径；`train` 只作为 transform ablation。
- 所有 GMM replay 结果必须明确区分 `raw` vs `sphere`、`mean` vs `sample`，不能混用旧的错误 raw-space 结果。
- `test_gmm_raw_mean_seed42` 旧结果不得与 fixed mean 三种子结果混合。
- 若声称超过 LADA，平均优势应接近或超过 `+1 pp`，且每个 seed 不出现反向；否则写成持平、trade-off 或场景性优势。
- 论文中删除或移出主线：无条件 SOTA、OOD router、LR-RGDA 天然 OOD 近零置信度、真实特征场景全面超过 LADA。

## 5. 实验里程碑

### M1：推理端 replay 边界定稿

必做：

- 汇总 `test_gmm_raw_sample` 三种子。
- 若 raw sample 稳定超过 LADA，将 claim 扩展为 raw statistical replay。
- 若 raw sample 不稳定或低于 LADA，将 claim 收紧为 component-mean replay。
- 补跑 `gmm_sphere sample`，区分 raw-space GMM 和 sphere-space GMM 的影响。

验收产物：

- replay matrix：`test_real`、`test_gmm_raw_mean_fixed`、`test_gmm_raw_sample`、`test_gmm_sphere_sample`。
- 每组包含 CLIP-ZS、LR-RGDA、LADA、LR-RGDA+ZS、LADA+ZS。
- 每组包含三种子 mean/std 和 per-seed deltas。

### M2：LoRA-NSP 训练端贡献

必做：

- 跑 LoRA baseline、LoRA-NSP、LoRA-NSP+FD、LoRA-NSP+FD+CD。
- 在同一 classifier protocol 下评估训练端收益，避免把推理端收益误归因给 LoRA-NSP。
- 报告 Transfer、Average、Last，以及必要的 forgetting 指标。

验收产物：

- 训练端主表。
- LoRA-NSP 的单独 ablation。
- LoRA-NSP 与 LR-RGDA ensemble 的组合表。

### M3：LADA 对齐与官方 DPT 边界

必做：

- 保留 real-feature LADA 强基线，并在论文中明确其优势。
- 跑或复核 LADA official DPT 风格消融，避免把当前“从 GMM 伪样本重建分类器”的 replay 实验误写成官方 DPT 等价实验。
- 给出 LADA 与 LR-RGDA 在存储预算上的对照。

验收产物：

- LADA official protocol 对齐说明。
- DPT vs compact replay 的语义差异说明。
- 存储预算表：真实特征、LADA centers、GMM component means、GMM parameters、LR-RGDA statistics。

### M4：论文包与复现包

必做：

- 更新 `paper_draft.tex`：摘要、贡献、方法、实验、限制全部与当前证据一致。
- 补齐主表、消融表、存储预算表和 replay protocol 表。
- 清理旧文档中的 OOD router / SOTA / near-zero OOD confidence 叙事。
- 提供可复现命令和结果目录说明。

验收产物：

- 可编译 PDF。
- `README` 或 experiment script 说明。
- 每个表格可追溯到 JSON 结果和命令。

## 6. Stop/Go 判据

### 可以继续推进为主论文的条件

- `test_gmm_raw_mean_fixed` 保持为主证据，并至少再有一个 replay 或 storage-budget 维度支持该结论。
- LoRA-NSP 训练端至少在 Average/Last 或 forgetting 上提供稳定正贡献。
- 论文叙事能清晰解释为什么“真实特征下 LADA 强、压缩统计回放下 LR-RGDA 强”不是矛盾，而是存储预算改变后的分类器归纳偏置差异。

### 需要降级为 workshop/short paper 的条件

- LoRA-NSP 训练端贡献不稳定，只剩推理端 replay 贡献。
- LR-RGDA 优势只存在于 `component-mean replay`，且 raw/sphere sample 都不支持。
- 无法在 LADA official DPT 语义下建立合理边界。

### 需要暂停主 claim 的条件

- 后续无泄漏复核发现 replay 设置仍混入真实特征统计。
- 多种子下 LR-RGDA 对 LADA 的优势消失或反向。
- LoRA-NSP 训练端与推理端收益无法拆分，导致贡献归因不清。

## 7. 下一步执行顺序

1. 查询 `test_gmm_raw_sample` 三种子是否完成，并记录结果。
2. 根据结果决定是否启动 `gmm_sphere sample` 三种子。
3. 更新 replay matrix 和论文 `Statistical Replay Protocol`。
4. 启动 LoRA-NSP 训练端主消融。
5. 补齐 LADA official DPT 边界和存储预算表。
