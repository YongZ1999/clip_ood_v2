# Agent 接手与 LR-RGDA 初步诊断

**日期**: 2026-06-12
**会话概况**: 新 agent 接手项目，复核最近 LADA/LR-RGDA 实验、核心实现、原始 LR-RGDA 推导和历史超参数搜索，确认当前瓶颈不是简单的 ensemble alpha 失调。

---

## 1. 当前可信结果

- 2026-06-07 早期出现的 RGDA 约 86% 结果不可信：评估脚本解包错误，把 `train_loader4updating` 当成完整测试集。
- 修复后，在真实测试集、分类器构建使用 `test_transform` 的条件下：
  - LR-RGDA 4-center + 200 轮 CE 微调：77.7%
  - LR-RGDA + ZS 最优集成：79.0%
  - LADA k=16 + 200 轮 CE 微调：78.8%
  - LADA + ZS 最优集成：79.9%
- 当前 LADA 稳定领先 LR-RGDA 约 0.9-1.1 个百分点。

## 2. 初步诊断

### 2.1 当前“微调 LR-RGDA”偏离原始方法定位

原始 LR-RGDA 推导强调解析式分类器和无需迭代优化。当前代码的 `fit()`：

- 仅训练 `affine_weights` 和 `affine_biases`
- 冻结 `U_eff_T_B_inv`、`M_invs` 等低秩协方差结构
- 训练后更接近“线性分类器 + 固定二次残差”，不再是纯解析 LR-RGDA

因此，用该结果支撑“Bayes-optimal analytic classifier”会存在方法论不一致。

### 2.2 16-shot 下低秩类协方差信息天然受限

- 每类只有 16 个样本，中心化样本协方差秩最多为 15。
- 当前默认 `rank=32` 超过可观测秩，多出的方向不包含真实类内统计信息。
- 512 维单位球面特征下，类协方差估计噪声很大；LADA k=16 基本直接保留每个训练样本作为原型，容量与 16-shot 更匹配。
- 多中心解析 LR-RGDA 反而明显下降，说明“类内多模态 + 共享类协方差”的当前组合没有形成有效增益。

### 2.3 现有超参数搜索目标不适用于当前问题

`optimization/lr_rgda_alpha_grid*` 的优化指标是 OOD AUROC、FPR@95TPR 和 detection error。OOD 检测模块已在 2026-05-31 删除，当前目标是分类准确率与持续学习指标。

因此，文档中的“最优 LR-RGDA α₁/α₂/α₃”不能视为分类任务最优参数，需要按验证集分类指标重新搜索。

### 2.4 当前比较仍缺少关键消融

尚未回答低秩二次项本身是否有效。需要至少比较：

- LDA：`alpha1=0`
- 解析 LR-RGDA：`alpha1>0`
- 只微调线性项的 LR-RGDA
- 普通线性 probe
- nearest class mean / cosine prototype
- LADA k=16

如果“LR-RGDA fit”不能稳定超过 linear probe，则提升来自 CE 微调而非低秩 Gaussian 结构。

## 3. 下一步实验优先级

1. 建立统一、无数据泄漏的分类器 benchmark，固定同一 16-shot 子集、`test_transform`、完整测试集和 per-dataset macro average。
2. 做 `rank ∈ {0, 1, 2, 4, 8, 15}` 与分类目标的 `alpha1/alpha2/alpha3` 搜索；不再沿用 OOD 搜索结果。
3. 加入 linear probe、LDA、nearest-class-mean 基线，量化低秩二次修正的净贡献。
4. 分离“解析 LR-RGDA”和“可训练判别式变体”，避免混用同一名称和理论主张。
5. 只有确认二次项有稳定增益后，再研究更合理的 LR-RGDA + ZS 校准；当前 ensemble alpha 不是主要瓶颈。

## 4. 文档风险

- `PROJECT_DOCUMENTATION.md` 和论文草稿仍包含已删除的 OOD 路由叙述。
- 论文草稿声称 LR-RGDA 对 OOD 输出近零置信度，这一说法对 softmax 后验一般不成立。
- 论文草稿的超参数与当前代码/历史记录不一致。
- 在完成上述消融前，不应宣称 LR-RGDA 优于 LADA 或达到 SOTA。

## 5. 相关文件

- `src/classifiers/gaussian_classifier.py`: LR-RGDA 解析构建、低秩二次项和 `fit()`
- `src/classifiers/gaussian_statistics.py`: 16-shot 类协方差与多中心统计
- `src/lada/lada_classifier.py`: LADA 原型构建与微调
- `debug_ensemble_alpha.py`: 修复后的 LR-RGDA 对比实验
- `debug_lada_eval.py`: 最新 LADA 对比实验
- `optimization/lr_rgda_alpha_grid*/`: 以 OOD 指标为目标的旧搜索结果
- `paper_writing/my_original_papers/LR-RGDA_images/`: LR-RGDA 原始推导
