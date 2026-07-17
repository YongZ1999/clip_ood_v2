# 分类器证据的论文呈现边界

**日期**：2026-07-17  
**状态**：当前有效的实验表述补充。

## 呈现原则

论文主表必须同时区分：

- **ZS**：LoRA-NF 训练端是否保留/改善 CLIP 的直接分类能力；
- **Ensemble**：训练端加 LR-RGDA 推理端后的完整方法能力；
- **RGDA Last**：仅作为最终阶段判别头能力的辅助分解，不能报告为有效的前向 Transfer/Average。

因此，论文不得只以 Ensemble 结果来推导所有 LoRA-NF 的贡献；ZS 的改进（16-shot Last 80.14 vs 78.58）是训练端 NSP + 蒸馏的直接证据，而 Ensemble 的额外提升是推理端协同的证据。

## 需避免的表述

- 不把 RGDA 的 0 Transfer 当作模型性能；该值来自未见类别没有构建判别头的占位。
- 不将 E7 写成 ZS 对 ZS 或严格公平的分类器比较；当前为 LoRA-NF Ensemble 与 Native LADA label-specific classifier 的端到端输出比较。
- K×K 矩阵的 Transfer 必须按“任务 k 被训练前的全部阶段平均”计算，不能用仅训练 k-1 后的单点结果代替。

## 相关证据

- `docs/paper_experiment_results.md`。
- `chat-history/2026-07-17-07-classifier-metrics-completeness.md`。
