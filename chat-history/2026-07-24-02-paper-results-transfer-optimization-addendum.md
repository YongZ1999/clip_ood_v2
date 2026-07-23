# 正式结果文档补充：Transfer/Average 优化证据

**日期**: 2026-07-24

**会话概况**: 将 transfer-aware 分支的 E0/E1/E2/E3/D1/D2 结果写入 `docs/paper_experiment_results.md`，重点记录 preserve-aspect 的受控三-seed 对照、逐任务影响、与 LADA 的可比边界，以及所有未形成正式优化的负结果。

---

## 1. 已写入的核心证据

- E0-PA 是历史 LoRA-NF 16-shot 配置的严格匹配对照：唯一有意变量为 `legacy_square -> preserve_aspect`，并保持 `classwise` routing。
- 三 seed Ensemble 结果：Transfer 从 60.14 ± 0.20 提升到 61.88 ± 0.09（+1.74）；Average 从 71.75 ± 0.10 提升到 72.57 ± 0.10（+0.82）；Last 基本持平（83.74 -> 83.69）。
- 增益主要来自 Stanford Cars（Transfer/Average +7.61/+6.11）、Oxford Pets（+3.44/+3.09）和 MNIST（+2.60/+1.48）；Caltech101、EuroSAT 存在下降，故结论是总体净收益而非全任务一致增长。
- LADA 官方复现使用同样的等比例 resize；E0-PA 相对本地 LADA 复现为 +0.28 Transfer / +0.27 Average / +0.72 Last。由于分类头和训练方法不同且 Transfer/Average 差距小，只可表述为一致预处理下具有竞争力或略优。

## 2. 负结果与最终协议

- `zs_predicted_seen` routing 的同 artifact 直接对照与历史 `classwise` 得到完全相同的准确率矩阵；不作为贡献，推荐使用 `classwise`。
- alpha、RGDA rank、centers、GMM source、fit iterations 的 D1/D2 单 seed 信号最大约 +0.08 Average，未达到预先规定的 +0.20 门槛；不进入三 seed，不替换正式设置。
- E2-TA 中仅 CD 与 FD+CD 的差异远小于方差；完整 FD+CD 保留。E3-TA 中 LoRA-Null 和 GradProj 均不超过 LoRA-NF。
- 当前推荐配置：`preserve_aspect + classwise`，LoRA-NF rank=4、nsp_eps=.20、nsp_weight=.02、FD=1、CD=2、alpha=.05、M=4、gmm_sample、rank=32、fit=200。

## 3. 相关文件

- `docs/paper_experiment_results.md`: 新增第 11 节，含宏观、同 seed、十任务、LADA 与后续探索表。
- `experiments/paper_transfer_aware/E0_resize_only/`: preserve-aspect 原始三 seed JSON。
- `experiments/paper_transfer_aware/E2_distill_3seed/`: 新协议 FD/CD 三 seed JSON。
- `experiments/paper_transfer_aware/E3_adapters_3seed/`: 新协议 adapter 三 seed JSON。
- `chat-history/2026-07-24-01-transfer-optimization-final-results-and-doc-cleanup.md`: 分支快照、文档清理和最终配置记录。
