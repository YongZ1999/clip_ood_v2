# 结果章节重组与主张边界更新

**日期**：2026-07-17  
**状态**：当前有效的结果叙事组织补充。

## 变化

结果叙事改为：E1 分类主表 → E2 的分类与 FD/CD 检索证据 → E3 的分类与 adapter 检索证据 → E4/E5 超参数与覆盖范围说明 → LADA 基线及主方法检索 → 汇总 → E7 跨 backbone → 核心结论。

核心结论必须置于所有实验（包括 SigLIP2）之后，以免读者在看到 E7 前接受过强的总括性主张。

## 主张边界

- 检索结论必须分别报告 I2T 与 T2I；不能将 I2T 近似不变扩展为双向检索完全不变。
- NSP 的检索主张应为“E3 中未见系统性损害”，而不是数学意义的“不影响”。
- SigLIP2 条目是各自原生配方下的 backbone robustness evidence，不是 controlled-compute superiority claim。它显示 LoRA-NF 的 Transfer 较高、Native LADA 的 Average/Last 略高。
- E4/E5 没有检索超参数曲线；文稿应说明其为单 seed 调参，FD/CD 与 NSP 的机制级检索证据来自 E2/E3 的 3-seed 对照。

## 相关证据

- `docs/paper_experiment_results.md`：当前正式结果汇总。
- `chat-history/2026-07-17-06-results-document-reorganization.md`：工程事实、数值与文件位置。
