# 主表分类器指标补全与矩阵口径修正

**日期**：2026-07-17  
**会话概况**：用户要求结果文档为论文写作保留更完整的分类器证据；据 E1 原始 JSON 补齐 ZS、RGDA Last 和 Ensemble 指标，并修正旧 K×K 展示矩阵的 Transfer 行口径。

## 关键决定

- E1 的 16-shot 与 full-shot 主表同时报告 ZS Transfer/Average/Last 和 Ensemble Transfer/Average/Last。
- RGDA 对未见任务没有有效类别统计，其前向位置是实现中的零占位；因此只报告其最终 Last，不把 RGDA 的 Transfer=0 或 Average 写作可比较的连续学习分数。
- 四张 Ensemble K×K 矩阵保留在正文；ZS/RGDA 的逐 seed 原始矩阵位置在文档中明确，避免正文增加四张高度重复的大表。
- 全文三种子离散度统一说明为 population standard deviation，与 `scripts/summarize_continual_metrics.py` 和检索聚合脚本一致。

## 发现与修复

- E1 full-shot 旧表中部分 ± 数值使用了 sample std，而其它正式表使用 population std；已按原始 JSON 与正式聚合器改为统一的 population std。
- 旧 K×K 矩阵的 Transfer 行误填为“紧邻训练前一阶段”的单点准确率，而项目的 LADA Transfer 定义为同一未来任务上所有此前阶段的平均。已重算四张矩阵的 Transfer 行及其 Mean；新值与 JSON 主表一致。
- E7 当前只归档 LoRA-NF Ensemble 输出，LADA 为 native label-specific classifier 输出；文档已显式标记该 head mismatch，禁止把它表述为 ZS 对 ZS 的严格比较。

## 相关文件

- `docs/paper_experiment_results.md`：补全后的主表、单 seed 明细和修正后的矩阵。
- `experiments/paper_formal/E1_main/*_{zs,rgda,ens}_results.json`：唯一数值来源。
- `scripts/summarize_continual_metrics.py`：正式的 population-std 聚合逻辑。
- `chat-history-for-paper-writing/2026-07-17-04-classifier-evidence-clarification.md`：论文主张边界。
