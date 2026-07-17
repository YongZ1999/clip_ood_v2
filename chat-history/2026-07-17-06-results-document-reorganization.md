# 结果文档重组与检索证据归位

**日期**：2026-07-17  
**会话概况**：将正式实验结果文档按“分类实验—对应检索证据—主方法/外部基线检索—跨 backbone—结论”重组，并明确未为全部单点超参数补跑检索的证据边界。

## 关键决定

- 原 E8 的 FD/CD 检索表移入 E2，adapter 检索表移入 E3；两者均复用对应的既有 3-seed 模型和任务后检索 JSON，不产生新训练。
- E4/E5 不新增每个单点配置的检索曲线：E2 已提供 FD/CD 存在性的三种子因果对照，E3 已提供标准 LoRA、hard-null 与 leaky NSP 的三种子对照；E4/E5 为单 seed 分类敏感性/调参扫描。
- 第 7 节保留 Frozen CLIP、官方 LADA 和 full-shot 主方法的检索；16-shot adapter 对照以第 3.1 节为唯一位置，避免重复表格。
- E7 明确为两种方法保留各自原生训练日程的 SigLIP2 迁移实验，不能解释为同训练预算效率比较。

## 重要发现与表述修正

- E2 的 I2T 与 T2I 结论不同：CD/完整模型 I2T 接近 C0，但 Flickr30K T2I 仍低 1.5--2.3 点；后续论文不得笼统写成“所有检索能力不变”。
- E3 三种 adapter 的表中最大均值差为 0.60 R@1，支持“没有证据表明 NSP 系统性损害检索”，而非绝对不变。
- E7：LoRA-NF Transfer 高 1.80，Native LADA Average/Last 高 0.71/0.38；仅支持跨 backbone 可运行性与竞争力。

## 相关文件

- `docs/paper_experiment_results.md`：重组后的正式结果文档。
- `experiments/paper_formal/E7_siglip2/`：E7 原始三种子 JSON 和汇总表。
- `chat-history-for-paper-writing/2026-07-17-03-results-narrative-reorganization.md`：对应的论文叙事与 claim 边界记录。
