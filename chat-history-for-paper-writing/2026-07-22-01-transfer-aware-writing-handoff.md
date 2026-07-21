# Transfer-aware 论文叙事与交接：合并记录

**日期**: 2026-07-20 至 2026-07-22

**会话概况**: 合并本轮关于 Transfer 基线、transfer-aware 推理协议及论文证据边界的写作记录，并提供给合作方/写作模型的统一入口。

## 1. 叙事起点

历史 LoRA-NF 的 Last 很高，但其绝对 Transfer 与 LADA 原文数值存在差异。这个差异不能直接解释为方法的前向迁移劣势，因为 backbone 实现、分类测试预处理、类别模板及分类器协议可能不同。特别是历史项目使用 `Resize((224,224))`，而标准 CLIP/LADA 风格流程通常保持宽高比后 crop。

因此，论文应优先比较同一 transfer-aware protocol 下的 LoRA-NF 与 Standard LoRA；官方 LADA 论文数值、本地 LADA 复现和本项目 LR-RGDA 结果须明确来源与实现边界，不能混为严格 head-matched 比较。

## 2. 当前可用的论文结论与限制

- E1 三 seed 结果支持：在相同 `preserve_aspect + zs_predicted_seen` 评估协议下，LoRA-NF 的 Transfer、Average、Last 均高于 Standard LoRA。
- 新 protocol 相对旧 protocol 的数值变化不能归因于训练端 LoRA-NF，因为预处理和推理融合同时改变。
- R1 seed-42 routing 对照中，`classwise` 与 `zs_predicted_seen` 的结果完全相同；不可声称 gate 已被实证证明提高 Transfer。
- R2 是单 seed FD/CD + retrieval 分析。CD 对分类 Average/Last 有主要信号，但检索是权衡而非所有方向均提高；正文的强机制结论需要额外多 seed。
- 对 LADA 可写既定端到端协议下的数值对比，并清楚说明分类器/训练流程不同；不可写作严格等训练预算、同 head 的直接胜负。

## 3. 合作方写作入口

详尽的代码改动、三 seed 主表、检索、R1/R2、D1/D2，以及推荐/禁止的论文措辞都集中在：

- `docs/partner_handoff_transfer_aware_lora_nf.md`

原始证据分别在：

- `experiments/paper_transfer_aware/E1_main/`
- `experiments/paper_transfer_aware/E6_gate_seed42/`
- `experiments/paper_transfer_aware/E2_distill_seed42/`
- `docs/paper_experiment_results.md`

## 4. 已合并并删除的论文记录

- `chat-history-for-paper-writing/2026-07-20-02-transfer-baseline-and-tuning-plan.md`
- `chat-history-for-paper-writing/2026-07-20-03-transfer-aware-main-protocol.md`

保留未合并的审计记录：

- `chat-history-for-paper-writing/2026-07-20-01-lada-comparison-narrative-audit.md`
