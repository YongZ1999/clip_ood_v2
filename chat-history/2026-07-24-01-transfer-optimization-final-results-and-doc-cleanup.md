# Transfer 优化结果定论与服务器指令文档清理

**日期**: 2026-07-24

**会话概况**: 基于服务器审计快照 `4af314c`（文件树与 `c5d2bfc` 相同），复核本轮 transfer-aware 实验的目标、代码和原始 JSON；明确哪些改动相对历史正式配置真正提高 Transfer/Average，并清理已完成的服务器运行提示与实验计划文档。

---

## 1. 审计快照与范围

- 服务器分支 `server/audit-snapshot-20260724` 的 commit `4af314c` 以 `c5d2bfc` 为父提交，两个 commit 的 Git tree 相同；它是完整性快照，不引入新的代码或实验结果。
- 本地工作分支已安全快进到该快照。原有未跟踪 audit 文件和 `tmp/` 未修改。
- 本记录只总结本轮“在历史正式 LoRA-NF 配置上提高 Transfer / Average”的实验；不会把已有 LoRA-NF、CD 或 LR-RGDA 基线组件误记为新的优化发现。

## 2. 已验证的有效改动

历史 `paper_formal` LoRA-NF 16-shot 使用 `legacy_square + classwise`，三 seed Ensemble 为：

| 配置 | Transfer | Average | Last |
|---|---:|---:|---:|
| Historical `legacy_square` | 60.14 +/- 0.20 | 71.75 +/- 0.10 | 83.74 +/- 0.04 |
| E0 `preserve_aspect`（其余与历史配置匹配） | 61.88 +/- 0.09 | 72.57 +/- 0.10 | 83.69 +/- 0.15 |
| Delta | +1.74 | +0.82 | -0.05 |

- `preserve_aspect` 是本轮唯一有受控三 seed 证据支持、能提高 Transfer 和 Average 的新改动。
- 它将测试预处理从 `Resize((224, 224))` 改为标准 CLIP 的 `Resize(shorter_edge=224) + CenterCrop(224)`；不改变训练、LoRA-NF、蒸馏或检索编码。
- 官方 LADA 已采用等比例 resize，因此该协议也消除了与 LADA 的预处理不一致。

## 3. 未形成新配置的探索结论

- `zs_predicted_seen` routing 相比历史 `classwise` 的直接 R1 对照得到完全相同的准确率矩阵；不能宣称有增益。E1 相对 E0 的极小均值差异来自独立重训波动，不能归因于 routing。
- `alpha`、RGDA centers、GMM source、fit iterations 与 rank 的 D1/D2 单 seed 扫描，最大的 Average 信号约 +0.08，低于预先规定的 +0.20 门槛；不进入三 seed，也不替换正式超参数。
- E2 中只保留 CD（C2）与 FD+CD（C3）的差异远小于三 seed 方差；FD-only 虽有极小 Transfer 均值变化，但 Average 明显下降。保持完整 FD+CD 配置。
- E3 中 LoRA-Null 与 Gradient-projected LoRA 均没有超过 LoRA-NF；不构成可提升当前最佳 LoRA-NF 的操作。

## 4. 当前推荐协议

- 保留 LoRA-NF 原训练超参数：rank=4、nsp_eps=0.20、nsp_weight=0.02、FD=1、CD=2。
- 采用 `eval_resize_mode=preserve_aspect`。
- 使用历史、较简单的 `ensemble_routing=classwise`；zs-predicted-seen 不作为方法贡献。
- 保持三 seed 已验证的 LR-RGDA 设置：alpha=0.05、num_centers=4、gmm_sample、rank=32、fit=200。

## 5. 文档清理

以下已完成的服务器运行提示和阶段性实验计划已删除，避免它们在项目 docs 中被误当作当前待执行任务：

- `docs/server_model_prompt_transfer_aware_main.md`
- `docs/server_model_prompt_transfer_aware_followups.md`
- `docs/server_model_prompt_final_evidence.md`
- `docs/transfer_aware_main_experiment.md`
- `docs/transfer_aware_followup_experiment_plan.md`
- `docs/final_evidence_experiment_plan.md`

实验设计、执行过程、审计与最终结论保留在 `chat-history/`；合作作者交接摘要 `docs/partner_handoff_transfer_aware_lora_nf.md` 保留，因为它不是服务器运行提示。

## 6. 相关文件

- `experiments/paper_transfer_aware/E0_resize_only/E0_RESIZE_ONLY_SUMMARY.md`: preserve-aspect 三 seed 对照。
- `experiments/paper_transfer_aware/E1_main/TRANSFER_AWARE_SUMMARY.md`: transfer-aware 主表。
- `experiments/paper_transfer_aware/E2_distill_3seed/E2_TA_3SEED_SUMMARY.md`: FD/CD 三 seed 消融。
- `experiments/paper_transfer_aware/E3_adapters_3seed/E3_TA_3SEED_SUMMARY.md`: adapter 三 seed 对比。
- `chat-history/2026-07-20-02-transfer-hyperparameter-diagnosis.md`: 初始诊断。
- `chat-history/2026-07-21-03-phase-d1-single-seed-audit.md`: D1 审计。
- `chat-history/2026-07-22-01-phase-d2-rank-sweep-audit.md`: D2 审计。
