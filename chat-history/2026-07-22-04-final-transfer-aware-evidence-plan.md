# Transfer-aware 最终证据补齐计划

**日期**: 2026-07-22

**会话概况**: 在 E1 三 seed 主表、R1/R2 单 seed 结果与 D1/D2 停止规则的基础上，制定最终多 seed 补充实验与服务器交接提示。

## 1. 决策

- E1 已经是正式三 seed 主表，保持不变。
- R1 gate 在 seed 42 无可测差异，不扩大；D1/D2 classifier-only 扫描未达到 +0.20 Average 阈值，不继续。
- 对 R2 中有意义但尚无方差的 FD/CD 组件结论，补 C0/C1/C2 的 seeds 43/44，结合已有 seed 42 和 E1 的完整 C3，形成新协议下完整三 seed E2-TA。
- 对 legacy E3 无法与新 E1 绝对横比的问题，补 LoRA-Null 与 Gradient-projected LoRA 各三 seed；Standard LoRA 和 LoRA-NF 复用 E1。

## 2. 工作量与输出

- 新增训练任务共 12 个：E2-TA 6 个 + E3-TA 6 个。
- Phase A、Phase B 各占用六张卡的一波，不混合输出目录。
- E2 输出 `experiments/paper_transfer_aware/E2_distill_3seed/`；E3 输出 `experiments/paper_transfer_aware/E3_adapters_3seed/`。
- 两个 summary 必须从原始 JSON 计算 sample std，并明确复用的 seed42/E1 文件。

## 3. 不运行的项目

- 不重跑 E1、LADA、Frozen CLIP、SigLIP2、full-shot、R1、D1/D2、检索-only 或历史 E4/E5。
- 不执行额外超参数搜索，避免事后调参。

## 4. 相关文件

- `docs/final_evidence_experiment_plan.md`
- `docs/server_model_prompt_final_evidence.md`
- `docs/partner_handoff_transfer_aware_lora_nf.md`
