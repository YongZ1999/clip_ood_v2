# Preserve-aspect 隔离三-seed 消融计划

**日期**: 2026-07-22

**会话概况**: 用户要求将 `preserve_aspect` 从 transfer-aware E1 的 resize+routing 联合变化中单独隔离，形成严格保持旧最佳配置的三-seed 对照。

## 1. 关键澄清

- 历史 `paper_formal` E1 默认使用 `legacy_square + classwise`。
- transfer-aware E1 同时改为 `preserve_aspect + zs_predicted_seen`。
- R1 表明 routing 在 D1 seed 42 上没有可测分类差异，但它不是 resize 的独立证据。
- 官方 LADA 已经使用 `Resize(shorter_edge)+CenterCrop`，因此无需为预处理一致性重跑 LADA；E0-PA 仅用于量化内部 CLIP 路径中 resize 的独立影响。

## 2. E0-PA 设计

- LoRA-NF 16-shot，seeds 42/43/44。
- 保留旧 E1 的训练、LR-RGDA、retrieval、alpha sensitivity 和 `classwise` routing。
- 唯一有意变化：`--eval_resize_mode preserve_aspect`。
- 同三个 seed 的 legacy source 是 `experiments/paper_formal/E1_main/E1__lora_nf__16shot__seed{42,43,44}_*.json`。
- 新输出目录是 `experiments/paper_transfer_aware/E0_resize_only/`。

## 3. 结论边界

E0-PA 将提供受控三-seed 协议比较，但因为重训轨迹仍可能有微小非确定性，只能表述为“在匹配配置与 seeds 下，标准测试 resize 的平均影响”；不能把它包装为 LoRA-NF 新训练组件。

## 4. 相关文件

- `docs/final_evidence_experiment_plan.md`
- `docs/server_model_prompt_final_evidence.md`
