# Transfer-Aware 后续实验最小计划

**日期**: 2026-07-22
**会话概况**: 基于 E1 三-seed 主表与 D1/D2 单 seed classifier sweep，制定仅保留高论文价值证据的后续实验；避免因新评估协议而不必要地重跑所有历史消融。

---

## 1. 决策

- 不再为主表数值继续调 classifier：D1/D2 的最高 Average 提升不到预设 +0.20 门槛。
- 必跑 R1/E6-TA：零训练成本，在 D1 artifacts 上直接比较旧 classwise 与新 `zs_predicted_seen` gate，给新推理组件提供隔离证据。
- 必跑 R2/E2-TA：在新协议下用同一 seed 跑 C0/C1/C2 三个蒸馏条件；D1 的 FD+CD seed-42 轨迹作为 C3，不重训。每个条件同时输出分类与检索，直接回答组件对检索保持的作用。
- R3 adapter family 仅当论文要把所有 adapter 放到新协议同表排序时才执行；否则保留清晰标注 legacy protocol 的历史 E3。

## 2. 不重跑范围

- 检索-only、E4 超参、LADA、Frozen CLIP、SigLIP2 和三-seed E1 均不重跑。
- 历史 E2-E5 可作为 legacy protocol 机制证据，但不得与 transfer-aware E1 的绝对分类数值混表。

## 3. 相关文件

- `docs/transfer_aware_followup_experiment_plan.md`
- `docs/server_model_prompt_transfer_aware_followups.md`
- `chat-history/2026-07-21-03-phase-d1-single-seed-audit.md`
- `chat-history/2026-07-22-01-phase-d2-rank-sweep-audit.md`
