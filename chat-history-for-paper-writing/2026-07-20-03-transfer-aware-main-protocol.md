# Transfer-aware 主实验的论文证据边界

**日期**: 2026-07-20
**关联工程记录**: `chat-history/2026-07-20-03-transfer-aware-main-protocol-implementation.md`

## 原问题

现有 LoRA-NF 有较高 Last，但其绝对 Transfer 低于 LADA 原文数字。直接用 alpha 或 RGDA 训练次数追逐 Transfer，容易产生事后选参风险，也无法区分 backbone 起点、预处理和分类器偏置的影响。

## 新的、待验证协议

新分支只为 E1 主实验引入两项具有明确推理含义的改变：

1. 用标准 CLIP 的保持宽高比测试预处理，消除与 LADA/标准 CLIP 流程的明显协议差异；
2. 在 ZS 已将样本预测为未来类时保留 ZS，避免已见类 LR-RGDA 分支把该样本重新拉回历史类别。

这不是训练端 LoRA-NF 的新组件，也不是利用真实 OOD 标签的 router。门控只使用当前样本的 ZS argmax，属于可在部署时执行的确定性 inference rule。

## 论文使用边界

- 新结果生成前，不能声称该规则提高 Transfer。
- 若新协议同时提高或保持 Transfer、Average、Last，可将其作为完整推理端 `LR-RGDA + ZS-gated ensemble` 的定义，并与旧 classwise 版本在附录中说明差异。
- 若只提高 Transfer 但明显损失 Last，应报告为 forward-transfer / final-ID trade-off，不能写作无条件改进。
- 主表内部 LoRA 与 LoRA-NF 必须使用同一新协议；不能用新 LoRA-NF 对比旧 square-resize LoRA。
- LADA 官方数字、LADA 本地复现与新内部协议应继续分栏；跨实现绝对值差异不得只归因于方法本身。

## 实验最小集

公平 E1 主表需要 Standard LoRA 与 LoRA-NF 的 16-shot/full-shot、三个 seeds，共 12 个训练任务。无需为这次 inference/evaluation 改动重跑 E2--E6；这些旧结果仍属于历史 `classwise + legacy_square` 协议，若正文采用新协议需明确标注，不能混合求均值。
