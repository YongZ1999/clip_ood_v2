# Transfer 基线对齐与调参叙事建议

## 核心判断

论文不宜直接把 LoRA-NF 的 Transfer 60.14 与 LADA 原文 61.5 解释成方法造成的前向迁移劣势，因为两者 Frozen CLIP 起点和评估预处理并不一致。

- 本项目 Frozen CLIP Transfer：60.37。
- LoRA-NF ZS / Ensemble Transfer：59.84 / 60.14。
- LoRA-NF Ensemble 相对自身 Frozen CLIP 仅下降 0.23。
- LADA 原文的 Transfer 61.5 与其报告的各数据集 Frozen CLIP 零样本精度所对应的未来任务均值基本一致。

因此，更准确的表述是：LoRA-NF 在显著提高 Last 的同时，基本保留了自身基座的前向零样本迁移能力。跨论文绝对值比较前，应先统一模型实现、图像预处理、类别与模板协议。

## 需要优先核验的协议差异

当前项目分类测试使用 `Resize((224,224))`，官方 LADA 使用保持宽高比的 `Resize(224)+CenterCrop(224)`。该差异可能解释 Stanford Cars、Oxford Pets 等数据集上的大部分 Frozen CLIP 起点差异。

建议先报告双预处理 Frozen CLIP 诊断结果。若采用标准 CLIP 预处理，则所有需要横向比较的分类方法应使用同一预处理重新评估或重跑，不能只更新 LoRA-NF 一行。

## 超参数叙事边界

- 调大 alpha 只会在 Transfer 与 Average/Last 之间做很小的重新平衡；已有数据不支持其带来大幅 Transfer 提升。
- 增加 RGDA 拟合步数主要作用于已见类，理论上不是提高未来未见类 Transfer 的直接手段。
- 当前工程内置 LADA prototype classifier 不等同于官方 LADA，不能以“直接换用 LADA 分类器”作为官方方法对比。
- 若引入“ZS 判为未见类时不启用适配分类头”的条件融合，它应作为新的 inference gating 组件单独报告，并通过验证集确定规则。

## 推荐论文实验呈现

1. 先给出 Frozen backbone、LoRA-NF ZS、LoRA-NF Ensemble 的 Transfer 差值，强调相对基座保留率。
2. 再给出统一预处理后的绝对 Transfer/Average/Last。
3. 对 alpha、RGDA iterations/rank/source 报告 Pareto，而不是只报告一个对测试集最有利的点。
4. 将官方 LADA 论文数字与本地复现分栏，明确实现、预处理和分类器协议差异。

本记录为诊断与实验规划，不代表已经修改正式代码或产生新的实验结果。
