# 可发表目标定义：LoRA-NSP + 集成分类器对齐 LADA

**日期**: 2026-06-14
**会话概况**: 用户希望在 goal 模式下定义一个可执行的研究目标：让 LoRA-NSP + LR-RGDA/集成分类器达到超过 LADA 并足以支撑论文发表的水平。本记录将目标收敛为可验收的实验、方法与写作标准。

---

## 1. 核心目标

将论文目标从泛化的“超过 LADA”收紧为：

> 在严格对齐 LADA/X-TAIL 协议、无数据泄漏、完整测试集和多随机种子评估下，证明 LoRA-NSP + LR-RGDA/零样本集成在至少一个清晰场景中相对 LADA 具有实质性优势，并用充分消融解释优势来源。

优先可发表场景为：

1. 真实 16-shot 特征可用时，LoRA-NSP 训练端提升最终持续学习指标。
2. 历史真实特征不可保存、只能保存轻量统计量时，LR-RGDA 比 LADA 更能利用 GMM/均值回放。
3. 训练端 LoRA-NSP 与推理端 LR-RGDA 的收益可相加，且不依赖 OOD 路由旧叙事。

## 2. 硬性验收标准

- 主表必须使用 LADA 协议的 X-TAIL 10 数据集、16-shot、Transfer/Average/Last 指标。
- 每个主结论至少 3 个 seeds，报告 mean +/- std。
- 不使用曾经出现过问题的评估路径：不能把 `train_loader4updating` 当测试集，不能把已归一化特征误称 raw 特征。
- LADA 和 LR-RGDA 必须在同一特征源、同一训练子集、同一测试集、同一 transform 口径下比较。
- 如果声称超越 LADA，优势应至少约 1 个百分点或超过随机种子标准差；否则表述为持平或场景性优势。
- 论文中必须删除或弱化 OOD router、LR-RGDA 对 OOD 近零置信度、无条件 SOTA 等当前证据不足的表述。

## 3. 优先实验路线

1. 完成 `classifier_feature_transform=train|test` ablation，验证 test_transform 是否提升最新 joint replay 中的 LR-RGDA/LADA 真实特征结果。
2. 若 test_transform 提升成立，重跑 `real`、`gmm_raw_mean`、必要时 `gmm_raw/gmm_sphere` 的 3-seed 对照。
3. 在增量设置中跑 LADA official DPT 消融，并加入 LR-RGDA 统计回放对照，报告 Average/Last/forgetting。
4. 做 LR-RGDA 净贡献消融：LDA、解析 LR-RGDA、LR-RGDA fit、linear probe、nearest class mean、LADA k=16。
5. 在 LoRA-NSP 训练端跑 B0/B2/B3/B4 主实验，验证训练端和推理端收益是否可叠加。

## 4. 论文叙事调整

当前更稳妥的论文主张不是“LR-RGDA 全面优于 LADA”，而是：

- LADA 是强标签记忆基线，在真实特征可用时非常强。
- LR-RGDA 的优势在于统计压缩与回放：当不能保存历史真实特征时，它比 LADA 更适合从轻量分布参数重建判别结构。
- LoRA-NSP 是正交的训练端贡献，目标是提升 CLIP 表征稳定性，并与统计分类器结合。

## 5. 相关文件

- `main_joint.py`: 新增 `--classifier_feature_transform train|test`，用于验证确定性特征构建。
- `scripts/run_joint_classifier_replay.sh`: 支持 `CLASSIFIER_FEATURE_TRANSFORMS` 和 `REPLAY_MODES`。
- `chat-history/2026-06-14-01-joint-gmm-replay-results.md`: 最新 joint GMM 回放结果。
- `chat-history/2026-06-12-01-agent-handoff-lr-rgda-diagnosis.md`: 当前方法论风险与消融优先级。
- `paper_writing/paper-template/paper_draft.tex`: 需要按新目标重写实验叙事和部分方法表述。
