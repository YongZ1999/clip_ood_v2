# Transfer 偏低的超参数与评估协议诊断

## 用户问题

LoRA-NF 当前 Last 很高，但 Transfer 不够高。需要判断能否通过 ensemble alpha、RGDA 拟合迭代数，或改用 LADA 分类器缓解。

## 主要结论

1. 当前 LoRA-NF 并没有明显破坏其自身的 Frozen CLIP 前向迁移能力。
   - Frozen CLIP ZS Transfer：60.37。
   - LoRA-NF ZS Transfer：59.84，下降 0.53。
   - LoRA-NF Ensemble Transfer：60.14，下降 0.23。
   - 因此，与 LADA 原文 61.5 的绝对差距不能全部归因于持续训练遗忘。

2. 当前工程与官方 LADA 的 CLIP 测试预处理不一致，是优先级高于超参数扫描的混杂因素。
   - 当前 `src/utils/data.py` 使用 `Resize((224, 224))`，直接把图像拉伸为正方形。
   - 官方 `LADA/trainer.py` 使用 `Resize(224) + CenterCrop(224)`，保持宽高比，符合标准 CLIP 评估方式。
   - 两边 Frozen CLIP 差异主要集中在 Stanford Cars、Oxford Pets、Flowers 等数据集；尤其 Cars 的 Frozen 基线约差 5.2 个点。
   - 应先做无需训练的 Frozen CLIP 双预处理对照，确认基线差距，再决定是否重跑正式主表。

3. `alpha` 只能沿 Transfer–Average–Last 的 Pareto 曲线小幅移动，不能单独补足约 1.4 个点的绝对 Transfer 差距。
   - 现有旧实验中，alpha 从 0.05 增至 0.10/0.20/0.50，Transfer 仅增加约 0.03–0.06，但 Average 和 Last 持续下降。
   - 正式配置 alpha=0.05 已接近现有证据下的综合最优点。

4. RGDA 训练迭代主要优化已见类判别，对未来未见类没有直接学习信号。
   - Transfer 在任务尚未学习时评估未来类别；LR-RGDA 只有已见类统计量。
   - 增加 `rgda_train_iter` 更可能提高 Average/Last，并加强 seen-class bias，而不是实质提高 Transfer。
   - 16-shot 下每类可观测类内协方差秩最多约 15，默认 rank=32 值得重新验证。

5. 当前 `main_incremental.py` 中的 `LADAClassifier` 不能等同于官方完整 LADA。
   - 它使用所有历史真实 16-shot 特征重建 label-specific prototypes，并可在这些真实特征上拟合。
   - 它不包含官方 LADA 的完整 AdaptFormer、DPT 和原生逐任务训练过程。
   - 直接替换为该分类器会改变历史样本记忆假设，而且更可能提升 seen-task Average/Last，不保证提升 Transfer；论文中只能称为 prototype-head diagnostic，不能称为官方 LADA。

6. 官方 LADA 的一个重要行为是条件性使用其分类头：当零样本预测落在未见类时保留 ZS；落在已见类时才使用 LADA logits。当前固定 LR-RGDA ensemble 会对所有样本施加融合。仿照该行为可能更直接保护 Transfer，但属于新增路由/门控组件，需要单独消融和验证集选参，不能事后按测试集挑结果。

## 现有组件结果揭示的权衡

- 无蒸馏：Transfer 60.71，Average 71.19，Last 83.21。
- 完整 FD+CD：Transfer 60.14，Average 71.75，Last 83.74。
- 蒸馏提高 Average/Last，但现有结果中约牺牲 0.57 Transfer。
- LoRA-NF 相比 Standard LoRA 的 Ensemble Transfer 已提高约 0.30，说明 NF 的保护方向有效，但仍存在稳定性–可塑性权衡。

## 推荐实验顺序

### Phase 0：零训练成本的预处理核验

为 CLIP 分类评估加入显式 `square` / `aspect_preserving` resize 模式，先只跑 Frozen CLIP。若标准宽高比预处理明显恢复 Cars/Pets 和整体 Transfer，再决定是否以统一协议重跑分类主表。

### Phase 1：固定一个编码器轨迹，离线扫描分类器

先用一个代表性 seed 保存每任务 step artifacts，再离线扫描：

- alpha：0、0.02、0.05、0.10、0.20、0.50；
- RGDA train iterations：0、50、100、200、400；
- fit source：none、gmm_mean、gmm_sample；
- covariance rank：8、15、32；
- 可额外加入 LADA-style prototype head 作为诊断，但须明确其不是官方 LADA。

优先进行分阶段扫描而不是完整笛卡尔积，并用预先定义的验证协议选参；最终报告 Transfer/Average/Last Pareto，而不是依据测试 Transfer 反向挑参数。

### Phase 2：必要时调整训练侧

若分类器扫描仍无法得到满意折中，再验证 CD weight（0/1/2）、视觉学习率或视觉更新步数。现有结果已经表明降低蒸馏可提高 Transfer，但会牺牲 Average/Last，因此应作为明确的权衡实验。

## 本次代码状态

本次只做诊断和实验设计记录，没有修改训练、评估或正式实验配置。
