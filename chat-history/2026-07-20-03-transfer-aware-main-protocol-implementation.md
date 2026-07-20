# Transfer-aware 主实验协议实现

**日期**: 2026-07-20
**会话概况**: 为缓解 LoRA-NF 高 Last、相对较低 Transfer 的现象，实现了标准 CLIP 分类测试预处理与按零样本预测的 LR-RGDA 门控融合，并提供了独立分支上的主实验六卡启动器。

---

## 1. 关键决策

- 新实现位于 Git 分支 `agent/transfer-gated-ensemble`，不修改 `v4` 分支的默认行为。
- `--eval_resize_mode legacy_square` 继续是默认值，严格复现历史结果；新正式实验显式传入 `--eval_resize_mode preserve_aspect`。
- `--ensemble_routing classwise` 继续是默认值；新正式实验显式传入 `--ensemble_routing zs_predicted_seen`。
- 本轮保持已经验证的 LR-RGDA 参数 `M=4/rank=32/fit=200/gmm_sample/alpha=0.05`。不在没有验证的情况下同时改 rank、replay source 与 fit iterations。

## 2. 实现内容

### 标准 CLIP 分类测试预处理

- `src/utils/data.py:get_transforms` 新增 `test_resize_mode`：
  - `legacy_square`: `Resize((224,224)) -> CenterCrop`；
  - `preserve_aspect`: `Resize(224) -> CenterCrop(224)`。
- `main_incremental.py` 新增 `--eval_resize_mode`，所有预构建的 X-TAIL test loader 使用该值。
- 该设置只影响分类测试 transform，不改变训练 augmentation 或检索 evaluator。

### ZS-predicted-seen 门控融合

- `src/utils/main_utils.py:combine_ensemble_logits` 新增 `routing` 参数。
- `classwise` 维持原公式：每张样本的已见类列都叠加 LR-RGDA。
- `zs_predicted_seen` 使用 `argmax(raw ZS logits) < current_num_classes` 作为逐样本 gate：
  - ZS 预测未见类时保留纯 ZS 预测；
  - ZS 预测已见类时使用旧 classwise ensemble。
- 门控不访问样本真实标签，也不假设测试集任务身份。
- 主评估、批量评估和 artifact 离线 sweep evaluator 都支持该字段；`EnsembleClassifier` 同步实现相同行为。

### 可复现实验启动器

- 新增 `scripts/run_transfer_aware_main_table.sh`。
- 默认运行 12 个 E1 主实验任务：Standard LoRA 与 LoRA-NF，分别在 16-shot/full-shot 和 seeds 42/43/44 下运行。
- 六张 GPU 的首波并发 6 个任务，其余 6 个由各 worker 自动顺序接续。
- 设置 `RUN_STANDARD_LORA=0` 可只跑 LoRA-NF 的 6 个诊断任务，但不应用于方法间公平比较。
- 结果保存至 `experiments/paper_transfer_aware/E1_main/`，日志保存至 `logs/paper_transfer_aware/`，不会覆盖历史 `paper_formal`。

## 3. 验证

- `git diff --check` 通过。
- `PYTHONPYCACHEPREFIX=/private/tmp/clip_ood_v2_pycache python3 -m compileall ...` 通过。
- `bash -n scripts/run_transfer_aware_main_table.sh` 通过。
- 新增 `tests/test_transfer_aware_ensemble.py`，覆盖：未见 ZS 预测严格保留、已见 ZS 预测沿用 classwise、两种 resize mode 的 transform 形状。
- 本机 Python 环境不含 `torch`/`torchvision`，故数值单测只能在服务器的训练环境运行。

## 4. 待办事项

- [ ] 在服务器运行 `python tests/test_transfer_aware_ensemble.py`。
- [ ] 运行 12 个 transfer-aware E1 任务并汇总 ZS/Ensemble Transfer、Average、Last 与检索 Average/Last。
- [ ] 使用新 Standard LoRA 与新 LoRA-NF 结果更新公平内部比较；保留旧 `paper_formal` 结果作为历史协议记录。
- [ ] LADA 绝对数值比较前核验类别、模板和预处理协议。

## 5. 相关文件

- `src/utils/data.py`
- `src/utils/main_utils.py`
- `src/classifiers/lr_rgda_classifier.py`
- `main_incremental.py`
- `scripts/evaluate_incremental_rgda_sweep_artifacts.py`
- `scripts/run_transfer_aware_main_table.sh`
- `docs/transfer_aware_main_experiment.md`
- `tests/test_transfer_aware_ensemble.py`
