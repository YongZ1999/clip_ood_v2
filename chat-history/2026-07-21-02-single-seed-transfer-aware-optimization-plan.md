# Transfer-Aware 单 Seed 后续优化计划

**日期**: 2026-07-21
**会话概况**: transfer-aware E1 已在 `preserve_aspect + zs_predicted_seen` 协议下获得更高的 Transfer/Average。用户希望先用单 seed 验证仍可能改进的方向，只有明确有效后才补齐三 seed。

---

## 1. 前提与防止结果选择的规则

- 固定 seed 42 为开发 seed，目标先放在 16-shot LoRA-NF；该阶段是诊断，不作为最终论文报告。
- 在扫描前预先定义选择规则：优先 Ensemble Average，同时要求 Transfer 不低于当前 16-shot LoRA-NF 开发基线超过 0.2 点、Last 不下降；没有候选满足则保留当前配置。
- 选中唯一候选后，才对 LoRA-NF 和 Standard LoRA 补齐 42/43/44 的同协议实验。分类器配置若改变，最终必须同样应用于 Standard LoRA，保证公平比较。

## 2. 当前结果与可复用性

- 当前 transfer-aware E1 LoRA-NF 16-shot Ensemble：`61.91 ± 0.22 / 72.63 ± 0.05 / 83.72 ± 0.10`（T/A/L）。
- 已提交的 12 个正式 run 只有最终 JSON，没有 `--save_step_artifacts` 产生的每任务模型/统计量，不能直接做离线 RGDA sweep。
- 重新训练一个 LoRA-NF 16-shot seed 42，同时开启 `--save_step_artifacts`，即可为全部后续 classifier sweep 提供相同的 10 个编码器快照；不需要为每个 alpha、M 或 fit iteration 重训。

## 3. 推荐顺序

### Phase D1：一个编码器轨迹 + 离线分类器扫描（最高优先级）

- 以当前 transfer-aware 16-shot LoRA-NF 配置重跑 seed 42；保留 inline evaluation/检索，同时加入 `--save_step_artifacts`、固定 `--async_eval_dir`，并保存 `--artifact_num_centers 1,2,4`。
- 对同一套 step artifacts 分阶段扫描：
  1. 当前 MC4 + gmm_sample + 200 iterations 下的 ensemble alpha（`0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20`）；
  2. M=`1,2,4`、fit=`0,50,100,200,400`、source=`gmm_mean/gmm_sample` 的少量预先指定候选；
  3. `zs_predicted_seen` 与 classwise 仅作路由 sanity check，不以降低 Transfer 的候选作为最终配置。
- 现有离线 evaluator 已支持 alpha、M、fit iterations、fit source、routing 和 resize；M=2 需在 artifact run 中显式保存。

### Phase D2：LR-RGDA rank 的离线扩展（次高优先级）

- 16-shot 每类可观测样本有限，当前 rank=32 值得和 8/15/24/32 比较。
- 当前 artifact evaluator 的 variant format 尚未暴露 rank；为此增加一个只影响 evaluator 的 `rank=` variant 字段后，可完全复用 D1 artifacts，不需再训练编码器。

### Phase D3：训练端单 seed（只有 D1/D2 无满意改进时）

- 优先 CD weight `1.0` vs 当前 `2.0`；历史证据显示 CD 是主要分类贡献，同时也可能改变 Transfer–Last 权衡。
- NSP epsilon 只建议小范围 `0.15/0.20/0.25`；历史 E4 已显示 0.20 最优，优先级低。
- 不优先扫 nsp_weight（已有证据不敏感）、训练步数（加长更可能降低 Transfer）或用工程内 prototype head 替代 LR-RGDA（不能称为官方 LADA）。

## 4. 相关文件

- `scripts/evaluate_incremental_rgda_sweep_artifacts.py`
- `main_incremental.py` (`--save_step_artifacts`, `--artifact_num_centers`)
- `scripts/run_transfer_aware_main_table.sh`
- `chat-history/2026-07-21-01-transfer-aware-e1-result-audit.md`
