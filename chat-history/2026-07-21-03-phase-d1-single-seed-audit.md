# Phase D1 单 Seed 分类器扫描审计

**日期**: 2026-07-21
**会话概况**: 服务器提交 `b6b0849`，包含一条 LoRA-NF 16-shot seed 42 的 artifact 训练轨迹，以及 alpha 与 M/fit/source 离线扫描。本地审计精确 JSON 后评估是否存在足以进入多 seed 确认的候选。

---

## 1. 完整性

- 一次完整 10-task LoRA-NF 训练完成，并保存了 10 个 queue ready records；服务器本地保留 artifact.pt 以供后续离线分析，未将 checkpoint 上传 GitHub。
- 产生 alpha sweep（8 个 alpha）与 M/fit/source sweep（M=1/2/4，fit=0/50/100/200，gmm_mean/gmm_sample）的完整 JSON。
- 开发 run 的 inline Ensemble 为 `61.6788 / 72.7247 / 83.8013`（Transfer/Average/Last），与原 E1 seed-42 的微小差异属单独训练轨迹，不能直接将旧 run 的 rounded baseline 当作该 sweep 的精确基线。

## 2. 结果

- Alpha=0.02 与当前 alpha=0.05 的 matching offline results 几乎持平：
  - alpha 0.02：`61.6381 / 72.6962 / 83.8389`；
  - alpha 0.05：`61.6828 / 72.6895 / 83.8298`。
- 精确 Average 最好的是 `M=4, gmm_mean, fit=200, alpha=0.02`：
  `61.6366 / 72.7520 / 83.9325`。
- 与匹配的 M=4/gmm_sample/fit=200/alpha=.02 离线结果相比，该候选为：
  `-0.004 / +0.047 / +0.141`；与 alpha=.05 当前 offline baseline 相比约为
  `-0.046 / +0.062 / +0.103`。
- 服务器总结推荐的 M=1/gmm_mean（`61.6673 / 72.7487 / 83.9072`）并非精确 Average 的最高者；它仅在四舍五入后与 M=4 均显示 72.75，且 Transfer 略高。

## 3. 决策

- D1 未满足事先规定的选择门槛：Average 至少 +0.20，同时保护 Transfer/Last。因此不进入三 seed 确认，也不应将 M=1 或 M=4/gmm_mean 写为新正式配置。
- `gmm_mean` 对比 `gmm_sample` 有轻微正向信号，但幅度太小，且 offline evaluator 没有全局固定 RNG；`gmm_sample` 使用 `torch.randn`，因此亚 0.1 点差异不能当作稳健优化结论。
- 下一步应进行 D2，但先给 evaluator 加入确定性 evaluation seed 和 `rank=` variant 支持，再直接复用服务器保留的 D1 artifacts 扫描 rank=8/15/24/32；不需再训练编码器。

## 4. 相关文件

- `experiments/paper_transfer_aware/dev_seed42/D1_SUMMARY.md`
- `experiments/paper_transfer_aware/dev_seed42/sweep_alpha/`
- `experiments/paper_transfer_aware/dev_seed42/sweep_mft/`
- `scripts/evaluate_incremental_rgda_sweep_artifacts.py`
