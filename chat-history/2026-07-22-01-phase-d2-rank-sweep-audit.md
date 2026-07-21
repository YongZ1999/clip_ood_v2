# Phase D2 Rank Sweep 审计

**日期**: 2026-07-22
**会话概况**: 服务器提交 `1578060`（离线 evaluator 加入 rank variant 与 eval seed）和 `4517dc1`（D2 结果）。本地审计 rank 扫描是否带来足以进入多 seed 确认的提升。

---

## 1. 实现核验

- `scripts/evaluate_incremental_rgda_sweep_artifacts.py` 已支持 `rank=` / `r=` variant，且该值实际传给 `LRRGDAClassifier`。
- 新增 `--eval_seed`（默认 42），在 evaluator 启动时固定 Python、NumPy、Torch CPU/CUDA 随机状态。
- D2 使用服务器保存的 D1 artifacts，无新增 LoRA-NF 训练。
- 输出 JSON 的 `args` 仍来自训练 artifact，不包含 `eval_seed`；D2 的 Markdown 记录了 seed，但若此 evaluator 将来用于正式结果，应把 sweep 参数写入 JSON metadata。

## 2. 结果

- D1 matching baseline（M=4/gmm_mean/fit=200/rank=32/alpha=.02）：
  `61.6366 / 72.7520 / 83.9325`。
- D2 的最佳表面候选为 rank=8/alpha=.05：
  `61.6815 / 72.8349 / 84.0744`，相对上述 baseline 为 `+0.045 / +0.083 / +0.142`。
- 但在 alpha=.05 下：rank=8 为 `72.8349`，rank=15/24/32 均为 `72.8307`；rank 的净 Average 差异仅约 `+0.004`。
- rank=15、24、32 在所有 alpha 下完全相同，说明 16-shot 数据下有效低秩已经在约 15 处饱和；rank=8 也只带来极小变化。
- 主要变化来自 alpha=.05 相对 .02 的重加权，而非 rank；两者仍未达到预先规定的 Average `+0.20` 选择门槛。

## 3. 决策

- D2 不提供值得替换当前正式 main-table 配置的稳健候选；不启动新的多 seed 确认。
- D1/D2 已充分覆盖当前 classifier-only 方向（alpha、M、fit、replay source、rank），继续在同一 test seed 上微调风险大于预期收益。
- 当前 transfer-aware E1 的三 seed 配置应保持为正式结果。若仍研究训练端机制，应将 CD/NSP 扫描明确作为新的 ablation/方法开发，而非为了在测试结果上追逐小数点提升。

## 4. 相关文件

- `scripts/evaluate_incremental_rgda_sweep_artifacts.py`
- `experiments/paper_transfer_aware/dev_seed42/sweep_rank/D2_SUMMARY.md`
- `experiments/paper_transfer_aware/dev_seed42/sweep_rank/D2_rank_ens_*.json`
- `chat-history/2026-07-21-03-phase-d1-single-seed-audit.md`
