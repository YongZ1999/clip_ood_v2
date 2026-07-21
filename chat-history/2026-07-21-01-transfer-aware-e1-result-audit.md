# Transfer-Aware E1 结果拉取与独立审计

**日期**: 2026-07-21
**会话概况**: 服务器已将 transfer-aware E1 的 12 个完整训练结果推送为 commit `a0bb9b5`。本地拉取后直接从原始 JSON 独立重算并核验分类矩阵、配置与检索记录。

---

## 1. 完整性与配置核验

- 结果目录 `experiments/paper_transfer_aware/E1_main/` 包含 12 个 ZS、12 个 Ensemble、12 个 RGDA 和 12 个 retrieval JSON；对应 LoRA-NF/Standard LoRA × 16-shot/full-shot × seeds 42/43/44。
- 每个结果的 Accuracy Matrix 以 `ContinualLearningMetrics` 的列均值定义独立复算后，与 JSON 的 Transfer/Average/Last 完全一致（36 个 classification JSON 均通过）。
- 12 个运行均记录：OpenAI CLIP ViT-B/16、800 iterations、`preserve_aspect` 分类测试预处理、`zs_predicted_seen` 融合门控、`alpha=0.05`、RGDA rank=32/200 iterations、开启检索评估；LoRA-NF 记录 `nsp_eps=0.2/nsp_weight=0.02/CD=2.0`。
- Standard LoRA 16-shot seed 44 首次被外部 `raoxuan` 进程 OOM 干扰，随后在 GPU 0 成功重跑；结果文件齐全。其余 11 个任务首次完成。

## 2. 主要分类结果（sample std，n=3）

| Method | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|
| LoRA-NF 16-shot | 61.91 ± 0.22 | 72.63 ± 0.05 | 83.72 ± 0.10 |
| Standard LoRA 16-shot | 61.59 ± 0.15 | 71.75 ± 0.34 | 82.04 ± 0.29 |
| LoRA-NF full-shot | 61.89 ± 0.15 | 74.98 ± 0.01 | 86.19 ± 0.02 |
| Standard LoRA full-shot | 61.56 ± 0.33 | 74.33 ± 0.11 | 84.90 ± 0.06 |

- LoRA-NF − Standard LoRA（Ensemble）：16-shot `+0.32 / +0.89 / +1.68`，full-shot `+0.33 / +0.65 / +1.29`（Transfer/Average/Last）。
- 对 LoRA-NF，门控 Ensemble 相对 ZS：16-shot `+0.37 / +1.35 / +2.84`，full-shot `+0.41 / +1.57 / +2.48`。
- 该新协议相对历史 `paper_formal` 16-shot LoRA-NF Ensemble（60.14/71.75/83.74）为 61.91/72.63/83.72；Transfer 和 Average 更高，Last 基本持平。但这同时改变了评估 resize 与 routing，不能将差异归因于单个组件，也不能与旧协议的 LoRA-Null/GradProj 直接混排。

## 3. 检索核验

- 所有 12 个 run 都有 10 个任务后的 MSCOCO 5K I2T/T2I R@1/5/10 记录。
- R@1 均值：LoRA-NF 16-shot `51.97/33.30`（I2T/T2I），Standard LoRA `51.83/33.28`；full-shot LoRA-NF `52.15/33.53`，Standard LoRA `52.00/33.43`。
- NF 相对 LoRA 的检索差异均很小（R@1 的平均差最多约 0.15），支持“未见显著检索退化”，但不应把小幅的双向差异表述为所有方向均严格改善。

## 4. 汇总文件注意事项

- 服务器生成的 `TRANSFER_AWARE_SUMMARY.md` 采用 sample standard deviation（论文表格应使用该值）。
- 同目录 `TRANSFER_AWARE_SUMMARY.json` 的 `std` 使用 population standard deviation，因此两者不一致。原始三 seed JSON 正确；后续更新论文/正式结果文档时应统一使用 sample std，并避免直接引用该 JSON 的 `std` 字段。

## 5. 相关文件

- `experiments/paper_transfer_aware/E1_main/TRANSFER_AWARE_SUMMARY.md`
- `experiments/paper_transfer_aware/E1_main/TRANSFER_AWARE_SUMMARY.json`
- `experiments/paper_transfer_aware/E1_main/E1TA__*_{zs,ens,rgda}_results.json`
- `experiments/paper_transfer_aware/E1_main/E1TA__*_retrieval.json`
