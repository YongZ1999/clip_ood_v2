# LoRA-NF 项目核心代码阅读摘要

**日期**: 2026-07-16
**会话概况**: 按用户要求通读项目指南、正式实验结果和增量学习核心实现，梳理 LoRA-NF 从训练、任务后处理、统计分类器构建到评估与检索的完整流程。

---

## 1. 关键发现

- `main_incremental.py` 将每个任务组织为 2a 数据加载、2b adapter 初始化、2c 训练、2d 协方差提取与 merge/reset、2e 特征/统计提取、2f LR-RGDA/LADA 分类器构建、2g 全任务评估、2h 图文检索评估。
- LoRA-NF 的 runtime P 在任务训练期间保持不变：任务 t 使用任务 t-1 结束时的历史协方差构建的 P；任务 t 结束后才把新协方差做等权平均并生成任务 t+1 使用的 P。
- QKV 默认共享输入侧投影对象；协方差提取只对同层 q/k/v 挂一个代表 hook，再把同一协方差复制到 q/k/v 模块。
- `SGPBaseLoRA`/`SGPBaseDoRA` 的 full 模式计算 `B @ A @ P`；`fixed_basis` 计算 `B_basis @ U_h.T`；`core_basis` 计算 `B @ C @ U_h.T`。P 的 identity leakage 由 hard projection 中的 `nsp_weight` 提供满秩表达集。
- LR-RGDA 先从每类特征构建均值/协方差，使用数据集等权全局协方差、低秩 Woodbury 修正和三项正则化；M>1 时对每类多个中心做 log-sum-exp 聚合。评估时将其仅加到全局零样本 logits 的 ID 类区间。
- `ContinualLearningMetrics` 维护完整 K×K 矩阵，未见任务位置保留 -1；Transfer/Average/Last 分别按 LADA 协议计算。

## 2. 当前代码与文档的口径差异/注意事项

- 当前 checkout 为分支 `v4`、提交 `fecacbd`；`docs/paper_experiment_results.md` 记录的代码版本是 `main_v3 (ea74d94)`，正式结果应继续以文档和对应实验 JSON 为证据。
- 当前 `parse_args()` 默认 `lora_rank=4`、`nsp_eps=0.20`、`nsp_weight=0.02`、`lr=1e-4`、`batch_size=32`、`cosine_with_warmup`、`fd_weight=1.0`、`cd_weight=2.0`、`cd_temperature=4.0`、`num_centers=4`、`rgda_train_iter=200`；这与 AGENTS.md 中较早的 FD/CD 表格（1.0/1.0）不同。
- `build_projection()` 当前硬投影先保留协方差谱尾部方向，再用 `(1-nsp_weight)P+nsp_weight I` 泄漏回 identity；它的设备实现固定写入 `cuda`，因此 CPU 路径存在兼容性风险。
- `extract_covariances` 中 `total_observations` 在所有输入分组之间累计，而非每组单独计数；由于各组通常样本数相同，主要表现为整体缩放，谱向量通常不受影响，但实现语义值得后续核验。
- 用户所称的 `LADA/trainer.py` 在当前 checkout 中确实存在；同时还有 `src/lada/lada_trainer.py` 的项目内复现/扩展实现，二者不是同一个训练器。正式基线结果来自 `LADA/trainer.py` 官方代码。

## 3. 实验事实

- 正式结果文档报告：16-shot LoRA-NF Ens Last 为 83.74 ± 0.03，Standard LoRA 为 82.12 ± 0.13；full-shot LoRA-NF Ens Last 为 86.09 ± 0.07。
- LoRA 系列 Ens Last 排序为 LoRA-NF > Gradient-projected LoRA > LoRA-Null > Standard LoRA。
- CD 蒸馏比 FD 更稳定/有效；MSCOCO 5K I2T R@1 上 LoRA-NF 16-shot 与 Frozen CLIP 差距小于 0.3%。

## 4. 相关文件

- `AGENTS.md`: 项目协作、实验和历史记录规范。
- `docs/paper_experiment_results.md`: E1-E5、LADA 复现与检索正式结果。
- `main_incremental.py`: 增量主循环与结果保存。
- `src/trainers/lora_nsp_trainer.py`: AMP、SCE、FD/CD、协方差历史与 merge/reset。
- `src/models/lora_sgp.py`: LoRA/DoRA、P、basis 和 NSP 谱投影。
- `src/classifiers/lr_rgda_classifier.py`、`src/classifiers/gaussian_classifier.py`: LR-RGDA 与集成分类器。
- `src/utils/continual_metrics.py`: LADA 指标矩阵与汇总。
- `LADA/trainer.py`: 官方 AdaptFormer + DPT + label-specific memory 基线。
