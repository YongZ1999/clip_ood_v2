# LoRA-NF 正式实验实现与证据链审计

**日期**: 2026-07-16  
**范围**: `main_incremental.py`、LoRA-NF 训练/模型/分类器/指标/检索代码、官方 `LADA/` 路径、正式实验 JSON、launch/audit 脚本和实验历史。

---

## 1. 结论摘要

1. LoRA/LoRA-NF 的正式运行参数、任务顺序、训练预算、FD/CD、文本调度、LR-RGDA 与检索数据集在 E1/E3 的结果 JSON 中一致，公平性没有发现配置漂移。
2. `src/utils/retrieval_eval.py` 的图像—标题索引、双向正样本判定和逐任务状态均正确。LoRA-NF 的检索会随任务、方法和 seed 变化，排除了“误把所有模型评成 Frozen CLIP”的主要怀疑。
3. 检索与 Frozen CLIP 基本持平是方法目标可预期的结果：FD/CD 使用冻结 CLIP 作为教师，LoRA-NF 也限制更新。因此它支持“保持对齐”，并不能推出应必然超过 Frozen CLIP 或 LADA。
4. 不能将 Frozen CLIP 当作 LADA 的检索数值。官方 LADA 每个任务训练 AdaptFormer 文本端；I2T 与 T2I 都依赖图像、文本两端的嵌入。现有 checkpoint 未保存逐任务文本适配器，无法离线补算 LADA 的 Retrieval Average/Last。
5. 发现并修复协方差分组归一化错误。该错误影响所有依赖历史协方差的 LoRA-NF / LoRA-Null / Gradient-projected LoRA 实验；旧结果保留作可追溯证据，但不能作为修复后实现的最终数字。

## 2. 已修复问题

| 问题 | 影响 | 修复 |
|---|---|---|
| 所有 hook group 共用 `total_observations`，每组 `X^T X` 被错误除以组数 | NSP/LoRA-Null/Gradient-projected LoRA 的协方差数值尺度错误；与 `build_projection()` 的稳定项结合后可能改变 hard P 切分 | `lora_nsp_trainer.py` 改为每组独立 `observation_counts`；新增 `tests/test_covariance_grouping.py` 回归测试 |
| `build_projection()` 强制把特征分解结果移动到 `cuda` | 无 GPU 时解析 CPU 设备后仍崩溃；与声明的 CPU fallback 矛盾 | 保持在协方差设备；视觉/文本 wrapper 显式将 covariance 移至 adapter 设备 |
| 内联检索 JSON `step` 使用 0--9 | 结果元数据与日志的 Task 1--10 不一致 | `main_incremental.py` 输出改为 1--10 |
| Flickr8K 蒸馏根目录硬编码且不写入运行参数 | 正式实验无法仅从 JSON 复现参考集位置 | 新增 `--reference_root`，默认保持原服务器路径，并更新正式命令模板 |
| 文档把 `Frozen CLIP (= LADA baseline)` 作为检索基线 | 不成立的 LADA 对标/机制表述 | 更正 `docs/paper_experiment_results.md`，将其限定为实际测得的 Frozen CLIP 基线 |

## 3. 检索结果的正确解释

- Frozen CLIP：MSCOCO 5K I2T R@1=52.32，Flickr30K I2T R@1=81.10。
- LoRA-NF 16-shot 三 seed 的 MSCOCO I2T R@1 平均值为 52.07，最终任务为 52.55；数值在冻结教师附近波动，未显示评估器被固定的迹象。
- 因为训练损失显式锚定冻结教师特征和图文相似度，合理的主要假设是**不显著退化**，而不是“必须超过 Frozen CLIP”。超过冻结模型需要针对检索任务的额外监督/目标，当前协议没有该项。
- 正式表可主张 LoRA-NF 相对 Frozen CLIP 保持检索能力；不得主张已经超过或等同官方 LADA 的检索性能，直到 LADA 训练过程内每个任务后直接完成同一评测。

## 4. 实验设计与证据状态

- E1/E3 的 16-shot 配置均使用 42/43/44、800 iterations、rank 4、全层、FD=1、CD=2、T=4、`lada_hybrid`、MC=4、RGDA=200、`alpha=.05`、`maxshift`、COCO 5K + Flickr30K；公平对照成立。
- Full-shot 的 JSON 已记录 `full_shot=true`，该早期元数据问题已修复。
- E4/E5 存在单 seed 扫描；其趋势只能作为单 seed 消融，不能报告为三 seed 显著性结论。协方差修复后也需重跑。
- `experiments/paper_formal/` 没有 `E6_ensemble/` 结果。固定 `alpha=.05` 的主表还没有完成正式计划所要求的 validation-only 选择规则和离线 alpha 消融。
- `chat-history/2026-07-16-01-experiment-completion-summary.md` 关于 “E2/E3 未覆盖检索” 的句子已过期：当前 E2/E3 JSON 中存在逐任务 retrieval 文件。其 “E6=0 训练 runs” 仅表示无需训练，不等于 E6 已完成。

## 5. 重跑与补评估清单

1. 用本次修复后的 commit 重跑 E1 LoRA-NF（16/full，3 seeds）、E2、E3 的 LoRA-Null/GradProj、E4、E5；Standard LoRA 可复用旧结果。
2. 将每次训练的结果写入新目录或在文件名中标识修复后版本，不能覆盖旧 JSON。
3. 给官方 LADA 增加训练过程内的同一检索评估（或保留每个任务的 text-tuner state 后准确重建）；随后才报告 LADA Retrieval Average/Last。
4. 依照正式计划完成 E6：只在 validation protocol 上选择全局 alpha，测试集只评一次固定 alpha；不要用逐任务测试最优 alpha。
5. 在远程 GPU 环境运行 `python tests/test_covariance_grouping.py` 及一个 2-task smoke run，确认 P 构建和新元数据；本地环境缺少 PyTorch，未执行动态测试。

## 6. 本地验证

- 对核心修改文件执行了无导入的 Python 语法编译检查：通过。
- 执行了 `git diff --check`：通过。
- 动态回归测试未在本地执行，原因是当前 Python 环境没有 `torch`；GPU 训练与动态验证须遵循项目规范在远程环境进行。
