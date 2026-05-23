# 评估指标对齐与代码重构

**日期**: 2026-05-23
**会话概况**: 对齐 LADA 论文的 Transfer/Average/Last 指标定义，修复代码中不一致的实现，重构函数位置，调整 joint 模式的评估报告方式。

---

## 1. 关键决策

### 1.1 Transfer 指标的精确定义（与 LADA 对齐）

- **Transferₖ** = $\frac{1}{k-1}\sum_{j=1}^{k-1} \hat{a}_k^{(j)}$，衡量任务 k **被学习之前**的平均准确率（前向遗忘）
- 最终报告 **Transfer** = $\frac{1}{K-1}\sum_{k=2}^{K} \text{Transfer}_k$（对 K-1 个任务取均值，任务 1 无定义）
- 来源：LADA 论文 Appendix A + 源码 `result_process.py` 确认

### 1.2 `ContinualLearningMetrics` 是正确的

- `src/utils/continual_metrics.py` 中的 `calculate_transfer/average/last` 与 LADA 精确一致 ✅
- 删除了文件底部两个不属于 LADA 的额外函数：`calculate_forgetting()` 和 `calculate_forward_transfer()`

### 1.3 `main.py` 和 `main_utils.py` 的错误修复

旧代码的错误：
- **Transfer 取对角线**：`matrix[k][k]` 是"学完任务 k 后在该任务上的准确率"，而非"学之前"的平均值
- **Average 只算子三角**：只从 row=j 到 K-1 求均值，而非全部 K 行
- **Transfer 最终值除以 K**：应该除以 K-1（任务 1 无定义）

已统一修正为 LADA 公式。

### 1.4 `main_joint.py` 的评估方式

联合微调没有任务序列概念，不适用 LADA 的 Transfer/Average/Last：
- 改为报告 **ID 准确率**（微调过的数据集）和 **OOD 准确率**（未微调的数据集）
- 不再使用 `get_full_stats` / `print_paper_metrics`
- 添加 `ALL_XTAIL_DATASETS` 列表，`--ood_datasets` 改为可选参数，不传时自动推断为 X-TAIL 补集

### 1.5 函数迁移

`build_stats_dict_from_features` 和 `extract_stats_dict_from_model`：
- 从 `src/detectors/ood_detector.py` 迁移到 `src/classifiers/gaussian_statistics.py`
- 逻辑原因：这两个函数构建 `GaussianStatistics` 对象，与 OOD 检测无关
- `ood_detector.py` 保留向后兼容的 re-export

## 2. 受影响文件清单

| 文件 | 改动 |
|------|------|
| `src/utils/continual_metrics.py` | 删除 `calculate_forgetting()`, `calculate_forward_transfer()` |
| `src/utils/main_utils.py` | 修正 `get_full_stats()` 中的 Transfer/Average 公式 |
| `main.py` | 修正 2 处 inline `get_full_stats` + 1 处 `print_paper_metrics` |
| `main_joint.py` | 移除 LADA 指标改为 ID/OOD 报告；添加 `ALL_XTAIL_DATASETS`；`--ood_datasets` 可选 |
| `src/classifiers/gaussian_statistics.py` | 添加 `build_stats_dict_from_features()`, `extract_stats_dict_from_model()` |
| `src/detectors/ood_detector.py` | 删除上述两函数，保留 re-export |
| `src/experiments/run_continual_learning.py` | 移除 `calculate_forgetting` import 和调用 |
| `src/experiments/run_continual_learning_routing.py` | 同上 |
| `src/experiments/run_continual_learning_routing_v2.py` | 同上 |
| `scripts/run_cached_experiment.py` | import 路径切换到 `gaussian_statistics` |
| `scripts/run_classification_eval.py` | 同上 |
| `scripts/run_ood_detector_eval.py` | 同上 |
| `debug_classifier_router.py` | 同上 |

## 3. 待办事项 / 遗留问题

- [ ] `gaussian_statistics.py` 中新增的 `extract_stats_dict_from_model` 引入了 `from utils_data import ...`，可能导致循环导入，运行时需验证

## 4. 相关文件

- `chat-history/2026-05-23-07-lada-evaluation-metrics.md`: LADA 指标定义详细记录
- `src/utils/continual_metrics.py`: 持续学习指标计算（LADA 对齐）
- `src/utils/main_utils.py`: `get_full_stats()` 修正后的实现
- `main_joint.py`: 修改后的 ID/OOD 评估逻辑
