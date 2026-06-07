# ensemble_alpha 诊断实验：评估不一致问题排查

**日期**: 2026-06-07
**会话概况**: 排查 5/31（main_joint.py RGDA 69.2%）与 6/7（debug_ensemble_alpha.py RGDA 78.8%）的评估不一致问题，定位根因并修复。

---

## 1. 问题

5/31 用 `main_joint.py --iterations 0` 评测 RGDA(分析) = 69.2%，6/7 用 `debug_ensemble_alpha.py` 评测 A:1c分析 = 78.8%。
两个脚本使用相同的 frozen CLIP、相同的特征提取、相同的 LRRGDAClassifier 构建参数、相同的 task-agnostic per-dataset 平均评估，但结果差 10%。

## 2. 排查过程

### 2.1 `global_cov` 假设（排除）
- 假设：`fb9be96` 提交引入的 `dataset_balanced_global_cov` 可能提升了精度
- 测试：per-class average vs per-dataset balanced global_cov 对比
- 结果：**两者完全一致（68.9%），差异 < 0.1%**，`global_cov` 对精度无影响

### 2.2 训练数据随机增强假设（部分成立）
- 假设：`RandomResizedCrop` + `RandomHorizontalFlip` 导致每次特征提取不同
- 测试：TRAIN transform vs TEST transform 用于分类器构建
- 结果：TRAIN transform 68.9% / TEST transform 72.6%，差 3.7%
- 多次 TRAIN transform 运行**完全一致**（68.9%），说明随机增强在固定 seed 下是确定性的
- 但即使 TEST transform 也只有 72.6%，无法解释到 78.8% 的全部差距

### 2.3 解包顺序 Bug（根因）
- **发现**：`debug_ensemble_alpha.py:219` 的解包顺序错误
  ```python
  # buggy:
  tr_loader, te_loader, _, c_names = get_xtail_trainloader(...)
  
  # fixed:
  tr_loader, _, te_loader, c_names = get_xtail_trainloader(...)
  ```
- `build_cur_task_data_loader` 返回 `(train_loader, train_loader4updating, test_loader, classnames)`
- buggy 版本中 `te_loader` 实际拿到的是 `train_loader4updating`（16-shot 训练数据 + test_transform）
- 分类器在 16-shot 训练数据上评估（而非完整测试集），导致"虚高"约 10%

### 2.4 验证
- 编写独立测试脚本，严格复现 `main_joint.py` 的评估流程
- TRAIN transform 多次运行结果完全一致：68.9%
- 完美匹配 5/31 的 `main_joint.py` 结果（69.2%，差 0.3% 在正常波动范围内）

## 3. 核心结论

| 评估数据 | RGDA 分析版 | 来源 |
|---------|-----------|------|
| 16-shot 训练集（test transform） | ~78.8% | buggy `debug_ensemble_alpha.py` |
| 完整测试集 | ~68.9% | `main_joint.py` / 独立测试脚本 |

**根因**：`debug_ensemble_alpha.py` 一行解包顺序错误，导致用 16-shot 训练数据代替完整测试集进行评估。

**修复**：`debug_ensemble_alpha.py:219` 改为 `tr_loader, _, te_loader, c_names = get_xtail_trainloader(...)`

## 4. 重要教训

- `get_xtail_trainloader` 返回 4 个值，第 2 个是 `train_loader4updating`（训练数据 + test_transform），第 3 个才是 `test_loader`（完整测试集）
- `main_utils.py` 中的 `evaluate_dataset` 早期也有同样的解包 bug，在 `fb9be96` 中修复
- 训练数据上的评估不能代表测试集性能，差约 10%
- 分类器构建用 TRAIN transform（有随机增强）vs TEST transform（无增强）差 ~3.7%

## 5. 相关文件

- `debug_ensemble_alpha.py:219`: 修复位置
- `src/utils/main_utils.py:93`: `evaluate_dataset` 的正确解包（`_, _, te_loader, c_names`）
- `scenario_datasets/build_functions.py:45-76`: `build_cur_task_data_loader` 返回值说明
- `utils_data.py:49-59`: `get_xtail_trainloader` 封装
- `debug_cov_compare.py`: global_cov 对比测试脚本
- `debug_transform_test.py`: transform 对比测试脚本
