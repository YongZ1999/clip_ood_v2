# 移除 OOD 检测模块

**日期**: 2026-05-31
**会话概况**: 应研究者要求，从项目中完整移除了 OOD 检测和自适应路由相关的所有代码。

---

## 1. 关键讨论 / 决策

- 研究者的推理端聚焦点从 OOD 检测缓解（自适应路由）转向纯粹的集成分类器（ZS + LR-RGDA 固定 α 融合）
- OOD 检测（`ood_detector.py`）和自适应路由（`adaptive_router.py`）被视为不再必要，全程删除

## 2. 删除/修改的文件

### 完全删除（3 个）
| 文件 | 原因 |
|------|------|
| `src/detectors/ood_detector.py` | OOD 检测器核心实现 |
| `src/routing/adaptive_router.py` | 自适应路由核心实现 |
| `scripts/run_ood_detector_eval.py` | OOD 检测器专用评估脚本 |

### 清理 OOD 引用的文件（6 个）
| 文件 | 改动 |
|------|------|
| `debug_classifier_router.py` | 移除 OOD 和路由的 import |
| `scripts/run_classification_eval.py` | 移除 routing strategy、OOD 检测器参数、OOD 相关代码路径 |
| `scripts/run_cached_experiment.py` | 移除 routing 评估、evaluate_with_routing 函数、OOD 参数 |
| `src/utils/hyperparameter_optimizer.py` | 移除 OOD 检测和路由相关的超参优化代码 |
| `src/experiments/run_continual_learning_routing.py` | 标记为已废弃，移除 broken imports |
| `src/experiments/run_continual_learning_routing_v2.py` | 标记为已废弃，移除 broken imports |

### 更新
| 文件 | 改动 |
|------|------|
| `AGENTS.md` | 移除 B1 基线、OOD 检测相关说明、更新代码结构描述 |

## 3. 重要发现

- `main_incremental.py` 和 `main.py` 原本就已经不依赖 OOD 检测，移除后主训练管线不受影响
- 实验基线从 5 种（B0-B4）减少到 4 种（删除 B1 自适应路由）
- 两个 routing 实验脚本依然保留在仓库中但标记为已废弃，供历史参考

## 4. 待办事项 / 遗留问题

- [ ] 更新 PROJECT_DOCUMENTATION.md 以反映删除
- [ ] 检查服务器上是否还有对 OOD 检测的引用需要同步清理
- [ ] 考虑 `main.py` 行 11 注释是否要更新

## 5. 相关文件

- `AGENTS.md`: 已更新
- `main_incremental.py`: 未受影响
- `main.py`: 未受影响（注释行 11 保留）
