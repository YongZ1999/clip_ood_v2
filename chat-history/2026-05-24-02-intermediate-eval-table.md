# 中间评估添加格式化总表输出

**日期**: 2026-05-24
**会话概况**: main_joint.py 的中间评估（每500步）缺少格式化总表输出，修复后统一打印紧凑表格。

---

## 1. 修改内容

| 文件 | 改动 |
|------|------|
| `main_joint.py` | 在 `run_full_evaluation()` 函数中增加格式化表格打印（带框线的 Dataset/ZR/RGDA/Ensemble 表），替换原有的松散逐行日志 |

## 2. 改动说明

- `run_full_evaluation()` 在 per-dataset logging + average 行之后，增加了一段 `print()` 输出表格
- 中间评估（`eval_callback` 每500步调用）和最终评估共用此函数，所以现在都会打印表格
- 最终评估还会额外打印原有的 JOINT FINE-TUNING RESULTS 总表（含 ID + OOD 分开的部分），无冗余冲突

## 3. 相关文件

- `main_joint.py`: 主要改动
