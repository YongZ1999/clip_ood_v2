# 完整评估流程重构：中间评估与最终评估统一

**日期**: 2026-05-23
**会话概况**: 将训练中的中间评估改为完整评估（含 LR-RGDA 构建和 Ensemble），抽取为共用函数 `run_full_evaluation`

---

## 1. 关键决策

### 1.1 评估流程统一

将提取特征 → 构建 LR-RGDA 分类器 → 评估 ZS/RGDA/Ensemble 的完整流程抽取为独立函数 `run_full_evaluation(eval_model, tag="")`：

- **中间评估**（每 500 步）：`eval_callback` 调用 `run_full_evaluation(current_model, tag="Iter 500")`
- **最终评估**（训练后）：调用 `run_full_evaluation(model, tag="Final")`
- **不微调模式**：调用 `run_full_evaluation(model, tag="No-tune")`

三者使用完全相同的评估逻辑，结果一致可比。

### 1.2 `run_full_evaluation` 函数定义

位置：`main_joint.py` 中 `tune_student` 判断之前，确保 `if` 和 `else` 分支都能调用。

返回值：

```python
return (id_zs, id_rgda, id_ens,           # 每个数据集的准确率列表
        id_zs_avg, id_rgda_avg, id_ens_avg # 平均值
        id_dataset_offset_map,              # 各数据集标签偏移
        num_id_classes,                     # 总类别数
        zs_classifier, lr_rgda_classifier)  # 分类器实例
```

后续 OOD 评估复用这些返回的分类器实例，避免重复构建。

## 2. 解决了的问题

1. **中间评估设备不一致**：`extract_features` 返回 CPU tensor，需 `.to()`
2. **标签偏移未对齐**：各数据集 0-based 标签需加上偏移量才匹配联合分类器
3. **重复的 `else` 分支**：重构过程中残留，已清理
4. **函数体丢失**：删除重复定义时误删，已恢复

## 3. 相关文件

- `main_joint.py`: 主要改动（run_full_evaluation + eval_callback + 结构清理）
- `src/trainers/lora_nsp_trainer.py`: eval_interval/eval_callback 支持
