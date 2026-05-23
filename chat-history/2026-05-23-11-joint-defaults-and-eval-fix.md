# 联合微调默认配置调整与中间评估修复

**日期**: 2026-05-23
**会话概况**: 联合微调默认禁用了教师蒸馏，修正了中间评估的设备不一致问题，调整了 LoRA 类型默认值

---

## 1. 关键决策

### 1.1 默认禁用教师蒸馏

- `--reference_dataset` 默认值从 `"flickr8k"` 改为 `""`（空字符串），默认不加载参考数据集
- `load_reference_dataset` 中检查条件改为 `if args.reference_dataset != "flickr8k"` 时跳过
- 如需蒸馏：显式传 `--reference_dataset flickr8k`

### 1.2 `main_joint.py` 默认参数汇总

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--lora_type` | `lora_vanilla` | 最基础 LoRA |
| `--reference_dataset` | `""` | 禁用蒸馏 |
| `--iterations` | 5000 | 总迭代次数 |
| `--alpha` | 0.5 | 集成权重 |
| `--tune_student` | True | 微调模式 |
| `--num_workers` | 6 | DataLoader 进程数 |

### 1.3 中间评估设备修复

`eval_callback` 中 `extract_features` 返回 CPU tensor，需要用 `features.to(zs_for_eval.device)` 将特征移到与分类器相同的设备上。

## 2. 相关文件

- `main_joint.py`: 默认值 + eval_callback 设备修复 + reference_dataset 提示更新
- `src/utils/reference_loader.py`: 跳过条件更新
- `src/trainers/lora_nsp_trainer.py`: EMA + eval_interval/eval_callback 支持
