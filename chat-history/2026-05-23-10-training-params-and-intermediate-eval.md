# 训练参数与中间评估功能新增

**日期**: 2026-05-23
**会话概况**: 添加训练过程 EMA 损失记录、每 500 步中间评估、联合微调默认 LoRA 类型改为 lora_vanilla

---

## 1. 关键决策

### 1.1 损失函数与准确率滑动平均

训练日志中的 Loss 和 Acc 由单 batch 瞬时值改为 **EMA（指数滑动平均，动量 0.95）**，使趋势更平滑稳定。

### 1.2 每 500 步中间评估

`LoRANSPTrainer.train()` 新增 `eval_interval` 和 `eval_callback` 参数：
- 每 500 步调用回调函数，传入当前模型和 step 编号
- 回调中执行轻量零样本评估，输出每个 ID 数据集准确率和均值
- 评估完后模型切回训练模式继续训练

### 1.3 联合微调默认 LoRA 类型

| 参数 | 旧值 | 新值 |
|------|:----:|:----:|
| `--lora_type` 默认 | `lora_nsp` | **`lora_vanilla`** |
| `choices` | `["lora_sgp", "lora_nsp"]` | `["lora_vanilla", "lora_sgp", "lora_nsp"]` |

三种 LoRA 类型：
- **`lora_vanilla`** — 最基础 LoRA
- `lora_sgp` — LoRA + SGP
- `lora_nsp` — LoRA + 零空间投影（NSP）

### 1.4 训练 DataLoader

`num_workers` 从默认 0 改为 6，加速数据加载。

### 1.5 训练相关参数汇总

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--batch_size` | 32 | 训练 batch 大小 |
| `--lr` | 1e-4 | 学习率，AdamW + CosineAnnealingLR 到 lr/3 |
| `--iterations` | 5000 | 总迭代次数 |
| `--num_workers` | 6 | 数据加载进程数 |
| `--lora_rank` | 4 | LoDA 秩 |
| `--alpha` | 0.5 | 集成分类器权重 |

## 2. 相关文件

- `src/trainers/lora_nsp_trainer.py`: EMA 记录 + eval_interval/eval_callback
- `main_joint.py`: 默认 lora_vanilla + 中间评估回调 + num_workers=6

## 3. 待办事项

- [ ] `main_incremental.py` 的 lora_type choices 尚未同步更新（当前只有 `["lora_sgp", "lora_nsp"]`）
