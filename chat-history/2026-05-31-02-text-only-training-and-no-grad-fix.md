# 仅文本编码器训练 + 视觉侧 no_grad 修复

**日期**: 2026-05-31
**会话概况**: 分析了 `main_joint.py` 在 `--tune_text_encoder True`（仅文本编码器可训）下的行为，发现视觉编码器未加 `no_grad()` 导致显存浪费，并进行了修复。

---

## 1. 关键讨论 / 决策

### 1.1 仅文本编码器训练的行为分析

命令：`python3 main_joint.py --tune_text_encoder True --id_datasets [全部10个] --lora_type lora_vanilla --alpha 0.5 --gpu 1`

| 环节 | 行为 |
|------|------|
| **视觉编码器** | 冻结（`has_vision_lora=False`），但未加 `no_grad()` |
| **文本编码器** | 挂载 LoRA（`has_text_lora=True`），每步重新编码 ZS 分类器 |
| **~1100 类处理** | 每步随机采样 ~128 类，只在采样类上计算 loss，其余 mask |
| **FD 蒸馏** | 视觉侧冻结 → student 特征与 teacher 完全相同 → FD ≈ 0，无效 |
| **CD 蒸馏** | 文本侧被 LoRA 修改 → CD 约束文本编码器不破坏跨模态对齐 → **唯一有效** |
| **LR-RGDA** | 基于冻结的 CLIP 视觉特征构建，与原始 CLIP 无异 |

### 1.2 视觉编码器 `no_grad()` 缺失问题

两个位置缺少 `no_grad()`：
1. 主前向（`vision_model(images)`）
2. 蒸馏前向（`vision_model(r_imgs)`）

固定 `has_vision_lora=False` 时，PyTorch 仍会：
- 保存所有 ViT 层中间激活值（浪费显存）
- 构建完整计算图，反向传播时计算全零梯度后丢弃

### 1.3 决定：条件式 no_grad

在 `train()` 方法中，对视觉编码器前向添加条件判断：
- `has_vision_lora=True` → `torch.enable_grad()`（梯度回传 LoRA）
- `has_vision_lora=False` → `torch.no_grad()`（跳过激活存储）

文本编码器同理但已天然正确处理（`has_text_lora=False` 时使用预计算的 ZS 分类器和 teacher 文本特征，不调用 text encoder）。

## 2. 修改的文件

`src/trainers/lora_nsp_trainer.py`

- **L348-352**: 主前向添加 `vision_ctx`，条件 `no_grad` / `enable_grad`
- **L424-428**: 蒸馏前向添加 `vision_ctx_ref`，同理

## 3. 待办事项 / 遗留问题

- [ ] 仅文本编码器训练的实验价值需验证（预期 ID 提升有限）

## 4. 相关文件

- `src/trainers/lora_nsp_trainer.py` (modified)
- `main_joint.py` (analyzed, not modified)
