# LADA 集成实现与 Text-Only 训练支持

**日期**: 2026-05-31
**会话概况**: 实现了 LADA 分类器的完整集成（包括 DPT），添加了 text-only 训练模式，修复了多个 LoRA 包装模型的兼容性问题。

---

## 1. 关键讨论 / 决策

### 1.1 Text-Only 训练模式
- **背景**: 现有两种模式：vision-only（只微调视觉编码器）和 both（双塔都微调）
- **决策**: 添加第三种模式 text-only（只微调文本编码器，视觉编码器冻结）
- **实现**: 在 `src/models/clip.py` 添加 `tune_vision_encoder` 条件判断
- **参数**: `--tune_vision_encoder True/False`（默认 True）

### 1.2 LADA 集成方案选择
- **讨论**: 三种方案对比
  - 方案 A: 独立并行（LADA 和 LR-RGDA 独立评估）
  - 方案 B: 集成三分支（LADA + LR-RGDA + ZS）
  - 方案 C: LADA 替代 LR-RGDA
- **决策**: 选择方案 A（独立并行），便于消融实验和正交性论证
- **理由**: 可以分别评估各组件贡献，符合论文的正交性主张

### 1.3 LADA 与官方实现对齐
- **发现**: 官方 LADA 在推理时使用 mask 机制
  ```python
  # 官方实现
  if text_logits.argmax < lada_classes:  # ZS 认为是 ID
      output = text_logits + α * lada_logits
  else:  # ZS 认为是 OOD
      output = text_logits  # 不加 LADA
  ```
- **决策**: 对齐官方实现，添加 mask 机制
- **实现**: 在训练循环和评估函数中都添加了 mask

### 1.4 DPT（Distribution-Preserved Training）
- **决策**: 实现完整的 DPT 机制
- **功能**: 
  - 用 GMM 拟合旧类特征分布
  - 训练新任务时采样幻影数据进行蒸馏
  - 防止新任务覆盖旧类决策边界

---

## 2. 重要发现

### 2.1 LoRA 包装模型的兼容性问题
**问题**: `model.get_image_features()` 和 `model.get_text_features()` 在 LoRA 包装后返回 `BaseModelOutputWithPooling` 对象，而非 tensor。

**影响范围**:
- `src/utils/feature_extractor.py` 的 `extract_features()`
- `src/trainers/lora_nsp_trainer.py` 的 `encode_text()`
- `src/utils/main_utils.py` 的 `get_zeroshot_classifier()`

**解决方案**: 手动提取特征，绕过 `get_*_features()` 接口
```python
# 图像特征
vision_outputs = self.model.vision_model(images)
pooled = vision_outputs[1]
proj_feats = self.model.visual_projection(pooled)
feats = proj_feats / proj_feats.norm(dim=-1, keepdim=True)

# 文本特征
text_outputs = self.model.text_model(**text_inputs)
pooled = text_outputs.pooler_output
text_features = self.model.text_projection(pooled)
```

### 2.2 text_prototypes 维度不匹配 Bug
**问题**: `text_protos` 存储为 `(C_prev, D)`，`live_text_feats` 为 `(D, C_curr)`，直接 `cat` 会报错。

**修复**: 转置 `text_protos` 为 `(D, C_prev)` 后再拼接
```python
text_protos = self.dpt.text_prototypes.detach().t()  # (D, C_prev)
full_text_classifier = torch.cat([text_protos, live_text_feats], dim=1)
```

### 2.3 main_joint.py 默认行为变更
**问题**: 添加 `--tune_vision_encoder` 参数后，默认值设为 `False`，改变了原有行为。

**影响**: `main_joint.py` 现在默认双塔都不微调，需要显式传参才能恢复原行为。

**建议**: 用户需要显式传 `--tune_vision_encoder True` 来微调视觉编码器。

---

## 3. 实现细节

### 3.1 新增文件
```
src/lada/
├── __init__.py              # 模块导出
├── lada_classifier.py       # LADA 分类器（k-means + 指数亲和）
├── lada_trainer.py          # LADA 训练器（继承 LoRANSPTrainer）
└── dpt.py                   # Distribution-Preserved Training（GMM 采样）

main_incremental_lada.py     # LADA 增量学习入口脚本
```

### 3.2 修改的现有文件
| 文件 | 改动 | 影响 |
|------|------|------|
| `src/models/clip.py` | 添加 `tune_vision_encoder` 条件 | 无（默认 True） |
| `main_incremental.py` | 添加 `--tune_vision_encoder` 参数 | 无（默认 True） |
| `main_joint.py` | 添加 `--tune_vision_encoder` 参数 | ⚠️ 默认改为 False |
| `src/trainers/lora_nsp_trainer.py` | 添加 `has_vision_lora` + 修复 `encode_text` | 无（更健壮） |
| `src/utils/main_utils.py` | 修复 `get_zeroshot_classifier` | 无（兼容两种模型） |

### 3.3 LADA 核心公式
```python
# 构建 LADA features（k-means 聚类）
lada_features = kmeans(train_features, k=16)  # (D, K_total)

# 指数亲和变换
affinity = image_features @ lada_features  # (N, K_total)
lada_logits = exp(-β * (1 - affinity)) @ joint_classifier  # (N, C_total)

# 集成（带 mask）
mask = (text_preds < lada_classes).float().unsqueeze(1)
total_logits = text_logits + mask * α * lada_logits
```

### 3.4 DPT 采样公式
```python
# GMM 拟合旧类分布
gmm = GaussianMixture(n_components=k, covariance_type='spherical')
gmm.fit(old_class_features)

# 采样幻影数据
p̃ = p + e * sqrt(Tr(Σ) / d), e ~ N(0, I)
```

---

## 4. 待办事项 / 遗留问题

### 4.1 已完成
- [x] 实现 LADA 分类器核心（k-means + 指数亲和）
- [x] 实现 LADA 训练器（继承 LoRANSPTrainer）
- [x] 实现 DPT（GMM 拟合 + 采样）
- [x] 添加 mask 机制（对齐官方实现）
- [x] 创建 `main_incremental_lada.py` 入口脚本
- [x] 修复 LoRA 包装模型的兼容性问题
- [x] 修复 text_prototypes 维度不匹配 bug
- [x] 添加 text-only 训练模式支持

### 4.2 待完成
- [ ] 完整测试 LADA 增量学习流程（2 个任务，10+ iterations）
- [ ] 验证 DPT 蒸馏效果
- [ ] 对比 LADA vs LR-RGDA 性能
- [ ] 评估 mask 机制对 X-TAIL 的影响
- [ ] 考虑是否需要将 `main_joint.py` 的默认值改回 `True`

### 4.3 已知问题
- 测试运行时进程卡住（CPU 时间 430 分钟），原因未明
- 可能是 NSP 协方差提取或投影矩阵更新耗时过长
- 需要进一步调试

---

## 5. 运行示例

### 5.1 LADA 增量学习（基础版，无 DPT）
```bash
python3 main_incremental_lada.py \
    --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --num_shots 16 --iterations 800 \
    --lora_type lora_nsp \
    --lada_k 16 --lada_beta 1.0 --lada_alpha 1.0
```

### 5.2 LADA 增量学习（完整版，含 DPT）
```bash
python3 main_incremental_lada.py \
    --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
    --num_shots 16 --iterations 800 \
    --lora_type lora_nsp \
    --lada_k 16 --lada_beta 1.0 --lada_alpha 1.0 \
    --enable_dpt --prototype_k 4 --image_prototypes_weight_coef 64.0
```

### 5.3 Text-Only 训练
```bash
# 增量学习
python3 main_incremental.py \
    --tune_vision_encoder False --tune_text_encoder True \
    --dataset_sequence aircraft caltech101 ... \
    --num_shots 16 --iterations 800

# 联合训练
python3 main_joint.py \
    --tune_text_encoder True \
    --id_datasets aircraft caltech101 ... \
    --num_shots 16 --iterations 800
```

---

## 6. 输出指标

每个任务结束后评估 5 种方法：
- **ZS**: Zero-shot baseline
- **LADA**: LADA only
- **LADA+ZS**: LADA + Zero-shot ensemble（带 mask）
- **RGDA**: LR-RGDA only（独立对比）
- **RGDA+ZS**: LR-RGDA + Zero-shot ensemble（独立对比）

结果保存到 `experiments/lada_incremental_results_*.json`。

---

## 7. 相关文件

- `src/lada/lada_classifier.py`: LADA 分类器实现
- `src/lada/lada_trainer.py`: LADA 训练器实现
- `src/lada/dpt.py`: DPT 实现
- `main_incremental_lada.py`: LADA 入口脚本
- `src/trainers/lora_nsp_trainer.py`: 修复了 encode_text
- `src/utils/main_utils.py`: 修复了 get_zeroshot_classifier
- `/tmp/LADA/`: 官方 LADA 源码（服务器上）
