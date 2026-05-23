# main_joint.py OOD 评估修复与超参数配置

**日期**: 2026-05-23
**会话概况**: 修复 main_joint.py 中 OOD 评估的零样本准确率为 0 的问题，添加集成分类器支持，添加 ALL 简写，更新 LR-RGDA 默认超参数

---

## 1. 关键决策

### 1.1 OOD 零样本准确率修复

问题：零样本分类器只从 ID 数据集构建类别权重，OOD 数据集类别不在其中 → OOD 准确率全为 0%。

修复：对每个不在 ID 列表中的 novel OOD 数据集，单独构建 `combined_class_names = ID类名 + 该OOD类名` 的联合零样本分类器，传给 `evaluate_dataset` 进行评估。

### 1.2 OOD 集成分类器

OOD 也要用 ensemble，集成权重 `--alpha` 控制（默认 0.5）。

逻辑：`evaluate_dataset` 中 `ensemble_logits` 前 `num_id_classes` 个位置叠加 RGDA，OOD 类别位置（索引 ≥ num_id_classes）仅靠零样本贡献。因此 OOD 集成准确率 ≈ 零样本准确率（因为 RGDA 对 OOD 类别无贡献）。

### 1.3 添加 `ALL` 简写

`--id_datasets ALL` 自动展开为全部 10 个 X-TAIL 数据集：
```
aircraft, caltech101, dtd, eurosat, flowers,
food101, mnist, oxford_pets, stanford_cars, sun397
```
不传 `--id_datasets` 时默认为全部数据集，`--ood_datasets` 不传时自动推断为补集。

### 1.4 LR-RGDA 默认超参数更新

基于网格搜索结果：

| 参数 | 旧值 | 新值 | 说明 |
|------|:----:|:----:|------|
| `--rgda_alpha1` | 0.6 | **0.3** | 每个类各自协方差的权重 |
| `--rgda_alpha2` | 1.0 | **2.0** | 全局共享协方差的权重 |
| `--rgda_alpha3` | 0.5 | 0.5 | 单位矩阵正则化的权重 |

协方差正则化：Σ_k = α₁·Σ_k^(class) + α₂·Σ^(global) + α₃·I

## 2. 实验结果（不微调，全部 10 个数据集）

| 分类器 | ID Average |
|--------|:----------:|
| Zero-shot | 57.2% |
| LR-RGDA | 70.1% |
| Ensemble (α=0.5) | **73.7%** |

集成分类器优于零样本和纯 LR-RGDA。

## 3. 相关文件

- `main_joint.py`: 主要改动（OOD 评估 + ALL 简写 + 超参数默认值）
- `main_incremental.py`: 同步超参数默认值
- `src/utils/main_utils.py`: `evaluate_dataset` 中的 ensemble 逻辑
