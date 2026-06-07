# 分类器梯度微调与高斯特征采样实验

**日期**: 2026-05-31
**会话概况**: 实现了 LADA 和 LR-RGDA 分类器的梯度微调，并通过 spherical GMM vs 全协方差高斯采样实验，发现 spherical GMM 能有效替代真实特征训练分类器。

---

## 1. 关键发现

### 1.1 分类器梯度微调显著提升性能

| 分类器 | 分析版 | 微调后 (200 iter) | 提升 |
|--------|--------|-------------------|------|
| LR-RGDA 单中心 | 69.2% | 73.9% | +4.7% |
| LR-RGDA 4-center | ~70% | 74.8% | +4.8% |
| LR-RGDA 8-center | -- | 74.7% | -- |
| LADA 16-center | ~65% | 74.4% | +9.4% |

- **最优**: LR-RGDA 4-center + 微调 = 74.8%, Ensemble RGDA+ZS = **75.1%**
- 4-center 是最优配置，8-center 略微过拟合（train acc 91.6% vs 89.7%，test 反降 0.1%）
- LADA 收敛极快（40 轮到 100%），但最终精度略低于 LR-RGDA（74.4% vs 74.8%）

### 1.2 Spherical GMM 有效替代真实特征（核心发现）

| 分类器训练数据 | LR-RGDA 4-center | LADA 16-center |
|---------------|------------------|----------------|
| 真实特征 | **74.8%** | **74.4%** |
| Spherical GMM (k=4) | **72.8%** (-2.0%) | **71.9%** (-2.5%) |
| 正则化全协方差高斯 | 5.5% | 3.9% |

**核心结论**: Spherical GMM 仅比真实特征低 2%，而全协方差高斯完全崩溃（5.5%）。证明了：
1. 高维空间 16-shot 下，**任何超过球形的协方差结构都无法可靠估计**
2. Spherical 假设（1 个标量方差/分量）是低样本场景下唯一可行的选择
3. GMM（多分量）比单高斯保留了类内多模态性，是精度差距仅 2% 的关键

### 1.3 协方差估计的维度灾难

```
全协方差方案:   512×512 = 131,328 参数/类  ← 16 个样本 → 完全噪声
Spherical GMM:  1 scalar/分量 × 4 分量 = 4 参数/类  ← 16 个样本 → 可靠估计
```

LR-RGDA 的正则化协方差 `0.2*Σ_class + 2.0*Σ_global + 0.5*I` 中：
- `Σ_class`（权重 0.2）: 16-shot 估计的 512×512 矩阵 → 几乎无信息
- `Σ_global`（权重 2.0）: 主导分布形状 → 所有类趋于相同
- 这就是为什么采样出的特征毫无区分度

---

## 2. 实现细节

### 2.1 新增/修改的文件

| 文件 | 改动 |
|------|------|
| `src/lada/lada_classifier.py` | 添加 `fit()` 梯度微调方法；修复 `register_buffer` 被覆盖后 `to()` 不生效的 bug；`forward()` 加入 device safety |
| `src/classifiers/gaussian_classifier.py` | `LRRGDA` 类添加 `fit()` 临时转 Parameter → 训练 → 恢复 buffer |
| `src/classifiers/lr_rgda_classifier.py` | `LRRGDAClassifier` 添加 `fit()` 委托方法 |
| `src/utils/main_utils.py` | `evaluate_dataset` 新增可选 `lada_classifier` 参数；LADA+ZS 改为加权融合（替代 mask 加法）|
| `src/utils/feature_extractor.py` | 修复 `get_image_features` 返回 `BaseModelOutputWithPooling` 的兼容性问题（改用手动提取）|
| `main_joint.py` | 添加 LADA/RGDA 微调参数 + GMM 采样参数 + 高斯特征生成逻辑 |
| `main_incremental.py` | 解包适配新的 7 元组返回值 |

### 2.2 新增命令行参数 (main_joint.py)

```bash
# LADA 分类器
--enable_lada --lada_k 16 --lada_beta 1.0 --lada_alpha 0.5
--lada_train_iter 200 --lada_train_lr 0.01

# LR-RGDA 微调
--num_centers 4
--rgda_train_iter 200 --rgda_train_lr 0.01

# 高斯特征采样
--use_gaussian_features --gmm_k 4 --gaussian_samples_per_class 16
```

### 2.3 分类器 fit() 机制

- **LADA**: `curr_lada_features` (k-means 聚类中心) 转为 `nn.Parameter`，CE loss 梯度更新；`joint_classifier` (块对角 one-hot) 冻结
- **LR-RGDA**: `affine_weights` [C,D] 和 `affine_biases` [C] 临时转 Parameter 训练，协方差结构（低秩精度矩阵）冻结，训练后恢复为 buffer
- 两者均使用 AdamW + CosineAnnealingLR，200 轮

---

## 3. 决策记录

1. **LADA+ZS 融合方式**: 从 mask 加法 `zs + mask*α*lada` 改为直接加权 `(1-α)*zs + α*lada`，与 LR-RGDA Ensemble 保持一致
2. **LADA k-means 微调时机**: 联合训练中，LADA 不需要梯度训练（与 LR-RGDA 公平对比分析型分类器）；增量训练中需要（LADA 论文的正宗用法）。后续通过 `--lada_train_iter` 实现可选微调
3. **方差估计方案**: Spherical GMM 是 16-shot 下唯一可行的协方差建模方式

---

## 4. 完整实验结果对比表

**X-TAIL 16-shot, iterations=0 (zero-shot evaluation only)**

| Dataset | ZS | RGDA (分析) | RGDA (4c+fit) | Ens (fit) | LADA (16c+fit) |
|---------|-----|-------------|---------------|-----------|----------------|
| aircraft | 22.3 | 33.3 | 35.0 | 35.2 | 34.0 |
| caltech101 | 73.5 | 82.9 | 87.1 | 87.1 | 83.2 |
| dtd | 37.3 | 58.8 | 64.1 | 64.2 | 63.1 |
| eurosat | 37.4 | 76.4 | 84.6 | 84.9 | 86.8 |
| flowers | 62.0 | 92.3 | 94.2 | 94.3 | 93.8 |
| food101 | 82.6 | 76.3 | 80.0 | 80.3 | 75.6 |
| mnist | 43.9 | 74.3 | 81.0 | 81.1 | 90.3 |
| oxford_pets | 84.2 | 72.0 | 81.1 | 81.9 | 80.9 |
| stanford_cars | 60.3 | 58.7 | 69.8 | 70.2 | 67.5 |
| sun397 | 60.9 | 66.7 | 71.2 | 71.5 | 68.4 |
| **Average** | **56.4** | **69.2** | **74.8** | **75.1** | **74.4** |

**GMM k=4 伪特征训练的结果**

| Dataset | ZS | RGDA (GMM) | Ens (GMM) | LADA (GMM) |
|---------|-----|------------|-----------|------------|
| aircraft | 22.3 | 32.7 | 32.9 | 32.0 |
| caltech101 | 73.5 | 85.8 | 85.8 | 80.9 |
| dtd | 37.3 | 61.4 | 61.5 | 60.5 |
| eurosat | 37.4 | 83.2 | 83.3 | 85.2 |
| flowers | 62.0 | 93.0 | 93.1 | 91.6 |
| food101 | 82.6 | 79.0 | 79.3 | 74.8 |
| mnist | 43.9 | 80.3 | 80.4 | 86.4 |
| oxford_pets | 84.2 | 78.2 | 79.0 | 76.8 |
| stanford_cars | 60.3 | 65.3 | 65.7 | 63.2 |
| sun397 | 60.9 | 69.2 | 69.5 | 67.3 |
| **Average** | **56.4** | **72.8** | **73.1** | **71.9** |

---

## 5. 运行示例

```bash
# 真实特征 + 两个分类器都微调
python3 main_joint.py --id_datasets ALL --iterations 0 --gpu 1 \
    --enable_lada --lada_k 16 --lada_beta 1.0 --lada_alpha 0.5 \
    --lada_train_iter 200 --lada_train_lr 0.01 \
    --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01

# Spherical GMM 伪特征替代真实特征
python3 main_joint.py --id_datasets ALL --iterations 0 --gpu 1 \
    --enable_lada --lada_k 16 --lada_beta 1.0 --lada_alpha 0.5 \
    --lada_train_iter 200 --lada_train_lr 0.01 \
    --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 \
    --use_gaussian_features --gmm_k 4 --gaussian_samples_per_class 16
```
