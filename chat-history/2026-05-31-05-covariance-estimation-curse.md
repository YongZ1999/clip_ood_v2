# 高维低样本下协方差估计的维度灾难实验

**日期**: 2026-05-31
**会话概况**: 通过全协方差 vs spherical GMM vs rank-1 GMM 的系统对比实验，证明 16-shot + 512 维场景下全协方差矩阵完全无法估计，spherical 假设是唯一可行方案，伪特征仅比真实特征低 2%。

---

## 1. 核心结论

### 1.1 全协方差矩阵在低样本下完全失效

| 采样方案 | α1/α2/α3 | LR-RGDA |
|---------|-----------|------|
| 真实特征（基准） | -- | **74.8%** |
| Spherical GMM (k=4) | -- | **72.8%** |
| Rank-1 GMM (k=4) | -- | **72.7%** |
| Full 不正则 | 1.0 / 0 / 0.5 | 6.8% |
| Full 正则 | 0.2 / 2.0 / 0.5 | 6.0% |
| Full 正则（球面估计，错误） | 0.2 / 2.0 / 0.5 | 5.5% |

**关键发现**: 无论正则化强度如何（α2=0 到 2.0），全协方差方案都彻底崩溃（6-7%）。这不是正则化参数选择的问题，而是 **512×512 协方差矩阵在 16 个样本下数学上无法估计**——自由度 512×513/2 ≈ 131,328 远超样本量 16。

### 1.2 Spherical 接近理论上界

Spherical GMM 72.8% vs 真实特征 74.8%，差距仅 2.0%。Rank-1（72.7%）几乎与 Spherical 完全一致，说明一阶主成分方向不提供额外判别信息。

### 1.3 实验流程（所有方案一致）

```
原始空间    mean/cov 估计 → 正则化 → 采样（范数 ~10-20）
   ↓
球面投影    samples / samples.norm（范数 = 1）
   ↓
分类器     训练在球面特征上
   ↓
评估       测试集特征也 L2 归一化
```

---

## 2. 协方差建模方案对比

| 方案 | 协方差形式 | 参数量 | 16-shot 可估计？ | 精度 |
|------|----------|--------|-----------------|------|
| Full | Σ (512×512) | 131,328 | ❌ | 6.0% |
| Spherical GMM | σ²×I (标量/分量) | 1/分量 | ✅ | 72.8% |
| Rank-1 GMM | σ²_base×I + λ·vvᵀ (球面基+主成分) | 1+512/分量 | ⚠️边界 | 72.7% |
| Real features | -- | -- | -- | 74.8% |

---

## 3. 实现细节

### 3.1 新增命令行参数 (main_joint.py)

```
--sample_alpha1/2/3   高斯采样正则化（独立于 RGDA 分类器构建参数）
--gmm_k               GMM 分量数（0=全协方差，>0=GMM 模式）
--gmm_cov_type        spherical / rank1
--gmm_reg             标量方差正则化
--use_gaussian_features  启用高斯特征采样
```

### 3.2 分类器 fit() 微调

- **LADA**: `curr_lada_features` (k-means 聚类中心) → `nn.Parameter` → CE loss 优化
- **LR-RGDA**: `affine_weights` + `affine_biases` 临时转 Parameter → CE loss 优化 → 恢复 buffer
- 协方差结构（低秩精度矩阵）始终保持冻结

### 3.3 GRGDA 多中心

```bash
--num_centers 4 --rgda_train_iter 200  # 73.9% (单中心) → 74.8% (4-center)
```

每类 k-means 到 M 个中心 → 共享协方差 → log-sum-exp 归约为类概率。

---

## 4. 完整实验结果

### 4.1 真实特征 + 分类器微调（X-TAIL 16-shot）

| Dataset | ZS | RGDA(分析) | RGDA(4c+fit) | LADA(16c+fit) | Ens(4c+fit) |
|---------|-----|------------|-------------|----------------|-------------|
| aircraft | 22.3 | 33.3 | 35.0 | 34.0 | 35.2 |
| caltech101 | 73.5 | 82.9 | 87.1 | 83.2 | 87.1 |
| dtd | 37.3 | 58.8 | 64.1 | 63.1 | 64.2 |
| eurosat | 37.4 | 76.4 | 84.6 | 86.8 | 84.9 |
| flowers | 62.0 | 92.3 | 94.2 | 93.8 | 94.3 |
| food101 | 82.6 | 76.3 | 80.0 | 75.6 | 80.3 |
| mnist | 43.9 | 74.3 | 81.0 | 90.3 | 81.1 |
| oxford_pets | 84.2 | 72.0 | 81.1 | 80.9 | 81.9 |
| stanford_cars | 60.3 | 58.7 | 69.8 | 67.5 | 70.2 |
| sun397 | 60.9 | 66.7 | 71.2 | 68.4 | 71.5 |
| **Average** | **56.4** | **69.2** | **74.8** | **74.4** | **75.1** |

### 4.2 伪特征训练（高斯采样替代真实特征）

| Dataset | Spherical GMM | Rank-1 GMM | Full noreg | Full reg |
|---------|--------------|-----------|------------|----------|
| aircraft | 32.7 | -- | -- | -- |
| caltech101 | 85.8 | -- | -- | -- |
| dtd | 61.4 | -- | -- | -- |
| eurosat | 83.2 | -- | -- | -- |
| flowers | 93.0 | -- | -- | -- |
| food101 | 79.0 | -- | -- | -- |
| mnist | 80.0 | -- | -- | -- |
| oxford_pets | 78.2 | -- | -- | -- |
| stanford_cars | 65.3 | -- | -- | -- |
| sun397 | 69.2 | -- | -- | -- |
| **Average** | **72.8** | **72.7** | **6.8** | **6.0** |

---

## 5. 运行示例

```bash
# 真实特征 + 两个分类器微调
python3 main_joint.py --id_datasets ALL --iterations 0 --gpu 1 \
    --enable_lada --lada_k 16 --lada_beta 1.0 --lada_alpha 0.5 \
    --lada_train_iter 200 --lada_train_lr 0.01 \
    --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01

# Spherical GMM 采样
python3 main_joint.py --id_datasets ALL --iterations 0 --gpu 1 \
    --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 \
    --use_gaussian_features --gmm_k 4 --gaussian_samples_per_class 16

# Rank-1 GMM 采样
python3 main_joint.py --id_datasets ALL --iterations 0 --gpu 1 \
    --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 \
    --use_gaussian_features --gmm_k 4 --gmm_cov_type rank1

# 全协方差 + 自定义正则化
python3 main_joint.py --id_datasets ALL --iterations 0 --gpu 1 \
    --num_centers 4 --rgda_train_iter 200 --rgda_train_lr 0.01 \
    --use_gaussian_features --gmm_k 0 \
    --sample_alpha1 1.0 --sample_alpha2 0 --sample_alpha3 0.5
```
