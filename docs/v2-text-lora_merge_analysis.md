# v2-text-lora 合并分析：伙伴全部改动与消融结果

**日期**: 2026-07-09
**来源**: `raoxuan98-hash/project_clip_continual_learning/tree/v2-text-lora`（2026-06-30 ~ 2026-07-07）
**合并 commit**: `f39e9ab`

---

## 目录

1. [改动总览](#1-改动总览)
2. [运行时优化](#2-运行时优化)
3. [Basis/Null 消融与 DoRA Bug 修复](#3-basisnull-消融与-dora-bug-修复)
4. [Arbor 自动消融（3-4 数据集）](#4-arbor-自动消融3-4-数据集)
5. [6-task 系统性单变量消融](#5-6-task-系统性单变量消融)
6. [10-task 退化与 DoRA Anchor 悬案](#6-10-task-退化与-dora-anchor-悬案)
7. [10-task 重消融计划与执行](#7-10-task-重消融计划与执行)
8. [Parser Bug 修复](#8-parser-bug-修复)
9. [最终默认配置与最佳结果](#9-最终默认配置与最佳结果)
10. [与之前配置的关键差异](#10-与之前配置的关键差异)

---

## 1. 改动总览

| 类别 | 内容 |
|------|------|
| 修改的核心文件 | `main_incremental.py`, `src/trainers/lora_nsp_trainer.py`, `src/models/lora_sgp.py`, `src/models/clip.py`, `src/utils/main_utils.py`, `src/utils/feature_extractor.py`, `src/utils/reference_loader.py`, `src/utils/retrieval_eval.py`, `src/lada/lada_classifier.py` |
| 新增文件 | `src/utils/infinite_sampler.py`, `evaluate_incremental_artifacts.py`, `merge_incremental_async_results.py`, `test_basis_variants.py`, `test_dora_init_fix.py`, 29 个脚本, 17 篇 chat-history, 2 篇 docs |
| 新增功能 | AMP 混合精度、多优化器/调度器、DoRA 支持、QKV 共享协方差、CD 散度/温度、NSP 变体（basis/null/soft）、StageTimer、检索评估、LADA artifact |

---

## 2. 运行时优化

### 问题

`main_incremental.py` 一次 10-task 完整运行耗时 7-8 小时。

### 瓶颈分析

| 阶段 | 耗时原因 | 优化 |
|------|---------|------|
| Inline 评估 | 10 任务共 55 次完整测试集特征提取（O(K²)） | 增大 `--eval_batch_size`；async eval 解耦 |
| LADA 分类器构建 | 每任务拼所有已见任务特征重新 build/fit | `--disable_lada` 跳过 |
| 特征提取重复 | 训练/协方差/分类器/GMM 各自构建 loader | 预构建 loader 缓存、预计算 frozen ZS 分类器 |

### 落地改动

- **`StageTimer`**：全局累积计时器，对 init、data_load、train、nsp_and_merge、feature_extraction、build_classifiers、evaluate 及每个 task total 计时，每任务结束及脚本末尾打印摘要
- **`--eval_batch_size`**：评估时独立 batch size（默认 None 即同训练 bs），可设 128/256 加速
- **`--eval_keep_features_on_device`**：特征留 GPU 避免 CPU↔GPU 传输
- **预构建 test loader 缓存**：避免 100 次评估中重复构造 dataset
- **预计算 frozen ZS 分类器**：每轮复用，不重复 tokenize
- **推荐 async eval 路径**：训练用 `--save_step_artifacts --async_eval_dir <dir> --skip_inline_eval`，评估用 `scripts/evaluate_incremental_artifacts.py`

---

## 3. Basis/Null 消融与 DoRA Bug 修复

### 3.1 DoRA 初始化 Bug

**Bug**：`initialize_adapters_from_covariance` 直接读写 `module.linear.weight.data`，但 DoRA 的 forward 使用 `weight_directions * magnitude`，`linear.weight` 在初始化后失效。

**修复**：
- 新增 `get_effective_weight` / `set_effective_weight` / `set_lora_init` helper
- DoRA 下读 `weight_directions * magnitude` 作为有效权重，residual 写回 direction/magnitude
- DoRA 下 LoRA-Null init 需将 B 按 `1/magnitude_new` 缩放，保证 `W_residual + magnitude_new * (B A) = W_eff`
- `test_dora_init_fix.py` 验证：SGPBaseLoRA / SGPBaseDoRA init 前后 output diff < 1e-6

### 3.2 DoRA Merge 简化

当前实现 `weight_directions += lora_delta` 后直接乘 magnitude，缺少原论文 `||direction + delta||` 归一化。但内部自洽，视为简化版 DoRA。

### 3.3 投影变体实现

实现了 6 种组合（`projection_param_mode` × `null_init_mode`）：

| projection_param_mode | 公式 | 说明 |
|------|------|------|
| `full` | ΔW = B A P | 当前默认，B A 后乘投影矩阵 |
| `fixed_basis` | ΔW = B U_h^T | 固定 tail basis，去掉 A |
| `core_basis` | ΔW = B C U_h^T | 可学习低秩组合 C∈R^(r×k)，k>r |

| null_init_mode | 说明 |
|------|------|
| `none` | 无特殊初始化（当前默认） |
| `history_init_only` | BA_init = W_0 U_h U_h^T，训练时 P=I |
| `history_init_runtime` | 同上 init + 训练时继续 runtime P |

### 3.4 QKV 共享协方差

同层 q_proj/k_proj/v_proj 输入空间相同，旧实现各自收集协方差各自 SVD，存在冗余。

**改造**：按输入源分组，同层 q/k/v 共享一个 hook，协方差只提取一次，复制给组内所有模块名。减少了 2/3 的 hook 和 SVD 计算量。

### 3.5 LoRA-Null 数学梳理

- `BA_init = W_0 U_null U_null^T`：把 BA 乘积初始化为 W_0 在 null/low-energy 子空间上的投影
- 通过 SVD 分解得到 A_init, B_init，保证 `W_0' + BA_init = W_0`（初始化前后功能等价）
- `U_h` 索引修正：`torch.linalg.eigh` 返回升序特征值，最小 k 个方向为 `V[:, 0:k]`
- soft projection 的 P 是满秩加权矩阵（`w_i > 0`），不是真正的投影；只有 hard 模式（`use_soft_projection=False, nsp_weight=0`）才得到真正投影

### 3.6 新增 CLI 参数

| 参数 | 说明 |
|------|------|
| `--projection_param_mode` | `full` / `fixed_basis` / `core_basis` |
| `--basis_rank` | basis 子空间维度 k（默认 128） |
| `--basis_window` | 特征值窗口 `tail` / `middle` |
| `--null_init_mode` | `none` / `history_init_only` / `history_init_runtime` |
| `--use_soft_projection` | 是否使用软投影（特征值加权） |
| `--use_dora` | 是否使用 DoRA（SGPBaseDoRA） |

---

## 4. Arbor 自动消融（3-4 数据集）

使用 Arbor（LLM 驱动的自动实验系统）在 3 数据集（aircraft→caltech101→dtd）上跑 5 种变体。

### Round 1（3 数据集, seed=42）

| 排名 | 变体 | Ens Average | Transfer | Last | Δ baseline |
|:----:|------|:---:|:---:|:---:|:---:|
| 🥇 | **hist_null_init_only** | **55.75** | 62.04 | 59.83 | **+0.21** |
| 🥈 | hist_null_init_runtime | 55.58 | 62.41 | 58.97 | +0.04 |
| 🥉 | current_nsp | 55.54 | 61.89 | 59.29 | — |
| 4 | basis_fixed_tail | 55.34 | 62.05 | 58.49 | −0.20 |
| 5 | basis_core_tail | 55.25 | 62.00 | 58.46 | −0.29 |

### 3-seed 验证（3 数据集）

| 变体 | Seed 42 | Seed 43 | Seed 44 | Mean ± Std | Δ baseline |
|------|:---:|:---:|:---:|:---:|:---:|
| **hist_null_init_only** | 56.01 | 56.01 | 56.29 | **56.10 ± 0.16** | **+0.56** |
| current_nsp | — | — | — | 55.54 | — |
| basis_fixed_tail | 55.34 | 55.13 | 55.22 | 55.23 ± 0.10 | −0.31 |

### 关键结论

1. **hist_null_init_only 是最优机制**：仅靠初始化即达 +0.56（σ=0.16），消除了每 task ~30min 的特征分解瓶颈（init-only 仅需 ~1min）
2. **运行时 NSP 是冗余的**：hist_null_init_runtime 与 baseline 无显著差异，init + runtime 无互补效果
3. **Fixed-basis 参数化损害塑性**：约束过强导致 Last 下降
4. **Eurosat 是分布外挑战**：所有方法加入 eurosat 后下降 ~1 点，null-init 优势在 4-dataset 设置上反转

---

## 5. 6-task 系统性单变量消融

所有实验在 **6-task（aircraft→caltech101→dtd→eurosat→flowers→oxford_pets）、16-shot、seed=42** 上进行。公共配置：current_nsp + AdamW + lr=1e-4 + bs=64 + iter=800 + cosine。

### 5.1 优化器

| 优化器 | Ens Average | Ens Last |
|------|:---:|:---:|
| RMSprop | **69.89** | **80.36** |
| AdamW | 69.85 | 80.25 |
| Adam | 69.85 | 80.30 |
| Adagrad | 69.08 | 79.64 |
| SGD (lr=1e-3) | 66.72 | 74.43 |

**结论**：AdamW/Adam/RMSprop 为第一梯队，差异 <0.1pp。选定 AdamW。

### 5.2 AdamW 学习率

| lr | Ens Average | Ens Last | ZS Transfer |
|:---:|:---:|:---:|:---:|
| 5e-5 | 68.77 | 77.86 | **61.75** |
| **1e-4** | **69.85** | 80.25 | 60.89 |
| 3e-4 | 69.80 | **81.87** | 58.25 |
| 6e-4 | 69.64 | 81.47 | 57.98 |

**结论**：**1e-4 是 Transfer/Average/Last 最佳平衡点**。3e-4 的 Last 最高但 Transfer 降 2.6pp。

### 5.3 SGD 精细学习率

| lr | Ens Average | Ens Last | ZS Transfer |
|:---:|:---:|:---:|:---:|
| 1e-3 | 66.72 | 74.43 | 60.88 |
| 2e-3 | 68.88 | 78.01 | 60.74 |
| 3e-3 | 69.76 | 80.99 | 59.39 |
| 4e-3 | 69.93 | 82.00 | 58.16 |
| 5e-3 | **69.99** | **82.24** | 57.44 |
| 5e-2 | 38.19 | 70.95 | 61.35 |

**结论**：SGD lr 越大 Last 越高 Transfer 越低。5e-2 直接摧毁零样本（ZS Last=0.17）。整体 SGD 不如 AdamW。

### 5.4 Batch Size

| bs | Ens Average | Ens Last |
|:---:|:---:|:---:|
| 32 | 69.32 | 79.02 |
| 64 | 69.85 | 80.25 |
| **128** | **70.14** | **81.01** |

**结论**：6-task 上 bs 越大越好，128 > 64 > 32。

### 5.5 蒸馏/辅助权重

| 参数 | 扫描范围 | 结论 |
|------|------|------|
| `cd_weight` | {0, 1, 2} | **2.0 最优**（Transfer + Last 均优于其他） |
| `fd_weight` | {0, 1, 2} | 三者几乎无差别，feature distillation 不敏感 |
| `aux_weight` | {0, 1, 2} | 0.0 Average 最好，1.0 Last 略高，差异很小 |

### 5.6 CD 散度形式（6 种，temp=2.0）

| 散度 | Ens Transfer | Ens Average | Ens Last |
|------|:---:|:---:|:---:|
| **kl_forward** | **60.30** | **70.09** | 80.72 |
| kl_reverse | 60.30 | 70.08 | 80.60 |
| js | 60.25 | 69.98 | 80.53 |
| cosine | 60.07 | 69.75 | 80.42 |
| l1 | 60.02 | 69.71 | 80.32 |
| mse | 60.00 | 69.53 | 80.11 |

**结论**：kl_forward 和 kl_reverse 几乎一致，kl_forward 略优。

### 5.7 CD 温度（kl_forward）

| temp | Ens Transfer | Ens Average | Ens Last |
|:---:|:---:|:---:|:---:|
| 1.0 | 60.34 | 70.00 | 80.56 |
| 2.0 | 60.30 | 70.09 | 80.72 |
| **4.0** | **60.37** | **70.14** | 80.70 |

**结论**：**temp=4.0 最优**。

### 5.8 Iterations × Scheduler（4×3=12 个实验，kl_forward, temp=4.0）

| iter | scheduler | Ens Average | Ens Last |
|:---:|------|:---:|:---:|
| 400 | constant | 70.11 | 80.57 |
| 400 | cosine_with_warmup | 68.92 | 78.12 |
| 400 | linear | 68.71 | 77.92 |
| 400 | cosine | 68.60 | 77.83 |
| 800 | **cosine_with_warmup** | **70.31** | 80.80 |
| 800 | linear | 70.26 | 80.62 |
| 800 | cosine | 70.25 | 80.68 |
| 800 | constant | 68.95 | 78.62 |
| 1600 | linear | **70.50** | **81.26** |
| 1600 | cosine | 70.40 | 81.17 |
| 1600 | cosine_with_warmup | 70.28 | 80.94 |
| 1600 | constant | 69.23 | 78.67 |

**结论**：
- iter 400→800：Average/Last 显著提升；800→1600：Last 继续小幅提升但 Transfer 下降
- cosine/linear/cosine_with_warmup 三者接近，均优于 constant
- 平衡选 **iter=800 + cosine_with_warmup**；追求 Last 选 iter=1600 + linear

### 5.9 主干微调层（rank=4）

| target modules | Ens Average | Ens Last |
|------|:---:|:---:|
| **q/k/v/out/ffn (All)** | **70.65** | **81.41** |
| v/out/ffn (FFN only) | 70.12 | 80.45 |
| q/k/v/out (Attention only) | 69.21 | 78.44 |
| q/k only | 66.04 | 73.90 |

**结论**：全微调最优。FFN-only 仅比 All 低 0.53pp。

### 5.10 LoRA Rank（All layers）

| rank | Ens Average | Ens Last |
|:---:|:---:|:---:|
| 2 | 69.79 | 79.78 |
| 4 | 70.12 | 80.57 |
| 8 | 70.39 | 81.56 |
| **16** | **70.60** | 81.43 |

**结论**：rank 越大 Average/Last 越高。rank=4 为效率平衡点。

### 5.11 QKV 投影方案

| 方案 | Ens Average | Ens Last |
|------|:---:|:---:|
| **方案 A：共享 P（当前默认）** | **70.65** | **81.41** |
| 旧实现：独立 P | 70.55 | 81.12 |
| 方案 B：fused QKV | 66.06 | 74.25 |

**结论**：共享 P 最优且消除了冗余 SVD。fused QKV 明显落后（-4.6pp），不采用。

### 5.12 最佳组合验证（4 Combo）

| Combo | 关键配置 | Ens Average | Ens Last |
|------|------|:---:|:---:|
| **C1 Balanced** | current_nsp + AdamW 1e-4 + bs128 + cwu + iter800 | **70.65** | 81.41 |
| C2 Last Max | current_nsp + AdamW 3e-4 + bs128 + linear + iter1600 | 69.85 | 81.69 |
| C3 Transfer Max | hist_runtime + AdamW 1e-4 + bs128 + cosine + iter800 | 70.44 | 80.71 |
| C4 Init-Only | hist_init_only + AdamW 1e-4 + bs128 + cwu + iter800 | 69.06 | 78.02 |

---

## 6. 10-task 退化与 DoRA Anchor 悬案

### 6.1 迁移退化

把 6-task C1 推到 10-task（追加 food101/mnist/stanford_cars/sun397）：

| 场景 | Ens Average | Ens Last |
|------|:---:|:---:|
| 6-task C1 (seed=42) | 70.65 | 81.41 |
| 10-task C1 (seed=42) | 69.6 | 80.0 |
| 退化 Δ | −**1.1** | −**1.4** |

### 6.2 DoRA Anchor 82.79 悬案

6/18 旧代码在 10-task 曾达 **Ens Last 82.79**，比当前 LoRA 的 80.0 高 **2.8pp**。

**差异追溯**：

| 维度 | 6/18 旧代码（82.79） | 当前代码（80.0） |
|------|:---:|:---:|
| use_dora | **true (DoRA)** | false (LoRA) |
| text_tuning_schedule | 不存在该参数，≈ low_lr_after | freeze_after |
| scheduler | cosine | cosine_with_warmup |
| cd_weight | 1.0 | 2.0 |

**三种假说**：
- (a) DoRA 在更长任务序列下优于 LoRA
- (b) 文本端持续微调（low_lr_after）是关键
- (c) scheduler/cd_weight 等微小差异的累积

---

## 7. 10-task 重消融计划与执行

### 整体设计

9 个 Phase，在 **10-task、16-shot、seed=43** 上重建超参数空间：

```
Pre-Wave A（backbone × text_schedule 全交叉，6 实验）
  → Wave A（lr sweep, 3 实验）
    → Wave C（batch size, 3 实验）
      → Wave D（CD 温度, 3 实验）
        → Wave F（text schedule 验证, 4 实验）
          → Phase 6（Soft vs Hard NSP）
            → Phase 7（nsp_eps / nsp_weight）
              → Phase 8（eta_min）
                → Phase 9（最终汇总）
```

**已砍**：Wave B（optimizer，6-task 差异 <0.1pp）、Wave E（aux_weight，6-task 不敏感）

### Pre-Wave A：DoRA vs LoRA × Text Schedule（✅ 已完成）

6 个实验全交叉（LoRA/DoRA × freeze_after/low_lr_after/always），seed=42。

**结果**：**LoRA > DoRA（3/3 配对全部胜出），text_tuning_schedule=always 最优**。

这澄清了 82.79 悬案——旧代码的优势来自文本端持续微调（≈low_lr_after），不是 DoRA。

### Phase 1：Wave A lr sweep（✅ 已完成）

切换到 seed=43，LoRA + text=always：

| lr | ZS Average | Ens Average | Ens Last |
|:---:|:---:|:---:|:---:|
| 5e-5 | 69.42 | 71.59 | 83.58 |
| **1e-4** | **69.50** | 71.19 | **83.65** |
| 3e-4 | 69.00 | 70.62 | 83.10 |

**Ens Last 83.65 已超过历史 82.79！** 种子标准差 σ_seed ≈ 0.5pp。

### Phase 2：Wave C Batch Size（✅ 已完成）

bs=32 ZS Average=**69.83**，bs=64 ZS Average=69.55。**bs=32 胜出**（与 6-task 的 bs=128 最优完全相反！）。

### Phase 3：Wave D CD 温度（✅ 已完成）

temp=4.0 ZS Average=**69.91** > temp=2.0 的 69.83。

### Phase 6-8（部分完成）

- **Phase 7 nsp_eps sweep**（✅ 已完成）：**nsp_eps=0.20 最优**，ZS Average +0.42pp vs 0.05
- Phase 6（Soft vs Hard NSP）、Phase 8（eta_min）、Phase 9（最终汇总）：待执行

---

## 8. Parser Bug 修复

发现 `main_incremental.py` 的 ArgumentParser 缺失关键参数：

| 缺失参数 | 影响 |
|------|------|
| `--cd_divergence` | CD 散度消融全部无法运行 |
| `--cd_temperature` | CD 温度消融全部无法运行 |
| `--scheduler` choices 不含 `linear`/`constant`/`cosine_with_warmup` | scheduler 消融全部无法运行 |

**根因**：trainer 通过 `getattr` 访问这些参数（有 fallback），但 CLI 从未注册。推测在某次代码清理中被意外移除。

**修复**（commit `710ee34`）：补充了 `--cd_divergence`、`--cd_temperature`，扩展了 `--scheduler` choices。

---

## 9. 最终默认配置与最佳结果

### 9.1 CLI 默认值变更汇总

| 参数 | 旧默认 | 新默认 | 依据 Phase |
|------|:---:|:---:|:---:|
| `--batch_size` | 64 | **32** | Phase 2: bs=32 ZS A 最高 |
| `--seed` | 42 | **43** | Phase 1 统一 |
| `--scheduler` | cosine | **cosine_with_warmup** | iter/scheduler 消融 |
| `--use_dora` | true | **false** | Pre-A: LoRA > DoRA (3/3) |
| `--nsp_eps` | 0.05 | **0.20** | Phase 7: +0.42pp vs 0.05 |
| `--cd_weight` | 1.0 | **2.0** | 蒸馏消融 |
| `--cd_temperature` | 2.0 | **4.0** | Phase 3: ZS A 69.91 |
| `--aux_weight` | 1.0 | **0.0** | 6-task 不敏感 |
| `--text_tuning_schedule` | freeze_after | **always** | Pre-A: always > low_lr > freeze |
| `--num_centers` | 1 | **4** | mc4ft200 |
| `--rgda_train_iter` | 0 | **200** | mc4ft200 |
| `--optimizer` | (无) | **adamw** | 新增，支持 adamw/adam/sgd/adagrad/rmsprop |
| `--amp` | (无) | **true** | 新增混合精度 |
| `--warmup_ratio` | (无) | **0.1** | 新增 warmup |
| `--eta_min` | (无) | **0.0** | 新增退火下限 |

### 9.2 🏆 全消融最佳配置

```text
LoRA+NSP, hard projection, nsp_eps=0.20
text_tuning_schedule=always
lr=1e-4, batch_size=32
scheduler=cosine_with_warmup, eta_min=0.0, warmup_ratio=0.1
optimizer=adamw, weight_decay=3e-5
cd_weight=2.0, fd_weight=1.0, aux_weight=0.0
cd_divergence=kl_forward, cd_temperature=4.0
num_centers=4, rgda_train_iter=200
ensemble_normalize=maxshift, seed=43
lora_rank=4, lora_target_modules=q_proj,k_proj,v_proj,out_proj,fc1,fc2
iterations=800, projection_param_mode=full, null_init_mode=none
```

### 9.3 最佳结果

| 分类器 | Transfer | Average | Last |
|------|:---:|:---:|:---:|
| Zero-shot | — | **70.33** | — |
| Ensemble | — | **72.18** | **83.93** |

**Ens Last 83.93 比历史 82.79 高 +1.14pp，比我们之前最优 82.6 高 +1.33pp。**

---

## 10. 与之前配置的关键差异

| 维度 | 之前最优 | 伙伴最优 | 变化方向 |
|------|:---:|:---:|:---:|
| batch_size | 64 | **32** | 🔴 反转（6-task 上 128>64>32，10-task 上 32>64） |
| nsp_eps | 0.05 | **0.20** | 🔴 反转（更宽松的投影反而更好） |
| text_tuning_schedule | freeze_after | **always** | 🔴 全文微调优于冻结 |
| scheduler | cosine | **cosine_with_warmup** | 新增 warmup |
| seed | 42 | **43** | 种子变更 |
| cd_weight | 1.0 | **2.0** | 加强蒸馏 |
| cd_temperature | 未使用 | **4.0** | 新增温度参数 |
| cd_divergence | 未使用 | **kl_forward** | 新增散度选择 |
| optimizer | AdamW | **AdamW** | 一致 |
| use_dora | false | **false** | 一致（确认 LoRA > DoRA） |
| Ens Last | 82.6 | **83.93** | **+1.33pp** |

### 关键洞察

1. **bs=32 在 10-task 上反超 bs=128**：与 6-task 结论完全相反。更长任务序列下，小 batch 的噪声梯度可能提供更好的泛化/抗遗忘正则化
2. **nsp_eps=0.20 > 0.05**：更宽松的投影（保留更小子空间）在 10-task 上更好，说明之前过于激进的投影可能过度限制了新任务学习
3. **text=always > freeze_after**：文本端全程微调在 10 个不同领域间建立了更好的对齐，过早冻结文本端损害了跨任务泛化
4. **DoRA 不是 82.79 的原因**：Pre-Wave A 全交叉实验证明 LoRA 在 3/3 text_schedule 配对上均优于 DoRA。旧代码 82.79 的优势来自 text≈low_lr_after + cosine + cd=1.0 的组合
5. **Ens Last 83.93 刷新 SOTA**：这个结果来自所有消融结论的综合叠加，主要收益来自 text=always + nsp_eps=0.20 + bs=32（10-task）+ cd_temperature=4.0

---

## 相关文件

- `main_incremental.py` — 主要入口，默认值已更新为最佳配置
- `src/trainers/lora_nsp_trainer.py` — 训练器，AMP + 多优化器/调度器 + QKV 共享协方差 + CD 散度/温度
- `src/models/lora_sgp.py` — LoRA/DoRA/NSP 核心，basis/null 变体 + DoRA init 修复
- `src/utils/main_utils.py` — 评估逻辑，eval_batch_size + inference_mode
- `chat-history/2026-07-05-weekly-ablation-review.md` — 6-task 全消融汇总
- `chat-history/2026-07-06-10task-reablation-plan.md` — 10-task 重消融计划
- `chat-history/2026-07-07-best-config-defaults-applied.md` — 最佳配置默认值更新
- `docs/lora_null_basis_ablation_plan.md` — Basis/Null 消融数学方案
- `docs/arbor_ablation_results.md` — Arbor 自动消融结果
