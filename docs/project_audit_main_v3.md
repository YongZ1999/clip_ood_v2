# main_v3 项目全盘审计报告

**日期**: 2026-07-13
**审计范围**: 完整代码库 (`main_v3`) — 代码、配置、实验、论文、文档、历史记录
**方法名变更**: LoRA-NSP → **LoRA-NF**（论文中已改名，代码中仍保留 `lora_nsp` 标识符以保持实验可复现性）

---

## 目录

1. [项目概览](#1-项目概览)
2. [顶层入口](#2-顶层入口)
3. [核心源码架构](#3-核心源码架构)
4. [当前最佳配置与结果](#4-当前最佳配置与结果)
5. [消融历史全景](#5-消融历史全景)
6. [论文现状](#6-论文现状)
7. [目录结构与项目管理](#7-目录结构与项目管理)
8. [已知问题与待办](#8-已知问题与待办)
9. [正式实验待执行 (E1-E6)](#9-正式实验待执行-e1-e6)

---

## 1. 项目概览

### 研究目标

CLIP 模型在类增量学习中的灾难性遗忘问题。提出**训练端 + 推理端协同优化**框架：

| 端 | 方法 | 职责 |
|:---|------|------|
| **训练端** | **LoRA-NF**（零空间滤波 + 跨模态蒸馏 FD/CD） | 保护历史任务知识 + 保持跨模态对齐 |
| **推理端** | **LR-RGDA 集成分类器**（低秩正则化高斯判别分析 + 零样本集成） | 提升分类精度 |

### 核心创新：LoRA-NF（Null-Space Filtering）

- **公式**：ΔW = P A B（paper 右乘约定），代码中为 `B @ A @ P`（PyTorch 左乘约定，两者互为转置）
- **机制**：P 由历史任务激活二阶矩（activation second moment）的特征分解构造，在 LoRA adapter 输入侧做持久的 forward filtering
- **不同于 gradient projection**：梯度投影是后处理梯度，LoRA-NF 的 P 在模型计算图中，影响前向响应、反向梯度和优化器状态
- **不同于 LoRA-Null**：LoRA-Null 是初始化时将 A 放在 null space，训练时无 P；LoRA-NF 是持久的运行时约束
- **当前默认**：leaky hard NSP，`P = (1-ρ) U U^T + ρ I`（ρ=0.02, nsp_eps=0.20）
- **关键结论**：在 leaky 设置下，LoRA-NF 不缩小标准 LoRA 的表达集（expressivity set），但改变了优化几何（P² 谱预处理效应）

---

## 2. 顶层入口

### `main_incremental.py`（1471 行）— 增量学习入口

**功能流程**:
1. 解析参数 → 初始化 `LoRANSPTrainer` → 预构建 test loader 缓存 + frozen ZS 分类器
2. **增量循环**（10 个任务）：
   - a. 准备训练数据（ConcatDataset + InfiniteSampler）
   - b. 适配器初始化（旧 `init_mode` + 新 `null_init_mode`）
   - c. 训练（根据 `text_tuning_schedule` 动态设置 `text_lr`）
   - d. 任务后处理：协方差提取 → NSP 投影更新 → merge LoRA weights
   - e. 缓存 LADA-style 文本原型（lada_hybrid 模式）
   - f. 特征提取 + 多中心统计字典构建 + GMM 拟合
   - g. 构建分类器：LR-RGDA（含 GMM/center replay fine-tuning）+ LADA（可选）
   - h. 按 LADA 协议评估所有已见任务（ZS/LR-RGDA/Ensemble/LADA/LADA+ZS 五种分类器）
   - i. 多模态检索评估（可选）
3. 打印 LADA 指标 → 保存 JSON 结果

**支持的全部 CLI 参数**: 约 100 个，覆盖训练、LoRA、蒸馏、分类器、GMM、检索、评估等所有方面。

**关键设计决策**:
- `text_classifier_mode=lada_hybrid`：已见类用各任务训练后的文本原型，未见类用 frozen CLIP
- `num_centers=4 + rgda_train_iter=200`（mc4ft200）：多中心 + GMM fine-tuned 集成分类器
- 支持 `--save_step_artifacts --skip_inline_eval` 异步评估路径
- `StageTimer` 全局计时器 + `--eval_batch_size` 加速

### `main_joint.py`（1284 行）— 联合训练入口

与 `main_incremental.py` 的关键区别：
- **一次性训练**：所有 ID 数据集合并为一个大数据集，用 WeightedRandomSampler 平衡采样
- 无 per-task 权重合并循环，训练完一次性 merge
- 文本/视觉编码器默认**不微调**（内存限制）
- 支持 classifier comparison mode（LR-RGDA vs LADA grid search）
- 评估在单点进行（+ 可选训练前 frozen 评估）

---

## 3. 核心源码架构

```
src/
├── trainers/
│   └── lora_nsp_trainer.py      # LoRANSPTrainer: 训练循环、协方差、蒸馏、AMP
├── models/
│   ├── clip.py                  # get_clip_model(): CLIP 模型工厂
│   ├── lora_sgp.py              # LoRA/DoRA/NSP 核心: SGPBaseLoRA, SGPBaseDoRA, FixedProjection, build_projection
│   ├── lada_text_adapter.py     # LADA 风格 text AdaptFormer
│   ├── lora_baseline.py         # 标准 LoRA (Vanilla)
│   ├── trainer.py               # ⚠️ 旧版遗留 trainer
│   └── utils.py                 # 蒸馏损失函数 (FD, CD, SCE)
├── classifiers/
│   ├── lr_rgda_classifier.py    # LRRGDAClassifier + EnsembleClassifier
│   ├── gaussian_statistics.py   # GaussianStatistics, k-means, multi-center stats
│   └── da_classifier_builder.py # LR-RGDA 构建器（Woodbury 恒等式）
├── lada/
│   ├── lada_classifier.py       # LADAClassifier (k-means 原型 + 指数亲和度)
│   └── lada_trainer.py          # LADA 训练器
├── utils/
│   ├── data.py                  # 数据加载: get_xtail_trainloader, get_transforms, Flickr8kDataset
│   ├── main_utils.py            # 评估: evaluate_dataset, get_full_stats, combine_ensemble_logits
│   ├── continual_metrics.py     # ContinualLearningMetrics: K×K 矩阵 tracking
│   ├── feature_extractor.py     # extract_features: GPU 加速特征提取
│   ├── retrieval_eval.py        # 多模态检索评估 (COCO/Flickr30K)
│   ├── reference_loader.py      # 参考数据集加载 (蒸馏用)
│   ├── infinite_sampler.py      # InfiniteSampler: 无限循环采样
│   ├── config_manager.py        # 配置管理 (YAML)
│   ├── evaluation.py            # 评估辅助
│   └── hyperparameter_optimizer.py # 超参数优化
├── experiments/                 # 实验入口脚本
│   ├── run_continual_learning.py
│   ├── run_continual_learning_routing.py
│   ├── run_continual_learning_routing_v2.py
│   └── generate_paper_tables.py
├── detectors/                   # ⚠️ 已删除 (OOD detection)
└── routing/                     # ⚠️ 已删除 (adaptive routing)
```

### 关键模块详解

#### `lora_sgp.py` — LoRA/DoRA/NSP 核心

**投影参数化模式**（`projection_param_mode`）:

| 模式 | 公式 | 可训参数 |
|------|------|------|
| `full` (默认) | B A P | B[d_out,r] + A[r,d_in] |
| `fixed_basis` | B U_h^T | B[d_out,k] |
| `core_basis` | B C U_h^T | B[d_out,r] + C[r,k] |

**NSP 投影类型**:
- **Hard**（默认）: `P = (1-ρ) V_keep V_keep^T + ρ I`，`V_keep` 由累积能量阈值 `nsp_eps` 自适应选择
- **Soft**: `P = V diag(w_i) V^T`，所有权重 > 0，满秩

**DoRA vs LoRA**:
- DoRA 将 W 分解为 `direction * magnitude`，delta 作用于 direction
- 消融结论：**LoRA > DoRA**（3/3 text_schedule 配对），`--use_dora` 默认 `false`

**QKV 共享**:
- 同层 q/k/v 共享输入空间 → 共享 input hook → 共享协方差和投影矩阵
- 减少 2/3 的 hook 和 SVD 计算量

#### `lora_nsp_trainer.py` — 训练器

**优化器**: adamw/adam/sgd/adagrad/rmsprop
**调度器**: cosine/onecycle/linear/constant/cosine_with_warmup
**AMP**: GradScaler + autocast 混合精度
**蒸馏**:
- FD（特征蒸馏）：student vs teacher image features
- CD（跨模态蒸馏）：6 种散度（kl_forward/kl_reverse/js/mse/cosine/l1）+ 温度系数

#### `main_utils.py` — 评估核心

- `evaluate_dataset()`: 单数据集评估 + 21 点 alpha sensitivity sweep
- `combine_ensemble_logits()`: 归一化模式（maxshift/zscore/prob/raw）
- `get_full_stats()`: LADA Transfer/Average/Last 计算

#### `continual_metrics.py` — LADA 指标

- K×K 准确率矩阵在线 tracking
- Transfer = mean(前 k-1 步在任务 k 上的准确率), k=2..K
- Average = 每任务所有步准确率均值再平均
- Last = 最终模型每任务准确率均值

---

## 4. 当前最佳配置与结果

### 🏆 全消融最优配置（V4+V5 验证）

```text
LoRA+NSP, hard projection, nsp_eps=0.20, nsp_weight=0.02
text_tuning_schedule=always
lr=1e-4, batch_size=32
scheduler=cosine_with_warmup, warmup_ratio=0.1, eta_min=0.0
optimizer=adamw, weight_decay=3e-5
cd_weight=2.0, fd_weight=1.0, aux_weight=0.0
cd_divergence=kl_forward, cd_temperature=4.0
num_centers=4, rgda_train_iter=200, rgda_train_lr=0.01
rgda_fit_source=gmm_sample, gmm_k=4, gmm_sample_mode=mean
ensemble_normalize=maxshift, seed=43
lora_rank=4, lora_target_modules=q_proj,k_proj,v_proj,out_proj,fc1,fc2
iterations=800, projection_param_mode=full, null_init_mode=none
text_classifier_mode=lada_hybrid
```

### 最佳结果（10-task, 16-shot, seed=43）

| 分类器 | Transfer | Average | Last |
|------|:---:|:---:|:---:|
| Zero-shot | — | **70.33** | — |
| Ensemble | — | **72.18** | **83.93** |

### 所有 CLI 默认值

| 参数 | 值 | 依据 |
|------|:---:|------|
| `--lr` | 1e-4 | AdamW 最佳平衡点 |
| `--batch_size` | 32 | 10-task 最优（与 6-task 相反） |
| `--seed` | 43 | Phase 1 统一 |
| `--scheduler` | cosine_with_warmup | > cosine, linear, constant |
| `--nsp_eps` | 0.20 | > 0.02, 0.05, 0.08, 0.12 (Phase 7) |
| `--text_tuning_schedule` | always | > low_lr_after > freeze_after > never |
| `--use_dora` | false | LoRA > DoRA (Pre-Wave A) |
| `--cd_weight` | 2.0 | > 0.0, 1.0 (蒸馏消融) |
| `--cd_temperature` | 4.0 | > 1.0, 2.0 (温度消融) |
| `--cd_divergence` | kl_forward | > kl_reverse, js, mse, cosine, l1 |
| `--fd_weight` | 1.0 | 不敏感但 fd=0 下降 |
| `--aux_weight` | 0.0 | 不敏感 |
| `--num_centers` | 4 | mc4ft200 |
| `--rgda_train_iter` | 200 | mc4ft200 |
| `--warmup_ratio` | 0.1 | 10% warmup 最优 |
| `--optimizer` | adamw | 第一梯队 (adam/adamw/rmsprop) |
| `--weight_decay` | 3e-5 | V5 验证最优 |

---

## 5. 消融历史全景

### 6-task 消融（seed=42, 6 数据集）

| 维度 | 扫描 | 最优 | 关键发现 |
|------|------|:---:|------|
| 优化器 | adamw/adam/rmsprop/sgd/adagrad | AdamW | 第一梯队差异 <0.1pp |
| 学习率 | 5e-5/1e-4/3e-4/6e-4 | **1e-4** | 3e-4 Last 最高但 Transfer -2.6pp |
| SGD lr | 1e-3/2e-3/3e-3/4e-3/5e-3/5e-2 | 5e-3 (Last) | 越大 Last 越高 Transfer 越低；整体不如 AdamW |
| Batch Size | 32/64/128 | **128** | 6-task: 越大越好 |
| CD 散度 | kl_forward/kl_reverse/js/mse/cosine/l1 | **kl_forward** | 与 kl_reverse 几乎一致 |
| CD 温度 | 1.0/2.0/4.0 | **4.0** | Transfer/Average/Last 均最优 |
| CD 权重 | 0.0/1.0/2.0 | **2.0** | |
| FD 权重 | 0.0/1.0/2.0 | 不敏感 | {0,1,2} 差异 <0.1pp |
| Aux 权重 | 0.0/1.0/2.0 | **0.0** | |
| Iter × Scheduler | 400/800/1600 × 4 | **800 + cwu** | 1600 Last 最高但 Transfer 降 |
| 微调层 | qkv+ffn/ffn/attn/qk | **qkv+ffn (All)** | ffh-only 仅低 0.53pp |
| LoRA Rank | 2/4/8/16 | **16 (Avg)** / 4 (平衡) | |
| QKV 方案 | 共享P/独立P/fused QKV | **共享 P** | fused QKV 差 4.6pp |
| Backbone × Text | LoRA/DoRA × 3 schedules | **LoRA + always** | 3/3 配对 LoRA 胜 |

### 10-task 重消融（seed=43, 10 数据集）

| Phase | 内容 | 结论 |
|:---:|------|------|
| Pre-A | LoRA/DoRA × text_schedule 全交叉 | LoRA > DoRA, always 最优 |
| Phase 1 | Wave A lr sweep | lr=1e-4, Ens Last=83.65 (超历史 82.79) |
| Phase 2 | Wave C batch size | **bs=32 最优**（与 6-task 的 128 相反！） |
| Phase 3 | Wave D CD 温度 | temp=4.0 最优 |
| Phase 7 | nsp_eps sweep | **0.20 最优**（+0.42pp vs 0.05） |
| V5 Phases 10-14 | Rank, iter1600, warmup, fd, wd | **全部未超过 V4 基线** |

### 关键洞察

1. **bs=32 在 10-task 反超 128**：更长任务序列下，小 batch 的噪声梯度提供更好的抗遗忘正则化
2. **nsp_eps=0.20 > 0.05**：更宽松的投影在 10-task 上更好，过度约束损害新任务学习
3. **text=always > freeze_after**：文本端全程微调建立更好的跨领域对齐
4. **DoRA 不是 82.79 的原因**：Pre-Wave A 证明 LoRA > DoRA，旧代码优势来自 text≈low_lr_after
5. **V4 配置已收敛**：V5 的 12 个实验全部未超过基线

---

## 6. 论文现状

### 基本信息

- **标题**: "Null-Space Filtered Low-Rank Adaptation for CLIP Class-Incremental Learning"
- **方法名**: **LoRA-NF**（论文用名，代码中用 `lora_nsp`）
- **模板**: NeurIPS 2026 (`neurips_2026.sty`)
- **作者**: Xuan Rao, Yong Zeng, Bo Zhao, Derong Liu, Cesare Alippi

### 完成度

| 章节 | 状态 | 说明 |
|------|:---:|------|
| Abstract | ✅ | 已定稿 |
| Introduction | ✅ 首稿完成 | 7 段结构，待替换占位结果 |
| Related Work | ✅ 定稿 | 三段结构：CIL+PEFT+预测方法 |
| Method | ⚠️ | 主要内容已写但 `\iffalse` 注释，仅 Table 5 有真实数据 |
| Experiments | 🔴 占位 | 所有主表数字为 `\todo{--}` |
| Conclusion | ⚠️ | 有框架但待最终数据 |
| Appendix | ⚠️ | 推导框架已有，证明未补完 |

### 论文叙事核心（当前版本）

**中心矛盾**：有监督持续适配提升新类别辨别但同时侵蚀 CLIP 的预训练跨模态知识

**贡献层次**:
1. **LoRA-NF**（主要创新 + 理论）：ΔW = P A B，持久前向滤波压制历史高能方向的适配
2. **跨模态知识蒸馏 FD/CD**（辅助机制）：维护表示空间的跨模态对齐，主管检索保持
3. **LR-RGDA + CLIP text ensemble**（重要预测组件）：统计判别 + 语义先验 fusion

**机制责任明确划分**:
- LoRA-NF → 历史任务保护（参数空间）
- FD/CD → 跨模态检索保持（表示空间）
- LR-RGDA 集成 → 分类决策（预测空间）

### 正式实验计划（E1-E6）

| 包 | 内容 | 最少实验数 |
|:---:|------|:---:|
| E1 | LADA-style 主实验 (16-shot + full-shot, 3 seeds) | 12 |
| E2 | 组件消融 (C0-C4 累积链) | 12 |
| E3 | LoRA 系列公平对比 (LoRA/LoRA-Null/LoRA-NF) | 9 |
| E4 | LoRA-NF 超参数 (nsp_eps, nsp_weight, 应用层) | 10 |
| E5 | 跨模态蒸馏消融 (cd_weight, cd_temperature) | 8 |
| E6 | 集成分类器消融 (alpha sweep, 离线) | 0 额外训练 |

---

## 7. 目录结构与项目管理

### main_v3 整理后的顶层结构

```
project_clip_continual_learning/
├── main_incremental.py          # 🔑 增量学习入口
├── main_joint.py                # 🔑 联合训练入口
├── src/                         # 核心代码（见 §3）
├── configs/                     # YAML 配置文件
│   ├── base/default.yaml        # 基础配置（⚠️ 参数不是最新消融值）
│   └── experiments/             # 13 个实验配置 (B0-B4 + ablation + joint)
├── scripts/                     # 规范脚本位置
│   ├── debug/                   # 诊断/探索脚本
│   ├── legacy/                  # 旧版入口（可复现性保留）
│   └── README.md                # 脚本分类说明
├── tests/                       # 研究测试 (test_basis_variants, test_dora_init_fix)
├── demos/                       # 探索性 notebook
├── artifacts/                   # 本地中间产物 (不入库)
│   ├── logs/
│   ├── results/
│   ├── figures/
│   ├── tool-runs/
│   └── archive/
├── chat-history/                # 工程/实验记录 (100+ 篇)
├── chat-history-for-paper-writing/ # 论文叙事演化记录 (23 篇)
├── meta-prompts/                # 跨会话项目全局认知 (4 篇)
├── paper_writing/               # 论文写作
│   ├── paper-template/          # LaTeX 模板 + 草稿
│   ├── reference_papers/        # 参考论文 (不入库)
│   ├── my_original_papers/      # 作者推导笔记 (不入库)
│   └── kimi/                    # Kimi paper-writing skills
├── LADA/                        # LADA 源码 (不入库，.gitignore)
├── experiments/                 # 实验输出 (不入库)
├── scenario_datasets/           # 数据集加载器
├── models -> src/models         # 符号链接
├── utils/                       # 旧工具 (部分冗余)
└── configs/                     # 配置文件
```

### .gitignore 关键排除

- `experiments/`, `optimization/` — 实验输出
- `LADA/` — 第三方源码
- `paper_writing/reference_papers/`, `my_original_papers/` — 版权/隐私
- `artifacts/` — 本地中间产物
- `*.pt`, `*.pth`, `*.pkl` — 模型权重
- `*.pdf`, `*.aux`, `*.out` — 编译产物

### ⚠️ 注意事项

- **`.git` 目录已被覆盖**：本地 git 历史只剩 1 个 commit (`ea74d94`)。如需恢复旧历史需从 `YongZ1999` GitHub 重新 clone
- `models/` 是 `src/models/` 的符号链接
- `src/models/trainer.py` 是旧版遗留代码，新训练逻辑在 `src/trainers/lora_nsp_trainer.py`
- `utils/` 和 `src/utils/` 存在部分重复，需注意导入一致性
- `configs/base/default.yaml` 参数不是最新消融最优值（lr=1e-4, nsp_eps=0.05 等是旧默认），以 `main_incremental.py` CLI defaults 为准

---

## 8. 已知问题与待办

### Bug 修复历史（均已在当前代码中修复）

| Bug | 影响 | 修复 |
|------|------|:---:|
| Parser 缺少 `--cd_temperature/--cd_divergence` | CD 消融崩溃 | ✅ 已添加 |
| Parser `--scheduler` choices 不完整 | scheduler 消融崩溃 | ✅ 已扩展 |
| `--lora_target_modules` 覆盖失效 | 层消融结果错误 | ✅ 已修复 |
| `reference_loader` batch_size 硬编码 32 | ref_bs 变体实验白跑 | ✅ 已修复 |
| `--use_soft_projection` parser 缺失 | Soft NSP 无法启动 | ✅ 已添加 |
| DoRA `initialize_adapters_from_covariance` | DoRA 下 init 基于错误权重 | ✅ 已修复 |
| `get_full_stats` NameError | alpha sweep 崩溃 | ✅ 已导入 |
| `main._global_center_means` 全局泄露 | 多进程冲突 | ✅ 改为局部变量 |

### 遗留问题

- [ ] 恢复有效 Git 历史（当前 `.git` 仅 1 个 commit）
- [ ] `configs/base/default.yaml` 同步为最新消融最优值
- [ ] 审计 `src/models/trainer.py` 与 `src/trainers/lora_nsp_trainer.py` 的重复关系
- [ ] 审计 `utils/` 与 `src/utils/` 的重复关系
- [ ] Phase 7a 的 `nsp_weight` 扫描在 Soft NSP 下跑了但 Soft NSP 不读该参数 → **结果无效，不能用于论文**
- [ ] 检索数据集下载（COCO/Flickr30K，当前无权限访问 `/mnt/raoxuan/`）

---

## 9. 正式实验待执行 (E1-E6)

### Gate 0 前置条件

- [ ] 冻结数据划分（train/test splits）
- [ ] 审计检索数据与蒸馏参考数据的去重
- [ ] CLI 参数全面验证
- [ ] Dry run (2 tasks, 100 iter)
- [ ] JSON schema 冻结

### 三波执行

**Wave 1 — 论文循环闭合**（最高优先级）
- E3: LoRA 系列公平对比 (9 runs)
- E5: 跨模态蒸馏消融 (8 runs)
- LoRA-NF 主配置 3-seed 完整跑

**Wave 2 — 主表 + 关键消融**
- E1: LADA-style 主实验 16-shot (12 runs)
- E1: full-shot (需 12 runs)
- E2: 组件消融累积链 (12 runs)

**Wave 3 — 预测 + 机制 + 附录**
- E6: Ensemble alpha sweep (离线，无需训练)
- E4: LoRA-NF 超参数 (10 runs)
- 检索评估（需先下载数据集）

---

## 相关文件索引

| 类型 | 文件 | 说明 |
|------|------|------|
| 入口 | `main_incremental.py` | 增量学习主入口 |
| 入口 | `main_joint.py` | 联合训练入口 |
| 训练 | `src/trainers/lora_nsp_trainer.py` | LoRA-NF 训练器 |
| 模型 | `src/models/lora_sgp.py` | LoRA/DoRA/NSP 核心实现 |
| 模型 | `src/models/clip.py` | CLIP 模型工厂 |
| 分类器 | `src/classifiers/lr_rgda_classifier.py` | LR-RGDA + Ensemble |
| 分类器 | `src/classifiers/gaussian_statistics.py` | 高斯统计 |
| 分类器 | `src/lada/lada_classifier.py` | LADA 分类器 |
| 评估 | `src/utils/main_utils.py` | 数据集评估 + alpha sweep |
| 评估 | `src/utils/continual_metrics.py` | LADA Transfer/Average/Last |
| 数据 | `src/utils/data.py` | X-TAIL 数据加载 |
| 检索 | `src/utils/retrieval_eval.py` | 多模态检索评估 |
| 配置 | `configs/base/default.yaml` | 基础配置 |
| 消融 | `chat-history/2026-07-05-weekly-ablation-review.md` | 6-task 全消融汇总 |
| 消融 | `chat-history/2026-07-06-10task-reablation-plan.md` | 10-task 重消融计划 |
| 消融 | `chat-history/2026-07-12-01-v5-completion-summary.md` | V5 完成总结 |
| 论文 | `paper_writing/paper-template/paper_draft.tex` | 当前论文草稿 |
| 论文 | `chat-history-for-paper-writing/` | 23 篇论文叙事记录 |
| 认知 | `meta-prompts/` | 4 篇项目全局认知 |
| 前次 | `docs/v2-text-lora_merge_analysis.md` | v2-text-lora 合并分析 |
