# 论文正式实验执行计划

**日期**: 2026-07-13
**基于**: `chat-history-for-paper-writing/2026-07-12-20-formal-experiment-plan.md`（原始计划）
**代码版本**: `main_v3` (`ea74d94`)
**执行人**: Yong Zeng

---

## 0. 环境确认

### 硬件
| 资源 | 状态 |
|------|:---:|
| GPU 0-5 | 6× RTX 4090 24GB，全部空闲 |
| 单实验耗时 | ~50 分钟（10 task × 800 iter） |

### 软件
| 组件 | 版本/值 |
|------|------|
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| HF_ENDPOINT | `https://hf-mirror.com` |

### 数据路径
| 数据 | 路径 | 状态 |
|------|------|:---:|
| X-TAIL 训练 | `/data1/open_datasets/X-TAIL/` | ✅ |
| flickr8k 蒸馏参考 | `/data1/open_datasets/flickr8k/` (8091 images) | ✅ |
| 检索数据集 | `/mnt/raoxuan/open_datasets/` → 需换路径 | ❌ 无权限 |

---

## 1. Gate 0：正式实验前必须完成的代码修复（5 项）

### G0-1: Full-shot 结果 JSON 元数据修复 🔴

**问题**：当 `--full_shot` 使用时，输出 JSON 仍写 `"num_shots": 16`，无法区分 16-shot 和 full-shot。

**位置**：`main_incremental.py` 第 1387 行和第 1412 行

```python
# 当前（错误）
"num_shots": args.num_shots,

# 修复为
"num_shots": args.num_shots,
"full_shot": args.full_shot,
```

**影响文件**：`_zs_results.json`、`_rgda_results.json`、`_ens_results.json`、`_lada_results.json`、汇总 JSON。

### G0-2: `--no-alpha_sensitivity` 无法使用 🔴

**问题**：`parser.add_argument("--alpha_sensitivity", action="store_true", default=True)` 配合 `store_true` + `default=True` → argparse 无法生成 `--no-alpha_sensitivity` 前缀，参数永远为 True，无法从 CLI 关闭。

**验证**：已实测 `--no-alpha_sensitivity` 报 `unrecognized arguments`。

**位置**：`main_incremental.py` 第 622-624 行

**修复**：
```python
parser.add_argument("--alpha_sensitivity", action="store_true", default=True,
                    help="...")
parser.add_argument("--no-alpha_sensitivity", action="store_false",
                    dest="alpha_sensitivity",
                    help="Disable per-task alpha sensitivity sweep.")
```

### G0-3: 检索数据集路径修改 🔴

**问题**：`main_incremental.py` 中所有检索路径默认指向 `/mnt/raoxuan/open_datasets/`，无读取权限。

**涉及参数**（第 738-747 行）：
- `--retrieval_root`：`/mnt/raoxuan/open_datasets`
- `--retrieval_roots`：`mscoco_2014_5k=/mnt/raoxuan/open_datasets/mscoco_2014_5k_test_hf`, `flickr30k_hf=/mnt/raoxuan/open_datasets/flickr30k_hf`

**修复**：修改默认值为 `/data/home/zengyong1/dataset/`：
```python
parser.add_argument("--retrieval_root", type=str, default="/data/home/zengyong1/dataset", ...)
parser.add_argument("--retrieval_roots", type=str,
                    default="flickr8k=/mnt/open_datasets/flickr8k,"
                            "flickr30k_hf=/data/home/zengyong1/dataset/flickr30k_hf,"
                            "mscoco_2014_5k=/data/home/zengyong1/dataset/mscoco_2014_5k_test_hf",
                    ...)
```

### G0-4: 下载检索数据集到新路径 🔴

下载 `mscoco_2014_5k`（5000 images + CSV）和 `flickr30k_hf`（31014 rows, parquet）到 `/data/home/zengyong1/dataset/`。

已存在 `scripts/download_retrieval_datasets.sh`，修改 `ROOT` 即可。

### G0-5: Dry run 验证 🟡

用 2 个任务（aircraft → caltech101）、100 iter、seed=42 完成端到端 dry run，验证：
- 分类结果 JSON 格式正确（含 `full_shot` 字段）
- `--enable_retrieval_eval` 检索评估正常
- `--no-alpha_sensitivity` 可正常关闭 sweep
- 无 OOM、NaN

---

## 2. 统一 CLI 模板

所有正式实验使用以下公共模板（仅修改被消融的变量）：

```bash
python -u main_incremental.py \
  --root /data1/open_datasets/X-TAIL \
  --dataset_sequence aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397 \
  --num_shots 16 \
  --batch_size 32 \
  --eval_batch_size 128 \
  --iterations 800 \
  --train_budget_mode uniform \
  --optimizer adamw \
  --lr 1e-4 \
  --weight_decay 3e-5 \
  --scheduler cosine_with_warmup \
  --warmup_ratio 0.1 \
  --eta_min 0.0 \
  --lora_type lora_nsp \
  --init_mode lora_nsp \
  --use_dora false \
  --lora_rank 4 \
  --lora_alpha 4 \
  --lora_dropout 0.0 \
  --lora_target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
  --projection_param_mode full \
  --null_init_mode none \
  --nsp_eps 0.20 \
  --nsp_weight 0.02 \
  --reference_dataset flickr8k \
  --reference_batch_size 32 \
  --fd_weight 1.0 \
  --cd_weight 2.0 \
  --cd_divergence kl_forward \
  --cd_temperature 4.0 \
  --aux_weight 0.0 \
  --tune_vision_encoder true \
  --tune_text_encoder true \
  --text_lora_rank 4 \
  --text_tuning_schedule always \
  --text_classifier_mode lada_hybrid \
  --classifier_feature_transform test \
  --rgda_rank 32 \
  --rgda_alpha1 0.2 \
  --rgda_alpha2 2.0 \
  --rgda_alpha3 0.5 \
  --num_centers 4 \
  --rgda_train_iter 200 \
  --rgda_train_lr 0.01 \
  --rgda_fit_source gmm_sample \
  --alpha 0.05 \
  --ensemble_normalize maxshift \
  --temperature 1.0 \
  --enable_retrieval_eval \
  --retrieval_datasets mscoco_2014_5k,flickr30k_hf \
  --retrieval_batch_size 128 \
  --retrieval_recall_ks 1,5,10 \
  --retrieval_max_images 0 \
  --disable_lada \
  --seed 43 \
  --gpu 0
```

**说明**：
- `--disable_lada`：主实验不跑 LADA 分类器（节省时间）；跑 LADA 基线时移除
- Full-shot 时：移除 `--num_shots 16`，添加 `--full_shot`
- 三个方法的精确覆盖（E3）见 §4.3

### 三种 LoRA 对比方法的 CLI 覆盖

| 方法 | CLI 覆盖 |
|------|------|
| **Standard LoRA** | `--lora_type lora_vanilla --init_mode lora_vanilla --null_init_mode none` |
| **LoRA-Null** | `--lora_type lora_nsp --init_mode lora_nsp --projection_param_mode full --null_init_mode history_init_only` |
| **LoRA-NF** | 公共模板（无需改动） |

---

## 3. 六组实验矩阵

### E1: LADA-style 主实验

**目的**：论文主表。16-shot + full-shot × 3 seeds × 分类 + 检索。

**内部方法（必须）**：

| 方法 | 16-shot seeds | Full-shot seeds | 训练 runs |
|------|:---:|:---:|:---:|
| Standard LoRA + FD/CD | 42/43/44 | 42/43/44 | 6 |
| LoRA-NF + FD/CD | 42/43/44 | 42/43/44 | 6 |
| **小计** | | | **12** |

**外部基线（强烈建议）**：

| 方法 | 16-shot seeds | Full-shot seeds | 入口 |
|------|:---:|:---:|------|
| Frozen CLIP | 1 次评估 | 1 次评估 | 不训练，仅评估 |
| LADA | 42/43/44 | 42/43/44 | `scripts/legacy/main_incremental_lada.py` |

**输出**：分类 Transfer/Average/Last（ZS + Ensemble）、检索 R_Avg/R_Last（I2T/T2I R@1/5/10）、逐任务曲线。

### E2: 组件消融

**目的**：归因 LoRA-NF、FD、CD、Ensemble 各自的贡献。

**全用 LoRA-NF + 16-shot**：

| 编号 | 配置 | fd_weight | cd_weight | 训练 runs |
|:---:|------|:---:|:---:|:---:|
| C0 | 纯 LoRA-NF | 0 | 0 | 3 |
| C1 | +FD | 1.0 | 0 | 3 |
| C2 | +CD | 0 | 2.0 | 3 |
| C3 | C0+C1+C2 = 完整训练端 | 1.0 | 2.0 | **复用 E1** |
| C4 | C3 + Ensemble | — | — | **复用 E1** |

**实际新增**：C0 + C1 + C2 × 3 seeds = **9 runs**

### E3: LoRA 系列公平对比

**目的**：验证 LoRA-NF 持续前向过滤优于普通 LoRA 和仅在初始化时使用零空间的 LoRA-Null。

| 方法 | seeds | 新增 runs | 说明 |
|------|:---:|:---:|------|
| Standard LoRA + FD/CD | 42/43/44 | — | **复用 E1 16-shot** |
| LoRA-NF + FD/CD | 42/43/44 | — | **复用 E1 16-shot** |
| LoRA-Null + FD/CD | 42/43/44 | **3** | 唯一新增 |

**实际新增**：**3 runs**

> **未冻结决策**：Gradient-projected LoRA 是否纳入 E3 正文核心对照？计划 §13 列为未冻结。如需纳入需先实现 gradient projection 代码路径（新增 3 runs）。该问题**需要你拍板**。

### E4: LoRA-NF 超参数消融

**16-shot，先用 seed=43，关键端点补 42/44。**

#### E4a: nsp_eps（4 runs）
| 实验名 | nsp_eps |
|------|:---:|
| eps0p02 | 0.02 |
| eps0p05 | 0.05 |
| eps0p10 | 0.10 |
| eps0p20 | 0.20（复用公共模板） |

#### E4b: nsp_weight（新增 4 runs）
| 实验名 | nsp_weight |
|------|:---:|
| weight0 | 0（严格投影） |
| weight0p02 | 0.02（复用公共模板） |
| weight0p04 | 0.04 |
| weight0p08 | 0.08 |
| weight0p16 | 0.16 |

#### E4c: 应用层（新增 2 runs）
| 实验名 | lora_target_modules |
|------|------|
| attn_only | `q_proj,k_proj,v_proj,out_proj` |
| ffn_only | `fc1,fc2` |
| attn_ffn | 全层（复用公共模板） |

**E4 总计**：4 + 4 + 2 = **10 runs**（seed=43）

### E5: 跨模态蒸馏消融

**16-shot, LoRA-NF, seed=43。固定 `fd_weight=0`。**

#### E5a: cd_weight（5 runs）
| 实验名 | cd_weight |
|------|:---:|
| cd0 | 0 |
| cd0p5 | 0.5 |
| cd1p0 | 1.0 |
| cd2p0 | 2.0（复用公共模板） |
| cd4p0 | 4.0 |

#### E5b: cd_temperature（新增 3 runs，固定 `cd_weight=2.0`）
| 实验名 | cd_temperature |
|------|:---:|
| temp1p0 | 1.0 |
| temp2p0 | 2.0 |
| temp4p0 | 4.0（复用公共模板） |
| temp8p0 | 8.0 |

**E5 总计**：5 + 3 = **8 runs**（seed=43）

### E6: 集成分类器消融

**0 训练 runs**。从 E1 LoRA-NF 16-shot seed=43 的已保存 checkpoint/artifact 离线扫描 Ensemble alpha ∈ `{0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0}`。不需要重新训练。

---

## 4. 实验去重与复用总表

| 实验包 | 配置 | seeds | 总需求 | 复用来源 | 实际新增 |
|:---:|------|:---:|:---:|------|:---:|
| E1 | LoRA 16-shot | 42/43/44 | 3 | — | 3 |
| E1 | LoRA-NF 16-shot | 42/43/44 | 3 | — | 3 |
| E1 | LoRA full-shot | 42/43/44 | 3 | — | 3 |
| E1 | LoRA-NF full-shot | 42/43/44 | 3 | — | 3 |
| E2 | C0 | 42/43/44 | 3 | — | 3 |
| E2 | C1 | 42/43/44 | 3 | — | 3 |
| E2 | C2 | 42/43/44 | 3 | — | 3 |
| E2 | C3 | 42/43/44 | — | E1 LoRA-NF 16-shot | 0 |
| E2 | C4 | 42/43/44 | — | E1 LoRA-NF 16-shot | 0 |
| E3 | LoRA-Null | 42/43/44 | 3 | — | 3 |
| E3 | LoRA | 42/43/44 | — | E1 LoRA 16-shot | 0 |
| E3 | LoRA-NF | 42/43/44 | — | E1 LoRA-NF 16-shot | 0 |
| E4 | eps/weight/层 | 43 | 10 | — | 10 |
| E5 | cd_weight/temp | 43 | 8 | — | 8 |
| E6 | alpha sweep | — | 0 | E1 checkpoint | 0 |
| **16-shot 小计** | | | | | **36** |
| **Full-shot 小计** | | | | | **6** |
| **总计** | | | | | **42 encoder training runs** |

不含 LADA 基线（每个 shot × 3 seeds = 额外 6 runs）、Gradient-projected LoRA（如纳入额外 3 runs）、E4/E5 关键端点补种子（约 10 runs）。

---

## 5. 执行顺序与时间估算

**单实验: ~50 min，保守按 1h。6 GPU 可并行，每轮 6 实验。**

### Wave 1: 论文核心闭环（最高优先级）

| 批次 | 实验 | GPU 分配 | 耗时 | 累积 |
|------|------|:---:|:---:|:---:|
| W1.1 | E1 LoRA 16-shot ×3 + LoRA-NF 16-shot ×3 | GPU 0-5 各 1 | ~1h | 1h |
| W1.2 | E2 C0/C1/C2 seed=42(3) + E3 LoRA-Null seed=42/43/44(3) | GPU 0-5 | ~1h | 2h |
| W1.3 | E2 C0/C1/C2 seed=43(3) + C0/C1/C2 seed=44(3) | GPU 0-5 | ~1h | 3h |

**Wave 1 完成 = 论文 16-shot 主表 + 组件消融 + LoRA 对比 + 检索 全部可用。约 3 小时。**

### Wave 2: 主表 + 关键消融

| 批次 | 实验 | GPU 分配 | 耗时 | 累积 |
|------|------|:---:|:---:|:---:|
| W2.1 | E1 LoRA full-shot ×3 + LoRA-NF full-shot ×3 | GPU 0-5 | ~1h | 4h |
| W2.2 | E4 nsp_eps{0.02,0.05,0.10} + nsp_weight{0,0.04,0.08} | GPU 0-5 | ~1h | 5h |
| W2.3 | E4 nsp_weight{0.16} + 应用层{attn,ffn} + E5 cd_weight{0,0.5,1.0} | GPU 0-5 | ~1h | 6h |
| W2.4 | E5 cd_weight{4.0} + cd_temp{1.0,2.0,8.0} | GPU 0-3 | ~1h | 7h |

**Wave 2 完成 = full-shot 主表 + LoRA-NF 超参数 + CD 消融全部可用。约 4 小时。**

### Wave 3: 附录 + 补种子 + LADA 基线

| 批次 | 实验 | GPU 分配 | 耗时 | 累积 |
|------|------|:---:|:---:|:---:|
| W3.1 | E4 关键端点 seeds 42/44（约 6 runs） | GPU 0-5 | ~1h | 8h |
| W3.2 | E5 关键端点 seeds 42/44（约 6 runs） | GPU 0-5 | ~1h | 9h |
| W3.3 | LADA 16-shot × 3 seeds + full-shot × 3 seeds | GPU 0-5 | ~1h | 10h |
| W3.4 | E6 离线 alpha sweep | CPU | ~0.5h | 10.5h |

**Wave 3 完成后全部实验数据齐全。总训练时间约 10-11 小时。**

---

## 6. 输出与命名规范

### 目录结构
```
experiments/paper_formal/
├── E1_main/
│   ├── 16shot/
│   └── fullshot/
├── E2_components/
├── E3_adapters/
├── E4_lora_nf_hparams/
├── E5_distillation/
└── E6_ensemble/
```

### 实验命名
```
{实验包}__{方法}__{shots}shot__seed{seed}__{覆盖参数}
```

示例：
```
E1__lora_nf__16shot__seed43
E4__lora_nf__16shot__seed43__nsp_eps0p02
E5__lora_nf__16shot__seed43__cd_weight0
```

### 每个 run 必须生成
- `{name}_zs_results.json` — ZS 分类结果 + accuracy_matrix
- `{name}_ens_results.json` — Ensemble 分类结果 + per-task alpha
- `{name}_rgda_results.json` — RGDA-only 内部诊断
- `{name}_retrieval.json` — 每任务双向 R@K
- `{name}.json` — 汇总（含完整 arguments）
- `stdout.log` — 完整训练日志

---

## 7. 当前未冻结事项（需要你确认）

| # | 事项 | 计划中的状态 | 建议 |
|:---:|------|:---:|------|
| 1 | **Gradient-projected LoRA 是否纳入 E3？** | §13 未冻结 | 建议：先跑完 Wave 1 核心三方法，如果 LoRA-NF vs LoRA-Null 差异不够显著，再加 gradient-projected LoRA 强化论证 |
| 2 | **LADA 基线 full-shot 训练预算** | 统一 800 iter vs lada_epochs | 建议：先用 800 iter 统一预算跑全部方法；LADA 原论文数字单独标注 |
| 3 | **其他强基线（ZSCL, Mod-X, RAIL 等）** | §13 未冻结 | 建议：E1 先跑内部方法 + LADA，其他基线视投稿需要再补 |
| 4 | **Checkpoint 保存策略** | 全量 vs 增量 artifact | 建议：全用 `--save_step_artifacts --async_eval_dir` 保存每任务 snapshot，离线扫 alpha |

---

## 8. 实验启动前 Checklist

- [ ] G0-1: Full-shot JSON 元数据修复
- [ ] G0-2: `--no-alpha_sensitivity` 修复
- [ ] G0-3: 检索数据集路径改为 `/data/home/zengyong1/dataset/`
- [ ] G0-4: 下载 `mscoco_2014_5k` + `flickr30k_hf` 到本地
- [ ] G0-5: 2-task dry run 通过
- [ ] 确认三个未冻结事项（§7）
- [ ] 冻结结果 JSON schema
- [ ] Git commit 所有 Gate 0 修改
