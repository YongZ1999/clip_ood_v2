# Transfer-aware 最终证据补齐计划

**分支**: `agent/transfer-gated-ensemble`

**目标**: 在不继续进行 classifier-only 调参的前提下，补齐当前论文最缺少的多 seed 证据：新协议下的蒸馏组件结论与 adapter family 对比。所有训练使用已完成 E1 完全一致的 OpenAI CLIP ViT-B/16、X-TAIL 顺序和训练配方。

## 1. 已完成、不可重做的结论

| 项目 | 当前证据 | 决策 |
|---|---|---|
| E1 LoRA-NF vs Standard LoRA | 16-shot/full-shot，各 3 seeds | 正式主表；不重跑 |
| D1/D2 LR-RGDA sweep | seed 42；最大 Average 改善低于预注册的 +0.20 | 停止 classifier 调参 |
| R1/E6 gate | seed 42；classwise 与 `zs_predicted_seen` 准确率矩阵完全相同 | 不扩大 seed |
| R2/E2-TA FD/CD | seed 42；CD 有分类信号，检索存在方向性权衡 | 补齐多 seed |
| LADA、Frozen CLIP、SigLIP2 | 已有独立结果；本轮未改其 encoder/retrieval 流程 | 不重跑 |
| E4/E5 NSP/CD 超参数 | 已有 legacy protocol 的机制证据 | 不再扫超参数 |

不要在本计划之外扫描 alpha、RGDA rank/fit/M、训练步数、`nsp_weight`、NSP epsilon、CD temperature 或 CD weight。这些会将最终结果变成事后优化，且 D1/D2 已显示分类器方向饱和。

## 2. 统一固定配置

除实验表中明确列出的差异外，**每个训练任务必须逐字匹配** `scripts/run_transfer_aware_main_table.sh` 的 16-shot LoRA-NF 配置：

```text
OpenAI CLIP ViT-B/16
X-TAIL 10-task order
16-shot; batch_size=32; iterations=800; AdamW lr=1e-4; cosine_with_warmup
rank=4; targets=q,k,v,out,fc1,fc2; tune vision + text; text LoRA rank=4
hard NSP eps=.20, weight=.02 (where the method uses NSP)
CD divergence=kl_forward; temperature=4; aux=0
LR-RGDA: M=4, rank=32, fit=200, gmm_sample, alpha=.05, maxshift
classification: eval_resize_mode=preserve_aspect;
ensemble_routing=zs_predicted_seen
MSCOCO 2014 5K retrieval after every incremental task (I2T/T2I R@1/5/10)
```

必需的离线加载环境变量与 E1 相同：

```text
CLIP_USE_SAFETENSORS=0
CLIP_LOCAL_FILES_ONLY=1
CLIP_OPENAI_PT_FALLBACK=1
```

训练前必须运行：

```bash
python scripts/check_openai_pt_hf_compat.py --device cuda:0
```

若该检查失败，停止并报告，不要将实验静默改成原生 `clip.load()` 或改用不同 backbone。

## 3. Phase A — E2-TA 蒸馏组件补齐（最高优先级，6 个训练任务）

### 科学问题

在新的 classification protocol 下，FD 与 CD 分别如何影响 continual classification 和跨模态 retrieval？当前 seed-42 R2 只能说明趋势，不能作为带方差的组件结论。

### 运行矩阵

seed 42 的 C0/C1/C2 已完成于 `experiments/paper_transfer_aware/E2_distill_seed42/`；只补 43、44。完整 C3 不重训，直接复用 E1 的 LoRA-NF 16-shot 三 seed 结果。

| Condition | FD | CD | 新跑 seeds | 新增训练 |
|---|---:|---:|---|---:|
| C0 | 0 | 0 | 43, 44 | 2 |
| C1 | 1 | 0 | 43, 44 | 2 |
| C2 | 0 | 2 | 43, 44 | 2 |
| C3 | 1 | 2 | none; reuse E1 seeds 42/43/44 | 0 |

新结果写入 `experiments/paper_transfer_aware/E2_distill_3seed/`，日志写入 `logs/paper_transfer_aware/E2_distill_3seed/`。命名必须是：

```text
E2TA__C0__seed43, E2TA__C0__seed44,
E2TA__C1__seed43, E2TA__C1__seed44,
E2TA__C2__seed43, E2TA__C2__seed44.
```

最终聚合时，C0/C1/C2 的 seed42 来自旧 R2 目录，43/44 来自本轮目录；C3 三个 seed 都来自 `experiments/paper_transfer_aware/E1_main/E1TA__lora_nf__16shot__seed{42,43,44}_*`。不得用 `dev_seed42` 的 C3 混入最终三 seed 表。

### 产物与验收

生成 `experiments/paper_transfer_aware/E2_distill_3seed/E2_TA_3SEED_SUMMARY.md`，其中必须含：

- 每个 C0--C3 的 ZS 与 Ensemble Transfer/Average/Last，mean ± **sample std**；
- 每个条件的 MSCOCO 5K I2T/T2I R@1/5/10：跨任务 Average 与 Task-10 Last，再对三 seed 计算 mean ± sample std；
- 每个原始 JSON 路径及 seed；
- C1-C0、C2-C0、C3-C2 的差值；
- 完整 command、commit、失败/重跑说明。

只有当 C2/C3 相对 C0 的趋势在三个 seed 中一致时，才可在正文作较强的“CD 是主要组件”表述；否则仅保留描述性表格。

## 4. Phase B — E3-TA adapter family 公平对比（6 个训练任务）

### 科学问题

当前 LoRA-Null 与 GradProj 的三 seed 结果使用 legacy `legacy_square + classwise` 分类协议。若论文要在主文中给出 “LoRA-NF > GradProj > LoRA-Null > LoRA” 的绝对排名，必须让所有方法处在同一 transfer-aware protocol 下。

E1 已提供 Standard LoRA 与 LoRA-NF 的 16-shot 三 seed；因此只补齐两个缺失方法。

| Method | 关键覆盖 | seeds | 新增训练 |
|---|---|---|---:|
| Standard LoRA | reuse E1 | 42,43,44 | 0 |
| LoRA-Null | `lora_nsp`, `null_init_mode=history_init_only`, `projection_param_mode=full` | 42,43,44 | 3 |
| Gradient-projected LoRA | `lora_nsp`, `use_gradient_projection`, `null_init_mode=none`, `projection_param_mode=full` | 42,43,44 | 3 |
| LoRA-NF | reuse E1 | 42,43,44 | 0 |

两种补跑方法都必须保留 `fd_weight=1`、`cd_weight=2`，并启用相同分类及 retrieval 评估。仅改变上述 method flag；不得改变任务顺序、训练步数、LoRA target、NSP 参数、分类器或 eval protocol。

输出写入 `experiments/paper_transfer_aware/E3_adapters_3seed/`，日志写入 `logs/paper_transfer_aware/E3_adapters_3seed/`；命名：

```text
E3TA__lora_null__seed{42,43,44}
E3TA__gradproj__seed{42,43,44}
```

### 产物与验收

生成 `experiments/paper_transfer_aware/E3_adapters_3seed/E3_TA_3SEED_SUMMARY.md`，将两种新运行与 E1 的 Standard LoRA / LoRA-NF 16-shot JSON 合并，报告：

- 四种方法各自的 ZS、Ensemble Transfer/Average/Last，mean ± sample std；
- 四种方法的 MSCOCO 5K I2T/T2I R@1/5/10 的 Average/Last；
- 每种 method 的三个 seed 原始 JSON 路径；
- 只根据 Ensemble 主表给出排名，同时说明检索差异是否仅为小幅差异。

若排序与 legacy E3 不同，应如实报告新协议结果并不再沿用旧排名；不允许选择性保留更好的一套数值。

## 5. 资源与执行顺序

总新增 encoder-training runs：**12**。六张卡可分两波：

1. 第一波：Phase A 的 6 个 E2 任务，一卡一个；
2. 第二波：Phase B 的 6 个 E3 任务，一卡一个；
3. 阶段完成后在一张空闲 GPU 上做结果汇总和原始 JSON 完整性检查。

不要将 Phase A 与 B 混在同一输出目录；若 GPU 暂时不空闲，排队等待，绝不杀无关任务。每个 run 完成后检查存在 `*_zs_results.json`、`*_rgda_results.json`、`*_ens_results.json`、`*_retrieval.json` 与主 `*.json`。

## 6. 最终的证据层级

完成本计划后，论文可使用：

- **正式三 seed 主结果**：E1（LoRA-NF vs Standard LoRA）；
- **正式三 seed 组件与检索证据**：E2-TA（C0--C3）；
- **正式三 seed adapter family 对比**：E3-TA（LoRA / LoRA-Null / GradProj / LoRA-NF）；
- **辅助单 seed 诊断**：R1 gate、D1/D2 classifier saturation；不作为强性能主张。

本计划完成前，E2-TA 与 E3-TA 只能作为单 seed / legacy-support evidence，不应与 E1 主表具有同等强度。
