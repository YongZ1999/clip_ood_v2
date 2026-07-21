# LoRA-NF transfer-aware 分支：合作方论文交接

**代码分支**: `agent/transfer-gated-ensemble`

**结果提交**: `39e403e`（R1/R2 结果）；本文提交会继续记录文档与 history 整理。
**阅读对象**: 接手实验结果并撰写论文的研究者或写作模型。

## 1. 一句话结论

在新的、对 Standard LoRA 与 LoRA-NF 完全一致的 `preserve_aspect + ZS-predicted-seen` 分类协议下，LoRA-NF 的三 seed E1 主表仍同时优于 Standard LoRA 的 Transfer、Average、Last；但 R1 表明 seed 42 上 gate 本身没有改变结果，因此不能将新协议相对旧协议的提升归因于 gate。单 seed R2 进一步支持 CD 对分类 Average/Last 的主要作用，但其检索结果呈现权衡，不能表述为“蒸馏无条件提升检索”。

## 2. 本轮具体修改了什么

| 范畴 | 修改 | 影响范围 |
|---|---|---|
| 分类测试预处理 | `legacy_square` 之外新增 `preserve_aspect = Resize(shorter_edge=224) + CenterCrop(224)` | 只改变 X-TAIL classification test transform；不改变训练或 retrieval |
| 集成推理 | 新增 `zs_predicted_seen`：ZS 预测未来类时保留 ZS，否则使用历史 classwise ZS + LR-RGDA | 只改变分类 logits 融合；不读任务 ID/真值标签，不改编码器 |
| 命令行与离线评估 | `--eval_resize_mode`、`--ensemble_routing` 进入主训练和 artifact evaluator | 支持完整可复现的统一分类协议 |
| 服务器离线加载 | 新增 OpenAI ViT-B/16 `.pt` 到 Hugging Face `CLIPModel` 的兼容转换 | 仅为损坏 HF cache/DNS 不可用时恢复 E1；保持 LoRA-NF 依赖的独立 Q/K/V 模块接口 |
| 开发工具 | 统一六 GPU 的 E1 launcher；artifact sweep 支持 rank 与固定 eval seed | D1/D2 只用于开发诊断，不替换主表配置 |

关键实现文件：

- `src/utils/data.py`
- `src/utils/main_utils.py`
- `src/classifiers/lr_rgda_classifier.py`
- `main_incremental.py`
- `src/models/clip.py`
- `src/models/openai_clip_compat.py`
- `scripts/run_transfer_aware_main_table.sh`
- `scripts/evaluate_incremental_rgda_sweep_artifacts.py`
- `scripts/check_openai_pt_hf_compat.py`

## 3. 当前推荐的正式主表协议

### 模型与训练

OpenAI CLIP ViT-B/16；X-TAIL 任务顺序 aircraft → caltech101 → dtd → eurosat → flowers → food101 → mnist → oxford_pets → stanford_cars → sun397；16-shot 或 full-shot；seeds 42/43/44。

LoRA-NF 保持原有主配方：rank=4、hard NSP `eps=.20` / `weight=.02`、800 iterations/task、AdamW `lr=1e-4`、FD=1、CD=2（temperature=4）；同时调视觉/文本 LoRA。分类器为 LR-RGDA：`M=4`、rank=32、fit=200、`gmm_sample`、alpha=.05、maxshift。Standard LoRA 使用相同训练预算、分类器、预处理与推理协议，只移除 NSP。

### 新分类评估/推理协议

```text
classification test: Resize(shorter_edge=224) -> CenterCrop(224)

if argmax(raw ZS logits) is an unseen class:
    output raw ZS logits
else:
    output historical classwise maxshift(ZS + LR-RGDA) logits
```

这里的“unseen”由当前第 t 个增量任务的已见类别总数定义；规则并不知道测试样本来自哪个任务。

跨模态检索仍在每次完成一个 X-TAIL 任务后，以当时的 encoder 在 MSCOCO 2014 5K 上测 I2T/T2I R@1/5/10；它不使用 LR-RGDA 或分类 gate。

## 4. E1 主结果：三 seed、sample standard deviation

结果来源：`experiments/paper_transfer_aware/E1_main/TRANSFER_AWARE_SUMMARY.md`。论文表格使用该 Markdown 的 **sample std**，不要直接使用同目录 JSON 的 population std 字段。

| 方法 | Ens Transfer | Ens Average | Ens Last |
|---|---:|---:|---:|
| LoRA-NF 16-shot | **61.91 ± 0.22** | **72.63 ± 0.05** | **83.72 ± 0.10** |
| Standard LoRA 16-shot | 61.59 ± 0.15 | 71.75 ± 0.34 | 82.04 ± 0.29 |
| LoRA-NF full-shot | **61.89 ± 0.15** | **74.98 ± 0.01** | **86.19 ± 0.02** |
| Standard LoRA full-shot | 61.56 ± 0.33 | 74.33 ± 0.11 | 84.90 ± 0.06 |

LoRA-NF 减 Standard LoRA（Ensemble）：

- 16-shot：`+0.32 / +0.89 / +1.68`（Transfer / Average / Last）；
- full-shot：`+0.33 / +0.65 / +1.29`。

建议论文主张：在相同 transfer-aware 评估、相同 LR-RGDA 设置下，LoRA-NF 在两个 shot regime 中稳定优于 Standard LoRA 的三项 continual metrics。不要把 LoRA-NF 与历史 `paper_formal` 的 LoRA-Null/GradProj 数字直接置入同一绝对值排名，因为两套分类协议不同。

### E1 检索结果（MSCOCO 5K R@1）

| 方法 | Avg I2T | Avg T2I | Last I2T | Last T2I |
|---|---:|---:|---:|---:|
| LoRA-NF 16-shot | 51.97 ± 0.03 | 33.30 ± 0.08 | 51.88 ± 0.27 | 33.68 ± 0.15 |
| Standard LoRA 16-shot | 51.83 ± 0.04 | 33.28 ± 0.07 | 51.71 ± 0.18 | 33.83 ± 0.13 |
| LoRA-NF full-shot | 52.15 ± 0.16 | 33.53 ± 0.02 | 51.92 ± 0.37 | 33.97 ± 0.07 |
| Standard LoRA full-shot | 52.00 ± 0.09 | 33.43 ± 0.02 | 52.05 ± 0.24 | 33.93 ± 0.06 |

可写“未观察到显著检索退化”。不应写“两种检索方向都严格提高”：差异较小，且 16-shot Last T2I 与 full-shot Last I2T 对 Standard LoRA 略低。

## 5. 这轮改动对旧结果的实际影响

历史 `paper_formal` 的 LoRA-NF 16-shot Ensemble 是 `60.14 / 71.75 / 83.74`；E1 transfer-aware 新协议为 `61.91 / 72.63 / 83.72`（T/A/L）。差值为 `+1.77 / +0.88 / -0.02`。

这是**评估/推理协议组合**的变化，不是新的 LoRA-NF 训练模块或证明 NSP/蒸馏变强的证据。R1 对 seed 42 的控制比较见下一节，直接表明 gate 在该轨迹上不产生数值差异；因此不能把上述跨协议变化单独归因于 gate，也不宜与 legacy 结果做“方法提升”叙事。

本地 native LADA 复现的既有 16-shot结果为 `61.6 ± 0.2 / 72.3 ± 0.4 / 83.0 ± 0.2`，见 `docs/paper_experiment_results.md` 第 6.2 节。E1 LoRA-NF 16-shot 在数值上为 `+0.31 / +0.33 / +0.72`，但 LADA 和 LoRA-NF 的训练流程、分类器并不相同；将其写作既定端到端 protocol 的对比，而非严格 head-matched 或训练预算等价的结论。LADA 原论文数字需在最终定稿前按任务顺序、模板、预处理与指标定义再次核验，不能与本地复现混为同一来源。

## 6. R1 / E6-TA：新路由的隔离检查（seed 42）

对同一 D1 编码器 snapshots、同一 `M=4/gmm_sample/fit=200/alpha=.05`，仅切换 routing：

| Routing | Transfer | Average | Last |
|---|---:|---:|---:|
| `classwise` | 61.6812 | 72.7391 | 83.7318 |
| `zs_predicted_seen` | 61.6812 | 72.7391 | 83.7318 |

两个 accuracy matrix 完全相同。这说明在此 seed/配置下，gate 没有改变最终分类结果；它可以保留为安全的、部署可用的规则，但不能作为提高 Transfer 的实证贡献来宣传。R1 是推理端 ablation，不会改变 embedding，所以没有重复运行 retrieval。

原始文件：`experiments/paper_transfer_aware/E6_gate_seed42/`。

## 7. R2 / E2-TA：FD/CD 与分类、检索（单 seed 42）

所有条件均为 transfer-aware 16-shot LoRA-NF；C3 是已有的同配置 D1 轨迹，未重训。**这只是一个 seed，不能报告显著性。**

| 条件 | FD | CD | Ens T | Ens A | Ens L | Avg I2T | Avg T2I | Last I2T | Last T2I |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0 | 0 | 61.95 | 71.84 | 83.11 | 51.17 | 33.49 | 52.62 | 34.35 |
| C1 | 1 | 0 | **61.99** | 72.08 | 83.35 | 50.15 | 32.45 | 51.92 | 33.83 |
| C2 | 0 | 2 | 61.68 | 72.71 | **83.89** | 51.93 | 33.20 | 52.12 | 33.74 |
| C3 | 1 | 2 | 61.68 | **72.72** | 83.80 | **51.94** | 33.20 | 52.16 | 33.71 |

相对于 C0，C1（FD-only）的分类 Average/Last 为 `+0.24/+0.24`，但两个检索 Average R@1 均下降约 1 点。C2（CD-only）带来 `-0.27/+0.87/+0.78` 的分类 T/A/L 变化，I2T Average 增加 0.76、T2I Average 减少 0.29。C3 相对 C2 仅有极小的分类/检索差异，尚无 FD 额外稳定收益的强证据。

稳妥写法：CD 是该单 seed 上提高 classification Average/Last 的主要成分；但它与检索的关系是细粒度 trade-off，必须以“逐任务 MSCOCO 5K 监测结果”呈现，不能声称任意蒸馏组合都更好地维持双向检索。若要把 E2-TA 放在正文并作强机制结论，应优先补 C0 vs C2（或 C0/C2/C3）多 seed。

原始文件：`experiments/paper_transfer_aware/E2_distill_seed42/`。

## 8. D1/D2 classifier sweep：不要作为正式结果替换 E1

D1/D2 复用一条 seed-42 encoder artifact 轨迹扫描 alpha、centers、fit iterations、GMM replay source 与 rank。表面最优的 rank=8/alpha=.05 为 `61.6815 / 72.8349 / 84.0744`，但相对基线的 Average 只有约 `+0.08`；rank=8 与 rank>=15 的净差约 `.004`。未达到事先规定的 Average `+0.20` 门槛。

所以正式 E1 保持 `rank=32/M=4/gmm_sample/fit=200/alpha=.05`；D1/D2 可在补充材料说明“classifier-only tuning 已饱和”，但不要以此替换三 seed 主表或制造新的最优版本。

## 9. 哪些旧实验需要重跑、哪些不需要

| 实验 | 状态 | 论文使用建议 |
|---|---|---|
| E1 LoRA-NF vs Standard LoRA | 已按新协议三 seed 完成 | 使用 transfer-aware 主表 |
| R1 gate | 已完成，单 seed，无训练 | 仅说明 gate 在 seed 42 无可测差异 |
| R2 FD/CD | 已完成，单 seed，含 retrieval | 机制趋势；避免显著性/绝对优劣措辞 |
| 历史 E2--E5 | 不重跑 | 标记为 legacy protocol 机制/超参证据，不能混入新分类主表 |
| 旧 E3 LoRA-Null/GradProj | 不重跑，除非需要同新协议横排 | 否则保留为 legacy adapter comparison |
| retrieval-only、E4、Frozen CLIP、LADA、SigLIP2 | 不因本轮改动重跑 | 本轮未改其 encoder/retrieval 流程 |

## 10. 复现与材料入口

- E1 launcher：`scripts/run_transfer_aware_main_table.sh`
- E1 protocol：`docs/transfer_aware_main_experiment.md`
- R1/R2 plan：`docs/transfer_aware_followup_experiment_plan.md`
- server prompts：`docs/server_model_prompt_transfer_aware_main.md`、`docs/server_model_prompt_transfer_aware_followups.md`
- E1 source results：`experiments/paper_transfer_aware/E1_main/`
- R1 source results：`experiments/paper_transfer_aware/E6_gate_seed42/`
- R2 source results：`experiments/paper_transfer_aware/E2_distill_seed42/`
- legacy/full experiment ledger：`docs/paper_experiment_results.md`
- independent result audits：`chat-history/2026-07-21-01-transfer-aware-e1-result-audit.md`、`chat-history/2026-07-21-03-phase-d1-single-seed-audit.md`、`chat-history/2026-07-22-01-phase-d2-rank-sweep-audit.md`

## 11. 推荐论文表述边界

可以写：

- “在统一的 transfer-aware protocol 下，LoRA-NF 在 16-shot 与 full-shot 上均优于 Standard LoRA 的 Transfer、Average 和 Last。”
- “在任务序列过程中，LoRA-NF 的 MSCOCO 5K 检索没有观察到显著退化。”
- “单 seed 组件分析显示 CD 是分类 Average/Last 的主要贡献来源，FD 的附加作用较弱；该趋势仍需多 seed 验证。”

不要写：

- “gate 导致了 Transfer 提升”（R1 未支持）。
- “新协议相对旧协议的所有增益来自 LoRA-NF 方法”（预处理与推理共同改变）。
- “所有蒸馏组合都维持或改善检索”（R2 不支持）。
- “LoRA-NF 在不同 backbone / 不同分类器 / 不同训练预算下全面胜过 native LADA”（当前证据不支持）。
