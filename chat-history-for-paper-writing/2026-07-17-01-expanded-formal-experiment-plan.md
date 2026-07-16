# 扩展正式实验计划：SigLIP 2 鲁棒性与检索组件归因

**日期**：2026-07-17  
**状态**：当前有效；补充并优先于 `2026-07-12-20-formal-experiment-plan.md` 中与 E7/E8 冲突的部分。

## 1. 新增实验的目标与边界

论文原有证据只覆盖 OpenAI CLIP ViT-B/16。新增 **E7** 检验 LoRA-NF 是否依赖这一特定 backbone：在 X-TAIL 16-shot、相同十任务顺序下，比较 SigLIP 2 上的 LADA-style 基线和 LoRA-NF，并报告 Transfer / Average / Last。

同时，原始 E2--E5 的许多运行已经保存每个任务后的 `*_retrieval.json`，但 `paper_experiment_results.md` 没有系统报告它们。新增 **E8** 只聚合这些已存在的机器可读结果，回答 FD、CD 与前向 NSP 对检索保持的作用；它不是新的 encoder 训练实验。

SigLIP 2 checkpoint 固定为 `google/siglip2-base-patch16-224`。这是 SigLIP 2 的 FixRes base/16/224 模型，分辨率和 patch size 与现有 CLIP ViT-B/16 最接近，避免将更大容量或 NaFlex 可变分辨率的收益混入“方法跨 backbone 鲁棒性”结论。SigLIP 2 文本预处理固定为官方 processor 的 length-64 padding/truncation。

## 2. E7：SigLIP 2 鲁棒性主表

### 2.1 固定协议

| 项目 | 设置 |
|---|---|
| Backbone | `google/siglip2-base-patch16-224` |
| 数据 | X-TAIL，16-shot，十任务固定顺序 |
| Seeds | `42, 43, 44` |
| 训练预算 | 800 iterations/task |
| LoRA-NF | rank=4，q/k/v/out/fc1/fc2，hard NSP，`eps=0.20`，`weight=0.02` |
| 蒸馏（LoRA-NF） | FD=1，CD=2，temperature=4 |
| 分类器（LoRA-NF） | LR-RGDA + ZS，`num_centers=4`，maxshift ensemble |
| 指标 | Zero-shot 与 Ensemble 的 Transfer / Average / Last；主结论使用 Ensemble |

### 2.2 方法和运行量

| 方法 | 实现与主报告项 | 训练 runs |
|---|---|---:|
| LADA-style SigLIP 2 | 冻结视觉编码器、文本 AdaptFormer、label-specific prototype memory；报告 `LADA+ZS` T/A/L | 3 |
| LoRA-NF SigLIP 2 | 完整 LoRA-NF + FD/CD + LR-RGDA ensemble；报告 Ens T/A/L | 3 |
| **总计** | 16-shot，3 seeds | **6** |

LADA 官方仓库绑定 OpenAI CLIP 的 tokenizer、visual encoder、DPT 和权重结构，不能直接加载 SigLIP 2。因此 E7 的 LADA 项必须写作 **“LADA-style architecture-adapted reimplementation on SigLIP 2”**，不能声称是官方 LADA checkpoint 或与 CLIP 官方复现完全同一实现。其目的仍是测试两种持续学习策略跨 backbone 的相对趋势。

不为 E7 增加 full-shot 或检索：本轮新增问题仅是主表分类鲁棒性，增加这些项目会扩大计算预算而不直接回答该问题。

### 2.3 执行与验收

服务器运行：

```bash
cd /data/home/zengyong1/projects/clip_ood_v2
git pull origin v4
python -c "import transformers; print(transformers.__version__)"
python scripts/check_siglip2_compatibility.py --device cuda:0
bash scripts/run_siglip2_robustness.sh 0
```

运行前必须以单任务 smoke test 验证 `google/siglip2-base-patch16-224` 可被当前 `transformers` 加载；若版本缺少 SigLIP 2 checkpoint support，升级到项目现有的 `transformers==4.57.6` 后再开始正式 runs。每个正式 run 必须产生十行任务级分类矩阵以及：

```text
E7__siglip2__lora_nf__16shot__seed{42,43,44}_ens_results.json
E7__siglip2__lada_style__16shot__seed{42,43,44}_lada_zs_results.json
```

聚合命令：

```bash
python scripts/summarize_continual_metrics.py \
  --output experiments/paper_formal/E7_siglip2/E7_siglip2_main_table \
  --input 'LoRA-NF Ensemble=experiments/paper_formal/E7_siglip2/E7__siglip2__lora_nf__16shot__seed42_ens_results.json' \
  --input 'LoRA-NF Ensemble=experiments/paper_formal/E7_siglip2/E7__siglip2__lora_nf__16shot__seed43_ens_results.json' \
  --input 'LoRA-NF Ensemble=experiments/paper_formal/E7_siglip2/E7__siglip2__lora_nf__16shot__seed44_ens_results.json' \
  --input 'LADA-style+ZS=experiments/paper_formal/E7_siglip2/E7__siglip2__lada_style__16shot__seed42_lada_zs_results.json' \
  --input 'LADA-style+ZS=experiments/paper_formal/E7_siglip2/E7__siglip2__lada_style__16shot__seed43_lada_zs_results.json' \
  --input 'LADA-style+ZS=experiments/paper_formal/E7_siglip2/E7__siglip2__lada_style__16shot__seed44_lada_zs_results.json'
```

## 3. E8：检索组件归因（不重训）

### 3.1 研究问题

检索只取决于 encoder 与文本适配器，不受 LR-RGDA/Ensemble 预测分支影响。因此：

- **E2** 检验 FD/CD 的独立与联合作用；
- **E3** 检验持续前向过滤相对 LoRA/LoRA-Null 的作用；
- **E4/E5** 的已有 retrieval JSON 可作为机制附录，分析 NSP/蒸馏强度；
- **E6 不需要、也不能通过重训来评估检索**，因为其只改变推理分类器融合权重。

### 3.2 已有结果的可复用性

经文件覆盖检查，E2 的 C0/C1/C2（三 seeds）及完整 C3/C4（复用 E1 LoRA-NF 三 seeds）均有两套检索数据集各十个任务的记录。E3 的 LoRA、LoRA-Null、LoRA-NF 三 seeds 同样完整。因此这两张核心检索组件表**不需要重新训练**；只需以新增汇总脚本从原 JSON 生成表格。E4/E5 的已有单-seed retrieval JSON 也可直接汇总，只有在要把单 seed 曲线升级为 3-seed 显著性结论时才需要补跑。

所有 E8 结果报告：每任务均值 Retrieval Average、最终任务 Retrieval Last、I2T R@1、T2I R@1 及二者等权平均 mR@1；完整 R@5/R@10 留在 JSON。3-seed 的 `±` 采用与既有主表一致的 population standard deviation。

### 3.3 重新运行判定

| 已有实验包 | 是否重跑 | 原因 |
|---|---|---|
| E1 CLIP 主表 | 否 | 正式分类与检索结果已完成 |
| E2 FD/CD | 否 | 三 seeds、十任务、两检索集 JSON 已完整 |
| E3 LoRA family | 否 | 三 seeds、十任务、两检索集 JSON 已完整 |
| E4 NSP hyperparameters | 否（默认） | 已有 JSON 可作单-seed 附录；非核心 3-seed 结论不强求 |
| E5 CD sweep | 否（默认） | 同上 |
| E6 ensemble | 否 | encoder 不变，检索理论上不变 |
| LADA-CLIP | 否 | 官方逐任务检索已经完成 |
| **E7 SigLIP 2** | **是** | 唯一新增的 6 个训练 runs |

## 4. 论文主张边界

E7 若 LoRA-NF 在 SigLIP 2 上仍优于 LADA-style 基线，可写“在第二个视觉语言 backbone 上观察到一致趋势”；不能只凭两个 backbone 写“与 backbone 无关”。E8 若检索差异不跨 MSCOCO/Flickr 一致，应报告其稳定性界限，不将分类消融的正效应自动归因成检索改善。

## 5. 结果写入位置

- `docs/new_experiment_results_report.md`：E7 主表和 E8 检索组件表；
- `docs/paper_experiment_results.md`：完成 E7 后只摘录最终可用于正文的表和结论；
- `experiments/paper_formal/E7_siglip2/`：新训练和聚合 JSON；
- `experiments/paper_formal/E8_retrieval_ablation/`：从已有 JSON 生成的只读汇总。
