# Transfer-aware 实现与后续实验：合并记录

**日期**: 2026-07-20 至 2026-07-22

**分支**: `agent/transfer-gated-ensemble`

**会话概况**: 合并 transfer-aware 工程实现、服务器离线加载修复及后续实验计划；审计记录保持独立。
**目的**: 合并本轮的实现、故障修复与实验计划记录，减少重复交接材料。结果核验记录不在本文重复，仍保留为独立 audit。

## 1. 研究动机与范围

历史 LoRA-NF 主实验呈现“Last 较高、绝对 Transfer 相对 LADA 论文数字偏低”的现象。排查发现，直接围绕 alpha 或 LR-RGDA 拟合步数追逐 Transfer 既不能隔离 backbone/预处理差异，也会带来事后选参风险。因此，本轮只在独立分支加入两项可部署的分类评估/推理改动：标准 CLIP 的保持宽高比测试预处理，以及基于零样本预测的 ZS--LR-RGDA 融合门控。

这两项改动不改变 LoRA-NF 的训练目标、NSP、FD/CD 或检索编码过程。历史 `paper_formal` 结果保持不动；新结果写入 `experiments/paper_transfer_aware/`，不得与旧协议的绝对分类数值直接混排。

## 2. 已实现的分类协议

### 2.1 标准 CLIP 测试预处理

- `src/utils/data.py:get_transforms` 增加 `test_resize_mode`：
  - `legacy_square`（默认）：`Resize((224,224)) -> CenterCrop`，用于复现历史结果；
  - `preserve_aspect`：`Resize(shorter_edge=224) -> CenterCrop(224)`，用于本轮 transfer-aware 实验。
- `main_incremental.py` 增加 `--eval_resize_mode`，并传给所有 X-TAIL test loader。
- 训练 augmentation、训练数据与跨模态检索预处理均未改变。

### 2.2 ZS-predicted-seen 集成门控

- `src/utils/main_utils.py:combine_ensemble_logits` 与 `EnsembleClassifier` 增加 `routing`；默认仍为历史 `classwise`。
- 新的 `zs_predicted_seen` 规则：若原始 ZS logits 的 argmax 指向当前尚未学过的类别（索引不小于当前已见类别数），直接使用 ZS logits；否则使用历史的 classwise maxshift ZS + LR-RGDA 融合。
- 它不访问真实任务 ID、真实标签或 OOD 标签，只依赖部署时可取得的 ZS 预测。
- `main_incremental.py` 增加 `--ensemble_routing`；离线 RGDA evaluator 也支持同一字段。

### 2.3 固定的主实验分类器

主实验未同时改动分类器超参数，固定为：`num_centers=4`、`rgda_rank=32`、`rgda_train_iter=200`、`rgda_fit_source=gmm_sample`、`alpha=0.05`、`maxshift`。这样 Standard LoRA 与 LoRA-NF 使用相同推理协议。

## 3. 离线 OpenAI CLIP 恢复路径

服务器的 Hugging Face CLIP cache 损坏且 DNS 不可用，但仍有 OpenAI `~/.cache/clip/ViT-B-16.pt`。直接使用 OpenAI `clip.load()` 不可行，因为其注意力使用 fused QKV，而 LoRA-NF 注入器依赖 Hugging Face 模块树和独立 Q/K/V 投影。

- 新增 `src/models/openai_clip_compat.py`：将 OpenAI ViT-B/16 的 `.pt` 权重映射到等价 Hugging Face `CLIPModel`，包括拆分 QKV、映射视觉/文本 transformer、投影层与 logit scale，并复用项目内 OpenAI BPE tokenizer。
- `src/models/clip.py:get_clip_model()` 保持 Hugging Face `from_pretrained()` 为优先路径；仅当正式 OpenAI ViT-B/16 加载失败、`CLIP_OPENAI_PT_FALLBACK=1` 且本地 `.pt` 存在时才回退转换。
- 损坏 cache 有时抛出 `AttributeError` 而不是 `OSError`，故加载/分词器回退均捕获 `(OSError, AttributeError)`。
- 该回退不作用于 SigLIP2、其他模型或 OpenAI ResNet CLIP。
- `scripts/check_openai_pt_hf_compat.py` 在启动前验证 tokenizer 一致性及原生/转换模型的图像、文本特征 cosine（阈值 `0.9999`）。

## 4. 实验与工具支持

- `scripts/run_transfer_aware_main_table.sh`：在 GPU 0--5 上队列运行 LoRA-NF / Standard LoRA × 16-shot / full-shot × seeds 42/43/44，共 12 个 E1 主实验；显式使用 OpenAI CLIP、离线加载变量、`preserve_aspect` 与 `zs_predicted_seen`。
- `tests/test_transfer_aware_ensemble.py`：覆盖两种 resize mode 及门控的已见/未见 ZS 分支。
- `scripts/evaluate_incremental_rgda_sweep_artifacts.py`：离线 artifact evaluator 支持 routing、`rank=`/`r=` variant 和 `--eval_seed`；后者固定 Python、NumPy、Torch CPU/CUDA RNG。注意：现有离线 JSON 的 `args` 继承训练 artifact，未完整写入 evaluator 参数；若未来将该 evaluator 用于正式表，应补充 JSON metadata。
- 相关操作文档：
  - `docs/transfer_aware_main_experiment.md`
  - `docs/server_model_prompt_transfer_aware_main.md`
  - `docs/transfer_aware_followup_experiment_plan.md`
  - `docs/server_model_prompt_transfer_aware_followups.md`

## 5. 后续实验决策

1. E1 三 seed 主表已完成，作为本分支的正式 LoRA-NF vs Standard LoRA 内部比较。
2. D1/D2 仅为单 seed classifier 开发扫描；其提升未达预定义门槛，因此不替换 E1 的正式超参数，也不扩展为三 seed。
3. R1/E6-TA 是不重训编码器的路由 ablation；R2/E2-TA 是 FD/CD 组件的单 seed 分类与逐任务 MSCOCO 5K 检索 ablation。
4. 历史 E2--E5 是 legacy `legacy_square + classwise` 机制证据。若在论文中与 transfer-aware 主表同时出现，必须明确标注协议不同，不能混合求均值或做绝对横比。
5. 不因本轮分类协议改动而重跑 retrieval-only、E4、LADA、Frozen CLIP 或 SigLIP2；只有当论文要求 LoRA / LoRA-Null / GradProj / LoRA-NF 在同一新协议表内排名时，才考虑补跑缺失的 adapter family。

## 6. 保留的独立审计记录

以下记录按“审核不合并”的约定保留，包含原始 JSON 的独立核验与具体结论：

- `chat-history/2026-07-20-01-lada-main-table-metric-audit.md`
- `chat-history/2026-07-20-02-transfer-hyperparameter-diagnosis.md`
- `chat-history/2026-07-21-01-transfer-aware-e1-result-audit.md`
- `chat-history/2026-07-21-03-phase-d1-single-seed-audit.md`
- `chat-history/2026-07-22-01-phase-d2-rank-sweep-audit.md`
- `chat-history-for-paper-writing/2026-07-20-01-lada-comparison-narrative-audit.md`

## 7. 本次合并替代的记录

本文替代并删除以下重复的实现/计划记录：

- `chat-history/2026-07-20-03-transfer-aware-main-protocol-implementation.md`
- `chat-history/2026-07-20-04-openai-clip-loader-launcher-fix.md`
- `chat-history/2026-07-20-05-offline-openai-pt-hf-compatibility.md`
- `chat-history/2026-07-20-06-hf-cache-attributeerror-fallback-fix.md`
- `chat-history/2026-07-21-02-single-seed-transfer-aware-optimization-plan.md`
- `chat-history/2026-07-22-02-transfer-aware-followup-plan.md`

论文叙事类源记录由 `chat-history-for-paper-writing/2026-07-22-01-transfer-aware-writing-handoff.md` 单独合并，以保持工程记录与论文叙事的目录边界。
