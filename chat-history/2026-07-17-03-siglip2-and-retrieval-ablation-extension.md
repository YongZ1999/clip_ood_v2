# SigLIP 2 鲁棒性与检索组件归因扩展

**日期**: 2026-07-17  
**会话概况**: 新增 SigLIP 2 X-TAIL 主表支持，并将已有逐任务检索 JSON 转为可审计的组件消融结果；不重跑已完成的 CLIP 实验。

## 1. 关键决定

- 新 backbone 固定为 `google/siglip2-base-patch16-224`，而不是 NaFlex 或更大模型，以匹配现有 CLIP ViT-B/16 的 16x16 patch 和 224 分辨率；
- E7 仅运行 16-shot、3 seeds 的 LADA-style SigLIP 2 与 LoRA-NF SigLIP 2，共 6 个训练 runs；
- 官方 LADA 代码耦合 OpenAI CLIP，SigLIP 2 条目明确标记为 architecture-adapted LADA-style reimplementation；
- E2/E3 的 retrieval JSON 已经完整，不应重训；E6 的分类器融合不改变 encoder，也不需要检索重跑。

## 2. 新增代码

- `src/models/backbone_utils.py`: 统一 CLIP/SigLIP 的图像、文本 embedding 和 SigLIP 2 text tokenization；
- `src/models/clip.py`: 基于 `--model_name` 加载默认 CLIP 或 SigLIP 2；
- `src/models/lada_text_adapter.py`: AdaptFormer 支持 SigLIP encoder layer；
- 训练、特征、参考蒸馏和检索路径使用统一 embedding helper；
- `scripts/run_siglip2_robustness.sh`: E7 六个 runs；
- `scripts/check_siglip2_compatibility.py`: 在正式 runs 前验证 SigLIP 2 的 LoRA-NF 与 LADA-style 分支；
- `scripts/summarize_continual_metrics.py` 与 `scripts/summarize_retrieval_ablation.py`: 分类和检索的只读聚合工具；
- `tests/test_backbone_utils.py`: CPU-only helper regression test（服务器执行）。

## 3. 已核验的 E8 结果

`scripts/summarize_retrieval_ablation.py` 已成功读取 E2 C0/C1/C2 与 E1 完整配置的 12 个输入 JSON，每个输入均覆盖两套数据集和十个任务。详细数值及谨慎解释写入 `docs/new_experiment_results_report.md`。

## 4. 运行前检查

- 服务器 `transformers` 必须能够通过 `AutoModel` 加载 `google/siglip2-base-patch16-224`；项目历史使用 `transformers==4.57.6`；
- 先跑一任务 smoke test，确认 SigLIP 2 的 q/k/v/out/fc1/fc2 注入、AdaptFormer 与结果文件均正常；
- 正式 E7 不启用检索评估，因为其研究问题只要求分类主表；检索组件结论复用现有 E2--E5 原始结果。
