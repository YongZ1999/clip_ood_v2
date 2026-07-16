# 服务器交接：SigLIP 2 鲁棒性与检索组件报告

**日期**：2026-07-17  
**适用分支**：`v4`

## 可直接发给服务器模型的提示词

```text
你在服务器上的仓库是 clip_ood_v2。请严格执行以下新增实验，不要修改、删除或覆盖任何既有 E1--E6、CLIP 主表或 LADA-CLIP 的原始结果。

目标：
1) 新增 E7：在 SigLIP 2 上测试 LoRA-NF 与 **Native LADA algorithm port** 的 X-TAIL 16-shot 持续学习分类鲁棒性；
2) 新增 E8：从既有 retrieval JSON 汇总 FD/CD 组件和 LoRA/LoRA-Null/LoRA-NF 的检索结果。E8 不训练模型。

先同步代码并运行兼容性门槛：
cd /data/home/zengyong1/projects/clip_ood_v2
git checkout v4
git pull origin v4
python -c "import transformers; print(transformers.__version__)"
python scripts/check_siglip2_compatibility.py --device cuda:0

只有最后一条打印 `SigLIP 2 compatibility gate: passed` 才能开始正式 E7。若 checkpoint 无法加载或 LoRA-NF/LADA AdaptFormer 路径失败，停止，不要开始正式 runs；完整保存 traceback 和 transformers 版本。必要时安装/切换到仓库实验使用的 transformers==4.57.6 后，重新运行门槛。

通过后，在一张空闲 GPU 上启动全部 6 个 E7 runs（16-shot，seeds 42/43/44；每个 seed 先 LoRA-NF、后 Native LADA）：
bash scripts/run_siglip2_robustness.sh 0

若 SigLIP2 LoRA-NF 三个 seeds 已成功完成，只重跑 Native LADA，绝不能再次运行上面的完整脚本；改用：

```bash
# 单卡顺序运行三个 Native LADA seeds
bash scripts/run_siglip2_lada_native.sh 0

# 或为并行调度分别提交三个独立 jobs（推荐）：
bash scripts/run_siglip2_lada_native.sh 3 42
bash scripts/run_siglip2_lada_native.sh 4 43
bash scripts/run_siglip2_lada_native.sh 5 44
```

注意：
- E7 只测分类 Transfer/Average/Last，不开启 retrieval；
- SigLIP2 backbone 固定为 google/siglip2-base-patch16-224；
- Native LADA SigLIP2 port 保留原 LADA 的冻结视觉端、task-local text AdaptFormer、label-specific memory、DPT replay、AdamW+OneCycle 与各数据集 epoch schedule；仅适配 SigLIP2 接口，不能称为官方代码零修改直接运行；
- 任何此前以 `E7__siglip2__lada_style__...` 命名、使用统一 800 iterations 的结果都属于旧 controlled-budget 版本，不能写入 E7 主表；
- 不要为 E7 重跑 full-shot，也不要重跑 E1--E6 或官方 LADA-CLIP。

6 个 run 全部完成后，确认下列六个文件存在且每个都含完整 T/A/L：
experiments/paper_formal/E7_siglip2/E7__siglip2__lora_nf__16shot__seed{42,43,44}_ens_results.json
experiments/paper_formal/E7_siglip2/E7__siglip2__lada_native__16shot__seed{42,43,44}_lada_zs_results.json

然后汇总 E7：
python scripts/summarize_continual_metrics.py \
  --output experiments/paper_formal/E7_siglip2/E7_siglip2_main_table \
  --input 'LoRA-NF Ensemble=experiments/paper_formal/E7_siglip2/E7__siglip2__lora_nf__16shot__seed42_ens_results.json' \
  --input 'LoRA-NF Ensemble=experiments/paper_formal/E7_siglip2/E7__siglip2__lora_nf__16shot__seed43_ens_results.json' \
  --input 'LoRA-NF Ensemble=experiments/paper_formal/E7_siglip2/E7__siglip2__lora_nf__16shot__seed44_ens_results.json' \
  --input 'Native LADA+ZS=experiments/paper_formal/E7_siglip2/E7__siglip2__lada_native__16shot__seed42_lada_zs_results.json' \
  --input 'Native LADA+ZS=experiments/paper_formal/E7_siglip2/E7__siglip2__lada_native__16shot__seed43_lada_zs_results.json' \
  --input 'Native LADA+ZS=experiments/paper_formal/E7_siglip2/E7__siglip2__lada_native__16shot__seed44_lada_zs_results.json'

E8 不要重跑任何训练。只执行下列只读汇总，并把生成的 Markdown/JSON 交回：
python scripts/summarize_retrieval_ablation.py \
  --output experiments/paper_formal/E8_retrieval_ablation/E8_components \
  --input C0=experiments/paper_formal/E2_components/E2__C0__s42_retrieval.json \
  --input C0=experiments/paper_formal/E2_components/E2__C0__s43_retrieval.json \
  --input C0=experiments/paper_formal/E2_components/E2__C0__s44_retrieval.json \
  --input C1=experiments/paper_formal/E2_components/E2__C1__s42_retrieval.json \
  --input C1=experiments/paper_formal/E2_components/E2__C1__s43_retrieval.json \
  --input C1=experiments/paper_formal/E2_components/E2__C1__s44_retrieval.json \
  --input C2=experiments/paper_formal/E2_components/E2__C2__s42_retrieval.json \
  --input C2=experiments/paper_formal/E2_components/E2__C2__s43_retrieval.json \
  --input C2=experiments/paper_formal/E2_components/E2__C2__s44_retrieval.json \
  --input C3_C4_full=experiments/paper_formal/E1_main/E1__lora_nf__16shot__seed42_retrieval.json \
  --input C3_C4_full=experiments/paper_formal/E1_main/E1__lora_nf__16shot__seed43_retrieval.json \
  --input C3_C4_full=experiments/paper_formal/E1_main/E1__lora_nf__16shot__seed44_retrieval.json

python scripts/summarize_retrieval_ablation.py \
  --output experiments/paper_formal/E8_retrieval_ablation/E8_lora_family \
  --input 'Standard LoRA=experiments/paper_formal/E1_main/E1__lora__16shot__seed42_retrieval.json' \
  --input 'Standard LoRA=experiments/paper_formal/E1_main/E1__lora__16shot__seed43_retrieval.json' \
  --input 'Standard LoRA=experiments/paper_formal/E1_main/E1__lora__16shot__seed44_retrieval.json' \
  --input 'LoRA-Null=experiments/paper_formal/E3_adapters/E3__lora_null__s42_retrieval.json' \
  --input 'LoRA-Null=experiments/paper_formal/E3_adapters/E3__lora_null__s43_retrieval.json' \
  --input 'LoRA-Null=experiments/paper_formal/E3_adapters/E3__lora_null__s44_retrieval.json' \
  --input 'LoRA-NF=experiments/paper_formal/E1_main/E1__lora_nf__16shot__seed42_retrieval.json' \
  --input 'LoRA-NF=experiments/paper_formal/E1_main/E1__lora_nf__16shot__seed43_retrieval.json' \
  --input 'LoRA-NF=experiments/paper_formal/E1_main/E1__lora_nf__16shot__seed44_retrieval.json'

最后回复：每个 E7 run 的退出状态、E7 汇总表路径/数值、两份 E8 汇总路径，以及任何失败的完整 traceback。不要自行改写 docs/paper_experiment_results.md；把结果交回给我统一审核后再写论文表格。
```

## 重跑清单

仅需新增 E7 的 6 次训练。E8 只读汇总。E1--E6、full-shot、原 CLIP 检索、官方 LADA-CLIP 都不重跑。
