# Server-model prompt: final transfer-aware evidence runs

Copy the following prompt to the server-side model. It is intentionally self-contained; do not let the server model change algorithm code or search extra hyperparameters.

```text
请在服务器项目中拉取并切换到 GitHub 分支 agent/transfer-gated-ensemble 的最新代码：

cd /data/home/zengyong1/projects/clip_ood_v2
git fetch origin
git switch agent/transfer-gated-ensemble
git pull --ff-only origin agent/transfer-gated-ensemble
git log -1 --oneline

完整阅读并严格执行：
docs/final_evidence_experiment_plan.md

这是最终证据补齐，不是继续调参。禁止修改算法代码、禁止搜索 alpha/RGDA rank/fit/M、训练步数、NSP/CD 超参数，禁止重跑 E1、LADA、SigLIP2、full-shot、R1 或 D1/D2。

先检查环境与权重兼容性：
python scripts/check_openai_pt_hf_compat.py --device cuda:0

检查失败就停止并报告；禁止用原生 clip.load() 替换项目的 Hugging Face-compatible fallback。

本轮总共只训练 12 个任务，按两个六卡波次执行。先用 nvidia-smi 选择 GPU 0--5 中的空闲卡；若某张不空闲则等待/排队，不要杀任何无关进程。

Phase A：先并行运行 6 个 E2-TA 训练：
  C0(fd=0,cd=0), C1(fd=1,cd=0), C2(fd=0,cd=2)，各 seed=43、44。
  只改 fd_weight、cd_weight、seed、experiment_name、output/log path；其余所有参数必须逐字匹配 scripts/run_transfer_aware_main_table.sh 的 LoRA-NF 16-shot recipe，包括 preserve_aspect、zs_predicted_seen、每任务 MSCOCO 5K retrieval、OpenAI CLIP 离线加载变量。
  输出：experiments/paper_transfer_aware/E2_distill_3seed/
  日志：logs/paper_transfer_aware/E2_distill_3seed/
  命名：E2TA__C{0,1,2}__seed{43,44}。

Phase B：等 Phase A 六个任务全部成功结束后，再并行运行 6 个 E3-TA 训练：
  LoRA-Null（seed 42/43/44）：--lora_type lora_nsp --init_mode lora_nsp --projection_param_mode full --null_init_mode history_init_only
  GradProj（seed 42/43/44）：--lora_type lora_nsp --init_mode lora_nsp --projection_param_mode full --null_init_mode none --use_gradient_projection
  两类任务都保留 fd_weight=1, cd_weight=2、完整 LoRA-NF 16-shot 公共参数、preserve_aspect、zs_predicted_seen 和每任务 retrieval；除 method flag 外不得修改任何参数。
  输出：experiments/paper_transfer_aware/E3_adapters_3seed/
  日志：logs/paper_transfer_aware/E3_adapters_3seed/
  命名：E3TA__lora_null__seed{42,43,44}、E3TA__gradproj__seed{42,43,44}。

允许新增一个纯 launcher/summary 脚本以避免手工命令出错，但不得改 main_incremental.py、src/ 下算法文件或既有 E1 结果。新脚本必须先 bash -n；所有训练命令、git SHA、物理 GPU 都写入对应 log。

每个训练任务完成后核对同名主 JSON、*_zs_results.json、*_rgda_results.json、*_ens_results.json、*_retrieval.json 均存在。任务失败时保留日志，其他任务可继续；最后集中报告失败任务，不能伪造或补写结果。

全部 12 个任务完成后：
1. 生成 experiments/paper_transfer_aware/E2_distill_3seed/E2_TA_3SEED_SUMMARY.md。C0/C1/C2 的 seed42 复用 E2_distill_seed42，43/44 用本轮；C3 三个 seed 全部复用 E1_main 的 LoRA-NF 16-shot JSON，绝不能混入 dev_seed42。报告 ZS/Ens T/A/L 和 MSCOCO I2T/T2I R@1/5/10 的任务均值与 Last，全部用 mean ± sample std，并列出原始文件路径。
2. 生成 experiments/paper_transfer_aware/E3_adapters_3seed/E3_TA_3SEED_SUMMARY.md。LoRA 与 LoRA-NF 复用 E1_main 16-shot 三 seed；LoRA-Null 与 GradProj 用本轮结果。报告同样的分类和检索指标、样本标准差、原始路径与排名。
3. 从原始 JSON 独立复算每个 K×K accuracy matrix 的 Transfer/Average/Last，确认与 JSON metrics 一致；汇总中记录检查数量与任何异常。
4. 不修改 docs/paper_experiment_results.md，也不替换历史 legacy 结果。
5. 仅 git add/commit/push 本轮的 JSON、两份 summary Markdown、必要新增 launcher/summary 脚本和一份 chat-history 记录；绝不提交 logs、artifact.pt、checkpoint、数据集或缓存。推到同一分支。

最后回复：commit SHA、12 个任务逐项成功/失败状态、两个 summary 文件路径、每个表的核心数值、原始 JSON 数量、是否存在重跑或失败。
```
