# Server-model prompt: transfer-aware follow-up experiments

Copy the following prompt to the server-side model after pulling the latest
`agent/transfer-gated-ensemble` branch.

---

```text
请在服务器项目中拉取并切换到 GitHub 分支 agent/transfer-gated-ensemble 的最新代码：

cd /data/home/zengyong1/projects/clip_ood_v2
git fetch origin
git switch agent/transfer-gated-ensemble
git pull --ff-only origin agent/transfer-gated-ensemble
git log -1 --oneline

完整阅读并严格执行：
docs/transfer_aware_followup_experiment_plan.md

本轮只执行 R1/E6-TA 与 R2/E2-TA；不要执行 Optional R3，不要重跑 E1、LADA、SigLIP2、full-shot 或任何三-seed 主表。

先确认 D1 artifacts 仍存在：
find experiments/paper_transfer_aware/dev_seed42/async_eval/artifacts -name artifact.pt | wc -l

预期是 10。若不是 10，停止并报告，不要重训 D1。

R1：复用 D1 artifacts 做两次离线评估。设置完全相同（M=4、gmm_sample、fit=200、alpha=.05、eval_seed=42、preserve_aspect、完整测试集），仅分别使用 classwise 与 zs_predicted_seen routing。结果写入：
experiments/paper_transfer_aware/E6_gate_seed42/

R2：只启动 3 个 LoRA-NF 16-shot seed-42 训练任务 C0(fd=0,cd=0)、C1(fd=1,cd=0)、C2(fd=0,cd=2)。所有其它训练、NSP、LR-RGDA、评估、检索参数必须逐字匹配 transfer-aware E1/D1 的固定配置；保持 inline classification 和每任务一次 MSCOCO 5K retrieval。C3(fd=1,cd=2) 已在 dev_seed42 中存在，绝不重跑。

先用 nvidia-smi 选择三张空闲 GPU 并行启动 C0/C1/C2；没有空闲卡则排队，不要杀掉无关进程。结果与日志必须分别写入：
experiments/paper_transfer_aware/E2_distill_seed42/
logs/paper_transfer_aware/E2_distill_seed42/

训练全部结束后，读取 C0/C1/C2 的原始 ZS、Ensemble、retrieval JSON，并结合 dev_seed42 的 C3，生成：
experiments/paper_transfer_aware/E2_distill_seed42/E2_TA_SUMMARY.md

汇总必须包含分类 ZS/Ens Transfer/Average/Last，检索 I2T/T2I R@1/5/10 的 Average 与 Last，C0/C1/C2 相对 C3 的差值，全部命令、commit SHA、失败情况。

不要根据一个 seed 宣称统计显著性，不要修改 docs/paper_experiment_results.md。完成后只强制提交本轮 JSON、R1/R2 汇总 Markdown、必要的启动脚本（若新增）；不要提交 artifact.pt、checkpoint、logs 或数据集。push 到同一分支，并报告 commit SHA、结果文件数量及完整汇总。
```
