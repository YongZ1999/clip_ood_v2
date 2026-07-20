# Server-model prompt: transfer-aware E1 main table

Copy the following prompt into the server-side model after the branch has been
pushed to GitHub.

---

你现在要运行 LoRA-NF 的 **transfer-aware E1 主实验**。不要修改算法代码、不要改现有 `v4` 分支、不要运行 E2--E8 消融，也不要重跑 LADA。目标是使用新分支上的统一评估协议，公平重跑 Standard LoRA 与 LoRA-NF 的主表。

先执行只读检查：

```bash
cd /data/home/zengyong1/projects/clip_ood_v2
git fetch origin
git switch agent/transfer-gated-ensemble
git pull --ff-only origin agent/transfer-gated-ensemble
git status --short
git log -1 --oneline
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
nvidia-smi
python tests/test_transfer_aware_ensemble.py
```

如果 `git status --short` 显示任何**已跟踪代码文件**被修改，停止运行并报告这些文件；不要使用 `git reset --hard`。`experiments/`、`logs/` 下的未跟踪结果无需删除。

通过单测后，直接启动官方 launcher：

```bash
bash scripts/run_transfer_aware_main_table.sh
```

launcher 会自动使用 GPU `0,1,2,3,4,5`：第一波同时运行六个任务，第二波每张卡自动继续一个排队任务。完整公平主表共 12 个训练任务：

```text
LoRA-NF:       16-shot / full-shot × seeds 42,43,44 = 6
Standard LoRA: 16-shot / full-shot × seeds 42,43,44 = 6
```

不要手动再开额外训练进程，不要覆盖 `experiments/paper_formal/`。新结果和日志必须分别保存在：

```text
experiments/paper_transfer_aware/E1_main/
logs/paper_transfer_aware/
```

新协议已由 launcher 固定：

```text
--eval_resize_mode preserve_aspect
--ensemble_routing zs_predicted_seen
--num_centers 4 --rgda_rank 32 --rgda_train_iter 200
--rgda_fit_source gmm_sample --alpha 0.05
```

含义：标准 CLIP 宽高比测试预处理；当 ZS 的预测类别尚未学习时保持纯 ZS，否则使用现有 maxshift ZS+LR-RGDA 融合。训练端 LoRA-NF、蒸馏和 LR-RGDA 已验证超参数保持不变。

运行期间每隔一段时间报告一次：每张 GPU 当前任务、最近日志最后 30 行、是否有报错。不要因单个任务报错而杀掉其他 GPU；launcher 会继续该 GPU 队列中的后续任务。

全部结束后执行并报告：

```bash
find experiments/paper_transfer_aware/E1_main -name '*_ens_results.json' | sort
find experiments/paper_transfer_aware/E1_main -name '*_zs_results.json' | sort
find experiments/paper_transfer_aware/E1_main -name '*_retrieval.json' | sort
grep -R "Traceback\|ERROR\|FAILED" -n logs/paper_transfer_aware || true
```

核验 12 个 Ensemble JSON、12 个 ZS JSON、12 个 retrieval JSON 是否都存在。然后读取每个 JSON 的 `metrics.transfer`、`metrics.average`、`metrics.last`，按方法和 shot 分成三个 seeds，计算 mean ± std；同时汇总 retrieval 的 Average 和 Last。保存一个新的汇总 Markdown 到：

```text
experiments/paper_transfer_aware/E1_main/TRANSFER_AWARE_SUMMARY.md
```

最后只提交结果摘要、每个 run 的 JSON 路径、检索 JSON 路径、失败情况（如有）和当前 commit SHA；不要修改 `docs/paper_experiment_results.md`，等本地人工复核后再更新。

---
