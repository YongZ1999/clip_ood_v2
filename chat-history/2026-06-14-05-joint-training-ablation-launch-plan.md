# Joint Training Ablation 启动计划与 Smoke Test

**日期**: 2026-06-14

## 1. 目标

补齐训练端 LoRA-NSP 贡献归因，避免把推理端 LR-RGDA/LADA 分类器差异误归因给 LoRA-NSP 表征训练。

本轮实验使用 `main_joint.py`，因为它已经和当前 LADA/LR-RGDA 公平评估口径对齐：

- X-TAIL 10 数据集，16-shot。
- `classifier_feature_transform=test`。
- LADA 和 LR-RGDA 使用同一训练特征、同一测试集、同一 label space。
- 输出同一 JSON schema，可由 `scripts/summarize_joint_classifier_replay.py` 汇总。

注意：这是 joint training ablation，不是 strict task-incremental Average/Last 实验。它用于快速验证训练端表征贡献；后续仍需要增量协议下的 Average/Last/forgetting 表。

## 2. 训练端配置

新增脚本：

```text
scripts/run_joint_training_ablation.sh
```

脚本特性：

- 如果当前没有 conda 环境且存在 `~/miniconda3/etc/profile.d/conda.sh`，会自动激活 `CONDA_ENV`，默认 `raoxuan`。
- 支持断点恢复：如果 `${OUT_DIR}/${experiment}_seed${seed}.json` 已存在，则跳过该 seed，避免远程中断后重跑覆盖已完成结果。
- 运行结束自动调用 `scripts/summarize_joint_classifier_replay.py` 生成 `summary.csv` 和 `summary.md`。

默认运行 4 个配置，每个配置 3 个 seed：

| Experiment | LoRA type | FD | CD | Purpose |
|---|---|---:|---:|---|
| `lora_vanilla` | `lora_vanilla` | 0.0 | 0.0 | 标准 LoRA baseline |
| `lora_nsp` | `lora_nsp` | 0.0 | 0.0 | 纯 NSP 方向约束贡献 |
| `lora_nsp_fd` | `lora_nsp` | 1.0 | 0.0 | NSP + feature distillation |
| `lora_nsp_fd_cd` | `lora_nsp` | 1.0 | 1.0 | 完整 LoRA-NSP + FD + CD |

默认参数：

```bash
ROOT=/data1/open_datasets/X-TAIL
OUT_DIR=experiments/joint_training_ablation_20260614
SEEDS="42 43 44"
ITERATIONS=800
BATCH_SIZE=64
GPUS="0 2 4 5"
```

核心命令口径：

```bash
python main_joint.py \
  --id_datasets ALL \
  --ood_datasets \
  --root /data1/open_datasets/X-TAIL \
  --gpu 0 \
  --iterations 800 \
  --num_shots 16 \
  --batch_size 64 \
  --tune_vision_encoder true \
  --enable_lada \
  --lada_k 16 \
  --lada_beta 1.0 \
  --lada_train_iter 200 \
  --lada_train_lr 0.01 \
  --num_centers 4 \
  --rgda_rank 32 \
  --rgda_train_iter 200 \
  --rgda_train_lr 0.01 \
  --classifier_feature_transform test \
  --output_dir experiments/joint_training_ablation_20260614 \
  --seed <42|43|44> \
  --lora_type <lora_vanilla|lora_nsp> \
  --fd_weight <0.0|1.0> \
  --cd_weight <0.0|1.0> \
  --experiment_name <name>
```

## 3. Smoke Test 结果

服务器路径：

```text
/home/raoxuan/projects/project_clip_continual_learning
```

第一次 smoke test：

```bash
--iterations 1
--id_datasets aircraft caltech101
--lora_type lora_nsp
--fd_weight 1.0
--cd_weight 1.0
```

失败原因：

```text
ValueError: optimizer got an empty parameter list
```

根因：

- `main_joint.py` 当前默认 `--tune_vision_encoder false` 且 `--tune_text_encoder false`。
- 如果不显式设置 `--tune_vision_encoder true`，训练时没有任何 LoRA 参数。

第二次 smoke test 加入：

```bash
--tune_vision_encoder true
```

结果：

- Flickr8K reference loader 正常加载：`8091` samples。
- LoRA-NSP + FD + CD 训练 1 iteration 正常。
- LoRA merge/reset 正常。
- `classifier_feature_transform=test` 的 LR-RGDA/LADA 同源评估正常。
- JSON 正常保存到：

```text
experiments/joint_training_smoke/smoke_full_seed42.json
```

两任务 smoke result：

| Dataset | ZS | LR-RGDA | LR-RGDA+ZS | LADA |
|---|---:|---:|---:|---:|
| aircraft | 22.5 | 31.5 | 33.7 | 33.2 |
| caltech101 | 81.5 | 86.5 | 87.2 | 90.4 |
| Average | 52.0 | 59.0 | 60.5 | 61.8 |

该数值只用于 smoke，不可作为论文结果。

## 4. 当前服务器状态

最后一次成功查询时：

- 无 `main_joint.py` 主实验进程。
- GPU 2/4/5 空闲。
- GPU 3 忙，继续避免使用。
- GPU 0/1 有少量残留显存/利用率，但没有本项目主进程。

随后尝试通过 SSH 写入并启动远程脚本时多次失败：

```text
ssh: connect to host 10.20.34.30 port 22: Operation not permitted
```

这看起来是远程 SSH 临时连接限制，而不是实验脚本或代码错误。

后续补充：

- 2026-06-14 04:27:53 CST 曾短暂恢复 SSH，确认服务器无 `main_joint.py` 主实验进程，GPU 2/4/5 空闲，GPU 3 忙。
- 随后 `scp` / 写入远程脚本再次触发 `Operation not permitted`。
- 已在本地更新 `scripts/run_joint_training_ablation.sh` 为可恢复脚本，并通过：

```bash
bash -n scripts/run_joint_training_ablation.sh
git diff --check -- scripts/run_joint_training_ablation.sh
```

下一次 SSH 恢复后可直接同步并启动。

## 6. 复现包与 Storage Budget 补充

在 SSH 持续不稳定期间，已补齐本地复现与论文审计工件：

```text
experiments/README_publication_repro.md
```

该文件记录：

- 当前服务器环境和 CLIP 本地缓存 env vars。
- LR-RGDA/LADA 公平比较规则。
- 已完成 inference-side 结果目录和 summary。
- classifier replay 复现命令。
- pending training-side ablation 启动/查询命令。
- 当前 paper claim 边界和禁止越界表述。
- remaining publication gates。

新增 storage-budget 计算，使用当前 X-TAIL 16-shot 设置：

```text
C = 1100 classes
d = 512 CLIP ViT-B/16 feature dimension
float32 = 4 bytes/value
```

| Stored representation | Items per class | Formula | Storage |
|---|---:|---|---:|
| Real 16-shot features | 16 features | `C * 16 * d` | 34.38 MiB |
| LADA `k=16` centers | 16 centers | `C * 16 * d` | 34.38 MiB |
| GMM `k=4` component means | 4 means | `C * 4 * d` | 8.59 MiB |
| GMM `k=4` spherical params | 4 means, variances, weights | `C * 4 * (d + 2)` | 8.63 MiB |

关键边界：

- storage claim 只针对 historical replay source，不针对最终分类器参数。
- 不声称 LR-RGDA 最终分类器参数总是小于 LADA。
- 当前最强叙事是：`k=4` GMM replay source 约为 real features / LADA `k=16` centers 的四分之一，而 LR-RGDA 在该压缩 replay source 下更稳定地利用统计信息。

论文同步：

- 已在 `paper_writing/paper-template/paper_draft.tex` 的 `Statistical Replay Protocol` 后加入 storage-budget 表。
- 已隐藏未验证的训练端主表、ablation 表和旧 classifier analysis 表，避免 PDF 出现未完成 TODO 数字。
- 已明确写入：strict incremental Transfer/Average/Last 和 LoRA-NSP training-side ablation 尚不能作为当前 paper evidence。

本地验证：

```bash
bash -n scripts/run_joint_training_ablation.sh
git diff --check
pdflatex -interaction=nonstopmode -halt-on-error paper_draft.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_draft.tex
```

均通过。LaTeX 仍有旧 warning（首页长行、duplicate table destination、附录公式 overfull），但无 fatal error；新增 storage 表的交叉引用已稳定。

## 5. 下次继续

SSH 恢复后，在本地同步脚本并启动：

```bash
scp scripts/run_joint_training_ablation.sh \
  raoxuan@10.20.34.30:/home/raoxuan/projects/project_clip_continual_learning/scripts/run_joint_training_ablation.sh

ssh raoxuan@10.20.34.30 "
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate raoxuan &&
  cd /home/raoxuan/projects/project_clip_continual_learning &&
  chmod +x scripts/run_joint_training_ablation.sh &&
  nohup scripts/run_joint_training_ablation.sh \
    > experiments/joint_training_ablation_20260614_master.log 2>&1 &
"
```

查询进度：

```bash
ssh raoxuan@10.20.34.30 "
  cd /home/raoxuan/projects/project_clip_continual_learning &&
  pgrep -af 'run_joint_training_ablation|main_joint.py' || true &&
  find experiments/joint_training_ablation_20260614 -maxdepth 1 -name '*_seed*.json' -printf '%f\n' | sort &&
  tail -40 experiments/joint_training_ablation_20260614/logs/launch.log 2>/dev/null || true
"
```

完成后汇总：

```bash
ssh raoxuan@10.20.34.30 "
  source ~/miniconda3/etc/profile.d/conda.sh &&
  conda activate raoxuan &&
  cd /home/raoxuan/projects/project_clip_continual_learning &&
  python scripts/summarize_joint_classifier_replay.py \
    --input_dir experiments/joint_training_ablation_20260614 \
    --output_csv experiments/joint_training_ablation_20260614/summary.csv \
    --output_markdown experiments/joint_training_ablation_20260614/summary.md &&
  cat experiments/joint_training_ablation_20260614/summary.md
"
```

解释规则：

- 如果 `lora_nsp_fd_cd` 提升 ZS 和/或 LR-RGDA/LADA real-feature accuracy，说明训练端贡献成立。
- 如果训练端提升只出现在 ZS，而不传递到 LR-RGDA/LADA，需要把 LoRA-NSP 写成 representation/zero-shot preservation contribution，而不是分类器性能主因。
- 如果训练端不稳定或负向，需要降低 LoRA-NSP 主 claim，把论文重点进一步转向 statistical replay classifier。
