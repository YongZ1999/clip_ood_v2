# 分类器特征 Transform 消融与 GMM 回放跟进

**日期**: 2026-06-14
**会话概况**: 验证用户提出的假设：在 `test_transform` 模式下，LR-RGDA 分类器准确率应显著提升。新增显式实验开关并在服务器上运行真实特征和 GMM mean 回放对照。

---

## 1. 代码改动

### `main_joint.py`

新增参数：

```bash
--classifier_feature_transform train|test
```

- `train`: 保持历史行为，使用带随机增强的 `train_loader` 提取 16-shot 分类器构建特征。
- `test`: 使用确定性 `train_loader4updating` 提取 16-shot 分类器构建特征，避免训练增强噪声。

该开关同时作用于 LR-RGDA 和 LADA，保证两者使用相同特征源。

### `scripts/run_joint_classifier_replay.sh`

新增环境变量：

```bash
CLASSIFIER_FEATURE_TRANSFORMS="train test"
REPLAY_MODES="real gmm_raw gmm_sphere gmm_raw_mean"
```

实验名自动加上 transform 前缀，例如 `train_real`、`test_real`、`test_gmm_raw_mean`，避免不同口径的 JSON 互相覆盖。

### CLIP 本地缓存加载修复

服务器访问 `hf-mirror.com` 超时，且 `use_safetensors=True` 会触发 Hugging Face 元数据查询。为保证服务器离线缓存可用，`src/models/clip.py` 新增环境变量：

```bash
CLIP_USE_SAFETENSORS=0
CLIP_LOCAL_FILES_ONLY=1
CLIP_MODEL_NAME=openai/clip-vit-base-patch16
```

同时将 trainer 的导入从根目录 `models.*` 改成 `src.models.*`，避免服务器根目录 `models/` 不完整导致 `models.utils` 缺失。

## 2. 已完成验证

本地检查通过：

- `python -m py_compile main_joint.py src/models/clip.py src/models/trainer.py src/trainers/lora_nsp_trainer.py scripts/summarize_joint_classifier_replay.py`
- `bash -n scripts/run_joint_classifier_replay.sh`
- `git diff --check`

服务器最小加载测试通过：

```bash
CLIP_USE_SAFETENSORS=0 CLIP_LOCAL_FILES_ONLY=1
```

可从 Hugging Face 本地缓存加载 `CLIPModel` 和 `CLIPProcessor`。

## 3. 当前实验结果

服务器路径：

```text
/home/raoxuan/projects/project_clip_continual_learning/experiments/joint_classifier_transform_ablation
```

已完成结果：

| Replay | CLIP-ZS | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS |
|---|---:|---:|---:|---:|---:|
| train_real | 56.40 +/- 0.00 | 74.78 +/- 0.18 | 74.74 +/- 0.20 | 75.00 +/- 0.20 | 74.74 +/- 0.20 |
| test_real seed42 | 56.40 | 77.66 | 78.84 | 77.90 | 78.84 |
| test_gmm_raw_mean seed42 | 56.40 | 76.73 | 76.01 | 76.93 | 76.01 |

## 4. 初步结论

1. 用户假设的前半成立：`test_transform` 显著提升 LR-RGDA。
   - seed42 真实特征下，LR-RGDA 从 `74.84` 提升到 `77.66`，约 `+2.82 pp`。
2. 但真实特征下 LADA 也同步提升，且提升更大。
   - seed42 真实特征下，LADA 从 `74.82` 提升到 `78.84`。
   - 因此真实特征场景暂时仍是 LADA 领先约 `1.18 pp`。
3. 统计回放场景继续支持 LR-RGDA 的论文差异点。
   - seed42 `test_gmm_raw_mean` 下，LR-RGDA `76.73`，LADA `76.01`，LR-RGDA+ZS `76.93`。
   - 这说明当真实历史特征不可保存、只能用轻量 GMM/均值统计回放时，LR-RGDA 仍可能比 LADA 更能重建判别结构。

## 5. 正在运行 / 待完成

正在服务器上运行：

- `test_gmm_raw_mean seed43`
- `test_gmm_raw_mean seed44`
- 主脚本继续运行 `test_real` 的 3-seed 队列

待完成后需要：

- 汇总 `test_gmm_raw_mean` 的 3-seed mean +/- std。
- 若 LR-RGDA 领先稳定，再将“统计回放优势”作为推理端主要论文 claim。
- 后续补跑 `test_gmm_raw` / variance-scale ablation，解释为什么 mean replay 强于 noisy spherical sampling。

## 6. 对可发表目标的影响

当前证据不支持“真实特征条件下 LR-RGDA 全面超过 LADA”。更稳妥的叙事是：

- 真实特征可用时，LADA 仍是强标签记忆基线。
- `test_transform` 是所有公平分类器构建实验应采用的主口径。
- LR-RGDA 的可发表优势应聚焦在“轻量统计回放 / 不能保存真实历史特征”的场景。

## 7. 论文草稿同步清理

已同步修改 `paper_writing/paper-template/paper_draft.tex`，把旧叙事中证据不足的 claim 降级：

- 摘要和贡献中不再声称 LR-RGDA 是无条件 `Bayes-optimal` 分类器。
- 不再声称 LR-RGDA 天然区分 ID/OOD、对 OOD 输出近零置信度。
- 不再把推理端方法写成 adaptive router / OOD router 主线；当前实现按固定 ensemble 和同源特征构建分类器比较。
- 不再声称当前方法已经在 X-TAIL 上达到 state-of-the-art。
- 新增 `Statistical Replay Protocol` 小节，将主叙事收敛到真实特征回放 vs 统计回放的分类器构建比较。

本地 LaTeX 检查：

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper_draft.tex
```

已能完整生成 PDF。仍存在旧草稿遗留 warning/TODO，例如未补齐最终实验表格、部分引用/交叉引用需要后续整理、表格编号因旧 auxiliary 文件有重复 destination warning。这些不是当前 claim 清理引入的 fatal error。

## 8. 最新等待状态

截至本次检查，服务器仍未产出新的 `test_gmm_raw_mean_seed43.json` / `test_gmm_raw_mean_seed44.json`。仍在运行的关键进程：

- `test_real seed43`
- `test_gmm_raw_mean seed43`
- `test_gmm_raw_mean seed44`

当前 summary 仍只包含 `test_gmm_raw_mean seed42`，不能作为 3-seed 论文结论。下一次继续时应先重新运行：

```bash
ssh raoxuan@10.20.34.30 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate raoxuan && cd /home/raoxuan/projects/project_clip_continual_learning && python scripts/summarize_joint_classifier_replay.py --input_dir experiments/joint_classifier_transform_ablation --output_csv experiments/joint_classifier_transform_ablation/summary.csv --output_markdown experiments/joint_classifier_transform_ablation/summary.md && cat experiments/joint_classifier_transform_ablation/summary.md"
```

只有当 `test_gmm_raw_mean` 的 seed 42/43/44 都完成并且 LR-RGDA 或 LR-RGDA+ZS 稳定超过 LADA 时，才能把“统计回放优势”提升为主论文 claim。

## 9. GMM mean replay 重复中心修复

后续检查发现旧的 `test_gmm_raw_mean seed43/44` 进程长时间不落盘，并且日志持续刷：

```text
ConvergenceWarning: Number of distinct clusters (4) found smaller than n_clusters (16).
```

根因是 `gmm_sample_mode=mean` 会把每类 4 个 GMM 分量均值按权重重复到 16 个伪样本；LADA 的 `build_from_data(..., k=16)` 虽然限制了 `actual_k <= n_samples`，但没有限制到唯一特征数，导致 sklearn 对重复点请求 16 个中心。该问题会造成大量 warning 和无效 CPU 消耗，也会让 LADA 在 mean replay 下拥有 16 个可训练重复原型，和“只保存紧凑统计量”的论文叙事不一致。

本地已修复：

- `src/lada/lada_classifier.py`
  - 每类先计算 `np.unique(lbl_features, axis=0)`。
  - LADA 的 `actual_k` 限制为 `min(k, unique_count)`。
  - 如果唯一点数已经不超过 `actual_k`，直接使用唯一点作为 LADA centers，跳过 sklearn k-means。
- `src/classifiers/gaussian_statistics.py`
  - 多中心 LR-RGDA 的 torch k-means 先去重。
  - 若唯一中心数少于请求的 `M`，循环补齐到固定 `M`，保持 LR-RGDA 多中心张量形状不变。

本地验证通过：

```bash
python -m py_compile src/lada/lada_classifier.py src/classifiers/gaussian_statistics.py main_joint.py
python - <<'PY'
import torch
from src.classifiers.gaussian_statistics import kmeans
from src.lada.lada_classifier import LADAClassifier

base = torch.randn(4, 8)
features = base.repeat_interleave(4, dim=0)
features = features / features.norm(dim=-1, keepdim=True)
labels = torch.zeros(features.shape[0], dtype=torch.long)
centers = kmeans(features, 4)
assert centers.shape == (4, 8)
assert torch.unique(centers, dim=0).shape[0] == 4
clf = LADAClassifier(feature_dim=8)
clf.build_from_data(features, labels, k=16)
assert clf.curr_lada_features.shape == (8, 4)
PY
git diff --check -- src/lada/lada_classifier.py src/classifiers/gaussian_statistics.py
```

服务器同步：

- 已在服务器 `/home/raoxuan/projects/project_clip_continual_learning` 应用同样补丁。
- 服务器 `python -m py_compile src/lada/lada_classifier.py src/classifiers/gaussian_statistics.py` 通过。
- 已停止旧的、持续刷 warning 的 `test_gmm_raw_mean seed43/44` 进程。
- 保留 `test_real` 主队列继续运行。
- 已在新目录启动 fixed mean replay 三种子：

```text
experiments/joint_classifier_transform_ablation_fixed_mean
```

运行设置：

- `seed=42` on GPU 2
- `seed=43` on GPU 4
- `seed=44` on GPU 5
- `classifier_feature_transform=test`
- `gmm_fit_space=raw`
- `gmm_sample_mode=mean`
- `experiment_name=test_gmm_raw_mean_fixed`

截至最后一次检查，fixed run 已正常进入特征提取阶段，未再出现 sklearn `ConvergenceWarning`。尚未产出 JSON。下一次继续时应优先查看：

```bash
ssh raoxuan@10.20.34.30 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate raoxuan && cd /home/raoxuan/projects/project_clip_continual_learning && find experiments/joint_classifier_transform_ablation_fixed_mean -maxdepth 1 -type f -name '*_seed*.json' -printf '%f\n' | sort && python scripts/summarize_joint_classifier_replay.py --input_dir experiments/joint_classifier_transform_ablation_fixed_mean --output_csv experiments/joint_classifier_transform_ablation_fixed_mean/summary.csv --output_markdown experiments/joint_classifier_transform_ablation_fixed_mean/summary.md && cat experiments/joint_classifier_transform_ablation_fixed_mean/summary.md"
```

注意：fixed mean replay 的 LADA 数值不应和旧 `test_gmm_raw_mean_seed42.json` 直接混合求三种子均值，因为 LADA center 构造口径已经改变。应以 `test_gmm_raw_mean_fixed_seed42/43/44` 作为同一口径的完整三种子结果。

## 10. Fixed mean replay 三种子结果

fixed mean replay 三种子已完成，结果目录：

```text
/home/raoxuan/projects/project_clip_continual_learning/experiments/joint_classifier_transform_ablation_fixed_mean
```

文件：

```text
test_gmm_raw_mean_fixed_seed42.json
test_gmm_raw_mean_fixed_seed43.json
test_gmm_raw_mean_fixed_seed44.json
```

summary：

| Replay | CLIP-ZS | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS |
|---|---:|---:|---:|---:|---:|
| test_gmm_raw_mean_fixed | 56.40 +/- 0.00 | 76.93 +/- 0.16 | 76.08 +/- 0.35 | 77.12 +/- 0.13 | 76.08 +/- 0.35 |

per-seed：

| Seed | CLIP-ZS | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS | LR-RGDA - LADA | LR-RGDA+ZS - LADA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 56.4011 | 76.7484 | 75.7929 | 76.9723 | 75.7929 | +0.9555 | +1.1794 |
| 43 | 56.4011 | 76.9786 | 75.9857 | 77.1801 | 75.9857 | +0.9930 | +1.1944 |
| 44 | 56.4011 | 77.0659 | 76.4729 | 77.2157 | 76.4729 | +0.5930 | +0.7427 |

结论：

- 三个 seed 下 LR-RGDA 均超过 LADA。
- LR-RGDA 平均领先 LADA `+0.85 pp`。
- LR-RGDA+ZS 平均领先 LADA `+1.04 pp`。
- 这是当前最强、最干净的推理端论文证据：在 `test_transform + raw GMM mean replay + compact component-mean replay` 场景下，LR-RGDA/ensemble 对 LADA 有稳定优势。
- 该结论必须保持窄化，不能推广为真实特征场景或所有 replay 场景下都超过 LADA。

已同步更新 `paper_writing/paper-template/paper_draft.tex` 的 `Statistical Replay Protocol` 小节，将原 TODO 替换为 fixed mean replay 三种子表和窄化结论。

## 11. test_real 三种子完成后的方法边界

原目录 `experiments/joint_classifier_transform_ablation` 中 `test_real` 三种子也已完成：

| Replay | CLIP-ZS | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS |
|---|---:|---:|---:|---:|---:|
| test_real | 56.40 +/- 0.00 | 77.87 +/- 0.16 | 78.92 +/- 0.16 | 78.05 +/- 0.14 | 78.92 +/- 0.16 |

per-seed：

| Seed | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS | LR-RGDA - LADA | LR-RGDA+ZS - LADA |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 77.6900 | 78.8373 | 77.9066 | 78.8373 | -1.1473 | -0.9307 |
| 43 | 77.9112 | 78.8118 | 78.0690 | 78.8118 | -0.9006 | -0.7428 |
| 44 | 78.0065 | 79.1008 | 78.1820 | 79.1008 | -1.0943 | -0.9188 |

这给出了很清楚的方法边界：

- 真实 16-shot 特征可用时，LADA 更强，领先 LR-RGDA 约 `+1.05 pp`。
- 统计 mean replay 时，LR-RGDA 更强，领先 LADA 约 `+0.85 pp`，LR-RGDA+ZS 领先约 `+1.04 pp`。
- 因此论文主张应写成：
  - 不声称 LR-RGDA 全面超过 LADA；
  - 声称 LR-RGDA/ensemble 在“不能保存真实历史特征、只保留紧凑 GMM 均值统计量”的低存储 replay 场景中稳定超过 LADA；
  - 将真实特征结果作为诚实边界报告，反而增强可信度。

`paper_writing/paper-template/paper_draft.tex` 已补一句真实特征对照：同样 deterministic transform 下，LADA `78.92 +/- 0.16` 高于 LR-RGDA `77.87 +/- 0.16` 和 LR-RGDA+ZS `78.05 +/- 0.14`。

## 12. 后续统计采样 replay 消融启动

为了判断 LR-RGDA 的优势是否只属于 `gmm_sample_mode=mean`，已启动 `gmm_raw sample` 三种子对照：

```text
/home/raoxuan/projects/project_clip_continual_learning/experiments/joint_classifier_sampling_ablation
```

运行命令等价于：

```bash
for seed in 42 43 44; do
  CUDA_VISIBLE_DEVICES=<gpu> \
  CLIP_USE_SAFETENSORS=0 \
  CLIP_LOCAL_FILES_ONLY=1 \
  python main_joint.py \
    --id_datasets ALL \
    --ood_datasets \
    --root /data1/open_datasets/X-TAIL \
    --gpu 0 \
    --iterations 0 \
    --num_shots 16 \
    --batch_size 64 \
    --enable_lada \
    --lada_k 16 \
    --lada_beta 1.0 \
    --lada_train_iter 200 \
    --lada_train_lr 0.01 \
    --num_centers 4 \
    --rgda_rank 32 \
    --rgda_train_iter 200 \
    --rgda_train_lr 0.01 \
    --gmm_k 4 \
    --gaussian_samples_per_class 16 \
    --use_gaussian_features \
    --gmm_fit_space raw \
    --gmm_sample_mode sample \
    --output_dir experiments/joint_classifier_sampling_ablation \
    --seed "$seed" \
    --classifier_feature_transform test \
    --experiment_name test_gmm_raw_sample
done
```

进程：

- `seed=42` on GPU 0
- `seed=43` on GPU 1
- `seed=44` on GPU 2

判据：

- 如果 `test_gmm_raw_sample` 也稳定超过 LADA，则论文可声称 LR-RGDA 在统计 replay 中有更一般的优势。
- 如果 `test_gmm_raw_sample` 不稳定或低于 LADA，而 `test_gmm_raw_mean_fixed` 稳定领先，则论文 claim 应进一步收窄为“紧凑均值统计 replay / component-mean replay”。
- 后续再补 `gmm_sphere sample`，用于区分 raw-space GMM 与 sphere-space GMM 的影响。

当前状态：

- 三个 `test_gmm_raw_sample` 进程均仍在运行。
- 尚未产出 JSON，当前 summary 为空表。
- 日志未出现 `Traceback` / `ERROR` / `ConvergenceWarning` 命中。
- 暂未启动 `gmm_sphere sample`，避免与 raw-sample 三种子并发过多，影响这组结果稳定性。

下一次继续时优先查询：

```bash
ssh raoxuan@10.20.34.30 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate raoxuan && cd /home/raoxuan/projects/project_clip_continual_learning && find experiments/joint_classifier_sampling_ablation -maxdepth 1 -type f -name '*_seed*.json' -printf '%f\n' | sort && python scripts/summarize_joint_classifier_replay.py --input_dir experiments/joint_classifier_sampling_ablation --output_csv experiments/joint_classifier_sampling_ablation/summary.csv --output_markdown experiments/joint_classifier_sampling_ablation/summary.md && cat experiments/joint_classifier_sampling_ablation/summary.md"
```

## 13. Raw GMM sample replay 三种子结果

`test_gmm_raw_sample` 三种子已完成，结果目录：

```text
/home/raoxuan/projects/project_clip_continual_learning/experiments/joint_classifier_sampling_ablation
```

文件：

```text
test_gmm_raw_sample_seed42.json
test_gmm_raw_sample_seed43.json
test_gmm_raw_sample_seed44.json
```

summary：

| Replay | CLIP-ZS | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS |
|---|---:|---:|---:|---:|---:|
| test_gmm_raw_sample | 56.40 +/- 0.00 | 76.24 +/- 0.11 | 75.63 +/- 0.28 | 76.46 +/- 0.16 | 75.63 +/- 0.28 |

per-seed：

| Seed | CLIP-ZS | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS | LR-RGDA - LADA | LR-RGDA+ZS - LADA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 56.4011 | 76.1413 | 75.3047 | 76.3524 | 75.3047 | +0.8365 | +1.0476 |
| 43 | 56.4011 | 76.2200 | 75.8120 | 76.3951 | 75.8120 | +0.4080 | +0.5830 |
| 44 | 56.4011 | 76.3504 | 75.7740 | 76.6414 | 75.7740 | +0.5764 | +0.8674 |

结论：

- raw-space GMM stochastic sample replay 下，LR-RGDA 和 LR-RGDA+ZS 仍然三个 seed 都超过 LADA。
- 平均优势小于 fixed component-mean replay：
  - raw sample: LR-RGDA `+0.61 pp`，LR-RGDA+ZS `+0.83 pp`。
  - raw component mean: LR-RGDA `+0.85 pp`，LR-RGDA+ZS `+1.04 pp`。
- 因此 raw sample 支持“统计 replay 下 LR-RGDA 的优势不只存在于 mean replay”，但 margin 不够支撑强版本的“普遍 +1pp 优势”。
- 当前最稳论文表述应为：
  - 主证据仍放在 compact GMM component-mean replay；
  - raw stochastic sample 作为支持性消融，说明加噪采样下优势仍为正但变弱；
  - mean replay 可能更好地保留 16-shot CLIP 特征中的判别中心，而 spherical covariance noise 会削弱可分性。

## 14. Sphere GMM sample replay 已启动

为区分 raw-space GMM 与 sphere-space GMM 的影响，已启动 `test_gmm_sphere_sample` 三种子：

```text
seed=42 on GPU 2
seed=43 on GPU 4
seed=44 on GPU 5
```

运行设置：

```bash
CUDA_VISIBLE_DEVICES=<2|4|5> \
CLIP_USE_SAFETENSORS=0 \
CLIP_LOCAL_FILES_ONLY=1 \
python main_joint.py \
  --id_datasets ALL \
  --ood_datasets \
  --root /data1/open_datasets/X-TAIL \
  --gpu 0 \
  --iterations 0 \
  --num_shots 16 \
  --batch_size 64 \
  --enable_lada \
  --lada_k 16 \
  --lada_beta 1.0 \
  --lada_train_iter 200 \
  --lada_train_lr 0.01 \
  --num_centers 4 \
  --rgda_rank 32 \
  --rgda_train_iter 200 \
  --rgda_train_lr 0.01 \
  --gmm_k 4 \
  --gaussian_samples_per_class 16 \
  --use_gaussian_features \
  --gmm_fit_space sphere \
  --gmm_sample_mode sample \
  --output_dir experiments/joint_classifier_sampling_ablation \
  --seed <seed> \
  --classifier_feature_transform test \
  --experiment_name test_gmm_sphere_sample
```

下一次继续时查询：

```bash
ssh raoxuan@10.20.34.30 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate raoxuan && cd /home/raoxuan/projects/project_clip_continual_learning && find experiments/joint_classifier_sampling_ablation -maxdepth 1 -type f -name '*_seed*.json' -printf '%f\n' | sort && python scripts/summarize_joint_classifier_replay.py --input_dir experiments/joint_classifier_sampling_ablation --output_csv experiments/joint_classifier_sampling_ablation/summary.csv --output_markdown experiments/joint_classifier_sampling_ablation/summary.md && cat experiments/joint_classifier_sampling_ablation/summary.md && pgrep -af 'test_gmm_sphere_sample|main_joint.py' || true"
```

## 15. Sphere GMM sample replay 三种子结果

`test_gmm_sphere_sample` 三种子已完成，与 raw sample 放在同一结果目录：

```text
/home/raoxuan/projects/project_clip_continual_learning/experiments/joint_classifier_sampling_ablation
```

summary：

| Replay | CLIP-ZS | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS |
|---|---:|---:|---:|---:|---:|
| test_gmm_raw_sample | 56.40 +/- 0.00 | 76.24 +/- 0.11 | 75.63 +/- 0.28 | 76.46 +/- 0.16 | 75.63 +/- 0.28 |
| test_gmm_sphere_sample | 56.40 +/- 0.00 | 76.22 +/- 0.18 | 75.66 +/- 0.31 | 76.45 +/- 0.17 | 75.66 +/- 0.31 |

per-seed：

| Replay | Seed | CLIP-ZS | LR-RGDA | LADA | LR-RGDA+ZS | LADA+ZS | LR-RGDA - LADA | LR-RGDA+ZS - LADA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw sample | 42 | 56.4011 | 76.1413 | 75.3047 | 76.3524 | 75.3047 | +0.8365 | +1.0476 |
| raw sample | 43 | 56.4011 | 76.2200 | 75.8120 | 76.3951 | 75.8120 | +0.4080 | +0.5830 |
| raw sample | 44 | 56.4011 | 76.3504 | 75.7740 | 76.6414 | 75.7740 | +0.5764 | +0.8674 |
| sphere sample | 42 | 56.4011 | 76.3925 | 75.3202 | 76.5980 | 75.3202 | +1.0723 | +1.2778 |
| sphere sample | 43 | 56.4011 | 76.0269 | 75.7255 | 76.2565 | 75.7255 | +0.3013 | +0.5310 |
| sphere sample | 44 | 56.4011 | 76.2431 | 75.9305 | 76.4834 | 75.9305 | +0.3126 | +0.5530 |

结论：

- sphere sample 与 raw sample 几乎重合：
  - raw sample: LR-RGDA `76.24 +/- 0.11`，LADA `75.63 +/- 0.28`。
  - sphere sample: LR-RGDA `76.22 +/- 0.18`，LADA `75.66 +/- 0.31`。
- 两种 stochastic sample replay 下，LR-RGDA 和 LR-RGDA+ZS 都在三个 seed 上超过 LADA。
- 但 sampling replay 的优势小于 compact component-mean replay：
  - raw mean fixed: LR-RGDA+ZS `77.12 +/- 0.13`。
  - raw sample: LR-RGDA+ZS `76.46 +/- 0.16`。
  - sphere sample: LR-RGDA+ZS `76.45 +/- 0.17`。
- 因此，raw-vs-sphere 不是当前主要影响因素；更关键的是 mean replay vs noisy sample replay。
- 论文 claim 建议保持：
  - 主 claim：compact GMM component-mean replay 下 LR-RGDA/ensemble 稳定优于 LADA。
  - 支持性消融：raw/sphere stochastic sample replay 下优势仍为正，但 margin 变小。
  - 方法边界：真实特征可保存时 LADA 仍更强。

论文草稿同步：

- 已更新 `paper_writing/paper-template/paper_draft.tex` 的 `Statistical Replay Protocol` 表格。
- 表格现在同时包含：
  - `Real 16-shot features`
  - `GMM raw component means`
  - `GMM raw samples`
  - `GMM sphere samples`
- 叙事已改成 storage-dependent boundary：
  - real feature 下 LADA 更强；
  - compact statistical replay 下 LR-RGDA 稳定超过 LADA；
  - component-mean replay 是最强主证据；
  - stochastic sample replay 是支持性消融，说明优势仍为正但变弱；
  - raw-vs-sphere 不是主要影响因素。
- 本地验证：

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper_draft.tex
git diff --check
```

均通过。LaTeX 仍有旧草稿遗留 warning，例如首页长行、旧 aux duplicate destination、附录公式 overfull，但没有 fatal error；新 replay 表格的 overfull 已通过 `\resizebox{\textwidth}{!}{...}` 消除。
