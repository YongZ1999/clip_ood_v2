# 服务器交接：官方 LADA 分类 + 跨模态检索（3 seeds）

**交接日期**: 2026-07-16  
**目标**: 在不改变官方 LADA 训练/分类协议的前提下，补齐真实 LADA 的逐任务图文检索结果，并最终与 LoRA-NF 的第 7 节进行同协议比较。

---

## 1. 必须同步的提交

服务器代码必须至少包含以下两个提交（按顺序）：

```text
a00c34e fix: correct NSP covariance accounting and retrieval audit
ea3f610 feat: evaluate retrieval in official LADA tasks
```

本次 LADA 检索功能的核心是 `ea3f610`。确认：

```bash
cd /home/raoxuan/projects/project_clip_continual_learning
git log --oneline -3
test -f LADA/retrieval_eval.py && echo "LADA retrieval evaluator present"
```

若服务器工作树存在未提交改动，先检查其是否与以下文件冲突，不能直接覆盖：

```text
LADA/retrieval_eval.py
LADA/trainer.py
LADA/utils/config.py
LADA/run_TAIL_16shot_seed.sh
```

## 2. 改动内容与边界

本次没有改变 LADA 的 AdaptFormer、DPT、label-specific memory、优化器、epoch recipe 或分类指标。

仅新增了一个**任务后评估**：每个任务训练结束、当前 `Text_Tuner` 仍在内存时，调用 `LADA/retrieval_eval.py`：

1. 用 `model.image_encoder` 和 `model.text_encoder(tokens, model.text_tuner)` 测真实当前 LADA；
2. 用相同模型、相同 OpenAI CLIP tokenizer、相同图像预处理，但 `text_tuner=None` 测 Frozen CLIP 对照；
3. 复用项目 `src.utils.retrieval_eval` 的 COCO/Flickr30K 数据读取、双向正样本定义和 R@K 计算；
4. 将每任务的 `lada` 与 `frozen_clip` 结果追加到同一个 `retrieval.json`，并自动计算 LADA 的 Average / Last。

因此，此评测得到的是**真实 LADA 检索**。不要再写 `Frozen CLIP (= LADA)`；Frozen CLIP 只是同协议参考。

## 3. 数据集与环境前置检查

LADA 配置默认使用：

```text
MSCOCO 5K:  /data/home/zengyong1/dataset/mscoco_2014_5k_test_hf
Flickr30K:  /data/home/zengyong1/dataset/flickr30k_hf
X-TAIL:     /data1/open_datasets/X-TAIL
```

开始前检查：

```bash
test -d /data1/open_datasets/X-TAIL && echo "X-TAIL OK"
test -d /data/home/zengyong1/dataset/mscoco_2014_5k_test_hf/images_mscoco_2014_5k_test && echo "MSCOCO OK"
test -f /data/home/zengyong1/dataset/mscoco_2014_5k_test_hf/test_5k_mscoco_2014.csv && echo "MSCOCO captions OK"
test -d /data/home/zengyong1/dataset/flickr30k_hf/data && echo "Flickr30K OK"
```

使用原 LADA 环境（历史记录中为 `raoxuan`）。该环境需要 PyTorch、torchvision、scikit-learn、yacs、pandas 及 parquet backend；不要在本地 Mac 运行此实验。

## 4. 正式运行：16-shot、3 个 seed

每个 seed 仍是官方的 10 个独立任务进程，训练预算不变。只需在原命令前增加 `LADA_RETRIEVAL_EVAL=1`。

```bash
cd /home/raoxuan/projects/project_clip_continual_learning/LADA
source ~/miniconda3/etc/profile.d/conda.sh
conda activate raoxuan

LADA_RETRIEVAL_EVAL=1 bash run_TAIL_16shot_seed.sh 42 0
LADA_RETRIEVAL_EVAL=1 bash run_TAIL_16shot_seed.sh 43 1
LADA_RETRIEVAL_EVAL=1 bash run_TAIL_16shot_seed.sh 44 2
```

若多卡并行，确保每个 seed 使用独占 GPU，并将 stdout/stderr 写到各自日志。不要让同一输出目录被两个 seed 共享。

`LADA_RETRIEVAL_EVAL` 默认关闭；必须显式设为 `1`，否则只会得到旧的分类结果，不会生成检索 JSON。

## 5. 每个 seed 的预期产物与验收

以 seed 42 为例：

```text
LADA/output/LADA_official_s42/result.txt
LADA/output/LADA_official_s42/retrieval.json
```

`result.txt` 仍是原官方分类 Transfer/Average/Last。`retrieval.json` 是新增文件，结构要点：

```json
{
  "protocol": "official_lada_in_process_text_tuner",
  "rows": [
    {"method": "frozen_clip", "step": 1, "task": "aircraft", "dataset": "mscoco_2014_5k"},
    {"method": "lada", "step": 1, "task": "aircraft", "dataset": "mscoco_2014_5k"}
  ],
  "lada_summary": {
    "mscoco_2014_5k": {"average": {}, "last": {}},
    "flickr30k_hf": {"average": {}, "last": {}}
  }
}
```

完整 10-task 运行的行数应为：

- 默认含 Frozen 对照：`10 tasks × 2 datasets × 2 methods = 40 rows`；
- 只看真实 LADA：每个数据集恰有 10 个 `method == "lada"` 行；
- `lada_summary[dataset].num_tasks` 必须为 10。

快速验收：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("output/LADA_official_s42/retrieval.json")
payload = json.loads(path.read_text())
assert payload["protocol"] == "official_lada_in_process_text_tuner"
for dataset, summary in payload["lada_summary"].items():
    assert summary["num_tasks"] == 10, (dataset, summary["num_tasks"])
print("rows:", len(payload["rows"]))
print("LADA retrieval summary:", json.dumps(payload["lada_summary"], indent=2))
PY
```

## 6. 三 seed 汇总规范

从每个 seed 的 `lada_summary` 取同一数据集、同一方向、同一统计量：

- `average.i2t_r@1`、`average.t2i_r@1`；
- `last.i2t_r@1`、`last.t2i_r@1`。

对 42/43/44 计算 arithmetic mean 和 sample standard deviation（`ddof=1`），以 `mean ± std` 报告。Frozen rows也可汇总作 sanity check，但论文比较 LADA 时必须使用 `method=lada` 的数值。

与 LoRA-NF 对比时保持：MSCOCO 5K、Flickr30K、完整测试集、I2T/T2I R@1、每任务评测、Average/Last 一致。不要混用旧文档中标注为 “Frozen CLIP (= LADA)” 的值。

## 7. 失败排查

- `ModuleNotFoundError: src`：确认从 `LADA/` 目录运行且 `LADA/retrieval_eval.py` 来自提交 `ea3f610`；该文件会把项目根目录加入 `sys.path`。
- `FileNotFoundError`（COCO/Flickr30K）：先核对第 3 节路径；必要时以 yacs opts 覆盖 `retrieval_root` / `retrieval_roots`，例如：
  `retrieval_roots "mscoco_2014_5k=/new/coco,flickr30k_hf=/new/flickr"`。
- 只得到分类 `result.txt`：检查启动环境变量是否为 `LADA_RETRIEVAL_EVAL=1`，以及日志 config 中是否为 `retrieval_eval: True`。
- `retrieval.json` 少于 40 行：检查是否有某个任务进程失败；不要只重跑 `result_process.py`，应重跑缺失的那个任务命令，因为 Text_Tuner 不在旧 checkpoint 中。

## 8. 结果解释约束

此实验回答的是：“真实官方 LADA 在 X-TAIL 连续训练后，其当前文本 AdaptFormer 对通用图文检索的影响是什么？”

它不测 LADA 的分类器/DPT 在 retrieval 上的作用，因为这些模块不参与 CLIP 的通用 image-text embedding 相似度。LADA 的视觉编码器冻结，而文本端会改变；所以真实 LADA 检索可能等于、优于或劣于 Frozen CLIP，必须以此新评测为准。
