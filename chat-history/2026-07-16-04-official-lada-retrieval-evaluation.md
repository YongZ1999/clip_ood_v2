# 官方 LADA 跨模态检索评测接入

**日期**: 2026-07-16  
**目的**: 直接测量官方 LADA 在每个 X-TAIL 任务结束时的真实图文检索能力，替代 “Frozen CLIP 等同 LADA” 的错误代理。

## 实现

- 新增 `LADA/retrieval_eval.py`。
  - 复用 `src.utils.retrieval_eval` 的 COCO/Flickr30K 数据读取、caption-to-image 正样本映射、双向 R@K 计算。
  - 图像嵌入从官方 `model.image_encoder` 得到；LADA 文本嵌入从 `model.text_encoder(tokens, model.text_tuner)` 得到。
  - 同时评测 `text_tuner=None` 的 `frozen_clip`，它与 LADA 使用相同 OpenAI CLIP、tokenizer、图像预处理和检索集合。
- `LADA/trainer.py` 在每个任务训练、原型更新、原始分类评测后调用 evaluator。此时当前 Text_Tuner 尚在内存，避免官方 checkpoint 不保存其状态造成的不可重建问题。
- `LADA/utils/config.py` 新增检索开关与数据集路径参数，默认关闭，不改变原官方分类复现。
- `LADA/run_TAIL_16shot_seed.sh` 支持 `LADA_RETRIEVAL_EVAL=1` opt-in。

## 运行

```bash
cd LADA
LADA_RETRIEVAL_EVAL=1 bash run_TAIL_16shot_seed.sh 42 0
```

输出为 `output/LADA_official_s42/retrieval.json`：每个任务/数据集包含 `lada` 与 `frozen_clip` 行，并自动汇总 LADA 的 Retrieval Average 与 Last。三个 seed 都执行后，再按与 E1 相同方式汇总 mean ± std。

## 验证状态

- 新增/修改的 LADA Python 文件已经过无导入语法编译检查；启动脚本通过 `bash -n`；`git diff --check` 通过。
- 本地环境没有 PyTorch 和检索数据，未运行动态评测。必须在正式远程 GPU 环境用完整 COCO 5K 与 Flickr30K 跑至少一个 seed 验证输出，再运行 42/43/44 的完整比较。
