# 2026-07-17：回退 NSP 审计期间的训练代码修改

## 决定

LoRA 系列的正式分类实验已经使用原始实现完成，论文结果也以这些原始运行生成的 JSON 为准。因此，仓库的训练主线恢复到正式实验所用的实现；仅保留为官方 LADA 新增的逐任务跨模态检索评估。

## 已回退的内容

恢复提交 `a00c34e` 之前的实现：

- `src/trainers/lora_nsp_trainer.py`：恢复全局 `total_observations` 的协方差归一化；
- `src/models/lora_sgp.py`：恢复原始投影矩阵设备处理；
- `main_incremental.py`：移除 `reference_root` 参数，并恢复原始 LoRA 检索任务编号；
- `src/utils/reference_loader.py`：恢复原始 Flickr8k 固定路径；
- 删除只服务于上述协方差修改的 `tests/test_covariance_grouping.py`。

## 保留的 LADA 检索能力

以下内容保持不变，且不依赖被回退的 LoRA 训练改动：

- `LADA/retrieval_eval.py`；
- `LADA/trainer.py` 中每个任务训练结束后的评估调用；
- `LADA/utils/config.py` 的检索参数；
- `LADA/run_TAIL_16shot_seed.sh` 的 `LADA_RETRIEVAL_EVAL=1` 开关与依赖预检；
- `LADA/requirements.txt` 中的 `pyarrow`；
- 已测得的 LADA 检索结果和论文结果文档。

这样，后续重新运行 LADA 时仍会得到每任务的真实 LADA 跨模态检索指标，而 LoRA-NF 的代码版本与已经完成的 LoRA 系列正式分类实验一致。
