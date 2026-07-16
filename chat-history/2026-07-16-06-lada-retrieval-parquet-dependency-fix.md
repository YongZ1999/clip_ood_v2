# LADA 检索实验 parquet 依赖修复

**日期**: 2026-07-16  
**触发**: 三个 LADA retrieval seed 在 Task 1 的 MSCOCO 评测完成后，加载 Flickr30K parquet shard 时失败。

## 根因

日志的异常为：

```text
ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'
```

`src.utils.retrieval_eval.load_retrieval_dataset()` 对 `flickr30k_hf` 调用 `pandas.read_parquet()`；实际启动 LADA 的服务器 `clip` conda 环境没有 `pyarrow` 或 `fastparquet`。截图中 MSCOCO 5K 已产生真实 LADA Task 1 指标，证明 LADA retrieval evaluator 和 COCO 路径正常；失败仅是 Flickr30K parquet 可选依赖缺失。

## 修改

- `LADA/requirements.txt` 增加 `pyarrow`。
- `LADA/run_TAIL_16shot_seed.sh` 在 `LADA_RETRIEVAL_EVAL=1` 时，训练开始前运行 `python3 -c 'import pyarrow'`。若缺失，明确提示安装命令并退出，避免耗费一个任务后才失败。
- `LADA/README.md` 和服务器交接文档写明同一 conda 环境中需要执行：

```bash
python3 -m pip install pyarrow
python3 -c "import pyarrow; print(pyarrow.__version__)"
```

## 重新运行要求

由于每个任务进程的当前 Text_Tuner 不保存在 checkpoint 中，失败的 seed 不能从 Task 2 继续获得完整逐任务检索曲线。安装依赖并同步本修复后，42/43/44 都应从 Task 1 完整重新运行。

## 本地验证

- `bash -n LADA/run_TAIL_16shot_seed.sh` 通过。
- LADA 检索 Python 文件无导入语法编译通过。
- 未在本地运行动态检索：本地缺少 PyTorch、GPU 和数据集。
