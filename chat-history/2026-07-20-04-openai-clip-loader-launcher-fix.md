# OpenAI CLIP 加载失败与 transfer-aware launcher 修复

**日期**: 2026-07-20
**会话概况**: 服务器首次启动 transfer-aware E1 时，12 个任务都在加载 OpenAI CLIP 阶段失败。已定位到 safetensors 默认值与服务器本地权重缓存不兼容，并在独立分支修复。

---

## 1. 故障事实

- 失败发生在 `scripts/run_transfer_aware_main_table.sh` 启动后的模型加载阶段，所有六个首波 worker 都退出；后续队列任务也未成功开始。
- 服务器已确认没有残留 `main_incremental.py` 或 launcher 进程。
- 统一错误为：`openai/clip-vit-base-patch16` 没有 `model.safetensors`，却被要求 `use_safetensors=True`。

## 2. 根因

- `src/models/clip.py:get_clip_model()` 过去在未设置环境变量时默认 `CLIP_USE_SAFETENSORS=True`。
- 服务器缓存的 OpenAI CLIP 权重为 `pytorch_model.bin`，这与项目其他正式 launcher 已长期使用的 `CLIP_USE_SAFETENSORS=0` 不兼容。
- 新 transfer-aware launcher 漏掉了这两个既有的环境变量，也未显式写出 `--model_name`，使问题被默认配置触发。

## 3. 修复

- `src/models/clip.py`：未设置环境变量时，OpenAI CLIP 默认 `use_safetensors=False`；SigLIP2 默认保持 safetensors。环境变量仍可显式覆盖。
- `scripts/run_transfer_aware_main_table.sh`：
  - 显式传递 `--model_name openai/clip-vit-base-patch16`；
  - 默认导出 `CLIP_USE_SAFETENSORS=0`；
  - 默认导出 `CLIP_LOCAL_FILES_ONLY=1`；
  - 在 launcher log 记录模型和两个加载变量。
- 更新服务器交接提示与实验方案，禁止将该 OpenAI CLIP run 强制改为 safetensors。

## 4. 验证

- Python 语法编译通过。
- `bash -n scripts/run_transfer_aware_main_table.sh` 通过。
- 六卡 `DRY_RUN=1` 检查通过，12 条生成命令均包含模型名；launcher log 显示 `CLIP_USE_SAFETENSORS=0` 与 `CLIP_LOCAL_FILES_ONLY=1`。
- 本机无 PyTorch/Transformers，实际 `from_pretrained` 加载须在服务器用单测/启动器重新验证。

## 5. 服务器后续动作

```bash
git pull --ff-only origin agent/transfer-gated-ensemble
python tests/test_transfer_aware_ensemble.py
bash scripts/run_transfer_aware_main_table.sh
```

如果仍有加载错误，优先检查 launcher 日志首部的 `model`、`CLIP_USE_SAFETENSORS`、`CLIP_LOCAL_FILES_ONLY` 字段，而不是修改算法代码。
