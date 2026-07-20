# 损坏 Hugging Face 缓存的 AttributeError 回退修复

**日期**: 2026-07-20
**会话概况**: 服务器验证 offline OpenAI `.pt` fallback 时发现，部分损坏的 Hugging Face cache 不会抛出 `OSError`，而是内部抛出 `AttributeError`，导致回退分支未被执行。

---

## 1. 根因

- `transformers` 在解析失效 blob/缺失本地文件的某些路径会抛出 `AttributeError`。
- `src/models/clip.py:get_clip_model()` 原本只捕获 `OSError`，所以该异常直接终止，无法进入本地 `.pt` → Hugging Face CLIPModel 转换路径。

## 2. 修复

- 模型加载和 tokenizer 加载两处均改为捕获 `(OSError, AttributeError)`。
- `_load_openai_pt_fallback()` 的异常参数标注放宽为 `Exception`，以准确传递两种加载失败原因。
- 仍只对正式 OpenAI ViT-B/16 且 `CLIP_OPENAI_PT_FALLBACK=1` 的场景启用；SigLIP2 与其他模型不受影响。

## 3. 验证与后续

- [ ] 本机完成静态编译与 diff 检查后推送 `agent/transfer-gated-ensemble`。
- [ ] 服务器拉取后先运行 `python scripts/check_openai_pt_hf_compat.py --device cuda:0`；通过后重新启动 12 个 E1 任务。

## 4. 相关文件

- `src/models/clip.py`
- `src/models/openai_clip_compat.py`
- `scripts/check_openai_pt_hf_compat.py`
