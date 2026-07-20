# 离线 OpenAI CLIP `.pt` 与 LoRA-NF 兼容恢复

**日期**: 2026-07-20
**会话概况**: transfer-aware E1 在服务器上全部停在模型加载阶段。服务器的 Hugging Face CLIP cache 已损坏且 DNS 不可用，但本地仍保存 OpenAI `ViT-B-16.pt`。本次增加保持 LoRA-NF 模块接口的离线恢复路径。

---

## 1. 故障复核与关键决策

- 先前“服务器拥有可用 `pytorch_model.bin` cache”的判断不成立：实际 cache 的权重 blob 已失效，且服务器无法解析 Hugging Face/GitHub 域名，因此关闭 safetensors 仍不能加载模型。
- 不能把 `get_clip_model()` 直接改为 `LADA.clip.load()`/OpenAI `clip.load()`：OpenAI 原生 CLIP 将 QKV 融合为 `in_proj_weight`，而 LoRA-NF 的注入器依赖 Hugging Face `vision_model/text_model.encoder.layers` 及独立的 `q_proj/k_proj/v_proj`。
- 采用离线 **权重转换**，而不是替换模型架构：读取 `~/.cache/clip/ViT-B-16.pt`，建立等价 Hugging Face `CLIPModel`，拆分 QKV 并映射 vision/text transformer、投影层与 logit scale。转换后的模型仍走原有 LoRA-NF、蒸馏、LR-RGDA 路径。
- 该 fallback 仅支持 OpenAI ViT CLIP（正式 E1 的 `openai/clip-vit-base-patch16`），不作用于 SigLIP2，也不支持 OpenAI ResNet CLIP。

## 2. 代码与运行行为

- 新增 `src/models/openai_clip_compat.py`：
  - 从 JIT 或 state-dict `.pt` 读取权重；
  - 推断 ViT/text 配置，构造 Hugging Face `CLIPModel`；
  - 将 OpenAI 融合 QKV 拆分为 HF Q/K/V 线性层；
  - 复用项目内 OpenAI BPE tokenizer，避免依赖损坏的 HF tokenizer cache。
- `src/models/clip.py:get_clip_model()`：优先保持原有 HF `from_pretrained()` 路径；只有该路径在正式 OpenAI ViT-B/16 上触发 `OSError` 且本地 `.pt` 存在时才回退转换。
- 健康的 HF 模型 cache 如果只是 tokenizer 缺失，会保留已加载 HF 模型，仅使用本地 OpenAI tokenizer；不会不必要地替换权重。
- `scripts/check_openai_pt_hf_compat.py`：在正式启动前比对原生 OpenAI `.pt` 与转换模型的 tokenizer、图像特征和文本特征；任一 minimum cosine < 0.9999 即失败。
- launcher 新增 `CLIP_OPENAI_PT_FALLBACK=1`，文档已明确这不是直接 `clip.load()`。

## 3. 验证与限制

- 本机静态验证通过：Python 编译、shell 语法和 `git diff --check`。
- 本机没有 PyTorch/Transformers 及 GPU，无法读取 335 MB `.pt` 做数值验证；服务器必须先运行 `python scripts/check_openai_pt_hf_compat.py --device cuda:0`。
- 若该检查通过，才能运行 12 个 transfer-aware E1 任务；若失败，不应改用原生 `clip.load()` 训练，而应提交完整报错后继续修正映射或恢复服务器 HF cache/DNS。

## 4. 后续操作

- [ ] 推送本次修复到 `agent/transfer-gated-ensemble`。
- [ ] 服务器切到该最新 commit，执行兼容性检查。
- [ ] 检查通过后再启动 `scripts/run_transfer_aware_main_table.sh`；此前 12 个失败任务没有产生训练结果，需从头启动。

## 5. 相关文件

- `src/models/openai_clip_compat.py`
- `src/models/clip.py`
- `scripts/check_openai_pt_hf_compat.py`
- `scripts/run_transfer_aware_main_table.sh`
- `docs/server_model_prompt_transfer_aware_main.md`
- `docs/transfer_aware_main_experiment.md`
