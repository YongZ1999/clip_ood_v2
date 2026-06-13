# LoRA-NSP PEFT 方法在 LLM 上的首次验证

**日期**: 2026-06-13 ~ 2026-06-14  
**会话概况**: 将 LoRA-NSP 抽象为通用 PEFT 方法，在 Qwen2.5-1.5B 上用 GSM8K 进行首轮微调验证。创建了独立项目 lora_nsp_llm，搭建了完整的训练+评测管道，初步验证了 NSP 在保持通用知识（MMLU）和适配下游任务（Math）上的双赢效果。

---

## 1. 关键讨论 / 决策

- **项目定位**: 将 CLIP ViT 上的 LoRA-NSP 抽象为通用 PEFT 方法，测试在 LLM 上的有效性
- **模型选择**: Qwen2.5-1.5B（小模型快速迭代，fp16 ~3GB）
- **参考语料**: 最初计划用 Wikipedia，因 datasets 5.0 不再支持脚本格式数据集，改用已缓存的 MMLU 57 学科数据作为参考语料
- **NSP 模块策略**: down_proj 的 P 矩阵过大 (8960×8960)，排除之；对 Q/K/V/O/up/gate 做 NSP，down_proj 用 vanilla LoRA
- **评测指标**: MMLU（通用知识稳定性）+ 下游 PPL（任务可塑性），两者必须同时报告
- **环境**: 创建专用 conda env `lora_nsp_llm`（Python 3.10），解决 lm_eval + datasets 5.0 兼容性问题

## 2. 重要发现

- **LoRA 参数必须用 float32**: 基座加载为 float16 时，LoRA 参数若用 fp16 会在第二步就产生 NaN loss
- **device_map="auto" 导致跨 GPU 不匹配**: A/B 参数初始化时需显式用 `linear.weight.device`，forward 中 delta 需 cast 到 `linear.weight.dtype`
- **datasets 5.0 兼容性问题**: `trust_remote_code` 被移除，脚本格式数据集（wikipedia, bookcorpus）不可用；需用 MMLU 逐学科加载
- **PyTorch 2.12 forward hook 签名变化**: `register_forward_hook` 的 `with_kwargs=True` 会传入 4 个参数而非 3 个
- **MMLU 基线不一致**: 不同 `eval-limit` 导致同一模型的 MMLU 分数浮动；统一为 `limit=100` 后所有方法的 before 一致

## 3. 实验结果

### Math (GSM8K) — 首轮验证

| 方法 | MMLU Before | MMLU After | 保持率 | 遗忘率 | 下游 PPL ↓ |
|------|-----------|-----------|--------|--------|----------|
| Frozen | 60.84% | 60.84% | 100% | 0% | 3.98 |
| LoRA (rank=8) | 60.84% | 60.60% | 99.60% | 0.40% | 3.22 |
| **LoRA-NSP (rank=8)** | 60.84% | 60.77% | **99.88%** | **0.12%** | **3.20** |

- LoRA-NSP 遗忘率比标准 LoRA 低约 3.3 倍（0.12% vs 0.40%）
- 下游 PPL 略优于 LoRA（3.20 vs 3.22）
- **双赢效果**: 更多的通用知识保持 + 更好的下游适配

## 4. 技术踩坑记录

1. **fp16 训练 NaN**: 修复方法——LoRA A/B 强制 float32，forward 中 delta cast 回 weight.dtype
2. **device_map 设备不匹配**: forward 中 delta.to(linear.weight.device, linear.weight.dtype)
3. **P 矩阵占显存过大**: down_proj 的 P (8960²=80M) 排除，最终 168 个投影矩阵 ~1.6GB
4. **SSL 网络抖动**: MMLU 评测时 hf-mirror 偶发 SSL 断连，用 `HF_DATASETS_OFFLINE=1` 使用缓存
5. **batched tokenize 尺寸 bug**: `len(ds)` 在 map 的 batched 函数中不可用，改用 `len(examples["question"])`
6. **projection 名称匹配**: 从模糊子串匹配改为精确 dict key 匹配

## 5. 项目结构

```
lora_nsp_llm/
├── src/lora_nsp.py            # NSPLoRALinear + VanillaLoRALinear + build_nsp_projection
├── src/model_utils.py         # 模型加载 (hf-mirror) + 替换线性层
├── src/covariance.py          # hook + 激活协方差提取
├── src/trainer.py             # LoRATrainer
├── src/evaluate.py            # MMLU + 困惑度评测
├── scripts/extract_covariance.py  # 提取协方差 + 构建 P 矩阵
├── scripts/run_experiment.py      # 主实验入口
└── scripts/evaluate_mmlu.py       # 独立 MMLU 评测
```

## 6. 待办事项 / 遗留问题

- [ ] 在 MedQA (医疗)、CodeAlpaca (代码)、LegalBench (法律) 上跑剩余实验
- [ ] MMLU eval 目前只跑 `limit=100`，全量跑一遍获取精确结果
- [ ] 下游任务评测目前只用 PPL，可添加生成准确率 (GSM8K exact match)
- [ ] NSP 超参 (eps=0.05, mix_weight=0.02) 尚未调优
- [ ] 仅在 Math 单领域验证，需多领域交叉验证

## 7. 相关文件

- `lora_nsp_llm/PLAN.md`: 完整实验规划
- `lora_nsp_llm/src/lora_nsp.py`: LoRA-NSP 核心实现
- `lora_nsp_llm/scripts/run_experiment.py`: 主实验入口
- 服务器: `/home/raoxuan/projects/lora_nsp_llm/`
- conda env: `lora_nsp_llm` (Python 3.10)
