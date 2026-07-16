# E7 协议修正：使用 Native LADA，而非统一预算 LADA-style

**日期**：2026-07-17  
**状态**：当前有效；修正 `2026-07-17-01-expanded-formal-experiment-plan.md` 中 E7 的 LADA 条目。

## 决策

E7 的问题是跨 backbone 鲁棒性，而不是受控更新次数下的算法比较。因此两种方法应保留各自的原始方法协议，仅将 OpenAI CLIP 底座换为 SigLIP2，并作不可避免的 encoder API 适配。

- LoRA-NF：继续使用正式 CLIP 主表的 800 iterations/task、LoRA-NF + FD/CD + LR-RGDA 配方；
- LADA：改为 Native LADA SigLIP2 port，使用冻结视觉端、task-local AdaptFormer、label-specific memory、DPT spherical-GMM replay、AdamW、OneCycle，以及官方 16-shot 逐数据集 epoch schedule：`40/10/30/100/30/5/200/10/30/10`；
- SigLIP2 LADA 不可写为官方代码零修改运行，因为原仓库硬编码 OpenAI CLIP；论文中应表述为“native LADA algorithm port on SigLIP2”。

此前 `E7__siglip2__lada_style__*` 的统一 800-step 运行是 controlled-budget LADA-style 实验，不纳入 E7 主表。已完成或正在进行的 SigLIP2 LoRA-NF 运行可保留。

## 对运行量的影响

仍是 6 个完整 X-TAIL runs：LoRA-NF 三个 seeds、Native LADA 三个 seeds；每个 run 均有十个增量任务。仅 Native LADA 的三个 runs 必须重新运行。

## 相关实现

- `scripts/main_incremental_lada_native.py`：Native LADA 的独立 SigLIP2 入口；
- `src/lada/lada_trainer.py`：LADA memory/DPT/OneCycle 与 task-local tuner 生命周期；
- `src/lada/native_protocol.py`：官方 16-shot epoch schedule 与默认优化设置；
- `scripts/run_siglip2_robustness.sh`：E7 的 LADA 子命令改为 Native LADA 入口。
