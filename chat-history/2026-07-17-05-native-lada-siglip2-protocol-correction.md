# Native LADA SigLIP2 协议修正

**日期**：2026-07-17  
**会话概况**：用户明确 E7 的目标是方法跨 backbone 鲁棒性，因此修正原先统一 800-step 的 LADA-style 分支为保留 LADA 原始训练语义的 SigLIP2 native port。

## 关键决定

- LoRA-NF 保留其正式主表的 800 iterations/task 配方；
- Native LADA 保留冻结视觉端、task-local AdaptFormer、label-specific memory、DPT replay、AdamW、OneCycle 和官方 16-shot epoch schedule；
- 官方 LADA 不能零修改加载 SigLIP2，故该条目是 algorithm-native port，不是官方仓库的字节级直接运行；
- 原 `lada_style` 统一预算结果不再用于 E7 主表；E2/E3 检索组件结果与本修正无关。

## 代码变更

- 新增 `scripts/main_incremental_lada_native.py`；
- 新增 `src/lada/native_protocol.py`，集中保存 LADA 16-shot schedule；
- 重写 `src/lada/lada_trainer.py` 的 shared native LADA 路径，使其支持 SigLIP2 features、DPT、OneCycle 和每任务重置 AdaptFormer；
- `scripts/run_siglip2_robustness.sh` 的 LADA 子任务改为 native 入口；
- 更新 E7 计划、服务器交接和结果模板中的结果文件名为 `lada_native`。

## 服务器动作

- 保留已完成的 `E7__siglip2__lora_nf__*` 结果；
- 取消或忽略 `E7__siglip2__lada_style__*` 结果；
- 在新提交拉取后运行三次 `E7__siglip2__lada_native__*`，再以新的文件名汇总。
