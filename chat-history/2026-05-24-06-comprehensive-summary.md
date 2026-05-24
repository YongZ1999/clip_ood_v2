# 完整对话总结 — 2026-05-24 第二轮大规模修改

**日期**: 2026-05-24
**会话概况**: 围绕 main_joint.py / main_incremental.py 和 LR-RGDA 分类器进行了一系列修复、功能添加和实验讨论。

---

## 本次对话实际完成的修改

### 1. 评估流程

| 修改 | 文件 | 说明 |
|------|------|------|
| 中间评估格式化总表 | main_joint.py | `run_full_evaluation()` 末尾添加紧凑表格 |
| 测试集 bug 修复 | main_utils.py | `get_xtail_trainloader` 解包取 `train4update` 而非 `test_loader` |
| LADA 指标索引修复 | main_utils.py | Transfer `matrix[j][k]`→`matrix[k][j]`；Average `range(K)`→`range(k,K)` |

### 2. 训练配置

| 修改 | 文件 | 说明 |
|------|------|------|
| Aux head 学习率 5e-3 | lora_nsp_trainer.py | 独立参数组，不随 `--lr` 变化 |
| 默认 `--lr` 回到 1e-4 | main_joint / incremental | 之前临时改为 3e-4，已改回 |
| SCE 损失替代 CE | lora_nsp_trainer.py | `symmetric_cross_entropy_loss()` + `--sce_a/b` |
| Aux 准确率统计 + 所有 loss 的 EMA | lora_nsp_trainer.py | `ema_aux_ce/acc/fd/cd` |
| 默认启用蒸馏 | main_joint.py | `--reference_dataset` 默认 `""` → `"flickr8k"` |

### 3. 推理端

| 修改 | 文件 | 说明 |
|------|------|------|
| Adaptive Ensemble | main_utils.py / lr_rgda_classifier.py | `--adaptive_ensemble`，sigmoid 置信度加权 |
| Ensemble α 更新 | AGENTS.md | 0.8→0.5（对齐代码默认值） |
| LR-RGDA α₁ 更新 | AGENTS.md | 0.6→0.3（对齐代码已更新的默认值） |

### 4. LoRA-NSP 框架

| 修改 | 文件 | 说明 |
|------|------|------|
| 协方差等权平均 | lora_nsp_trainer.py | 替换指数衰减滑动平均 |
| NSP 流程顺序修正 | main_incremental.py / lora_sgp.py | extract→finalize(merge)→update_cov(build P) |
| 投影矩阵 device 修复 | lora_sgp.py | `P.to(device=module.A.device)` |
| Builder device 传递修复 | da_classifier_builder.py | 漏传 `device=self.device` 给 LRRGDA |

### 5. 自适应路由移除

- AGENTS.md 全部更新：两大创新、实验基线、超参数表、代码组织、重要约定

### 6. 多中心 LR-RGDA（已搁置）

- 4 个文件改动，测试效果一般，代码保留

---

## 本次对话前已存在的状态（来自 05-23 记录）

| 项目 | 说明 |
|------|------|
| `num_workers=6` | 已在 05-23-10 设置 |
| EMA 损失/准确率 | 已在 05-23-10 实现（仅 ema_loss / ema_acc） |
| 中间评估回调框架 | 已在 05-23-10 实现（缺格式化表格，本次补充） |
| `reference_dataset=""` 禁用蒸馏 | 已在 05-23-11 设置（本次改为 flickr8k 启用） |
| α₁=0.3, α₂=2.0, α₃=0.5 | 已在 05-23-09 在代码中更新，AGENTS.md 未同步（本次同步） |
| Proj-Σ 初始化实现 | 已在 05-24-07 实现（`initialize_adapters_from_covariance`） |
| `--init_mode` 参数 | 已在 05-24-07 添加到 main_incremental.py |
| `band_pass` / `high_cut` 权重 | 已在 05-24-07 实现 |

---

## 遗留问题

- [ ] 多中心 LR-RGDA 代码保留，待 full-shot 场景再启用
- [ ] Proj-Σ 的实验配置尚未生成
