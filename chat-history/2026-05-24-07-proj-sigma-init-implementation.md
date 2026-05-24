# Proj-Σ 初始化方案设计与实现

**日期**: 2026-05-24
**会话概况**: 设计并实现了无 P 的 Proj-Σ 初始化方案——从协方差矩阵 Σ 的特征分解取 V_small，投影 W，SVD 分解为 A, B，做减法消除初始偏移

---

## 1. 关键讨论 / 决策

- **Proj-Σ 方案确定**：无运行时 P，前向为 W = W' + B × A（标准 LoRA 形式）。保护完全来自初始化——从 Σ 的特征向量中取窗口（tail 或 middle），投影 W_t 得到 BA_init，然后做减法 W' = W - BA_init。

- **做减法的必要性澄清**：因为 B 非零（不像当前 LoRA-NSP 中 B=0），所以必须从 W 中减去 BA_init 才能保持初始输出不变：W' + BA_init = W ✅

- **两种窗口都要试**：`window='tail'`（最小特征值方向，类比 LoRA-Null）和 `window='middle'`（中间特征值方向，类比 Least）。

- **持续学习流程**：每个任务开始时从当前合并权重 W_{t-1} 和累积协方差 Σ 做 Proj-Σ 初始化 → 训练 → 合入 → 下一任务重复。

## 2. 架构决策讨论

当前实现将 `initialize_adapters_from_covariance` 作为方法添加到 `LoRACLIPVisionTransformer` 中。讨论中提出：Proj-Σ 本质上是一种不同的方法（无 P，结构化初始化），是否应该作为独立的新类/新方法存在，而不是作为现有框架的"插件"？

备选方案：
- A（当前）：LoRACLIPVisionTransformer 上新增方法，通过参数控制行为
- B：新建 `LoRAProjSigmaVisionTransformer` 类，与 `LoRACLIPVisionTransformer` 平级
- C：新建 `CovInitLoRA` 模块类（替代 SGPBaseLoRA），然后组合到 transformer 中

## 3. 代码改动

`src/models/lora_sgp.py`:
- 新增 `compute_proj_init(W, cov, r, window, s_ratio)` 函数 — Proj-Σ 初始化（从 Σ 投影 W）
- 新增 `compute_weight_svd_init(W, r, window, s_ratio)` 函数 — SVD-W 初始化（从 W 的 SVD 取窗口）
- 新增 `LoRACLIPVisionTransformer.initialize_adapters_from_covariance()` 方法
- 新增 `LoRACLIPVisionTransformer.initialize_adapters_from_weight_svd()` 方法
- （LoRA-NSP 路径）新增 `weight_kind="band_pass"` 和 `weight_kind="high_cut"` 权重函数

三种初始化路径对比：
| 路径 | 信息来源 | 方法 | 前向形式 |
|---|---|---|---|
| LoRA-NSP（现有） | Σ → soft P | `update_projection_matrices(cov)` | W₀ + BAP |
| Proj-Σ（新增） | Σ → V_small → 投影 W | `initialize_adapters_from_covariance(cov, window)` | W' + BA |
| SVD-W（新增） | W 直接 SVD | `initialize_adapters_from_weight_svd(window)` | W' + BA |

## 4. 集成改动

`main_incremental.py`:
- 新增 `--init_mode` 参数（lora_nsp / lora_vanilla / proj_sigma_tail / proj_sigma_middle / weight_svd_tail / weight_svd_middle）
- 训练前：非标准模式时（proj_sigma / weight_svd）调用对应初始化方法
- 训练后：lora_nsp 照常构建 P；proj_sigma 只累积 Σ 不构建 P；weight_svd 只合入

`src/trainers/lora_nsp_trainer.py`:
- `update_covariance_history()` 新增 `update_projection: bool = True` 参数

## 5. 待办事项 / 遗留问题

- [ ] 生成实验配置（tail vs middle 两组，对应不同 init_mode）

## 5. 相关文件

- `src/models/lora_sgp.py`: Proj-Σ 初始化函数 + 新方法 + 权重变体
- `paper_writing/deep-research-reports/lora-null-init/report.md`: 第 8 节讨论记录
