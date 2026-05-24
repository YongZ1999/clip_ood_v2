# LoRA-Null / Least but not Last 文献调研与结合讨论

**日期**: 2026-05-24
**会话概况**: 围绕 LoRA-NSP 框架与 LoRA-Null、Least but not Last、PiSSA、MiLoRA 等文献的对比讨论，在三轴设计空间中定位我们的方法，提炼可实验的变体方案，并记录到 deep-research-reports

---

## 1. 关键讨论 / 决策

- **三轴设计空间确认**：LoRA 知识保留方法可沿 X（奇异谱窗口）、Y（知识表征层次/分解对象）、Z（约束机制）三个轴定位。我们的 LoRA-NSP 是唯一同时使用特征协方差空间、运行时前向投影约束、软权重的方法。

- **分解对象的精确区分**：讨论澄清了各文献的分解对象差异——
  - PiSSA/MiLoRA/Least：对 **W₀（预训练权重矩阵）** 做 SVD
  - LoRA-Null：对 **X_pre（校准集激活矩阵）** 做 SVD，取左奇异向量 U_null
  - 我们 (LoRA-NSP)：对 **Σ_Cov = X^T X（特征 Gram 矩阵）** 做特征分解
  - 三者是不同的数学对象，类比时需要明确限定条件

- **U 形曲线是假说不是结论**：Least 的 U 形遗忘曲线在 **W₀ 的奇异值谱**上验证，推广到我们的 **Σ_Cov 特征值谱**是一个待实验验证的假说，不能直接断定。

- **校准集的角色澄清**：LoRA-Null 用校准集（NQ Open, 256条）构建 X_pre 以获取激活零空间；我们不需要校准集，因为 Σ_Cov 来自任务数据自身的特征。

## 2. 重要发现

- **带通 P 变体是最有理论贡献的方向**：将 P 从 low-pass 改为 band-pass（同时抑制极大和极小特征值方向），改动约 10 行代码，直接回应 U 形假说，文献中无人做过。

- **LoRA-NSP 与 LoRA-Null 本质互补**：LoRA-Null 保护预训练知识（校准集→初始化吸收式，无运行时 P），LoRA-NSP 保护任务知识（任务协方差→运行时投影式）。可结合为双投影 P = P_pre × P_task。

- **四类变体方案**：带通 P（变体 A）、Least 初始化 + 运行时 P（变体 B）、固定 B（变体 C，你之前提的 idea）、双投影 P（变体 D）

- **实验矩阵包含 6 个实验**：从基线到最强组合，覆盖不同三轴组合

## 3. 待办事项 / 遗留问题

- [ ] 带通 P 变体的代码实现（`compute_weights` 中新增 `weight_kind="band_pass"`）
- [ ] 实验验证 U 形假说是否在 Σ_Cov 特征值谱中成立

## 4. 相关文件

- `project_clip_continual_learning/paper_writing/deep-research-reports/lora-null-init/report.md`: 追加了第 8 节讨论内容
- `project_clip_continual_learning/paper_writing/deep-research-reports/lora-null-init/papers/2602.03493/essence_analysis.md`: Least but not Last 的本质分析
- `project_clip_continual_learning/paper_writing/deep-research-reports/lora-null-init/papers/2503.02659/essence_analysis.md`: LoRA-Null 的本质分析
- `project_clip_continual_learning/src/models/lora_sgp.py`: 当前 LoRA-NSP 实现
