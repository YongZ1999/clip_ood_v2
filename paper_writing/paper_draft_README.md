# CLIP Continual Learning 论文草稿

本目录包含基于项目"原始代码"生成的论文草稿，聚焦于CLIP持续学习的训练与推理协同优化。

## 论文核心创新

### 1. 训练端：LoRA-NSP (Low-Rank Adaptation with Null Space Parameterization)
- **问题**：CLIP在下游任务适配过程中面临灾难性遗忘问题
- **解决方案**：
  - 引入零空间约束到LoRA参数更新中
  - 通过投影矩阵限制对预训练关键知识的干扰
  - 设计跨模态蒸馏与模态内特征蒸馏相结合的复合蒸馏损失

### 2. 推理端：集成分类器 (Ensemble Classifier)
- **问题**：零样本分类器仅依赖视觉-文本相似度，未能充分利用下游监督数据中的视觉细粒度信息
- **解决方案**：
  - 提出基于LR-RGDA (Low-Rank Regularized Gaussian Discriminant Analysis) 与零样本分类器的集成框架
  - 通过贡献系数 α 动态调节两类分类器的权重分配
  - LR-RGDA仅对分布内(ID)样本输出正置信度，对分布外(OOD)样本输出零置信度

## 文件说明

```
paper_draft.tex          # 主论文文件 (LaTeX格式)
paper_draft_README.md    # 本说明文件
```

## 编译说明

### 环境要求
- TeX Live 2022 或更高版本
- 需要 neurips_2026.sty 样式文件（可从 NeurIPS 官网获取）

### 编译命令
```bash
# 方式1：使用pdflatex
pdflatex paper_draft.tex
bibtex paper_draft
pdflatex paper_draft.tex
pdflatex paper_draft.tex

# 方式2：使用latexmk（推荐）
latexmk -pdf paper_draft.tex
```

## 论文结构

### 1. Introduction (第1节)
- 任务背景与应用场景
- 技术挑战（灾难性遗忘、稳定性-可塑性权衡）
- 本文方法概述（LoRA-NSP + 集成分类器）
- 贡献总结

### 2. Related Work (第2节)
- **Continual Learning Settings**: TIL, CIL, MTIL, X-TAIL
- **Continual Learning with VLMs**: replay-based, regularization-based, architecture-based
- **Parameter-Efficient Fine-Tuning**: prompt-based methods, adapters, LADA
- **Inference-Time Adaptation**: RAIL 等方法
- **Relation to Our Work**: 与现有方法的区别

### 3. Method (第3节)
- **3.1 Preliminaries**: X-TAIL设置、CLIP模型
- **3.2 LoRA-NSP**: 训练端方法，包含动机、设计、复合蒸馏损失
- **3.3 Ensemble Classifier**: 推理端方法，包含LR-RGDA和集成框架
- **3.4 Adaptive Routing**: OOD检测与路由机制

### 4. Experiments (第4节)
- **4.1 Experimental Setup**: 数据集、评估指标、基线方法、实现细节
- **4.2 Main Results**: 16-shot和full-shot设置的主要结果
- **4.3 Ablation Study**: 各组件的贡献分析
- **4.4 Analysis of Ensemble Classifier**: 贡献系数α的影响
- **4.5 OOD Detection Performance**: OOD检测性能评估

### 5. Conclusion (第5节)

## 参考论文

论文写作参考了以下文献的相关章节：

### 核心参考（相关工作与实验结构）
- **LADA** (25-ICML): "Scalable Label-Specific CLIP Adapter for Continual Learning"
  - 相关工作组织结构参考
  - 实验章节结构参考
  - 评估指标定义参考

### 其他重要参考
- **ZSCL** (23-CVPR): Preventing Zero-Shot Transfer Degradation
- **Mod-X** (23-ICML): Continual Vision-Language Representation Learning with Off-Diagonal Information  
- **RAIL** (24-NIPS): Advancing Cross-domain Discriminability
- **MG-CLIP** (25-ICCV): Mind the Gap
- **C-CLIP** (25-ICLR): Multimodal Continual Learning

## 注意事项

1. **引用格式**: 论文使用 plain bibstyle，需要确保所有引用都有对应的bib条目
2. **表格数据**: 实验表格中的数据为示例数据，需要根据实际实验结果进行替换
3. **图表**: 论文中提到的 Figure 2, Figure 3 等需要在实际写作时添加
4. **超参数**: 实现细节中的超参数值需要根据实际调优结果进行更新

## 下一步工作

1. 补充实验数据到表格中
2. 添加可视化图表（方法框架图、实验结果图等）
3. 完善引用文献列表
4. 根据实际情况调整方法和实验描述
5. 进行内部一致性检查
