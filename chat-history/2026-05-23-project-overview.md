# project_clip_continual_learning — 项目概览与初始化沟通

**创建日期**: 2026-05-23
**目的**: 记录首次对项目进行全局阅读和理解的关键要点

---

## 1. 对话概况

用户要求阅读 `project_clip_continual_learning` 项目的全部内容，并在项目内建立 `chat-history` 文件夹保存对话要点。

## 2. 项目全景理解

### 项目目标
解决 CLIP 模型在下游任务适配中的**灾难性遗忘问题**，通过训练端与推理端协同优化，实现持续学习。

### 三大核心创新

| 创新 | 技术 | 位置 | 状态 |
|------|------|------|------|
| **LoRA-NSP 抗遗忘微调** | 零空间参数化 + 低秩适应 + 复合蒸馏损失 | `src/trainers/lora_nsp_trainer.py`, `src/models/lora_sgp.py` | ✅ 完成 |
| **LR-RGDA 集成分类器** | 低秩分解正则高斯判别分析 + 零样本分类器集成 | `src/classifiers/lr_rgda_classifier.py` | ✅ 完成 |
| **自适应路由分类器** | 基于 LR-RGDA 的 OOD 检测 + 动态分类器选择 | `src/routing/adaptive_router.py` | ✅ 完成 |

### 架构层次
- **核心模块层** (`src/`): trainers / classifiers / detectors / routing / models / utils
- **核心脚本层** (`scripts/core/`): train_clip / extract_stats / build_classifier / evaluate
- **工作流层** (`scripts/workflows/`): 串联核心脚本完成完整实验

### 关键设计决策
- **基于统计分布而非原始数据**: 分类器和 OOD 检测器完全基于 `stats_dict = {class_id: GaussianStatistics}` 构建，支持增量学习
- **训练端与推理端正交**: 两组创新可独立应用
- **滑动平均协方差历史**: 支持从 checkpoint 恢复，增量学习场景下自动应用零空间约束

### 实验设置
- **基础模型**: `openai/clip-vit-base-patch16`
- **数据集**: X-TAIL 10 个数据集 (aircraft, caltech101, dtd, eurosat, flowers, food101, mnist, oxford_pets, stanford_cars, sun397)
- **参考数据集**: Flickr8K（用于知识蒸馏）
- **LoRA 秩**: 4 (训练), rank=32 (LR-RGDA)
- **训练迭代**: 800 iterations, lr=1e-4
- **损失权重**: fd_weight=1.0, cd_weight=1.0

### 项目完成状态
- 核心功能 ~85% 完成
- 文档完善（PROJECT_DOCUMENTATION.md + PRESENTATION.md）
- 待完善: 消融实验框架、代码清理、单元测试

## 3. 相关文件对照

| 文件 | 内容 | 说明 |
|------|------|------|
| `PROJECT_DOCUMENTATION.md` | 完整项目文档 | 架构、模块详解、API 参考 |
| `PRESENTATION.md` | 研究汇报 | 从动机到技术的完整介绍 |
| `main.py` | 主程序 | 联合微调 + 增量学习双模式 |
| `main_incremental.py` | 增量学习入口 | - |
| `main_joint.py` | 联合微调入口 | - |
| `demo_ood.ipynb` | 完整 Demo | 最完整的实现参考 |
| `25-ICML-LADA*.pdf` | 论文原文 | LADA: Label-Specific CLIP Adapter |

## 4. 后续可能的关注点

- 消融实验框架的完善（alpha 敏感性分析、OOD 检测器对比）
- 超参数自动化优化流程
- 代码清理与模块化重构遗留问题
- 正式论文的实验结果复现
