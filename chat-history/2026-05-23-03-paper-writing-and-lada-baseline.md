# AGENTS.md 更新：补充 paper_writing 与 LADA 基线信息

**日期**: 2026-05-23
**会话概况**: 用户指出 `paper_writing/` 是论文写作文件夹，包含 LaTeX 模板和论文草稿；本篇论文的主要基线是 25-ICML 的 LADA 论文。要求将这些信息更新到 AGENTS.md 和 chat-history。

---

## 1. 关键决策

- **`paper_writing/` 的定位**：论文写作专用文件夹，包含 LaTeX 模板（NeurIPS 2026）、论文草稿、参考论文库、作者自己的推导笔记、以及 Kimi paper-writing skill
- **核心基线确认**：本篇论文的主要基线是 **LADA: Scalable Label-Specific CLIP Adapter for Continual Learning**（ICML 2025）
- **与 LADA 的关系**：
  - 评估协议（LADA 指标 Transfer/Average/Last）和实验设置（X-TAIL 10 数据集、16-shot）沿用自 LADA
  - 本论文的创新在于 LADA 之上增加训练端（LoRA-NSP）和推理端（自适应路由）的协同优化
  - LADA 仅做特征蒸馏 + 标签适配器，本文增加跨模态蒸馏、零空间投影、LR-RGDA 集成分类器

## 2. 修改的文件

| 文件 | 改动 |
|------|------|
| `AGENTS.md` | 新增第 2 节「论文写作（paper_writing/）」：目录结构、LADA 基线详情、论文创新对照、写作注意事项；更新目录树包含 paper_writing/；更新 chat-history 列表；后续章节号顺移 |
| `chat-history/2026-05-23-paper-writing-and-lada-baseline.md` | 本次记录 |

## 3. paper_writing/ 关键文件速查

| 文件 | 说明 |
|------|------|
| `paper-template/paper_draft.tex` | 主论文 LaTeX 源文件（NeurIPS 2026 模板） |
| `paper_draft_README.md` | 论文结构、创新点、参考论文说明 |
| `reference_papers/25-ICML-LADA*.pdf` | LADA 论文原文 |
| `my_original_papers/LR-RGDA_原始推导.pdf` | LR-RGDA 数学推导 |
| `my_original_papers/lora_nsp_原始推导.pdf` | LoRA-NSP 数学推导 |
| `kimi/xuan-research-paper-writing-v2/SKILL.md` | Kimi 论文写作 AI skill |

## 4. 待办事项

- [ ] 论文草稿中的实验数据（占位符）需替换为实际实验结果
- [ ] 方法图（Figure 2/3）需要制作添加到论文中
