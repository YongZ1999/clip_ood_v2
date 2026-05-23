# Xuan Research Paper Writing - 整体架构

## Skill 依赖关系图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Xuan Research Paper Writing                        │
│                         (主入口: SKILL.md)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
         ┌──────────────┬─────────────┼─────────────┬──────────────┐
         ▼              ▼             ▼             ▼              ▼
   ┌──────────┐   ┌──────────┐  ┌──────────┐  ┌──────────┐   ┌──────────┐
   │Abstract  │   │Introduc- │  │ Related  │  │  Method  │   │Experiments│
   │          │   │  tion    │  │  Work    │  │          │   │          │
   └────┬─────┘   └────┬─────┘  └────┬─────┘  └────┬─────┘   └────┬─────┘
        │              │             │             │              │
        │              │             ▼             │              │
        │              │      ┌──────────────┐    │              │
        │              │      │Paper         │    │              │
        │              │      │Summarizer    │    │              │
        │              │      └──────┬───────┘    │              │
        │              │             │            │              │
        │              │             ▼            │              │
        │              │      ┌──────────────┐    │              │
        │              │      │ pdf-to-images│    │              │
        │              │      │   (外部skill) │    │              │
        │              │      └──────────────┘    │              │
        │              │                          │              │
        └──────────────┴──────────────────────────┴──────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │      Paper Review              │
                     │  (最终自审, 调用所有section)    │
                     └───────────────────────────────┘
```

## 工作流顺序

```
Phase 1: 内容准备
├── 1. Paper Summarizer
│   └── 输入: PDF/arXiv 论文
│   └── 输出: 结构化文献总结
│
Phase 2: 核心章节写作
├── 2. Related Work
│   ├── 输入: 核心文献(3-5) + 候选文献池
│   ├── 依赖: Paper Summarizer (预处理)
│   └── 输出: 结构化相关工作
│
├── 3. Introduction  
│   ├── 输入: 背景 + 内容 + 贡献 (自由文本)
│   └── 输出: Introduction 草稿
│
Phase 3: 技术实现
├── 4. Method
│   ├── 输入: 代码仓库 + Introduction 上下文
│   └── 输出: Method 章节
│
├── 5. Experiments
│   ├── 输入: 实验草稿 + baseline论文 + 数据
│   ├── 阶段B: 设计 → 章节骨架
│   ├── 阶段A: 数据填充 → 完整章节
│   └── 阶段C: 补充完善
│
Phase 4: 摘要与结论
├── 6. Abstract
│   └── 基于全文生成
│
├── 7. Conclusion
│   └── 总结贡献 + 未来工作
│
Phase 5: 自审
└── 8. Paper Review
    └── 全文章节一致性检查
```

## LaTeX 模板与 Skill 对应

```
paper.tex
├── \\section{Abstract}              → Abstract Skill
│   └── (最后写, 基于全文)
│
├── \\section{Introduction}         → Introduction Skill
│   └── Para 1: Task & Application
│   └── Para 2: Technical Challenge  
│   └── Para 3: Our Method
│   └── Para 4: Contributions
│   └── Para 5: Experiments
│
├── \\section{Related Work}         → Related Work Skill
│   └── \\textbf{Topic A}: ...     (核心/基础方法)
│   └── \\textbf{Topic B}: ...     (相关参考)
│   └── \\textbf{Relation to Our Work}: ...
│
├── \\section{Method}               → Method Skill
│   └── \\subsection{Overview}
│   └── \\subsection{Module A}     (Motivation → Mechanism → Advantage)
│   └── \\subsection{Module B}
│   └── \\subsection{Training Objective}
│   └── \\subsection{Implementation Details}
│
├── \\section{Experiments}          → Experiments Skill
│   └── \\subsection{Experimental Setup}
│   └── \\subsection{Main Results}      (baseline重构 + 数据填充)
│   └── \\subsection{Ablation Study}    (方法定制)
│   └── \\subsection{Additional Analysis}
│
├── \\section{Conclusion}           → Conclusion Skill
│   └── (总结 + 局限 + 未来工作)
│
└── \\bibliography{...}
```

## 用户使用流程

```
1. 准备 LaTeX 模板
   └── 复制 template.tex 开始

2. 依次调用 Skill
   ├── 用户: "帮我写 Related Work"
   │   └── 提供: 核心文献PDF + 候选文献文件夹
   │   └── Agent: 输出 \section{Related Work} 内容
   │   └── 用户: 粘贴到 paper.tex
   │
   ├── 用户: "帮我写 Introduction"
   │   └── 提供: 背景、内容、贡献(自由文本)
   │   └── Agent: 输出 \section{Introduction} 内容
   │   └── 用户: 粘贴到 paper.tex
   │
   ├── ... (其他章节类似)

3. 最终整合
   └── Paper Review Skill 检查全文一致性
```

## 文件组织

```
working-directory/
├── paper.tex                    # 主 LaTeX 文件
├── references.bib               # 参考文献
├── paper-summaries/             # Paper Summarizer 输出
│   ├── paper1_summary.md
│   ├── paper2_summary.md
│   └── ...
├── figures/                     # 图表
│   ├── pipeline.pdf
│   ├── results/
│   └── ...
└── experiments/                 # 实验数据
    ├── main_results.csv
    └── ablation_results.xlsx
```
