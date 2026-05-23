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

## 工作流原则

```
该 skill 不假设用户按固定章节顺序写作，而是围绕“当前论文状态”处理操作。

Step 1: 识别 Operation Type
├── Draft
├── Polish
├── Restructure
├── Evidence Update
├── Claim Rollback
├── Consistency Repair
└── Derivation Check

Step 2: 判断 Evidence State
├── Idea only
├── Code available
├── Partial evidence
├── Strong evidence
└── Conflicting evidence

Step 3: 判断 Impact Scope
├── Local edit
├── Cross-section edit
└── Narrative reset

Step 4: 基于 `paper_state` 路由到对应 guide
├── Introduction: framing / contribution wording / narrative repair
├── Method: code-grounded drafting / module reprioritization / derivation verification
├── Experiments: evidence update / negative-result handling
├── Related Work: literature organization / novelty repositioning
├── Paper Review: quality gate / claim-evidence audit / derivation audit
└── Paper Summarizer: candidate paper preprocessing
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

2. 在任意项目阶段调用 skill
   ├── 从蓝图/代码起草章节
   ├── 基于现有草稿润色
   ├── 插入新文献后重构 Related Work 或 Introduction
   ├── 根据新实验结果更新 Experiments
   └── 当某个创新点不成立时回滚 claim 并联动修改多个章节

3. 输出时始终显式说明
   ├── 当前 Operation Type
   ├── 当前 Evidence State
   ├── 当前 Impact Scope
   ├── impacted sections
   ├── claim changes
   └── next best action

4. 定期运行 Paper Review
   └── 做 claim-evidence audit 和跨章节一致性检查
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
