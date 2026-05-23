---
name: xuan-research-paper-writing
description: Automatically generate structured Related Work based on user-provided core and candidate papers; generate Introduction based on user-provided background, content, and contributions. Supports semi-automated human-in-the-loop academic paper writing workflow.
---

# Xuan Research Paper Writing

## Overview

This skill is designed for **semi-automated human-in-the-loop** academic paper writing.

- **Related Work**: Based on user-provided core papers (3-5) and candidate paper pool, automatically organize technical topics and generate structured literature review
- **Introduction**: Based on user-provided research background, main content, and contributions, automatically generate academically sound introduction

## Quick Start

### Step 1: Copy the Template

```bash
cp xuan-research-paper-writing/references/latex_templates/template.tex paper.tex
cp xuan-research-paper-writing/references/latex_templates/neurips_2026.sty .
cp xuan-research-paper-writing/references/latex_templates/checklist.tex .
```

This template is based on **NeurIPS 2026** format and provides a structured LaTeX document with placeholders for each section.

### Template Features

- **NeurIPS 2026 Compliant**: Based on official neurips_2026.sty style file
- **Multiple Track Support**: Main, Position, E&D, Creative AI, Workshop
- **Preprint Mode**: Default mode for arXiv submission
- **Anonymous Ready**: Automatic anonymization for submission
- **Structured Comments**: Clear TODO markers and usage hints

### Step 2: Write Sections One by One

Follow the workflow below, calling the appropriate skill for each section.

## Core Workflow

### Related Work Workflow

```
Input:
  ├── Core papers: 3-5 papers (user-specified most critical papers)
  └── Candidate papers: Paper list or folder path (pool of papers to be filtered)

Processing Steps:
  Step 1: Paper summarization (call paper-summarizer skill)
         ├── For PDF papers: Convert to images using pdf-to-images skill
         ├── Analyze pages via multimodal understanding (ReadMediaFile)
         └── Output: Structured summaries for each candidate paper

  Step 2: Technical topic identification (if user did not specify)
         ├── Analyze core papers to identify technical lineage
         ├── Propose 2-4 technical topics
         └── User confirmation/adjustment

  Step 3: Paper grouping and importance classification
         ├── Assign candidate papers to each technical topic
         └── Determine detail level for each paper:
             ├── Foundational/core method → Medium detail (2-3 sentences)
             └── Related reference → Brief mention (1 sentence)

  Step 4: Generate Related Work draft
         ├── Organize by technical topics
         └── Each topic: Overview → Representative methods → Key limitations → Contrast with your work

Output:
  ├── Technical topic structure (for user confirmation)
  ├── Paper grouping table (paper → topic → importance level)
  └── Related Work draft (LaTeX/Markdown)

How to Use:
  1. Copy output to: template.tex \section{Related Work}
  2. Adjust if needed
```

### Introduction Workflow

```
Input (free text):
  ├── Research background: Problem importance, application scenarios
  ├── Main content: Technical challenge addressed, core method
  └── Contributions: Specific contributions (1-3) and their technical advantages

Processing Steps:
  Step 1: Information parsing
         └── Extract: Task definition, technical challenge, method, contributions

  Step 2: Automatic template matching
         └── Select most suitable Introduction template based on content characteristics

  Step 3: Generate complete Introduction

Output:
  ├── Parsed structured information (for confirmation)
  ├── Selected template description
  └── Introduction draft (5 paragraphs)

How to Use:
  1. Copy output to: template.tex \section{Introduction}
  2. Use paragraph-by-paragraph mode for iterations
```

### Method Workflow

```
Input:
  ├── Code repository/files: Implementation code (Python, PyTorch, etc.)
  └── Introduction context: Method overview from Introduction section

Processing Steps:
  Step 1: Parse input and extract structure
         ├── If code: Extract module structure, data flow, algorithms
         └── If description: Parse components and their relationships

  Step 2: Design section structure
         └── Propose subsection organization

  Step 3: Paragraph-by-paragraph generation
         └── Motivation → Mechanism → Advantage pattern

  Step 4: Algorithm and figure integration

  Step 5: Consistency check with Introduction

Output:
  ├── Method structure overview
  └── Complete Method section (LaTeX)

How to Use:
  1. Copy output to: template.tex \section{Method}
  2. Verify alignment with Introduction contributions
```

### Experiments Workflow

```
Input:
  ├── Experiment draft: Outline of experiments and evaluation metrics
  ├── Baseline papers: Reference papers for main experiment reconstruction
  └── Experiment data: CSV, Excel, JSON, or Markdown tables (Phase A)

Processing Phases:
  Phase B (Design):
    ├── Parse experiment draft
    ├── Reconstruct main experiments from baseline papers (if provided)
    ├── Design ablation study based on method
    └── Output: Section skeleton with placeholders

  Phase A (Data Population):
    ├── Parse and validate data files
    ├── Generate LaTeX tables
    └── Write result narratives

  Phase C (Refinement):
    └── Integrate supplementary experiments or feedback

Output:
  ├── Phase B: Section skeleton
  ├── Phase A: Complete section with data
  └── Phase C: Refined final version

How to Use:
  1. Phase B: Copy skeleton to template.tex \section{Experiments}
  2. Phase A: Replace placeholders with generated tables and text
  3. Phase C: Update with additional results
```

## Execution Rules

1. **DO NOT load all section files at once**; only load the guide for the current section being processed
2. **Related Work must complete paper summarization first** before topic identification and grouping
3. **Introduction must wait for user to provide complete information**, do not assume content
4. **Wait for user confirmation after each step output** before proceeding to next step
5. **Maintain terminology consistency**, unify technical terminology across the entire paper

## Section Guide Index

- **Related Work**: `references/related-work.md` - Paper summarization, topic identification, grouping strategy
- **Introduction**: `references/introduction.md` - Information parsing, template matching, paragraph-by-paragraph editing
- **Method**: `references/method.md` - Code analysis, structure extraction, technical writing
- **Experiments**: `references/experiments.md` - Experiment design, baseline reconstruction, data population
- **Paper Summarizer**: `references/paper-summarizer.md` - PDF to structured summary using multimodal analysis

## Output Format Convention

### Related Work Output

```latex
\section{Related Work}
\label{sec:related}

\textbf{[Topic A Name].}
[Paragraph content with 2-3 sentences for core papers, 1 sentence for related papers]

\textbf{[Topic B Name].}
[Paragraph content...]

\textbf{Relation to Our Work.}
[Optional: Brief summary of how your work differs]
```

### Introduction Output

```latex
\section{Introduction}
\label{sec:intro}

% Paragraph 1: Task and Applications
[Content...]

% Paragraph 2: Technical Challenge
[Content...]

% Paragraph 3: Our Method
[Content...]

% Paragraph 4: Contributions
[Content...]

% Paragraph 5: Experiments
[Content...]

---
For modifications, please specify the paragraph (e.g., "Paragraph 2 needs to emphasize...")
```

### Method Output

```latex
\section{Method}
\label{sec:method}

\subsection{Overview}
[Pipeline summary...]

\subsection{[Module A Name]}
\label{sec:module_a}
\textbf{Motivation.} [Why this module?]
\textbf{[Module Name] Design.} [How it works...]
\textbf{Advantages.} [Why better...]

\subsection{[Module B Name]}
...

\subsection{Training Objective}
...

\subsection{Implementation Details}
...
```

### Experiments Output

```latex
\section{Experiments}
\label{sec:exp}

\subsection{Experimental Setup}
\label{sec:setup}
\textbf{Datasets.} [...]
\textbf{Evaluation Metrics.} [...]
\textbf{Baselines.} [...]
\textbf{Implementation Details.} [...]

\subsection{Main Results}
\label{sec:main}
\begin{table}[t]
\centering
\caption{...}
\label{tab:main}
[Generated LaTeX table with actual numbers]
\end{table}
[Result narrative...]

\subsection{Ablation Study}
\label{sec:ablation}
...
```

## Usage Examples

### Example 1: Write Related Work

```
User: "帮我写 Related Work"
      "核心文献: /path/to/core_papers/*.pdf"
      "候选文献: /path/to/candidate_papers/"

Agent: 
  1. 调用 pdf-to-images 转换所有 PDF
  2. 分析核心文献，识别技术主题
  3. 输出主题结构供确认
  
User: "确认"

Agent:
  4. 分组候选文献
  5. 输出 Related Work LaTeX
  
User: [复制到 paper.tex]
```

### Example 2: Write Introduction

```
User: "帮我写 Introduction"

Agent: "请提供研究背景、主要内容和贡献"

User:
  "背景: 任务X在应用Y中很重要，现有方法Z存在局限..."
  "内容: 我们提出了方法W，通过技术V解决挑战U..."
  "贡献: (1) 新框架 (2) 高效算法 (3) SOTA性能"

Agent:
  1. 解析输入
  2. 输出结构化信息和选用的模板
  
User: "确认"

Agent:
  3. 生成完整 Introduction
  
User: "第3段需要更详细的方法描述"

Agent:
  4. 修改第3段，输出更新后的全文
```

## NeurIPS 2026 Specific Notes

### Page Limits

- **Main Content**: 9 pages (including figures)
- **Unlimited**: References, acknowledgments, checklist, and technical appendices
- **Strict**: Papers exceeding 9 pages will NOT be reviewed

### Submission Modes

The template supports multiple modes via package options:

```latex
% Default (Main Track, double-blind)
\usepackage{neurips_2026}

% Preprint (for arXiv)
\usepackage[preprint]{neurips_2026}

% Accepted final version
\usepackage[main,final]{neurips_2026}

% Other tracks
\usepackage[position]{neurips_2026}      % Position Paper
\usepackage[eandd]{neurips_2026}         % Evaluations & Datasets
\usepackage[creativeai]{neurips_2026}    % Creative AI
```

### Key Formatting Requirements

- **Abstract**: Single paragraph, indented 0.5 inch, 11pt leading
- **Figures**: Caption below figure, explain what it shows and key takeaway
- **Tables**: Caption above table, no vertical rules (use booktabs)
- **References**: Small font (9pt), unnumbered first-level heading
- **Acknowledgments**: Hidden in submission, shown in final version

### Required Checklist

NeurIPS requires a submission checklist. Uncomment at end of document:

```latex
\newpage
\input{checklist}
```

## File Structure

```
xuan-research-paper-writing/
├── SKILL.md                          # 本文件 - 主入口
├── OVERVIEW.md                       # 整体架构文档
└── references/
    ├── latex_templates/              # LaTeX 模板目录
    │   ├── template.tex              # NeurIPS 2026 论文模板
    │   ├── neurips_2026.sty          # NeurIPS 2026 样式文件
    │   └── checklist.tex             # NeurIPS 提交清单
    ├── related-work.md               # Related Work 详细指南
    ├── introduction.md               # Introduction 详细指南
    ├── method.md                     # Method 详细指南
    ├── experiments.md                # Experiments 详细指南
    ├── paper-summarizer.md           # Paper Summarizer 详细指南
    └── paper-review.md               # Paper Review 详细指南 (待完善)
```

## External Dependencies

- **pdf-to-images**: `skills/pdf-to-images/` (已存在)
  - 用于 PDF 转图片，支持多模态分析
  - 用法: `python skills/pdf-to-images/scripts/pdf_to_images.py paper.pdf ./images 300`

## Workflow Diagram

```
Start with template.tex
       │
       ├───► Related Work ───┐
       │    (需要: 核心+候选文献)│
       │                      │
       ├───► Introduction ────┤
       │    (需要: 背景+内容+贡献)│
       │                      ├───► paper.tex (完整论文)
       ├───► Method ──────────┤
       │    (需要: 代码仓库)    │
       │                      │
       └───► Experiments ─────┘
            (需要: 草稿+baseline+数据)
                     │
                     ▼
              Paper Review
              (全文章节一致性检查)
```
