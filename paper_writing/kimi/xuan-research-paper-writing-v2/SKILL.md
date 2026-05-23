---
name: xuan-research-paper-writing-v2
description: Maintains a research paper across drafting, revision, restructuring, evidence updates, and claim rollback. Invoke when the user wants to write or revise any paper section at any project stage.
---

# Xuan Research Paper Writing

## Overview

This skill is designed for **stateful, human-in-the-loop** academic paper writing.

It should treat paper writing as an evolving process rather than a fixed section-by-section pipeline. The user may draft from blueprints or code, revise existing sections, insert new references, restructure argument chains, or roll back claims after negative results.

## Primary Objective

Maintain a consistent paper narrative across project stages by operating on the **current paper state**.

A valid response should first identify:

1. **Operation Type**: what the user is trying to do now
2. **Evidence State**: what support already exists
3. **Impact Scope**: which sections or claims are affected
4. **Next Best Action**: draft, polish, restructure, rollback, or repair consistency

## Operation Types

Use these operation types before routing to any section guide:

- **Draft**: create new content from blueprints, notes, code, figures, or experiment plans
- **Polish**: improve clarity, academic tone, compression, and paragraph flow without changing claims
- **Restructure**: reorganize paragraph order, contribution structure, or narrative logic
- **Evidence Update**: revise text after adding new experiments, new references, or new observations
- **Claim Rollback**: weaken, delete, or reframe unsupported claims after negative or mixed results
- **Consistency Repair**: synchronize terminology, contributions, evidence, and conclusions across sections
- **Derivation Check**: verify an existing mathematical derivation, inspect assumptions, or explore a new formula or theorem extension

## Paper State

Track and preserve the following shared state across the full paper:

```yaml
paper_state:
  task:
    definition: ...
    setting: ...
    applications: [...]
  challenge:
    core_problem: ...
    why_existing_methods_fail: ...
  method:
    one_sentence_summary: ...
    key_modules:
      - name: ...
        role: core_contribution | auxiliary_design | implementation_detail
        evidence_status: supported | partial | conflicting | unknown
  derivations:
    target_formula: ...
    assumptions: [...]
    verified_steps: [...]
    questionable_steps: [...]
    open_questions: [...]
    exploration_goals: [...]
  contributions:
    - claim: ...
      priority: high | medium | low
      evidence: [...]
      status: supported | weakened | removed | pending
  literature_positioning:
    closest_work: [...]
    strongest_baselines: [...]
    novelty_boundary: ...
  knowledge_base:
    root_dir: ./paper-summaries/
    index_file: ./paper-summaries/index.md
    map_file: ./paper-summaries/literature-map.md
    paper_dirs: ./paper-summaries/papers/
    retrieval_sources:
      - arxiv_mcp_if_available
      - arxiv_search
      - web_search
    pending_ingest: [...]
    search_queries: [...]
    candidate_papers: [...]
    mapped_topics: [...]
    unresolved_links: [...]
    index_fields:
      - paper_slug
      - title
      - venue_year
      - primary_topics
      - category
      - relevance
      - novelty_risk
      - status
    map_sections:
      - topic_clusters
      - direct_competitor_links
      - foundational_links
      - technique_dependency_links
      - closest_to_our_work
      - novelty_risk_watchlist
      - open_questions
  terminology:
    preferred_terms: [...]
    banned_or_old_terms: [...]
    math_notation:
      vector: "\\mathbf{x}, \\boldsymbol{\\mu}_c"
      matrix: "\\mathbf{\\Sigma}_c, \\mathbf{I}_d"
      scalar: "c, d, \\lambda_i"
      set: "\\mathcal{C}_t, \\mathcal{H}_t"
      equation_env: "single-line -> equation; multi-step derivation -> align"
      operator_style: "use \\mathcal{L}, \\mathbb{E}, \\arg\max, \\mathrm{diag}, \\mathrm{softmax}, \\|\\cdot\\|"
      notation_rule: "define important symbols near first use or in a Notation paragraph"
  impacts:
    affected_sections: [...]
    next_best_action: ...
```

Use this state as the shared source of truth before revising any section. If part of the state is missing, infer conservatively and mark the missing fields explicitly.

## Mathematical Notation Standard

Use the following notation as the default convention for AI papers and derivation-heavy writing unless the user explicitly overrides it:

| Type | Format | Rule |
|---|---|---|
| Vector | `\mathbf{x}`, `\boldsymbol{\mu}_c` | lowercase bold; Greek letters use `\boldsymbol` |
| Matrix | `\mathbf{\Sigma}_c`, `\mathbf{I}_d` | uppercase bold |
| Scalar | `c`, `d`, `\lambda_i` | italic, not bold |
| Set | `\mathcal{C}_t`, `\mathcal{H}_t` | calligraphic |

Maintain this convention consistently across Method, Experiments, derivations, figure captions, and notation paragraphs.

## Equation Formatting Standard

Use the following equation-formatting rules unless the user explicitly overrides them:

- Use `equation` for a single important formula.
- Use `align` for multi-step derivations, parallel definitions, or equations that need aligned relation symbols.
- Define important symbols near first use or in a dedicated `\textbf{Notation.}` paragraph.
- Keep operators stylistically consistent, such as `\mathcal{L}`, `\mathbb{E}`, `\arg\max`, `\mathrm{diag}`, `\mathrm{softmax}`, and `\|\cdot\|`.
- Prefer readable derivations over compressed notation when explaining a new module or proof idea.

## Quick Start

### Step 1: Copy the Template

```bash
cp xuan-research-paper-writing-v2/references/latex_templates/template.tex paper.tex
cp xuan-research-paper-writing-v2/references/latex_templates/neurips_2026.sty .
cp xuan-research-paper-writing-v2/references/latex_templates/checklist.tex .
```

This template is based on **NeurIPS 2026** format and provides a structured LaTeX document with placeholders for each section.

### Template Features

- **NeurIPS 2026 Compliant**: Based on official neurips_2026.sty style file
- **Multiple Track Support**: Main, Position, E&D, Creative AI, Workshop
- **Preprint Mode**: Default mode for arXiv submission
- **Anonymous Ready**: Automatic anonymization for submission
- **Structured Comments**: Clear TODO markers and usage hints

### Step 2: Operate on Sections as Needed

Follow the workflow below based on the current operation, evidence state, and impacted sections.

## Core Workflow

### Routing Principle

Do **not** assume the user is working linearly from Introduction to Experiments.

Always decide the workflow in this order:

1. Identify the user's **operation type**
2. Read the **current paper state** or infer the missing parts conservatively
3. Detect whether the request is **local** or **cross-section**
4. Route to the relevant section guide only after the operation is clear

### Operation Routing

| Operation Type | Typical User Intent | Default Action | Section Routing |
|---|---|---|---|
| Draft | Start from blueprint, code, notes, figures, or partial ideas | extract structure, identify missing pieces, generate skeleton or first draft | Introduction / Method / Experiments / Related Work |
| Polish | Improve wording, tone, flow, or compression | preserve claims, rewrite locally, return improved text | current section only unless inconsistency is detected |
| Restructure | Reorganize argument order, contribution layout, or paragraph logic | redesign narrative structure before rewriting | Introduction / Method / Related Work / Conclusion |
| Evidence Update | Add new experiments, references, or observations | revise claims and supporting text according to new evidence | Experiments first, then linked sections |
| Claim Rollback | Remove or weaken unsupported contributions or statements | locate affected claims, downgrade wording, delete dependent text, rebuild narrative | Introduction / Method / Experiments / Conclusion |
| Consistency Repair | Align terminology, contributions, evidence, and conclusions | compare section states, list conflicts, and repair them together | all affected sections |
| Derivation Check | Verify an existing derivation or explore a new mathematical extension | inspect assumptions, validate step transitions, mark heuristic leaps, and propose corrected or extended formulas | Method / Paper Review |

### Reference Intake Inference

Do **not** invoke **Paper Summarizer** merely because the user mentions a new PDF or arXiv link.

Invoke it when the user's intent is clearly one of the following:

1. summarize a new paper
2. add a paper into the literature knowledge base
3. prepare references for Related Work or Introduction updates
4. build or refresh the literature map from summarized papers

If the user only wants a local writing revision, keep the current workflow local unless summarization is explicitly requested.

When summaries already exist, prefer reusing `./paper-summaries/index.md`, per-paper summaries, and `./paper-summaries/literature-map.md` before ingesting more files.

Treat `index.md` as the root-level browsing page for the literature knowledge base, and use it to quickly locate high-relevance papers, novelty-risk papers, and topic clusters before deeper reading.

### Literature Retrieval Inference

If the user explicitly asks to:

1. search for related papers
2. survey literature around a topic, method, setting, or challenge
3. expand a paper list from titles, keywords, or baseline names
4. find candidate papers before Related Work or Introduction updates

then trigger a **literature retrieval** step before summarization.

Recommended retrieval order:

1. Prefer **arXiv MCP** if available for arXiv-aware retrieval
2. Otherwise use arXiv search or web search to collect candidate papers
3. Deduplicate candidates by title and venue/year
4. Save the candidate list into the knowledge-base state
5. Summarize only the papers the user explicitly wants ingested or the papers most relevant to the current writing task

Useful retrieval outputs:

- search query used
- candidate paper list
- high-priority ingest list
- likely direct competitors
- likely foundational papers

### Evidence State

Classify the user's current support level before drafting or revising:

- **Idea only**: problem statement, intuition, or blueprint exists but no implementation or results
- **Code available**: implementation exists, but claims should stay mechanistic rather than empirical
- **Partial evidence**: early experiments or a few references exist; claims must stay tentative
- **Strong evidence**: main experiments, ablations, and comparisons support the main storyline
- **Conflicting evidence**: negative or mixed results weaken the original narrative and may require rollback

### Impact Scope

- **Local edit**: one paragraph, one subsection, or one section
- **Cross-section edit**: affects claim wording, contributions, or experiment interpretation in multiple sections
- **Narrative reset**: requires restructuring because evidence no longer supports the original storyline

### Output Strategy

Select the output form based on evidence state and impact scope:

- **Idea only + local edit**: produce outline, paragraph skeleton, or cautious draft with explicit assumptions
- **Code available + local edit**: produce method-oriented draft grounded in implementation details
- **Partial evidence + cross-section edit**: update wording conservatively and mark claims that still need validation
- **Strong evidence + cross-section edit**: revise the affected sections together and keep contribution-evidence alignment explicit
- **Conflicting evidence + narrative reset**: prioritize claim rollback, contribution reprioritization, and section rewrite plan before full redrafting

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

1. **DO NOT load all section files at once**; only load the guide needed for the current operation
2. **Always classify the operation first** before deciding which section workflow to use
3. **Never fabricate evidence, paper content, citations, or experimental conclusions**
4. **If information is incomplete**, prefer structured assumptions, missing-item lists, or skeletons over overconfident drafting
5. **If new evidence conflicts with current claims**, trigger claim rollback or narrative restructuring instead of preserving the old storyline
6. **Related Work must complete paper summarization first** before topic identification and grouping
7. **Maintain terminology consistency** across the entire paper
8. **When a revision affects multiple sections**, explicitly list the impacted sections and repair them together
9. **Maintain mathematical notation consistency** using the shared `paper_state.terminology.math_notation` convention unless the user requests a different style
10. **Maintain equation formatting consistency** for environment choice, operator style, and first-use symbol definitions
11. **For derivation-related requests**, make assumptions explicit, verify step transitions, and clearly label heuristic or exploratory extensions

## Section Dispatch Rules

Route to the following guides after classifying the operation:

- **Introduction**: use for task framing, challenge wording, contribution wording, and cross-section narrative repair
- **Method**: use for code-grounded drafting, module emphasis changes, contribution-to-module remapping, and derivation verification or exploration
- **Experiments**: use first when new results, ablations, or baselines change the evidence state
- **Related Work**: use for literature grouping, novelty repositioning, and updates after adding new papers
- **Paper Review**: use as the quality gate for claim-evidence audit, derivation audit, consistency audit, and narrative reset decisions
- **Paper Summarizer**: use before Related Work when candidate papers have not yet been summarized, or immediately when the user provides new PDFs or arXiv links that should enter the paper knowledge base

## Section Guide Index

- **Related Work**: `references/related-work.md` - operation-aware literature organization and novelty repair
- **Introduction**: `references/introduction.md` - operation-aware framing, claim wording, and paragraph revision
- **Method**: `references/method.md` - code analysis, module reprioritization, and evidence-aware method writing
- **Experiments**: `references/experiments.md` - experiment design, evidence updates, and claim rollback triggers
- **Paper Review**: `references/paper-review.md` - quality gate, claim-evidence audit, and cross-section consistency review
- **Paper Summarizer**: `references/paper-summarizer.md` - literature retrieval support, explicit reference ingest, PDF-to-images analysis, knowledge-base registration, and literature-map refresh

## Global Output Contract

Every non-trivial response should include the following when applicable:

- **Operation Type**: the current action being performed
- **Evidence State**: the current support level behind the wording
- **Impact Scope**: local edit, cross-section edit, or narrative reset
- **Impacted Sections**: all sections that should be revised together
- **Claim Changes**: none, weakened, removed, reframed, or newly supported
- **Next Best Action**: revise text, add evidence, run paper review, or trigger broader restructuring

When evidence is incomplete or conflicting, prefer structured assumptions, rewrite plans, or rollback suggestions over polished but unsafe prose.

For derivation-related requests, also include when applicable:

- **Target Formula or Claim**
- **Assumptions**
- **Verified Steps**
- **Questionable Steps**
- **Suggested Correction or Extension**

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

### Example 1: 从代码和蓝图起草

```
User: "根据项目蓝图和代码，先帮我起草 Introduction 和 Method"

Agent:
  1. 识别 Operation Type = Draft
  2. 识别 Evidence State = Code available
  3. 先构建 `paper_state`
  4. 生成谨慎的 Introduction framing 和 code-grounded Method draft
```

### Example 2: 插入新文献后重构定位

```
User: "我新增了3篇最相关论文，帮我重写 Related Work，并检查 Introduction 是否还在夸大 novelty"

Agent:
  1. 识别 Operation Type = Evidence Update
  2. 更新 literature_positioning 和 novelty_boundary
  3. 重写 Related Work
  4. 联动检查 Introduction 的 problem framing 和 contribution wording
```

### Example 3: 负结果触发 claim rollback

```
User: "新的消融结果说明组件B收益不稳定，帮我更新论文表述"

Agent:
  1. 识别 Operation Type = Claim Rollback
  2. 识别 Evidence State = Conflicting evidence
  3. 标出受影响章节: Introduction, Method, Experiments
  4. 先给出需要降级或删除的 claims
  5. 再联动重写相关章节
```

### Example 4: 投稿前做全局审查

```
User: "帮我做一次 reviewer-style 全文检查，重点看 claim 和 evidence 是否一致"

Agent:
  1. 识别 Operation Type = Consistency Repair
  2. 调用 Paper Review 做 claim-evidence audit
  3. 输出 risk summary, unsupported claims, impacted sections, next best actions
```

### Example 5: 先检索再入库

```
User: "帮我调研和 continual vision-language learning 相关的最新论文，并把最相关的几篇加入知识库"

Agent:
  1. 识别用户先需要 Literature Retrieval
  2. 生成检索 query，并通过 arXiv MCP / arXiv search / web search 收集候选论文
  3. 输出 candidate papers 和 recommended ingest list
  4. 对选中的论文执行 PDF → images → summary 的入库流程
  5. 更新 `index.md`、`literature-map.md` 和 `paper_state.literature_positioning`
```

### Example 6: 推导检验与新公式探索

```
User: "帮我检查这个 loss 推导是否正确，并看看能不能推广到带协方差约束的版本"

Agent:
  1. 识别 Operation Type = Derivation Check
  2. 从 `paper_state.derivations` 或用户输入中提取目标公式与假设
  3. 逐步验证关键变换，标出可证、存疑、跳步之处
  4. 给出修正后的推导链或一个合理的新公式扩展
  5. 如果该推导影响 Method 或理论表述，再联动更新对应章节
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
project-root/
├── paper-summaries/                  # 参考文献知识库根目录
│   ├── index.md                      # 已总结文献总索引
│   ├── literature-map.md             # 文献关系地图
│   └── papers/
│       └── <paper-slug>/
│           ├── images/               # pdf-to-images 输出
│           └── summary.md            # 结构化文献总结
├── xuan-research-paper-writing-v2/
│   ├── SKILL.md                      # 本文件 - 主入口
│   ├── OVERVIEW.md                   # 整体架构文档
│   └── references/
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
