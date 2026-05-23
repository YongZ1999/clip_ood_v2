# Introduction Writing Guide

## Goal

Automatically generate structured Introduction based on user-provided research background, main content, and contributions (free text).

## Input Format

User provides **free text descriptions** of the following:

1. **Research Background**
   - Why is this problem important?
   - What are the application scenarios?
   - What is the general situation of existing methods?

2. **Main Content**
   - What technical challenge does your work address?
   - What is the core method?
   - Where is the main innovation?

3. **Contributions**
   - Specific contribution points (1-3)
   - Technical advantages of each contribution
   - Performance/efficiency/effectiveness improvements brought

## Workflow

### Step 1: Information Parsing

Extract structured information from user-provided free text:

**Parsing Template**:

```
## Information Parsing Results

### Task and Applications
- **Task Definition**: [Extract from background: what input → what output]
- **Application Scenarios**: [Application domains mentioned in background]
- **Importance**: [Why this task is important]

### Technical Challenge
- **Problems with Existing Methods**: [Limitations of existing methods mentioned in background]
- **Technical Reasons**: [Why this limitation exists]
- **Core Challenge**: [Specific technical problem you solve]

### Core Method
- **Method Overview**: [Method described in main content]
- **Key Innovation**: [Main difference from existing methods]
- **Technical Mechanism**: [Brief implementation idea]

### Contributions
- **Contribution 1**: [Specific content + technical advantage]
- **Contribution 2**: [If applicable]
- **Contribution 3**: [If applicable]
- **Experimental Validation**: [Briefly mentioned if applicable]

---
Please confirm whether the above parsing is accurate. Point out any omissions or deviations.
```

**If parsing is uncertain**, clarify with user:
- "Does 'XXX' you mentioned refer to...?"
- "For technical challenges, my understanding is... Is this correct?"

### Step 2: Automatic Template Matching

Based on parsed information, automatically select the most suitable Introduction template.

**Template Selection Logic**:

| Condition | Selected Template |
|-----------|-------------------|
| Task is relatively new/niche | Version 1: Task → Application |
| Task is well-known | Version 2: Application First |
| Specific setting/new scenario | Version 3: General → Specific |
| Challenge is clear and prominent | Version 4: Open with Challenge |
| Existing task + clear method chain | Technical Challenge V1: Existing Task Chain |
| Traditional methods have similar insight | Technical Challenge V2: Insight-backed |
| Completely new task | Technical Challenge V3: Novel Task |
| Single contribution + multiple advantages | Pipeline V1: One Contribution |
| Two contributions | Pipeline V2: Two Contributions |
| Build on existing pipeline with new module | Pipeline V3: New Module |
| Key observation driven | Pipeline V4: Observation-driven |

**Output template selection description**:

```
## Selected Template

Based on your input, using the following template combination:

**Part 1 (Task Introduction)**: [Template Name]
- Reason: ...

**Part 2 (Technical Challenge)**: [Template Name]
- Reason: ...

**Part 3 (Method)**: [Template Name]
- Reason: ...

Provide feedback if template selection needs adjustment.
```

### Step 3: Generate Introduction Draft

Based on selected templates, generate complete Introduction.

**Standard Structure (5-paragraph)**:

```
Paragraph 1: Task and Applications
        ↓
Paragraph 2: Existing Methods and Limitations (leading to challenge)
        ↓
Paragraph 3: Our Method Overview
        ↓
Paragraph 4: Contributions and Technical Advantages
        ↓
Paragraph 5: Experimental Validation
```

**Generation Principles**:
- One core message per paragraph
- Opening sentence clearly states paragraph主旨
- Clear logic between sentences (cause, contrast, progression)
- Terminology consistent with user input

**Output Format**:

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
```

### Step 4: Iterative Editing (Paragraph-by-Paragraph Mode)

After generating initial draft, enter paragraph-by-paragraph editing mode.

**Edit Command Format**:

User can:
- `"Paragraph 2 needs to emphasize XXX"` → Targeted modification of Paragraph 2
- `"Paragraph 3 is too long, make it concise"` → Compress Paragraph 3
- `"Add a paragraph about YYY between Paragraph 2 and 3"` → Insert new paragraph
- `"Merge Paragraph 1 and 2"` → Paragraph restructuring
- `"Rewrite Paragraph 4 to highlight ZZZ"` → Complete rewrite of a paragraph

**Modification Response Format**:

```
## Modification: Paragraph N

**Modified Content**:
[New paragraph content]

**Modification Notes**:
- Adjusted...
- Highlighted...
- Maintained coherence with context

---

Current Full Introduction:

[Updated complete Introduction]

For further modifications, please specify the paragraph.
```

## Template Quick Reference

### Part 1: Task Introduction Templates

**Version 1 - Task → Application** (Task is relatively new)
```
[Task] targets at recovering/reconstructing/estimating [output] from [input].
It has a variety of applications such as [app1], [app2], and [app3].
```

**Version 2 - Application First** (Task is well-known)
```
[Task] has a variety of applications such as [app1], [app2], and [app3].
```

**Version 3 - General → Specific** (Specific setting)
```
[General task] has a variety of applications such as [app1] and [app2].
This paper focuses on the specific setting of recovering [output] from [input].
```

**Version 4 - Open with Challenge** (Challenge is prominent)
```
[Task/application importance].
Given [input], previous methods usually [approach].
Although they work in many cases, they fail at [failure case] because [reason].
```

### Part 2: Technical Challenge Templates

**V1 - Existing Task Chain** (Existing task, clear method chain)
```
This problem is particularly challenging due to [general challenge].
To overcome this, traditional methods [method1]. However, they [limitation1].
Recently, [method2] [approach]. However, they [limitation2] because [reason].
[Method3] try to [approach]. However, they [limitation3] because [reason].
```

**V2 - Insight-backed** (Traditional methods have similar insight)
```
Traditional methods [approach]. However, they [limitation] because [reason].
To overcome this, a typical approach is [insight], explored in [classical works].
However, these methods still [limitation] because [reason].
Newer methods [approach]. However, they [limitation] because [reason].
```

**V3 - Novel Task** (Completely new task)
```
In this work, our goal is to [goal]. This problem is challenging for [N] reasons.
First, [challenge1].
Second, [challenge2].
Finally, [challenge3].
```

### Part 3: Method Templates

**V1 - One Contribution, Multi Advantages**
```
In this paper, we propose [framework], named [Name], for [task].
The basic idea is illustrated in Figure [N].
Our innovation is in [key novelty].
Specifically, [implementation details].
In contrast to previous methods, [advantage1].
Another advantage is that [advantage2].
```

**V2 - Two Contributions**
```
In this paper, we propose [framework].
Our innovation is in [key novelty], illustrated in Figure [N].
Specifically, [contribution 1 details].
In contrast to ..., [advantage1].
However, [remaining challenge].
To address this, we [contribution 2].
Specifically, [contribution 2 details].
```

**V4 - Observation-driven**
```
Our innovation is [key innovation].
We observe that [key observation].
Considering that, we [method].
This leads to [advantage] and achieves [gain].
```

## Self-Checklist

After generating Introduction, check paragraph by paragraph:

- [ ] Opening sentence of each paragraph clearly states paragraph主旨
- [ ] Each paragraph conveys only one core message
- [ ] Technical challenge, technical reason, and solution mechanism are all clear
- [ ] Claims in Introduction are supported by experiments
- [ ] Terminology is consistent with other sections
- [ ] Logic chain from task to method to contributions is clear
