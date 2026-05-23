# Method Writing Guide

## Goal

Generate technically clear and complete Method section that fulfills the contributions promised in Introduction and enables experiment reproduction. Support code-based method extraction and structured technical writing.

## Input Types

User provides one or more of the following:

1. **Code Repository/Files**: Implementation code (Python, PyTorch, etc.)
2. **Method Description**: Free text or bullet points describing the method
3. **Algorithm Pseudocode**: Draft or outline of key algorithms
4. **Figure/Table Drafts**: Pipeline figure, module structure diagrams
5. **Introduction Context**: The contributions and method overview from Introduction section

## Core Principles

### Connection to Introduction
- Method section must **fulfill every contribution** claimed in Introduction
- Technical terms must be **consistent** with Introduction
- Pipeline steps must match the "Our Method" paragraph in Introduction

### Technical Clarity
- **One paragraph, one message**: Each paragraph describes one component/design choice
- **Motivation before mechanism**: Why this design? → How does it work?
- **Concrete, not abstract**: Specific operations, not vague concepts

### Reproducibility
- Implementation details sufficient for reproduction
- Hyperparameters, network architectures specified
- Clear algorithmic flow

## Workflow

### Step 1: Input Parsing and Structure Extraction

**If code repository provided**:

```
## Code Repository Analysis

### Main Module Structure
- **Entry point**: [Main file/function]
- **Core modules**:
  - Module A: [Purpose], [Key functions]
  - Module B: [Purpose], [Key functions]
- **Data flow**: [Input → Module A → Module B → Output]

### Key Algorithms Identified
1. **Algorithm 1**: [Function name], [Purpose]
   - Input: ...
   - Output: ...
   - Key steps: ...

2. **Algorithm 2**: ...

### Hyperparameters
- Learning rate: ...
- Batch size: ...
- Architecture details: ...

---
Please confirm the extracted structure or provide corrections.
```

**If method description provided**:

```
## Method Description Parsing

### Extracted Components
1. **Input/Output**: [What goes in/out]
2. **Main Pipeline**: [High-level steps]
3. **Module A**: [Description], [Purpose]
4. **Module B**: [Description], [Purpose]
5. **Key Design Choices**: [List innovations]

### Mapping to Introduction Contributions
| Contribution in Intro | Method Section Coverage |
|-----------------------|------------------------|
| Contribution 1: XXX | Section 3.2: [Module A] |
| Contribution 2: YYY | Section 3.3: [Module B] |

---
Confirm component mapping or propose adjustments.
```

### Step 2: Method Section Structure Design

Propose section structure based on method complexity:

```
## Proposed Method Section Structure

\section{Method}
\label{sec:method}

\subsection{Overview}
[1 paragraph: Pipeline summary, match Introduction description]
[Figure: Pipeline diagram with module labels]

\subsection{[Module A Name]}
\label{sec:module_a}
[Motivation paragraph: Why this module? What problem does it solve?]
[Mechanism paragraph: How does it work? Mathematical formulation or algorithm]
[Advantage paragraph: Why is this design better?]

\subsection{[Module B Name]}
\label{sec:module_b}
[Same structure: Motivation → Mechanism → Advantage]

\subsection{Training Objective}
\label{sec:objective}
[Loss functions, optimization strategy]

\subsection{Implementation Details}
\label{sec:implementation}
[Network architectures, hyperparameters, hardware]

---
Confirm section structure or propose modifications.
```

**Alternative Structure for Simple Methods**:

```
\section{Method}
\label{sec:method}

\subsection{Preliminaries}
[Background definitions needed]

\subsection{Method Overview}
[Pipeline description]

\subsection{Detailed Design}
[Component descriptions]

\subsection{Training and Inference}
[Algorithm and implementation]
```

### Step 3: Paragraph-by-Paragraph Generation

For each subsection, generate content following the **Motivation-Mechanism-Advantage** pattern.

**Template for Module Description**:

```latex
\subsection{[Module Name]}
\label{sec:[module_label]}

\textbf{Motivation.}
[Problem statement: What limitation does this module address?]
[Connection to challenge mentioned in Introduction]

\textbf{[Module Name] Design.}
[High-level description of the approach]
[Mathematical formulation or algorithm reference]
[Key operations explained step by step]

\textbf{Advantages.}
[Why this design works better than alternatives]
[Technical benefits: efficiency, accuracy, etc.]
```

**Key Requirements for Each Paragraph**:

1. **Motivation Paragraph**
   - Clearly state what problem this component solves
   - Link back to Introduction's technical challenge
   - Briefly mention why naive solutions fail

2. **Mechanism Paragraph**
   - Start with high-level intuition
   - Provide mathematical formulation if applicable
   - Reference Algorithm X or Figure Y
   - Explain each operation's purpose

3. **Advantage Paragraph** (optional for minor components)
   - Compare with alternative designs
   - Explain technical benefits
   - Connect to Introduction's claimed advantages

### Step 4: Algorithm and Figure Integration

**Algorithm Generation** (from code or description):

```latex
\begin{algorithm}[t]
\caption{[Algorithm Name]}
\label{alg:[name]}
\begin{algorithmic}[1]
\REQUIRE [Input requirements]
\ENSURE [Output]
\STATE [Step 1: description]
\STATE [Step 2: description]
\FOR{[condition]}
    \STATE [Loop operation]
\ENDFOR
\STATE [Final step]
\RETURN [Output]
\end{algorithmic}
\end{algorithm}
```

**Figure Description Template**:

```
Figure [N]: [Title]
- (a) Overview of the proposed method
- (b) Detailed design of [Module A]
- (c) [Specific operation/visualization]

Caption: Overview of our method. (a) The pipeline takes [input] and produces [output] through [stages]. (b) [Module A] addresses [problem] by [mechanism]. (c) [Additional detail].
```

### Step 5: Consistency Check

Before finalizing, verify:

```
## Method Section Consistency Check

### Introduction ↔ Method Alignment
- [ ] Contribution 1 ("We propose X") is covered in Section 3.X
- [ ] Contribution 2 ("We design Y") is covered in Section 3.Y
- [ ] Technical challenge mentioned in Intro is addressed in Method
- [ ] All terms used in Intro are defined in Method

### Technical Completeness
- [ ] Input/output clearly specified for each module
- [ ] Mathematical formulations are correct and consistent
- [ ] Algorithm steps match implementation (if code provided)
- [ ] Implementation details sufficient for reproduction

### Logical Flow
- [ ] Each section builds on previous sections
- [ ] Motivation precedes mechanism in each subsection
- [ ] Figures/algorithms referenced in text before they appear

---
Review and resolve any inconsistencies.
```

## LaTeX Templates

### Mathematical Notation

Establish notation at the beginning:

```latex
\textbf{Notation.}
We denote [description of symbols].
Let $\mathbf{x} \in \mathbb{R}^{d}$ represent [what].
[Additional definitions as needed].
```

### Module Description

```latex
\subsection{[Module Name]}
\label{sec:[label]}

Given [input], our goal is to [objective].
Existing approaches [brief mention of limitations].
To address this, we propose [module name] that [key idea].

Specifically, [module name] consists of [N] components:

\textbf{Component 1.}
[Description and formulation].
We compute:
\begin{equation}
\mathbf{y} = f(\mathbf{x}; \theta)
\end{equation}
where $\theta$ represents [parameters], and $f$ denotes [operation].

\textbf{Component 2.}
[Description]...

After [processing], we obtain [output] which [purpose].
This design [advantage explanation].
```

### Training Objective

```latex
\subsection{Training Objective}
\label{sec:training}

We train our model by minimizing the following objective:
\begin{equation}
\mathcal{L} = \mathcal{L}_{\text{main}} + \lambda \mathcal{L}_{\text{reg}}
\end{equation}
where $\mathcal{L}_{\text{main}}$ denotes [main loss description]:
\begin{equation}
\mathcal{L}_{\text{main}} = [formulation]
\end{equation}
and $\mathcal{L}_{\text{reg}}$ is [regularization term].

We use [optimizer] with learning rate [value] and [other hyperparameters].
```

### Implementation Details

```latex
\subsection{Implementation Details}
\label{sec:implementation}

\textbf{Network Architecture.}
Our [module] uses [architecture description].
Specifically, [layer details, dimensions, activation functions].

\textbf{Training Configuration.}
We train for [N] epochs with batch size [B].
The learning rate starts at [lr] and decays by [factor] every [interval].
We use [hardware] for training.

\textbf{Inference.}
During inference, [specific operations, any differences from training].
```

## Connection to Other Sections

### From Introduction
- Every "contribution" in Intro must have a corresponding subsection in Method
- Technical challenges mentioned in Intro must be addressed by Method design
- Pipeline overview in Intro should match Method overview

### To Experiments
- Implementation details must match experimental setup
- Modules validated in ablation study should be clearly defined
- Evaluation metrics mentioned should be computable from Method description

## Do and Don't

### Do
1. ✅ Start with motivation for each component (why, not just what)
2. ✅ Use consistent mathematical notation throughout
3. ✅ Reference figures and algorithms in the text before they appear
4. ✅ Provide enough detail for reproduction
5. ✅ Connect each design choice to a technical benefit
6. ✅ Define all symbols and terms before using them

### Don't
1. ❌ Don't describe implementation without explaining the rationale
2. ❌ Don't use vague descriptions ("we use a network" → specify architecture)
3. ❌ Don't introduce new terminology not in Introduction (or define it if you must)
4. ❌ Don't dump code; translate to algorithmic description
5. ❌ Don't omit key hyperparameters needed for reproduction

## Self-Checklist

After generating Method section:

- [ ] Every Introduction contribution has corresponding Method coverage
- [ ] Each subsection has clear motivation paragraph
- [ ] Mathematical notation is consistent and defined
- [ ] Algorithms match code (if code provided) or description
- [ ] Implementation details are sufficient for reproduction
- [ ] Figures are referenced and have descriptive captions
- [ ] Technical terms match Introduction
- [ ] Module advantages connect to Introduction claims
