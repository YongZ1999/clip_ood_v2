# Method Writing Guide

## Goal

Draft, revise, restructure, or repair the Method section based on the **current paper state**. This guide should support code-based extraction, narrative reorganization, contribution reprioritization, and claim rollback when experiments no longer justify emphasizing certain modules.

## Invocation Contract

Before writing, identify:

1. **Operation Type**: Draft / Polish / Restructure / Evidence Update / Claim Rollback / Consistency Repair / Derivation Check
2. **Evidence State**: Idea only / Code available / Partial evidence / Strong evidence / Conflicting evidence
3. **Impact Scope**: local subsection edit / full Method rewrite / cross-section repair

Useful inputs include one or more of the following:

1. **Code Repository/Files**
2. **Method Description**
3. **Algorithm Pseudocode**
4. **Figure/Table Drafts**
5. **Introduction Context**
6. **Experiments Context**: ablation findings, negative results, or updated evidence about each component
7. **Derivation Input**: target formula, theorem statement, proof sketch, or the user's exploratory idea

## Read From `paper_state`

Prioritize these shared fields before generating or revising text:

- `method.one_sentence_summary`, `method.key_modules[*].name`
- `method.key_modules[*].role`, `method.key_modules[*].evidence_status`
- `contributions[*].claim`, `contributions[*].status`
- `challenge.core_problem`
- `terminology.preferred_terms`, `terminology.banned_or_old_terms`
- `terminology.math_notation.vector`, `terminology.math_notation.matrix`
- `terminology.math_notation.scalar`, `terminology.math_notation.set`
- `terminology.math_notation.equation_env`, `terminology.math_notation.operator_style`
- `terminology.math_notation.notation_rule`
- `derivations.target_formula`, `derivations.assumptions`
- `derivations.verified_steps`, `derivations.questionable_steps`
- `derivations.open_questions`, `derivations.exploration_goals`
- `impacts.affected_sections`

## Core Principles

### Connection to Introduction and Experiments
- Method section must match the **current contribution set**, not an outdated one
- Technical terms must be **consistent** with Introduction
- Pipeline steps must match the current "Our Method" description in Introduction
- Module emphasis must reflect experimental support rather than the original intent alone

### Technical Clarity
- **One paragraph, one message**: Each paragraph describes one component/design choice
- **Motivation before mechanism**: Why this design? → How does it work?
- **Concrete, not abstract**: Specific operations, not vague concepts

### Reproducibility
- Implementation details sufficient for reproduction
- Hyperparameters, network architectures specified
- Clear algorithmic flow

## Workflow

### Step 0: Route the Request

Choose the workflow before editing text:

- **Draft**: extract pipeline structure from code, description, or figures and generate a first Method draft
- **Polish**: improve clarity, notation, and subsection flow without changing module priority
- **Restructure**: reorder subsections or redesign the method narrative around the actual main idea
- **Evidence Update**: revise module descriptions after new ablations, baselines, or observations
- **Claim Rollback**: downgrade a module from core contribution to auxiliary design or implementation detail when evidence is weak
- **Consistency Repair**: align Method with Introduction, Experiments, Abstract, and Conclusion
- **Derivation Check**: verify existing derivations, surface hidden assumptions, or explore a mathematically plausible extension

### Derivation Workflow

For derivation-related requests, use the following order:

1. restate the **target formula or theorem claim**
2. list all explicit and implicit **assumptions**
3. verify the derivation step by step
4. mark each step as verified, heuristic, or questionable
5. if needed, provide a corrected derivation or a bounded extension idea

Preferred output structure:

```
## Derivation Audit

### Target
- [formula / claim]

### Assumptions
- [assumption 1]
- [assumption 2]

### Step Check
1. [step] → verified / heuristic / questionable
2. ...

### Result
- Correct as written / needs correction / extension candidate

### Suggested Revision
- [corrected formula or extension]
```

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

Propose section structure based on method complexity and current claim strength:

- **Core contribution**: deserves its own subsection with motivation, mechanism, and advantage
- **Supported auxiliary design**: may share space inside a larger subsection
- **Weak or mixed-support component**: describe conservatively and avoid over-positioning it as a headline contribution
- **Implementation-only detail**: move to implementation details instead of treating it as a core module

Propose section structure accordingly:

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
   - Only claim advantages that are supported or at least not contradicted by current evidence

**Evidence-Aware Writing Rules**:
- **Idea only**: focus on design intent and mechanism, not validated benefit
- **Code available**: explain implementation-backed behavior, but avoid empirical superiority claims
- **Partial evidence**: describe promising components cautiously and avoid overcommitting
- **Strong evidence**: highlight experimentally supported modules as the main method story
- **Conflicting evidence**: reduce emphasis on weak modules and rewrite the method narrative around what still holds

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

### Step 5: Consistency Check and Claim Impact Review

Before finalizing, verify:

```
## Method Section Consistency Check

### Introduction ↔ Method Alignment
- [ ] Every current contribution is covered in Method
- [ ] Technical challenge mentioned in Intro is addressed in Method
- [ ] All terms used in Intro are defined in Method

### Method ↔ Experiments Alignment
- [ ] Every experimentally validated key module is clearly defined
- [ ] Weak or unsupported modules are not overstated as core contributions
- [ ] Ablation terminology matches the Method subsection names

### Technical Completeness
- [ ] Input/output clearly specified for each module
- [ ] Mathematical formulations are correct and consistent
- [ ] Algorithm steps match implementation (if code provided)
- [ ] Implementation details sufficient for reproduction

### Logical Flow
- [ ] Motivation precedes mechanism in each subsection
- [ ] Figures/algorithms referenced in text before they appear

---
If evidence weakens a module, revise its prominence before finalizing.
```

## LaTeX Templates

### Mathematical Notation

Establish notation at the beginning and follow the shared convention exactly:

- Vectors: `\mathbf{x}`, `\boldsymbol{\mu}_c`
- Matrices: `\mathbf{\Sigma}_c`, `\mathbf{I}_d`
- Scalars: `c`, `d`, `\lambda_i`
- Sets: `\mathcal{C}_t`, `\mathcal{H}_t`
- Single important formulas: `equation`
- Multi-step derivations: `align`
- Important symbols: define near first use or in a `\textbf{Notation.}` paragraph
- Operators: keep forms such as `\mathcal{L}`, `\mathbb{E}`, `\arg\max`, `\mathrm{diag}`, `\mathrm{softmax}`, and `\|\cdot\|` consistent

```latex
\textbf{Notation.}
We use bold lowercase letters such as $\mathbf{x}$ for vectors and bold uppercase letters such as $\mathbf{\Sigma}_c$ for matrices.
Greek vector symbols use $\boldsymbol{\mu}_c$.
Scalars such as $c$, $d$, and $\lambda_i$ are italic and not bold.
Sets such as $\mathcal{C}_t$ and $\mathcal{H}_t$ use calligraphic symbols.
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
\begin{align}
\mathcal{L}_{\text{main}} &= [formulation step 1] \\
&= [formulation step 2]
\end{align}
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

After generating or revising the Method section:

- [ ] Every current Introduction contribution has corresponding Method coverage
- [ ] Each subsection has clear motivation paragraph
- [ ] Mathematical notation is consistent and defined
- [ ] Vectors, matrices, scalars, and sets follow the shared notation convention
- [ ] Equation vs. align usage is appropriate and consistent
- [ ] Important symbols are defined near first use or in a notation paragraph
- [ ] Algorithms match code (if code provided) or description
- [ ] Implementation details are sufficient for reproduction
- [ ] Figures are referenced and have descriptive captions
- [ ] Technical terms match Introduction
- [ ] Module emphasis matches experimental support
- [ ] Weak or negative components are downgraded to the right narrative level
