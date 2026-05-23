# Experiments Writing Guide

## Goal

Generate structured Experiments section following a **hybrid workflow**: design → reconstruction → data population. Support baseline paper reconstruction for main experiments, custom ablation design, and automated data filling.

## Input Types

User provides one or more of the following:

1. **Experiment Draft**: Outline of experiments to conduct and evaluation metrics
2. **Baseline Papers**: Reference papers whose main experiments need to be reconstructed for our method
3. **Experiment Data**: CSV, Excel, JSON, or Markdown tables with actual results
4. **Code/Logs**: Optional links to code repositories or experiment logs

## Workflow

### Phase 1: Experiment Design (Mode B)

**Input**: Experiment draft + (optional) baseline papers

**Step 1: Parse Experiment Draft**

Extract structured information:

```
## Experiment Design Parsing

### Datasets
- **Dataset 1**: [Name], [Size], [Characteristics]
- **Dataset 2**: ...

### Evaluation Metrics
- **Metric 1**: [Name], [Definition/Purpose]
- **Metric 2**: ...

### Planned Experiments
1. **Main Comparison**: Compare with baselines on standard benchmarks
2. **Ablation Study**: Validate design choices
3. **Additional Analysis**: [Efficiency/Generalization/etc.]

### Baseline Methods
- **Strong baselines**: [List from draft or baseline papers]
- **Direct competitors**: [Most relevant methods]

---
Please confirm the experiment design.
```

**Step 2: Reconstruct Main Experiments (if baseline papers provided)**

When user provides baseline papers for reconstruction:

```
## Baseline Paper Analysis

For each baseline paper:
- **Paper**: [Title]
- **Main experiments**: [List their key experiments]
- **Evaluation protocol**: [How they evaluate]
- **Key tables/figures**: [Their main results presentation]

## Reconstruction Strategy

Based on baseline papers, propose reconstruction:

### Main Comparison Table Structure
| Method | Metric 1 | Metric 2 | ... |
|--------|----------|----------|-----|
| Baseline A | - | - | ... |
| Baseline B | - | - | ... |
| Ours | [To fill] | [To fill] | ... |

### Experiment Organization
1. **Comparison on Dataset X** (from Baseline Paper Y)
2. **Comparison on Dataset Y** (from Baseline Paper Z)
3. **Cross-dataset generalization** (custom addition)

### Adaptations for Our Method
- Baseline paper uses metric M, we add metric N because...
- Baseline paper evaluates on dataset D, we also include dataset E because...
- Presentation changes: [specific adaptations]

---
Confirm reconstruction strategy or propose modifications.
```

**Step 3: Design Ablation Study**

Based on the proposed method (from Introduction/Method sections):

```
## Ablation Study Design

### Key Components to Validate
1. **Component A**: [Description], [Expected impact]
2. **Component B**: [Description], [Expected impact]
3. **Design Choice C**: [Description], [Rationale]

### Proposed Ablation Table Structure
| Variant | Metric 1 | Metric 2 | Description |
|---------|----------|----------|-------------|
| Full model | - | - | Complete method |
| w/o Component A | - | - | Remove A |
| w/o Component B | - | - | Remove B |
| Alternative C' | - | - | Replace C with C' |

### Analysis Narrative Plan
- Expected finding for Component A: ...
- Expected finding for Component B: ...

---
Confirm ablation design or propose modifications.
```

**Step 4: Generate Experiment Section Skeleton**

Output complete structure with placeholders:

```latex
\section{Experiments}
\label{sec:exp}

\subsection{Experimental Setup}
\label{sec:setup}

\textbf{Datasets.}
[Dataset descriptions...]

\textbf{Evaluation Metrics.}
[Metric definitions...]

\textbf{Baselines.}
[Baseline descriptions...]

\textbf{Implementation Details.}
[Placeholder for hyperparameters, training details]

\subsection{Main Results}
\label{sec:main}

\textbf{Comparison on [Dataset].}
[Table placeholder with headers only]
[Paragraph placeholder describing expected findings]

\textbf{Comparison on [Dataset 2].}
...

\subsection{Ablation Study}
\label{sec:ablation}

[Table placeholder for ablation results]
[Paragraph placeholder for analysis]

\subsection{Additional Analysis}
\label{sec:analysis}

[Optional: Efficiency analysis, qualitative results, etc.]
```

### Phase 2: Data Population (Mode A)

**Input**: Experiment data (CSV, Excel, JSON, Markdown tables)

**Step 1: Parse and Validate Data**

```
## Data Import Summary

### File: [filename]
- **Format**: [CSV/Excel/JSON/Markdown]
- **Content**: [What experiment results]
- **Shape**: [Rows x Columns]
- **Columns**: [List column names]

### Data Mapping
- Column "X" → Table "Y" Row "Z"
- Metric "A" in data → Metric "B" in paper

### Missing Data Check
- [ ] Main results table: Complete
- [ ] Ablation table: Missing [specific cells]
- [ ] Analysis section: Data available

---
Please confirm data mapping or report discrepancies.
```

**Step 2: Generate LaTeX Tables**

Convert data to publication-ready LaTeX tables:

```latex
% Example: Main comparison table
\begin{table}[t]
\centering
\caption{Comparison with state-of-the-art methods on [Dataset].}
\label{tab:main}
\resizebox{\linewidth}{!}{
\begin{tabular}{lccc}
\toprule
Method & Metric 1 & Metric 2 & Metric 3 \\
\midrule
Baseline A & 0.85 & 0.72 & 0.91 \\
Baseline B & 0.87 & 0.75 & 0.93 \\
Ours & \textbf{0.92} & \textbf{0.81} & \textbf{0.95} \\
\bottomrule
\end{tabular}
}
\end{table}
```

**Step 3: Write Result Narrative**

Generate paragraph descriptions based on actual numbers:

```
## Results Narrative

### Main Results

\textbf{Performance on [Dataset].}
As shown in Table~\ref{tab:main}, our method achieves [X%] on [Metric], outperforming the best baseline [Baseline Name] by [Y%].
This improvement demonstrates [what this means technically].
Notably, our method shows particularly strong performance on [specific metric/scenario], suggesting [interpretation].

### Ablation Analysis

\textbf{Effect of Component A.}
Removing Component A leads to a [X%] drop in [Metric] (Table~\ref{tab:ablation}), confirming that [explanation of importance].

\textbf{Effect of Component B.}
...
```

### Phase 3: Supplementary & Refinement (Mode C)

**Input**: Additional experiments, reviewer feedback, or new data

**Handling Supplementary Experiments**:

```
## Supplementary Experiment Integration

New experiments to add:
1. **Additional baseline**: [Method] on [Dataset]
   - Insert into Table~\ref{tab:main}
   - Update narrative paragraph

2. **Additional metric**: [Metric] analysis
   - Add column to table
   - Add discussion paragraph

3. **New analysis section**: [Efficiency/Generalization/etc.]
   - Create new subsection
   - Add table/figure + narrative
```

## LaTeX Output Standards

### Table Guidelines

1. **Main Comparison Table**
```latex
\begin{table}[t]
\centering
\caption{Comparison with state-of-the-art on [Benchmark].}
\label{tab:main_[dataset]}
\resizebox{\linewidth}{!}{% If table is wide
\begin{tabular}{lcccc}
\toprule
Method & Metric$_1$($\uparrow$) & Metric$_2$($\downarrow$) & Metric$_3$($\uparrow$) & Params \\
\midrule
Baseline 1 & 0.85 & 0.12 & 0.91 & 10M \\
Baseline 2 & 0.87 & 0.10 & 0.93 & 15M \\
\midrule
Ours & \textbf{0.92} & \textbf{0.07} & \textbf{0.95} & 12M \\
\bottomrule
\end{tabular}
}
\end{table}
```

2. **Ablation Table**
```latex
\begin{table}[t]
\centering
\caption{Ablation study on key components.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Configuration & Metric$_1$ & Metric$_2$ & $\Delta$ \\
\midrule
Full model & 0.92 & 0.81 & - \\
\quad w/o Component A & 0.88 & 0.76 & -4\%/-5\% \\
\quad w/o Component B & 0.89 & 0.78 & -3\%/-3\% \\
\quad Alternative Design & 0.90 & 0.79 & -2\%/-2\% \\
\bottomrule
\end{tabular}
\end{table}
```

### Figure Guidelines

For qualitative results:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/qualitative.pdf}
\caption{Qualitative comparison on [Dataset]. Our method produces [better quality description] compared to baselines.}
\label{fig:qualitative}
\end{figure}
```

### Section Structure Template

```latex
\section{Experiments}
\label{sec:exp}

\subsection{Experimental Setup}
\label{sec:setup}

\textbf{Datasets.}
\textbf{Evaluation Metrics.}
\textbf{Baselines.}
\textbf{Implementation Details.}

\subsection{Main Results}
\label{sec:main}

\subsection{Ablation Study}
\label{sec:ablation}

\subsection{Additional Analysis}
\label{sec:analysis}
% Efficiency, scalability, generalization, etc.
```

## Do and Don't

### Do
1. ✅ Match baseline paper's evaluation protocol when reconstructing
2. ✅ Adapt table structure to highlight our method's strengths
3. ✅ Use consistent significant figures (usually 2-3 decimal places)
4. ✅ Bold best results, underline second best (standard convention)
5. ✅ Include parameter counts and runtime when relevant
6. ✅ Write narrative that interprets numbers, not just states them

### Don't
1. ❌ Don't copy baseline paper's exact wording when reconstructing
2. ❌ Don't present numbers without interpretation
3. ❌ Don't mix different evaluation protocols without clear explanation
4. ❌ Don't omit key implementation details needed for reproducibility
5. ❌ Don't overclaim based on small margins (e.g., 0.1% improvement)

## Self-Checklist

After generating Experiments section:

- [ ] All numbers from data files are correctly populated
- [ ] Table formatting follows conference/journal style
- [ ] Every table is referenced in the narrative text
- [ ] Baseline descriptions are accurate and rewritten in our words
- [ ] Ablation study validates all key design choices
- [ ] Implementation details are sufficient for reproducibility
- [ ] Claims in text match the numbers in tables
- [ ] Metric directions (↑/↓) are clearly indicated
