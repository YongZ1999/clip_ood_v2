# Related Work Writing Guide

## Goal

Draft, revise, restructure, or repair the Related Work section based on the **current literature state**. This guide should support initial topic organization, new-paper insertion, novelty repositioning, and cross-section updates when newly added references change how the paper should be framed.

## Invocation Contract

Before writing, identify:

1. **Operation Type**: Draft / Polish / Restructure / Evidence Update / Claim Rollback / Consistency Repair
2. **Literature State**: the literature-facing slice of `paper_state`, including core papers only / summarized candidate pool / expanded pool with new references / conflicting novelty signals
3. **Impact Scope**: local paragraph edit / full Related Work rewrite / cross-section repair

## Read From `paper_state`

Prioritize these shared fields before generating or revising text:

- `literature_positioning.closest_work`
- `literature_positioning.strongest_baselines`
- `literature_positioning.novelty_boundary`
- `knowledge_base.index_file`, `knowledge_base.map_file`
- `knowledge_base.search_queries`, `knowledge_base.candidate_papers`
- `contributions[*].claim`, `contributions[*].status`
- `terminology.preferred_terms`, `terminology.banned_or_old_terms`
- `impacts.affected_sections`

## Prerequisites

⚠️ **Important**: Before starting this section, candidate papers must have been summarized.

If the user provides **raw PDFs or arXiv links** during a Related Work revision, do **not** force summarization unless the user explicitly wants to summarize, ingest, or organize those papers into the literature knowledge base.

If the user asks to **search for more related papers** from a topic, method, baseline, or title list, trigger literature retrieval first, then decide which papers should enter the ingest queue.

If candidate papers have not been summarized yet and the user explicitly wants literature intake, please first call the `paper-summarizer` skill:
- Input: Candidate paper list/folder (PDF or arXiv links)
- Process: 
  1. PDF papers → Convert to images via `pdf-to-images` skill
  2. Analyze images via multimodal understanding (ReadMediaFile)
  3. Extract structured information visually
- Output: Structured summary for each paper (contribution, method, technical characteristics, limitations)

## Workflow

### Step 0: Route the Request

Choose the workflow before editing text:

- **Draft**: organize topics from core papers and summarized candidates
- **Polish**: improve wording, compression, and transitions without changing topic structure
- **Restructure**: redesign topic grouping when the current organization no longer highlights the right novelty gap
- **Literature Retrieval**: expand or resolve a paper list before summarization or topic grouping
- **Evidence Update**: insert new references, use existing knowledge-base summaries when available, and summarize new papers only when the user explicitly requests literature intake
- **Claim Rollback**: weaken novelty wording when newly added papers reduce the originality of a claimed contribution
- **Consistency Repair**: align Related Work with Introduction, Method, and contribution framing

### Step 1: Receive Input

Confirm user has provided:
1. **Core papers**: 3-5 papers (most critical papers, usually direct competitors or most relevant methods)
2. **Candidate paper summaries**: Structured summaries for all candidate papers
3. **(Optional) Technical topic template**: User-predefined topic structure

If item 2 is missing but summarized entries already exist in `./paper-summaries/papers/`, use `./paper-summaries/index.md` as the first-pass navigator, then drill into per-paper summaries and `./paper-summaries/literature-map.md` as needed.

If item 2 is missing and the user explicitly asks to ingest new references, summarize them into `./paper-summaries/papers/<paper-slug>/summary.md` first, then continue.

If the user only provides a topic description, method name, baseline list, or paper titles, retrieve candidate papers first, store them in `knowledge_base.candidate_papers`, and then decide which papers deserve summarization.

### Step 2: Technical Topic Identification

**If user provided technical topic template**:
- Use user-defined topic structure directly
- Confirm whether topic coverage is reasonable

**If user did not provide technical topics**:

1. **Analyze core papers**
   - Extract technical contribution of each core paper
   - Identify their research directions/technical paradigms
   - Find technical connections and differences between them

2. **Identify technical lineage**
   - What common problem do these core papers address?
   - What different technical approaches do they use?
   - What foundational methods/prior work exist?

3. **Propose technical topics (2-4)**

   Use the literature map first if it already exists:
   - reuse existing topic clusters when they are still reasonable
   - inspect direct competitor and foundational links before inventing a new grouping
   - check novelty risk nodes before overstating distinction from prior work

   Typical topic types include:
   - **Topic 1**: Mainstream/task-specific methods (Core Literature 1-2)
   - **Topic 2**: Methods closest to your core idea (Core Literature 2-3)
   - **Topic 3**: Auxiliary techniques your method builds on (if applicable)
   - **Topic 4**: Other related technical approaches (if needed)

4. **Output topic structure for confirmation**

   ```
   ## Proposed Technical Topic Structure

   1. [Topic A Name]: Mainstream methods (based on Core Paper X, Y)
      - Papers covered: ...
      - Key limitation: ...
   
   2. [Topic B Name]: Most relevant methods (based on Core Paper Z)
      - Papers covered: ...
      - Key limitation: ...
   
   3. [Topic C Name]: Foundational techniques (auxiliary methods)
      - Papers covered: ...
      
   Please confirm or adjust the above topics. You can:
   - Confirm (reply "confirm")
   - Adjust (point out what needs modification)
   - Fully customize (provide your own topic structure)
   ```

### Step 3: Paper Grouping and Importance Classification

After topic confirmation, proceed with paper grouping:

**Grouping principles**:
- Assign each candidate paper to the most appropriate technical topic
- A paper may belong to multiple topics (choose the most relevant one)

**Importance classification criteria**:

| Level | Criteria | Detail Level | Example |
|-------|----------|--------------|---------|
| **Core/Foundational** | Foundation of your method or direct comparison | Medium detail (2-3 sentences) | Your method builds on their technique; or direct competitor |
| **Related Reference** | Relevant but not core, provides technical context | Brief mention (1 sentence) | Other work in same research direction |

**Output paper grouping table**:

```
## Paper Grouping and Importance Levels

### Topic A: [Name]

| Paper | Importance Level | Assignment Reason |
|-------|------------------|---------------------|
| Paper X | Core | Our method builds on their XX technique |
| Paper Y | Related | Same XX direction, uses different approach |
| Paper Z | Related | Provides XX technical background |

### Topic B: [Name]

...

Please confirm paper grouping and importance levels. Provide feedback if adjustments are needed.
```

### Step 4: Generate Related Work Draft

Based on confirmed topic structure and paper grouping, generate the formal draft.

#### Paragraph Structure (one paragraph per topic)

1. **Topic sentence**: Define the scope and core problem of this topic
2. **Representative methods overview**:
   - Core papers: Describe method + technical characteristics in detail
   - Related papers: Briefly list
3. **Key limitations**: Point out limitations of existing methods (related to your technical challenge)
4. **Transition sentence**: Lead to how your work fills this gap

#### Detail Level Control

**Core/Foundational methods (2-3 sentences)**:
```
XXX et al. [N] propose a method that ... [method description]. 
They achieve ... [technical characteristics].
However, they ... [key limitation, related to your challenge].
```

**Related references (1 sentence)**:
```
Other works [M, N, O] explore ... [brief description of common direction].
```
or as enumeration:
```
Similar ideas have been explored in [M, N, O].
```

#### Output Format

```latex
\section{Related Work}
\label{sec:related}

\textbf{[Topic A Name].}
[Paragraph content...]

\textbf{[Topic B Name].}
[Paragraph content...]

\textbf{Relation to Our Work.}
[Optional: Brief summary of how your work relates to each category]
```

## Do and Don't

### Do
1. ✅ Organize by technical topics, not by year
2. ✅ Clearly specify technical comparison with your work for each topic
3. ✅ Describe technical mechanisms of core methods specifically, not just enumeration
4. ✅ Connect limitation descriptions to your technical challenges
5. ✅ Maintain terminology consistency (unified with Introduction/Method)

### Don't
1. ❌ Don't write as citation dump
2. ❌ Don't hide strongest baselines (core competing methods should be discussed in detail)
3. ❌ Don't vaguely say "poor performance", point out specific technical limitations
4. ❌ Don't introduce new terms in Related Work (should be already defined)

## Literature Update and Novelty Repair

When adding new references, explicitly review their impact:

```
## Novelty Impact Review

### New References That Strengthen Positioning
- [Paper A] clarifies the limitation our method addresses

### New References That Narrow Novelty
- [Paper B] already explores a similar design or observation

### Required Actions
1. Keep, narrow, or remove each affected novelty statement
2. Update topic grouping if the old structure hides the strongest competitors
3. Trigger cross-section revisions if Introduction or Method still overstates originality
```

## Self-Checklist

After generating or revising the draft, check the following:

- [ ] All core papers are covered and discussed adequately
- [ ] Each topic has clear scope definition
- [ ] Each core method has specific technical description
- [ ] Each topic's limitations relate to your technical challenges
- [ ] Technical distinction of your work is clearly stated without overstating novelty
- [ ] Citation coverage is complete, no key literature missed
- [ ] Terminology is consistent with Introduction/Method
- [ ] Newly added papers have been checked for cross-section impact
