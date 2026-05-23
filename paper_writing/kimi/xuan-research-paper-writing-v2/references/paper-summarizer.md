# Paper Summarizer Guide

## Goal

Generate structured, concise summaries of academic papers for literature review and Related Work writing. This guide also serves as the default **reference intake** workflow for building a reusable paper knowledge base from new PDFs or arXiv links.

## Invocation Contract

Use this guide when the user explicitly wants to:

- summarize a new PDF or arXiv paper
- add new references into the literature knowledge base
- prepare a batch of papers for Related Work or Introduction updates
- build or refresh the literature map from summarized papers
- retrieve candidate papers from a topic description, title list, or keyword list before ingestion

## Input Types

1. **PDF files**: Direct paper documents
2. **arXiv links**: arXiv abstract or PDF URLs
3. **Paper metadata**: Title, authors, venue, year
4. **User notes**: Brief description of paper content (if full text unavailable)

## Output Format

Standardized structured summary for literature management, root-level knowledge-base storage, and Related Work integration.

## Workflow Overview

```
Input: topic / title list / keyword list / PDF / arXiv URL / User Notes
    ↓
Step 0: Literature Retrieval (optional)
    ├── Topic or keyword query → arXiv MCP / arXiv search / web search
    ├── Title list → normalize and resolve metadata
    └── Output: candidate paper list
    ↓
Step 1: Source Recognition
    ├── PDF → Convert to images (pdf-to-images skill)
    ├── arXiv → Fetch metadata + PDF → images
    └── User Notes → Direct parsing
    ↓
Step 2: Multimodal Content Analysis
    ├── Read page images via ReadMediaFile
    ├── Analyze text, equations, tables visually
    └── Extract structured information
    ↓
Step 3: Structured Summary Generation
    └── Standardized format output
    ↓
Step 4: Usage Tagging
    └── Related Work integration suggestions
```

### Step 0: Literature Retrieval

Use this step only when the user explicitly asks for literature search, paper discovery, or expansion of a candidate paper list.

**Preferred retrieval order**:
1. arXiv MCP if available
2. arXiv search by title, keywords, or topic phrase
3. Web search for broader discovery such as venue pages or Google-Scholar-like results

**Typical retrieval inputs**:
- task description
- technical keywords
- baseline method names
- paper titles
- venue/year constraints

**Retrieval output contract**:
```
## Retrieval Results

### Search Query
- [query 1]
- [query 2]

### Candidate Papers
1. [Title] — [venue/year if known] — [why relevant]
2. ...

### Recommended Ingest List
- [paper-a]: direct competitor
- [paper-b]: foundational
- [paper-c]: novelty-risk candidate
```

### Step 1: Source Recognition and Preprocessing

Identify input type and preprocess for analysis.

**Knowledge Base Target**:
- Root folder: `./paper-summaries/`
- Global index: `./paper-summaries/index.md`
- Literature map: `./paper-summaries/literature-map.md`
- Per-paper folder: `./paper-summaries/papers/<paper-slug>/`
- Images: `./paper-summaries/papers/<paper-slug>/images/`
- Summary: `./paper-summaries/papers/<paper-slug>/summary.md`

Identify input type and preprocess for analysis:

```
## Source Analysis

**Input Type**: [PDF File / arXiv URL / Metadata / User Notes]
**Paper Identity**: [Title, Authors, Venue, Year]
**Availability**: [Full text / Abstract only / Metadata only]
```

**If input is PDF file**:
1. **Convert to images** using `pdf-to-images` skill:
   ```bash
   python skills/pdf-to-images/scripts/pdf_to_images.py paper.pdf ./paper-summaries/papers/<paper-slug>/images 300
   ```
2. **Read key pages** using multimodal analysis:
   - Page 1: Title, authors, abstract
   - Pages 2-3: Introduction (problem, motivation)
   - Method section pages: Technical approach
   - Experiment pages: Results and evaluation
3. **Analyze visually** for equations, tables, figures

**If input is arXiv URL**:
1. Fetch abstract and metadata from arXiv API
2. If full PDF needed, download and convert to images

**If input is paper titles or a paper list**:
1. Resolve titles against arXiv search or web search
2. Collect metadata and candidate PDF links where available
3. Ask for prioritization only when the candidate list is too broad

**Extraction Confidence**: [High / Medium / Low]
- High: PDF converted to images, full visual analysis possible
- Medium: arXiv abstract + introduction images
- Low: Metadata only, needs user input

### Step 2: Multimodal Content Analysis

**For PDF/Image-based analysis**:

Read and analyze converted images page by page:

```
## Page Analysis Plan

**Critical Pages to Analyze**:
1. **Page 1**: Title, authors, abstract, venue
2. **Pages 2-3**: Introduction (problem statement, motivation, contributions)
3. **Method pages**: Approach overview, key algorithms, architecture diagrams
4. **Experiment pages**: Tables, figures, results
5. **Related work page**: Positioning against prior work

**Analysis Strategy**:
- Use ReadMediaFile to view each page image
- Extract text visually (avoiding OCR/formatting issues)
- Pay special attention to:
  - Mathematical equations and symbols
  - Table structures and numbers
  - Figure captions and diagrams
  - Section headings and structure
```

**Content Extraction Checklist**:
- [ ] Title and authors identified
- [ ] Abstract captured in full
- [ ] Problem/task clearly stated
- [ ] Key contributions listed
- [ ] Method approach understood
- [ ] Main results and metrics extracted
- [ ] Limitations noted (if mentioned)

### Step 3: Structured Summary Generation

Generate summary in the following format:

```
## Paper Summary: [Title]

### Basic Information
- **Title**: 
- **Authors**: 
- **Venue**: 
- **Year**: 
- **URL**: [arXiv/conference link if available]
- **Code**: [GitHub link if available in paper]

### Problem & Motivation
- **Task**: [What problem does this paper address?]
- **Key Challenge**: [What technical difficulty do they target?]
- **Motivation**: [Why is this problem important?]

### Core Contribution
1. **[Contribution 1]**: [Specific technical contribution]
2. **[Contribution 2]**: [If applicable]
3. **[Contribution 3]**: [If applicable]

### Method Overview
- **Approach**: [High-level method category: e.g., transformer-based, optimization-based, hybrid]
- **Key Technique**: [Most important technical component]
- **Pipeline**: [Brief step-by-step: Input → [Module A] → [Module B] → Output]

### Technical Details
- **Input/Output**: [What data format in/out]
- **Architecture**: [Network type, key components]
- **Key Innovation**: [What distinguishes this method from prior work?]
- **Design Choices**: [Important hyperparameters or architectural decisions]

### Experiments & Results
- **Datasets**: [Benchmarks used]
- **Metrics**: [Evaluation criteria]
- **Main Results**: [Key performance numbers or comparisons]
- **Baselines**: [Methods they compare against]

### Limitations (Explicit or Inferred)
- **Stated Limitations**: [What authors acknowledge]
- **Inferred Limitations**: [What the method might struggle with]
- **Scope Constraints**: [Specific settings where method applies/doesn't apply]

### Relation to Our Work
- **Relevance**: [High / Medium / Low]
- **Category**: [Direct competitor / Foundational method / Related technique / Citation background]
- **Key Difference**: [How our method differs]
- **Citation Purpose**: [What point will we cite this for?]

### Literature Graph Fields
- **Primary Topics**: [Topic A, Topic B]
- **Closest Papers**: [Most similar papers in the current knowledge base]
- **Builds On**: [Foundational papers or techniques]
- **Competes With**: [Direct competitor papers]
- **Shares Technique With**: [Papers using similar modules or ideas]
- **Novelty Risk Level**: [Low / Medium / High]
- **Graph Notes**: [Why this paper should connect to those nodes]

### Key Citations
- **Cites**: [Important prior work they reference]
- **Cited by**: [If known, follow-up work building on this]

### One-Sentence Summary
[A single sentence capturing the essence of the paper]

### Quotes for Citation
[Direct quotes from the paper that might be useful for citing]
```

### Step 4: Knowledge Base Registration and Usage Tagging

After summarization, register the paper as a reusable knowledge-base entry and suggest how it should be used in Related Work.

```
## Knowledge Base Entry

- **Paper Slug**: [folder-safe short name]
- **Stored Images**: `./paper-summaries/papers/<paper-slug>/images/`
- **Stored Summary**: `./paper-summaries/papers/<paper-slug>/summary.md`
- **Index Update**: append the paper to `./paper-summaries/index.md`
- **Status**: summarized / needs relation-to-our-work completion

## Recommended `index.md` Structure

Use `./paper-summaries/index.md` as a lightweight entry page for all summarized references.

A useful index table should contain:

| Paper Slug | Title | Venue/Year | Primary Topics | Category | Relevance | Novelty Risk | Status |
|---|---|---|---|---|---|---|---|
| [paper-slug] | [Title] | [Venue Year] | [Topic A, Topic B] | [Direct competitor / Foundational / Related] | [High / Medium / Low] | [Low / Medium / High] | [summarized / needs relation-to-our-work] |

Optional index sections:
- **By Topic**: group papers under topic clusters
- **High-Relevance Papers**: quick access to the most important references
- **Novelty Risk Watchlist**: papers most likely to change positioning
- **Pending Ingest**: papers collected but not yet summarized

## Literature Map Update

After enough summaries exist, update `./paper-summaries/literature-map.md` with the following structure:

- **Topic Clusters**: group papers by task, setting, or technical paradigm
- **Direct Competitor Links**: connect papers that solve the same problem under comparable settings
- **Foundational Links**: connect papers to methods they build on
- **Technique Dependency Links**: connect papers sharing modules, objectives, or training strategies
- **Closest-to-Our-Work Links**: identify the papers most likely to affect novelty positioning
- **Novelty Risk Nodes**: highlight papers that narrow or threaten the claimed originality

A useful map entry should record:
- source paper
- target paper
- relation type
- short rationale
- confidence level

## Recommended `literature-map.md` Template

```markdown
# Literature Map

## Topic Clusters
- **[Topic A]**: [paper-a], [paper-b], [paper-c]
- **[Topic B]**: [paper-d], [paper-e]

## Direct Competitor Links
| Source | Target | Rationale | Confidence |
|---|---|---|---|
| [paper-a] | [paper-b] | Both address [same setting/task] | High |

## Foundational Links
| Source | Target | Rationale | Confidence |
|---|---|---|---|
| [paper-c] | [paper-f] | [paper-c] builds on [paper-f]'s objective/module | Medium |

## Technique Dependency Links
| Source | Target | Shared Technique | Confidence |
|---|---|---|---|
| [paper-d] | [paper-e] | prompt tuning / replay / adapter design | High |

## Closest-to-Our-Work
- **High overlap**: [paper-x], [paper-y]
- **Partial overlap**: [paper-z]

## Novelty Risk Watchlist
| Paper | Risk Level | Why It Matters |
|---|---|---|
| [paper-y] | High | overlaps with our claimed contribution on [aspect] |

## Open Questions
- Does [paper-a] already cover our setting under a different name?
- Should [paper-y] move from Related Reference to Direct Competitor?
```

Use paper slugs consistently so the map stays aligned with `index.md` and per-paper summaries.

## Suggested Usage in Related Work

**Primary Topic**: [Which technical topic this belongs to]

**Discussion Level**:
- [ ] Core / Foundational (2-3 sentences)
- [ ] Related Reference (1 sentence)
- [ ] Background Citation (brief mention)

**Key Comparison Points**:
- Their method: [X]
- Our method: [Y]
- Contrast: [How they differ]
- Best graph neighbors: [Which summarized papers this should be discussed with]

**Suggested Narrative**:
"[Author] et al. propose [method] that [key contribution]. 
However, they [limitation relevant to our work]."
```

## PDF to Images Integration

### When to Use Image-based Analysis

**Always use for PDF papers** to avoid:
- OCR errors in mathematical symbols
- Table formatting corruption
- Multi-column layout confusion
- Equation garbling

### Conversion Workflow

```bash
# Step 1: Convert PDF to images (300 DPI recommended for papers)
python skills/pdf-to-images/scripts/pdf_to_images.py paper.pdf ./paper_images 300

# Output: paper_images/page_001.png, page_002.png, etc.
```

### Page Selection Strategy

**For quick summary** (~10 pages):
- Page 1: Title, abstract
- Pages 2-3: Introduction
- First method section page: Approach overview
- First experiment page: Main results table

**For comprehensive summary** (~15-20 pages):
- All pages from Introduction through Conclusion
- Skip references section
- Focus on pages with tables and figures

### Multimodal Analysis Tips

**When viewing page images**:
1. **Text**: Read directly from image (avoid OCR errors)
2. **Equations**: Pay attention to subscripts, superscripts, special symbols
3. **Tables**: Note row/column alignment, exact numbers
4. **Figures**: Understand diagrams, pipeline illustrations
5. **Captions**: Don't skip - often contain key technical details

### Example Analysis Session

```
User: "Summarize this paper" (provides paper.pdf)

Agent:
1. Convert PDF: python skills/pdf-to-images/scripts/pdf_to_images.py paper.pdf ./tmp 300
2. Read page_001.png → Extract: Title, Authors, Abstract
3. Read page_002.png → Extract: Introduction, Problem
4. Read page_003.png → Extract: Contributions
5. Read page_005.png (method section) → Extract: Approach
6. Read page_008.png (experiment) → Extract: Results table
7. Generate structured summary
```

## Summary Templates by Paper Type

### Method Paper (Most Common)

```
### Method: [Name]
- **Paradigm**: [e.g., end-to-end, modular, two-stage]
- **Key Module**: [Most innovative component]
- **Technical Basis**: [e.g., attention mechanism, graph neural networks]
- **Improvement over Baseline**: [What metrics improve and by how much]
```

### Benchmark/Dataset Paper

```
### Contribution
- **New Task/Benchmark**: [What gap does it fill?]
- **Dataset Scale**: [Size, diversity, annotation quality]
- **Evaluation Protocol**: [How should methods be evaluated?]
- **Baseline Results**: [Initial performance numbers]
```

### Survey/Review Paper

```
### Contribution
- **Scope**: [What topic does it cover?]
- **Taxonomy**: [How do they categorize methods?]
- **Key Insights**: [Main takeaways about the field]
- **Trends**: [Future directions identified]
```

### Theoretical Paper

```
### Contribution
- **Theoretical Result**: [Theorem, bound, guarantee]
- **Assumptions**: [Under what conditions does it hold?]
- **Implications**: [What does this mean for practice?]
- **Proof Technique**: [Key proof approach if relevant]
```

## Quality Indicators

When generating summary, assess:

```
## Summary Quality Assessment

**Confidence Scores** (1-5):
- Problem understanding: [X/5]
- Method clarity: [X/5]
- Result comprehension: [X/5]
- Limitation identification: [X/5]

**Warning Flags**:
- [ ] Paper structure unclear (no standard sections)
- [ ] Method description too vague
- [ ] Results not reproducible from description
- [ ] Strong claims with weak evidence
- [ ] Ambiguous terminology

**Recommendations**:
- [ ] Read specific sections for clarification
- [ ] Check supplementary materials
- [ ] Look for code repository
- [ ] Verify key claims against experiments
```

### Step 5: Batch Processing (if multiple papers)

For processing multiple papers (Related Work preparation):

```
## Batch Summary Report

### Summary Statistics
- **Total Papers**: [N]
- **By Venue**: [Venue distribution]
- **By Year**: [Temporal distribution]
- **By Type**: [Method / Benchmark / Survey / Theory]

### Topic Clustering
**Emerging Topics**:
1. [Topic A]: [Papers X, Y, Z]
2. [Topic B]: [Papers M, N]

**Method Families**:
1. [Family 1]: [Common approach among papers]
2. [Family 2]: [Alternative approach]

### Gap Analysis
**What this literature covers**:
- [Point 1]
- [Point 2]

**Potential Gaps** (opportunities for your work):
- [Gap 1]: Not addressed by papers [A, B, C]
- [Gap 2]: Partially addressed but [limitation]
```

## Integration with Related Work

### Export Format for Related Work

Generate ready-to-use entries:

```
## Related Work Entry

**Citation**: \\cite{[bibkey]}

**For Core/Foundational Discussion**:
[Author] et al. [N] propose [method name], a [type] approach that 
[key technical contribution]. Their method [mechanism description], 
achieving [performance claim] on [benchmark]. However, [limitation 
relevant to your work].

**For Brief Mention**:
Similar approaches have been explored in [M, N, O].
```

## Do and Don't

### Do
1. ✅ Extract specific technical contributions, not just generic descriptions
2. ✅ Identify explicit and implicit limitations
3. ✅ Note key implementation details (architecture, hyperparameters)
4. ✅ Record exact performance numbers for quantitative comparison
5. ✅ Flag unclear or ambiguous claims in the paper
6. ✅ Note relationships between papers (which builds on which)

### Don't
1. ❌ Don't copy abstract verbatim
2. ❌ Don't make up details not in the paper
3. ❌ Don't ignore stated limitations
4. ❌ Don't overstate paper's contributions
5. ❌ Don't skip experimental validation section

## Quick Reference Card

For rapid paper assessment:

```
## Quick Scan Checklist (5-minute read)

□ What problem? ___________________
□ Key idea? ______________________
□ Main result? ____________________
□ Limitation? _____________________
□ Relevant to us? _________________
```

## Output Delivery

When delivering summaries:

1. **Single Paper**: Full structured format above
2. **Batch (Related Work prep)**: Summary table + individual full summaries
3. **Quick Reference**: One-sentence summaries only

Format choice based on user request:
- `"Summarize this paper"` → Full format
- `"Quick summary of [N] papers"` → One-sentence + key points only
- `"Prepare for Related Work"` → Full format + usage tagging + batch report
