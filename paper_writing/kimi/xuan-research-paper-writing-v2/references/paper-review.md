# Paper Review

## Goal

Use an adversarial, reviewer-style quality gate to detect reject risks early, audit claim-evidence alignment, and trigger targeted revisions before submission.

## Invocation Contract

Before reviewing, identify:

1. **Review Mode**: quick screening / full review / claim-evidence audit / cross-section consistency audit / narrative reset review / derivation audit
2. **Evidence State**: partial evidence / strong evidence / conflicting evidence
3. **Review Scope**: one section / multi-section / full paper

## Read From `paper_state`

Prioritize these shared fields before reviewing:

- `contributions[*].claim`, `contributions[*].status`, `contributions[*].evidence`
- `method.key_modules[*].role`, `method.key_modules[*].evidence_status`
- `literature_positioning.closest_work`, `literature_positioning.novelty_boundary`
- `terminology.preferred_terms`, `terminology.banned_or_old_terms`
- `terminology.math_notation.vector`, `terminology.math_notation.matrix`
- `terminology.math_notation.scalar`, `terminology.math_notation.set`
- `terminology.math_notation.equation_env`, `terminology.math_notation.operator_style`
- `terminology.math_notation.notation_rule`
- `derivations.target_formula`, `derivations.assumptions`
- `derivations.verified_steps`, `derivations.questionable_steps`
- `derivations.open_questions`, `derivations.exploration_goals`
- `impacts.affected_sections`, `impacts.next_best_action`

## Core Principle

Assume reviewers will probe every weak point and proactively fix it at the paper-state level rather than only patching isolated sentences.

## Critical Rule (Do Not Violate)

Every major claim, especially in Abstract, Introduction, Method emphasis, and Conclusion, must be:

1. technically correct,
2. consistent with the current literature positioning, and
3. explicitly supported by available evidence.

If a claim is not supported, do one of the following:
- add evidence,
- weaken or remove the claim,
- downgrade the corresponding module or contribution,
- or trigger a broader narrative reset.

## What Usually Gets a Paper Accepted

1. Sufficient contribution (for example: novel task, novel pipeline, novel module, novel design choices, new experimental findings, or new insight).
2. Better empirical performance than prior methods under fair comparisons.
3. Sufficient comparison experiments and ablation studies.

## Common Rejection Dimensions

| Rejection Dimension          | Typical Failure Signals                                                                                                                                                                                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Insufficient contribution | 1.1 Targeted failure cases are too common.<br /> 1.2 Proposed technique is already well explored; expected gains are predictable/well-known.                                                                                                                                               |
| 2. Unclear writing           | 2.1 Missing technical details; work is not reproducible.<br />2.2 A method module lacks clear motivation.                                                                                                                                                                                  |
| 3. Weak empirical effect     | 3.1 Improvement over prior methods is only marginal.<br /> 3.2 Even if better than previous methods, absolute performance is still not strong enough.                                                                                                                                      |
| 4. Incomplete evaluation     | 4.1 Missing ablation studies.<br />4.2 Missing important baselines or important evaluation metrics.<br /> 4.3 Datasets are too simple to prove the method truly works.                                                                                                                    |
| 5. Problematic method design | 5.1 Experimental setting is unrealistic.<br />5.2 Method has technical flaws and appears unreasonable.<br />5.3 Method is not robust and needs per-scenario hyperparameter tuning. <br /> 5.4 New design introduces stronger limitations than its benefits, leading to negative net value. |

## Review Outputs

Every review should produce:

- **Risk Summary**: highest-priority rejection risks
- **Claim-Evidence Map**: supported / weakened / unsupported claims
- **Cross-Section Conflicts**: terminology, contribution, and evidence mismatches
- **Required Actions**: revision / new experiment / claim rollback / narrative reset
- **Derivation Audit**: assumptions, verified steps, questionable steps, and correction suggestions when formulas are involved

## End-of-Paper Self-Review Question List

Add this checklist near the end of the draft while revising.
Use each question to trigger concrete edits before submission.

### 1. Contribution

1. What new knowledge does this paper give to readers?
2. Are we solving a truly meaningful failure case, not a trivial/common one?
3. Is the technical idea genuinely non-obvious beyond well-explored practice?
4. Is our gain surprising or insightful rather than a predictable improvement?
5. Is there at least one clear novelty type (task/pipeline/module/design finding/insight)?

### 2. Writing Clarity

1. Can a knowledgeable reader reproduce the method from the paper?
2. Did we provide enough technical detail for each key module?
3. Is the motivation of every module explicit and logically connected to a challenge?
4. Are terms and notation consistent across sections?
5. Does each paragraph carry one clear message with smooth transitions?

### 3. Experimental Strength

1. Are improvements over strong baselines meaningful, not just statistically tiny?
2. Is absolute performance competitive enough for the target venue?
3. Are gains consistent across multiple datasets/settings/metrics?
4. Do we report both strengths and failure cases honestly?

### 4. Evaluation Completeness

1. Do we include ablations for all key design choices?
2. Are all strong/recent baselines included under fair settings?
3. Are evaluation metrics standard and sufficient for this task?
4. Are datasets/scenarios challenging enough to validate real effectiveness?
5. Are comparison and ablation protocols clearly documented?

### 5. Method Design Soundness

1. Is the experimental setting realistic for practical use?
2. Does the method have hidden technical defects or unreasonable assumptions?
3. Is the method robust without heavy per-case hyperparameter retuning?
4. Do benefits outweigh added complexity and new limitations?
5. Could reviewers reasonably argue that the net benefit is negative?

## Claim-Evidence Audit

Use this audit before finalizing any major revision.

```
## Claim-Evidence Map

### Supported Claims
- [Claim A] ← supported by [table / figure / experiment / citation]

### Weakened Claims
- [Claim B] ← only supported in limited settings

### Unsupported Claims
- [Claim C] ← currently lacks evidence or is contradicted

### Required Actions
- [ ] keep
- [ ] weaken
- [ ] remove
- [ ] add experiment
- [ ] revise linked sections
```

## Derivation Audit

Use this audit when the paper contains nontrivial proofs, loss derivations, or theoretical claims.

```
## Derivation Check

### Target Formula or Claim
- [formula / theorem / proposition]

### Assumptions
- [assumption 1]
- [assumption 2]

### Step-by-Step Verification
1. [step] → verified / heuristic / questionable
2. [step] → verified / heuristic / questionable

### Main Issues
- hidden assumption
- invalid algebraic jump
- undefined symbol
- unjustified approximation

### Suggested Fix
- [corrected derivation / narrower claim / extension candidate]
```

## Cross-Section Consistency Audit

Check the following explicitly:

- [ ] Abstract does not overstate results beyond Experiments
- [ ] Introduction contributions match Method structure and Experiments evidence
- [ ] Method emphasis matches ablation support
- [ ] Related Work positioning matches the current literature landscape
- [ ] Vector, matrix, scalar, and set notation follow the shared convention consistently
- [ ] Equation and align environments are used appropriately
- [ ] Important symbols are defined near first use or in a notation paragraph
- [ ] Operator formatting is consistent across losses, expectations, norms, and functions
- [ ] Nontrivial derivations clearly state assumptions and avoid unjustified algebraic jumps
- [ ] Conclusion does not claim broader support than the paper provides

## Adversarial Writing Workflow

1. Read the paper as a skeptical reviewer.
2. Produce a claim-evidence map before giving stylistic feedback.
3. Mark each issue as `pass`, `needs revision`, `needs new experiment`, or `needs narrative reset`.
4. If multiple sections depend on the same weakened claim, revise them together.
5. If the current storyline is no longer the strongest one, recommend a narrative reset rather than local patching.
6. Repeat until no major rejection risk remains.
