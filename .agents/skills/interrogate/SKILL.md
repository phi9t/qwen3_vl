---
name: interrogate
description: "Use for \"interrogate\", \"adversarial review\", \"challenge this\", \"stress test this code\", or \"find blind spots\". Run independent review lenses and synthesize high-signal findings."
disable-model-invocation: true
---

# Interrogate

Run an adversarial review flow on the target work. The goal is insight, not change speed.

Do not auto-apply findings.

## Step 1, Gather Scope

Collect exactly what should be reviewed:

- Files named by the user
- Diff or working tree scope (`git diff` equivalent)
- Related context files needed to validate assumptions
- Runtime expectations and constraints from the request

## Step 2, State Intent

Write one clear paragraph stating:

- what the code is trying to achieve
- what constraints matter for this review
- what would count as failure

If intent is unclear, state the ambiguity and ask for confirmation before deep review.

## Step 3, Run Review Paths

Use independent perspectives when it materially improves signal:

1. Discover available multi-agent tooling first (for example via `tool_search`).
2. If available and useful, dispatch independent reviewers.
3. If unavailable, run all lenses locally.

### Option A: External independent reviews

Give each reviewer:

- the intent paragraph
- scope package
- source material (files / diff)
- review rubric from `references/rubric.md`

Request each reviewer focus on one lens:

- **Correctness**
- **Integration risk**
- **Maintainability**
- **Test coverage**

### Option B: Local lenses

Run each lens inline and independently:

- **Correctness:** correctness claims, edge cases, consistency, error handling
- **Integration risk:** interface mismatches, migration hazards, behavior coupling
- **Maintainability:** complexity growth, duplication, naming ownership, abstraction leakage
- **Test coverage:** gaps in confidence, brittle tests, untested failure modes

## Step 4, Synthesize

Merge findings into a single structured verdict:

- group repeated findings across reviewers/lenses
- keep model or reviewer identifiers with each item
- note disagreements explicitly
- mark low-confidence claims clearly

## Step 5, Lead Judgment

Output each finding in one of:

- **Act On**
- **Consider**
- **Noted**
- **Dismissed**

Include where each came from (reviewer/lens) and why.

## Output Format

### Intent
> [The stated intent paragraph from Step 2]

### Reviewers
- Independent Review A: [lens/model/tool], [N findings]
- Independent Review B: [lens/model/tool], [N findings]
- Independent Review C: [lens/model/tool], [N findings]
- Independent Review D: [lens/model/tool], [N findings]

### Act On
[Findings that should be addressed. For each: description, source(s), rationale.]

### Consider
[Findings worth thinking about. For each: description, source(s), tradeoff.]

### Noted
[Technically valid but low-priority observations.]

### Dismissed
[Findings rejected with brief rationale.]

### Agreement Map
[Where multiple reviewers/lenses align and where they diverge.]
