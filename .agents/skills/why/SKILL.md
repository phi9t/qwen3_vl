---
name: why
description: "Use for 'why does X work this way', 'why we picked Y', design rationale, regressions, postmortems, or threshold choices. Investigate source history, docs, and issue/observability evidence where available, then return calibrated conclusions."
---

# Why

Investigate the motivation and intent behind code. Why was this shape chosen, what edge cases drove it, and what constraints blocked alternative designs.

This is the rationale companion to `how`. `how` explains behavior; `why` explains forcing function.

## How this skill works

Treat explanation as evidence assembly, not speculation. Evidence usually lives in different systems, and the valuable signal can come from any one of them.

1. Gather a concrete code anchor first: files, symbols, and recent history.
2. Discover evidence sources that are actually available in the current environment.
3. Investigate in parallel per source category when possible.
4. Merge findings with explicit confidence tiers.
5. Surface unknowns and null-result sources as first-class gaps.

## Operating posture

- Collect evidence before writing the narrative.
- Prefer explicit citations from git, pull requests, issue tickets, docs, and connector-backed evidence.
- Show uncertainty clearly with hedged language.
- Never use code mechanics as proof of intent.
- Treat missing sources as evidence for process gaps.

## Core epistemics

- **Direct**: explicit evidence from a source (PR, issue, doc, commit message, chat record, dashboard entry).
- **Supported**: multiple direct sources agree.
- **Inferred**: plausible but indirect, with explicit reasoning chain.
- **Speculative**: weak or missing evidence; label clearly.
- **Unknown**: no evidence found.

Use these tiers in the final output and keep the confidence language intact.

## Step 1 — Define target and question

Identify:

- target file(s) and symbol(s) (or config/decision surface)
- user question wording and any assumptions to test
- likely impact areas (feature, failure mode, migration, performance behavior)

If the target is vague, state your interpretation up front and proceed with minimal assumptions.

## Step 2 — Build evidence anchor

Run local evidence commands first:

```bash
git blame -L <start>,<end> <file>
git log --follow -p -- <file>
git log --oneline -20 -- <file>
git log -1 --format=%B <commit>
```

For merge-level context:

```bash
gh pr view <number> --json title,body,author,createdAt,mergedAt,labels,closingIssuesReferences,comments,reviews
```

Always include the PR/ticket/date/time references you use.

## Step 3 — Discover available evidence sources

Before assigning investigations, discover what is runnable in this environment:

- Prefer `tool_search` to list available connectors/MCPs.
- If unavailable, inspect configured MCP/catalog surfaces that exist locally.
- Build a per-category coverage map from what is actually present.

Evidence categories remain:

- Source control history (required baseline via git/gh)
- Issue tracker
- Long-form documents (RFC/ADR/PRD/postmortem docs)
- Real-time/team chat archives
- Infrastructure observability
- Error/exception tracking
- Product analytics or warehouse evidence

For each available category, attach the specific tool/MCP and query shape (keywords, ids, dashboards, tables, ticket IDs).

If a category has no source, report it as "unavailable" rather than skipping silently.

## Step 4 — Investigate in parallel

Run investigations concurrently when possible. The skill should create distinct passes for different evidence types so they do not collapse each other’s assumptions.

- If you have helper mechanics that can run multiple passes in parallel, assign one pass per evidence category and execute in parallel.
- If parallel helpers are not available, run categories serially and explicitly note the limitation.

Each pass should return:

- what it searched
- what it found
- what it did not find
- any confidence-relevant constraints

## Step 5 — Synthesize

Synthesize all findings into one grounded explanation:

- The question and target
- direct evidence section with citations
- reasoned inferences with explicit hedging
- competing hypotheses if multiple stories remain viable
- explicit gaps and why they remain gaps
- source log: every category consulted and category result (found / no results / unavailable)

Never claim stronger certainty than the weakest supporting tier.

## Output format

Use this structure:

- **The Question**: concise restatement.
- **The Code in Question**: target files/lines and symbols.
- **What We Found**: direct/supportable evidence with citations.
- **What We Can Reasonably Infer**: hedged inferences with chains of evidence.
- **Competing Hypotheses**: optional when evidence supports multiple stories.
- **What We Don't Know**: explicit missing evidence and absent searches.
- **Sources Consulted**: one line per category with searched terms and result.

Include local repository evidence in the same line format as external sources (git/gh commands + paths examined).

## Failure modes to avoid

- Confident storytelling without evidence
- dropping caveats into "known" section
- replacing "no data" with plausible explanation
- using repository commit/structure alone as motive
- conflating "this is likely" with "this is proven"

## Reference files

- `references/epistemics.md` for confidence language.
- `references/investigator-prompt.md` for investigator framing.
- `references/source-playbook.md` and `references/sources/*.md` for connector query prompts.
- `references/synthesizer-prompt.md` for final narrative assembly.
