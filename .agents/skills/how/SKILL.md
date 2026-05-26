---
name: how
description: "Use for \"how does X work\", code walkthroughs before changing something, and placement / ownership / layering questions (\"where should this live\", \"which package owns this\", \"is this the right layer\"). Explains subsystem architecture, runtime flow, and onboarding mental models. Can critique architecture. Use why for motivation."
---

# How

Answer "how does X work?" questions so the user gets a correct mental model quickly.

Two modes:

1. **Explain** (default). Build the model and describe it.
2. **Critique.** Explain first, then run independent critiques for architectural risks.

## Explain Mode

### Step 1. Interpret scope and choose method

State your interpretation of scope before reading deeper. If it is unclear, use the narrowest interpretation and continue; adjust if the user steers.

Use `rg` first, then targeted reads:

- `rg` symbols, public types, and entry points for the subsystem.
- `rg` for call-site patterns that connect the pieces.
- Open only files that answer the specific scope.

Then choose:

- **Inline-first path** (default): one direct investigation pass.
- **Parallel slice path**: independent slices only for cross-cutting questions where separate viewpoints reduce risk.

For parallel slice work, discover available multi-agent tooling first (for example via `tool_search`). Use it only when it materially improves independent coverage.

### Step 2a. Parallel slice path (when useful)

Split the question into up to 4 independent slices with minimal overlap, for example:

- Slice 1: entry points and request/control flow
- Slice 2: data and state transitions
- Slice 3: configuration and boundaries
- Slice 4: tests and operational behavior

Each slice should trace:

- Trigger → handlers/callers
- Transformations and side effects
- Return/error paths and invariant assumptions
- Files touched

After all slices return, synthesize overlaps and contradictions into one coherent flow.

### Step 2b. Inline path (simple / low-separation questions)

Track one path end-to-end in one pass:

- identify one primary entry point
- trace to the terminal behavior
- capture key conditions and failure branches
- map where ownership changes or abstractions begin/end

### Step 3. Synthesize

You should be able to answer:

- what component owns the work at each stage
- where data and control move
- why the shape exists (or why it became that way)
- what is guaranteed vs "best effort"

If any link is uncertain, call it out explicitly and note the next file-level check.

### Step 4. Present

Write the explanation in the output format below, with minimal extra commentary.

## Output Format

**Overview.** 1-2 paragraphs. What is this thing, what does it do, and why does it exist. Someone should know whether to keep reading.

**Key Concepts.** Core abstractions only, with short definitions and ownership.

**How It Works.** Trigger → flow → decision points → effects → outputs. Reference paths, files, and behaviors without dumping unnecessary code.

**Where Things Live.** Small map of files/directories needed to work in this area.

**Gotchas.** Non-obvious assumptions, edge cases, and historical/operational traps.

## Critique Mode

### Step 1. Explain first

Run the explain flow above first. You cannot critique correctly before producing a faithful mental model.

### Step 2. Run critiques

After the explain output exists, run independent critique paths:

1. If available multi-agent tooling can materially help (for independent slices), dispatch independent critique passes.
2. Otherwise, run an inline multi-lens critique using the same scope and file set.

Each pass should evaluate architecture for:

- Correctness and invariants
- Integration risk and coupling
- Changeability/maintainability
- Sizing and failure diagnostics

### Step 3. Lead judgment

Prioritize findings with:

- **Act on.** Architectural issues worth fixing now.
- **Consider.** Real concerns with uncertain ROI.
- **Noted.** Valid but low priority.
- **Dismissed.** Invalid or context-missed observations.

Present explanation first, then the verdict. Keep the explanation readable even if the user only wants understanding.
