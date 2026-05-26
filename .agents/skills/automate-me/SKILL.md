---
name: automate-me
description: "Use for \"automate me\", \"create/update a -mode skill\", or translating recurring user behavior into reusable Codex skills."
---

# Automate me

Capture a user’s stable working preferences into a reusable `-mode` skill using local evidence and explicit confirmation.

## Flow

### 0. Determine whether a skill exists

Find candidate paths under `.agents/skills/` and `~/.codex/skills/` matching `*-mode` for the user.

- If found, plan an update-first pass.
- If not, plan a fresh skill.

### 1. Mine local evidence

Prefer repository-native, repeatable evidence before asking the user for new input:

- recent commands and edits in the active working context
- changed files and their review comments
- existing skill files recently updated by this user
- issue/PR patterns, review notes, or repo docs already used repeatedly
- any connector-backed project evidence exposed by available tools

When available, discover supporting connectors via `tool_search` and query only sources that exist in this environment.

### 2. Elicit missing intent directly

If evidence is insufficient, ask the user targeted free-form questions and record explicit answers.

Keep prompts to 1–2 small rounds:

- first: scope + constraints
- second (optional): preferred defaults and exception cases

Avoid broad one-time questionnaires unless absolutely necessary.

### 3. Synthesize candidate rules

Cluster observations into durable categories:

- response shape (brevity, tone, format)
- verification posture (tests/checks/commands considered complete)
- autonomy and parallelization preferences
- writing/prose discipline
- review and documentation workflow

Require repeated signal before encoding a rule as hard guidance.

### 4. Author or revise the skill

Use `skill-creator` and `writing-skills` for substantive edits.

- Place at `.agents/skills/<handle>-mode/SKILL.md` (or user home equivalent if requested).
- Keep frontmatter valid and minimal.
- Frontmatter `disable-model-invocation` should stay true by default.
- Keep scope explicit and narrow; skip unused sections.

### 5. Polish and validate prose

Run `unslop` and `writing-skills` guidance on generated text.

### 6. Deliver

- Show draft to the user and incorporate feedback.
- for lightweight, high-confidence updates, apply directly if user approves.
- for substantial skill changes, route through the normal drafting loop and iterate.

## Evaluation

There is no generic pass/fail metric; validate by:

- does the new skill match the user’s repeated behavior
- can the user apply it with fewer clarifications
- did we avoid overfitting one-off details

## Guardrails

- Do not mirror one-off preferences from a single session.
- Do not add speculative assumptions or private workspace details.
- Do not overfit to transient implementation facts.
- Keep sections minimal; if a theme does not repeat, do not force codification.

## Reference files

- `poteto-mode` skill as style benchmark
- `unslop` for prose discipline
- `skill-creator` and `writing-skills` for authoring
