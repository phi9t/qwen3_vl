# Plan

Produce a phased implementation plan grounded in the **Principles** section of the `poteto-mode` skill. The plan is the deliverable. Do not implement.

Open a todolist with one item per step below.

## 0. Triage

Skip the plan when the change is one or two files with an obvious approach. Say so and stop.

Plan when the change spans three or more files, introduces architecture, has competing approaches, has unclear scope, or the user asked for one.

## 1. Re-read principles

Read the **Principles** section of the `poteto-mode` skill end to end, and the leaf `principle-*` skills it indexes. The principles govern every plan decision; cross-link them.

## 2. Scope and constraints

State your read of scope and constraints in one paragraph. If intent is genuinely ambiguous, ask one focused clarification.

Resolve what is in scope vs explicitly out, technical or platform constraints, patterns to preserve, and the definition of done.

## 3. Explore in subagents

Delegate codebase exploration (the **guard-the-context-window** principle skill).

Use available helper delegation when useful.
If subagent tooling is available, discover it with `tool_search` first and dispatch by role:
`how`, `architect`, `interrogate`, and `why` are the default collaborators.

Each explorer returns file pointers, conventions, dependencies, test infrastructure, and entry points. No inlined dumps.

## 4. Write the plan

The user specifies where the plan lives.

Use a single file `NN-slug.md` for small plans. For plans with three or more phases, use a directory with `overview.md` plus phase files:

```
NN-slug/
├── overview.md
├── phase-1-scaffold.md
├── phase-2-...md
└── testing.md
```

### Phase sizing

- One function or type plus tests, or one bug fix. Not "one file"; file sizes vary too much.
- Two to three files touched, max.
- Prefer eight to ten small phases over three to four large ones to preserve option value (the **foundational-thinking** principle skill).
- Split if a phase has more than five test cases or three functions.

### Overview file

- **Context.** Problem and why now.
- **Scope.** Included; explicitly excluded.
- **Constraints.** Technical, platform, dependency, pattern.
- **Alternatives.** Two or three approaches sketched, choice and rationale (the **exhaust-the-design-space** principle skill). Skip when constraints dictate one.
- **Applicable skills.** Domain skills the implementer should invoke, by name.
- **Phases.** Ordered standard-markdown links to phase files.
- **Verification.** Project-level commands.
- **Implementation guidance.** Per section 6.

### Phase files

- Back-link to overview.
- **Goal.** What the phase accomplishes.
- **Changes.** Files affected and the change at a high level. What and why, not how. No code snippets.
- **Data structures.** Name the key types or schemas. One-line sketch only (the **foundational-thinking** principle skill).
- **Verification.** Per section 6.

Order phases so infrastructure and shared types land first (the **foundational-thinking** principle skill). Each phase should be independently shippable.

For changes touching existing code, apply the **redesign-from-first-principles** principle skill: if we'd built this with the new requirement on day one, what would it look like? Redesign holistically; deliver incrementally.

If a phase creates or edits a skill, the phase instructs the implementer to use the **skill-creator** and **writing-skills** skills (Codex skill authoring guidance).

## 5. Verification per phase

Each phase needs both:

**Static.** Type check, lint, project tests pass.

**Runtime.** Exercise the feature on the matching surface with matching surface tooling:

- Browser / Electron / Web UIs: use the available UI or browser harness.
- CLIs and TUIs: use the relevant CLI or runtime harness.
- Native mobile: use the available simulator tool.
- If no matching surface tooling exists, flag it in the plan and proceed with static checks.

For bug fixes, the loop is reproduce on the surface, fix, verify on the same surface. Unit tests show a branch behaves a certain way. They do not prove the bug is gone (the **prove-it-works** principle skill).

If a touched surface has no matching runtime tooling, flag it in the plan.

## 6. Implementation guidance

In the overview, name which poteto-mode non-negotiables the implementer must apply, by name:

- the **how** skill over each unfamiliar subsystem before changing it.
- the **interrogate** skill for adversarial review on contested designs before shipping.
- a diff/prose cleanup pass over each diff before commit. the **unslop** skill over any prose surface.
- the **show-me-your-work** skill to keep a decision trail when the plan is large enough to need an auditable record.
- evidence-backed validation for risky phases.

## 7. Hand back

Summarize phases, scope boundaries, applicable skills, and verification. Stop. The user decides when implementation starts.
