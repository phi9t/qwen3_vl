# Codex-Native Pstack Skills Design

## Purpose

Adapt the imported pstack skills under `.agents/skills/` so they are executable by Codex in this repository. This is a native port, not a compatibility shim. Skills should describe workflows that work with the tools and policies available in this environment, while preserving the useful judgment patterns from pstack.

The imported skill set contains two broad categories:

- Principle and style skills that mostly work as written.
- Workflow skills that still assume Cursor-specific agents, model names, transcript paths, MCP behavior, PR automation, and structured question tools.

The rewrite targets the second category first.

## Scope

Rewrite these workflow-heavy skills and their references:

- `how`
- `why`
- `interrogate`
- `reflect`
- `automate-me`
- `show-me-your-work`
- `poteto-mode`

Light-touch cleanup is allowed for supporting files in these skill directories when they contain the same platform assumptions. Principle skills, `tdd`, `typescript-best-practices`, `unslop`, `arena`, `architect`, and `figure-it-out` should only be changed where they reference the broken workflow mechanics directly.

The existing `python-style-touchup` skill stays repo-local and should only receive path wording fixes needed for `.agents/skills`.

## Codex-Native Rules

### Multi-Agent Work

Skills must not mention Cursor `Task` calls, `subagent_type`, `readonly`, or hardcoded model slugs as required mechanics. When parallel agents are useful, the skill should say:

1. Use `tool_search` to discover Multi-agent tools when available.
2. Dispatch separate agents only when the task has independent slices.
3. If no suitable multi-agent tool is available, do the work inline and keep the context small with `rg`, targeted file reads, and concise synthesis.

The skill can still recommend parallelism as a pattern, but it must have a valid inline fallback.

### User Questions

Skills must not reference `AskQuestion`. In Default mode, they should ask one concise plain-text question only when local context cannot answer safely. They may mention `request_user_input` only conditionally: use it when the tool is available and the session is in a mode that exposes it.

### Transcripts and Memory

Skills must not require an `agent-transcripts/` path. For reflection or historical analysis, use this evidence order:

1. Current conversation context.
2. Repo files and git history.
3. Codex memory, when relevant and available.
4. Any explicit transcript or artifact path provided by the user.

If transcript evidence is unavailable, the skill should say so and proceed from available evidence instead of failing.

### MCPs and Connectors

Skills must not assume a named MCP exists. They should use `tool_search` for connector or multi-agent tool discovery when the task implies external context. If no relevant connector is available, report the gap. External writes should follow existing Codex tool permissions and user intent; no skill may broadly authorize external actions just because they are "reversible."

### PR and Control Workflows

Skills must not require PR babysitting, UI control tools, or cleanup tools that are not installed. They may recommend:

- `gh` only when a GitHub remote and credentials are available.
- Browser or visual-control verification only when the relevant tool exists.
- Diff/prose cleanup as a manual review pass, not a required unavailable `deslop` command.

### Safety and Permissions

The port must not weaken Codex's repository and tool safety rules. Skills should preserve:

- No destructive git operations without explicit user request.
- No reverting unrelated user changes.
- Verification before completion claims.
- Clear reporting when a requested tool or evidence source is unavailable.

## Skill-Specific Design

### `how`

Make `how` an inline-first codebase explainer. For simple questions, inspect the relevant files directly. For complex subsystem questions, optionally use available multi-agent tooling after discovery. Output should remain a senior-engineer explanation with overview, key concepts, flow, locations, and gotchas.

### `interrogate`

Make `interrogate` a Codex-native adversarial review workflow. It should gather the intent, diff, and context, then either dispatch independent reviewers through available multi-agent tools or run a structured local review pass with multiple lenses. Remove fixed model tables.

### `why`

Keep the evidence-category model, but make connector discovery conditional. Source control through `git` is always available in repos; GitHub through `gh` is available only when installed and authenticated. Other categories depend on discovered connectors. The output must distinguish searched, unavailable, and skipped sources.

### `reflect`

Remove transcript-path dependency. Reflection should operate on current conversation context and available memory by default. If the user provides transcript paths, use them. The skill should produce concrete skill-edit proposals, not require nested reviewers.

### `automate-me`

Convert transcript mining into an optional evidence source. The core flow should be: inspect existing mode skills, gather current user preferences from conversation and memory, ask one focused question if needed, then draft or update a repo-local skill under `.agents/skills/`.

### `show-me-your-work`

Keep the TSV decision-trail format and helper script. Replace transcript auditing with an evidence audit against current conversation notes, command output, git diff, and any provided logs. Transcript audit becomes optional when an explicit transcript path exists.

### `poteto-mode`

Rewrite as a compact Codex mode skill. Preserve concise style, decision trails for long work, verification posture, and useful principle routing. Remove hardcoded subagent routing, model choices, PR babysitting, blanket MCP autonomy, and unavailable cleanup commands.

## Validation

The implementation is complete only when these checks pass:

```bash
rg -n 'Task tool|Task call|subagent_type|readonly|AskQuestion|agent-transcripts|composer-|claude-|gpt-|deslop|poteto-agent|Cursor|\\.cursor|~/\\.cursor|create-skill' .agents/skills
```

The command should return no matches except intentional references in examples that explicitly say they are not Codex mechanics.

Also run a reference check that verifies backtick references to `references/...`, `playbooks/...`, and `scripts/...` resolve from the skill root, ignoring documented glob placeholders such as `references/sources/*.md`.

Finally, manually review `how`, `interrogate`, `why`, `reflect`, `automate-me`, `show-me-your-work`, and `poteto-mode` to confirm each has a complete Codex execution path without requiring unavailable tools.

## Out of Scope

- Creating new plugin manifests.
- Reinstalling skills globally.
- Preserving Cursor-specific workflows for cross-editor compatibility.
- Validating behavior with external services that are not connected in this environment.
