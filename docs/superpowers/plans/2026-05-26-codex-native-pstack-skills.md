# Codex-Native Pstack Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the repo-local pstack skills under `.agents/skills/` so their workflow instructions are executable by Codex rather than Cursor.

**Architecture:** Keep the imported skill directory layout. Rewrite workflow-heavy skills inline, use conditional wording for optional tools, and validate mechanically with repository-local scans. Preserve principle/style skills except where they reference broken workflow mechanics.

**Tech Stack:** Markdown skills under `.agents/skills`, shell validation with `rg`, Python reference validation scriptlet, git commits.

---

## File Structure

- Modify: `.agents/skills/how/SKILL.md`
  - Inline-first codebase explanation workflow with optional multi-agent fallback.
- Modify: `.agents/skills/interrogate/SKILL.md`
  - Codex-native adversarial review without fixed model tables or Cursor Task fields.
- Modify: `.agents/skills/why/SKILL.md`
  - Conditional connector discovery and source-control-first rationale workflow.
- Modify: `.agents/skills/why/references/synthesizer-prompt.md`
  - Fix skill-root-relative reference wording for `references/epistemics.md`.
- Modify: `.agents/skills/reflect/SKILL.md`
  - Reflection from current context, memory, repo evidence, and optional user-provided transcripts.
- Modify: `.agents/skills/reflect/references/*.md`
  - Replace transcript/MCP assumptions with Codex-safe evidence language where necessary.
- Modify: `.agents/skills/automate-me/SKILL.md`
  - Repo-local skill authoring workflow using conversation, memory, and optional transcripts.
- Modify: `.agents/skills/show-me-your-work/SKILL.md`
  - Keep TSV log format; replace transcript-required audit with evidence audit.
- Modify: `.agents/skills/poteto-mode/SKILL.md`
  - Compact Codex mode, preserving style and verification while removing Cursor agent routing.
- Modify: `.agents/skills/poteto-mode/references/plan.md`
- Modify: `.agents/skills/poteto-mode/playbooks/*.md`
  - Remove mandatory control-skill, PR babysit, hardcoded subagent, and unavailable cleanup mechanics.
- Modify: `.agents/skills/architect/SKILL.md`, `.agents/skills/arena/SKILL.md`, `.agents/skills/figure-it-out/SKILL.md`
  - Light-touch cleanup for hardcoded model or subagent assumptions only.
- Modify: `.agents/skills/python-style-touchup/SKILL.md`
- Modify: `.agents/skills/python-style-touchup/scripts/touchup.sh`
  - Keep existing path fix to `.agents/skills`.

## Task 1: Baseline Skill Validation

**Files:**
- Read: `.agents/skills/**/SKILL.md`
- Read: `docs/superpowers/specs/2026-05-26-codex-native-pstack-skills-design.md`

- [ ] **Step 1: Record the current incompatibility scan**

Run:

```bash
rg -n 'Task tool|Task call|subagent_type|readonly|AskQuestion|agent-transcripts|composer-|claude-|gpt-|deslop|poteto-agent|Cursor|\.cursor|~/\.cursor|create-skill' .agents/skills
```

Expected: multiple matches in workflow skills. Save the themes mentally; this is the RED baseline.

- [ ] **Step 2: Record current skill inventory**

Run:

```bash
find .agents/skills -mindepth 2 -maxdepth 2 -name SKILL.md -print | sort
```

Expected: 33 `SKILL.md` files including `python-style-touchup`.

- [ ] **Step 3: Do not commit**

This task establishes baseline evidence only. No commit.

## Task 2: Rewrite Core Explanation and Review Skills

**Files:**
- Modify: `.agents/skills/how/SKILL.md`
- Modify: `.agents/skills/interrogate/SKILL.md`
- Modify: `.agents/skills/architect/SKILL.md`
- Modify: `.agents/skills/arena/SKILL.md`
- Modify: `.agents/skills/figure-it-out/SKILL.md`

- [ ] **Step 1: Rewrite `how`**

Replace Cursor agent sections with this Codex-native structure:

```markdown
## Explain Mode

1. Interpret the question and state the assumed scope when ambiguous.
2. Search with `rg`/`rg --files`, then read targeted files.
3. For complex subsystems, use `tool_search` to discover Multi-agent tools. If available, split independent slices across agents; if not, continue inline.
4. Synthesize the answer with Overview, Key Concepts, How It Works, Where Things Live, and Gotchas.
```

Keep the existing output format. Remove `Task`, `subagent_type`, `readonly`, and fixed model references.

- [ ] **Step 2: Rewrite `interrogate`**

Replace fixed four-model review with:

```markdown
## Review Flow

1. Determine scope from explicit files, staged diff, branch diff, or recent context.
2. State the intent in one paragraph.
3. If Multi-agent tools are available through `tool_search`, dispatch independent reviewers with the same prompt and rubric.
4. If not, run the review inline through four lenses: correctness, integration risk, maintainability, and test coverage.
5. Synthesize findings as Act On, Consider, Noted, Dismissed, and Agreement Map.
```

Remove the model table and any instruction to open a PR to update model slugs.

- [ ] **Step 3: Light-clean `architect`, `arena`, and `figure-it-out`**

Replace hardcoded model-slug requirements with conditional language:

```markdown
Use available multi-agent tooling when it materially improves independent exploration. If no such tool is available, perform the comparison inline and document the tradeoffs.
```

Do not rewrite these skills beyond broken platform mechanics.

- [ ] **Step 4: Validate this task**

Run:

```bash
rg -n 'Task tool|Task call|subagent_type|readonly|composer-|claude-|gpt-|poteto-agent' .agents/skills/how .agents/skills/interrogate .agents/skills/architect .agents/skills/arena .agents/skills/figure-it-out
```

Expected: no matches.

- [ ] **Step 5: Commit**

Run:

```bash
git add .agents/skills/how .agents/skills/interrogate .agents/skills/architect .agents/skills/arena .agents/skills/figure-it-out
git commit -m "docs: adapt review skills for codex"
```

## Task 3: Rewrite Historical and Reflection Skills

**Files:**
- Modify: `.agents/skills/why/SKILL.md`
- Modify: `.agents/skills/why/references/synthesizer-prompt.md`
- Modify: `.agents/skills/reflect/SKILL.md`
- Modify: `.agents/skills/reflect/references/*.md`
- Modify: `.agents/skills/automate-me/SKILL.md`

- [ ] **Step 1: Rewrite `why` connector discovery**

Keep the seven evidence categories, but rewrite the execution rules:

```markdown
1. Always inspect local git history and relevant files first.
2. Use `gh` only when it is installed, the repo has a GitHub remote, and auth works.
3. Use `tool_search` to discover connectors for tickets, docs, chat, observability, errors, or analytics.
4. If a connector is unavailable, mark that category as unavailable rather than skipped.
5. Use multi-agent tools only when discovered and useful; otherwise query sources inline.
```

Remove hard requirements for one investigator per MCP when no multi-agent tool exists.

- [ ] **Step 2: Fix `why` reference wording**

In `.agents/skills/why/references/synthesizer-prompt.md`, change:

```markdown
You MUST follow the framework in `references/epistemics.md`.
```

to:

```markdown
You MUST follow the framework in the sibling file `epistemics.md` when reading this prompt directly, or `references/epistemics.md` from the `why` skill root.
```

- [ ] **Step 3: Rewrite `reflect`**

Replace transcript-first flow with:

```markdown
## Evidence Order

1. Current conversation context.
2. Repo changes and git history.
3. Codex memory when relevant.
4. User-provided transcripts or logs.

If no transcript is available, proceed from the first three sources and state the limitation.
```

Remove required `Task` calls, `agent-transcripts`, model slugs, and `readonly`.

- [ ] **Step 4: Rewrite `automate-me`**

Replace `AskQuestion` and transcript mining requirements with:

```markdown
1. Look for an existing `*-mode` skill under `.agents/skills` and `~/.codex/skills`.
2. Gather current preferences from the conversation and relevant memory.
3. If a key preference is missing, ask one concise plain-text question.
4. Draft or update the repo-local skill under `.agents/skills/<handle>-mode/SKILL.md`.
5. Validate frontmatter and run a short self-review for specificity, trigger quality, and stale assumptions.
```

- [ ] **Step 5: Validate this task**

Run:

```bash
rg -n 'AskQuestion|agent-transcripts|Task call|subagent_type|readonly|composer-|claude-|gpt-' .agents/skills/why .agents/skills/reflect .agents/skills/automate-me
```

Expected: no matches.

- [ ] **Step 6: Commit**

Run:

```bash
git add .agents/skills/why .agents/skills/reflect .agents/skills/automate-me
git commit -m "docs: adapt reflection skills for codex"
```

## Task 4: Rewrite Decision-Trail and Mode Skills

**Files:**
- Modify: `.agents/skills/show-me-your-work/SKILL.md`
- Modify: `.agents/skills/poteto-mode/SKILL.md`
- Modify: `.agents/skills/poteto-mode/references/plan.md`
- Modify: `.agents/skills/poteto-mode/playbooks/*.md`
- Modify: `.agents/skills/python-style-touchup/SKILL.md`
- Modify: `.agents/skills/python-style-touchup/scripts/touchup.sh`

- [ ] **Step 1: Rewrite `show-me-your-work` audit**

Keep the TSV format and helper script. Replace transcript-required audit with:

```markdown
## Audit the log against evidence

Before handing back, check the log against available evidence: current conversation, command output, git diff, commits, test logs, generated artifacts, and any explicit transcript path the user provided. If no transcript path exists, do not fail the audit; state that transcript evidence was unavailable.
```

- [ ] **Step 2: Rewrite `poteto-mode`**

Keep:

- concise direct prose
- named data shapes before code
- verification before completion
- decision trails for long work
- principle-skill routing

Remove:

- `deslop`
- required control skills
- PR babysitting
- blanket MCP autonomy
- hardcoded subagent routing
- model slug defaults

Use this replacement for subagent guidance:

```markdown
Use multi-agent tooling only when it is available and the task has independent slices. Discover it with `tool_search`. If unavailable, work inline and keep the context small.
```

- [ ] **Step 3: Clean poteto playbooks**

For every file under `.agents/skills/poteto-mode/playbooks/` and `.agents/skills/poteto-mode/references/plan.md`, replace mandatory unavailable mechanisms with conditional Codex wording:

- "control skill" -> "available browser, runtime, CLI, or screenshot tooling; otherwise use the closest direct verification command"
- "PR follow-up workflow" -> "code review or PR follow-up only when requested and available"
- "subagent with model X" -> "available multi-agent worker when useful; otherwise inline"

- [ ] **Step 4: Keep python-style-touchup path fix**

Confirm the command examples use:

```bash
bash .agents/skills/python-style-touchup/scripts/touchup.sh
```

Do not otherwise rewrite `python-style-touchup`.

- [ ] **Step 5: Validate this task**

Run:

```bash
rg -n 'deslop|control skill|PR follow-up|subagent_type|poteto-agent|composer-|claude-|gpt-|agent-transcripts|Task call|AskQuestion' .agents/skills/show-me-your-work .agents/skills/poteto-mode .agents/skills/python-style-touchup
```

Expected: no matches.

- [ ] **Step 6: Commit**

Run:

```bash
git add .agents/skills/show-me-your-work .agents/skills/poteto-mode .agents/skills/python-style-touchup
git commit -m "docs: adapt mode skills for codex"
```

## Task 5: Final Validation and Import Commit

**Files:**
- Validate: `.agents/skills/**`

- [ ] **Step 1: Run the full forbidden-pattern scan**

Run:

```bash
rg -n 'Task tool|Task call|subagent_type|readonly|AskQuestion|agent-transcripts|composer-|claude-|gpt-|deslop|poteto-agent|Cursor|\.cursor|~/\.cursor|create-skill' .agents/skills
```

Expected: no matches except intentional examples that explicitly say they are not Codex mechanics. If matches remain, edit the relevant skill and rerun.

- [ ] **Step 2: Run frontmatter validation**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re
failed = False
for path in sorted(Path(".agents/skills").glob("*/SKILL.md")):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        print("missing frontmatter", path)
        failed = True
        continue
    end = text.find("\n---\n", 4)
    if end == -1:
        print("unterminated frontmatter", path)
        failed = True
        continue
    frontmatter = text[4:end]
    if not re.search(r"^name:\s*\S+", frontmatter, re.M):
        print("missing name", path)
        failed = True
    if not re.search(r"^description:\s*.+", frontmatter, re.M):
        print("missing description", path)
        failed = True
    if len(frontmatter) > 1024:
        print("frontmatter too long", path, len(frontmatter))
        failed = True
raise SystemExit(1 if failed else 0)
PY
```

Expected: exit 0 with no output.

- [ ] **Step 3: Run reference validation**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re
missing = []
for skill in Path(".agents/skills").iterdir():
    if not skill.is_dir():
        continue
    for path in skill.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"`((?:references|playbooks|scripts)/[^`]+?)`", text):
            ref = match.group(1)
            if "*" in ref or "<" in ref or " " in ref:
                continue
            if not (skill / ref).exists():
                missing.append((path, ref))
if missing:
    for path, ref in missing:
        print(f"{path}: missing {ref}")
    raise SystemExit(1)
PY
```

Expected: exit 0 with no output.

- [ ] **Step 4: Run git status check**

Run:

```bash
git status --short -- .agents/skills
```

Expected: only intended `.agents/skills` changes are listed.

- [ ] **Step 5: Commit remaining imported skills**

If prior tasks left uncommitted skill directories, run:

```bash
git add .agents/skills
git commit -m "docs: add codex-native pstack skills"
```

If there is nothing to commit, skip this step.

- [ ] **Step 6: Final status**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: no unstaged `.agents/skills` changes; recent commits show the adaptation commits.
