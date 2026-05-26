---
name: arena
description: "Spawn N parallel candidates at the same task, pick a base, graft the strongest parts of the losers into it. Use for /arena, 'arena this', 'throw it in the arena', or when one attempt at a non-trivial artifact would lock in the wrong shape."
disable-model-invocation: true
---

# Arena

Fan out N parallel attempts at the same task. Read every candidate end to end. Pick the strongest as the base. Graft the best ideas from the others into it. Verify the synthesized result.

## Start

Open a todolist with one entry per phase before launching anything. The arena runs autonomously and the list keeps phases from silently disappearing.

1. Frame
2. Fan out
3. Cross-judge
4. Pick
5. Graft
6. Verify

## Phase A: Frame

The candidates receive the same prompt, so the prompt is the contract. Get it right before launching anything.

1. State the artifact each candidate is producing.
2. Derive the rubric. State what success looks like for *this* task, then turn it into 3-6 concrete gradeable criteria. Concrete: `Adds a --dry-run flag that skips writes`. Vague: `code is correct`. The rubric is the picker's tool in Phase D; candidates only see the task.
3. Pick N candidates (default 4). Use independent tooling only when it improves separation of thought; otherwise run in parallel conceptually inside the same analysis stream.
4. Assign output paths. Each candidate must write to its own location; candidates writing to the same path is shared mutable state and violates independent synthesis.

## Phase B: Fan out

If available multi-agent tooling can materially improve independent generation, dispatch candidates in parallel through it; keep each candidate focused on:

- the shared task and rubric
- a single output path
- producing both artifact and rationale

If unavailable, run N internal candidate designs sequentially and track them separately in the same session.

Each rationale is mandatory. Without it, the parent cannot tell whether a candidate's structure is principled or accidental.

If a candidate fails to produce output, proceed with N-1 and note the dropout in the synthesis record.

## Phase C: Cross-judge

After all Phase B candidates complete, run a separate judge pass on a different reasoning profile when possible. It sees the rubric and all candidate artifacts, scores each criterion, and recommends a base with rationale. Do not start synthesis before this.

The judge can run in parallel with the parent reading in Phase D, but only after candidates are complete.

## Phase D: Pick a base

Read every candidate end to end before picking. Skimming N candidates surfaces only the candidate whose surface looks most familiar.

Score each candidate against the rubric criterion by criterion, then compare against the judge.

Agreement on the base confirms the pick. Disagreement means the rubric was ambiguous or one output is overfitted.

Pick the base on which a future maintainer can extend most easily without breaking invariants. Prefer the cleaner boundary or smaller surface area when two feel tied, per the Laziness Protocol.

Record the pick and reason in a short synthesis note alongside the base artifact, including the cross-judge's verdict.

## Phase E: Graft

Walk each losing candidate once more and identify what is worth porting into the base.

Fold each graft in by hand; keep coherence under one mental model.

Record what was grafted, from which candidate, and what was rejected and why. The rejection notes are the highest-signal part of the record. Future readers learn from what you considered and dropped, not just what you kept.

When candidates converge on the same shape, that is a strong agreement signal. Note convergence and ship the consensus shape. No graft is needed.

When candidates diverge widely, Phase A was under-specified. Reframe and re-run rather than averaging divergence.

## Phase F: Verify

The synthesized artifact has to hold up under the same scrutiny as any other output, per the **prove-it-works** principle skill. The arena is not a verification waiver.

If verification finds a missed issue, either Phase A was wrong (re-frame and re-run) or one candidate caught it and you missed the graft (go back to Phase E). Don’t paper over.

## Outputs

One synthesized artifact. One short synthesis note alongside, naming the base, the grafts (with source candidate), the rejections, the dropouts if any, and the verification result.
