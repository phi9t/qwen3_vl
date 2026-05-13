# Design: repo-local Symphony — `draccus/` + `autonomy/` monorepo

## Context

[openai/symphony](https://github.com/openai/symphony) specifies *autonomous implementation runs*: a work queue feeds tasks to isolated agents, each run produces "proof of work" (CI / PR / tests), and humans manage at the queue level rather than supervising agents one-by-one. We are building a **repo-local** implementation of that spec inside `/data02/home/philip.yang/workspace/qwen3_vl`, with three opinionated choices:

1. **Monorepo**: vendor `~/draccus` as a top-level `draccus/` directory (straight copy, no submodule, no subtree). Create a new top-level `autonomy/` directory for the Symphony implementation. Both live next to the existing `qwen-vl-finetune/` research stack.
2. **Temporal as orchestration**: reuse the local Temporal server pattern the repo already uses for ML campaigns (`qwen-vl-finetune/research/local_temporal.py`). One Temporal workflow = one `tracker.org` file.
3. **Org-mode is the queue *and* the journal**: the workflow scans `TODO` headings, executes them inside `draccus-run`, and flips state back on the same heading. Emacs / Magit users can read and edit at rest; the workflow re-reads on every pick.

Backed by user choices: executor is generic (claude-code | torchrun-trial | shell behind one interface); isolation is `draccus-run` (bwrap + Spack ML foundation); done-gate is `pytest` + typecheck + a synthetic-data GPT-2 smoke training step on GPU.

## Monorepo layout

```
/data02/home/philip.yang/workspace/qwen3_vl/
├── draccus/                  # VENDORED: straight `cp -a ~/draccus/. draccus/` then rm -rf draccus/.git
│   ├── bin/{draccus-run,draccus-build,draccus-uv,...}
│   ├── lib/, scripts/, envs/, shims/, rootfs/, state/, cache/, build/
│   ├── AGENTS.md, CLAUDE.md (symlink → AGENTS.md), DESIGN.md
│   └── .workstream/...        # historical draccus workstreams kept as read-only reference
├── autonomy/                  # NEW: the repo-local Symphony implementation
│   ├── pyproject.toml         # owns its deps: temporalio, orgparse, gitpython, transformers
│   ├── autonomy/
│   │   ├── workflows.py       # SymphonyTrackerWorkflow (one per tracker.org)
│   │   ├── activities.py
│   │   ├── org/
│   │   │   ├── schema.py      # OrgTask dataclass + property contract (canonical schema)
│   │   │   ├── parser.py      # orgparse-backed read
│   │   │   └── mutator.py     # flock+fsync atomic in-place write
│   │   ├── executors/
│   │   │   ├── base.py        # Executor Protocol + RunResult
│   │   │   ├── claude_code.py
│   │   │   ├── torch_trial.py # delegates to research/observability/trial_runner.py
│   │   │   └── shell.py
│   │   ├── sandbox.py         # invokes ../draccus/bin/draccus-run with DRACCUS_WORKSPACE
│   │   ├── worktree.py        # `git worktree add .autonomy/worktrees/<slug>/<task>`
│   │   ├── done_gate.py       # pytest + ruff + pyright + gpt2_synthetic_smoke
│   │   ├── smoke/
│   │   │   └── gpt2_synthetic.py
│   │   ├── worker.py          # Temporal worker on task queue "autonomy"
│   │   ├── cli.py             # `autonomy submit | status | signal | tail`
│   │   └── templates/
│   │       └── tracker.org.tmpl
│   ├── tests/
│   └── README.md
├── .autonomy/                 # runtime state, mostly gitignored
│   ├── runs/<slug>/
│   │   ├── design.md          # tracked
│   │   ├── tracker.org        # tracked
│   │   └── artifacts/         # gitignored: run.log, gate output, training logs
│   └── worktrees/             # gitignored
├── qwen-vl-finetune/          # existing, unchanged at this layer
└── pyproject.toml             # existing repo root pyproject; autonomy has its own
```

### Bringing draccus in (one-shot)

```
cp -a /data02/home/philip.yang/draccus/. ./draccus/
rm -rf ./draccus/.git
# keep draccus/.gitignore as-is so state/, cache/, build/, rootfs/ stay ignored from the outer repo's view too
git add draccus/
git commit -m "vendor draccus@<sha> into monorepo"
```

After the copy: `draccus/bin/draccus-run` resolves `DRACCUS_BUNDLE` via `lib/draccus-env.sh` (parent-of-`lib/`) so it works in its new home without environment tweaks. `rootfs/` and `state/spack/` are large built artifacts — gitignored, rebuilt per fresh clone via `draccus/scripts/bootstrap-rootfs.sh` and `draccus/bin/draccus-build`. The hard invariants in `draccus/AGENTS.md` (do-not-shadow list; two-layer Python model; `validate-static.sh` on every edit under `draccus/`) carry over verbatim — they apply only to changes inside `draccus/`, not to `autonomy/` or `qwen-vl-finetune/`.

## Canonical org-mode schema — `tracker.org`

A `tracker.org` file at `.autonomy/runs/<slug>/tracker.org` is the **input contract** for one `SymphonyTrackerWorkflow` execution. The schema reuses the keyword set and drawer conventions from `draccus/.workstream/*/tracker.org` (so Emacs configs already tuned for one work for the other), with Symphony-specific properties added.

### File header

```org
#+TITLE: <human-readable run name>
#+AUTHOR: <whoever bootstrapped — informational only>
#+STARTUP: overview logdone
#+TODO: TODO(t) READY(r) IN-PROGRESS(i!) BLOCKED(b@/!) AWAITING-GATE(g!) | DONE(d!) WONTFIX(w@/!) FAILED(f@/!)
#+PROPERTY: header-args :eval no
#+FILETAGS: :autonomy:<slug>:
#+AUTONOMY_VERSION: 1
#+AUTONOMY_RUN_SLUG: <slug>            # MUST match the parent directory name
#+AUTONOMY_DEFAULT_EXECUTOR: claude-code
#+AUTONOMY_DEFAULT_GATE: tests+typecheck+gpt2-smoke
```

Workflow IDs are deterministic: `autonomy-tracker-<slug>` — so resubmitting the same file is idempotent (Temporal rejects the duplicate start unless the prior execution is closed).

### TODO keyword semantics

| Keyword | Set by | Meaning |
|---|---|---|
| `TODO` | human | Authored, not yet evaluated. |
| `READY` | workflow | Workflow has verified all `:DEPENDS:` are `DONE` and the task is the next pick. Transient. |
| `IN-PROGRESS` | workflow | Executor is running for this heading. `:OWNER:` and `:STARTED:` set. |
| `AWAITING-GATE` | workflow | Executor exited 0; done-gate is running (pytest + typecheck + gpt2-smoke). |
| `BLOCKED` | workflow OR human | If workflow: gate failed or executor returned a recoverable error; `** Blocker` subtree filled. If human: paused intentionally. The workflow never auto-promotes BLOCKED → READY; the human must demote to `TODO`. |
| `DONE` | workflow | Executor ran, gate passed. `:FINISHED:` set; `:ARTIFACTS:` listed. |
| `FAILED` | workflow | Executor crashed or gate failed permanently (e.g., logical contradiction in the task spec). Distinct from BLOCKED in that the workflow asserts no retry will help. |
| `WONTFIX` | human | Human-skipped; workflow ignores. |

### Heading + properties (per-task contract)

```org
* Tasks
** TODO T03 Replace bilinear with nearest-neighbor in vision tower preproc
   :PROPERTIES:
   :ID:        T03                              ← REQUIRED. Stable, unique within the file. Workflow keys by this.
   :EXECUTOR:  claude-code                      ← claude-code | torch-trial | shell
   :GATE:      tests+typecheck+gpt2-smoke       ← +/- composition of: tests, typecheck, gpt2-smoke, none
   :DEPENDS:   T01 T02                          ← space-separated :ID:s; empty/missing = no deps
   :TIMEOUT:   2h                               ← Go-style duration; default 6h; bounds the activity start_to_close_timeout
   :GPUS:      1                                ← passed through to draccus-run; default 1, "0" disables CUDA visibility
   :BRANCH:    autonomy/<slug>/T03              ← optional override; default computed
   :WORKTREE:                                   ← workflow-owned, written when IN-PROGRESS
   :OWNER:                                      ← workflow-owned: <workflow-id>:<attempt>
   :STARTED:                                    ← workflow-owned: ISO-8601 UTC
   :FINISHED:                                   ← workflow-owned: ISO-8601 UTC
   :EXIT_CODE:                                  ← workflow-owned: executor's exit
   :GATE_RESULT:                                ← workflow-owned: pass | fail:<which-check>
   :ARTIFACTS:                                  ← workflow-owned: relative paths under artifacts/T03/
   :PR:                                         ← reserved for future; not part of v1 done-gate
   :END:

   *Goal.* One-paragraph plain-English statement of what success looks like. This is what the executor reads.

   *Constraints.* Bullets the executor must respect. E.g., "do not modify qwen-vl-finetune/research/."

   *Acceptance.* Free-form additional gate criteria beyond :GATE: (encoded as cmd:... lines the
   done-gate runs after the standard checks; one per line).

*** Logbook
    :LOGBOOK:
    - State "IN-PROGRESS" from "TODO" [2026-05-13 Wed 14:02] :: claimed by autonomy-tracker-<slug>:1
    - State "AWAITING-GATE" from "IN-PROGRESS" [2026-05-13 Wed 14:47] :: executor exit 0
    - State "DONE" from "AWAITING-GATE" [2026-05-13 Wed 14:51] :: gate pass
    :END:

*** Blocker        ← only present when state is BLOCKED or FAILED; workflow appends
    Reason: gate fail:gpt2-smoke — loss did not decrease.
    Excerpt:
    #+begin_src text
    [trainer] step 0 loss=10.91
    [trainer] step 29 loss=10.93
    AssertionError: smoke loss non-decreasing (delta=+0.02)
    #+end_src
    See artifacts/T03/gpt2_smoke.log.
```

#### Field-by-field contract

- **`:ID:`** — required, treated as opaque; conventionally `T<NN>` or `<phase>.<NN>`. The workflow keys all state by this. Renaming an `:ID:` while running is undefined behavior.
- **`:EXECUTOR:`** — one of `claude-code | torch-trial | shell`. Falls back to `#+AUTONOMY_DEFAULT_EXECUTOR:` then to `claude-code`.
- **`:GATE:`** — `+`-joined subset of `{tests, typecheck, gpt2-smoke}`, or the literal `none`. Empty falls back to `#+AUTONOMY_DEFAULT_GATE:`. The gate runs after the executor exits 0; if the executor exits non-zero, the gate is skipped and state goes BLOCKED (executor) or FAILED (depending on retry policy).
- **`:DEPENDS:`** — space-separated `:ID:`s in the same file. Cross-file deps are not supported in v1; if you need them, split into two trackers and submit sequentially.
- **`:TIMEOUT:`** — sets `start_to_close_timeout` on the `launch_run` activity. Beyond this, Temporal will cancel the activity and the workflow marks the task BLOCKED with a timeout reason.
- **`:GPUS:`** — number of GPUs the run requires. `sandbox.py` propagates this as `CUDA_VISIBLE_DEVICES` into the `draccus-run` invocation, picked from a simple round-robin pool against `nvidia-smi --query-gpu=index --format=csv,noheader`.
- **`:BRANCH:`** / **`:WORKTREE:`** — both workflow-owned in v1. User-supplied `:BRANCH:` is honored if set in the source; otherwise the worktree manager picks `autonomy/<slug>/<ID>`.

#### Body grammar

- **`*Goal.*`** paragraph — handed to `claude-code` as the prompt; informational for other executors.
- **`*Constraints.*`** bullets — concatenated to the claude-code prompt; informational for other executors.
- **`*Acceptance.*`** lines prefixed `cmd: ...` — extra shell commands the done-gate runs (each must exit 0). One per line. This is how Symphony-style "proof of work" extensions plug in without changing the executor.
- **`*** Logbook` `:LOGBOOK:` drawer** — workflow appends one line per state transition. Same format Org's `logdone` uses, so Emacs renders it natively.
- **`*** Blocker` subtree** — workflow appends on BLOCKED / FAILED; carries the gate summary and a 30-line log excerpt. Large logs spill to `artifacts/<ID>/`.

### Decisions section (optional, copied from draccus convention)

```org
* Decisions
** <ID> <short-decision-name>
   :PROPERTIES:
   :DECIDED_BY: <human | executor>
   :DECIDED_ON: <2026-05-13>
   :SIGN_OFF:   user-approved | executor-default
   :END:
   Body explaining the decision.
```

Workflow does not read this section — it's purely human-facing context.

## Temporal layout

Reuse `qwen-vl-finetune/research/local_temporal.py` (local dev server at `~/.automata/research/temporal/temporal.db`). The new worker uses a distinct task queue `autonomy` so it does not compete with `ProfiledExperimentWorkflow` on the existing `research` queue.

### `SymphonyTrackerWorkflow` (one per `tracker.org`)

```python
@workflow.defn
class SymphonyTrackerWorkflow:
    @workflow.run
    async def run(self, tracker_path: str) -> dict:
        while not (self.cancel_requested or self.pause_after_current):
            picked = await workflow.execute_activity(
                org_pick_next_ready, tracker_path,
                start_to_close_timeout=timedelta(seconds=30),
            )
            if picked is None:                              # nothing READY
                break
            task: OrgTask = picked
            await workflow.execute_activity(
                org_transition, args=[tracker_path, task.id, "IN-PROGRESS",
                                      {"OWNER": workflow.info().workflow_id,
                                       "STARTED": workflow.now().isoformat()}],
                start_to_close_timeout=timedelta(seconds=30),
            )
            worktree = await workflow.execute_activity(
                worktree_create, args=[task.slug, task.id, task.branch],
                start_to_close_timeout=timedelta(minutes=2),
            )
            try:
                exec_result = await workflow.execute_activity(
                    launch_run, args=[task, worktree],
                    start_to_close_timeout=task.timeout,    # honors :TIMEOUT:
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
                if exec_result.exit_code != 0:
                    await self._block(tracker_path, task, exec_result, reason="executor")
                    continue
                await workflow.execute_activity(
                    org_transition, args=[tracker_path, task.id, "AWAITING-GATE", {}],
                    start_to_close_timeout=timedelta(seconds=30),
                )
                gate = await workflow.execute_activity(
                    run_done_gate, args=[task, worktree],
                    start_to_close_timeout=timedelta(hours=1),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
                if gate.ok:
                    await workflow.execute_activity(
                        org_transition,
                        args=[tracker_path, task.id, "DONE",
                              {"FINISHED": workflow.now().isoformat(),
                               "EXIT_CODE": "0", "GATE_RESULT": "pass",
                               "ARTIFACTS": ",".join(gate.artifacts)}],
                        start_to_close_timeout=timedelta(seconds=30),
                    )
                else:
                    await self._block(tracker_path, task, gate, reason="gate")
            finally:
                await workflow.execute_activity(
                    worktree_destroy_if_clean, args=[worktree, task.id],
                    start_to_close_timeout=timedelta(minutes=2),
                )
            self.completed += 1
        return {"completed": self.completed, "phase": "paused" if self.pause_after_current else "drained"}
```

- Workflow ID: `autonomy-tracker-<slug>` — derived from `#+AUTONOMY_RUN_SLUG:` so resubmits are idempotent.
- Signals: `pause_after_current()`, `cancel()`, `skip_task(id)`, `rerun_task(id)` (last one demotes a `DONE`/`BLOCKED`/`FAILED` heading back to `TODO` via an org activity and re-loops).
- Queries: `status()` → `{active_task, completed, blocked_ids, last_gate_summary}`.
- Workflow code never touches the filesystem — all I/O sits in activities. Mirrors the discipline in `docs/superpowers/specs/2026-05-13-temporal-torch-experimentation-trial-runs-design.md`.

### Activities

- `org_pick_next_ready(path) -> OrgTask | None` — parse with orgparse; pick lowest-position `TODO` whose `:DEPENDS:` are all `DONE`. Returns the parsed task spec (frozen dataclass), no mutation.
- `org_transition(path, task_id, new_state, props_to_set)` — `flock(LOCK_EX)` on `path`; re-parse; mutate keyword + properties + append `:LOGBOOK:` line; write atomically via tempfile + `os.replace` + `fsync` on the dir. Returns nothing.
- `launch_run(task, worktree) -> RunResult` — invokes `autonomy.sandbox.draccus_exec(executor.build_command(task, worktree), workspace=worktree, gpus=task.gpus)` which shells out to `./draccus/bin/draccus-run`. Streams stdout/stderr to `.autonomy/runs/<slug>/artifacts/<task.id>/run.log`. Returns `{exit_code, log_path, summary, structured_result}`.
- `run_done_gate(task, worktree) -> GateResult` — composes the checks named in `:GATE:`. Each check runs inside its own `draccus-run` invocation against the worktree. Composes stdout into `artifacts/<task.id>/gate.log` and per-check summaries.
- `worktree_create(slug, task_id, branch_override) -> Path` — `git worktree add .autonomy/worktrees/<slug>/<task_id> -b <branch>` (or checks out existing branch if a rerun).
- `worktree_destroy_if_clean(worktree, task_id)` — `git worktree remove` iff the working tree is clean AND the task ended `DONE`; otherwise leaves it for the human to inspect.

### Executors

```python
class Executor(Protocol):
    name: str
    def build_command(self, task: OrgTask, worktree: Path) -> list[str]: ...
    def parse_result(self, exit_code: int, log_path: Path) -> dict: ...
```

- **`claude-code`** — `claude -p "$(render_prompt(task))" --output-format=stream-json --permission-mode=acceptEdits`. The render templates Goal/Constraints/Acceptance into a single prompt. Result parsed from the final `result` event.
- **`torch-trial`** — converts the task's `:PROPERTIES:` (`PROFILE`, `PHASE`, `TRIAL`, `ENV_*`) into a `TrialSpec` from `qwen-vl-finetune/research/models.py` and delegates to `qwen-vl-finetune/research/observability/trial_runner.SupervisorTrialRunner`. Returns the existing `TrialAnalysis` as the structured result; `FailureReason` (OOM/NCCL/etc.) is surfaced into the `** Blocker` subtree on failure.
- **`shell`** — runs the body's `cmd:` lines sequentially inside one `draccus-run`. Useful escape hatch and for the bootstrap smoke test.

### Done-gate composition

Order is fixed (cheapest first): `tests` → `typecheck` → `gpt2-smoke` → `cmd:` lines from Acceptance.

- `tests` — `uv run pytest -x --quiet` from the worktree root, against the existing repo-root `pyproject.toml` (which already configures pytest).
- `typecheck` — `uv run ruff check .` then `uv run pyright autonomy/ qwen-vl-finetune/` (both already in `pyproject.toml`).
- `gpt2-smoke` — `autonomy/autonomy/smoke/gpt2_synthetic.py`: loads `gpt2` (HuggingFace, full model — small enough for any GPU; uses `hf-internal-testing/tiny-random-gpt2` if `AUTONOMY_SMOKE_TINY=1`), builds synthetic `input_ids` of shape `(8, 128)` with random vocab tokens, runs 30 AdamW steps at `lr=5e-5`, asserts `mean(loss[-5:]) < mean(loss[:5]) - 0.05` AND no NaN/Inf. Fails BLOCKED with a 30-line log excerpt if CUDA is unavailable or the assertion fails. Runs inside `draccus-run` so torch/CUDA come from the pinned `base-ml` view.

The gate's stop-on-first-failure means we never spend GPU minutes when pytest already failed.

## Critical files to create

- `draccus/` — straight `cp -a` of `~/draccus/.` with `.git` removed.
- `autonomy/pyproject.toml` — declares deps: `temporalio`, `orgparse`, `gitpython`, `transformers`, `torch` (resolved from `draccus`'s `base-ml` view via `uv ... --system-site-packages` per draccus's two-layer rule).
- `autonomy/autonomy/{workflows,activities,sandbox,worktree,worker,cli,done_gate}.py`.
- `autonomy/autonomy/org/{schema,parser,mutator}.py`.
- `autonomy/autonomy/executors/{base,claude_code,torch_trial,shell}.py`.
- `autonomy/autonomy/smoke/gpt2_synthetic.py`.
- `autonomy/autonomy/templates/tracker.org.tmpl`.
- `.autonomy/` (repo-level), `.gitignore` additions for `.autonomy/worktrees/` and `.autonomy/runs/*/artifacts/`.

## Critical files reused unchanged

- `draccus/bin/draccus-run` — invoked by `autonomy/autonomy/sandbox.py` as `./draccus/bin/draccus-run -- <cmd>`.
- `qwen-vl-finetune/research/local_temporal.py` — local Temporal server lifecycle.
- `qwen-vl-finetune/research/observability/trial_runner.py` — the `torch-trial` executor delegates to this directly.
- `qwen-vl-finetune/research/models.py` — `TrialSpec` / `TrialMetrics`.
- `qwen-vl-finetune/research/observability/{schema,analyzer,artifacts}.py` — failure classification stays authoritative for `torch-trial`.
- Repo root `pyproject.toml` — pytest, ruff, pyright config already live here; the gate just shells out.

## Operational shape

```
$ ls .autonomy/runs/refactor-data-loader/
design.md  tracker.org

$ uv run -m autonomy.cli submit .autonomy/runs/refactor-data-loader/tracker.org
started workflow autonomy-tracker-refactor-data-loader

$ uv run -m autonomy.cli status refactor-data-loader
active: T02   completed: 1   blocked: []   last_gate: pass

$ uv run -m autonomy.cli tail refactor-data-loader       # streams active run.log

# Emacs reflects state on disk in real time — IN-PROGRESS / DONE flips live.
$ uv run -m autonomy.cli signal refactor-data-loader pause
```

## Verification (end-to-end)

0. **Vendor draccus**: `cp -a ~/draccus/. ./draccus/ && rm -rf ./draccus/.git`. Run `cd draccus && ./scripts/bootstrap-rootfs.sh && ./scripts/validate-static.sh` — expect Gate 0 green. Run `./bin/draccus-run -- python -c "import torch; print(torch.cuda.is_available())"` — expect `True`.
1. **Temporal up**: `uv run python -m research.local_temporal ensure` — server + DB ready.
2. **Worker up**: `uv run python -m autonomy.worker` — verify in Temporal UI (`http://localhost:8233`) both `SymphonyTrackerWorkflow` and `ProfiledExperimentWorkflow` appear and use different task queues.
3. **Shell smoke (no GPU)**: write `.autonomy/runs/echo-smoke/tracker.org` with a single `TODO T01` whose `:EXECUTOR: shell`, `:GATE: none`, body `cmd: echo hello`. `autonomy submit ...`. Expect on disk: keyword flip `TODO → IN-PROGRESS → DONE`, `:STARTED:`/`:FINISHED:` populated, `:LOGBOOK:` drawer has three lines, `artifacts/T01/run.log` contains `hello`.
4. **Full done-gate (GPU)**: add `TODO T02` with `:EXECUTOR: shell`, `:GATE: tests+typecheck+gpt2-smoke`, body `cmd: true`. Expect: pytest runs inside `draccus-run`, ruff+pyright pass, `gpt2_synthetic.py` loads `gpt2`, runs 30 AdamW steps, asserts loss decreased, task lands DONE with `:GATE_RESULT: pass`.
5. **Gate failure path**: intentionally introduce a failing pytest and rerun. Expect: state goes `IN-PROGRESS → AWAITING-GATE → BLOCKED`, `** Blocker` subtree appears with the pytest excerpt, GPU step is skipped (fast-fail order).
6. **`torch-trial`**: `TODO T03` with `:EXECUTOR: torch-trial`, `:PROFILE: b200`, `:PHASE: probe`, `:TRIAL: t-smoke`. Expect: `SupervisorTrialRunner` runs, writes `analysis.json` under its own artifacts root, and the workflow copies the analysis summary into the `:LOGBOOK:` drawer.
7. **`claude-code`**: `TODO T04` with `:EXECUTOR: claude-code` and Goal "rename `_DATASET_CACHE` to `_dataset_cache` in `qwen-vl-finetune/qwenvl/data/__init__.py`". Expect: the per-task worktree's branch contains the edit, gate passes, heading lands DONE.
8. **Signals**: `autonomy signal echo-smoke pause` between tasks → status `paused`; `cancel` aborts cleanly; manually flipping a `TODO` to `WONTFIX` in Emacs causes the next pick to skip it.
9. **Concurrency**: submit two tracker files simultaneously. Two workflow executions; two worktree subtrees; neither file shows interleaved writes (flock holds).
10. **Crash safety**: kill the worker mid-run; restart it. Temporal replays the workflow; the in-progress task either resumes (if the activity was idempotent and re-runs) or surfaces a clean retry decision in `tracker.org`.
