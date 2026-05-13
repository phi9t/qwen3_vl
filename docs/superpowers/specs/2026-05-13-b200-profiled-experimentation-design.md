# B200 Profiled Experimentation and Temporal Orchestration Design

## Context

The current Qwen3-VL fine-tuning workspace already has an autoresearch-style
experiment tree under `qwen-vl-finetune/experiments/`, a Qwen3-VL-8B launcher,
VisualWebInstruct train/eval registration, and completed run logs. The
bring-up notes in `qwen-vl-experimentation-bringup.txt` are useful historical
context, but the live repository has drifted from that plan:

- `experiments/run_all.sh` still calls `scripts/sft_qwen3_4b.sh`, although the
  design docs describe an 8B campaign.
- The completed sweep used smaller batch settings (`bs=1/2`, `ga=4`) than the
  original B200 target (`bs=4/8`, `ga=2`).
- A later 8B smoke run on 2026-05-12 passed with `scripts/sft_qwen3_8b.sh`,
  `bs=2`, `ga=4`, and gradient checkpointing enabled, reaching validation loss
  around `0.5600` and peak VRAM around `95 GB`.
- Earlier high-throughput attempts at `bs=8`, `ga=2` crashed.

The design goal is to use the B200 GPUs deliberately without hard-coding B200
assumptions throughout the experiment code. A100 and H100 profiles should be
possible later, but only B200 needs to be implemented and validated now.

## Goals

Build a profile-aware experiment mechanism that:

1. Keeps hardware capacity decisions separate from scientific sweep choices.
2. Uses a B200 probe phase to discover a reliable high-throughput envelope.
3. Runs the real Qwen3-VL-8B VisualWebInstruct sweep only from passing capacity
   settings.
4. Records crashes and metrics consistently instead of depending on a single
   long-running shell process.
5. Adds Temporal orchestration for durable probe/select/sweep/summarize runs.
6. Leaves A100 and H100 as explicit profile stubs for future validation.

## Non-Goals

- Do not implement or validate A100/H100 profiles in this pass.
- Do not replace the existing training launcher internals unless required by
  the profile runner.
- Do not store orchestration code or reusable profile definitions under
  `qwen-vl-finetune/experiments/`.
- Do not build a general experiment platform beyond this Qwen3-VL campaign.

## Approach

Use a capability-gated runner with a stable concept of hardware profiles. The
B200 profile is active. H100 and A100 profiles exist only as disabled stubs.

The lifecycle is:

1. `probe --profile b200`: run short Qwen3-VL-8B jobs over a bounded B200
   capacity grid.
2. `select --profile b200`: parse probe metrics and choose the fastest passing
   capacity config below the B200 VRAM ceiling.
3. `sweep --profile b200`: run the scientific A/B/C/D sweep using the selected
   capacity config.
4. `summarize --profile b200`: write a report with probe results, selected
   capacity, phase winners, final checkpoint, and crashes.

Temporal becomes the durable orchestrator for that lifecycle. Shell scripts and
log parsing remain activities, not workflow code.

## Files and Boundaries

Autoresearch-style mechanics and Temporal orchestration live under:

```text
qwen-vl-finetune/research/
  profiles/
  selected/
  lib/
  workflows.py
  activities.py
  worker.py
  start.py
  models.py
  run_profiled.sh
```

Planned research files:

- `research/profiles/b200.env`: active B200 profile and probe grid values.
- `research/profiles/h100.env`: reserved profile stub, not used by default.
- `research/profiles/a100.env`: reserved profile stub, not used by default.
- `research/selected/b200.env`: selected B200 capacity config written by
  selection.
- `research/run_profiled.sh`: profile-aware shell entrypoint.
- `research/lib/common.sh`: shared shell helpers for profile loading, run
  directory creation, metric extraction, TSV appends, and winner selection.
- `research/workflows.py`: deterministic Temporal workflow definitions.
- `research/activities.py`: shell execution, log parsing, filesystem I/O, and
  summary writing.
- `research/worker.py`: Temporal worker process for the GPU node.
- `research/start.py`: CLI for starting campaign workflows.
- `research/models.py`: typed profile, trial, metric, and status payloads.

Experiment artifacts stay under:

```text
qwen-vl-finetune/experiments/
  runs/
  results.tsv
  summary.md
```

- `experiments/results.tsv`: canonical machine-readable table. Add columns for
  `profile`, `phase`, `throughput_steps_per_sec`, and `selected_capacity`.
- `experiments/summary.md`: campaign report.

`research/` contains reusable orchestration and experiment mechanics.
`experiments/` contains only captured details from concrete training trials and
campaigns: logs, metrics, summaries, run manifests, and checkpoints.

The profiled runner always calls `qwen-vl-finetune/scripts/sft_qwen3_8b.sh` for
this campaign. It must not accidentally call `sft_qwen3_4b.sh`.

Hardware assumptions live in `research/profiles/`. Scientific knobs such as
tuning regime, learning rate, warmup, weight decay, resolution phase, and
context phase remain in the sweep definition under `research/`.

## B200 Probe Policy

The initial B200 grid is intentionally bounded:

```text
max_pixels: 50176, 112896
model_max_length: 8192
batch_size: 2, 4, 8
gradient_accumulation_steps: 2, 4
gradient_checkpointing: True, False
```

Probe guardrails:

- `MAX_STEPS=50`
- `EVAL_STEPS=25`
- `SAVE_STEPS=50`
- `DATASETS=visualwebinstruct_train`
- `EVAL_DATASETS=visualwebinstruct_val`
- Record crashes and continue.
- Require both footer metrics: `val_loss:` and `peak_vram_mb:`.
- Parse throughput from trainer output when possible.
- Reject configs above `B200_VRAM_CEILING_MB`, default `160000`.

Selection rule:

1. Keep only successful probes with parseable `val_loss` and `peak_vram_mb`.
2. Reject logs with obvious OOM, NCCL, launcher, or missing-footer failures.
3. Keep only probes with `peak_vram_mb <= B200_VRAM_CEILING_MB`.
4. Prefer higher throughput.
5. Break ties by lower `val_loss`.
6. Break remaining ties by lower VRAM.
7. Write selected capacity to `research/selected/b200.env`.

The selected file provides concrete shell assignments copied from the chosen
probe row: `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `GRADIENT_CHECKPOINTING`,
`MAX_PIXELS`, and `MODEL_MAX_LENGTH`.

If no B200 probe passes, the runner may print the last known-good conservative
fallback (`bs=2`, `ga=4`, `gradient_checkpointing=True`) but must not use it
silently. It can use the fallback only when `ALLOW_PROFILE_FALLBACK=1`.

Resolution still belongs to the scientific sweep. If `336^2` passes the probe,
that proves feasibility, but Phase C remains responsible for comparing
resolution/context choices.

## Sweep Policy

The real sweep reuses the current A/B/C/D funnel, parameterized by
`research/selected/b200.env`.

- Phase A: tuning regime selection across projector-only, projector+LLM, and
  LoRA.
- Phase B: learning-rate sweep. If capacity probing has already selected a
  strong batch config, Phase B should avoid duplicating batch science unless the
  experiment explicitly asks for it.
- Phase C: resolution/context comparison through explicit overrides.
- Phase D: warmup and weight-decay polish.

Each run writes under:

```text
qwen-vl-finetune/experiments/runs/<profile>/<phase>/<trial>/
```

Each `results.tsv` row records:

- timestamp
- git commit
- profile
- phase
- trial name
- regime
- learning rate
- batch size
- gradient accumulation
- gradient checkpointing
- max pixels
- model max length
- warmup
- weight decay
- validation loss
- peak VRAM
- throughput when available
- status
- failure reason when available
- output directory
- log path

Failure reasons should be normalized as `oom`, `nccl`, `missing_footer`,
`launcher_error`, `cancelled`, or `unknown`.

## Temporal Workflow Design

Use Python Temporal SDK code under `qwen-vl-finetune/research/`.

Primary workflow:

```text
ProfiledExperimentWorkflow(profile="b200", campaign="qwen3vl-8b-vwi")
```

Workflow phases:

1. `load_profile` activity reads `research/profiles/b200.env`.
2. `plan_probe_trials` activity expands the B200 probe matrix.
3. The workflow runs probe trials with bounded concurrency. The default is
   sequential because each trial uses all 8 GPUs.
4. `select_capacity` activity parses probe results and writes
   `research/selected/b200.env`.
5. The workflow runs sweep phases A/B/C/D.
6. `summarize_campaign` activity writes `experiments/summary.md`.

Temporal code responsibilities:

- `workflows.py`: deterministic orchestration only. It schedules activities,
  stores returned results, makes deterministic decisions from those results,
  exposes queries/signals, and never shells out or reads logs directly.
- `activities.py`: subprocess execution, filesystem I/O, profile reading,
  log parsing, TSV updates, summary writing, and process cleanup.
- `worker.py`: runs on the GPU node and polls a task queue such as
  `qwen3vl-b200`.
- `start.py`: starts workflows for `probe`, `select`, `sweep`, `summarize`, or
  the full campaign.
- `models.py`: typed payloads for profiles, trial specs, metrics, statuses,
  and workflow state.

Trial execution activity:

```text
RunTrialActivity(trial_spec)
```

`RunTrialActivity` launches `scripts/sft_qwen3_8b.sh` with explicit environment
variables, writes output to the trial log, heartbeats periodically with current
log offset and best-known progress, terminates the subprocess group on
cancellation, and returns parsed metrics.

Training failures are trial results, not always activity failures. OOM, NCCL
timeouts, missing footers, and nonzero launcher exits should usually return a
`status=crash` result so the workflow can continue. Infrastructure failures,
such as inability to start a subprocess or write the run directory, can raise
activity errors and use Temporal retries.

Queries:

- current phase
- active trial
- completed trial count
- latest metrics
- selected capacity config

Signals or updates:

- pause after current trial
- skip a pending trial
- stop after current phase

Workflow code must remain deterministic. All shell commands, log reads,
`nvidia-smi` checks, TSV parsing, checkpoint inspection, and filesystem writes
belong in activities.

## Verification

Shell verification:

```bash
bash -n qwen-vl-finetune/research/run_profiled.sh
bash -n qwen-vl-finetune/research/lib/common.sh
```

Temporal verification:

- Import check for `qwen-vl-finetune/research/workflows.py`,
  `activities.py`, and `models.py`.
- Unit-test selection logic with synthetic probe rows.
- Start a Temporal dev server and worker locally.
- Run a dry-run workflow that schedules activities without launching training.
- Run `probe --profile b200` for real and require at least one passing probe.
- Run `select --profile b200` and require
  `qwen-vl-finetune/research/selected/b200.env`.

Campaign verification:

- Every completed trial has `val_loss` and `peak_vram_mb`.
- Every failed trial has a normalized failure reason.
- `summary.md` identifies selected B200 capacity, phase winners, final best
  checkpoint, and any crashes.
- The final best checkpoint is qualitatively spot-checked on held-out
  VisualWebInstruct prompts.

## Open Follow-Up

After B200 is implemented and validated, add real H100 and A100 profile grids
by running the same probe/select flow on those machines. Until then, their
profile files should remain explicit stubs and should not be selectable by
default.
