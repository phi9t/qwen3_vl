# Temporal/Torch Experimentation Trial Runs Design

## Purpose

Build a debugging and observability mechanism for Qwen-VL experimentation that
captures why a trial was run, what happened during the run, what failed or
worked, what actions were taken, and what root cause was found after analysis.

The mechanism must use Temporal for both local research runs and remote or
long-lived campaigns. Local runs use a local Temporal development server with a
file-backed SQLite database. Temporal provides durable orchestration and
operator control. A shared supervisor/analyzer library provides the actual
per-trial evidence capture and diagnosis inside activities.

## Goals

- Represent individual training trials in Temporal with clear progress,
  cancellation, and result visibility.
- Make the normal local command path Temporal-backed by starting or connecting
  to a local development server.
- Persist local Temporal state to a SQLite file outside committed source by
  default, under `~/.automata/research/temporal/`.
- Keep actual GPU training out of workflow code and inside activities.
- Preserve the trainer as a plain `torchrun` subprocess.
- Capture mandatory lightweight observability for every trial.
- Capture deeper diagnostics when failures occur.
- Produce machine-readable artifacts for aggregation and human-readable
  analysis for debugging.
- Distinguish expected capacity-probing failures from unexpected
  infrastructure or training failures.
- Keep the current `experiments/runs/...` area focused on concrete trial
  evidence and outputs.
- Keep heavyweight artifacts out of Git-managed paths by resolving them from a
  runtime artifact root and storing only relative references in metadata.

## Non-Goals

- Do not make trainer code depend on Temporal.
- Do not parse large logs or launch subprocesses from workflow code.
- Do not retry capacity OOMs as Temporal activity retries.
- Do not replace raw `run.log`; keep it as source evidence.
- Do not introduce a non-Temporal local campaign path for normal probe, sweep,
  select, or summarize commands.
- Do not write model weights, checkpoints, tokenizer files, full logs, or
  trainer state into Git-managed metadata directories.
- Do not store hard-coded local absolute paths in committed metadata artifacts.

## Boundary

### Temporal Workflow

The workflow is the durable campaign state machine. It owns campaign-level
decisions and visibility:

- profile loading and trial planning
- trial ordering
- phase transitions: `probe`, `select`, `sweep`, `summarize`
- signals: pause after current trial, stop after current phase, skip trial,
  cancel campaign, rerun trial
- queries: phase, active trial, completed count, latest analysis, selected
  capacity
- high-level action records such as stop, skip, select, or rerun

Workflow code must stay deterministic and lightweight. It should not inspect
GPU state, read large logs, run shell commands, or parse raw trainer output.

### Temporal Activity

Each trial attempt is one side-effectful activity. The activity owns:

- creating the per-trial artifact directory
- writing intent and resolved config artifacts
- taking pre-run and post-run system snapshots
- launching and monitoring the `torchrun` subprocess in its own process group
- streaming stdout/stderr to `run.log`
- handling activity cancellation by terminating the process group
- enforcing trial stall timeouts before the Temporal activity timeout is reached
- parsing trainer events and raw logs
- classifying failures
- writing `analysis.json` and `analysis.md`
- returning a compact `TrialAnalysis` payload to the workflow

The first implementation should use one activity per trial attempt. Child
workflows are reserved for a later stage if each trial needs independent
signals, multi-step remediation, or richer Temporal UI state than an activity
can provide.

### Trainer Process

The trainer remains a normal external process launched by the activity:

```text
torchrun ... qwenvl/train/train_qwen.py ...
```

It does not know about Temporal. When observability environment variables are
provided, it emits structured JSONL events in addition to normal stdout/stderr.
The existing greppable footers, such as `val_loss:` and `peak_vram_mb:`, remain
for compatibility.

### Artifact Store

The in-repo attempt directory is the source of lightweight trial evidence.
Temporal stores compact status and result payloads; it does not store large
logs. Heavy artifacts are written under a runtime-resolved artifact root and are
referenced from metadata by relative paths.

The artifact root is resolved at runtime:

```text
RESEARCH_ARTIFACT_ROOT if set
otherwise ~/.automata/research/experiments/
```

The default keeps most experimentation data outside the repository. A developer
may set `RESEARCH_ARTIFACT_ROOT` to a repo-local `.artifacts/research/` tree for
isolated project-local work, but `.artifacts/` must be ignored by Git. Metadata
stores the symbolic root and relative references, not expanded absolute paths:

```json
{
  "artifact_root_ref": "~/.automata/research/experiments",
  "artifact_refs": {
    "full_log": "logs/b200/probe/trial/attempt_001/run.log",
    "train_events": "events/b200/probe/trial/attempt_001/train_events.jsonl",
    "output_dir": "outputs/b200/probe/trial/attempt_001"
  }
}
```

The resolver expands these references at runtime when reading or writing large
artifacts.

### Heavy Artifact Garbage Collection

Heavy artifacts (e.g., checkpoints, tensorboard logs) can quickly consume local disk space, especially during large sweeps or capacity probes. To manage this:

- The system must provide a lightweight cleanup mechanism (e.g., `python -m research.runner gc`) to purge unreferenced or failed-probe heavy artifacts.
- By default, capacity probes that fail with expected OOMs should not retain any checkpoints.
- Artifact cleanup must only delete files under `RESEARCH_ARTIFACT_ROOT` and must not delete committed metadata.

## Per-Trial Artifact Contract

Each trial attempt writes under an attempt directory:

```text
experiments/runs/<profile>/<phase>/<trial>/
  attempt_001/
    intent.json
    resolved_config.json
    system.pre.json
    lifecycle.jsonl
    actions.jsonl
    system.post.json
    analysis.json
    analysis.md
```

`run.log`, `train_events.jsonl`, checkpoints, saved model shards, tokenizer
files, processor files, trainer state, and other model artifacts live under the
artifact root. The in-repo `analysis.json` links to them through relative
`artifact_refs`.

Retries and reruns must allocate a new `attempt_<N>` directory and must not
overwrite prior evidence. A convenience pointer such as `latest_attempt.txt` or
`latest` may be added, but readers must treat attempt directories as the
authoritative history.

Every JSON artifact and every JSONL record includes:

```json
{"schema_version": 1}
```

Readers must reject unsupported versions or fall back through an explicit
compatibility path. Schema migrations are handled in-memory by `analyzer.py`, which normalizes older schema versions (e.g., `v1`) into the current internal model during read operations. We do not require on-disk migration scripts that rewrite historical metadata.

### `intent.json`

The intended experiment before launch:

- `trial_id`, `profile`, `phase`, `trial`, `attempt`
- research question or intent label
- planned hyperparameters
- expected hardware profile
- expected failure class, if this is a capacity probe
- success criteria
- parent campaign id and workflow id when present

### `resolved_config.json`

The exact launch configuration:

- git commit
- trial-specific environment variables
- command argv with secrets redacted
- resolved model path
- dataset names
- output paths
- `artifact_root_ref`
- relative `artifact_refs`
- GPU count and distributed settings
- max steps, eval steps, save steps

### `system.pre.json` and `system.post.json`

System snapshots around the trial. Both snapshots include:

- GPU inventory, memory total, memory used, utilization
- driver and CUDA versions
- Python executable and version
- installed versions for torch, transformers, accelerate, deepspeed, peft, and
  temporalio when present
- important environment flags: CUDA, NCCL, HF cache/offline, Spack, venv
- available disk space for `RESEARCH_ARTIFACT_ROOT` and `HF_HOME` to diagnose ENOSPC failures

`system.post.json` also includes process topology and worker identity observed
during the run. `system.pre.json` does not include training worker topology
because the worker processes do not exist before launch.

### `lifecycle.jsonl`

Append-only events written by the supervisor:

```json
{"ts":"...","source":"supervisor","stage":"trial_started","message":"..."}
```

Required stages:

- `trial_started`
- `system_snapshot_pre`
- `launcher_started`
- `trainer_started`
- `first_step_seen`
- `eval_seen`
- `save_seen`
- `process_exited`
- `system_snapshot_post`
- `analysis_started`
- `analysis_finished`

Stages that do not occur should be inferable from `analysis.json`.

`lifecycle.jsonl` has single-writer semantics. Trainer processes must not append
to it directly. Trainer events go only to `train_events.jsonl`; the supervisor
may mirror selected trainer milestones into `lifecycle.jsonl` after reading or
observing them.

### `actions.jsonl`

Append-only supervisor decisions:

```json
{
  "ts": "...",
  "action": "mark_capacity_boundary",
  "reason": "oom during expected capacity probe",
  "input_evidence": ["analysis.json#/failure_reason"],
  "config_before": {"BATCH_SIZE": "8"},
  "config_after": null,
  "outcome": "continued"
}
```

Actions include retry, skip, cancel, select profile, mark capacity boundary,
stop sweep, rerun, or reduce capacity. Every action must include a reason and
evidence reference.

### `train_events.jsonl`

Structured trainer events:

- `trainer_started`
- `model_loaded`
- `data_module_ready`
- `train_step`
- `eval`
- `checkpoint_save_started`
- `checkpoint_save_finished`
- `trainer_footer`

Training metrics include step, loss, grad norm, learning rate, epoch, eval
loss, eval runtime, throughput, and rank/world metadata where available.

This file is optional during the migration phase. Until trainer instrumentation
is added, the analyzer must tolerate a missing or empty `train_events.jsonl` and
fall back to `run.log` and existing summary footers.

### `analysis.json`

Machine-readable post-trial diagnosis:

- status: `ok`, `crash`, `cancelled`
- failure reason
- root cause
- symptoms
- evidence references
- parsed metrics
- artifact root reference and relative artifact references
- lifecycle durations
- whether the failure was expected for the trial intent
- recommended next actions

### `analysis.md`

Human-readable diagnosis:

- what was attempted
- what happened
- observed symptoms
- likely root cause
- action taken by the supervisor
- recommended follow-up

## Data Models

### `TrialIntent`

```text
trial_id: str
attempt: int
profile: str
phase: str
trial: str
research_intent: str
planned_env: dict[str, str]
success_criteria: dict
expected_failure_reason: str | null
campaign_id: str | null
workflow_id: str | null
```

### `ResolvedTrialConfig`

```text
trial_id: str
attempt: int
git_commit: str
command: list[str]
env: dict[str, str]
model_path: str
datasets: list[str]
eval_datasets: list[str]
output_dir: str
run_dir: str
hardware_profile: str
distributed: dict
artifact_root_ref: str
artifact_refs: dict[str, str]
```

### `TrialAnalysis`

```text
trial_id: str
attempt: int
status: ok | crash | cancelled
failure_reason: none | oom | nccl | launcher_error | model_load_error |
  dataset_error | hub_or_cache_error | missing_footer | timeout | cancelled |
  hardware_error | unknown
root_cause: str
expected_failure: bool
metrics: TrialMetrics
symptoms: list[dict]
evidence_refs: list[str]
artifact_root_ref: str
artifact_refs: dict[str, str]
actions: list[dict]
recommendations: list[str]
artifact_dir: str  # repo-relative metadata attempt directory
```

`TrialMetrics` remains the compact metrics model produced by log parsing and is
not removed. `TrialAnalysis` wraps `TrialMetrics` with root-cause
classification, symptoms, actions, evidence references, and recommendations.

`append_result_row()` continues to write TSV rows by extracting the
backward-compatible fields from `TrialAnalysis.metrics`. During migration,
existing callers may still pass `TrialMetrics`; new supervisor paths should pass
`TrialAnalysis`. TSV log and output columns should store relative artifact
references for post-migration rows; readers resolve them through the artifact
root.

Workflow query compatibility is preserved by keeping `latest_metrics` as the
compact metrics payload and adding `latest_analysis` as a separate query field.
Local runner output keeps the current TSV behavior while also writing per-trial
analysis artifacts.

## Failure Classification

The analyzer classifies failures in two layers.

`failure_reason` is the observed failure category:

- `oom`
- `nccl`
- `launcher_error`
- `model_load_error`
- `dataset_error`
- `hub_or_cache_error`
- `missing_footer`
- `timeout`
- `cancelled`
- `hardware_error`
- `unknown`

`root_cause` explains the underlying diagnosis:

- `capacity_exceeded`
- `invalid_config`
- `dependency_missing`
- `dataset_unavailable`
- `model_snapshot_unavailable`
- `distributed_runtime_failure`
- `trainer_instrumentation_gap`
- `operator_cancelled`
- `hardware_fault`
- `unknown`

Default mapping:

| `failure_reason` | Default `root_cause` | Expected in capacity probe? |
| --- | --- | --- |
| `none` | `none` | yes |
| `oom` | `capacity_exceeded` | yes, if the intent marks OOM as expected |
| `nccl` | `distributed_runtime_failure` | no |
| `launcher_error` | `dependency_missing` or `invalid_config` after symptom analysis | no |
| `model_load_error` | `model_snapshot_unavailable` | no |
| `dataset_error` | `dataset_unavailable` | no |
| `hub_or_cache_error` | `model_snapshot_unavailable` | no |
| `missing_footer` | `trainer_instrumentation_gap` | no |
| `timeout` | `distributed_runtime_failure` | no |
| `cancelled` | `operator_cancelled` | no |
| `hardware_error` | `hardware_fault` | no |
| `unknown` | `unknown` | no |

The analyzer may refine the default root cause when stronger evidence is
available. For example, a launcher error containing an unknown dataset name
should become `dataset_unavailable`, while a launcher error containing an import
failure should become `dependency_missing`.

For example, a B200 `BATCH_SIZE=8` OOM during a probe is:

```text
failure_reason=oom
root_cause=capacity_exceeded
expected_failure=true
action=mark_capacity_boundary
```

## Temporal Execution Model

The first implementation uses one activity per trial attempt:

```text
ProfiledExperimentWorkflow
  load_profile_activity
  plan_probe_trials_activity
  run_trial_attempt_activity(trial 1)
  run_trial_attempt_activity(trial 2)
  ...
  select_capacity_activity
  plan_sweep_trials_activity
  run_trial_attempt_activity(sweep trial 1)
  summarize_campaign_activity
```

The workflow records compact results and exposes them through queries. The
activity writes detailed artifacts.

## Local Temporal Development Mode

Local commands run through Temporal, not through a direct in-process campaign
loop. The local runner owns a development harness:

```text
python -m research.runner probe --profile b200 --dry-run
  -> ensure local Temporal server is reachable
  -> if not reachable, start:
     temporal server start-dev --db-filename ~/.automata/research/temporal/temporal.db
  -> start a local worker on task queue qwen3vl-local
  -> start ProfiledExperimentWorkflow(command="probe")
  -> wait for workflow completion
```

The local Temporal database lives under:

```text
~/.automata/research/temporal/temporal.db
```

The runner must create the parent directory before starting the dev server
because current Temporal CLI versions do not guarantee parent directory
creation for `--db-filename`. To prevent race conditions where multiple local runners attempt to start the server simultaneously, the runner must acquire an OS-level file lock (e.g., `temporal.lock`) before checking reachability and bootstrapping the server.

The Temporal database path is resolved at runtime. `RESEARCH_TEMPORAL_DB` may
override it, including to a repo-local `.artifacts/temporal/temporal.db` for
project-local development. If `.artifacts/` is used, it must be ignored by Git.
Metadata artifacts must not reference the absolute SQLite path.

If `temporal` is not on `PATH`, local campaign commands fail fast with a message
that names the missing executable and the command that would have been run.
There is no silent fallback to the old direct runner.

The local harness may expose a narrow developer command for unit tests or
debugging a single activity implementation, but normal user-facing experiment
commands use Temporal.

## Subprocess Lifecycle

The trial activity launches the trainer with `subprocess.Popen`:

```python
proc = subprocess.Popen(
    command,
    cwd=root,
    env=resolved_env,
    stdout=artifact_log_file,
    stderr=subprocess.STDOUT,
    text=True,
    start_new_session=True,
)
```

`start_new_session=True` creates a process group for `torchrun` and all worker
processes. Cancellation, timeout, and teardown use the process group:

```python
pgid = os.getpgid(proc.pid)
os.killpg(pgid, signal.SIGTERM)
```

After `SIGTERM`, the supervisor waits for a bounded grace period. If any process
in the group remains, it sends `SIGKILL` with `os.killpg(pgid, signal.SIGKILL)`.
After process termination, the supervisor polls `nvidia-smi` until GPU memory
returns to the pre-run baseline within a configured tolerance, or records
`gpu_memory_not_released` as a symptom in `analysis.json`. To prevent indefinite hangs caused by bad driver states, this polling loop must enforce a strict timeout (e.g., 60 seconds). If the timeout is reached, the supervisor records a `gpu_driver_hang` symptom and proceeds.

The supervisor never kills only the parent `torchrun` process.

## Timeout and Stall Detection

Temporal activity timeouts are a final ceiling, not the primary hang detector.
The supervisor enforces application-level watchdogs:

- `max_trial_wall_time_sec`: hard upper bound for a trial attempt
- `stall_timeout_sec`: maximum time with no artifact-root `run.log` growth and
  no artifact-root `train_events.jsonl` growth after trainer start
- `startup_timeout_sec`: maximum time from launch to `trainer_started` or first
  meaningful log output

When a watchdog fires, the supervisor records a `timeout` failure, terminates
the process group, captures post-run system state, and writes analysis before
returning to the workflow.

## Retry and Cancellation Policy

Training failures are not automatically retried unless classified as transient.

Do not retry:

- OOM capacity failures
- invalid config
- dataset schema/config errors
- deterministic model load errors
- missing footer after successful return code

Retry may be allowed with action records:

- master port collision
- Hub 429 or temporary network/cache failure
- temporary filesystem failure
- worker interruption before trainer launch

Retries must be explicit in `actions.jsonl`, with before/after config or
environment changes when applicable.

Each retry is a new attempt directory. The workflow records the relationship
between attempts by trial id and attempt number.

Cancellation must terminate the full process group. The activity writes:

```text
status=cancelled
failure_reason=cancelled
root_cause=operator_cancelled
```

and records the cancellation source in `actions.jsonl`.

## Shared Supervisor Compatibility

The Temporal activity calls the shared supervisor code:

```text
SupervisorTrialRunner.run(intent, config, dry_run=False) -> TrialAnalysis
```

Convenience factories build these inputs from existing `TrialSpec` values:

```text
TrialIntent.from_spec(spec, campaign_context=None)
ResolvedTrialConfig.from_spec(root, spec, attempt)
```

This preserves one implementation for local Temporal, remote Temporal, and
activity-level tests. The supervisor can be invoked directly by tests, but not
by normal local campaign commands.

## Post-Run Aggregation

Campaign-level analysis reads all `analysis.json` files plus
`results.profiled.tsv` and produces:

- safe capacity boundary
- fastest safe configuration
- best validation-loss configuration
- repeated failure patterns
- unexpected infrastructure failures
- action history
- recommended next campaign

`analysis.json` is authoritative for post-migration trials. The TSV remains a
secondary compatibility output used by existing selection code and older
scripts. If `analysis.json` and TSV disagree for the same attempt, the
summarizer reports the disagreement and uses `analysis.json`. For pre-migration
trials without analysis artifacts, the summarizer falls back to TSV rows and raw
logs.

The aggregation should explicitly distinguish:

- capacity findings, such as `bs=8 exceeded memory`
- quality findings, such as lower validation loss
- operational findings, such as missing diagnostics or launcher issues

## Implementation Modules

Add shared modules under `qwen-vl-finetune/research/observability/`:

```text
schema.py
artifact_store.py
artifacts.py
system_snapshot.py
event_writer.py
trial_runner.py
log_parser.py
analyzer.py
policy.py
summarizer.py
```

`research.runner` should call `trial_runner.py` for local runs.
`research.activities.run_trial_activity` should call the same `trial_runner.py`
for Temporal runs.

Trainer instrumentation should be small and optional:

- if `RESEARCH_EVENTS_PATH` is set, append JSONL events
- if not set, continue current behavior
- always keep existing stdout/stderr logs and summary footers

Redaction utilities live in the shared observability layer. Secret values are
redacted when keys match `TOKEN`, `SECRET`, `PASSWORD`, `API_KEY`, or
`CREDENTIAL` case-insensitively. Redacted values are replaced with
`***REDACTED***`. `resolved_config.json` records trial-specific environment
overrides and selected safe runtime metadata, not the full inherited process
environment.

Artifact storage utilities live in `artifact_store.py`. They resolve
`RESEARCH_ARTIFACT_ROOT` at runtime, default to
`~/.automata/research/experiments/`, create local artifact directories, and
convert paths to relative references before writing metadata. They reject
metadata payloads that contain local absolute artifact paths.

## Testing

Unit tests:

- schema serialization and redaction
- artifact root resolution and relative reference generation
- artifact path creation
- system snapshot command fallback behavior
- lifecycle/action JSONL append behavior
- log parser failure classification
- analyzer root-cause classification
- local dry-run artifact creation
- process group teardown with a synthetic subprocess tree
- GPU memory release polling behavior with `nvidia-smi` mocked
- metadata absolute-path guard for `intent.json`, `resolved_config.json`, and
  `analysis.json`

Integration tests:

- dry-run local trial writes all mandatory non-trainer artifacts
- failed synthetic log produces `analysis.json` with expected root cause
- Temporal modules import with mandatory `temporalio`
- Temporal activity dry run returns `TrialAnalysis`
- cancellation of a synthetic long-running trial terminates the whole process
  group and records `failure_reason=cancelled`

Manual verification:

- run a B200 dry-run probe and inspect artifacts
- verify heavy dry-run artifacts are under `RESEARCH_ARTIFACT_ROOT` or
  `~/.automata/research/experiments/`, not under `experiments/runs/...`
- run one short real trial and verify trainer events and analysis
- cancel an activity and verify process group teardown and cancellation
  analysis

## Migration Plan

1. Introduce observability schemas and artifact writer.
2. Introduce artifact root resolver and Git ignore rules for local heavy
   artifacts.
3. Add supervisor trial runner that wraps the existing launcher.
4. Update local runner to manage a local Temporal dev server and start workflows.
5. Add trainer JSONL event emission behind `RESEARCH_EVENTS_PATH`.
6. Expand analyzer beyond current regex-only metrics parsing.
7. Update Temporal activity to return `TrialAnalysis`.
8. Update workflow queries to expose latest trial analysis without removing
   `latest_metrics`.
9. Add pause, stop, cancel, skip, and rerun signals with attempt tracking.
10. Add campaign summarizer based on `analysis.json` files.
11. Remove the direct local campaign path after Temporal-backed dry runs pass.

This keeps the current B200 profile runner usable throughout the migration.
