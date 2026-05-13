# B200 Real Probe Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch the real B200 capacity probe, select the best passing B200 capacity profile, summarize the result, and stop before any full sweep.

**Architecture:** Use the committed local runner in `qwen-vl-finetune/research/` rather than Temporal for the first real GPU run. Preserve legacy `experiments/results.tsv`; write profiled metrics to `experiments/results.profiled.tsv`; capture launcher output in `experiments/b200_probe.nohup.log` with a PID file.

**Tech Stack:** Bash, `nvidia-smi`, existing `research/run_profiled.sh`, Qwen3-VL-8B launcher `scripts/sft_qwen3_8b.sh`.

---

## File Structure

Files read:

- `qwen-vl-finetune/research/profiles/b200.env`: B200 probe grid and guardrails.
- `qwen-vl-finetune/research/run_profiled.sh`: local runner wrapper.
- `qwen-vl-finetune/scripts/sft_qwen3_8b.sh`: trainer launcher used by every probe trial.

Files created or appended by the real run:

- `qwen-vl-finetune/experiments/b200_probe.nohup.log`
- `qwen-vl-finetune/experiments/b200_probe.pid`
- `qwen-vl-finetune/experiments/results.profiled.tsv`
- `qwen-vl-finetune/experiments/runs/b200/probe/<trial>/run.log`
- `qwen-vl-finetune/experiments/runs/b200/probe/<trial>/output/`
- `qwen-vl-finetune/research/selected/b200.env`
- `qwen-vl-finetune/experiments/summary.md`

Files intentionally not modified:

- `qwen-vl-finetune/experiments/results.tsv`
- trainer source files under `qwen-vl-finetune/qwenvl/`
- launcher scripts except normal execution of `scripts/sft_qwen3_8b.sh`

## Task 1: Preflight Checks

**Files:**
- Read: `qwen-vl-finetune/research/profiles/b200.env`
- Read: `qwen-vl-finetune/research/run_profiled.sh`
- Read: `qwen-vl-finetune/research/lib/common.sh`

- [ ] **Step 1: Confirm B200 GPUs**

Run:

```bash
nvidia-smi --list-gpus
```

Expected: eight lines, each containing `NVIDIA B200`.

- [ ] **Step 2: Confirm launcher and shell syntax**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl
bash -n qwen-vl-finetune/research/run_profiled.sh
bash -n qwen-vl-finetune/research/lib/common.sh
test -f qwen-vl-finetune/scripts/sft_qwen3_8b.sh
```

Expected: exit code `0`, no shell syntax output, launcher file exists.

- [ ] **Step 3: Confirm no active B200 probe is already running**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
if [[ -f experiments/b200_probe.pid ]] && kill -0 "$(cat experiments/b200_probe.pid)" 2>/dev/null; then
  echo "ACTIVE $(cat experiments/b200_probe.pid)"
  exit 1
else
  echo "NO_ACTIVE_B200_PROBE"
fi
```

Expected: `NO_ACTIVE_B200_PROBE`.

## Task 2: Launch Probe

**Files:**
- Create: `qwen-vl-finetune/experiments/b200_probe.nohup.log`
- Create: `qwen-vl-finetune/experiments/b200_probe.pid`
- Append: `qwen-vl-finetune/experiments/results.profiled.tsv`
- Create: `qwen-vl-finetune/experiments/runs/b200/probe/`

- [ ] **Step 1: Start the real B200 probe**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
mkdir -p experiments
nohup bash research/run_profiled.sh probe --profile b200 \
  > experiments/b200_probe.nohup.log 2>&1 &
echo $! > experiments/b200_probe.pid
cat experiments/b200_probe.pid
```

Expected: prints a numeric PID.

- [ ] **Step 2: Confirm the probe process is alive**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
kill -0 "$(cat experiments/b200_probe.pid)"
```

Expected: exit code `0`.

- [ ] **Step 3: Confirm first run directory starts appearing**

Run after 30-60 seconds:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
find experiments/runs/b200/probe -maxdepth 2 -name run.log | sort | head
tail -n 40 experiments/b200_probe.nohup.log
```

Expected: at least one probe `run.log` appears, or launcher startup logs are visible in `b200_probe.nohup.log`.

## Task 3: Monitor Probe

**Files:**
- Read: `qwen-vl-finetune/experiments/b200_probe.nohup.log`
- Read: `qwen-vl-finetune/experiments/results.profiled.tsv`
- Read: `qwen-vl-finetune/experiments/runs/b200/probe/*/run.log`

- [ ] **Step 1: Check current progress**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
echo "pid=$(cat experiments/b200_probe.pid)"
if kill -0 "$(cat experiments/b200_probe.pid)" 2>/dev/null; then echo RUNNING; else echo EXITED; fi
find experiments/runs/b200/probe -maxdepth 2 -name run.log | wc -l
tail -n 5 experiments/results.profiled.tsv 2>/dev/null || true
```

Expected while running: `RUNNING`, a growing count of probe logs, and rows appearing after completed trials.

- [ ] **Step 2: Watch for hard failures**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
grep -RniE "out of memory|cuda oom|nccl.*(error|timeout|watchdog)|traceback|error:" \
  experiments/runs/b200/probe/*/run.log experiments/b200_probe.nohup.log 2>/dev/null | tail -n 40 || true
```

Expected: either no output, or failures isolated to individual probe trials. The runner should continue after trial crashes.

- [ ] **Step 3: Wait for process completion**

Run periodically:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
if kill -0 "$(cat experiments/b200_probe.pid)" 2>/dev/null; then
  echo RUNNING
else
  echo EXITED
fi
```

Expected final state: `EXITED`.

## Task 4: Select B200 Capacity

**Files:**
- Read: `qwen-vl-finetune/experiments/results.profiled.tsv`
- Create: `qwen-vl-finetune/research/selected/b200.env`

- [ ] **Step 1: Run selection**

Run after the probe process exits:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
bash research/run_profiled.sh select --profile b200
```

Expected: prints the path to `research/selected/b200.env`.

- [ ] **Step 2: Inspect selected capacity**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
cat research/selected/b200.env
```

Expected: contains concrete values for `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `GRADIENT_CHECKPOINTING`, `MAX_PIXELS`, and `MODEL_MAX_LENGTH`.

- [ ] **Step 3: If selection fails**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
awk -F'\t' 'NR==1 || $3=="b200" {print}' experiments/results.profiled.tsv | tail -n 40
```

Expected: enough rows to diagnose whether every probe crashed or exceeded the VRAM ceiling. Do not set `ALLOW_PROFILE_FALLBACK=1` in this run.

## Task 5: Summarize and Stop

**Files:**
- Create: `qwen-vl-finetune/experiments/summary.md`
- Read: `qwen-vl-finetune/research/selected/b200.env`
- Read: `qwen-vl-finetune/experiments/results.profiled.tsv`

- [ ] **Step 1: Generate summary**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
bash research/run_profiled.sh summarize --profile b200
```

Expected: prints `.../qwen-vl-finetune/experiments/summary.md`.

- [ ] **Step 2: Report final probe state**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
echo "Selected B200 capacity:"
cat research/selected/b200.env
echo
echo "Last profiled rows:"
tail -n 10 experiments/results.profiled.tsv
echo
echo "Summary:"
sed -n '1,160p' experiments/summary.md
```

Expected: selected capacity, last probe rows, and summary are visible.

- [ ] **Step 3: Stop before sweep**

Do not run:

```bash
bash research/run_profiled.sh sweep --profile b200
```

Expected: full sweep remains unstarted until separately approved.

## Self-Review Notes

- Spec coverage: preflight, real probe launch, monitoring, selection, summary, and stop-before-sweep are covered.
- Placeholder scan: no TBD/TODO/fill-later language remains.
- Type consistency: commands use the committed `research/run_profiled.sh` interface and `b200` profile throughout.
