# B200 Real Probe Run Design

## Context

The B200 research orchestration has been implemented under
`qwen-vl-finetune/research/` and committed in `2545c99`. The machine currently
shows eight NVIDIA B200 GPUs. The existing `qwen-vl-finetune/experiments/`
directory already contains legacy run artifacts and a legacy `results.tsv`
schema, so the new profiled runner must preserve that file and write profiled
metrics to `experiments/results.profiled.tsv`.

System Python does not currently have `temporalio` or `pytest` installed. The
local non-Temporal runner is sufficient for the first real capacity probe, so
Temporal setup is deferred until after the B200 capacity envelope is known.

## Goal

Run the real B200 capacity probe only, select the fastest successful config
under the B200 VRAM ceiling, and stop before launching the full sweep.

## Run Shape

Use the local runner from `qwen-vl-finetune/`:

```bash
nohup bash research/run_profiled.sh probe --profile b200 \
  > experiments/b200_probe.nohup.log 2>&1 &
echo $! > experiments/b200_probe.pid
```

The B200 profile grid is:

- `batch_size`: `2,4,8`
- `gradient_accumulation_steps`: `2,4`
- `gradient_checkpointing`: `True,False`
- `max_pixels`: `50176,112896`
- `model_max_length`: `8192`
- `MAX_STEPS=50`, `EVAL_STEPS=25`, `SAVE_STEPS=50`

This produces 24 short probe trials. Each trial uses
`scripts/sft_qwen3_8b.sh` and writes:

```text
qwen-vl-finetune/experiments/runs/b200/probe/<trial>/run.log
qwen-vl-finetune/experiments/runs/b200/probe/<trial>/output/
```

Metrics are appended to:

```text
qwen-vl-finetune/experiments/results.profiled.tsv
```

The legacy `experiments/results.tsv` remains untouched.

## Selection

After the probe process exits, run:

```bash
bash research/run_profiled.sh select --profile b200
bash research/run_profiled.sh summarize --profile b200
```

Selection writes:

```text
qwen-vl-finetune/research/selected/b200.env
```

The selection rule is the committed runner rule:

1. Keep successful probe rows with parseable `val_loss` and `peak_vram_mb`.
2. Reject rows above `B200_VRAM_CEILING_MB=160000`.
3. Prefer higher throughput.
4. Break ties by lower validation loss.
5. Break remaining ties by lower VRAM.

## Monitoring

Use these commands:

```bash
tail -f qwen-vl-finetune/experiments/b200_probe.nohup.log
tail -n 5 qwen-vl-finetune/experiments/results.profiled.tsv
find qwen-vl-finetune/experiments/runs/b200/probe -maxdepth 2 -name run.log | wc -l
```

If the probe is still running, the PID is stored at:

```text
qwen-vl-finetune/experiments/b200_probe.pid
```

## Stop Conditions

Do not start `sweep` automatically. Stop after:

1. `research/selected/b200.env` exists.
2. `experiments/summary.md` exists.
3. Probe crashes, if any, are visible in `results.profiled.tsv`.
4. The selected config is reported back for review.

If every probe trial fails, do not use the conservative fallback automatically.
Report the failure reasons and leave `research/selected/b200.env` absent unless
the operator explicitly sets `ALLOW_PROFILE_FALLBACK=1` in a later run.

## Verification

Before starting:

```bash
nvidia-smi --list-gpus
bash -n qwen-vl-finetune/research/run_profiled.sh
bash -n qwen-vl-finetune/research/lib/common.sh
```

After completion:

```bash
test -f qwen-vl-finetune/experiments/results.profiled.tsv
test -f qwen-vl-finetune/research/selected/b200.env
test -f qwen-vl-finetune/experiments/summary.md
cat qwen-vl-finetune/research/selected/b200.env
```
