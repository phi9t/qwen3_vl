# Qwen3-VL Research Orchestration

`research/` contains reusable experiment mechanics and Temporal orchestration.
`experiments/` contains concrete run artifacts: logs, metrics, summaries, and checkpoints.

## Local Dry Run

```bash
cd qwen-vl-finetune
PYTHONPATH=. bash research/run_profiled.sh probe --profile b200 --dry-run
PYTHONPATH=. bash research/run_profiled.sh select --profile b200
```

## Subagent Command Preview

```bash
cd qwen-vl-finetune
PYTHONPATH=. python -m research.runner subagents --provider all
PYTHONPATH=. python -m research.runner subagents --provider coco
PYTHONPATH=. python -m research.runner subagents --provider cursor_agent
```

The preview prints Trae `coco` and Cursor `agent` commands when those CLIs are
available on PATH. It does not launch them.

## Temporal Dry Run

Start Temporal dev server:

```bash
temporal server start-dev
```

Start worker:

```bash
cd qwen-vl-finetune
PYTHONPATH=. python -m research.worker --task-queue qwen3vl-b200
```

Start workflow:

```bash
cd qwen-vl-finetune
PYTHONPATH=. python -m research.start --profile b200 --dry-run
```

## Real Probe

```bash
cd qwen-vl-finetune
bash research/run_profiled.sh probe --profile b200
bash research/run_profiled.sh select --profile b200
```

The runner always launches `scripts/sft_qwen3_8b.sh` for this campaign.
