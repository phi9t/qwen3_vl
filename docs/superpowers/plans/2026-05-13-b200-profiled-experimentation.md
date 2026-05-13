# B200 Profiled Experimentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Use Codex subagents for review/integration work, and use external implementation subagents through Trae `coco` and Cursor `agent` CLI adapters where assigned below. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a B200-first, profile-aware experiment runner with Temporal orchestration under `qwen-vl-finetune/research/`, while keeping `qwen-vl-finetune/experiments/` as the trial artifact store.

**Architecture:** Put reusable research mechanics in a small Python package plus thin shell wrappers. Pure Python modules own profile parsing, trial planning, log parsing, metric selection, subagent command construction, and summary generation; shell scripts launch the Qwen3-VL-8B trainer; Temporal workflows orchestrate activities without touching the filesystem directly.

**Tech Stack:** Python 3 stdlib dataclasses/argparse/subprocess/shutil, pytest, Bash, Temporal Python SDK (`temporalio`) with graceful fallback import guidance for environments that do not have Temporal installed, Trae `coco` CLI, Cursor `agent` CLI.

---

## File Structure

Create these files:

- `qwen-vl-finetune/research/__init__.py`: package marker.
- `qwen-vl-finetune/research/models.py`: dataclasses and constants for profiles, trial specs, metrics, and statuses.
- `qwen-vl-finetune/research/profile_config.py`: `.env` profile parser and probe-grid expansion.
- `qwen-vl-finetune/research/metrics.py`: log footer parsing, throughput extraction, and failure classification.
- `qwen-vl-finetune/research/selection.py`: B200 capacity selection and selected-profile writer.
- `qwen-vl-finetune/research/subagents.py`: external subagent provider detection and command construction for `coco` and `agent`.
- `qwen-vl-finetune/research/runner.py`: local non-Temporal probe/select/summarize CLI implementation.
- `qwen-vl-finetune/research/activities.py`: Temporal activities, including subprocess execution and artifact writes.
- `qwen-vl-finetune/research/workflows.py`: deterministic Temporal workflow definitions.
- `qwen-vl-finetune/research/worker.py`: Temporal worker process.
- `qwen-vl-finetune/research/start.py`: Temporal workflow starter CLI.
- `qwen-vl-finetune/research/run_profiled.sh`: shell entrypoint that invokes `runner.py`.
- `qwen-vl-finetune/research/lib/common.sh`: small shell helper library for path resolution and launcher validation.
- `qwen-vl-finetune/research/profiles/b200.env`: active B200 profile.
- `qwen-vl-finetune/research/profiles/a100.env`: disabled future profile stub.
- `qwen-vl-finetune/research/profiles/h100.env`: disabled future profile stub.
- `qwen-vl-finetune/research/README.md`: usage and boundary documentation.
- `qwen-vl-finetune/tests/research/test_profile_config.py`
- `qwen-vl-finetune/tests/research/test_metrics.py`
- `qwen-vl-finetune/tests/research/test_selection.py`
- `qwen-vl-finetune/tests/research/test_subagents.py`
- `qwen-vl-finetune/tests/research/test_runner_dry_run.py`
- `qwen-vl-finetune/tests/research/test_workflows_import.py`

Modify these files only if needed:

- `qwen-vl-finetune/experiments/results.tsv`: do not edit existing data rows in implementation. New runs may append rows later.
- Do not modify `qwen-vl-finetune/scripts/sft_qwen3_8b.sh` unless verification shows it is missing an env override needed by the runner.

## Subagent Execution Strategy

Implementation should be split across subagents with disjoint ownership:

- **Codex integration agent:** owns final review, conflict resolution, and cross-task verification.
- **Trae Coco worker:** use `coco -p -y --worktree <name> "<prompt>"` for a focused code task when `coco` is available.
- **Cursor Agent worker:** use `agent -p --trust --workspace <path> -w <name> "<prompt>"` for a focused code task when `agent` is available.

Provider rules:

- The implementation must include a `research/subagents.py` adapter that detects `coco` and `agent` with `shutil.which`.
- External subagents are optional at runtime. If a CLI is unavailable, the runner reports it as unavailable and does not fail unrelated local operations.
- Prompts must include file ownership, expected tests, and the instruction not to revert unrelated changes.
- Use external subagents for coding/review support only. GPU trial execution still happens through `scripts/sft_qwen3_8b.sh` and the Temporal/local runner, not through an LLM CLI.
- Do not run multiple external subagents against overlapping write sets.

Recommended assignment:

- Task 1 and Task 2: Trae Coco worker, because they are bounded file creation tasks.
- Task 3 and Task 4: Cursor Agent worker, because metrics and subagent adapters are isolated pure Python modules.
- Task 5: Codex integration agent, because runner wiring touches profile parsing, metrics, shell, and artifact layout.
- Task 6 and Task 7: Trae Coco or Cursor Agent worker with Codex review, because Temporal import boundaries are easy to inspect but must be checked carefully for deterministic workflow behavior.
- Task 8 and Task 9: Codex integration agent, because they are verification and GPU handoff tasks.

Example external dispatch commands from the repo root:

```bash
coco -p -y --worktree b200-research-profiles "Implement Tasks 1-2 from docs/superpowers/plans/2026-05-13-b200-profiled-experimentation.md. Own only qwen-vl-finetune/research/models.py, qwen-vl-finetune/research/profile_config.py, qwen-vl-finetune/research/profiles/, and qwen-vl-finetune/tests/research/test_profile_config.py. Do not revert unrelated changes. Run: cd qwen-vl-finetune && PYTHONPATH=. pytest tests/research/test_profile_config.py -v"

agent -p --trust --workspace /data02/home/philip.yang/workspace/qwen3_vl -w b200-research-metrics "Implement Tasks 3-4 from docs/superpowers/plans/2026-05-13-b200-profiled-experimentation.md. Own only qwen-vl-finetune/research/metrics.py, qwen-vl-finetune/research/selection.py, qwen-vl-finetune/research/subagents.py, qwen-vl-finetune/tests/research/test_metrics.py, qwen-vl-finetune/tests/research/test_selection.py, and qwen-vl-finetune/tests/research/test_subagents.py. Do not revert unrelated changes. Run: cd qwen-vl-finetune && PYTHONPATH=. pytest tests/research/test_metrics.py tests/research/test_selection.py tests/research/test_subagents.py -v"
```

## Task 1: Models and Package Skeleton

**Files:**
- Create: `qwen-vl-finetune/research/__init__.py`
- Create: `qwen-vl-finetune/research/models.py`
- Create: `qwen-vl-finetune/tests/research/test_profile_config.py`

- [ ] **Step 1: Write failing tests for basic model imports**

Create `qwen-vl-finetune/tests/research/test_profile_config.py`:

```python
from pathlib import Path

from research.models import FailureReason, TrialMetrics, TrialSpec


def test_trial_spec_derives_run_paths() -> None:
    spec = TrialSpec(
        profile="b200",
        phase="probe",
        trial="probe_bs2_ga4_gcTrue_px50176",
        env={
            "BATCH_SIZE": "2",
            "GRAD_ACCUM_STEPS": "4",
            "GRADIENT_CHECKPOINTING": "True",
            "MAX_PIXELS": "50176",
            "MODEL_MAX_LENGTH": "8192",
        },
    )

    run_dir = spec.run_dir(Path("/repo/qwen-vl-finetune"))

    assert run_dir == Path(
        "/repo/qwen-vl-finetune/experiments/runs/b200/probe/probe_bs2_ga4_gcTrue_px50176"
    )
    assert spec.log_path(Path("/repo/qwen-vl-finetune")).name == "run.log"


def test_trial_metrics_success_requires_footer_values() -> None:
    metrics = TrialMetrics(
        status="ok",
        val_loss=0.56,
        peak_vram_mb=95280.6,
        throughput_steps_per_sec=0.31,
        failure_reason=FailureReason.NONE,
    )

    assert metrics.is_successful
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_profile_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'research'`.

- [ ] **Step 3: Create package skeleton and models**

Create `qwen-vl-finetune/research/__init__.py`:

```python
"""Research orchestration utilities for Qwen3-VL experimentation."""
```

Create `qwen-vl-finetune/research/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class FailureReason(StrEnum):
    NONE = "none"
    OOM = "oom"
    NCCL = "nccl"
    MISSING_FOOTER = "missing_footer"
    LAUNCHER_ERROR = "launcher_error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    enabled: bool
    vram_ceiling_mb: float
    max_steps: int
    eval_steps: int
    save_steps: int
    datasets: str
    eval_datasets: str
    batch_sizes: tuple[int, ...]
    grad_accum_steps: tuple[int, ...]
    gradient_checkpointing: tuple[bool, ...]
    max_pixels: tuple[int, ...]
    model_max_lengths: tuple[int, ...]


@dataclass(frozen=True)
class TrialSpec:
    profile: str
    phase: str
    trial: str
    env: dict[str, str] = field(default_factory=dict)

    def run_dir(self, root_dir: Path) -> Path:
        return root_dir / "experiments" / "runs" / self.profile / self.phase / self.trial

    def log_path(self, root_dir: Path) -> Path:
        return self.run_dir(root_dir) / "run.log"

    def output_dir(self, root_dir: Path) -> Path:
        return self.run_dir(root_dir) / "output"


@dataclass(frozen=True)
class TrialMetrics:
    status: str
    val_loss: float | None = None
    peak_vram_mb: float | None = None
    throughput_steps_per_sec: float | None = None
    failure_reason: FailureReason = FailureReason.NONE

    @property
    def is_successful(self) -> bool:
        return (
            self.status == "ok"
            and self.val_loss is not None
            and self.peak_vram_mb is not None
            and self.failure_reason == FailureReason.NONE
        )
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_profile_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qwen-vl-finetune/research/__init__.py qwen-vl-finetune/research/models.py qwen-vl-finetune/tests/research/test_profile_config.py
git commit -m "feat: add research model types"
```

## Task 2: Profile Parser and Probe Planning

**Files:**
- Create: `qwen-vl-finetune/research/profile_config.py`
- Create: `qwen-vl-finetune/research/profiles/b200.env`
- Create: `qwen-vl-finetune/research/profiles/a100.env`
- Create: `qwen-vl-finetune/research/profiles/h100.env`
- Modify: `qwen-vl-finetune/tests/research/test_profile_config.py`

- [ ] **Step 1: Extend tests for profile parsing and probe grid expansion**

Append to `qwen-vl-finetune/tests/research/test_profile_config.py`:

```python
from research.profile_config import load_profile, plan_probe_trials


def test_load_profile_parses_b200_env(tmp_path: Path) -> None:
    profile_path = tmp_path / "b200.env"
    profile_path.write_text(
        "\n".join(
            [
                "PROFILE_NAME=b200",
                "PROFILE_ENABLED=true",
                "B200_VRAM_CEILING_MB=160000",
                "MAX_STEPS=50",
                "EVAL_STEPS=25",
                "SAVE_STEPS=50",
                "DATASETS=visualwebinstruct_train",
                "EVAL_DATASETS=visualwebinstruct_val",
                "PROBE_BATCH_SIZES=2,4",
                "PROBE_GRAD_ACCUM_STEPS=2,4",
                "PROBE_GRADIENT_CHECKPOINTING=True,False",
                "PROBE_MAX_PIXELS=50176",
                "PROBE_MODEL_MAX_LENGTHS=8192",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profile = load_profile(profile_path)

    assert profile.name == "b200"
    assert profile.enabled is True
    assert profile.vram_ceiling_mb == 160000
    assert profile.batch_sizes == (2, 4)
    assert profile.gradient_checkpointing == (True, False)


def test_plan_probe_trials_expands_grid(tmp_path: Path) -> None:
    profile_path = tmp_path / "b200.env"
    profile_path.write_text(
        "\n".join(
            [
                "PROFILE_NAME=b200",
                "PROFILE_ENABLED=true",
                "B200_VRAM_CEILING_MB=160000",
                "MAX_STEPS=50",
                "EVAL_STEPS=25",
                "SAVE_STEPS=50",
                "DATASETS=visualwebinstruct_train",
                "EVAL_DATASETS=visualwebinstruct_val",
                "PROBE_BATCH_SIZES=2,4",
                "PROBE_GRAD_ACCUM_STEPS=2",
                "PROBE_GRADIENT_CHECKPOINTING=True,False",
                "PROBE_MAX_PIXELS=50176",
                "PROBE_MODEL_MAX_LENGTHS=8192",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    profile = load_profile(profile_path)

    trials = plan_probe_trials(profile)

    assert len(trials) == 4
    assert trials[0].env["MODEL_NAME_OR_PATH"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert trials[0].env["DATASETS"] == "visualwebinstruct_train"
    assert trials[0].phase == "probe"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_profile_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'research.profile_config'`.

- [ ] **Step 3: Implement profile parsing and grid planning**

Create `qwen-vl-finetune/research/profile_config.py`:

```python
from __future__ import annotations

from itertools import product
from pathlib import Path

from research.models import HardwareProfile, TrialSpec


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid profile line in {path}: {raw!r}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _get(values: dict[str, str], key: str) -> str:
    try:
        return values[key]
    except KeyError as exc:
        raise ValueError(f"Missing required profile key: {key}") from exc


def _bool(value: str) -> bool:
    if value.lower() in {"1", "true", "yes"}:
        return True
    if value.lower() in {"0", "false", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _bool_tuple(value: str) -> tuple[bool, ...]:
    return tuple(_bool(part.strip()) for part in value.split(",") if part.strip())


def load_profile(path: str | Path) -> HardwareProfile:
    profile_path = Path(path)
    values = _parse_env(profile_path)
    return HardwareProfile(
        name=_get(values, "PROFILE_NAME"),
        enabled=_bool(_get(values, "PROFILE_ENABLED")),
        vram_ceiling_mb=float(_get(values, "B200_VRAM_CEILING_MB")),
        max_steps=int(_get(values, "MAX_STEPS")),
        eval_steps=int(_get(values, "EVAL_STEPS")),
        save_steps=int(_get(values, "SAVE_STEPS")),
        datasets=_get(values, "DATASETS"),
        eval_datasets=_get(values, "EVAL_DATASETS"),
        batch_sizes=_int_tuple(_get(values, "PROBE_BATCH_SIZES")),
        grad_accum_steps=_int_tuple(_get(values, "PROBE_GRAD_ACCUM_STEPS")),
        gradient_checkpointing=_bool_tuple(_get(values, "PROBE_GRADIENT_CHECKPOINTING")),
        max_pixels=_int_tuple(_get(values, "PROBE_MAX_PIXELS")),
        model_max_lengths=_int_tuple(_get(values, "PROBE_MODEL_MAX_LENGTHS")),
    )


def plan_probe_trials(profile: HardwareProfile) -> list[TrialSpec]:
    if not profile.enabled:
        raise ValueError(f"Profile {profile.name!r} is disabled")

    trials: list[TrialSpec] = []
    for batch, accum, checkpointing, max_pixels, model_max_length in product(
        profile.batch_sizes,
        profile.grad_accum_steps,
        profile.gradient_checkpointing,
        profile.max_pixels,
        profile.model_max_lengths,
    ):
        trial = (
            f"probe_bs{batch}_ga{accum}_gc{checkpointing}"
            f"_px{max_pixels}_ctx{model_max_length}"
        )
        env = {
            "MODEL_NAME_OR_PATH": "Qwen/Qwen3-VL-8B-Instruct",
            "RUN_NAME": trial,
            "DATASETS": profile.datasets,
            "EVAL_DATASETS": profile.eval_datasets,
            "MAX_STEPS": str(profile.max_steps),
            "EVAL_STEPS": str(profile.eval_steps),
            "SAVE_STEPS": str(profile.save_steps),
            "BATCH_SIZE": str(batch),
            "GRAD_ACCUM_STEPS": str(accum),
            "GRADIENT_CHECKPOINTING": str(checkpointing),
            "MAX_PIXELS": str(max_pixels),
            "MODEL_MAX_LENGTH": str(model_max_length),
        }
        trials.append(TrialSpec(profile=profile.name, phase="probe", trial=trial, env=env))
    return trials
```

- [ ] **Step 4: Add profile files**

Create `qwen-vl-finetune/research/profiles/b200.env`:

```bash
PROFILE_NAME=b200
PROFILE_ENABLED=true
B200_VRAM_CEILING_MB=160000
MAX_STEPS=50
EVAL_STEPS=25
SAVE_STEPS=50
DATASETS=visualwebinstruct_train
EVAL_DATASETS=visualwebinstruct_val
PROBE_BATCH_SIZES=2,4,8
PROBE_GRAD_ACCUM_STEPS=2,4
PROBE_GRADIENT_CHECKPOINTING=True,False
PROBE_MAX_PIXELS=50176,112896
PROBE_MODEL_MAX_LENGTHS=8192
```

Create `qwen-vl-finetune/research/profiles/a100.env`:

```bash
PROFILE_NAME=a100
PROFILE_ENABLED=false
B200_VRAM_CEILING_MB=0
MAX_STEPS=50
EVAL_STEPS=25
SAVE_STEPS=50
DATASETS=visualwebinstruct_train
EVAL_DATASETS=visualwebinstruct_val
PROBE_BATCH_SIZES=1
PROBE_GRAD_ACCUM_STEPS=4
PROBE_GRADIENT_CHECKPOINTING=True
PROBE_MAX_PIXELS=50176
PROBE_MODEL_MAX_LENGTHS=8192
```

Create `qwen-vl-finetune/research/profiles/h100.env`:

```bash
PROFILE_NAME=h100
PROFILE_ENABLED=false
B200_VRAM_CEILING_MB=0
MAX_STEPS=50
EVAL_STEPS=25
SAVE_STEPS=50
DATASETS=visualwebinstruct_train
EVAL_DATASETS=visualwebinstruct_val
PROBE_BATCH_SIZES=1
PROBE_GRAD_ACCUM_STEPS=4
PROBE_GRADIENT_CHECKPOINTING=True
PROBE_MAX_PIXELS=50176
PROBE_MODEL_MAX_LENGTHS=8192
```

- [ ] **Step 5: Run tests**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_profile_config.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add qwen-vl-finetune/research/profile_config.py qwen-vl-finetune/research/profiles qwen-vl-finetune/tests/research/test_profile_config.py
git commit -m "feat: add research hardware profiles"
```

## Task 3: Metrics Parsing and Capacity Selection

**Files:**
- Create: `qwen-vl-finetune/research/metrics.py`
- Create: `qwen-vl-finetune/research/selection.py`
- Create: `qwen-vl-finetune/tests/research/test_metrics.py`
- Create: `qwen-vl-finetune/tests/research/test_selection.py`

- [ ] **Step 1: Write failing tests for log parsing**

Create `qwen-vl-finetune/tests/research/test_metrics.py`:

```python
from research.metrics import classify_failure, parse_trial_metrics
from research.models import FailureReason


def test_parse_trial_metrics_extracts_footer_and_throughput() -> None:
    text = """
{'train_runtime': '159.9', 'train_steps_per_second': '0.313'}
val_loss: 0.5600338578224182
peak_vram_mb: 95280.6162109375
"""

    metrics = parse_trial_metrics(text, return_code=0)

    assert metrics.status == "ok"
    assert metrics.val_loss == 0.5600338578224182
    assert metrics.peak_vram_mb == 95280.6162109375
    assert metrics.throughput_steps_per_sec == 0.313


def test_classify_failure_detects_oom() -> None:
    assert classify_failure("CUDA out of memory", return_code=1) == FailureReason.OOM


def test_missing_footer_is_crash() -> None:
    metrics = parse_trial_metrics("training ended without footer", return_code=0)

    assert metrics.status == "crash"
    assert metrics.failure_reason == FailureReason.MISSING_FOOTER
```

- [ ] **Step 2: Write failing tests for selection**

Create `qwen-vl-finetune/tests/research/test_selection.py`:

```python
from pathlib import Path

from research.models import TrialMetrics, TrialSpec
from research.selection import choose_capacity, write_selected_env


def test_choose_capacity_prefers_throughput_under_vram_ceiling() -> None:
    slow = TrialSpec("b200", "probe", "slow", {"BATCH_SIZE": "2"})
    fast = TrialSpec("b200", "probe", "fast", {"BATCH_SIZE": "4"})
    too_big = TrialSpec("b200", "probe", "too_big", {"BATCH_SIZE": "8"})
    rows = [
        (slow, TrialMetrics("ok", val_loss=0.50, peak_vram_mb=90000, throughput_steps_per_sec=0.2)),
        (fast, TrialMetrics("ok", val_loss=0.60, peak_vram_mb=120000, throughput_steps_per_sec=0.4)),
        (too_big, TrialMetrics("ok", val_loss=0.40, peak_vram_mb=170000, throughput_steps_per_sec=0.5)),
    ]

    selected = choose_capacity(rows, vram_ceiling_mb=160000)

    assert selected.trial == "fast"


def test_write_selected_env_writes_shell_assignments(tmp_path: Path) -> None:
    spec = TrialSpec(
        "b200",
        "probe",
        "chosen",
        {
            "BATCH_SIZE": "4",
            "GRAD_ACCUM_STEPS": "2",
            "GRADIENT_CHECKPOINTING": "False",
            "MAX_PIXELS": "50176",
            "MODEL_MAX_LENGTH": "8192",
        },
    )

    path = write_selected_env(tmp_path, spec)

    assert path == tmp_path / "b200.env"
    assert "BATCH_SIZE=4" in path.read_text(encoding="utf-8")
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_metrics.py tests/research/test_selection.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 4: Implement metrics parsing**

Create `qwen-vl-finetune/research/metrics.py`:

```python
from __future__ import annotations

import re

from research.models import FailureReason, TrialMetrics


_VAL_RE = re.compile(r"^val_loss:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
_VRAM_RE = re.compile(r"^peak_vram_mb:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
_THROUGHPUT_RE = re.compile(r"'train_steps_per_second':\s*'([0-9.eE+-]+)'")


def classify_failure(log_text: str, return_code: int) -> FailureReason:
    lower = log_text.lower()
    if "out of memory" in lower or "cuda oom" in lower:
        return FailureReason.OOM
    if "nccl" in lower and ("error" in lower or "timeout" in lower or "watchdog" in lower):
        return FailureReason.NCCL
    if return_code != 0:
        return FailureReason.LAUNCHER_ERROR
    return FailureReason.UNKNOWN


def _last_float(regex: re.Pattern[str], text: str) -> float | None:
    matches = regex.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_trial_metrics(log_text: str, return_code: int) -> TrialMetrics:
    val_loss = _last_float(_VAL_RE, log_text)
    peak_vram_mb = _last_float(_VRAM_RE, log_text)
    throughput = _last_float(_THROUGHPUT_RE, log_text)

    if return_code == 0 and val_loss is not None and peak_vram_mb is not None:
        return TrialMetrics(
            status="ok",
            val_loss=val_loss,
            peak_vram_mb=peak_vram_mb,
            throughput_steps_per_sec=throughput,
            failure_reason=FailureReason.NONE,
        )

    reason = FailureReason.MISSING_FOOTER
    if return_code != 0:
        reason = classify_failure(log_text, return_code)
    return TrialMetrics(
        status="crash",
        val_loss=val_loss,
        peak_vram_mb=peak_vram_mb,
        throughput_steps_per_sec=throughput,
        failure_reason=reason,
    )
```

- [ ] **Step 5: Implement selection**

Create `qwen-vl-finetune/research/selection.py`:

```python
from __future__ import annotations

from pathlib import Path

from research.models import TrialMetrics, TrialSpec


REQUIRED_SELECTED_KEYS = (
    "BATCH_SIZE",
    "GRAD_ACCUM_STEPS",
    "GRADIENT_CHECKPOINTING",
    "MAX_PIXELS",
    "MODEL_MAX_LENGTH",
)


def choose_capacity(
    rows: list[tuple[TrialSpec, TrialMetrics]],
    *,
    vram_ceiling_mb: float,
) -> TrialSpec:
    candidates: list[tuple[TrialSpec, TrialMetrics]] = []
    for spec, metrics in rows:
        if not metrics.is_successful:
            continue
        if metrics.peak_vram_mb is None or metrics.peak_vram_mb > vram_ceiling_mb:
            continue
        candidates.append((spec, metrics))

    if not candidates:
        raise ValueError("No successful probe rows under the VRAM ceiling")

    def sort_key(row: tuple[TrialSpec, TrialMetrics]) -> tuple[float, float, float]:
        _, metrics = row
        throughput = metrics.throughput_steps_per_sec or 0.0
        val_loss = metrics.val_loss if metrics.val_loss is not None else float("inf")
        vram = metrics.peak_vram_mb if metrics.peak_vram_mb is not None else float("inf")
        return (-throughput, val_loss, vram)

    return sorted(candidates, key=sort_key)[0][0]


def write_selected_env(selected_dir: str | Path, spec: TrialSpec) -> Path:
    out_dir = Path(selected_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.profile}.env"

    lines = [f"# Selected from {spec.trial}"]
    for key in REQUIRED_SELECTED_KEYS:
        if key not in spec.env:
            raise ValueError(f"Selected trial missing env key: {key}")
        lines.append(f"{key}={spec.env[key]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
```

- [ ] **Step 6: Run tests**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_metrics.py tests/research/test_selection.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add qwen-vl-finetune/research/metrics.py qwen-vl-finetune/research/selection.py qwen-vl-finetune/tests/research/test_metrics.py qwen-vl-finetune/tests/research/test_selection.py
git commit -m "feat: parse and select research trial metrics"
```

## Task 4: External Subagent Provider Adapter

**Files:**
- Create: `qwen-vl-finetune/research/subagents.py`
- Create: `qwen-vl-finetune/tests/research/test_subagents.py`

- [ ] **Step 1: Write failing tests for provider detection and command construction**

Create `qwen-vl-finetune/tests/research/test_subagents.py`:

```python
from pathlib import Path

from research.subagents import AgentProvider, SubagentTask, build_command, discover_providers


def test_build_coco_command() -> None:
    task = SubagentTask(
        provider=AgentProvider.COCO,
        worktree="b200-models",
        workspace=Path("/repo/qwen3_vl"),
        prompt="Implement Task 1. Do not revert unrelated changes.",
    )

    command = build_command(task)

    assert command[:4] == ["coco", "-p", "-y", "--worktree"]
    assert command[4] == "b200-models"
    assert command[-1].startswith("Implement Task 1")


def test_build_cursor_agent_command() -> None:
    task = SubagentTask(
        provider=AgentProvider.CURSOR_AGENT,
        worktree="b200-metrics",
        workspace=Path("/repo/qwen3_vl"),
        prompt="Implement Task 3. Do not revert unrelated changes.",
    )

    command = build_command(task)

    assert command[:4] == ["agent", "-p", "--trust", "--workspace"]
    assert command[4] == "/repo/qwen3_vl"
    assert "-w" in command
    assert command[-1].startswith("Implement Task 3")


def test_discover_providers_uses_supplied_which() -> None:
    providers = discover_providers(which=lambda name: f"/bin/{name}" if name in {"coco", "agent"} else None)

    assert providers[AgentProvider.COCO] == "/bin/coco"
    assert providers[AgentProvider.CURSOR_AGENT] == "/bin/agent"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_subagents.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'research.subagents'`.

- [ ] **Step 3: Implement subagent adapter**

Create `qwen-vl-finetune/research/subagents.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from shutil import which as shutil_which


class AgentProvider(StrEnum):
    COCO = "coco"
    CURSOR_AGENT = "cursor_agent"


@dataclass(frozen=True)
class SubagentTask:
    provider: AgentProvider
    worktree: str
    workspace: Path
    prompt: str


def discover_providers(
    which: Callable[[str], str | None] = shutil_which,
) -> dict[AgentProvider, str]:
    providers: dict[AgentProvider, str] = {}
    coco = which("coco")
    if coco:
        providers[AgentProvider.COCO] = coco
    cursor_agent = which("agent")
    if cursor_agent:
        providers[AgentProvider.CURSOR_AGENT] = cursor_agent
    return providers


def build_command(task: SubagentTask) -> list[str]:
    if task.provider == AgentProvider.COCO:
        return [
            "coco",
            "-p",
            "-y",
            "--worktree",
            task.worktree,
            "--query-timeout",
            "30m",
            task.prompt,
        ]
    if task.provider == AgentProvider.CURSOR_AGENT:
        return [
            "agent",
            "-p",
            "--trust",
            "--workspace",
            str(task.workspace),
            "-w",
            task.worktree,
            "--output-format",
            "text",
            task.prompt,
        ]
    raise ValueError(f"Unsupported provider: {task.provider}")


def implementation_prompt(task_name: str, files: list[str], test_command: str) -> str:
    file_list = "\n".join(f"- {path}" for path in files)
    return (
        f"Implement {task_name} in this repository.\n"
        "You are not alone in the codebase. Do not revert unrelated changes.\n"
        "Own only these files:\n"
        f"{file_list}\n"
        f"Run this verification command before finishing:\n{test_command}\n"
        "Return a concise summary and list changed files."
    )
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_subagents.py -v
```

Expected: PASS.

- [ ] **Step 5: Verify available providers in this environment**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. python - <<'PY'
from research.subagents import discover_providers
print(discover_providers())
PY
```

Expected in this workspace: output includes `coco` and `cursor_agent` if `/data02/home/philip.yang/.local/bin/coco` and `/data02/home/philip.yang/.local/bin/agent` are still on PATH.

- [ ] **Step 6: Commit**

```bash
git add qwen-vl-finetune/research/subagents.py qwen-vl-finetune/tests/research/test_subagents.py
git commit -m "feat: add research subagent adapters"
```

## Task 5: Local Profile Runner and Shell Entrypoint

**Files:**
- Create: `qwen-vl-finetune/research/runner.py`
- Create: `qwen-vl-finetune/research/run_profiled.sh`
- Create: `qwen-vl-finetune/research/lib/common.sh`
- Create: `qwen-vl-finetune/tests/research/test_runner_dry_run.py`

- [ ] **Step 1: Write failing dry-run tests**

Create `qwen-vl-finetune/tests/research/test_runner_dry_run.py`:

```python
from pathlib import Path

from research.runner import append_result_row, build_trial_env, results_header
from research.models import FailureReason, TrialMetrics, TrialSpec


def test_build_trial_env_sets_output_dir(tmp_path: Path) -> None:
    root = tmp_path / "qwen-vl-finetune"
    spec = TrialSpec("b200", "probe", "probe1", {"BATCH_SIZE": "2"})

    env = build_trial_env(root, spec)

    assert env["BATCH_SIZE"] == "2"
    assert env["OUTPUT_DIR"].endswith("experiments/runs/b200/probe/probe1/output")
    assert env["RUN_NAME"] == "probe1"


def test_append_result_row_writes_profile_and_failure_reason(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"
    spec = TrialSpec("b200", "probe", "probe1", {"BATCH_SIZE": "2"})
    metrics = TrialMetrics(
        status="crash",
        failure_reason=FailureReason.MISSING_FOOTER,
    )

    append_result_row(path, spec, metrics, git_commit="abc123", log_path=Path("run.log"), output_dir=Path("out"))

    text = path.read_text(encoding="utf-8")
    assert results_header().split("\t")[2] == "profile"
    assert "b200\tprobe\tprobe1" in text
    assert "missing_footer" in text


def test_runner_exposes_subagent_command() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "research.runner", "subagents", "--provider", "all"],
        cwd=Path(__file__).resolve().parents[2],
        env={"PYTHONPATH": "."},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_runner_dry_run.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'research.runner'`.

- [ ] **Step 3: Implement runner helpers and CLI**

Create `qwen-vl-finetune/research/runner.py`:

```python
from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from research.metrics import parse_trial_metrics
from research.models import TrialMetrics, TrialSpec
from research.profile_config import load_profile, plan_probe_trials
from research.selection import choose_capacity, write_selected_env
from research.subagents import AgentProvider, SubagentTask, build_command, discover_providers, implementation_prompt


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def results_header() -> str:
    return "\t".join(
        [
            "ts",
            "git",
            "profile",
            "phase",
            "trial",
            "batch_size",
            "grad_accum",
            "gradient_checkpointing",
            "max_pixels",
            "model_max_length",
            "val_loss",
            "peak_vram_mb",
            "throughput_steps_per_sec",
            "status",
            "failure_reason",
            "output_dir",
            "log",
        ]
    )


def git_short(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def build_trial_env(root: Path, spec: TrialSpec) -> dict[str, str]:
    env = dict(os.environ)
    env.update(spec.env)
    env["RUN_NAME"] = spec.trial
    env["OUTPUT_DIR"] = str(spec.output_dir(root))
    return env


def append_result_row(
    path: Path,
    spec: TrialSpec,
    metrics: TrialMetrics,
    *,
    git_commit: str,
    log_path: Path,
    output_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(results_header() + "\n", encoding="utf-8")

    row = [
        datetime.now(timezone.utc).isoformat(),
        git_commit,
        spec.profile,
        spec.phase,
        spec.trial,
        spec.env.get("BATCH_SIZE", ""),
        spec.env.get("GRAD_ACCUM_STEPS", ""),
        spec.env.get("GRADIENT_CHECKPOINTING", ""),
        spec.env.get("MAX_PIXELS", ""),
        spec.env.get("MODEL_MAX_LENGTH", ""),
        "" if metrics.val_loss is None else str(metrics.val_loss),
        "" if metrics.peak_vram_mb is None else str(metrics.peak_vram_mb),
        "" if metrics.throughput_steps_per_sec is None else str(metrics.throughput_steps_per_sec),
        metrics.status,
        metrics.failure_reason.value,
        str(output_dir),
        str(log_path),
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def run_trial(root: Path, spec: TrialSpec, *, dry_run: bool) -> TrialMetrics:
    run_dir = spec.run_dir(root)
    log_path = spec.log_path(root)
    run_dir.mkdir(parents=True, exist_ok=True)

    launcher = root / "scripts" / "sft_qwen3_8b.sh"
    if dry_run:
        log_path.write_text(f"DRY_RUN launcher={launcher}\n", encoding="utf-8")
        return TrialMetrics(status="ok", val_loss=0.0, peak_vram_mb=0.0, throughput_steps_per_sec=0.0)

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            ["bash", str(launcher)],
            cwd=root,
            env=build_trial_env(root, spec),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return parse_trial_metrics(log_path.read_text(encoding="utf-8", errors="replace"), proc.returncode)


def command_probe(args: argparse.Namespace) -> int:
    root = repo_root()
    profile = load_profile(root / "research" / "profiles" / f"{args.profile}.env")
    rows: list[tuple[TrialSpec, TrialMetrics]] = []
    results_path = root / "experiments" / "results.tsv"
    for spec in plan_probe_trials(profile):
        metrics = run_trial(root, spec, dry_run=args.dry_run)
        rows.append((spec, metrics))
        append_result_row(
            results_path,
            spec,
            metrics,
            git_commit=git_short(root),
            log_path=spec.log_path(root),
            output_dir=spec.output_dir(root),
        )
    return 0


def command_select(args: argparse.Namespace) -> int:
    root = repo_root()
    profile = load_profile(root / "research" / "profiles" / f"{args.profile}.env")
    specs = {spec.trial: spec for spec in plan_probe_trials(profile)}
    rows: list[tuple[TrialSpec, TrialMetrics]] = []
    results_path = root / "experiments" / "results.tsv"
    if not results_path.exists():
        raise SystemExit(f"Missing results file: {results_path}")

    for line in results_path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 17 or parts[2] != args.profile or parts[3] != "probe":
            continue
        spec = specs.get(parts[4])
        if spec is None:
            continue
        metrics = TrialMetrics(
            status=parts[13],
            val_loss=float(parts[10]) if parts[10] else None,
            peak_vram_mb=float(parts[11]) if parts[11] else None,
            throughput_steps_per_sec=float(parts[12]) if parts[12] else None,
        )
        rows.append((spec, metrics))

    selected = choose_capacity(rows, vram_ceiling_mb=profile.vram_ceiling_mb)
    out = write_selected_env(root / "research" / "selected", selected)
    print(out)
    return 0


def command_subagents(args: argparse.Namespace) -> int:
    root = repo_root().parent
    providers = discover_providers()
    if args.provider == "all":
        selected = sorted(providers)
    else:
        selected = [AgentProvider(args.provider)]

    if not selected:
        print("No requested subagent providers are available")
        return 0

    prompt = implementation_prompt(
        "B200 profiled experimentation implementation",
        [
            "qwen-vl-finetune/research/",
            "qwen-vl-finetune/tests/research/",
        ],
        "cd qwen-vl-finetune && PYTHONPATH=. pytest tests/research -v",
    )
    for provider in selected:
        if provider not in providers:
            print(f"{provider.value}: unavailable")
            continue
        task = SubagentTask(
            provider=provider,
            worktree=f"b200-research-{provider.value}",
            workspace=root,
            prompt=prompt,
        )
        print(" ".join(build_command(task)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile-aware Qwen3-VL experiment runner")
    parser.add_argument("command", choices=["probe", "select", "subagents"])
    parser.add_argument("--profile", default="b200")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provider", choices=["all", "coco", "cursor_agent"], default="all")
    args = parser.parse_args()

    if args.command == "probe":
        return command_probe(args)
    if args.command == "select":
        return command_select(args)
    if args.command == "subagents":
        return command_subagents(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add shell wrapper and common helper**

Create `qwen-vl-finetune/research/lib/common.sh`:

```bash
#!/usr/bin/env bash

research_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

finetune_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

require_8b_launcher() {
  local root
  root="$(finetune_root)"
  if [[ ! -x "$root/scripts/sft_qwen3_8b.sh" && ! -f "$root/scripts/sft_qwen3_8b.sh" ]]; then
    echo "Missing launcher: $root/scripts/sft_qwen3_8b.sh" >&2
    return 1
  fi
}
```

Create `qwen-vl-finetune/research/run_profiled.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

require_8b_launcher
cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR" python -m research.runner "$@"
```

- [ ] **Step 5: Run tests and shell syntax checks**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_runner_dry_run.py -v
bash -n research/run_profiled.sh
bash -n research/lib/common.sh
```

Expected: PASS and no shell syntax output.

- [ ] **Step 6: Commit**

```bash
git add qwen-vl-finetune/research/runner.py qwen-vl-finetune/research/run_profiled.sh qwen-vl-finetune/research/lib/common.sh qwen-vl-finetune/tests/research/test_runner_dry_run.py
git commit -m "feat: add profiled research runner"
```

## Task 6: Temporal Activities and Workflow Imports

**Files:**
- Create: `qwen-vl-finetune/research/activities.py`
- Create: `qwen-vl-finetune/research/workflows.py`
- Create: `qwen-vl-finetune/tests/research/test_workflows_import.py`

- [ ] **Step 1: Write failing import and workflow shape tests**

Create `qwen-vl-finetune/tests/research/test_workflows_import.py`:

```python
import importlib.util


def test_temporal_modules_import_or_skip_cleanly() -> None:
    if importlib.util.find_spec("temporalio") is None:
        return

    from research.activities import load_profile_activity, plan_probe_trials_activity
    from research.workflows import ProfiledExperimentWorkflow

    assert load_profile_activity is not None
    assert plan_probe_trials_activity is not None
    assert ProfiledExperimentWorkflow is not None
```

- [ ] **Step 2: Run test and verify it fails when Temporal is installed**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_workflows_import.py -v
```

Expected when `temporalio` is installed: FAIL with missing modules. Expected when not installed: PASS because the test returns early.

- [ ] **Step 3: Implement activities**

Create `qwen-vl-finetune/research/activities.py`:

```python
from __future__ import annotations

from pathlib import Path

from temporalio import activity

from research.models import TrialMetrics, TrialSpec
from research.profile_config import load_profile, plan_probe_trials
from research.runner import append_result_row, git_short, repo_root, run_trial
from research.selection import choose_capacity, write_selected_env


@activity.defn
def load_profile_activity(profile: str) -> dict:
    root = repo_root()
    loaded = load_profile(root / "research" / "profiles" / f"{profile}.env")
    return loaded.__dict__


@activity.defn
def plan_probe_trials_activity(profile: str) -> list[dict]:
    root = repo_root()
    loaded = load_profile(root / "research" / "profiles" / f"{profile}.env")
    return [spec.__dict__ for spec in plan_probe_trials(loaded)]


@activity.defn
def run_trial_activity(spec_payload: dict, dry_run: bool = False) -> dict:
    root = repo_root()
    spec = TrialSpec(**spec_payload)
    metrics = run_trial(root, spec, dry_run=dry_run)
    append_result_row(
        root / "experiments" / "results.tsv",
        spec,
        metrics,
        git_commit=git_short(root),
        log_path=spec.log_path(root),
        output_dir=spec.output_dir(root),
    )
    return metrics.__dict__


@activity.defn
def select_capacity_activity(profile: str, rows_payload: list[dict]) -> dict:
    root = repo_root()
    loaded = load_profile(root / "research" / "profiles" / f"{profile}.env")
    rows: list[tuple[TrialSpec, TrialMetrics]] = []
    for row in rows_payload:
        rows.append((TrialSpec(**row["spec"]), TrialMetrics(**row["metrics"])))
    selected = choose_capacity(rows, vram_ceiling_mb=loaded.vram_ceiling_mb)
    path = write_selected_env(root / "research" / "selected", selected)
    return {"selected": selected.__dict__, "path": str(path)}


@activity.defn
def summarize_campaign_activity(profile: str) -> str:
    root = repo_root()
    results = root / "experiments" / "results.tsv"
    summary = root / "experiments" / "summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    text = f"# Qwen3-VL Research Summary\n\nProfile: `{profile}`\n\nResults: `{results}`\n"
    summary.write_text(text, encoding="utf-8")
    return str(summary)
```

- [ ] **Step 4: Implement workflow**

Create `qwen-vl-finetune/research/workflows.py`:

```python
from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from research.activities import (
        load_profile_activity,
        plan_probe_trials_activity,
        run_trial_activity,
        select_capacity_activity,
        summarize_campaign_activity,
    )


@workflow.defn
class ProfiledExperimentWorkflow:
    def __init__(self) -> None:
        self.phase = "created"
        self.active_trial = ""
        self.completed_trials = 0
        self.latest_metrics: dict | None = None
        self.selected_capacity: dict | None = None
        self.pause_after_current = False
        self.stop_after_phase = False

    @workflow.run
    async def run(self, profile: str = "b200", dry_run: bool = False) -> dict:
        self.phase = "load_profile"
        await workflow.execute_activity(
            load_profile_activity,
            profile,
            start_to_close_timeout=timedelta(seconds=30),
        )

        self.phase = "probe"
        specs = await workflow.execute_activity(
            plan_probe_trials_activity,
            profile,
            start_to_close_timeout=timedelta(seconds=30),
        )

        rows: list[dict] = []
        for spec in specs:
            if self.stop_after_phase:
                break
            self.active_trial = spec["trial"]
            metrics = await workflow.execute_activity(
                run_trial_activity,
                args=[spec, dry_run],
                start_to_close_timeout=timedelta(hours=6),
                heartbeat_timeout=timedelta(minutes=5),
            )
            self.completed_trials += 1
            self.latest_metrics = metrics
            rows.append({"spec": spec, "metrics": metrics})
            if self.pause_after_current:
                break

        self.phase = "select"
        self.selected_capacity = await workflow.execute_activity(
            select_capacity_activity,
            args=[profile, rows],
            start_to_close_timeout=timedelta(minutes=2),
        )

        self.phase = "summarize"
        summary = await workflow.execute_activity(
            summarize_campaign_activity,
            profile,
            start_to_close_timeout=timedelta(minutes=2),
        )
        self.phase = "complete"
        return {"selected_capacity": self.selected_capacity, "summary": summary}

    @workflow.query
    def status(self) -> dict:
        return {
            "phase": self.phase,
            "active_trial": self.active_trial,
            "completed_trials": self.completed_trials,
            "latest_metrics": self.latest_metrics,
            "selected_capacity": self.selected_capacity,
        }

    @workflow.signal
    def pause_after_current_trial(self) -> None:
        self.pause_after_current = True

    @workflow.signal
    def stop_after_current_phase(self) -> None:
        self.stop_after_phase = True
```

- [ ] **Step 5: Run import tests**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_workflows_import.py -v
```

Expected: PASS. If `temporalio` is not installed, the test exits early and passes.

- [ ] **Step 6: Commit**

```bash
git add qwen-vl-finetune/research/activities.py qwen-vl-finetune/research/workflows.py qwen-vl-finetune/tests/research/test_workflows_import.py
git commit -m "feat: add Temporal research workflow"
```

## Task 7: Temporal Worker, Starter, and Documentation

**Files:**
- Create: `qwen-vl-finetune/research/worker.py`
- Create: `qwen-vl-finetune/research/start.py`
- Create: `qwen-vl-finetune/research/README.md`

- [ ] **Step 1: Create worker**

Create `qwen-vl-finetune/research/worker.py`:

```python
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures

from temporalio.client import Client
from temporalio.worker import Worker

from research.activities import (
    load_profile_activity,
    plan_probe_trials_activity,
    run_trial_activity,
    select_capacity_activity,
    summarize_campaign_activity,
)
from research.workflows import ProfiledExperimentWorkflow


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Qwen3-VL research Temporal worker")
    parser.add_argument("--address", default="localhost:7233")
    parser.add_argument("--task-queue", default="qwen3vl-b200")
    args = parser.parse_args()

    client = await Client.connect(args.address)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as activity_executor:
        worker = Worker(
            client,
            task_queue=args.task_queue,
            workflows=[ProfiledExperimentWorkflow],
            activities=[
                load_profile_activity,
                plan_probe_trials_activity,
                run_trial_activity,
                select_capacity_activity,
                summarize_campaign_activity,
            ],
            activity_executor=activity_executor,
        )
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Create starter**

Create `qwen-vl-finetune/research/start.py`:

```python
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from temporalio.client import Client

from research.workflows import ProfiledExperimentWorkflow


async def main() -> None:
    parser = argparse.ArgumentParser(description="Start a Qwen3-VL research workflow")
    parser.add_argument("--address", default="localhost:7233")
    parser.add_argument("--task-queue", default="qwen3vl-b200")
    parser.add_argument("--profile", default="b200")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workflow-id", default="")
    args = parser.parse_args()

    client = await Client.connect(args.address)
    workflow_id = args.workflow_id or (
        f"qwen3vl-{args.profile}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    handle = await client.start_workflow(
        ProfiledExperimentWorkflow.run,
        args=[args.profile, args.dry_run],
        id=workflow_id,
        task_queue=args.task_queue,
    )
    print(handle.id)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Create README**

Create `qwen-vl-finetune/research/README.md`:

```markdown
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
```

- [ ] **Step 4: Run syntax/import checks**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. python -m py_compile research/*.py
bash -n research/run_profiled.sh
bash -n research/lib/common.sh
```

Expected: PASS. If `temporalio` is not installed, `py_compile` may fail on Temporal imports; in that case install `temporalio` or defer this compile check to the Temporal environment and record the missing dependency.

- [ ] **Step 5: Commit**

```bash
git add qwen-vl-finetune/research/worker.py qwen-vl-finetune/research/start.py qwen-vl-finetune/research/README.md
git commit -m "docs: add research orchestration entrypoints"
```

## Task 8: End-to-End Dry Run Verification

**Files:**
- Modify only if dry-run failures require fixes: files created in Tasks 1-7.

- [ ] **Step 1: Run all research tests**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. pytest tests/research -v
```

Expected: all tests pass.

- [ ] **Step 2: Run local dry-run probe**

Run:

```bash
cd qwen-vl-finetune
bash research/run_profiled.sh probe --profile b200 --dry-run
```

Expected: exits 0 and writes dry-run logs under `experiments/runs/b200/probe/`.

- [ ] **Step 3: Run local selection on dry-run results**

Run:

```bash
cd qwen-vl-finetune
bash research/run_profiled.sh select --profile b200
```

Expected: prints `.../qwen-vl-finetune/research/selected/b200.env`.

- [ ] **Step 4: Inspect selected profile**

Run:

```bash
cd qwen-vl-finetune
cat research/selected/b200.env
```

Expected: contains `BATCH_SIZE=`, `GRAD_ACCUM_STEPS=`, `GRADIENT_CHECKPOINTING=`, `MAX_PIXELS=`, and `MODEL_MAX_LENGTH=`.

- [ ] **Step 5: Preview external subagent commands**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=. python -m research.runner subagents --provider all
```

Expected: exits 0. In this workspace it should print commands for Trae `coco`
and Cursor `agent` when both CLIs are on PATH; if either CLI is unavailable, it
prints only the available provider commands or a clean availability message.

- [ ] **Step 6: Commit dry-run fixes only**

If any implementation fixes were needed, commit them:

```bash
git add qwen-vl-finetune/research qwen-vl-finetune/tests/research
git commit -m "fix: verify research dry run"
```

Do not commit generated `experiments/runs`, `experiments/results.tsv`, or `research/selected/b200.env` unless the user explicitly wants generated artifacts tracked.

## Task 9: Optional Real B200 Probe Handoff

**Files:**
- No code changes expected.

- [ ] **Step 1: Confirm GPUs are visible**

Run:

```bash
nvidia-smi --list-gpus
```

Expected: eight B200 GPUs are listed.

- [ ] **Step 2: Run real B200 probe**

Run:

```bash
cd qwen-vl-finetune
nohup bash research/run_profiled.sh probe --profile b200 > experiments/b200_probe.nohup.log 2>&1 &
echo $!
```

Expected: prints a process ID. Probe logs appear under `experiments/runs/b200/probe/`.

- [ ] **Step 3: Select capacity**

After probe completes, run:

```bash
cd qwen-vl-finetune
bash research/run_profiled.sh select --profile b200
```

Expected: writes `research/selected/b200.env` with the fastest successful config under the B200 VRAM ceiling.

- [ ] **Step 4: Report result**

Summarize:

```bash
cd qwen-vl-finetune
tail -n 10 experiments/results.tsv
cat research/selected/b200.env
```

Expected: enough rows to identify probe outcomes and the selected capacity profile.

## Self-Review Notes

- Spec coverage: profiles, B200 probe, selected capacity, artifact boundaries, subagent CLI adapters, shell runner, Temporal workflow, worker/starter, queries/signals, and verification all map to tasks above.
- Placeholder scan: no planned step uses TBD/TODO/fill-later language.
- Type consistency: `TrialSpec`, `TrialMetrics`, `HardwareProfile`, `FailureReason`, `AgentProvider`, and `SubagentTask` are introduced before reuse and keep consistent names throughout.
