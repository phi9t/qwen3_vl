# Temporal Torch Experimentation Trial Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Temporal-backed trial observability and diagnosis for local and remote Qwen-VL torch experiments.

**Architecture:** Add a `research/observability/` package that owns schemas, artifact-root resolution, lightweight metadata artifacts, snapshots, subprocess supervision, analysis, and summaries. Temporal activities call the shared `SupervisorTrialRunner`; the local command path starts or connects to a local Temporal dev server using `temporal server start-dev --db-filename ~/.automata/research/temporal/temporal.db`, starts a worker, and runs the workflow. The trainer optionally emits structured JSONL events through `RESEARCH_EVENTS_PATH`, and heavy logs/model outputs default to `~/.automata/research/experiments/` rather than Git-managed metadata directories.

**Tech Stack:** Python 3.11+, stdlib dataclasses/json/subprocess/os/signal, existing `temporalio`, existing `pytest`, Bash launcher scripts, `nvidia-smi`.

---

## File Structure

- Create `qwen-vl-finetune/research/observability/__init__.py`: package exports.
- Create `qwen-vl-finetune/research/observability/schema.py`: versioned dataclasses and enum values.
- Create `qwen-vl-finetune/research/observability/redaction.py`: secret redaction helpers.
- Create `qwen-vl-finetune/research/observability/artifact_store.py`: runtime artifact root resolver and relative refs.
- Create `qwen-vl-finetune/research/observability/artifacts.py`: attempt directories and JSON/JSONL writers.
- Create `qwen-vl-finetune/research/observability/system_snapshot.py`: GPU/system/env snapshots.
- Create `qwen-vl-finetune/research/observability/log_parser.py`: log symptom extraction and metrics parsing wrapper.
- Create `qwen-vl-finetune/research/observability/analyzer.py`: root-cause classification and analysis writing.
- Create `qwen-vl-finetune/research/observability/trial_runner.py`: supervised subprocess execution.
- Create `qwen-vl-finetune/research/observability/summarizer.py`: campaign aggregation from analysis artifacts with TSV fallback.
- Create `qwen-vl-finetune/research/local_temporal.py`: local Temporal dev-server and worker harness.
- Modify `qwen-vl-finetune/.gitignore`: ignore local artifact root and heavyweight run outputs.
- Modify `qwen-vl-finetune/research/runner.py`: preserve TSV helpers and route local campaign commands through Temporal.
- Modify `qwen-vl-finetune/research/activities.py`: return `TrialAnalysis` payloads while preserving metrics fields.
- Modify `qwen-vl-finetune/research/workflows.py`: add `latest_analysis`, preserve `latest_metrics`, add cancel/rerun signal state.
- Modify `qwen-vl-finetune/qwenvl/train/train_qwen.py`: optional `RESEARCH_EVENTS_PATH` event emission.
- Add tests under `qwen-vl-finetune/tests/research/observability/`.

## Task 1: Versioned Schemas and Redaction

**Files:**
- Create: `qwen-vl-finetune/research/observability/__init__.py`
- Create: `qwen-vl-finetune/research/observability/schema.py`
- Create: `qwen-vl-finetune/research/observability/redaction.py`
- Test: `qwen-vl-finetune/tests/research/observability/test_schema.py`

- [ ] **Step 1: Write schema and redaction tests**

Create `qwen-vl-finetune/tests/research/observability/test_schema.py`:

```python
from research.models import FailureReason, TrialMetrics
from research.observability.redaction import redact_mapping
from research.observability.schema import (
    SCHEMA_VERSION,
    ResolvedTrialConfig,
    TrialAnalysis,
    TrialIntent,
)


def test_trial_intent_payload_has_schema_version_and_attempt() -> None:
    intent = TrialIntent.from_spec_payload(
        profile="b200",
        phase="probe",
        trial="probe_bs2",
        env={"BATCH_SIZE": "2"},
        attempt=1,
        campaign_id="campaign-1",
        workflow_id="wf-1",
        research_intent="capacity probe",
        expected_failure_reason="oom",
    )

    payload = intent.to_payload()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["trial_id"] == "b200/probe/probe_bs2"
    assert payload["attempt"] == 1
    assert payload["planned_env"] == {"BATCH_SIZE": "2"}
    assert payload["expected_failure_reason"] == "oom"


def test_resolved_config_redacts_secrets() -> None:
    config = ResolvedTrialConfig(
        trial_id="b200/probe/t1",
        attempt=2,
        git_commit="abc1234",
        command=["bash", "scripts/sft_qwen3_8b.sh"],
        env={
            "BATCH_SIZE": "2",
            "HF_TOKEN": "secret",
            "AWS_SECRET_ACCESS_KEY": "secret2",
            "NORMAL": "visible",
        },
        model_path="/models/qwen",
        datasets=["train"],
        eval_datasets=["eval"],
        output_dir="outputs/b200/probe/t1/attempt_002",
        run_dir="experiments/runs/b200/probe/t1/attempt_002",
        hardware_profile="b200",
        distributed={"nproc_per_node": 8},
        artifact_root_ref="~/.automata/research/experiments",
        artifact_refs={
            "full_log": "logs/b200/probe/t1/attempt_002/run.log",
            "train_events": "events/b200/probe/t1/attempt_002/train_events.jsonl",
            "output_dir": "outputs/b200/probe/t1/attempt_002",
        },
    )

    payload = config.to_payload()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["env"]["HF_TOKEN"] == "***REDACTED***"
    assert payload["env"]["AWS_SECRET_ACCESS_KEY"] == "***REDACTED***"
    assert payload["env"]["NORMAL"] == "visible"


def test_redact_mapping_matches_secret_key_patterns_case_insensitive() -> None:
    redacted = redact_mapping(
        {
            "api_key": "a",
            "PASSWORD": "b",
            "credential_path": "c",
            "TOKENIZERS_PARALLELISM": "false",
            "BATCH_SIZE": "2",
        }
    )

    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["PASSWORD"] == "***REDACTED***"
    assert redacted["credential_path"] == "***REDACTED***"
    assert redacted["TOKENIZERS_PARALLELISM"] == "***REDACTED***"
    assert redacted["BATCH_SIZE"] == "2"


def test_trial_analysis_wraps_trial_metrics() -> None:
    metrics = TrialMetrics(
        status="ok",
        val_loss=0.5,
        peak_vram_mb=100.0,
        throughput_steps_per_sec=1.25,
        failure_reason=FailureReason.NONE,
    )
    analysis = TrialAnalysis(
        trial_id="b200/probe/t1",
        attempt=1,
        status="ok",
        failure_reason=FailureReason.NONE,
        root_cause="none",
        expected_failure=False,
        metrics=metrics,
        symptoms=[],
        evidence_refs=[],
        actions=[],
        recommendations=[],
        artifact_root_ref="~/.automata/research/experiments",
        artifact_refs={
            "full_log": "logs/b200/probe/t1/attempt_001/run.log",
            "train_events": "events/b200/probe/t1/attempt_001/train_events.jsonl",
            "output_dir": "outputs/b200/probe/t1/attempt_001",
        },
        artifact_dir="experiments/runs/b200/probe/t1/attempt_001",
    )

    payload = analysis.to_payload()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["metrics"]["val_loss"] == 0.5
    assert payload["failure_reason"] == "none"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_schema.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'research.observability'`.

- [ ] **Step 3: Implement schema and redaction modules**

Create `qwen-vl-finetune/research/observability/__init__.py`:

```python
from research.observability.schema import (
    SCHEMA_VERSION,
    ResolvedTrialConfig,
    TrialAnalysis,
    TrialIntent,
)

__all__ = [
    "SCHEMA_VERSION",
    "ResolvedTrialConfig",
    "TrialAnalysis",
    "TrialIntent",
]
```

Create `qwen-vl-finetune/research/observability/redaction.py`:

```python
from __future__ import annotations

import re
from collections.abc import Mapping


SECRET_KEY_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL)", re.IGNORECASE)
REDACTED = "***REDACTED***"


def is_secret_key(key: str) -> bool:
    return bool(SECRET_KEY_RE.search(key))


def redact_mapping(values: Mapping[str, object]) -> dict[str, object]:
    redacted: dict[str, object] = {}
    for key, value in values.items():
        redacted[key] = REDACTED if is_secret_key(key) else value
    return redacted
```

Create `qwen-vl-finetune/research/observability/schema.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from research.models import FailureReason, TrialMetrics
from research.observability.redaction import redact_mapping


SCHEMA_VERSION = 1


def _metrics_to_payload(metrics: TrialMetrics) -> dict[str, Any]:
    return {
        "status": metrics.status,
        "val_loss": metrics.val_loss,
        "peak_vram_mb": metrics.peak_vram_mb,
        "throughput_steps_per_sec": metrics.throughput_steps_per_sec,
        "failure_reason": metrics.failure_reason.value,
    }


@dataclass(frozen=True)
class TrialIntent:
    trial_id: str
    attempt: int
    profile: str
    phase: str
    trial: str
    research_intent: str
    planned_env: dict[str, str]
    success_criteria: dict[str, Any] = field(default_factory=dict)
    expected_failure_reason: str | None = None
    campaign_id: str | None = None
    workflow_id: str | None = None

    @classmethod
    def from_spec_payload(
        cls,
        *,
        profile: str,
        phase: str,
        trial: str,
        env: dict[str, str],
        attempt: int,
        campaign_id: str | None = None,
        workflow_id: str | None = None,
        research_intent: str = "profiled experiment trial",
        expected_failure_reason: str | None = None,
    ) -> "TrialIntent":
        return cls(
            trial_id=f"{profile}/{phase}/{trial}",
            attempt=attempt,
            profile=profile,
            phase=phase,
            trial=trial,
            research_intent=research_intent,
            planned_env=dict(env),
            success_criteria={},
            expected_failure_reason=expected_failure_reason,
            campaign_id=campaign_id,
            workflow_id=workflow_id,
        )

    def to_payload(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True)
class ResolvedTrialConfig:
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
    distributed: dict[str, Any]
    artifact_root_ref: str
    artifact_refs: dict[str, str]

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["env"] = redact_mapping(payload["env"])
        return {"schema_version": SCHEMA_VERSION, **payload}


@dataclass(frozen=True)
class TrialAnalysis:
    trial_id: str
    attempt: int
    status: str
    failure_reason: FailureReason
    root_cause: str
    expected_failure: bool
    metrics: TrialMetrics
    symptoms: list[dict[str, Any]]
    evidence_refs: list[str]
    artifact_root_ref: str
    artifact_refs: dict[str, str]
    actions: list[dict[str, Any]]
    recommendations: list[str]
    artifact_dir: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "trial_id": self.trial_id,
            "attempt": self.attempt,
            "status": self.status,
            "failure_reason": self.failure_reason.value,
            "root_cause": self.root_cause,
            "expected_failure": self.expected_failure,
            "metrics": _metrics_to_payload(self.metrics),
            "symptoms": self.symptoms,
            "evidence_refs": self.evidence_refs,
            "artifact_root_ref": self.artifact_root_ref,
            "artifact_refs": self.artifact_refs,
            "actions": self.actions,
            "recommendations": self.recommendations,
            "artifact_dir": self.artifact_dir,
        }
```

- [ ] **Step 4: Run schema tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_schema.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add qwen-vl-finetune/research/observability qwen-vl-finetune/tests/research/observability/test_schema.py
git commit -m "feat: add trial observability schemas"
```

## Task 2: Artifact Root Resolver and Lightweight Metadata Store

**Files:**
- Create: `qwen-vl-finetune/research/observability/artifact_store.py`
- Create: `qwen-vl-finetune/research/observability/artifacts.py`
- Modify: `qwen-vl-finetune/.gitignore`
- Test: `qwen-vl-finetune/tests/research/observability/test_artifacts.py`

- [ ] **Step 1: Write artifact root and metadata tests**

Create `qwen-vl-finetune/tests/research/observability/test_artifacts.py`:

```python
import json
from pathlib import Path

from research.observability.artifact_store import (
    ArtifactRoot,
    default_artifact_root,
    reject_local_absolute_paths,
)
from research.observability.artifacts import TrialArtifactStore
from research.observability.schema import SCHEMA_VERSION


def test_attempt_directory_is_incremented(tmp_path: Path) -> None:
    base = tmp_path / "experiments" / "runs" / "b200" / "probe" / "trial"
    first = TrialArtifactStore.create_next(base)
    second = TrialArtifactStore.create_next(base)

    assert first.attempt == 1
    assert second.attempt == 2
    assert first.attempt_dir.name == "attempt_001"
    assert second.attempt_dir.name == "attempt_002"
    assert (base / "latest_attempt.txt").read_text(encoding="utf-8") == "attempt_002\n"


def test_artifact_root_uses_env_and_produces_relative_refs(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "external-artifacts"
    monkeypatch.setenv("RESEARCH_ARTIFACT_ROOT", str(root))
    artifact_root = ArtifactRoot.from_env(repo_root=tmp_path / "repo")

    ref = artifact_root.ref_for("logs", "b200", "probe", "trial", "attempt_001", "run.log")
    resolved = artifact_root.resolve(ref)

    assert artifact_root.root == root
    assert ref == "logs/b200/probe/trial/attempt_001/run.log"
    assert resolved == root / ref
    assert not ref.startswith("/")


def test_default_artifact_root_is_user_research_dir(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    assert default_artifact_root(tmp_path) == home / ".automata" / "research" / "experiments"


def test_reject_local_absolute_paths_rejects_metadata_paths() -> None:
    payload = {"artifact_refs": {"full_log": "/data02/home/user/run.log"}}

    try:
        reject_local_absolute_paths(payload)
    except ValueError as exc:
        assert "absolute local path" in str(exc)
    else:
        raise AssertionError("expected absolute path rejection")


def test_write_json_adds_schema_version(tmp_path: Path) -> None:
    store = TrialArtifactStore.create_next(tmp_path / "trial")

    store.write_json("intent.json", {"trial_id": "x"})

    payload = json.loads((store.attempt_dir / "intent.json").read_text(encoding="utf-8"))
    assert payload == {"schema_version": SCHEMA_VERSION, "trial_id": "x"}


def test_append_jsonl_adds_schema_version(tmp_path: Path) -> None:
    store = TrialArtifactStore.create_next(tmp_path / "trial")

    store.append_jsonl("lifecycle.jsonl", {"stage": "trial_started"})
    store.append_jsonl("lifecycle.jsonl", {"stage": "process_exited"})

    lines = (store.attempt_dir / "lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["schema_version"] == SCHEMA_VERSION
    assert json.loads(lines[0])["stage"] == "trial_started"
    assert json.loads(lines[1])["stage"] == "process_exited"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_artifacts.py -v
```

Expected: FAIL with `ModuleNotFoundError`, missing `ArtifactRoot`, or missing `TrialArtifactStore`.

- [ ] **Step 3: Implement artifact root resolver**

Create `qwen-vl-finetune/research/observability/artifact_store.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


ARTIFACT_ROOT_ENV = "RESEARCH_ARTIFACT_ROOT"


def default_artifact_root(repo_root: Path) -> Path:
    return Path.home() / ".automata" / "research" / "experiments"


@dataclass(frozen=True)
class ArtifactRoot:
    root: Path
    root_ref: str

    @classmethod
    def from_env(cls, repo_root: Path) -> "ArtifactRoot":
        configured = os.environ.get(ARTIFACT_ROOT_ENV)
        if configured:
            return cls(root=Path(configured), root_ref=f"${{{ARTIFACT_ROOT_ENV}}}")
        return cls(root=default_artifact_root(repo_root), root_ref="~/.automata/research/experiments")

    def ref_for(self, *parts: str) -> str:
        ref = PurePosixPath(*[str(part).strip("/") for part in parts])
        if ref.is_absolute():
            raise ValueError(f"artifact ref must be relative: {ref}")
        return str(ref)

    def resolve(self, ref: str) -> Path:
        path = PurePosixPath(ref)
        if path.is_absolute():
            raise ValueError(f"artifact ref must be relative: {ref}")
        return self.root / Path(*path.parts)


def reject_local_absolute_paths(payload: Any) -> None:
    if isinstance(payload, dict):
        for value in payload.values():
            reject_local_absolute_paths(value)
    elif isinstance(payload, list):
        for value in payload:
            reject_local_absolute_paths(value)
    elif isinstance(payload, str):
        if payload.startswith(("/data", "/home/", "/Users/", "file:///")):
            raise ValueError(f"metadata contains absolute local path: {payload}")
```

- [ ] **Step 4: Implement lightweight metadata store**

Create `qwen-vl-finetune/research/observability/artifacts.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.observability.artifact_store import reject_local_absolute_paths
from research.observability.schema import SCHEMA_VERSION


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TrialArtifactStore:
    trial_dir: Path
    attempt: int
    attempt_dir: Path

    @classmethod
    def create_next(cls, trial_dir: Path) -> "TrialArtifactStore":
        trial_dir.mkdir(parents=True, exist_ok=True)
        attempts = []
        for path in trial_dir.glob("attempt_*"):
            if path.is_dir():
                try:
                    attempts.append(int(path.name.split("_", 1)[1]))
                except ValueError:
                    continue
        attempt = max(attempts, default=0) + 1
        attempt_dir = trial_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=False, exist_ok=False)
        (trial_dir / "latest_attempt.txt").write_text(f"attempt_{attempt:03d}\n", encoding="utf-8")
        return cls(trial_dir=trial_dir, attempt=attempt, attempt_dir=attempt_dir)

    def path(self, name: str) -> Path:
        return self.attempt_dir / name

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        out = dict(payload)
        out.setdefault("schema_version", SCHEMA_VERSION)
        if name in {"intent.json", "resolved_config.json", "analysis.json"}:
            reject_local_absolute_paths(out)
        path = self.path(name)
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def append_jsonl(self, name: str, payload: dict[str, Any]) -> Path:
        out = dict(payload)
        out.setdefault("schema_version", SCHEMA_VERSION)
        out.setdefault("ts", utc_now_iso())
        if name in {"actions.jsonl"}:
            reject_local_absolute_paths(out)
        path = self.path(name)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, sort_keys=True) + "\n")
        return path

    def lifecycle(self, stage: str, *, message: str = "", source: str = "supervisor", **extra: Any) -> None:
        payload = {"source": source, "stage": stage, "message": message}
        payload.update(extra)
        self.append_jsonl("lifecycle.jsonl", payload)

    def action(self, action: str, *, reason: str, outcome: str, **extra: Any) -> None:
        payload = {"action": action, "reason": reason, "outcome": outcome}
        payload.update(extra)
        self.append_jsonl("actions.jsonl", payload)
```

- [ ] **Step 5: Add Git ignore rules for local heavy artifacts**

Modify `qwen-vl-finetune/.gitignore` and add:

```gitignore
# Optional repo-local runtime state such as Temporal dev-server SQLite files or
# a project-local RESEARCH_ARTIFACT_ROOT override.
.artifacts/

# Heavy training outputs should be stored through RESEARCH_ARTIFACT_ROOT.
experiments/runs/**/output/
experiments/runs/**/checkpoint-*/
experiments/runs/**/*.safetensors
experiments/runs/**/*.bin
experiments/runs/**/trainer_state.json
```

- [ ] **Step 6: Run artifact tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_artifacts.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add qwen-vl-finetune/.gitignore qwen-vl-finetune/research/observability/artifact_store.py qwen-vl-finetune/research/observability/artifacts.py qwen-vl-finetune/tests/research/observability/test_artifacts.py
git commit -m "feat: add trial artifact storage roots"
```

## Task 3: System Snapshots

**Files:**
- Create: `qwen-vl-finetune/research/observability/system_snapshot.py`
- Test: `qwen-vl-finetune/tests/research/observability/test_system_snapshot.py`

- [ ] **Step 1: Write snapshot tests**

Create `qwen-vl-finetune/tests/research/observability/test_system_snapshot.py`:

```python
import json
import subprocess

from research.observability.system_snapshot import collect_system_snapshot


def test_collect_system_snapshot_parses_gpu_query(monkeypatch) -> None:
    def fake_check_output(cmd, text=True, stderr=None):
        joined = " ".join(cmd)
        if "--query-gpu" in joined:
            return "0, NVIDIA B200, 183359, 10, 0\n1, NVIDIA B200, 183359, 20, 5\n"
        if cmd[:2] == ["nvidia-smi", "--query"]:
            return ""
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    payload = collect_system_snapshot(env={"BATCH_SIZE": "2", "HF_TOKEN": "secret"}, include_process_topology=False)

    assert payload["schema_version"] == 1
    assert payload["gpus"][0]["name"] == "NVIDIA B200"
    assert payload["gpus"][1]["memory_used_mb"] == 20
    assert payload["env"]["HF_TOKEN"] == "***REDACTED***"
    assert "process_topology" not in payload


def test_collect_system_snapshot_can_include_process_topology(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "")

    payload = collect_system_snapshot(env={}, include_process_topology=True, process_topology=[{"pid": 1}])

    assert payload["process_topology"] == [{"pid": 1}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_system_snapshot.py -v
```

Expected: FAIL with missing `system_snapshot`.

- [ ] **Step 3: Implement system snapshot collector**

Create `qwen-vl-finetune/research/observability/system_snapshot.py`:

```python
from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

from research.observability.redaction import redact_mapping
from research.observability.schema import SCHEMA_VERSION


VERSION_PACKAGES = ("torch", "transformers", "accelerate", "deepspeed", "peft", "temporalio")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in VERSION_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _gpu_inventory() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    gpus: list[dict[str, Any]] = []
    for raw in output.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": int(parts[2]),
                "memory_used_mb": int(parts[3]),
                "utilization_gpu_percent": int(parts[4]),
            }
        )
    return gpus


def collect_system_snapshot(
    *,
    env: dict[str, str],
    include_process_topology: bool,
    process_topology: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "packages": _package_versions(),
        "gpus": _gpu_inventory(),
        "env": redact_mapping(env),
    }
    if include_process_topology:
        payload["process_topology"] = process_topology or []
    return payload
```

- [ ] **Step 4: Run snapshot tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_system_snapshot.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add qwen-vl-finetune/research/observability/system_snapshot.py qwen-vl-finetune/tests/research/observability/test_system_snapshot.py
git commit -m "feat: add trial system snapshots"
```

## Task 4: Analyzer and Root-Cause Classification

**Files:**
- Create: `qwen-vl-finetune/research/observability/log_parser.py`
- Create: `qwen-vl-finetune/research/observability/analyzer.py`
- Test: `qwen-vl-finetune/tests/research/observability/test_analyzer.py`

- [ ] **Step 1: Write analyzer tests**

Create `qwen-vl-finetune/tests/research/observability/test_analyzer.py`:

```python
from pathlib import Path

from research.models import FailureReason
from research.observability.analyzer import analyze_trial


def test_analyze_expected_oom_capacity_probe(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("RuntimeError: CUDA out of memory. Tried to allocate 1.00 GiB\n", encoding="utf-8")

    analysis = analyze_trial(
        trial_id="b200/probe/t1",
        attempt=1,
        artifact_dir=tmp_path,
        log_path=log,
        return_code=1,
        expected_failure_reason="oom",
    )

    assert analysis.failure_reason == FailureReason.OOM
    assert analysis.root_cause == "capacity_exceeded"
    assert analysis.expected_failure is True
    assert analysis.actions[0]["action"] == "mark_capacity_boundary"


def test_analyze_missing_footer_with_zero_return_code(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("{'train_steps_per_second': 0.2}\n", encoding="utf-8")

    analysis = analyze_trial(
        trial_id="b200/probe/t2",
        attempt=1,
        artifact_dir=tmp_path,
        log_path=log,
        return_code=0,
        expected_failure_reason=None,
    )

    assert analysis.failure_reason == FailureReason.MISSING_FOOTER
    assert analysis.root_cause == "trainer_instrumentation_gap"
    assert analysis.expected_failure is False


def test_analyze_dataset_error_refines_launcher_error(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("KeyError: 'unknown_dataset'\n", encoding="utf-8")

    analysis = analyze_trial(
        trial_id="b200/sweep/t3",
        attempt=1,
        artifact_dir=tmp_path,
        log_path=log,
        return_code=1,
        expected_failure_reason=None,
    )

    assert analysis.failure_reason == FailureReason.LAUNCHER_ERROR
    assert analysis.root_cause == "dataset_unavailable"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_analyzer.py -v
```

Expected: FAIL with missing analyzer module.

- [ ] **Step 3: Implement log parser wrapper and analyzer**

Create `qwen-vl-finetune/research/observability/log_parser.py`:

```python
from __future__ import annotations

from research.metrics import parse_trial_metrics
from research.models import FailureReason, TrialMetrics


def parse_metrics_and_symptoms(log_text: str, return_code: int) -> tuple[TrialMetrics, list[dict]]:
    metrics = parse_trial_metrics(log_text, return_code)
    lower = log_text.lower()
    symptoms: list[dict] = []
    if "out of memory" in lower or "cuda oom" in lower:
        symptoms.append({"kind": "oom", "message": "CUDA out of memory detected"})
    if "nccl" in lower and ("error" in lower or "timeout" in lower or "watchdog" in lower):
        symptoms.append({"kind": "nccl", "message": "NCCL failure signature detected"})
    if "keyerror" in lower and "dataset" in lower:
        symptoms.append({"kind": "dataset", "message": "Dataset registry/config lookup failed"})
    if "429" in lower or "retry after" in lower:
        symptoms.append({"kind": "hub_or_cache", "message": "Hub/cache transient failure signature detected"})
    return metrics, symptoms
```

Create `qwen-vl-finetune/research/observability/analyzer.py`:

```python
from __future__ import annotations

from pathlib import Path

from research.models import FailureReason
from research.observability.log_parser import parse_metrics_and_symptoms
from research.observability.schema import TrialAnalysis


DEFAULT_ROOT_CAUSE = {
    FailureReason.NONE: "none",
    FailureReason.OOM: "capacity_exceeded",
    FailureReason.NCCL: "distributed_runtime_failure",
    FailureReason.LAUNCHER_ERROR: "invalid_config",
    FailureReason.MISSING_FOOTER: "trainer_instrumentation_gap",
    FailureReason.CANCELLED: "operator_cancelled",
    FailureReason.UNKNOWN: "unknown",
}


def _refine_root_cause(log_text: str, failure_reason: FailureReason) -> str:
    lower = log_text.lower()
    if failure_reason == FailureReason.LAUNCHER_ERROR:
        if "keyerror" in lower and "dataset" in lower:
            return "dataset_unavailable"
        if "modulenotfounderror" in lower or "importerror" in lower:
            return "dependency_missing"
        if "snapshot" in lower or "huggingface" in lower or "hf_hub" in lower:
            return "model_snapshot_unavailable"
    return DEFAULT_ROOT_CAUSE.get(failure_reason, "unknown")


def analyze_trial(
    *,
    trial_id: str,
    attempt: int,
    artifact_dir: Path,
    artifact_dir_ref: str | None = None,
    log_path: Path,
    return_code: int,
    expected_failure_reason: str | None,
    artifact_root_ref: str = "~/.automata/research/experiments",
    artifact_refs: dict[str, str] | None = None,
) -> TrialAnalysis:
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    metrics, symptoms = parse_metrics_and_symptoms(log_text, return_code)
    root_cause = _refine_root_cause(log_text, metrics.failure_reason)
    expected_failure = expected_failure_reason == metrics.failure_reason.value
    actions = []
    recommendations = []
    if metrics.failure_reason == FailureReason.OOM and expected_failure:
        actions.append(
            {
                "action": "mark_capacity_boundary",
                "reason": "oom during expected capacity probe",
                "outcome": "continued",
            }
        )
        recommendations.append("Exclude this capacity point from sweep candidates.")
    elif metrics.failure_reason != FailureReason.NONE:
        recommendations.append("Inspect analysis evidence before continuing larger campaigns.")
    return TrialAnalysis(
        trial_id=trial_id,
        attempt=attempt,
        status=metrics.status,
        failure_reason=metrics.failure_reason,
        root_cause=root_cause,
        expected_failure=expected_failure,
        metrics=metrics,
        symptoms=symptoms,
        evidence_refs=[(artifact_refs or {}).get("full_log", str(log_path))],
        artifact_root_ref=artifact_root_ref,
        artifact_refs=artifact_refs or {},
        actions=actions,
        recommendations=recommendations,
        artifact_dir=artifact_dir_ref or artifact_dir.name,
    )
```

- [ ] **Step 4: Run analyzer tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_analyzer.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add qwen-vl-finetune/research/observability/log_parser.py qwen-vl-finetune/research/observability/analyzer.py qwen-vl-finetune/tests/research/observability/test_analyzer.py
git commit -m "feat: add trial analyzer"
```

## Task 5: Supervisor Trial Runner With Attempts, Watchdogs, and Process Groups

**Files:**
- Create: `qwen-vl-finetune/research/observability/trial_runner.py`
- Test: `qwen-vl-finetune/tests/research/observability/test_trial_runner.py`

- [ ] **Step 1: Write trial runner tests**

Create `qwen-vl-finetune/tests/research/observability/test_trial_runner.py`:

```python
import os
import signal
import subprocess
import sys
from pathlib import Path

from research.models import TrialSpec
from research.observability.trial_runner import SupervisorTrialRunner


def test_dry_run_writes_attempt_artifacts(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    artifact_root = tmp_path / "external-artifacts"
    monkeypatch.setenv("RESEARCH_ARTIFACT_ROOT", str(artifact_root))
    launcher = root / "scripts" / "sft_qwen3_8b.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    spec = TrialSpec("b200", "probe", "trial", {"BATCH_SIZE": "2"})

    analysis = SupervisorTrialRunner(root).run_from_spec(spec, dry_run=True)

    attempt_dir = root / "experiments" / "runs" / "b200" / "probe" / "trial" / "attempt_001"
    assert analysis.status == "ok"
    assert (attempt_dir / "intent.json").exists()
    assert (attempt_dir / "resolved_config.json").exists()
    assert (attempt_dir / "system.pre.json").exists()
    assert (attempt_dir / "analysis.json").exists()
    assert not (attempt_dir / "run.log").exists()
    assert not (attempt_dir / "output").exists()
    assert (artifact_root / "logs" / "b200" / "probe" / "trial" / "attempt_001" / "run.log").exists()
    assert (artifact_root / "outputs" / "b200" / "probe" / "trial" / "attempt_001").is_dir()


def test_process_group_teardown_kills_child_processes(tmp_path: Path) -> None:
    script = tmp_path / "spawn_child.py"
    marker = tmp_path / "child.pid"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    proc = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
    try:
        for _ in range(100):
            if marker.exists():
                break
            import time
            time.sleep(0.05)
        child_pid = int(marker.read_text(encoding="utf-8"))
        SupervisorTrialRunner.terminate_process_group(proc, grace_seconds=0.1)

        assert proc.poll() is not None
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            child_alive = False
        else:
            child_alive = True
        assert child_alive is False
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_trial_runner.py -v
```

Expected: FAIL with missing `trial_runner`.

- [ ] **Step 3: Implement supervised trial runner**

Create `qwen-vl-finetune/research/observability/trial_runner.py`:

```python
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from research.models import FailureReason, TrialMetrics, TrialSpec
from research.observability.analyzer import analyze_trial
from research.observability.artifact_store import ArtifactRoot
from research.observability.artifacts import TrialArtifactStore
from research.observability.schema import ResolvedTrialConfig, TrialAnalysis, TrialIntent
from research.observability.system_snapshot import collect_system_snapshot


class SupervisorTrialRunner:
    def __init__(
        self,
        root: Path,
        *,
        startup_timeout_sec: float = 600.0,
        stall_timeout_sec: float = 1800.0,
        max_trial_wall_time_sec: float = 21600.0,
    ) -> None:
        self.root = Path(root)
        self.artifact_root = ArtifactRoot.from_env(self.root)
        self.startup_timeout_sec = startup_timeout_sec
        self.stall_timeout_sec = stall_timeout_sec
        self.max_trial_wall_time_sec = max_trial_wall_time_sec

    @staticmethod
    def terminate_process_group(proc: subprocess.Popen, *, grace_seconds: float = 30.0) -> None:
        if proc.poll() is not None:
            return
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.05)
        if proc.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=10)

    def _artifact_refs(self, spec: TrialSpec, attempt: int) -> dict[str, str]:
        attempt_name = f"attempt_{attempt:03d}"
        return {
            "full_log": self.artifact_root.ref_for("logs", spec.profile, spec.phase, spec.trial, attempt_name, "run.log"),
            "train_events": self.artifact_root.ref_for("events", spec.profile, spec.phase, spec.trial, attempt_name, "train_events.jsonl"),
            "output_dir": self.artifact_root.ref_for("outputs", spec.profile, spec.phase, spec.trial, attempt_name),
        }

    def _build_env(self, spec: TrialSpec, output_dir: Path, events_path: Path) -> dict[str, str]:
        env = dict(os.environ)
        env.update(spec.env)
        env["RUN_NAME"] = spec.trial
        env["OUTPUT_DIR"] = str(output_dir)
        env["RESEARCH_EVENTS_PATH"] = str(events_path)
        env["RESEARCH_TRIAL_ID"] = f"{spec.profile}/{spec.phase}/{spec.trial}"
        return env

    def _resolved_config(self, spec: TrialSpec, attempt: int, run_dir: Path, output_dir: Path, env: dict[str, str], artifact_refs: dict[str, str]) -> ResolvedTrialConfig:
        trial_env = {key: env[key] for key in sorted(spec.env) if key in env}
        trial_env.update(
            {
                "RUN_NAME": spec.trial,
                "OUTPUT_DIR": artifact_refs["output_dir"],
                "RESEARCH_TRIAL_ID": f"{spec.profile}/{spec.phase}/{spec.trial}",
            }
        )
        return ResolvedTrialConfig(
            trial_id=f"{spec.profile}/{spec.phase}/{spec.trial}",
            attempt=attempt,
            git_commit="unknown",
            command=["bash", "scripts/sft_qwen3_8b.sh"],
            env=trial_env,
            model_path=env.get("MODEL_NAME_OR_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
            datasets=[env.get("DATASETS", "visualwebinstruct_train")],
            eval_datasets=[env.get("EVAL_DATASETS", "visualwebinstruct_val")],
            output_dir=artifact_refs["output_dir"],
            run_dir=str(run_dir.relative_to(self.root)),
            hardware_profile=spec.profile,
            distributed={"nproc_per_node": env.get("NPROC_PER_NODE", "")},
            artifact_root_ref=self.artifact_root.root_ref,
            artifact_refs=artifact_refs,
        )

    def run_from_spec(self, spec: TrialSpec, *, dry_run: bool = False) -> TrialAnalysis:
        trial_dir = spec.run_dir(self.root)
        store = TrialArtifactStore.create_next(trial_dir)
        artifact_refs = self._artifact_refs(spec, store.attempt)
        output_dir = self.artifact_root.resolve(artifact_refs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        events_path = self.artifact_root.resolve(artifact_refs["train_events"])
        events_path.parent.mkdir(parents=True, exist_ok=True)
        env = self._build_env(spec, output_dir, events_path)
        intent = TrialIntent.from_spec_payload(
            profile=spec.profile,
            phase=spec.phase,
            trial=spec.trial,
            env=spec.env,
            attempt=store.attempt,
            expected_failure_reason="oom" if spec.phase == "probe" else None,
        )
        config = self._resolved_config(spec, store.attempt, store.attempt_dir, output_dir, env, artifact_refs)
        return self.run(intent, config, env=env, store=store, dry_run=dry_run)

    def run(
        self,
        intent: TrialIntent,
        config: ResolvedTrialConfig,
        *,
        env: dict[str, str],
        store: TrialArtifactStore,
        dry_run: bool = False,
    ) -> TrialAnalysis:
        store.write_json("intent.json", intent.to_payload())
        store.write_json("resolved_config.json", config.to_payload())
        store.write_json("system.pre.json", collect_system_snapshot(env=config.env, include_process_topology=False))
        store.lifecycle("trial_started")
        log_path = self.artifact_root.resolve(config.artifact_refs["full_log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            log_path.write_text("DRY_RUN\nval_loss: 0.0\npeak_vram_mb: 0.0\n", encoding="utf-8")
            return_code = 0
        else:
            return_code = self._run_process(config.command, env=env, log_path=log_path, store=store)
        store.lifecycle("process_exited", return_code=return_code)
        store.write_json("system.post.json", collect_system_snapshot(env=config.env, include_process_topology=True))
        store.lifecycle("analysis_started")
        analysis = analyze_trial(
            trial_id=intent.trial_id,
            attempt=intent.attempt,
            artifact_dir=store.attempt_dir,
            artifact_dir_ref=str(store.attempt_dir.relative_to(self.root)),
            log_path=log_path,
            return_code=return_code,
            expected_failure_reason=intent.expected_failure_reason,
            artifact_root_ref=config.artifact_root_ref,
            artifact_refs=config.artifact_refs,
        )
        store.write_json("analysis.json", analysis.to_payload())
        store.path("analysis.md").write_text(self._analysis_markdown(analysis), encoding="utf-8")
        store.lifecycle("analysis_finished", status=analysis.status, failure_reason=analysis.failure_reason.value)
        for action in analysis.actions:
            store.action(
                action.get("action", "record_analysis_action"),
                reason=action.get("reason", "analysis action"),
                outcome=action.get("outcome", "recorded"),
            )
        return analysis

    def _run_process(self, command: list[str], *, env: dict[str, str], log_path: Path, store: TrialArtifactStore) -> int:
        start = time.monotonic()
        last_growth = start
        last_size = 0
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                command,
                cwd=self.root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            store.lifecycle("launcher_started", pid=proc.pid)
            while proc.poll() is None:
                now = time.monotonic()
                size = log_path.stat().st_size if log_path.exists() else 0
                if size > last_size:
                    last_growth = now
                    last_size = size
                if now - start > self.max_trial_wall_time_sec or now - last_growth > self.stall_timeout_sec:
                    store.lifecycle("timeout", pid=proc.pid)
                    self.terminate_process_group(proc, grace_seconds=30.0)
                    return 124
                time.sleep(1.0)
            return proc.returncode or 0

    def _analysis_markdown(self, analysis: TrialAnalysis) -> str:
        return (
            f"# Trial Analysis\n\n"
            f"- trial: `{analysis.trial_id}`\n"
            f"- attempt: `{analysis.attempt}`\n"
            f"- status: `{analysis.status}`\n"
            f"- failure_reason: `{analysis.failure_reason.value}`\n"
            f"- root_cause: `{analysis.root_cause}`\n"
        )
```

- [ ] **Step 4: Run trial runner tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_trial_runner.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add qwen-vl-finetune/research/observability/trial_runner.py qwen-vl-finetune/tests/research/observability/test_trial_runner.py
git commit -m "feat: add supervised trial runner"
```

## Task 6: TSV Compatibility Helper

**Files:**
- Modify: `qwen-vl-finetune/research/runner.py`
- Test: `qwen-vl-finetune/tests/research/test_runner_dry_run.py`

- [ ] **Step 1: Add a dry-run test for attempt artifacts and TSV rows**

Modify `qwen-vl-finetune/tests/research/test_runner_dry_run.py` to include:

```python
from pathlib import Path

from research.models import TrialSpec
from research.runner import append_result_row_from_analysis
from research.observability.trial_runner import SupervisorTrialRunner


def test_append_result_row_from_analysis_preserves_profiled_tsv(tmp_path: Path) -> None:
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts" / "sft_qwen3_8b.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    spec = TrialSpec("b200", "probe", "trial", {"BATCH_SIZE": "2"})
    analysis = SupervisorTrialRunner(root).run_from_spec(spec, dry_run=True)
    result_path = root / "experiments" / "results.profiled.tsv"

    append_result_row_from_analysis(result_path, spec, analysis, git_commit="abc123")

    text = result_path.read_text(encoding="utf-8")
    assert "trial" in text
    assert "\tok\t" in text
    assert "outputs/b200/probe/trial/attempt_001" in text
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_runner_dry_run.py::test_append_result_row_from_analysis_preserves_profiled_tsv -v
```

Expected: FAIL with missing `append_result_row_from_analysis`.

- [ ] **Step 3: Update runner TSV compatibility**

Modify `qwen-vl-finetune/research/runner.py`:

```python
from research.observability.schema import TrialAnalysis
from research.observability.trial_runner import SupervisorTrialRunner
```

Add this function near `append_result_row`:

```python
def append_result_row_from_analysis(
    path: Path,
    spec: TrialSpec,
    analysis: TrialAnalysis,
    *,
    git_commit: str,
) -> None:
    append_result_row(
        path,
        spec,
        analysis.metrics,
        git_commit=git_commit,
        log_path=Path(analysis.artifact_refs["full_log"]),
        output_dir=Path(analysis.artifact_refs["output_dir"]),
    )
```

Keep the existing `run_trial()` function until the Temporal activity is migrated in Task 8. Do not route local campaign commands directly through `SupervisorTrialRunner`; local commands are made Temporal-backed in Task 10.

- [ ] **Step 4: Run runner dry-run tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_runner_dry_run.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all research tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add qwen-vl-finetune/research/runner.py qwen-vl-finetune/tests/research/test_runner_dry_run.py
git commit -m "feat: add trial analysis tsv helper"
```

## Task 7: Optional Trainer JSONL Events

**Files:**
- Modify: `qwen-vl-finetune/qwenvl/train/train_qwen.py`
- Test: `qwen-vl-finetune/tests/research/observability/test_trainer_events.py`

- [ ] **Step 1: Write trainer event helper test**

Create `qwen-vl-finetune/tests/research/observability/test_trainer_events.py`:

```python
import json
from pathlib import Path

from qwenvl.train.train_qwen import research_event


def test_research_event_writes_jsonl_when_enabled(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("RESEARCH_EVENTS_PATH", str(path))
    monkeypatch.setenv("RESEARCH_TRIAL_ID", "b200/probe/t1")

    research_event("trainer_started", rank=0)

    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["schema_version"] == 1
    assert payload["event"] == "trainer_started"
    assert payload["trial_id"] == "b200/probe/t1"
    assert payload["rank"] == 0


def test_research_event_noops_when_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RESEARCH_EVENTS_PATH", raising=False)

    research_event("trainer_started")

    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_trainer_events.py -v
```

Expected: FAIL with missing `research_event`.

- [ ] **Step 3: Add trainer event helper and call sites**

Modify `qwen-vl-finetune/qwenvl/train/train_qwen.py` near imports:

```python
import json
from datetime import datetime, timezone
```

Add near `rank0_print`:

```python
def research_event(event: str, **payload):
    path = os.environ.get("RESEARCH_EVENTS_PATH")
    if not path:
        return
    record = {
        "schema_version": 1,
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "trial_id": os.environ.get("RESEARCH_TRIAL_ID", ""),
    }
    record.update(payload)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
```

Add these calls in `train()`:

```python
research_event("trainer_started", local_rank=local_rank)
```

after model construction:

```python
research_event("model_loaded", model_class=model.__class__.__name__, model_name_or_path=model_args.model_name_or_path)
```

after `make_supervised_data_module(...)`:

```python
research_event("data_module_ready", do_train=training_args.do_train, do_eval=training_args.do_eval)
```

inside the world-process-zero footer block before printing footers:

```python
research_event(
    "trainer_footer",
    val_loss=val_loss,
    peak_vram_mb=peak_vram_mb,
    world_process_zero=True,
)
```

- [ ] **Step 4: Run trainer event tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_trainer_events.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add qwen-vl-finetune/qwenvl/train/train_qwen.py qwen-vl-finetune/tests/research/observability/test_trainer_events.py
git commit -m "feat: emit optional trainer events"
```

## Task 8: Temporal Activity and Workflow Integration

**Files:**
- Modify: `qwen-vl-finetune/research/activities.py`
- Modify: `qwen-vl-finetune/research/workflows.py`
- Test: `qwen-vl-finetune/tests/research/test_workflows_import.py`

- [ ] **Step 1: Extend Temporal import/query tests**

Modify `qwen-vl-finetune/tests/research/test_workflows_import.py`:

```python
def test_temporal_modules_import() -> None:
    from research.activities import load_profile_activity, plan_probe_trials_activity, run_trial_activity
    from research.workflows import ProfiledExperimentWorkflow

    assert load_profile_activity is not None
    assert plan_probe_trials_activity is not None
    assert run_trial_activity is not None
    assert ProfiledExperimentWorkflow is not None


def test_workflow_status_shape_includes_latest_analysis() -> None:
    from research.workflows import ProfiledExperimentWorkflow

    workflow = ProfiledExperimentWorkflow()
    status = workflow.status()

    assert "latest_metrics" in status
    assert "latest_analysis" in status
    assert status["latest_analysis"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_workflows_import.py -v
```

Expected: FAIL because `latest_analysis` is absent.

- [ ] **Step 3: Update activity to return analysis payload**

Modify `qwen-vl-finetune/research/activities.py` imports:

```python
from research.observability.trial_runner import SupervisorTrialRunner
from research.runner import append_result_row_from_analysis
```

Replace the body of `run_trial_activity` with:

```python
@activity.defn
def run_trial_activity(spec_payload: dict, dry_run: bool = False) -> dict:
    root = repo_root()
    spec = TrialSpec(**spec_payload)
    analysis = SupervisorTrialRunner(root).run_from_spec(spec, dry_run=dry_run)
    append_result_row_from_analysis(
        results_path(root),
        spec,
        analysis,
        git_commit=git_short(root),
    )
    return analysis.to_payload()
```

- [ ] **Step 4: Update workflow status and metric compatibility**

Modify `qwen-vl-finetune/research/workflows.py`:

```python
self.latest_analysis: dict | None = None
```

In `_run_trial_specs`, after the activity returns:

```python
analysis = await workflow.execute_activity(...)
self.latest_analysis = analysis
self.latest_metrics = analysis.get("metrics")
rows.append({"spec": spec, "analysis": analysis, "metrics": analysis.get("metrics")})
```

Update `status()`:

```python
"latest_analysis": self.latest_analysis,
```

Keep `latest_metrics` unchanged in the returned dict.

- [ ] **Step 5: Add signal state for cancel/rerun tracking**

In `ProfiledExperimentWorkflow.__init__`, add:

```python
self.cancel_requested = False
self.rerun_trials: list[str] = []
```

Add signal methods:

```python
@workflow.signal
def cancel_campaign(self) -> None:
    self.cancel_requested = True
    self.stop_after_phase = True

@workflow.signal
def rerun_trial(self, trial: str) -> None:
    self.rerun_trials.append(trial)
```

Add these fields to `status()`:

```python
"cancel_requested": self.cancel_requested,
"rerun_trials": list(self.rerun_trials),
```

Do not implement rerun execution in this task; this signal records the queue and attempt tracking support added in Task 5 preserves evidence once execution is wired.

- [ ] **Step 6: Run Temporal tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_workflows_import.py -v
```

Expected: PASS.

- [ ] **Step 7: Run all research tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add qwen-vl-finetune/research/activities.py qwen-vl-finetune/research/workflows.py qwen-vl-finetune/tests/research/test_workflows_import.py
git commit -m "feat: return trial analysis from temporal activities"
```

## Task 9: Campaign Summarizer

**Files:**
- Create: `qwen-vl-finetune/research/observability/summarizer.py`
- Modify: `qwen-vl-finetune/research/runner.py`
- Modify: `qwen-vl-finetune/research/activities.py`
- Test: `qwen-vl-finetune/tests/research/observability/test_summarizer.py`

- [ ] **Step 1: Write summarizer tests**

Create `qwen-vl-finetune/tests/research/observability/test_summarizer.py`:

```python
import json
from pathlib import Path

from research.observability.summarizer import summarize_campaign


def test_summarizer_uses_analysis_json_as_authoritative(tmp_path: Path) -> None:
    attempt = tmp_path / "experiments" / "runs" / "b200" / "probe" / "trial" / "attempt_001"
    attempt.mkdir(parents=True)
    (attempt / "analysis.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trial_id": "b200/probe/trial",
                "attempt": 1,
                "status": "ok",
                "failure_reason": "none",
                "root_cause": "none",
                "expected_failure": False,
                "metrics": {
                    "status": "ok",
                    "val_loss": 0.5,
                    "peak_vram_mb": 100.0,
                    "throughput_steps_per_sec": 2.0,
                    "failure_reason": "none",
                },
                "symptoms": [],
                "evidence_refs": [],
                "actions": [],
                "recommendations": [],
                "artifact_dir": str(attempt),
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_campaign(tmp_path, profile="b200")

    assert "fastest safe configuration" in summary
    assert "b200/probe/trial" in summary
    assert "0.5" in summary
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_summarizer.py -v
```

Expected: FAIL with missing summarizer.

- [ ] **Step 3: Implement summarizer**

Create `qwen-vl-finetune/research/observability/summarizer.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_analysis(root: Path, profile: str) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    base = root / "experiments" / "runs" / profile
    for path in sorted(base.glob("*/*/attempt_*/analysis.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            continue
        analyses.append(payload)
    return analyses


def summarize_campaign(root: Path, *, profile: str) -> str:
    analyses = _load_analysis(root, profile)
    ok = [a for a in analyses if a.get("status") == "ok"]
    crashes = [a for a in analyses if a.get("status") != "ok"]
    fastest = sorted(
        ok,
        key=lambda a: a.get("metrics", {}).get("throughput_steps_per_sec") or 0.0,
        reverse=True,
    )
    best_val = sorted(
        ok,
        key=lambda a: a.get("metrics", {}).get("val_loss")
        if a.get("metrics", {}).get("val_loss") is not None
        else float("inf"),
    )
    lines = ["# Qwen3-VL Research Summary", "", f"Profile: `{profile}`", ""]
    lines.append("## Fastest Safe Configuration")
    if fastest:
        first = fastest[0]
        lines.append(f"- {first['trial_id']} attempt={first['attempt']} throughput={first['metrics'].get('throughput_steps_per_sec')}")
    else:
        lines.append("No successful analysis artifacts found.")
    lines.append("")
    lines.append("## Best Validation Configuration")
    if best_val:
        first = best_val[0]
        lines.append(f"- {first['trial_id']} attempt={first['attempt']} val_loss={first['metrics'].get('val_loss')}")
    else:
        lines.append("No successful analysis artifacts found.")
    lines.append("")
    lines.append("## Crashes")
    if crashes:
        for analysis in crashes:
            lines.append(f"- {analysis['trial_id']} attempt={analysis['attempt']} root_cause={analysis.get('root_cause')}")
    else:
        lines.append("No crashes recorded.")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Wire summarizer into runner and activity**

Modify `command_summarize` in `qwen-vl-finetune/research/runner.py`:

```python
from research.observability.summarizer import summarize_campaign
```

At the start of `command_summarize`, after `root = repo_root()`:

```python
analysis_summary = summarize_campaign(root, profile=args.profile)
if "No successful analysis artifacts found." not in analysis_summary or (root / "experiments" / "runs" / args.profile).exists():
    path = root / "experiments" / "summary.md"
    path.write_text(analysis_summary, encoding="utf-8")
    print(path)
    return 0
```

Keep the existing TSV-based code below as fallback for pre-migration trials.

Modify `summarize_campaign_activity` in `qwen-vl-finetune/research/activities.py` to call `summarize_campaign(root, profile=profile)` first and write `experiments/summary.md`.

- [ ] **Step 5: Run summarizer tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/observability/test_summarizer.py -v
```

Expected: PASS.

- [ ] **Step 6: Run all research tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add qwen-vl-finetune/research/observability/summarizer.py qwen-vl-finetune/research/runner.py qwen-vl-finetune/research/activities.py qwen-vl-finetune/tests/research/observability/test_summarizer.py
git commit -m "feat: summarize trial analyses"
```

## Task 10: Local Temporal Runner Harness

**Files:**
- Create: `qwen-vl-finetune/research/local_temporal.py`
- Modify: `qwen-vl-finetune/research/runner.py`
- Test: `qwen-vl-finetune/tests/research/test_local_temporal.py`

- [ ] **Step 1: Write local Temporal harness tests**

Create `qwen-vl-finetune/tests/research/test_local_temporal.py`:

```python
import shutil
import subprocess
from pathlib import Path

import pytest

from research.local_temporal import (
    LOCAL_TEMPORAL_DB,
    build_start_dev_command,
    default_temporal_db_path,
    ensure_temporal_cli,
)


def test_build_start_dev_command_uses_file_backed_sqlite(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    command = build_start_dev_command(tmp_path)

    assert command[:3] == ["temporal", "server", "start-dev"]
    assert "--db-filename" in command
    db_path = command[command.index("--db-filename") + 1]
    assert db_path == str(home / LOCAL_TEMPORAL_DB)


def test_build_start_dev_command_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / ".artifacts" / "temporal" / "temporal.db"
    monkeypatch.setenv("RESEARCH_TEMPORAL_DB", str(configured))

    assert default_temporal_db_path() == configured


def test_build_start_dev_command_creates_parent_dir(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    command = build_start_dev_command(tmp_path)
    db_path = Path(command[command.index("--db-filename") + 1])

    assert db_path.parent.exists()


def test_ensure_temporal_cli_fails_fast_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="temporal CLI not found"):
        ensure_temporal_cli()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_local_temporal.py -v
```

Expected: FAIL with missing `research.local_temporal`.

- [ ] **Step 3: Implement local Temporal harness helpers**

Create `qwen-vl-finetune/research/local_temporal.py`:

```python
from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from temporalio.client import Client


TEMPORAL_DB_ENV = "RESEARCH_TEMPORAL_DB"
LOCAL_TEMPORAL_DB = Path(".automata") / "research" / "temporal" / "temporal.db"
DEFAULT_ADDRESS = "localhost:7233"
DEFAULT_TASK_QUEUE = "qwen3vl-local"


def default_temporal_db_path() -> Path:
    configured = os.environ.get(TEMPORAL_DB_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / LOCAL_TEMPORAL_DB


def ensure_temporal_cli() -> str:
    temporal = shutil.which("temporal")
    if temporal is None:
        raise RuntimeError(
            "temporal CLI not found on PATH; local runs require "
            "`temporal server start-dev --db-filename ~/.automata/research/temporal/temporal.db`"
        )
    return temporal


def build_start_dev_command(root: Path) -> list[str]:
    db_path = default_temporal_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return ["temporal", "server", "start-dev", "--db-filename", str(db_path)]


async def temporal_is_reachable(address: str = DEFAULT_ADDRESS) -> bool:
    try:
        await Client.connect(address)
    except Exception:
        return False
    return True


async def ensure_local_temporal_server(root: Path, address: str = DEFAULT_ADDRESS) -> subprocess.Popen | None:
    if await temporal_is_reachable(address):
        return None
    ensure_temporal_cli()
    proc = subprocess.Popen(
        build_start_dev_command(root),
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    for _ in range(120):
        if await temporal_is_reachable(address):
            return proc
        await asyncio.sleep(1)
    proc.terminate()
    raise RuntimeError(f"local Temporal server did not become reachable at {address}")


def terminate_local_temporal_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    if proc.poll() is None:
        os.killpg(pgid, signal.SIGKILL)
```

Do not call `research.start.main` or `research.worker.main` from this module; their current CLIs parse process arguments. Workflow start/worker orchestration is wired in `runner.py` in the next step.

- [ ] **Step 4: Update runner to make local campaign commands Temporal-backed**

Modify `qwen-vl-finetune/research/runner.py`:

```python
import asyncio
from datetime import datetime, timezone
from temporalio.client import Client
from temporalio.worker import Worker
from research.local_temporal import (
    DEFAULT_ADDRESS,
    DEFAULT_TASK_QUEUE,
    ensure_local_temporal_server,
    terminate_local_temporal_server,
)
from research.activities import (
    load_profile_activity,
    plan_probe_trials_activity,
    plan_sweep_trials_activity,
    run_trial_activity,
    select_capacity_from_results_activity,
    select_capacity_activity,
    summarize_campaign_activity,
)
from research.workflows import ProfiledExperimentWorkflow
```

Add:

```python
async def run_temporal_command(profile: str, command: str, dry_run: bool) -> dict:
    root = repo_root()
    server_proc = await ensure_local_temporal_server(root, DEFAULT_ADDRESS)
    client = await Client.connect(DEFAULT_ADDRESS)
    worker = Worker(
        client,
        task_queue=DEFAULT_TASK_QUEUE,
        workflows=[ProfiledExperimentWorkflow],
        activities=[
            load_profile_activity,
            plan_probe_trials_activity,
            plan_sweep_trials_activity,
            run_trial_activity,
            select_capacity_from_results_activity,
            select_capacity_activity,
            summarize_campaign_activity,
        ],
    )
    worker_task = asyncio.create_task(worker.run())
    try:
        handle = await client.start_workflow(
            ProfiledExperimentWorkflow.run,
            args=[profile, dry_run, command],
            id=f"qwen3vl-{profile}-{command}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            task_queue=DEFAULT_TASK_QUEUE,
        )
        result = await handle.result()
        return result
    finally:
        await worker.shutdown()
        await worker_task
        terminate_local_temporal_server(server_proc)
```

Change `main()` so normal commands call Temporal:

```python
if args.command in {"probe", "select", "sweep", "summarize"}:
    asyncio.run(run_temporal_command(args.profile, args.command, args.dry_run))
    return 0
```

Keep `subagents` as a direct local command because it only prints implementation commands and does not run experiments.

- [ ] **Step 5: Run local Temporal tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research/test_local_temporal.py -v
```

Expected: PASS.

- [ ] **Step 6: Run all research tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add qwen-vl-finetune/research/local_temporal.py qwen-vl-finetune/research/runner.py qwen-vl-finetune/tests/research/test_local_temporal.py
git commit -m "feat: run local campaigns through temporal"
```

## Task 11: End-to-End Dry Run Verification

**Files:**
- Read: all files changed above
- Generated during test: `qwen-vl-finetune/experiments/runs/b200/probe/...`

- [ ] **Step 1: Run all research tests**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. pytest tests/research -v
```

Expected: PASS.

- [ ] **Step 2: Check Temporal CLI availability**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
command -v temporal
```

Expected when the CLI is installed: prints the `temporal` executable path.
Expected when the CLI is absent: exit code nonzero; install the Temporal CLI or
stop before running the local campaign command because the runner must fail fast
rather than falling back to direct execution.

- [ ] **Step 3: Run a Temporal-backed dry-run B200 probe**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. python -m research.runner probe --profile b200 --dry-run
```

Expected: starts or connects to local Temporal at `localhost:7233`, uses
`~/.automata/research/temporal/temporal.db`, exits `0`, and writes attempt directories under
`experiments/runs/b200/probe/*/attempt_*`.

- [ ] **Step 4: Verify mandatory artifacts exist**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
first_attempt=$(find experiments/runs/b200/probe -path '*/attempt_001' -type d | sort | head -n 1)
test -f "$first_attempt/intent.json"
test -f "$first_attempt/resolved_config.json"
test -f "$first_attempt/system.pre.json"
test -f "$first_attempt/lifecycle.jsonl"
test -f "$first_attempt/system.post.json"
test -f "$first_attempt/analysis.json"
test -f "$first_attempt/analysis.md"
test ! -e "$first_attempt/run.log"
test ! -e "$first_attempt/output"
```

Expected: exit code `0`.

- [ ] **Step 5: Verify heavy artifacts are under the artifact root**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
first_attempt=$(find experiments/runs/b200/probe -path '*/attempt_001' -type d | sort | head -n 1)
export first_attempt
log_ref=$(python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["first_attempt"]) / "analysis.json"
print(json.loads(p.read_text())["artifact_refs"]["full_log"])
PY
)
test -f "$HOME/.automata/research/experiments/$log_ref"
```

Expected: exit code `0`; the log reference is relative and resolves under `~/.automata/research/experiments`.

- [ ] **Step 6: Verify no absolute local paths or secret values are present in resolved config**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
first_attempt=$(find experiments/runs/b200/probe -path '*/attempt_001' -type d | sort | head -n 1)
! rg '/data02/|/home/|file:///' "$first_attempt/resolved_config.json" "$first_attempt/analysis.json"
if rg -i 'HF_TOKEN|AWS_SECRET_ACCESS_KEY|PASSWORD' "$first_attempt/resolved_config.json"; then
  rg '\\*\\*\\*REDACTED\\*\\*\\*' "$first_attempt/resolved_config.json"
fi
```

Expected: no absolute local paths; any secret-like keys that appear have `***REDACTED***` values.

- [ ] **Step 7: Generate summary through Temporal**

Run:

```bash
cd /data02/home/philip.yang/workspace/qwen3_vl/qwen-vl-finetune
PYTHONPATH=. python -m research.runner summarize --profile b200
sed -n '1,120p' experiments/summary.md
```

Expected: summary references fastest safe configuration or reports that no successful analysis artifacts exist.

- [ ] **Step 8: Commit verification fixes if needed**

If any verification command exposed a defect, fix it with a focused commit:

```bash
git add qwen-vl-finetune/research qwen-vl-finetune/tests
git commit -m "fix: complete trial observability verification"
```

If no fixes were needed, do not create an empty commit.
