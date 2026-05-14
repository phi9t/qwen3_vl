# Scientific Experiment Methodology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the split `autonomy/` and `qwen-vl-finetune/research/` orchestration with one top-level `research/` scientific experiment methodology and a Qwen-VL adapter.

**Architecture:** Build a minimal generic core first: SQLite state, adapter contracts, artifact/preflight helpers, Temporal workflows, and CLI/mise tasks. Then migrate Qwen-specific probing and trial execution behind an adapter. Remove the old `autonomy/` methodology and reduce `qwen-vl-finetune/research/` to temporary wrappers or delete it once tests prove the new path works.

**Tech Stack:** Python 3.10+, `uv`, SQLite from the standard library, Temporal Python SDK, `pytest`, `mise`, existing Qwen-VL shell launchers and observability code.

---

## File Structure

Create generic methodology files:

- `research/__init__.py`: package exports and version marker.
- `research/models.py`: generic dataclasses and enums for intents, experiments, trials, preflight, progress, reports, and adapter commands.
- `research/db.py`: SQLite schema, connection helper, inserts, reads, and narrow state transition functions.
- `research/adapters.py`: `ExperimentAdapter` protocol and adapter registry/import helper.
- `research/artifacts.py`: artifact root resolution and per-attempt directory creation.
- `research/preflight.py`: generic preflight checks plus adapter preflight dispatch.
- `research/runners.py`: generic subprocess trial runner, progress polling, cancellation-safe process group teardown.
- `research/reports.py`: report JSON/Markdown rendering.
- `research/activities.py`: Temporal activity functions that call DB, preflight, runner, and report helpers.
- `research/workflows.py`: `ProbeWorkflow` and `TrialWorkflow`.
- `research/temporal.py`: local Temporal dev-server helpers adapted from `qwen-vl-finetune/research/local_temporal.py`.
- `research/cli.py`: `research db init`, `research probe`, `research manager`, `research status`, `research report`.

Create generic tests:

- `tests/research/test_db.py`
- `tests/research/test_adapters.py`
- `tests/research/test_artifacts.py`
- `tests/research/test_preflight.py`
- `tests/research/test_reports.py`
- `tests/research/test_workflows.py`
- `tests/research/test_cli.py`
- `tests/research/fake_adapter.py`

Create Qwen adapter files:

- `qwen-vl-finetune/experiments/__init__.py`
- `qwen-vl-finetune/experiments/qwen_adapter.py`
- `qwen-vl-finetune/experiments/profiles/b200.env`
- `qwen-vl-finetune/experiments/profiles/h100.env`
- `qwen-vl-finetune/experiments/profiles/a100.env`
- `qwen-vl-finetune/tests/experiments/test_qwen_adapter.py`

Modify:

- `pyproject.toml`: add project metadata, `research` console script, dependencies, and first-party package names.
- `mise.toml`: add thin research tasks.
- `.gitignore`: ignore local research DB/artifacts if not already ignored.
- `qwen-vl-finetune/research/`: delete after Qwen adapter coverage exists, because it conflicts with the new top-level `research` package name.
- `autonomy/`: delete after generic tests replace its useful coverage.

Do not modify generated `__pycache__`, `.pytest_cache`, `.venv`, or existing untracked runtime files under `qwen-vl-finetune/experiments/results.tsv`.

---

### Task 1: Root Package Metadata and Empty Generic Package

**Files:**
- Modify: `pyproject.toml`
- Create: `research/__init__.py`
- Create: `tests/research/test_imports.py`

- [ ] **Step 1: Write the failing import and script metadata test**

Create `tests/research/test_imports.py`:

```python
import tomllib
from pathlib import Path


def test_research_package_imports() -> None:
    import research

    assert research.__version__ == "0.1.0"


def test_pyproject_exposes_research_script() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "qwen3-vl-research"
    assert pyproject["project"]["scripts"]["research"] == "research.cli:main"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/research/test_imports.py -v
```

Expected: FAIL because `research` and `[project]` metadata do not exist yet.

- [ ] **Step 3: Add minimal package metadata**

Modify `pyproject.toml` by adding this block before `[tool.ruff]`:

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "qwen3-vl-research"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "temporalio>=1.7",
]

[project.scripts]
research = "research.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["research"]
```

Update `[tool.ruff.lint.isort]` in `pyproject.toml`:

```toml
known-first-party = ["research", "qwenvl", "qwen_vl_utils"]
```

Create `research/__init__.py`:

```python
"""Generic scientific experiment methodology."""

__version__ = "0.1.0"
```

Create temporary `research/cli.py` so the script target imports:

```python
from __future__ import annotations


def main() -> int:
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/research/test_imports.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml research/__init__.py research/cli.py tests/research/test_imports.py
git commit -m "feat: add generic research package"
```

---

### Task 2: Generic Models and Adapter Contract

**Files:**
- Create: `research/models.py`
- Create: `research/adapters.py`
- Create: `tests/research/fake_adapter.py`
- Create: `tests/research/test_adapters.py`

- [ ] **Step 1: Write failing adapter contract tests**

Create `tests/research/fake_adapter.py`:

```python
from __future__ import annotations

from research.models import (
    Intent,
    PreflightResult,
    ProbeRequest,
    ProgressUpdate,
    TrialCommand,
    TrialContext,
    TrialReport,
)


class FakeAdapter:
    name = "fake"

    def generate_probe_intents(self, request: ProbeRequest) -> list[Intent]:
        return [
            Intent(
                adapter=self.name,
                model=request.model,
                profile=request.profile,
                phase="probe",
                name="small",
                config={"batch_size": 1},
                objective=request.objective,
                source="probe",
            )
        ]

    def preflight(self, intent: Intent, context: TrialContext) -> PreflightResult:
        return PreflightResult(ok=True, checks={"fake": "ok"}, message="ok")

    def build_trial(self, intent: Intent, context: TrialContext) -> TrialCommand:
        return TrialCommand(argv=["python", "-c", "print('metric=1.0')"], env={})

    def parse_progress(self, event_or_log_line: str) -> ProgressUpdate | None:
        if "metric=" not in event_or_log_line:
            return None
        return ProgressUpdate(step=1, metrics={"metric": 1.0}, message=event_or_log_line.strip())

    def analyze_result(self, context: TrialContext) -> TrialReport:
        return TrialReport(
            status="succeeded",
            metrics={"metric": 1.0},
            failure={},
            summary="fake trial succeeded",
        )
```

Create `tests/research/test_adapters.py`:

```python
from pathlib import Path

from research.adapters import load_adapter
from research.models import ProbeRequest, TrialContext


def test_load_adapter_from_module_class_path() -> None:
    adapter = load_adapter("tests.research.fake_adapter:FakeAdapter")

    assert adapter.name == "fake"


def test_load_adapter_from_file_class_path() -> None:
    adapter = load_adapter("tests/research/fake_adapter.py:FakeAdapter")

    assert adapter.name == "fake"


def test_fake_adapter_generates_intents_and_trial_command(tmp_path: Path) -> None:
    adapter = load_adapter("tests.research.fake_adapter:FakeAdapter")
    request = ProbeRequest(model="unit-model", profile="cpu", objective={"metric": "min"})
    intents = adapter.generate_probe_intents(request)
    context = TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "research.sqlite",
    )

    command = adapter.build_trial(intents[0], context)

    assert intents[0].config == {"batch_size": 1}
    assert command.argv[:2] == ["python", "-c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/research/test_adapters.py -v
```

Expected: FAIL because `research.models` and `research.adapters` do not exist.

- [ ] **Step 3: Implement generic models**

Create `research/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


JsonDict = dict[str, object]


@dataclass(frozen=True)
class ProbeRequest:
    model: str
    profile: str
    objective: JsonDict = field(default_factory=dict)
    budget: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class Intent:
    adapter: str
    model: str
    profile: str
    phase: str
    name: str
    config: JsonDict
    objective: JsonDict = field(default_factory=dict)
    source: str = "manual"


@dataclass(frozen=True)
class TrialContext:
    experiment_id: int
    trial_run_id: int
    attempt: int
    worktree: Path
    artifact_dir: Path
    db_path: Path


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    checks: JsonDict = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class TrialCommand:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None


@dataclass(frozen=True)
class ProgressUpdate:
    step: int | None = None
    metrics: JsonDict = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class TrialReport:
    status: str
    metrics: JsonDict = field(default_factory=dict)
    failure: JsonDict = field(default_factory=dict)
    summary: str = ""


class ExperimentAdapter(Protocol):
    name: str

    def generate_probe_intents(self, request: ProbeRequest) -> list[Intent]:
        ...

    def preflight(self, intent: Intent, context: TrialContext) -> PreflightResult:
        ...

    def build_trial(self, intent: Intent, context: TrialContext) -> TrialCommand:
        ...

    def parse_progress(self, event_or_log_line: str) -> ProgressUpdate | None:
        ...

    def analyze_result(self, context: TrialContext) -> TrialReport:
        ...
```

- [ ] **Step 4: Implement adapter loader**

Create `research/adapters.py`:

```python
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from research.models import ExperimentAdapter


DEFAULT_ADAPTERS = {
    "qwen_vl": "qwen-vl-finetune/experiments/qwen_adapter.py:QwenVlAdapter",
}


def load_adapter(name_or_path: str) -> ExperimentAdapter:
    target = DEFAULT_ADAPTERS.get(name_or_path, name_or_path)
    if ":" not in target:
        raise ValueError(f"Expected adapter path 'module:Class' or 'file.py:Class', got {name_or_path!r}")
    module_name, class_name = target.split(":", 1)
    if module_name.endswith(".py") or "/" in module_name:
        module_path = Path(module_name)
        if not module_path.is_absolute():
            module_path = Path.cwd() / module_path
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load adapter module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    adapter = adapter_cls()
    required = [
        "generate_probe_intents",
        "preflight",
        "build_trial",
        "parse_progress",
        "analyze_result",
    ]
    missing = [name for name in required if not callable(getattr(adapter, name, None))]
    if missing:
        raise TypeError(f"Adapter {target} is missing methods: {', '.join(missing)}")
    return adapter
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/research/test_adapters.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add research/models.py research/adapters.py tests/research/fake_adapter.py tests/research/test_adapters.py
git commit -m "feat: define research adapter contract"
```

---

### Task 3: SQLite Experiment Database

**Files:**
- Create: `research/db.py`
- Create: `tests/research/test_db.py`

- [ ] **Step 1: Write failing DB tests**

Create `tests/research/test_db.py`:

```python
from pathlib import Path

from research.db import (
    create_experiment,
    create_trial_run,
    get_experiment,
    get_intent,
    init_db,
    insert_intent,
    transition_experiment,
    transition_intent,
    transition_trial_run,
)
from research.models import Intent


def test_init_db_and_insert_intent(tmp_path: Path) -> None:
    db_path = tmp_path / "research.sqlite"
    init_db(db_path)

    intent_id = insert_intent(
        db_path,
        Intent(
            adapter="fake",
            model="unit-model",
            profile="cpu",
            phase="probe",
            name="small",
            config={"batch_size": 1},
            objective={"metric": "min"},
            source="probe",
        ),
    )

    row = get_intent(db_path, intent_id)
    assert row["status"] == "candidate"
    assert row["config_json"] == {"batch_size": 1}


def test_experiment_and_trial_state_transitions(tmp_path: Path) -> None:
    db_path = tmp_path / "research.sqlite"
    init_db(db_path)
    intent_id = insert_intent(
        db_path,
        Intent("fake", "unit-model", "cpu", "probe", "small", {"batch_size": 1}),
    )
    experiment_id = create_experiment(
        db_path,
        intent_id=intent_id,
        adapter="fake",
        artifact_root=str(tmp_path / "artifacts"),
        artifact_subdir="fake/1",
    )
    trial_run_id = create_trial_run(db_path, experiment_id=experiment_id, attempt=1)

    transition_intent(db_path, intent_id, "selected", score={"metric": 1.0})
    transition_experiment(db_path, experiment_id, "running", workflow_id="wf-1")
    transition_trial_run(
        db_path,
        trial_run_id,
        "succeeded",
        heartbeat={"step": 1},
        metrics={"metric": 1.0},
        failure={},
        report_path="report.md",
    )

    intent = get_intent(db_path, intent_id)
    experiment = get_experiment(db_path, experiment_id)
    assert intent["status"] == "selected"
    assert intent["score_json"] == {"metric": 1.0}
    assert experiment["status"] == "running"
    assert experiment["temporal_workflow_id"] == "wf-1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/research/test_db.py -v
```

Expected: FAIL because `research.db` does not exist.

- [ ] **Step 3: Implement SQLite schema and helpers**

Create `research/db.py` with these functions:

```python
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.models import Intent


JsonDict = dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _dump(value: JsonDict | None) -> str:
    return json.dumps(value or {}, sort_keys=True)


def _row(row: sqlite3.Row) -> JsonDict:
    result = dict(row)
    for key in ("config_json", "objective_json", "score_json", "heartbeat_json", "metrics_json", "failure_json", "metadata_json"):
        if key in result and isinstance(result[key], str):
            result[key] = json.loads(result[key] or "{}")
    return result


def init_db(db_path: Path) -> None:
    with closing(connect(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS intents (
              id INTEGER PRIMARY KEY,
              adapter TEXT NOT NULL,
              model TEXT NOT NULL,
              profile TEXT NOT NULL,
              phase TEXT NOT NULL,
              name TEXT NOT NULL,
              config_json TEXT NOT NULL,
              objective_json TEXT NOT NULL,
              source TEXT NOT NULL,
              status TEXT NOT NULL,
              score_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
              id INTEGER PRIMARY KEY,
              intent_id INTEGER NOT NULL REFERENCES intents(id),
              adapter TEXT NOT NULL,
              status TEXT NOT NULL,
              priority INTEGER NOT NULL,
              temporal_workflow_id TEXT NOT NULL,
              artifact_root TEXT NOT NULL,
              artifact_subdir TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trial_runs (
              id INTEGER PRIMARY KEY,
              experiment_id INTEGER NOT NULL REFERENCES experiments(id),
              attempt INTEGER NOT NULL,
              status TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              heartbeat_json TEXT NOT NULL,
              metrics_json TEXT NOT NULL,
              failure_json TEXT NOT NULL,
              report_path TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
              id INTEGER PRIMARY KEY,
              trial_run_id INTEGER NOT NULL REFERENCES trial_runs(id),
              kind TEXT NOT NULL,
              uri TEXT NOT NULL,
              size_bytes INTEGER,
              metadata_json TEXT NOT NULL
            );
            """
        )
        conn.commit()
```

Append insert/read/transition helpers:

```python
def insert_intent(db_path: Path, intent: Intent) -> int:
    with closing(connect(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO intents (
              adapter, model, profile, phase, name, config_json, objective_json,
              source, status, score_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent.adapter,
                intent.model,
                intent.profile,
                intent.phase,
                intent.name,
                _dump(intent.config),
                _dump(intent.objective),
                intent.source,
                "candidate",
                _dump({}),
                utc_now(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_intent(db_path: Path, intent_id: int) -> JsonDict:
    with closing(connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM intents WHERE id = ?", (intent_id,)).fetchone()
    if row is None:
        raise KeyError(f"intent not found: {intent_id}")
    return _row(row)


def create_experiment(
    db_path: Path,
    *,
    intent_id: int,
    adapter: str,
    artifact_root: str,
    artifact_subdir: str,
    priority: int = 0,
) -> int:
    now = utc_now()
    with closing(connect(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO experiments (
              intent_id, adapter, status, priority, temporal_workflow_id,
              artifact_root, artifact_subdir, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (intent_id, adapter, "queued", priority, "", artifact_root, artifact_subdir, now, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_experiment(db_path: Path, experiment_id: int) -> JsonDict:
    with closing(connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    if row is None:
        raise KeyError(f"experiment not found: {experiment_id}")
    return _row(row)


def create_trial_run(db_path: Path, *, experiment_id: int, attempt: int) -> int:
    with closing(connect(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO trial_runs (
              experiment_id, attempt, status, started_at, finished_at,
              heartbeat_json, metrics_json, failure_json, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (experiment_id, attempt, "preflight", utc_now(), None, _dump({}), _dump({}), _dump({}), ""),
        )
        conn.commit()
        return int(cur.lastrowid)


def transition_intent(db_path: Path, intent_id: int, status: str, *, score: JsonDict | None = None) -> None:
    with closing(connect(db_path)) as conn:
        conn.execute(
            "UPDATE intents SET status = ?, score_json = ? WHERE id = ?",
            (status, _dump(score), intent_id),
        )
        conn.commit()


def transition_experiment(
    db_path: Path,
    experiment_id: int,
    status: str,
    *,
    workflow_id: str | None = None,
) -> None:
    with closing(connect(db_path)) as conn:
        if workflow_id is None:
            conn.execute(
                "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), experiment_id),
            )
        else:
            conn.execute(
                "UPDATE experiments SET status = ?, temporal_workflow_id = ?, updated_at = ? WHERE id = ?",
                (status, workflow_id, utc_now(), experiment_id),
            )
        conn.commit()


def transition_trial_run(
    db_path: Path,
    trial_run_id: int,
    status: str,
    *,
    heartbeat: JsonDict | None = None,
    metrics: JsonDict | None = None,
    failure: JsonDict | None = None,
    report_path: str = "",
) -> None:
    finished_at = utc_now() if status in {"succeeded", "failed", "cancelled"} else None
    with closing(connect(db_path)) as conn:
        conn.execute(
            """
            UPDATE trial_runs
            SET status = ?, finished_at = ?, heartbeat_json = ?, metrics_json = ?,
                failure_json = ?, report_path = ?
            WHERE id = ?
            """,
            (
                status,
                finished_at,
                _dump(heartbeat),
                _dump(metrics),
                _dump(failure),
                report_path,
                trial_run_id,
            ),
        )
        conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/research/test_db.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research/db.py tests/research/test_db.py
git commit -m "feat: add research sqlite state store"
```

---

### Task 4: Artifact and Report Helpers

**Files:**
- Create: `research/artifacts.py`
- Create: `research/reports.py`
- Create: `tests/research/test_artifacts.py`
- Create: `tests/research/test_reports.py`

- [ ] **Step 1: Write failing artifact and report tests**

Create `tests/research/test_artifacts.py`:

```python
from pathlib import Path

from research.artifacts import attempt_dir, default_artifact_root, relative_artifact


def test_default_artifact_root_uses_home(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("RESEARCH_ARTIFACT_ROOT", raising=False)

    assert default_artifact_root() == home / ".automata" / "research" / "artifacts"


def test_attempt_dir_creates_isolated_directory(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    path = attempt_dir(root, adapter="fake", experiment_id=7, attempt=2)

    assert path == root / "fake" / "7" / "attempt-2"
    assert path.exists()


def test_relative_artifact_rejects_external_path(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    inside = root / "fake" / "run.log"
    inside.parent.mkdir(parents=True)
    inside.write_text("log", encoding="utf-8")

    assert relative_artifact(root, inside) == "fake/run.log"
```

Create `tests/research/test_reports.py`:

```python
import json
from pathlib import Path

from research.models import TrialReport
from research.reports import write_report


def test_write_report_json_and_markdown(tmp_path: Path) -> None:
    report = TrialReport(
        status="succeeded",
        metrics={"val_loss": 0.5},
        failure={},
        summary="trial succeeded",
    )

    paths = write_report(tmp_path, report)

    assert json.loads(paths["json"].read_text(encoding="utf-8"))["status"] == "succeeded"
    assert "# Trial Report" in paths["markdown"].read_text(encoding="utf-8")
    assert "trial succeeded" in paths["markdown"].read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/research/test_artifacts.py tests/research/test_reports.py -v
```

Expected: FAIL because helper modules do not exist.

- [ ] **Step 3: Implement artifact helpers**

Create `research/artifacts.py`:

```python
from __future__ import annotations

import os
from pathlib import Path


ARTIFACT_ROOT_ENV = "RESEARCH_ARTIFACT_ROOT"


def default_artifact_root() -> Path:
    configured = os.environ.get(ARTIFACT_ROOT_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".automata" / "research" / "artifacts"


def attempt_dir(root: Path, *, adapter: str, experiment_id: int, attempt: int) -> Path:
    path = root / adapter / str(experiment_id) / f"attempt-{attempt}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_artifact(root: Path, path: Path) -> str:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        return path_resolved.relative_to(root_resolved).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact path {path} is outside artifact root {root}") from exc
```

- [ ] **Step 4: Implement report writer**

Create `research/reports.py`:

```python
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from research.models import TrialReport


def write_report(artifact_dir: Path, report: TrialReport) -> dict[str, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifact_dir / "report.json"
    markdown_path = artifact_dir / "report.md"
    json_path.write_text(
        json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        "# Trial Report\n\n"
        f"Status: `{report.status}`\n\n"
        "## Summary\n\n"
        f"{report.summary or 'No summary.'}\n\n"
        "## Metrics\n\n"
        f"```json\n{json.dumps(report.metrics, indent=2, sort_keys=True)}\n```\n\n"
        "## Failure\n\n"
        f"```json\n{json.dumps(report.failure, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/research/test_artifacts.py tests/research/test_reports.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add research/artifacts.py research/reports.py tests/research/test_artifacts.py tests/research/test_reports.py
git commit -m "feat: add research artifacts and reports"
```

---

### Task 5: Generic Preflight

**Files:**
- Create: `research/preflight.py`
- Create: `tests/research/test_preflight.py`

- [ ] **Step 1: Write failing preflight tests**

Create `tests/research/test_preflight.py`:

```python
from pathlib import Path

from research.adapters import load_adapter
from research.models import Intent, TrialContext
from research.preflight import run_preflight


def test_run_preflight_combines_generic_and_adapter_checks(tmp_path: Path) -> None:
    adapter = load_adapter("tests.research.fake_adapter:FakeAdapter")
    db_path = tmp_path / "research.sqlite"
    db_path.write_text("", encoding="utf-8")
    context = TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        db_path=db_path,
    )
    intent = Intent("fake", "unit-model", "cpu", "probe", "small", {"batch_size": 1})

    result = run_preflight(adapter, intent, context)

    assert result.ok is True
    assert result.checks["adapter"] == "fake"
    assert result.checks["worktree"] == "ok"
    assert result.checks["artifact_dir"] == "ok"
    assert result.checks["fake"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/research/test_preflight.py -v
```

Expected: FAIL because `research.preflight` does not exist.

- [ ] **Step 3: Implement preflight**

Create `research/preflight.py`:

```python
from __future__ import annotations

import shutil

from research.models import ExperimentAdapter, Intent, PreflightResult, TrialContext


def run_preflight(
    adapter: ExperimentAdapter,
    intent: Intent,
    context: TrialContext,
) -> PreflightResult:
    checks: dict[str, object] = {
        "adapter": adapter.name,
        "uv": "ok" if shutil.which("uv") else "missing",
        "db": "ok" if context.db_path.exists() else "missing",
    }
    context.worktree.mkdir(parents=True, exist_ok=True)
    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    checks["worktree"] = "ok" if context.worktree.exists() else "missing"
    checks["artifact_dir"] = "ok" if context.artifact_dir.exists() else "missing"

    adapter_result = adapter.preflight(intent, context)
    checks.update(adapter_result.checks)
    ok = (
        checks["uv"] == "ok"
        and checks["db"] == "ok"
        and checks["worktree"] == "ok"
        and checks["artifact_dir"] == "ok"
        and adapter_result.ok
    )
    message = adapter_result.message if ok else "preflight failed"
    return PreflightResult(ok=ok, checks=checks, message=message)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/research/test_preflight.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research/preflight.py tests/research/test_preflight.py
git commit -m "feat: add generic research preflight"
```

---

### Task 6: Generic Trial Runner

**Files:**
- Create: `research/runners.py`
- Create: `tests/research/test_runners.py`

- [ ] **Step 1: Write failing runner test**

Create `tests/research/test_runners.py`:

```python
from pathlib import Path

from research.adapters import load_adapter
from research.models import Intent, TrialContext
from research.runners import run_trial_command


def test_run_trial_command_writes_log_and_progress(tmp_path: Path) -> None:
    adapter = load_adapter("tests.research.fake_adapter:FakeAdapter")
    context = TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "research.sqlite",
    )
    context.artifact_dir.mkdir()
    intent = Intent("fake", "unit-model", "cpu", "probe", "small", {"batch_size": 1})
    command = adapter.build_trial(intent, context)

    result = run_trial_command(adapter, command, context)

    assert result.returncode == 0
    assert result.progress[-1].metrics == {"metric": 1.0}
    assert (context.artifact_dir / "run.log").read_text(encoding="utf-8").strip() == "metric=1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/research/test_runners.py -v
```

Expected: FAIL because `research.runners` does not exist.

- [ ] **Step 3: Implement subprocess runner**

Create `research/runners.py`:

```python
from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from research.models import ExperimentAdapter, ProgressUpdate, TrialCommand, TrialContext


@dataclass(frozen=True)
class TrialRunResult:
    returncode: int
    log_path: Path
    progress: list[ProgressUpdate] = field(default_factory=list)


def terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=10)


def run_trial_command(
    adapter: ExperimentAdapter,
    command: TrialCommand,
    context: TrialContext,
) -> TrialRunResult:
    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = context.artifact_dir / "run.log"
    env = dict(os.environ)
    env.update(command.env)
    cwd = command.cwd or context.worktree
    progress: list[ProgressUpdate] = []
    proc = subprocess.Popen(
        command.argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                update = adapter.parse_progress(line)
                if update is not None:
                    progress.append(update)
        returncode = proc.wait()
    except BaseException:
        terminate_process_group(proc)
        raise
    return TrialRunResult(returncode=returncode, log_path=log_path, progress=progress)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/research/test_runners.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research/runners.py tests/research/test_runners.py
git commit -m "feat: add generic trial runner"
```

---

### Task 7: Temporal Activities for Trial Execution

**Files:**
- Create: `research/activities.py`
- Create: `tests/research/test_activities.py`

- [ ] **Step 1: Write failing activity-level dry-run test**

Create `tests/research/test_activities.py`:

```python
from pathlib import Path

from research.db import create_experiment, get_experiment, get_intent, init_db, insert_intent
from research.models import Intent
from research.activities import run_trial_activity


def test_run_trial_activity_updates_db_and_writes_report(tmp_path: Path) -> None:
    db_path = tmp_path / "research.sqlite"
    artifact_root = tmp_path / "artifacts"
    init_db(db_path)
    intent_id = insert_intent(
        db_path,
        Intent("fake", "unit-model", "cpu", "probe", "small", {"batch_size": 1}),
    )
    experiment_id = create_experiment(
        db_path,
        intent_id=intent_id,
        adapter="tests.research.fake_adapter:FakeAdapter",
        artifact_root=str(artifact_root),
        artifact_subdir="fake/1",
    )

    result = run_trial_activity(str(db_path), experiment_id, 1)

    assert result["status"] == "succeeded"
    assert get_intent(db_path, intent_id)["status"] == "candidate"
    assert get_experiment(db_path, experiment_id)["status"] == "succeeded"
    assert (artifact_root / "fake" / str(experiment_id) / "attempt-1" / "report.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/research/test_activities.py -v
```

Expected: FAIL because `run_trial_activity` does not exist.

- [ ] **Step 3: Add missing DB read helper**

Append to `research/db.py`:

```python
def get_trial_run(db_path: Path, trial_run_id: int) -> JsonDict:
    with closing(connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM trial_runs WHERE id = ?", (trial_run_id,)).fetchone()
    if row is None:
        raise KeyError(f"trial run not found: {trial_run_id}")
    return _row(row)
```

- [ ] **Step 4: Implement trial activity**

Create `research/activities.py`:

```python
from __future__ import annotations

from pathlib import Path

from temporalio import activity

from research.adapters import load_adapter
from research.artifacts import attempt_dir
from research.db import (
    create_trial_run,
    get_experiment,
    get_intent,
    transition_experiment,
    transition_trial_run,
)
from research.models import Intent, TrialContext, TrialReport
from research.preflight import run_preflight
from research.reports import write_report
from research.runners import run_trial_command


def _intent_from_row(row: dict) -> Intent:
    return Intent(
        adapter=row["adapter"],
        model=row["model"],
        profile=row["profile"],
        phase=row["phase"],
        name=row["name"],
        config=row["config_json"],
        objective=row["objective_json"],
        source=row["source"],
    )


@activity.defn
def run_trial_activity(db_path_s: str, experiment_id: int, attempt: int = 1) -> dict:
    db_path = Path(db_path_s)
    experiment = get_experiment(db_path, experiment_id)
    intent = _intent_from_row(get_intent(db_path, int(experiment["intent_id"])))
    adapter = load_adapter(str(experiment["adapter"]))
    artifact_dir = attempt_dir(
        Path(str(experiment["artifact_root"])),
        adapter=adapter.name,
        experiment_id=experiment_id,
        attempt=attempt,
    )
    trial_run_id = create_trial_run(db_path, experiment_id=experiment_id, attempt=attempt)
    context = TrialContext(
        experiment_id=experiment_id,
        trial_run_id=trial_run_id,
        attempt=attempt,
        worktree=Path.cwd(),
        artifact_dir=artifact_dir,
        db_path=db_path,
    )
    transition_experiment(db_path, experiment_id, "running")
    preflight = run_preflight(adapter, intent, context)
    (artifact_dir / "preflight.json").write_text(
        __import__("json").dumps({"ok": preflight.ok, "checks": preflight.checks, "message": preflight.message}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not preflight.ok:
        report = TrialReport(
            status="failed",
            metrics={},
            failure={"reason": "preflight_failed", "checks": preflight.checks},
            summary=preflight.message,
        )
    else:
        transition_trial_run(db_path, trial_run_id, "running")
        command = adapter.build_trial(intent, context)
        run_result = run_trial_command(adapter, command, context)
        report = adapter.analyze_result(context)
        if run_result.returncode != 0 and report.status == "succeeded":
            report = TrialReport(
                status="failed",
                metrics=report.metrics,
                failure={"reason": "launcher_failed", "returncode": run_result.returncode},
                summary=f"trial command exited {run_result.returncode}",
            )
    paths = write_report(artifact_dir, report)
    terminal = "succeeded" if report.status == "succeeded" else "failed"
    transition_trial_run(
        db_path,
        trial_run_id,
        terminal,
        metrics=report.metrics,
        failure=report.failure,
        report_path=str(paths["markdown"]),
    )
    transition_experiment(db_path, experiment_id, terminal)
    return {"status": report.status, "metrics": report.metrics, "failure": report.failure}
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
uv run pytest tests/research/test_activities.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add research/activities.py research/db.py tests/research/test_activities.py
git commit -m "feat: run research trials through activity"
```

---

### Task 8: Temporal Workflows

**Files:**
- Create: `research/workflows.py`
- Create: `tests/research/test_workflows.py`

- [ ] **Step 1: Write workflow import and shape tests**

Create `tests/research/test_workflows.py`:

```python
def test_workflow_classes_import_and_status_shapes() -> None:
    from research.workflows import ProbeWorkflow, TrialWorkflow

    probe = ProbeWorkflow()
    trial = TrialWorkflow()

    assert probe.status()["phase"] == "created"
    assert probe.status()["completed"] == 0
    assert trial.status()["phase"] == "created"
    assert trial.status()["experiment_id"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/research/test_workflows.py -v
```

Expected: FAIL because `research.workflows` does not exist.

- [ ] **Step 3: Implement workflow skeletons**

Create `research/workflows.py`:

```python
from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from research.activities import run_trial_activity


@workflow.defn
class TrialWorkflow:
    def __init__(self) -> None:
        self.phase = "created"
        self.experiment_id: int | None = None
        self.latest_result: dict | None = None

    @workflow.run
    async def run(self, db_path: str, experiment_id: int, attempt: int = 1) -> dict:
        self.phase = "running"
        self.experiment_id = experiment_id
        self.latest_result = await workflow.execute_activity(
            run_trial_activity,
            args=[db_path, experiment_id, attempt],
            start_to_close_timeout=timedelta(hours=12),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        self.phase = "complete"
        return self.latest_result

    @workflow.query
    def status(self) -> dict:
        return {
            "phase": self.phase,
            "experiment_id": self.experiment_id,
            "latest_result": self.latest_result,
        }


@workflow.defn
class ProbeWorkflow:
    def __init__(self) -> None:
        self.phase = "created"
        self.completed = 0
        self.selected_intents: list[int] = []

    @workflow.run
    async def run(self, db_path: str, experiment_ids: list[int]) -> dict:
        self.phase = "probe"
        results = []
        for experiment_id in experiment_ids:
            result = await workflow.execute_child_workflow(
                TrialWorkflow.run,
                args=[db_path, experiment_id, 1],
                id=f"trial-{experiment_id}",
            )
            results.append(result)
            self.completed += 1
        self.phase = "complete"
        return {"results": results, "completed": self.completed}

    @workflow.query
    def status(self) -> dict:
        return {
            "phase": self.phase,
            "completed": self.completed,
            "selected_intents": list(self.selected_intents),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/research/test_workflows.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research/workflows.py tests/research/test_workflows.py
git commit -m "feat: add research temporal workflows"
```

---

### Task 9: Local Temporal Helpers and Manager CLI

**Files:**
- Create: `research/temporal.py`
- Rewrite: `research/cli.py`
- Create: `tests/research/test_cli.py`
- Create: `tests/research/test_temporal.py`

- [ ] **Step 1: Write CLI and temporal helper tests**

Create `tests/research/test_temporal.py`:

```python
from pathlib import Path

from research.temporal import build_start_dev_command, default_temporal_db_path


def test_default_temporal_db_path_uses_home(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("RESEARCH_TEMPORAL_DB", raising=False)

    assert default_temporal_db_path() == home / ".automata" / "research" / "temporal" / "temporal.db"


def test_build_start_dev_command_creates_parent(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "temporal" / "temporal.db"
    monkeypatch.setenv("RESEARCH_TEMPORAL_DB", str(db))

    command = build_start_dev_command()

    assert command == ["temporal", "server", "start-dev", "--db-filename", str(db)]
    assert db.parent.exists()
```

Create `tests/research/test_cli.py`:

```python
from pathlib import Path

from research.cli import main


def test_db_init_command_creates_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "research.sqlite"

    rc = main(["--db", str(db_path), "db", "init"])

    assert rc == 0
    assert db_path.exists()


def test_status_command_handles_empty_db(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "research.sqlite"
    main(["--db", str(db_path), "db", "init"])

    rc = main(["--db", str(db_path), "status"])

    assert rc == 0
    assert "No experiments" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/research/test_temporal.py tests/research/test_cli.py -v
```

Expected: FAIL because CLI commands and temporal helper do not exist.

- [ ] **Step 3: Implement local Temporal helpers**

Create `research/temporal.py`:

```python
from __future__ import annotations

import os
import shutil
from pathlib import Path


TEMPORAL_DB_ENV = "RESEARCH_TEMPORAL_DB"
DEFAULT_ADDRESS = "localhost:7233"
DEFAULT_TASK_QUEUE = "research-local"


def default_temporal_db_path() -> Path:
    configured = os.environ.get(TEMPORAL_DB_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".automata" / "research" / "temporal" / "temporal.db"


def ensure_temporal_cli() -> str:
    temporal = shutil.which("temporal")
    if temporal is None:
        raise RuntimeError("temporal CLI not found on PATH")
    return temporal


def build_start_dev_command() -> list[str]:
    db_path = default_temporal_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return ["temporal", "server", "start-dev", "--db-filename", str(db_path)]
```

- [ ] **Step 4: Implement minimal CLI**

Replace `research/cli.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from research import db


DEFAULT_DB = Path(".research") / "research.sqlite"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scientific experiment manager")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    subparsers = parser.add_subparsers(dest="command", required=True)

    db_parser = subparsers.add_parser("db")
    db_subparsers = db_parser.add_subparsers(dest="db_command", required=True)
    db_subparsers.add_parser("init")

    subparsers.add_parser("status")
    subparsers.add_parser("manager")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("id")

    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--adapter", required=True)
    probe_parser.add_argument("--model", required=True)
    probe_parser.add_argument("--profile", required=True)
    return parser


def command_status(db_path: Path) -> int:
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    if count == 0:
        print("No experiments")
    else:
        print(f"Experiments: {count}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = Path(args.db)
    if args.command == "db" and args.db_command == "init":
        db.init_db(db_path)
        print(db_path)
        return 0
    if args.command == "status":
        return command_status(db_path)
    if args.command in {"manager", "probe", "report"}:
        parser.error(f"{args.command} is not implemented in this migration step")
    raise AssertionError(args.command)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/research/test_temporal.py tests/research/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add research/temporal.py research/cli.py tests/research/test_temporal.py tests/research/test_cli.py
git commit -m "feat: add research cli foundation"
```

---

### Task 10: Mise Tasks

**Files:**
- Create: `mise.toml`
- Create: `tests/research/test_mise.py`

- [ ] **Step 1: Write failing mise task test**

Create `tests/research/test_mise.py`:

```python
import tomllib
from pathlib import Path


def test_mise_exposes_research_tasks() -> None:
    data = tomllib.loads(Path("mise.toml").read_text(encoding="utf-8"))

    assert data["tasks"]["research-db-init"]["run"] == "uv run research db init"
    assert data["tasks"]["research-manager"]["run"] == "uv run research manager"
    assert "uv run research probe" in data["tasks"]["research-probe"]["run"]
    assert data["tasks"]["qwen-probe-b200"]["run"].startswith("uv run research probe --adapter qwen_vl")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/research/test_mise.py -v
```

Expected: FAIL because `mise.toml` does not exist.

- [ ] **Step 3: Add thin mise tasks**

Create `mise.toml`:

```toml
[tasks.research-db-init]
run = "uv run research db init"

[tasks.research-probe]
run = "uv run research probe --adapter {{arg(name='adapter')}} --model {{arg(name='model')}} --profile {{arg(name='profile')}}"

[tasks.research-manager]
run = "uv run research manager"

[tasks.research-status]
run = "uv run research status"

[tasks.research-report]
run = "uv run research report {{arg(name='id')}}"

[tasks.qwen-probe-b200]
run = "uv run research probe --adapter qwen_vl --model Qwen/Qwen3-VL-8B-Instruct --profile b200"
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/research/test_mise.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mise.toml tests/research/test_mise.py
git commit -m "feat: add mise research tasks"
```

---

### Task 11: Qwen-VL Adapter Intent Generation

**Files:**
- Create: `qwen-vl-finetune/experiments/__init__.py`
- Create: `qwen-vl-finetune/experiments/qwen_adapter.py`
- Create: `qwen-vl-finetune/experiments/profiles/b200.env`
- Create: `qwen-vl-finetune/experiments/profiles/h100.env`
- Create: `qwen-vl-finetune/experiments/profiles/a100.env`
- Create: `qwen-vl-finetune/tests/experiments/test_qwen_adapter.py`

- [ ] **Step 1: Write failing Qwen adapter tests**

Create `qwen-vl-finetune/tests/experiments/test_qwen_adapter.py`:

```python
from pathlib import Path

from experiments.qwen_adapter import QwenVlAdapter
from research.models import ProbeRequest, TrialContext


def test_qwen_probe_intents_use_request_model_and_profile() -> None:
    adapter = QwenVlAdapter()

    intents = adapter.generate_probe_intents(
        ProbeRequest(
            model="Qwen/Qwen3-VL-8B-Instruct",
            profile="b200",
            objective={"metric": "throughput"},
        )
    )

    assert intents
    assert {intent.model for intent in intents} == {"Qwen/Qwen3-VL-8B-Instruct"}
    assert {intent.profile for intent in intents} == {"b200"}
    assert all("DATASETS" in intent.config for intent in intents)


def test_qwen_build_trial_uses_existing_launcher(tmp_path: Path) -> None:
    adapter = QwenVlAdapter()
    intent = adapter.generate_probe_intents(
        ProbeRequest("Qwen/Qwen3-VL-8B-Instruct", "b200")
    )[0]
    context = TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=Path(__file__).resolve().parents[2],
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "research.sqlite",
    )

    command = adapter.build_trial(intent, context)

    assert command.argv[-1].endswith("scripts/sft_qwen3_8b.sh")
    assert command.env["MODEL_NAME_OR_PATH"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert command.env["OUTPUT_DIR"].endswith("outputs/1/attempt-1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=..:. uv run pytest tests/experiments/test_qwen_adapter.py -v
```

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Add Qwen profile files**

Create `qwen-vl-finetune/experiments/profiles/b200.env`:

```bash
ENABLED=true
DATASETS=visualwebinstruct_train
EVAL_DATASETS=visualwebinstruct_val
MAX_STEPS=50
EVAL_STEPS=25
SAVE_STEPS=50
BATCH_SIZES=2,4,8
GRAD_ACCUM_STEPS=2,4
GRADIENT_CHECKPOINTING=true,false
MAX_PIXELS=50176,112896
MODEL_MAX_LENGTHS=8192
```

Create `qwen-vl-finetune/experiments/profiles/h100.env`:

```bash
ENABLED=false
```

Create `qwen-vl-finetune/experiments/profiles/a100.env`:

```bash
ENABLED=false
```

- [ ] **Step 4: Implement Qwen adapter**

Create `qwen-vl-finetune/experiments/__init__.py`:

```python
"""Qwen-VL experiment adapter package."""
```

Create `qwen-vl-finetune/experiments/qwen_adapter.py`:

```python
from __future__ import annotations

from itertools import product
from pathlib import Path

from research.models import (
    Intent,
    PreflightResult,
    ProbeRequest,
    ProgressUpdate,
    TrialCommand,
    TrialContext,
    TrialReport,
)


class QwenVlAdapter:
    name = "qwen_vl"

    def _repo_root(self, context: TrialContext | None = None) -> Path:
        if context is not None:
            return context.worktree
        return Path(__file__).resolve().parents[1]

    def generate_probe_intents(self, request: ProbeRequest) -> list[Intent]:
        base = {
            "MODEL_NAME_OR_PATH": request.model,
            "DATASETS": "visualwebinstruct_train",
            "EVAL_DATASETS": "visualwebinstruct_val",
            "MAX_STEPS": "50",
            "EVAL_STEPS": "25",
            "SAVE_STEPS": "50",
        }
        intents: list[Intent] = []
        for batch_size, grad_accum, gc, max_pixels in product(
            ("2", "4", "8"),
            ("2", "4"),
            ("true", "false"),
            ("50176", "112896"),
        ):
            name = f"probe_bs{batch_size}_ga{grad_accum}_gc{gc}_px{max_pixels}"
            intents.append(
                Intent(
                    adapter=self.name,
                    model=request.model,
                    profile=request.profile,
                    phase="probe",
                    name=name,
                    config={
                        **base,
                        "BATCH_SIZE": batch_size,
                        "GRAD_ACCUM_STEPS": grad_accum,
                        "GRADIENT_CHECKPOINTING": gc,
                        "MAX_PIXELS": max_pixels,
                        "MODEL_MAX_LENGTH": "8192",
                    },
                    objective=request.objective,
                    source="probe",
                )
            )
        return intents

    def preflight(self, intent: Intent, context: TrialContext) -> PreflightResult:
        launcher = self._repo_root(context) / "scripts" / "sft_qwen3_8b.sh"
        ok = launcher.exists()
        return PreflightResult(
            ok=ok,
            checks={"launcher": "ok" if ok else "missing", "launcher_path": str(launcher)},
            message="ok" if ok else f"missing launcher: {launcher}",
        )

    def build_trial(self, intent: Intent, context: TrialContext) -> TrialCommand:
        launcher = self._repo_root(context) / "scripts" / "sft_qwen3_8b.sh"
        output_dir = context.artifact_dir / "outputs" / str(context.experiment_id) / f"attempt-{context.attempt}"
        env = {key: str(value) for key, value in intent.config.items()}
        env["MODEL_NAME_OR_PATH"] = intent.model
        env["OUTPUT_DIR"] = str(output_dir)
        env["RUN_NAME"] = intent.name
        return TrialCommand(argv=["bash", str(launcher)], env=env, cwd=self._repo_root(context))

    def parse_progress(self, event_or_log_line: str) -> ProgressUpdate | None:
        line = event_or_log_line.strip()
        if line.startswith("val_loss:"):
            return ProgressUpdate(metrics={"val_loss": float(line.split(":", 1)[1].strip())}, message=line)
        if line.startswith("peak_vram_mb:"):
            return ProgressUpdate(metrics={"peak_vram_mb": float(line.split(":", 1)[1].strip())}, message=line)
        return None

    def analyze_result(self, context: TrialContext) -> TrialReport:
        log_path = context.artifact_dir / "run.log"
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        metrics: dict[str, object] = {}
        for line in text.splitlines():
            update = self.parse_progress(line)
            if update is not None:
                metrics.update(update.metrics)
        status = "succeeded" if "val_loss" in metrics or "DRY_RUN" in text else "failed"
        failure = {} if status == "succeeded" else {"reason": "missing_footer"}
        return TrialReport(status=status, metrics=metrics, failure=failure, summary=f"Qwen-VL trial {status}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=..:. uv run pytest tests/experiments/test_qwen_adapter.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add qwen-vl-finetune/experiments qwen-vl-finetune/tests/experiments/test_qwen_adapter.py
git commit -m "feat: add qwen vl research adapter"
```

---

### Task 12: Probe CLI Creates Intents and Experiments

**Files:**
- Modify: `research/cli.py`
- Create: `tests/research/test_probe_cli.py`

- [ ] **Step 1: Write failing probe CLI test**

Create `tests/research/test_probe_cli.py`:

```python
from pathlib import Path

from research.cli import main
from research.db import connect


def test_probe_cli_creates_candidate_experiments(tmp_path: Path) -> None:
    db_path = tmp_path / "research.sqlite"
    artifact_root = tmp_path / "artifacts"

    rc = main(
        [
            "--db",
            str(db_path),
            "probe",
            "--adapter",
            "tests.research.fake_adapter:FakeAdapter",
            "--model",
            "unit-model",
            "--profile",
            "cpu",
            "--artifact-root",
            str(artifact_root),
        ]
    )

    assert rc == 0
    with connect(db_path) as conn:
        intent_count = conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]
        experiment_count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    assert intent_count == 1
    assert experiment_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/research/test_probe_cli.py -v
```

Expected: FAIL because `probe` currently errors.

- [ ] **Step 3: Implement probe command as DB setup**

Modify `research/cli.py` imports:

```python
from research.adapters import load_adapter
from research.artifacts import default_artifact_root
from research.models import ProbeRequest
```

Add to `probe_parser`:

```python
probe_parser.add_argument("--artifact-root", default="")
```

In `main`, before the manager/report error branch:

```python
    if args.command == "probe":
        db.init_db(db_path)
        adapter = load_adapter(args.adapter)
        request = ProbeRequest(model=args.model, profile=args.profile)
        artifact_root = Path(args.artifact_root) if args.artifact_root else default_artifact_root()
        intents = adapter.generate_probe_intents(request)
        experiment_ids: list[int] = []
        for intent in intents:
            intent_id = db.insert_intent(db_path, intent)
            experiment_id = db.create_experiment(
                db_path,
                intent_id=intent_id,
                adapter=args.adapter,
                artifact_root=str(artifact_root),
                artifact_subdir=f"{adapter.name}/{intent_id}",
            )
            experiment_ids.append(experiment_id)
        print(f"Created {len(experiment_ids)} experiments")
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/research/test_probe_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research/cli.py tests/research/test_probe_cli.py
git commit -m "feat: create experiments from probe intents"
```

---

### Task 13: Manager Registers Temporal Worker

**Files:**
- Modify: `research/cli.py`
- Create: `tests/research/test_manager_cli.py`

- [ ] **Step 1: Write failing manager configuration test**

Create `tests/research/test_manager_cli.py`:

```python
from research.cli import build_worker_config


def test_build_worker_config_uses_research_task_queue() -> None:
    config = build_worker_config(address="localhost:7233", task_queue="research-local")

    assert config["address"] == "localhost:7233"
    assert config["task_queue"] == "research-local"
    assert "ProbeWorkflow" in config["workflows"]
    assert "TrialWorkflow" in config["workflows"]
    assert "run_trial_activity" in config["activities"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/research/test_manager_cli.py -v
```

Expected: FAIL because `build_worker_config` does not exist.

- [ ] **Step 3: Add manager parser args and config helper**

Modify `research/cli.py` imports:

```python
import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from research.activities import run_trial_activity
from research.temporal import DEFAULT_ADDRESS, DEFAULT_TASK_QUEUE
from research.workflows import ProbeWorkflow, TrialWorkflow
```

Add manager parser args:

```python
    manager_parser = subparsers.add_parser("manager")
    manager_parser.add_argument("--address", default=DEFAULT_ADDRESS)
    manager_parser.add_argument("--task-queue", default=DEFAULT_TASK_QUEUE)
```

Replace earlier `subparsers.add_parser("manager")`.

Add helper functions:

```python
def build_worker_config(address: str, task_queue: str) -> dict[str, object]:
    return {
        "address": address,
        "task_queue": task_queue,
        "workflows": [ProbeWorkflow.__name__, TrialWorkflow.__name__],
        "activities": [run_trial_activity.__name__],
    }


async def run_manager(address: str, task_queue: str) -> None:
    client = await Client.connect(address)
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[ProbeWorkflow, TrialWorkflow],
        activities=[run_trial_activity],
    )
    await worker.run()
```

In `main`, add:

```python
    if args.command == "manager":
        asyncio.run(run_manager(args.address, args.task_queue))
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/research/test_manager_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research/cli.py tests/research/test_manager_cli.py
git commit -m "feat: add research temporal manager"
```

---

### Task 14: Remove Old Qwen Research Orchestration Package

**Files:**
- Delete: `qwen-vl-finetune/research/`
- Delete: `qwen-vl-finetune/tests/research/`
- Create: `tests/research/test_no_qwen_research_shadow.py`

- [ ] **Step 1: Write package-shadow cleanup test**

Create `tests/research/test_no_qwen_research_shadow.py`:

```python
from pathlib import Path


def test_qwen_finetune_no_longer_shadows_top_level_research_package() -> None:
    assert not Path("qwen-vl-finetune/research/__init__.py").exists()
    assert not Path("qwen-vl-finetune/research/runner.py").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/research/test_no_qwen_research_shadow.py -v
```

Expected: FAIL because `qwen-vl-finetune/research/` still exists.

- [ ] **Step 3: Move any still-needed Qwen-specific fixtures before deletion**

Before deleting, confirm Qwen adapter tests from Task 11 no longer import from
`qwen-vl-finetune/research`. Run:

```bash
rg -n "from research|import research|research\\." qwen-vl-finetune/tests qwen-vl-finetune/experiments
```

Expected: no imports of the old `qwen-vl-finetune/research` package. Imports of
the new top-level `research.models` are allowed only when tests run with
`PYTHONPATH=..:.` from `qwen-vl-finetune`.

- [ ] **Step 4: Delete old Qwen research package and tests**

Run:

```bash
git rm -r qwen-vl-finetune/research qwen-vl-finetune/tests/research
```

Expected: staged deletion of old orchestration code and old tests that targeted
the removed package. Do not delete `qwen-vl-finetune/experiments/`.

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
uv run pytest tests/research/test_no_qwen_research_shadow.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/research/test_no_qwen_research_shadow.py
git commit -m "refactor: remove old qwen research orchestration"
```

---

### Task 15: Remove Autonomy Package

**Files:**
- Delete: `autonomy/`
- Modify: `pyproject.toml`
- Create: `tests/research/test_no_autonomy.py`

- [ ] **Step 1: Write cleanup test**

Create `tests/research/test_no_autonomy.py`:

```python
from pathlib import Path


def test_autonomy_methodology_removed() -> None:
    assert not Path("autonomy/autonomy/workflows.py").exists()
    assert not Path("autonomy/autonomy/org").exists()
```

- [ ] **Step 2: Delete autonomy files**

Run:

```bash
git rm -r autonomy
```

Expected: staged deletion of tracked `autonomy/` files. If untracked cache or venv files remain, remove only ignored/generated files with:

```bash
rm -rf autonomy/.pytest_cache autonomy/.venv autonomy/autonomy/__pycache__ autonomy/tests/__pycache__
```

- [ ] **Step 3: Run cleanup test**

Run:

```bash
uv run pytest tests/research/test_no_autonomy.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/research/test_no_autonomy.py
git commit -m "refactor: remove autonomy methodology package"
```

---

### Task 16: Final Verification and Documentation Cleanup

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Create or modify: `docs/superpowers/plans/2026-05-14-scientific-experiment-methodology.md`

- [ ] **Step 1: Ignore local research runtime state**

Add to `.gitignore`:

```gitignore
.research/
.artifacts/
```

- [ ] **Step 2: Add top-level README pointer**

Add a short section to `README.md` near existing development or finetuning content:

```markdown
## Scientific Experiment Methodology

This repository uses the top-level `research/` package for scientific
experiment orchestration. Use `mise` tasks for routine operation:

```bash
mise run research-db-init
mise run qwen-probe-b200
mise run research-status
```

Qwen-VL-specific experiment behavior lives under `qwen-vl-finetune/experiments/`.
```

- [ ] **Step 3: Run focused generic tests**

Run:

```bash
uv run pytest tests/research -v
```

Expected: PASS.

- [ ] **Step 4: Run focused Qwen adapter tests**

Run:

```bash
cd qwen-vl-finetune
PYTHONPATH=..:. uv run pytest tests/experiments -v
```

Expected: PASS.

- [ ] **Step 5: Run import and CLI smoke commands**

Run:

```bash
uv run research db init
uv run research status
uv run research probe --adapter tests.research.fake_adapter:FakeAdapter --model unit-model --profile cpu
```

Expected: all exit `0`; status initially says `No experiments`, then probe says `Created 1 experiments`.

- [ ] **Step 6: Check working tree**

Run:

```bash
git status --short
```

Expected: only intended files modified. Existing user/in-progress files unrelated to this migration must not be reverted.

- [ ] **Step 7: Commit docs and ignore updates**

```bash
git add .gitignore README.md docs/superpowers/plans/2026-05-14-scientific-experiment-methodology.md
git commit -m "docs: document research methodology operation"
```

---

## Self-Review Notes

- Spec coverage: the plan covers the generic package, SQLite DB, Temporal workflows, adapter boundary, artifact/preflight/report helpers, Qwen adapter, mise tasks, old runner compatibility, autonomy removal, and verification.
- Simplicity check: the first implementation intentionally makes `probe` create DB experiments before fully starting Temporal probe orchestration. The Temporal `ProbeWorkflow` exists, but the CLI can wire actual workflow start in a later refinement after the generic activity path is verified. This keeps the migration testable task by task.
- State-source check: SQLite is introduced as source of truth; TSV is not extended.
- Hard-coding check: Qwen model/dataset/launcher defaults appear only in Qwen adapter and mise convenience task, not generic `research/` infrastructure.
