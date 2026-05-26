# COCO Deep Research Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Temporal deep-research workflow where planning, query generation, web search, and report synthesis all run through the local COCO CLI.

**Architecture:** Add focused generic modules under `research/`: models and parsers, COCO subprocess execution, COCO-backed activities, and a deterministic Temporal workflow. Register the workflow and activities through `research.temporal` and expose a small `research deep-research` CLI without changing Qwen experiment behavior.

**Tech Stack:** Python 3.10+, Temporal Python SDK, `uv`, `pytest`, `ruff`, local `coco` CLI invoked through `subprocess`.

---

## File structure

- Create `research/deep_research_models.py`
  - Dataclasses and parsing helpers for request, plans, search results, and reports.
- Create `research/coco_cli.py`
  - COCO command discovery, command construction, subprocess runner, and error type.
- Create `research/deep_research_activities.py`
  - Four Temporal activities backed by `research.coco_cli`.
- Create `research/deep_research_workflows.py`
  - `DeepResearchWorkflow`, workflow status query, partial search failure handling.
- Modify `research/temporal.py`
  - Register deep-research workflow and activities on demand. Add `start_deep_research_workflow`.
- Modify `research/cli.py`
  - Add `research deep-research worker` and `research deep-research start`.
- Create `tests/research/test_deep_research_models.py`
  - Model and parser tests.
- Create `tests/research/test_coco_cli.py`
  - COCO command and subprocess boundary tests.
- Create `tests/research/test_deep_research_activities.py`
  - Activity tests with fake COCO runners.
- Create `tests/research/test_deep_research_workflow.py`
  - Temporal workflow tests with fake activities.
- Create `tests/research/test_deep_research_cli.py`
  - CLI and registration tests.

Keep all imports at module top level. Do not add conditional imports or dependency-skipping paths. Use `uv` for dependency execution.

---

### Task 1: Add deep-research models and parsers

**Files:**
- Create: `research/deep_research_models.py`
- Test: `tests/research/test_deep_research_models.py`

- [ ] **Step 1: Write failing parser and dataclass tests**

Create `tests/research/test_deep_research_models.py`:

```python
"""Tests for deep-research model parsing."""

from __future__ import annotations

import pytest

import research.deep_research_models as models


def test_deep_research_request_defaults() -> None:
    request = models.DeepResearchRequest(topic="Temporal AI workflows")

    assert request.topic == "Temporal AI workflows"
    assert request.max_subtopics == 5
    assert request.queries_per_subtopic == 3
    assert request.max_search_results == 5


def test_parse_subtopic_plan_accepts_compact_json() -> None:
    plan = models.parse_subtopic_plan(
        '{"subtopics": ["durable execution", "activity retries"],'
        ' "rationale": "covers Temporal basics"}'
    )

    assert plan.subtopics == ["durable execution", "activity retries"]
    assert plan.rationale == "covers Temporal basics"


def test_parse_search_query_plan_rejects_empty_queries() -> None:
    with pytest.raises(models.DeepResearchParseError, match="queries"):
        models.parse_search_query_plan('{"queries": []}')


def test_parse_search_result_requires_summary() -> None:
    with pytest.raises(models.DeepResearchParseError, match="summary"):
        models.parse_search_result('{"query": "Temporal", "sources": []}')


def test_report_as_temporal_payload_uses_plain_dicts() -> None:
    report = models.DeepResearchReport(
        topic="Temporal",
        markdown="# Temporal",
        successful_queries=["Temporal activity retries"],
        failed_queries=[{"query": "bad", "error": "timeout"}],
    )

    assert report.as_dict() == {
        "topic": "Temporal",
        "markdown": "# Temporal",
        "successful_queries": ["Temporal activity retries"],
        "failed_queries": [{"query": "bad", "error": "timeout"}],
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/research/test_deep_research_models.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'research.deep_research_models'`.

- [ ] **Step 3: Implement models and parsing helpers**

Create `research/deep_research_models.py`:

```python
"""Data contracts for COCO-backed deep research."""

from __future__ import annotations

import dataclasses
import json
from typing import Any


JsonDict = dict[str, Any]


class DeepResearchParseError(ValueError):
    """Raised when a COCO response does not match the expected shape."""


@dataclasses.dataclass(frozen=True)
class DeepResearchRequest:
    """Input for one deep-research workflow run."""

    topic: str
    max_subtopics: int = 5
    queries_per_subtopic: int = 3
    max_search_results: int = 5

    def as_dict(self) -> JsonDict:
        """Return a Temporal payload-safe representation."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SubtopicPlan:
    """COCO-generated subtopic plan."""

    subtopics: list[str]
    rationale: str = ""

    def as_dict(self) -> JsonDict:
        """Return a Temporal payload-safe representation."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SearchQueryPlan:
    """COCO-generated search query plan."""

    queries: list[str]

    def as_dict(self) -> JsonDict:
        """Return a Temporal payload-safe representation."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SearchResult:
    """Summarized result for one COCO search query."""

    query: str
    summary: str
    sources: list[str]
    raw_output_path: str = ""

    def as_dict(self) -> JsonDict:
        """Return a Temporal payload-safe representation."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DeepResearchReport:
    """Final Markdown report and workflow result metadata."""

    topic: str
    markdown: str
    successful_queries: list[str]
    failed_queries: list[dict[str, str]]

    def as_dict(self) -> JsonDict:
        """Return a Temporal payload-safe representation."""
        return dataclasses.asdict(self)


def parse_subtopic_plan(raw_output: str) -> SubtopicPlan:
    """Parse COCO JSON into a subtopic plan."""
    payload = _json_object(raw_output)
    subtopics = _string_list(payload, "subtopics")
    rationale = _optional_string(payload, "rationale")
    return SubtopicPlan(subtopics=subtopics, rationale=rationale)


def parse_search_query_plan(raw_output: str) -> SearchQueryPlan:
    """Parse COCO JSON into search queries."""
    payload = _json_object(raw_output)
    return SearchQueryPlan(queries=_string_list(payload, "queries"))


def parse_search_result(raw_output: str) -> SearchResult:
    """Parse COCO JSON into one search result."""
    payload = _json_object(raw_output)
    query = _required_string(payload, "query")
    summary = _required_string(payload, "summary")
    sources = _string_list(payload, "sources", allow_empty=True)
    raw_output_path = _optional_string(payload, "raw_output_path")
    return SearchResult(
        query=query,
        summary=summary,
        sources=sources,
        raw_output_path=raw_output_path,
    )


def _json_object(raw_output: str) -> JsonDict:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise DeepResearchParseError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise DeepResearchParseError("expected JSON object")
    return payload


def _required_string(payload: JsonDict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DeepResearchParseError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(payload: JsonDict, key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DeepResearchParseError(f"{key} must be a string")
    return value.strip()


def _string_list(
    payload: JsonDict,
    key: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise DeepResearchParseError(f"{key} must be a list of strings")
    strings = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(strings) != len(value):
        raise DeepResearchParseError(f"{key} must contain only non-empty strings")
    if not strings and not allow_empty:
        raise DeepResearchParseError(f"{key} must not be empty")
    return strings
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m pytest tests/research/test_deep_research_models.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add research/deep_research_models.py tests/research/test_deep_research_models.py
git commit -m "feat: add deep research models"
```

---

### Task 2: Add the COCO CLI subprocess boundary

**Files:**
- Create: `research/coco_cli.py`
- Test: `tests/research/test_coco_cli.py`

- [ ] **Step 1: Write failing COCO CLI tests**

Create `tests/research/test_coco_cli.py`:

```python
"""Tests for COCO CLI subprocess execution."""

from __future__ import annotations

import pathlib
import subprocess

import pytest

import research.coco_cli as coco_cli


def test_build_coco_command_uses_prompt_mode_and_yes_flag(
    tmp_path: pathlib.Path,
) -> None:
    command = coco_cli.build_coco_command(
        coco_path="/usr/bin/coco",
        prompt="Return JSON",
        worktree="deep-research-search",
        workspace=tmp_path,
    )

    assert command == [
        "/usr/bin/coco",
        "-p",
        "-y",
        "--worktree",
        "deep-research-search",
        "--workspace",
        str(tmp_path),
        "Return JSON",
    ]


def test_build_coco_command_omits_workspace_when_absent() -> None:
    command = coco_cli.build_coco_command(
        coco_path="/usr/bin/coco",
        prompt="Return JSON",
        worktree="deep-research-plan",
        workspace=None,
    )

    assert command == [
        "/usr/bin/coco",
        "-p",
        "-y",
        "--worktree",
        "deep-research-plan",
        "Return JSON",
    ]


def test_resolve_coco_path_uses_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_COCO_BIN", "/opt/bin/coco")

    assert coco_cli.resolve_coco_path() == "/opt/bin/coco"


def test_run_coco_raises_on_missing_workspace(tmp_path: pathlib.Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(coco_cli.CocoExecutionError, match="workspace"):
        coco_cli.run_coco(
            prompt="Return JSON",
            phase="search",
            worktree="deep-research-search",
            workspace=missing,
            runner=subprocess.run,
            coco_path="/usr/bin/coco",
        )


def test_run_coco_raises_with_stderr_tail_on_failure(
    tmp_path: pathlib.Path,
) -> None:
    def failing_runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            2,
            stdout="some stdout",
            stderr="first line\nlast error line",
        )

    with pytest.raises(coco_cli.CocoExecutionError) as exc_info:
        coco_cli.run_coco(
            prompt="Return JSON",
            phase="search",
            worktree="deep-research-search",
            workspace=tmp_path,
            runner=failing_runner,
            coco_path="/usr/bin/coco",
        )

    assert exc_info.value.phase == "search"
    assert exc_info.value.returncode == 2
    assert "last error line" in str(exc_info.value)


def test_run_coco_returns_stdout_on_success(tmp_path: pathlib.Path) -> None:
    def passing_runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout='{"ok": true}', stderr="")

    result = coco_cli.run_coco(
        prompt="Return JSON",
        phase="subtopics",
        worktree="deep-research-plan",
        workspace=tmp_path,
        runner=passing_runner,
        coco_path="/usr/bin/coco",
    )

    assert result.stdout == '{"ok": true}'
    assert result.returncode == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/research/test_coco_cli.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'research.coco_cli'`.

- [ ] **Step 3: Implement the COCO CLI wrapper**

Create `research/coco_cli.py`:

```python
"""Subprocess boundary for local COCO CLI calls."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import subprocess
from collections.abc import Callable


COCO_BIN_ENV = "RESEARCH_COCO_BIN"
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclasses.dataclass(frozen=True)
class CocoResult:
    """Completed COCO subprocess output."""

    argv: list[str]
    stdout: str
    stderr: str
    returncode: int


class CocoExecutionError(RuntimeError):
    """Raised when COCO cannot be launched or exits unsuccessfully."""

    def __init__(
        self,
        *,
        phase: str,
        message: str,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def resolve_coco_path() -> str:
    """Return configured COCO binary path or discover it on PATH."""
    configured = os.environ.get(COCO_BIN_ENV)
    if configured:
        return configured
    resolved = shutil.which("coco")
    if resolved is None:
        raise CocoExecutionError(
            phase="discovery",
            message="coco CLI not found on PATH; set RESEARCH_COCO_BIN",
        )
    return resolved


def build_coco_command(
    *,
    coco_path: str,
    prompt: str,
    worktree: str,
    workspace: pathlib.Path | None,
) -> list[str]:
    """Build a non-interactive COCO command."""
    command = [coco_path, "-p", "-y", "--worktree", worktree]
    if workspace is not None:
        command.extend(["--workspace", str(workspace)])
    command.append(prompt)
    return command


def run_coco(
    *,
    prompt: str,
    phase: str,
    worktree: str,
    workspace: pathlib.Path | None = None,
    runner: Runner = subprocess.run,
    coco_path: str | None = None,
) -> CocoResult:
    """Run COCO and return captured process output."""
    if workspace is not None and not workspace.exists():
        raise CocoExecutionError(
            phase=phase,
            message=f"workspace does not exist: {workspace}",
        )
    resolved_coco = coco_path or resolve_coco_path()
    argv = build_coco_command(
        coco_path=resolved_coco,
        prompt=prompt,
        worktree=worktree,
        workspace=workspace,
    )
    completed = runner(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise CocoExecutionError(
            phase=phase,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            message=(
                f"COCO phase {phase} exited {completed.returncode}: "
                f"{_tail(completed.stderr) or _tail(completed.stdout)}"
            ),
        )
    return CocoResult(
        argv=list(argv),
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def _tail(text: str, limit: int = 500) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[-limit:]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run python -m pytest tests/research/test_coco_cli.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add research/coco_cli.py tests/research/test_coco_cli.py
git commit -m "feat: add coco cli runner"
```

---

### Task 3: Add COCO-backed deep-research activities

**Files:**
- Create: `research/deep_research_activities.py`
- Test: `tests/research/test_deep_research_activities.py`

- [ ] **Step 1: Write failing activity tests**

Create `tests/research/test_deep_research_activities.py`:

```python
"""Tests for COCO-backed deep-research activities."""

from __future__ import annotations

import pathlib

import pytest
import temporalio.exceptions

import research.deep_research_activities as activities
import research.deep_research_models as models


class FakeCoco:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return type("Result", (), {"stdout": self.stdout})()


def test_generate_subtopics_with_coco_parses_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    fake = FakeCoco('{"subtopics": ["a", "b"], "rationale": "r"}')
    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake)

    result = activities.generate_subtopics_with_coco(
        {"topic": "Temporal", "workspace": str(tmp_path)}
    )

    assert result == {"subtopics": ["a", "b"], "rationale": "r"}
    assert fake.calls[0]["phase"] == "subtopics"


def test_generate_search_queries_with_coco_uses_subtopics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeCoco('{"queries": ["Temporal retries", "Temporal activities"]}')
    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake)

    result = activities.generate_search_queries_with_coco(
        {"subtopics": ["retries"], "queries_per_subtopic": 2}
    )

    assert result == {"queries": ["Temporal retries", "Temporal activities"]}
    assert "retries" in str(fake.calls[0]["prompt"])


def test_search_web_with_coco_parses_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeCoco(
        '{"query": "Temporal", "summary": "Durable execution",'
        ' "sources": ["https://temporal.io"]}'
    )
    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake)

    result = activities.search_web_with_coco({"query": "Temporal"})

    assert result["query"] == "Temporal"
    assert result["summary"] == "Durable execution"
    assert result["sources"] == ["https://temporal.io"]


def test_synthesize_report_with_coco_wraps_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeCoco("# Report\n\nFindings.")
    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake)

    result = activities.synthesize_report_with_coco(
        {
            "topic": "Temporal",
            "research": [{"query": "Temporal", "summary": "Durable"}],
            "failed_queries": [{"query": "bad", "error": "timeout"}],
        }
    )

    assert result["topic"] == "Temporal"
    assert result["markdown"].startswith("# Report")
    assert result["successful_queries"] == ["Temporal"]
    assert result["failed_queries"] == [{"query": "bad", "error": "timeout"}]


def test_parse_failure_raises_application_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeCoco('{"queries": []}')
    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake)

    with pytest.raises(temporalio.exceptions.ApplicationError) as exc_info:
        activities.generate_search_queries_with_coco({"subtopics": ["x"]})

    assert exc_info.value.type == "deep_research_parse_failed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/research/test_deep_research_activities.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'research.deep_research_activities'`.

- [ ] **Step 3: Implement the COCO activities**

Create `research/deep_research_activities.py`:

```python
"""Temporal activities for COCO-backed deep research."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import temporalio.activity
import temporalio.exceptions

import research.coco_cli
import research.deep_research_models as models


JsonDict = dict[str, Any]


@temporalio.activity.defn
def generate_subtopics_with_coco(request: JsonDict) -> JsonDict:
    """Generate a subtopic plan through COCO."""
    topic = _required_string(request, "topic")
    max_subtopics = int(request.get("max_subtopics", 5))
    prompt = (
        "Return only compact JSON with keys subtopics and rationale. "
        f"Create at most {max_subtopics} research subtopics for: {topic}"
    )
    output = _run_phase(
        phase="subtopics",
        prompt=prompt,
        worktree="deep-research-subtopics",
        workspace=_workspace(request),
    )
    try:
        return output.as_dict()
    except AttributeError as exc:
        raise _application_error("invalid subtopic parser result", exc)


@temporalio.activity.defn
def generate_search_queries_with_coco(plan: JsonDict) -> JsonDict:
    """Generate search queries through COCO."""
    subtopics = _required_string_list(plan, "subtopics")
    queries_per_subtopic = int(plan.get("queries_per_subtopic", 3))
    prompt = (
        "Return only compact JSON with key queries. "
        f"Create {queries_per_subtopic} web search queries per subtopic for: "
        f"{json.dumps(subtopics)}"
    )
    raw = _raw_phase(
        phase="queries",
        prompt=prompt,
        worktree="deep-research-queries",
        workspace=_workspace(plan),
    )
    try:
        return models.parse_search_query_plan(raw).as_dict()
    except models.DeepResearchParseError as exc:
        raise _application_error("search query JSON was invalid", exc)


@temporalio.activity.defn
def search_web_with_coco(query_payload: JsonDict) -> JsonDict:
    """Delegate one web search query to COCO."""
    query = _required_string(query_payload, "query")
    max_search_results = int(query_payload.get("max_search_results", 5))
    prompt = (
        "Search the web and return only compact JSON with keys query, summary, "
        f"and sources. Use at most {max_search_results} sources. Query: {query}"
    )
    raw = _raw_phase(
        phase="search",
        prompt=prompt,
        worktree="deep-research-search",
        workspace=_workspace(query_payload),
    )
    try:
        return models.parse_search_result(raw).as_dict()
    except models.DeepResearchParseError as exc:
        raise _application_error("search result JSON was invalid", exc)


@temporalio.activity.defn
def synthesize_report_with_coco(payload: JsonDict) -> JsonDict:
    """Synthesize successful COCO search results into a Markdown report."""
    topic = _required_string(payload, "topic")
    research = payload.get("research")
    failed_queries = payload.get("failed_queries", [])
    if not isinstance(research, list) or not research:
        raise temporalio.exceptions.ApplicationError(
            "research must contain at least one successful result",
            type="deep_research_empty_research",
            non_retryable=True,
        )
    prompt = (
        "Write a concise Markdown research report. Base it only on this JSON "
        f"payload: {json.dumps({'topic': topic, 'research': research})}"
    )
    markdown = _raw_phase(
        phase="synthesis",
        prompt=prompt,
        worktree="deep-research-synthesis",
        workspace=_workspace(payload),
    ).strip()
    if not markdown:
        raise temporalio.exceptions.ApplicationError(
            "COCO synthesis returned empty Markdown",
            type="deep_research_empty_report",
        )
    successful_queries = [
        str(item["query"])
        for item in research
        if isinstance(item, dict) and isinstance(item.get("query"), str)
    ]
    report = models.DeepResearchReport(
        topic=topic,
        markdown=markdown,
        successful_queries=successful_queries,
        failed_queries=[
            {"query": str(item.get("query", "")), "error": str(item.get("error", ""))}
            for item in failed_queries
            if isinstance(item, dict)
        ],
    )
    return report.as_dict()


def _run_phase(
    *,
    phase: str,
    prompt: str,
    worktree: str,
    workspace: pathlib.Path | None,
) -> models.SubtopicPlan:
    raw = _raw_phase(
        phase=phase,
        prompt=prompt,
        worktree=worktree,
        workspace=workspace,
    )
    try:
        return models.parse_subtopic_plan(raw)
    except models.DeepResearchParseError as exc:
        raise _application_error("subtopic JSON was invalid", exc)


def _raw_phase(
    *,
    phase: str,
    prompt: str,
    worktree: str,
    workspace: pathlib.Path | None,
) -> str:
    try:
        return research.coco_cli.run_coco(
            prompt=prompt,
            phase=phase,
            worktree=worktree,
            workspace=workspace,
        ).stdout
    except research.coco_cli.CocoExecutionError as exc:
        raise temporalio.exceptions.ApplicationError(
            str(exc),
            dataclasses.asdict(
                models.DeepResearchReport(
                    topic="",
                    markdown="",
                    successful_queries=[],
                    failed_queries=[
                        {
                            "query": phase,
                            "error": str(exc),
                        }
                    ],
                )
            ),
            type="deep_research_coco_failed",
        )


def _application_error(
    message: str,
    exc: Exception,
) -> temporalio.exceptions.ApplicationError:
    return temporalio.exceptions.ApplicationError(
        f"{message}: {exc}",
        type="deep_research_parse_failed",
    )


def _required_string(payload: JsonDict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise temporalio.exceptions.ApplicationError(
            f"{key} must be a non-empty string",
            type="deep_research_invalid_payload",
            non_retryable=True,
        )
    return value.strip()


def _required_string_list(payload: JsonDict, key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise temporalio.exceptions.ApplicationError(
            f"{key} must be a list",
            type="deep_research_invalid_payload",
            non_retryable=True,
        )
    strings = [item for item in value if isinstance(item, str) and item.strip()]
    if len(strings) != len(value) or not strings:
        raise temporalio.exceptions.ApplicationError(
            f"{key} must contain non-empty strings",
            type="deep_research_invalid_payload",
            non_retryable=True,
        )
    return strings


def _workspace(payload: JsonDict) -> pathlib.Path | None:
    workspace = payload.get("workspace")
    if workspace in {None, ""}:
        return None
    if not isinstance(workspace, str):
        raise temporalio.exceptions.ApplicationError(
            "workspace must be a string",
            type="deep_research_invalid_payload",
            non_retryable=True,
        )
    return pathlib.Path(workspace)
```

- [ ] **Step 4: Run tests and fix any exact mismatch**

Run:

```bash
uv run python -m pytest tests/research/test_deep_research_activities.py -q
```

Expected: all tests pass. If the activity wrapper raises a different `ApplicationError.type`, make the code and tests use the type names from this plan.

- [ ] **Step 5: Commit**

```bash
git add research/deep_research_activities.py tests/research/test_deep_research_activities.py
git commit -m "feat: add coco deep research activities"
```

---

### Task 4: Add the Temporal deep-research workflow

**Files:**
- Create: `research/deep_research_workflows.py`
- Test: `tests/research/test_deep_research_workflow.py`

- [ ] **Step 1: Write failing workflow tests with fake activities**

Create `tests/research/test_deep_research_workflow.py`:

```python
"""Tests for the COCO deep-research Temporal workflow."""

from __future__ import annotations

import asyncio
import uuid

import pytest
import temporalio.activity
import temporalio.exceptions
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

import research.deep_research_workflows as workflows


@temporalio.activity.defn(name="generate_subtopics_with_coco")
async def fake_subtopics(request: dict) -> dict:
    return {"subtopics": ["durability", "retries"], "rationale": "coverage"}


@temporalio.activity.defn(name="generate_search_queries_with_coco")
async def fake_queries(plan: dict) -> dict:
    return {"queries": ["Temporal durability", "Temporal retries"]}


@temporalio.activity.defn(name="search_web_with_coco")
async def fake_search(payload: dict) -> dict:
    query = payload["query"]
    return {"query": query, "summary": f"summary for {query}", "sources": []}


@temporalio.activity.defn(name="synthesize_report_with_coco")
async def fake_synthesis(payload: dict) -> dict:
    return {
        "topic": payload["topic"],
        "markdown": "# Report",
        "successful_queries": [item["query"] for item in payload["research"]],
        "failed_queries": payload["failed_queries"],
    }


@temporalio.activity.defn(name="search_web_with_coco")
async def fake_partial_search(payload: dict) -> dict:
    if payload["query"] == "Temporal retries":
        raise RuntimeError("search failed")
    return {"query": payload["query"], "summary": "ok", "sources": []}


@temporalio.activity.defn(name="search_web_with_coco")
async def fake_failed_search(payload: dict) -> dict:
    raise RuntimeError("search failed")


def test_deep_research_workflow_happy_path() -> None:
    result = asyncio.run(_run_workflow([fake_search]))

    assert result["markdown"] == "# Report"
    assert result["successful_queries"] == ["Temporal durability", "Temporal retries"]
    assert result["failed_queries"] == []


def test_deep_research_workflow_tolerates_partial_search_failure() -> None:
    result = asyncio.run(_run_workflow([fake_partial_search]))

    assert result["successful_queries"] == ["Temporal durability"]
    assert result["failed_queries"][0]["query"] == "Temporal retries"


def test_deep_research_workflow_fails_when_all_searches_fail() -> None:
    with pytest.raises(temporalio.exceptions.WorkflowFailureError):
        asyncio.run(_run_workflow([fake_failed_search]))


async def _run_workflow(search_activities: list[object]) -> dict:
    task_queue = f"deep-research-{uuid.uuid4()}"
    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[workflows.DeepResearchWorkflow],
            activities=[
                fake_subtopics,
                fake_queries,
                *search_activities,
                fake_synthesis,
            ],
        ):
            return await env.client.execute_workflow(
                workflows.DeepResearchWorkflow.run,
                args=[
                    {
                        "topic": "Temporal AI",
                        "max_subtopics": 2,
                        "queries_per_subtopic": 1,
                        "max_search_results": 3,
                    }
                ],
                id=f"deep-research-{uuid.uuid4()}",
                task_queue=task_queue,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/research/test_deep_research_workflow.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'research.deep_research_workflows'`.

- [ ] **Step 3: Implement `DeepResearchWorkflow`**

Create `research/deep_research_workflows.py`:

```python
"""Temporal workflow for COCO-backed deep research."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import temporalio.exceptions
from temporalio import workflow


PLANNING_TIMEOUT = timedelta(seconds=60)
SEARCH_START_TO_CLOSE_TIMEOUT = timedelta(seconds=300)
SEARCH_SCHEDULE_TO_CLOSE_TIMEOUT = timedelta(seconds=900)
SYNTHESIS_TIMEOUT = timedelta(seconds=300)


@workflow.defn
class DeepResearchWorkflow:
    """Run a COCO-backed deep-research pipeline."""

    def __init__(self) -> None:
        self.phase = "created"
        self.topic = ""
        self.completed_searches = 0
        self.failed_searches = 0

    @workflow.run
    async def run(self, request: dict) -> dict:
        """Execute deep research for one topic."""
        self.phase = "planning"
        self.topic = str(request["topic"])
        subtopics = await workflow.execute_activity(
            "generate_subtopics_with_coco",
            request,
            task_queue=workflow.info().task_queue,
            start_to_close_timeout=PLANNING_TIMEOUT,
        )

        self.phase = "querying"
        query_payload = {
            **subtopics,
            "queries_per_subtopic": request.get("queries_per_subtopic", 3),
            "workspace": request.get("workspace", ""),
        }
        queries = await workflow.execute_activity(
            "generate_search_queries_with_coco",
            query_payload,
            task_queue=workflow.info().task_queue,
            start_to_close_timeout=PLANNING_TIMEOUT,
        )

        self.phase = "searching"
        search_tasks = [
            workflow.execute_activity(
                "search_web_with_coco",
                {
                    "query": query,
                    "max_search_results": request.get("max_search_results", 5),
                    "workspace": request.get("workspace", ""),
                },
                task_queue=workflow.info().task_queue,
                start_to_close_timeout=SEARCH_START_TO_CLOSE_TIMEOUT,
                schedule_to_close_timeout=SEARCH_SCHEDULE_TO_CLOSE_TIMEOUT,
            )
            for query in queries["queries"]
        ]
        raw_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        successful_results = [
            result for result in raw_results if not isinstance(result, Exception)
        ]
        failed_queries = _failed_query_payloads(queries["queries"], raw_results)
        self.completed_searches = len(successful_results)
        self.failed_searches = len(failed_queries)

        if not successful_results:
            self.phase = "failed"
            raise temporalio.exceptions.ApplicationError(
                "all COCO search activities failed",
                failed_queries,
                type="deep_research_all_searches_failed",
                non_retryable=True,
            )

        self.phase = "synthesizing"
        report = await workflow.execute_activity(
            "synthesize_report_with_coco",
            {
                "topic": self.topic,
                "research": successful_results,
                "failed_queries": failed_queries,
                "workspace": request.get("workspace", ""),
            },
            task_queue=workflow.info().task_queue,
            start_to_close_timeout=SYNTHESIS_TIMEOUT,
        )
        self.phase = "complete"
        return report

    @workflow.query
    def status(self) -> dict:
        """Return current workflow status."""
        return {
            "phase": self.phase,
            "topic": self.topic,
            "completed_searches": self.completed_searches,
            "failed_searches": self.failed_searches,
        }


def _failed_query_payloads(queries: list[str], results: list[object]) -> list[dict[str, str]]:
    failed = []
    for query, result in zip(queries, results):
        if isinstance(result, Exception):
            failed.append({"query": query, "error": str(result)})
    return failed
```

- [ ] **Step 4: Run workflow tests**

Run:

```bash
uv run python -m pytest tests/research/test_deep_research_workflow.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add research/deep_research_workflows.py tests/research/test_deep_research_workflow.py
git commit -m "feat: add coco deep research workflow"
```

---

### Task 5: Register worker helpers and CLI commands

**Files:**
- Modify: `research/temporal.py`
- Modify: `research/cli.py`
- Test: `tests/research/test_deep_research_cli.py`

- [ ] **Step 1: Write failing registration and CLI tests**

Create `tests/research/test_deep_research_cli.py`:

```python
"""Tests for deep-research worker registration and CLI wiring."""

from __future__ import annotations

import argparse

import research.cli
import research.deep_research_activities as activities
import research.deep_research_workflows as workflows
import research.temporal


def test_build_deep_research_worker_config_registers_workflow_and_activities() -> None:
    config = research.temporal.build_deep_research_worker_config(
        address="localhost:7233",
        task_queue="deep-research",
        activity_workers=2,
    )

    assert workflows.DeepResearchWorkflow in config.workflows
    assert activities.generate_subtopics_with_coco in config.activities
    assert activities.generate_search_queries_with_coco in config.activities
    assert activities.search_web_with_coco in config.activities
    assert activities.synthesize_report_with_coco in config.activities
    assert config.activity_workers == 2


def test_deep_research_start_parser_accepts_topic_and_wait() -> None:
    parser = research.cli.build_parser()
    args = parser.parse_args(
        [
            "deep-research",
            "start",
            "Temporal AI workflows",
            "--task-queue",
            "deep-research",
            "--wait",
            "--timeout-seconds",
            "1",
        ]
    )

    assert args.command == "deep-research"
    assert args.deep_research_command == "start"
    assert args.topic == "Temporal AI workflows"
    assert args.task_queue == "deep-research"
    assert args.wait is True
    assert args.timeout_seconds == 1


def test_deep_research_worker_parser_accepts_activity_workers() -> None:
    parser = research.cli.build_parser()
    args = parser.parse_args(
        [
            "deep-research",
            "worker",
            "--activity-workers",
            "3",
        ]
    )

    assert args.deep_research_command == "worker"
    assert args.activity_workers == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/research/test_deep_research_cli.py -q
```

Expected: fail because `build_deep_research_worker_config` and parser commands do not exist.

- [ ] **Step 3: Add Temporal registration helpers**

Modify `research/temporal.py`:

```python
import research.deep_research_activities
import research.deep_research_workflows
```

Add these functions near the existing worker and workflow helpers:

```python
def build_deep_research_worker_config(
    *,
    address: str = DEFAULT_ADDRESS,
    task_queue: str = DEFAULT_TASK_QUEUE,
    activity_workers: int = 1,
) -> WorkerConfig:
    """Build a worker config for COCO-backed deep research."""
    if activity_workers < 1:
        raise ValueError("activity_workers must be at least 1.")
    return WorkerConfig(
        address=address,
        task_queue=task_queue,
        workflows=[research.deep_research_workflows.DeepResearchWorkflow],
        activities=[
            research.deep_research_activities.generate_subtopics_with_coco,
            research.deep_research_activities.generate_search_queries_with_coco,
            research.deep_research_activities.search_web_with_coco,
            research.deep_research_activities.synthesize_report_with_coco,
        ],
        activity_workers=activity_workers,
    )


def make_deep_research_workflow_id(prefix: str = "deep-research") -> str:
    """Return a unique workflow id for deep research."""
    return f"{prefix}-{uuid.uuid4().hex}"


async def start_deep_research_workflow(
    *,
    address: str,
    task_queue: str,
    request: dict[str, object],
    workflow_id: str | None = None,
) -> str:
    """Start a deep-research workflow and return its workflow id."""
    resolved_workflow_id = workflow_id or make_deep_research_workflow_id()
    client = await temporalio.client.Client.connect(address)
    await client.start_workflow(
        research.deep_research_workflows.DeepResearchWorkflow.run,
        args=[request],
        id=resolved_workflow_id,
        task_queue=task_queue,
        id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
    )
    return resolved_workflow_id
```

- [ ] **Step 4: Add CLI parser and command handlers**

Modify `research/cli.py` imports:

```python
import asyncio
import json
```

Add parser setup inside `build_parser()`:

```python
    deep_research_parser = subparsers.add_parser("deep-research")
    deep_research_subparsers = deep_research_parser.add_subparsers(
        dest="deep_research_command",
        required=True,
    )

    deep_start_parser = deep_research_subparsers.add_parser("start")
    deep_start_parser.add_argument("topic")
    deep_start_parser.add_argument("--address", default=research.temporal.DEFAULT_ADDRESS)
    deep_start_parser.add_argument(
        "--task-queue",
        default=research.temporal.DEFAULT_TASK_QUEUE,
    )
    deep_start_parser.add_argument("--workflow-id", default="")
    deep_start_parser.add_argument("--max-subtopics", default=5, type=int)
    deep_start_parser.add_argument("--queries-per-subtopic", default=3, type=int)
    deep_start_parser.add_argument("--max-search-results", default=5, type=int)
    deep_start_parser.add_argument("--workspace", default="")
    deep_start_parser.add_argument("--wait", action="store_true")
    deep_start_parser.add_argument("--timeout-seconds", default=0.0, type=float)

    deep_worker_parser = deep_research_subparsers.add_parser("worker")
    deep_worker_parser.add_argument(
        "--address",
        default=research.temporal.DEFAULT_ADDRESS,
    )
    deep_worker_parser.add_argument(
        "--task-queue",
        default=research.temporal.DEFAULT_TASK_QUEUE,
    )
    deep_worker_parser.add_argument("--activity-workers", default=1, type=int)
```

Add handlers above `main()`:

```python
async def command_deep_research_start(args: argparse.Namespace) -> int:
    """Start one COCO-backed deep-research workflow."""
    request = {
        "topic": args.topic,
        "max_subtopics": args.max_subtopics,
        "queries_per_subtopic": args.queries_per_subtopic,
        "max_search_results": args.max_search_results,
        "workspace": args.workspace,
    }
    workflow_id = await research.temporal.start_deep_research_workflow(
        address=args.address,
        task_queue=args.task_queue,
        request=request,
        workflow_id=args.workflow_id or None,
    )
    print(f"Started Temporal workflow {workflow_id}")
    if args.wait:
        result = await research.temporal.wait_for_workflow_result(
            address=args.address,
            workflow_id=workflow_id,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


async def command_deep_research_worker(args: argparse.Namespace) -> None:
    """Run a worker for COCO-backed deep research."""
    config = research.temporal.build_deep_research_worker_config(
        address=args.address,
        task_queue=args.task_queue,
        activity_workers=args.activity_workers,
    )
    await research.temporal.run_worker(config)
```

Add dispatch inside `main()`:

```python
    if args.command == "deep-research" and args.deep_research_command == "start":
        return asyncio.run(command_deep_research_start(args))
    if args.command == "deep-research" and args.deep_research_command == "worker":
        asyncio.run(command_deep_research_worker(args))
        return 0
```

- [ ] **Step 5: Run CLI tests**

Run:

```bash
uv run python -m pytest tests/research/test_deep_research_cli.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add research/temporal.py research/cli.py tests/research/test_deep_research_cli.py
git commit -m "feat: register coco deep research worker"
```

---

### Task 6: Run focused and full verification

**Files:**
- Verify all files changed by Tasks 1-5.

- [ ] **Step 1: Run focused deep-research tests**

Run:

```bash
uv run python -m pytest \
  tests/research/test_deep_research_models.py \
  tests/research/test_coco_cli.py \
  tests/research/test_deep_research_activities.py \
  tests/research/test_deep_research_workflow.py \
  tests/research/test_deep_research_cli.py \
  -q
```

Expected: all deep-research tests pass.

- [ ] **Step 2: Run existing research and experiment tests**

Run:

```bash
uv run python -m pytest tests/research qwen-vl-finetune/tests/experiments -q
```

Expected: existing research and Qwen experiment tests still pass.

- [ ] **Step 3: Run lint and format checks**

Run:

```bash
uv run ruff check research tests/research qwen-vl-finetune/tests/experiments
uv run ruff format --check research tests/research qwen-vl-finetune/tests/experiments
```

Expected: both commands pass.

- [ ] **Step 4: Inspect help output for the new CLI**

Run:

```bash
uv run research deep-research --help
uv run research deep-research start --help
uv run research deep-research worker --help
```

Expected: help output lists `start`, `worker`, COCO workflow arguments, Temporal address/task queue arguments, and `--activity-workers`.

- [ ] **Step 5: Commit final verification note if docs changed**

If implementation changed the plan or spec while executing, commit those docs:

```bash
git add docs/superpowers/specs/2026-05-26-coco-deep-research-design.md \
  docs/superpowers/plans/2026-05-26-coco-deep-research-workflow.md
git commit -m "docs: update coco deep research plan"
```

If no docs changed, do not create an empty commit.

---

## Self-review notes

- Spec coverage: Tasks 1-5 cover models, COCO subprocess execution, all four COCO activities, deterministic workflow orchestration, partial search failure, all-search failure, worker registration, CLI entrypoints, and Qwen isolation.
- Test coverage: Unit tests cover parsers and command execution. Activity tests fake COCO. Workflow tests use fake activities and prove partial failure behavior without the real CLI.
- Dependency policy: The plan uses `uv run` and does not add conditional imports or dependency-skipping tests.
- OpenAI removal: No OpenAI SDK/API code is introduced.
