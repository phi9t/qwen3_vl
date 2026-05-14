"""Generic scientific experiment adapter models."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Protocol


JsonDict = dict[str, object]


@dataclasses.dataclass(frozen=True)
class ProbeRequest:
    """Request to generate candidate probe intents."""

    model: str
    profile: str
    objective: JsonDict = dataclasses.field(default_factory=dict)
    budget: JsonDict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Intent:
    """Candidate experiment configuration produced by an adapter."""

    adapter: str
    model: str
    profile: str
    phase: str
    name: str
    config: JsonDict
    objective: JsonDict = dataclasses.field(default_factory=dict)
    source: str = "manual"


@dataclasses.dataclass(frozen=True)
class TrialContext:
    """Runtime context for preflight, trial execution, and analysis."""

    experiment_id: int
    trial_run_id: int
    attempt: int
    worktree: pathlib.Path
    artifact_dir: pathlib.Path
    db_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class PreflightResult:
    """Result of adapter-specific readiness checks."""

    ok: bool
    checks: JsonDict = dataclasses.field(default_factory=dict)
    message: str = ""


@dataclasses.dataclass(frozen=True)
class TrialCommand:
    """Subprocess command an adapter wants the runner to execute."""

    argv: list[str]
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    cwd: pathlib.Path | None = None


@dataclasses.dataclass(frozen=True)
class ProgressUpdate:
    """Parsed progress update from a trial event or log line."""

    step: int | None = None
    metrics: JsonDict = dataclasses.field(default_factory=dict)
    message: str = ""


@dataclasses.dataclass(frozen=True)
class TrialReport:
    """Final trial report produced by adapter analysis."""

    status: str
    metrics: JsonDict = dataclasses.field(default_factory=dict)
    failure: JsonDict = dataclasses.field(default_factory=dict)
    summary: str = ""


class ExperimentAdapter(Protocol):
    """Contract implemented by model-specific experiment adapters."""

    name: str

    def generate_probe_intents(self, request: ProbeRequest) -> list[Intent]:
        """Generate candidate intents for a probe request."""
        ...

    def preflight(self, intent: Intent, context: TrialContext) -> PreflightResult:
        """Check whether an intent can run in the current context."""
        ...

    def build_trial(self, intent: Intent, context: TrialContext) -> TrialCommand:
        """Build the command used to run a trial attempt."""
        ...

    def parse_progress(self, event_or_log_line: str) -> ProgressUpdate | None:
        """Parse an optional progress update from an event or log line."""
        ...

    def analyze_result(self, context: TrialContext) -> TrialReport:
        """Analyze trial artifacts and produce the final report."""
        ...
