from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from temporalio.exceptions import ApplicationError

from research.models import FailureReason, TrialMetrics
from research.observability.schema import TrialAnalysis


def _analysis(status: str, failure_reason: FailureReason = FailureReason.NONE) -> TrialAnalysis:
    return TrialAnalysis(
        trial_id="b200/probe/probe1",
        attempt=1,
        status=status,
        failure_reason=failure_reason,
        root_cause="training_incomplete" if status != "ok" else "none",
        expected_failure=False,
        metrics=TrialMetrics(status=status, failure_reason=failure_reason),
        symptoms=[],
        evidence_refs=["logs/b200/probe/probe1/attempt_001/run.log"],
        actions=[],
        recommendations=[],
        artifact_root_ref="artifacts",
        artifact_refs={
            "full_log": "logs/b200/probe/probe1/attempt_001/run.log",
            "output_dir": "outputs/b200/probe/probe1/attempt_001",
        },
        artifact_dir="experiments/runs/b200/probe/probe1/attempt_001",
    )


class _FakeRunner:
    next_analysis = _analysis("ok")

    def __init__(self, root: Path) -> None:
        self.root = root

    def run_from_spec(self, spec, *, dry_run: bool = False) -> TrialAnalysis:
        return self.next_analysis


def test_run_trial_activity_returns_crashed_analysis_when_failure_tolerant(
    monkeypatch, tmp_path: Path
) -> None:
    from research import activities

    _FakeRunner.next_analysis = _analysis("crash", FailureReason.MISSING_FOOTER)
    monkeypatch.setattr(activities, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(activities, "git_short", lambda root: "abc123")
    monkeypatch.setattr(activities, "SupervisorTrialRunner", _FakeRunner)

    payload = activities.run_trial_activity(
        {"profile": "b200", "phase": "probe", "trial": "probe1", "env": {}},
        dry_run=True,
        fail_on_crash=False,
    )

    assert payload["status"] == "crash"
    results = (tmp_path / "experiments" / "results.tsv").read_text(encoding="utf-8")
    assert "\tb200\tprobe\tprobe1\t" in results
    assert "\tcrash\tmissing_footer\t" in results


def test_run_trial_activity_appends_results_before_failing_single_trial(
    monkeypatch, tmp_path: Path
) -> None:
    from research import activities

    _FakeRunner.next_analysis = _analysis("crash", FailureReason.MISSING_FOOTER)
    monkeypatch.setattr(activities, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(activities, "git_short", lambda root: "abc123")
    monkeypatch.setattr(activities, "SupervisorTrialRunner", _FakeRunner)

    with pytest.raises(ApplicationError) as exc_info:
        activities.run_trial_activity(
            {"profile": "b200", "phase": "probe", "trial": "probe1", "env": {}},
            dry_run=True,
            fail_on_crash=True,
        )

    assert exc_info.value.type == "TrialFailed"
    assert exc_info.value.non_retryable is True
    assert "b200/probe/probe1" in str(exc_info.value)
    assert "missing_footer" in str(exc_info.value)
    assert exc_info.value.details[0]["status"] == "crash"
    results = (tmp_path / "experiments" / "results.tsv").read_text(encoding="utf-8")
    assert "\tb200\tprobe\tprobe1\t" in results
    assert "\tcrash\tmissing_footer\t" in results


def test_run_trial_activity_returns_success_when_single_trial_failures_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    from research import activities

    _FakeRunner.next_analysis = _analysis("ok")
    monkeypatch.setattr(activities, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(activities, "git_short", lambda root: "abc123")
    monkeypatch.setattr(activities, "SupervisorTrialRunner", _FakeRunner)

    payload = activities.run_trial_activity(
        {"profile": "b200", "phase": "probe", "trial": "probe1", "env": {}},
        dry_run=True,
        fail_on_crash=True,
    )

    assert payload["status"] == "ok"


def test_single_trial_workflow_executes_trial_activity_with_failure_enabled(
    monkeypatch,
) -> None:
    from research import workflows

    calls = []

    async def fake_execute_activity(activity, *positional_args, **kwargs):
        calls.append((activity, positional_args, kwargs))
        return {"status": "ok", "metrics": {"status": "ok"}}

    monkeypatch.setattr(workflows.workflow, "execute_activity", fake_execute_activity)

    workflow = workflows.SingleTrialWorkflow()
    result = asyncio.run(
        workflow.run(
            {"profile": "b200", "phase": "probe", "trial": "probe1", "env": {}},
            dry_run=True,
        )
    )

    assert result["phase"] == "complete"
    assert calls[0][2]["args"] == [
        {"profile": "b200", "phase": "probe", "trial": "probe1", "env": {}},
        True,
        True,
    ]
