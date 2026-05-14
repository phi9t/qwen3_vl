"""Tests for generic trial command runners."""

from __future__ import annotations

import pathlib

import research.adapters
import research.models
import research.runners


def test_run_trial_command_writes_log_and_progress(
    tmp_path: pathlib.Path,
) -> None:
    adapter = research.adapters.load_adapter("tests.research.fake_adapter:FakeAdapter")
    context = research.models.TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "research.sqlite",
    )
    context.artifact_dir.mkdir()
    intent = research.models.Intent(
        "fake",
        "unit-model",
        "cpu",
        "probe",
        "small",
        {"batch_size": 1},
    )
    command = adapter.build_trial(intent, context)

    result = research.runners.run_trial_command(adapter, command, context)

    assert result.returncode == 0
    assert result.progress[-1].metrics == {"metric": 1.0}
    assert (
        (context.artifact_dir / "run.log").read_text(encoding="utf-8").strip()
        == "metric=1.0"
    )
