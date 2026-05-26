"""Tests for the research command-line interface."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import research.db
import research.models
from research import cli


def test_db_init_command_creates_sqlite(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "research.sqlite"

    rc = cli.main(["--db", str(db_path), "db", "init"])

    assert rc == 0
    assert db_path.exists()


def test_status_command_handles_empty_db(tmp_path: pathlib.Path, capsys) -> None:
    db_path = tmp_path / "research.sqlite"
    cli.main(["--db", str(db_path), "db", "init"])

    rc = cli.main(["--db", str(db_path), "status"])

    assert rc == 0
    assert "No experiments" in capsys.readouterr().out


def test_report_selected_prints_discovered_variants(
    tmp_path: pathlib.Path,
    capsys,
) -> None:
    """Selected report should expose the best discovered model variants."""
    db_path = tmp_path / "research.sqlite"
    _create_report_fixture(db_path, tmp_path / "artifacts")

    rc = cli.main(["--db", str(db_path), "report", "selected"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Selected Intents" in out
    assert "accurate" in out
    assert "val_loss" in out
    assert "0.2" in out
    assert "BATCH_SIZE" in out
    assert "slow" not in out


def test_report_experiment_prints_latest_trial_summary(
    tmp_path: pathlib.Path,
    capsys,
) -> None:
    """Experiment reports should show intent, metrics, and artifact report path."""
    db_path = tmp_path / "research.sqlite"
    experiment_ids = _create_report_fixture(db_path, tmp_path / "artifacts")

    rc = cli.main(["--db", str(db_path), "report", str(experiment_ids[0])])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Experiment 1" in out
    assert "Intent: slow" in out
    assert "Status: succeeded" in out
    assert "val_loss" in out
    assert "report.md" in out


def test_cli_module_runs_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "research.cli", "--help"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert "Scientific experiment manager" in result.stdout


def _create_report_fixture(
    db_path: pathlib.Path,
    artifact_root: pathlib.Path,
) -> list[int]:
    research.db.init_db(db_path)
    slow_intent_id = research.db.insert_intent(
        db_path,
        research.models.Intent(
            "fake",
            "unit-model",
            "cpu",
            "probe",
            "slow",
            {"BATCH_SIZE": 1},
        ),
    )
    accurate_intent_id = research.db.insert_intent(
        db_path,
        research.models.Intent(
            "fake",
            "unit-model",
            "cpu",
            "probe",
            "accurate",
            {"BATCH_SIZE": 2},
        ),
    )
    slow_experiment_id = research.db.create_experiment(
        db_path,
        intent_id=slow_intent_id,
        adapter="fake",
        artifact_root=str(artifact_root),
        artifact_subdir="fake/1",
    )
    accurate_experiment_id = research.db.create_experiment(
        db_path,
        intent_id=accurate_intent_id,
        adapter="fake",
        artifact_root=str(artifact_root),
        artifact_subdir="fake/2",
    )
    slow_trial_run_id = research.db.create_trial_run(
        db_path,
        experiment_id=slow_experiment_id,
        attempt=1,
    )
    accurate_trial_run_id = research.db.create_trial_run(
        db_path,
        experiment_id=accurate_experiment_id,
        attempt=1,
    )
    research.db.transition_trial_run(
        db_path,
        slow_trial_run_id,
        "succeeded",
        metrics={"val_loss": 0.7},
        report_path=str(artifact_root / "fake" / "1" / "attempt-1" / "report.md"),
    )
    research.db.transition_trial_run(
        db_path,
        accurate_trial_run_id,
        "succeeded",
        metrics={"val_loss": 0.2},
        report_path=str(artifact_root / "fake" / "2" / "attempt-1" / "report.md"),
    )
    research.db.transition_experiment(db_path, slow_experiment_id, "succeeded")
    research.db.transition_experiment(db_path, accurate_experiment_id, "succeeded")
    research.db.transition_intent(
        db_path,
        slow_intent_id,
        "rejected",
        score={"metric": "val_loss", "value": 0.7},
    )
    research.db.transition_intent(
        db_path,
        accurate_intent_id,
        "selected",
        score={"metric": "val_loss", "value": 0.2},
    )
    return [slow_experiment_id, accurate_experiment_id]
