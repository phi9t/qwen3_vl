"""Tests for the generic research SQLite state store."""

from __future__ import annotations

import pathlib

from research.db import (
    create_experiment,
    create_trial_run,
    get_experiment,
    get_intent,
    get_trial_run,
    init_db,
    insert_intent,
    transition_experiment,
    transition_intent,
    transition_trial_run,
)
from research.models import Intent


def test_init_db_and_insert_intent(tmp_path: pathlib.Path) -> None:
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


def test_experiment_and_trial_state_transitions(tmp_path: pathlib.Path) -> None:
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
    trial_run = get_trial_run(db_path, trial_run_id)
    assert intent["status"] == "selected"
    assert intent["score_json"] == {"metric": 1.0}
    assert experiment["status"] == "running"
    assert experiment["temporal_workflow_id"] == "wf-1"
    assert trial_run["status"] == "succeeded"
    assert trial_run["heartbeat_json"] == {"step": 1}
    assert trial_run["metrics_json"] == {"metric": 1.0}
    assert trial_run["failure_json"] == {}
    assert trial_run["report_path"] == "report.md"
    assert trial_run["finished_at"] is not None


def test_transition_missing_ids_raise_key_error(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "research.sqlite"
    init_db(db_path)

    for transition in (
        lambda: transition_intent(db_path, 999, "selected"),
        lambda: transition_experiment(db_path, 999, "running"),
        lambda: transition_trial_run(db_path, 999, "succeeded"),
    ):
        try:
            transition()
        except KeyError:
            continue
        raise AssertionError("missing row transition should raise KeyError")
