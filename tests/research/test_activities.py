"""Tests for research Temporal activities."""

from __future__ import annotations

import json
import pathlib

import research.activities
import research.db
import research.models


def test_run_trial_activity_updates_db_and_writes_report(
    tmp_path: pathlib.Path,
) -> None:
    db_path = tmp_path / "research.sqlite"
    artifact_root = tmp_path / "artifacts"
    research.db.init_db(db_path)
    intent_id = research.db.insert_intent(
        db_path,
        research.models.Intent(
            "fake",
            "unit-model",
            "cpu",
            "probe",
            "small",
            {"batch_size": 1},
        ),
    )
    experiment_id = research.db.create_experiment(
        db_path,
        intent_id=intent_id,
        adapter="tests.research.fake_adapter:FakeAdapter",
        artifact_root=str(artifact_root),
        artifact_subdir="fake/1",
    )

    result = research.activities.run_trial_activity(str(db_path), experiment_id, 1)

    artifact_dir = artifact_root / "fake" / str(experiment_id) / "attempt-1"
    report_path = artifact_dir / "report.md"
    report_json_path = artifact_dir / "report.json"
    preflight_path = artifact_dir / "preflight.json"
    trial_run = research.db.get_trial_run(db_path, 1)

    assert result["status"] == "succeeded"
    assert result["metrics"] == {"metric": 1.0}
    assert result["failure"] == {}
    assert research.db.get_intent(db_path, intent_id)["status"] == "candidate"
    assert research.db.get_experiment(db_path, experiment_id)["status"] == "succeeded"
    assert report_path.exists()
    assert report_json_path.exists()
    assert preflight_path.exists()
    assert json.loads(preflight_path.read_text(encoding="utf-8")) == {
        "checks": {
            "adapter": "fake",
            "artifact_dir": "ok",
            "db": "ok",
            "fake": "ok",
            "uv": "ok",
            "worktree": "ok",
        },
        "message": "ok",
        "ok": True,
    }
    assert json.loads(report_json_path.read_text(encoding="utf-8"))["status"] == (
        "succeeded"
    )
    assert trial_run["status"] == "succeeded"
    assert trial_run["metrics_json"] == {"metric": 1.0}
    assert trial_run["failure_json"] == {}
    assert trial_run["report_path"] == str(report_path)
