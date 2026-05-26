"""Tests for direct-object probe creation helpers."""

from __future__ import annotations

import contextlib
import pathlib

import research.db
import research.models
import research.probes
import tests.research.fake_adapter


def test_create_probe_experiments_uses_concrete_adapter(
    tmp_path: pathlib.Path,
) -> None:
    """Probe helper should persist intents without dynamic adapter loading."""
    db_path = tmp_path / "research.sqlite"
    artifact_root = tmp_path / "artifacts"
    adapter = tests.research.fake_adapter.FakeAdapter()

    result = research.probes.create_probe_experiments(
        db_path,
        adapter=adapter,
        request=research.models.ProbeRequest(
            model="unit-model",
            profile="cpu",
            objective={"metric": "min"},
        ),
        artifact_root=artifact_root,
        adapter_ref=adapter.name,
    )

    assert result.intent_ids == [1]
    assert result.experiment_ids == [1]
    with contextlib.closing(research.db.connect(db_path)) as conn:
        intent = conn.execute("SELECT * FROM intents").fetchone()
        experiment = conn.execute("SELECT * FROM experiments").fetchone()

    assert intent["adapter"] == "fake"
    assert experiment["adapter"] == "fake"
    assert experiment["artifact_root"] == str(artifact_root)
    assert experiment["artifact_subdir"] == "fake/1"


def test_select_probe_results_marks_lowest_metric_selected(
    tmp_path: pathlib.Path,
) -> None:
    """Probe selection should preserve the best metric and reject the rest."""
    db_path = tmp_path / "research.sqlite"
    artifact_root = tmp_path / "artifacts"
    research.db.init_db(db_path)
    first_intent_id = _insert_trial_result(
        db_path,
        artifact_root,
        name="slow",
        metrics={"val_loss": 0.7},
    )
    second_intent_id = _insert_trial_result(
        db_path,
        artifact_root,
        name="accurate",
        metrics={"val_loss": 0.3},
    )
    failed_intent_id = _insert_trial_result(
        db_path,
        artifact_root,
        name="failed",
        metrics={},
        status="failed",
        failure={"reason": "launcher_failed"},
    )

    result = research.probes.select_probe_results(
        db_path,
        [1, 2, 3],
        objective={"metric": "val_loss", "direction": "minimize", "top_k": 1},
    )

    assert result.selected_intent_ids == [second_intent_id]
    assert result.rejected_intent_ids == [first_intent_id, failed_intent_id]
    assert research.db.get_intent(db_path, second_intent_id)["status"] == "selected"
    assert research.db.get_intent(db_path, first_intent_id)["status"] == "rejected"
    assert research.db.get_intent(db_path, failed_intent_id)["status"] == "rejected"
    assert research.db.get_intent(db_path, second_intent_id)["score_json"] == {
        "metric": "val_loss",
        "value": 0.3,
    }


def _insert_trial_result(
    db_path: pathlib.Path,
    artifact_root: pathlib.Path,
    *,
    name: str,
    metrics: dict[str, object],
    status: str = "succeeded",
    failure: dict[str, object] | None = None,
) -> int:
    intent_id = research.db.insert_intent(
        db_path,
        research.models.Intent(
            "fake",
            "unit-model",
            "cpu",
            "probe",
            name,
            {},
        ),
    )
    experiment_id = research.db.create_experiment(
        db_path,
        intent_id=intent_id,
        adapter="fake",
        artifact_root=str(artifact_root),
        artifact_subdir=f"fake/{intent_id}",
    )
    trial_run_id = research.db.create_trial_run(
        db_path,
        experiment_id=experiment_id,
        attempt=1,
    )
    research.db.transition_trial_run(
        db_path,
        trial_run_id,
        status,
        metrics=metrics,
        failure=failure or {},
    )
    return intent_id
