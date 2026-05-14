"""Tests for research Temporal activities."""

from __future__ import annotations

import json
import pathlib

import research.activities
import research.db
import research.models


class FailingPreflightAdapter:
    name = "failing_preflight"

    def generate_probe_intents(
        self,
        request: research.models.ProbeRequest,
    ) -> list[research.models.Intent]:
        return []

    def preflight(
        self,
        intent: research.models.Intent,
        context: research.models.TrialContext,
    ) -> research.models.PreflightResult:
        return research.models.PreflightResult(
            ok=False,
            checks={"adapter_specific": "failed"},
            message="adapter preflight failed",
        )

    def build_trial(
        self,
        intent: research.models.Intent,
        context: research.models.TrialContext,
    ) -> research.models.TrialCommand:
        raise AssertionError("build_trial should not run after failed preflight")

    def parse_progress(self, event_or_log_line: str) -> None:
        return None

    def analyze_result(
        self,
        context: research.models.TrialContext,
    ) -> research.models.TrialReport:
        raise AssertionError("analyze_result should not run after failed preflight")


class BuildErrorAdapter:
    name = "build_error"

    def generate_probe_intents(
        self,
        request: research.models.ProbeRequest,
    ) -> list[research.models.Intent]:
        return []

    def preflight(
        self,
        intent: research.models.Intent,
        context: research.models.TrialContext,
    ) -> research.models.PreflightResult:
        return research.models.PreflightResult(ok=True, checks={}, message="ok")

    def build_trial(
        self,
        intent: research.models.Intent,
        context: research.models.TrialContext,
    ) -> research.models.TrialCommand:
        raise RuntimeError("cannot build command")

    def parse_progress(self, event_or_log_line: str) -> None:
        return None

    def analyze_result(
        self,
        context: research.models.TrialContext,
    ) -> research.models.TrialReport:
        raise AssertionError("analyze_result should not run after build failure")


def _create_experiment(
    tmp_path: pathlib.Path,
    adapter: str,
) -> tuple[pathlib.Path, pathlib.Path, int, int]:
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
        adapter=adapter,
        artifact_root=str(artifact_root),
        artifact_subdir="fake/1",
    )
    return db_path, artifact_root, intent_id, experiment_id


def test_run_trial_activity_updates_db_and_writes_report(
    tmp_path: pathlib.Path,
) -> None:
    db_path, artifact_root, intent_id, experiment_id = _create_experiment(
        tmp_path,
        "tests.research.fake_adapter:FakeAdapter",
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


def test_run_trial_activity_persists_preflight_failure(
    tmp_path: pathlib.Path,
) -> None:
    db_path, artifact_root, _intent_id, experiment_id = _create_experiment(
        tmp_path,
        "tests.research.test_activities:FailingPreflightAdapter",
    )

    result = research.activities.run_trial_activity(str(db_path), experiment_id, 1)

    artifact_dir = artifact_root / "failing_preflight" / str(experiment_id) / "attempt-1"
    trial_run = research.db.get_trial_run(db_path, 1)
    assert result["status"] == "failed"
    assert result["failure"]["reason"] == "preflight_failed"
    assert research.db.get_experiment(db_path, experiment_id)["status"] == "failed"
    assert trial_run["status"] == "failed"
    assert trial_run["finished_at"] is not None
    assert trial_run["failure_json"]["reason"] == "preflight_failed"
    assert (artifact_dir / "report.md").exists()


def test_run_trial_activity_persists_adapter_exception(
    tmp_path: pathlib.Path,
) -> None:
    db_path, artifact_root, _intent_id, experiment_id = _create_experiment(
        tmp_path,
        "tests.research.test_activities:BuildErrorAdapter",
    )

    result = research.activities.run_trial_activity(str(db_path), experiment_id, 1)

    artifact_dir = artifact_root / "build_error" / str(experiment_id) / "attempt-1"
    trial_run = research.db.get_trial_run(db_path, 1)
    assert result["status"] == "failed"
    assert result["failure"]["reason"] == "activity_exception"
    assert "cannot build command" in result["failure"]["message"]
    assert research.db.get_experiment(db_path, experiment_id)["status"] == "failed"
    assert trial_run["status"] == "failed"
    assert trial_run["failure_json"]["reason"] == "activity_exception"
    assert (artifact_dir / "report.md").exists()
