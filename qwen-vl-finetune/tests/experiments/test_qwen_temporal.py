"""Tests for Qwen-VL Temporal activity wrappers."""

from __future__ import annotations

import pathlib

import experiments.qwen_temporal

import research.db
import research.models


class TemporalFakeQwenAdapter:
    """Small adapter used to exercise Qwen Temporal activity wiring."""

    name = "qwen_vl"

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
        return research.models.TrialCommand(
            argv=["python", "-c", "print('val_loss: 0.5')"],
            env={},
        )

    def parse_progress(
        self,
        event_or_log_line: str,
    ) -> research.models.ProgressUpdate | None:
        if "val_loss:" not in event_or_log_line:
            return None
        return research.models.ProgressUpdate(
            metrics={"val_loss": 0.5},
            message=event_or_log_line.strip(),
        )

    def analyze_result(
        self,
        context: research.models.TrialContext,
    ) -> research.models.TrialReport:
        return research.models.TrialReport(
            status="succeeded",
            metrics={"val_loss": 0.5},
            failure={},
            summary="temporal fake succeeded",
        )


def test_qwen_trial_activity_uses_concrete_qwen_adapter(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    """Qwen activity should pass a concrete adapter into research helpers."""
    monkeypatch.setattr(
        experiments.qwen_temporal.experiments.qwen_adapter,
        "QwenVlAdapter",
        TemporalFakeQwenAdapter,
    )
    db_path, experiment_id = _create_experiment(tmp_path)

    result = experiments.qwen_temporal.qwen_run_trial_activity(
        str(db_path),
        experiment_id,
        1,
    )

    assert result["status"] == "succeeded"
    assert result["metrics"] == {"val_loss": 0.5}
    assert result["failure"] == {}


def _create_experiment(tmp_path: pathlib.Path) -> tuple[pathlib.Path, int]:
    db_path = tmp_path / "research.sqlite"
    artifact_root = tmp_path / "artifacts"
    research.db.init_db(db_path)
    intent_id = research.db.insert_intent(
        db_path,
        research.models.Intent(
            "qwen_vl",
            "unit-model",
            "b200",
            "probe",
            "small",
            {},
        ),
    )
    experiment_id = research.db.create_experiment(
        db_path,
        intent_id=intent_id,
        adapter="qwen_vl",
        artifact_root=str(artifact_root),
        artifact_subdir="qwen_vl/1",
    )
    return db_path, experiment_id
