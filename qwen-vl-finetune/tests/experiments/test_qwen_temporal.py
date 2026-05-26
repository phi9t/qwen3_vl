"""Tests for Qwen-VL Temporal activity wrappers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import pathlib
import uuid

import experiments.qwen_temporal
import pytest
import temporalio.exceptions
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

import research.activities
import research.db
import research.models
import research.workflows


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


class FailingTemporalFakeQwenAdapter(TemporalFakeQwenAdapter):
    """Adapter that fails during preflight after the activity creates state."""

    def preflight(
        self,
        intent: research.models.Intent,
        context: research.models.TrialContext,
    ) -> research.models.PreflightResult:
        return research.models.PreflightResult(
            ok=False,
            checks={"qwen": "missing"},
            message="qwen preflight failed",
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
    heartbeats = []
    monkeypatch.setattr(
        experiments.qwen_temporal.research.activities,
        "heartbeat_progress",
        heartbeats.append,
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
    assert heartbeats[-1].metrics == {"val_loss": 0.5}


def test_qwen_trial_activity_raises_temporal_failure_after_persisting_failure(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    """Failed Qwen trials should fail the Temporal activity after persistence."""
    monkeypatch.setattr(
        experiments.qwen_temporal.experiments.qwen_adapter,
        "QwenVlAdapter",
        FailingTemporalFakeQwenAdapter,
    )
    db_path, experiment_id = _create_experiment(tmp_path)

    with pytest.raises(temporalio.exceptions.ApplicationError) as exc_info:
        experiments.qwen_temporal.qwen_run_trial_activity(
            str(db_path),
            experiment_id,
            1,
        )

    trial_run = research.db.get_trial_run(db_path, 1)
    assert exc_info.value.type == "qwen_trial_failed"
    assert research.db.get_experiment(db_path, experiment_id)["status"] == "failed"
    assert trial_run["status"] == "failed"
    assert trial_run["failure_json"]["reason"] == "preflight_failed"


def test_qwen_trial_activity_can_run_real_adapter_dry_run_from_any_cwd(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    """Qwen activity should run a no-GPU script smoke without relying on cwd."""
    monkeypatch.chdir(tmp_path)
    db_path, experiment_id = _create_experiment(
        tmp_path,
        config={"QWEN_RESEARCH_DRY_RUN": "1"},
    )

    result = experiments.qwen_temporal.qwen_run_trial_activity(
        str(db_path),
        experiment_id,
        1,
    )

    trial_run = research.db.get_trial_run(db_path, 1)
    assert result["status"] == "succeeded"
    assert result["metrics"] == {"peak_vram_mb": 0.0, "val_loss": 0.0}
    assert research.db.get_experiment(db_path, experiment_id)["status"] == "succeeded"
    assert trial_run["status"] == "succeeded"


def test_qwen_probe_workflow_can_select_real_adapter_dry_runs(
    tmp_path: pathlib.Path,
) -> None:
    """A real Temporal worker should run Qwen dry-run trials and select one."""
    asyncio.run(_run_qwen_probe_workflow_dry_run(tmp_path))


async def _run_qwen_probe_workflow_dry_run(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "research.sqlite"
    experiment_ids = [
        _create_experiment(tmp_path, db_path=db_path, config=_dry_run_config(name))[1]
        for name in ("small", "large")
    ]
    task_queue = f"qwen-dry-run-{uuid.uuid4()}"
    async with await WorkflowEnvironment.start_local() as env:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[
                    research.workflows.ResearchProbeWorkflow,
                    research.workflows.ResearchTrialWorkflow,
                ],
                activities=[
                    experiments.qwen_temporal.qwen_run_trial_activity,
                    research.activities.select_probe_results_activity,
                ],
                activity_executor=executor,
            ):
                result = await env.client.execute_workflow(
                    research.workflows.ResearchProbeWorkflow.run,
                    args=[
                        str(db_path),
                        experiment_ids,
                        "qwen_run_trial_activity",
                        1,
                        {"metric": "val_loss", "direction": "minimize", "top_k": 1},
                        "select_probe_results_activity",
                    ],
                    id=f"qwen-probe-{uuid.uuid4()}",
                    task_queue=task_queue,
                )

    assert result["completed"] == 2
    assert result["failed"] == 0
    assert result["selection"]["selected_intent_ids"] == [1]
    assert research.db.get_intent(db_path, 1)["status"] == "selected"
    assert research.db.get_intent(db_path, 2)["status"] == "rejected"


def _create_experiment_with_config(
    tmp_path: pathlib.Path,
    *,
    config: dict[str, object],
    db_path: pathlib.Path | None = None,
) -> tuple[pathlib.Path, int]:
    db_path = db_path or tmp_path / "research.sqlite"
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
            config,
        ),
    )
    experiment_id = research.db.create_experiment(
        db_path,
        intent_id=intent_id,
        adapter="qwen_vl",
        artifact_root=str(artifact_root),
        artifact_subdir=f"qwen_vl/{intent_id}",
    )
    return db_path, experiment_id


def _create_experiment(
    tmp_path: pathlib.Path,
    *,
    config: dict[str, object] | None = None,
    db_path: pathlib.Path | None = None,
) -> tuple[pathlib.Path, int]:
    return _create_experiment_with_config(
        tmp_path,
        config=config or {},
        db_path=db_path,
    )


def _dry_run_config(name: str) -> dict[str, object]:
    return {
        "QWEN_RESEARCH_DRY_RUN": "1",
        "RUN_NAME": name,
    }
