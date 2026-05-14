from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from research.workflows import SingleTrialWorkflow


SPEC = {
    "profile": "b200",
    "phase": "probe",
    "trial": "temporal-single-trial",
    "env": {"BATCH_SIZE": "2"},
}


def _ok_analysis_payload() -> dict:
    return {
        "schema_version": 1,
        "trial_id": "b200/probe/temporal-single-trial",
        "attempt": 1,
        "status": "ok",
        "failure_reason": "none",
        "root_cause": "none",
        "expected_failure": False,
        "metrics": {
            "status": "ok",
            "val_loss": 0.0,
            "peak_vram_mb": 0.0,
            "throughput_steps_per_sec": None,
            "failure_reason": "none",
        },
        "symptoms": [],
        "evidence_refs": ["logs/b200/probe/temporal-single-trial/attempt_001/run.log"],
        "actions": [],
        "recommendations": [],
        "artifact_root_ref": "artifacts",
        "artifact_refs": {
            "full_log": "logs/b200/probe/temporal-single-trial/attempt_001/run.log",
            "output_dir": "outputs/b200/probe/temporal-single-trial/attempt_001",
        },
        "artifact_dir": "experiments/runs/b200/probe/temporal-single-trial/attempt_001",
    }


def _crash_analysis_payload() -> dict:
    payload = _ok_analysis_payload()
    payload.update(
        {
            "status": "crash",
            "failure_reason": "launcher_error",
            "root_cause": "launcher_failed",
            "metrics": {
                "status": "crash",
                "val_loss": None,
                "peak_vram_mb": None,
                "throughput_steps_per_sec": None,
                "failure_reason": "launcher_error",
            },
            "symptoms": [
                {"kind": "launcher", "message": "Launcher exited unsuccessfully"}
            ],
            "recommendations": [
                "Inspect analysis evidence before continuing larger campaigns."
            ],
        }
    )
    return payload


async def _execute_single_trial(activity_fn, workflow_id: str) -> dict:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        task_queue = f"single-trial-test-{uuid4()}"
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[SingleTrialWorkflow],
            activities=[activity_fn],
        ):
            return await env.client.execute_workflow(
                SingleTrialWorkflow.run,
                args=[SPEC, True],
                id=workflow_id,
                task_queue=task_queue,
            )


def test_single_trial_workflow_completes_in_temporal_test_environment() -> None:
    calls = []

    @activity.defn(name="run_trial_activity")
    async def fake_run_trial_activity(
        spec_payload: dict,
        dry_run: bool = False,
        fail_on_crash: bool = False,
    ) -> dict:
        calls.append((spec_payload, dry_run, fail_on_crash))
        return _ok_analysis_payload()

    result = asyncio.run(
        _execute_single_trial(
            fake_run_trial_activity,
            f"single-trial-success-{uuid4()}",
        )
    )

    assert calls == [(SPEC, True, True)]
    assert result["phase"] == "complete"
    assert result["spec"] == SPEC
    assert result["analysis"]["status"] == "ok"
    assert result["metrics"]["status"] == "ok"


def test_single_trial_workflow_fails_when_activity_reports_trial_failure() -> None:
    calls = []

    @activity.defn(name="run_trial_activity")
    async def fake_run_trial_activity(
        spec_payload: dict,
        dry_run: bool = False,
        fail_on_crash: bool = False,
    ) -> dict:
        calls.append((spec_payload, dry_run, fail_on_crash))
        payload = _crash_analysis_payload()
        raise ApplicationError(
            "Trial b200/probe/temporal-single-trial failed: launcher_error",
            payload,
            type="TrialFailed",
            non_retryable=True,
        )

    with pytest.raises(WorkflowFailureError) as exc_info:
        asyncio.run(
            _execute_single_trial(
                fake_run_trial_activity,
                f"single-trial-failure-{uuid4()}",
            )
        )

    assert calls == [(SPEC, True, True)]
    activity_error = exc_info.value.cause
    assert isinstance(activity_error, ActivityError)
    application_error = activity_error.cause
    assert isinstance(application_error, ApplicationError)
    assert application_error.type == "TrialFailed"
    assert application_error.non_retryable is True
    assert "launcher_error" in str(application_error)
    assert application_error.details[0]["status"] == "crash"
    assert application_error.details[0]["failure_reason"] == "launcher_error"
