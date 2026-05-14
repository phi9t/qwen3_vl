"""Qwen-VL Temporal workflows backed by concrete Qwen activities."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

import experiments.qwen_temporal


@workflow.defn
class QwenTrialWorkflow:
    """Runs one Qwen-VL experiment trial."""

    def __init__(self) -> None:
        self.phase = "created"
        self.experiment_id: int | None = None
        self.latest_result: dict | None = None

    @workflow.run
    async def run(self, db_path: str, experiment_id: int, attempt: int = 1) -> dict:
        """Execute a single Qwen-VL trial activity."""
        self.phase = "running"
        self.experiment_id = experiment_id
        self.latest_result = await workflow.execute_activity(
            experiments.qwen_temporal.qwen_run_trial_activity,
            args=[db_path, experiment_id, attempt],
            start_to_close_timeout=timedelta(hours=12),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        self.phase = "complete"
        return self.latest_result

    @workflow.query
    def status(self) -> dict:
        """Return the current trial workflow state."""
        return {
            "phase": self.phase,
            "experiment_id": self.experiment_id,
            "latest_result": self.latest_result,
        }


@workflow.defn
class QwenProbeWorkflow:
    """Runs a Qwen-VL probe campaign as a sequence of trial workflows."""

    def __init__(self) -> None:
        self.phase = "created"
        self.completed = 0

    @workflow.run
    async def run(self, db_path: str, experiment_ids: list[int]) -> dict:
        """Execute Qwen trial children for each experiment ID."""
        self.phase = "probe"
        results = []
        for experiment_id in experiment_ids:
            result = await workflow.execute_child_workflow(
                QwenTrialWorkflow.run,
                args=[db_path, experiment_id, 1],
                id=f"qwen-trial-{experiment_id}",
            )
            results.append(result)
            self.completed += 1
        self.phase = "complete"
        return {"results": results, "completed": self.completed}

    @workflow.query
    def status(self) -> dict:
        """Return the current probe workflow state."""
        return {"phase": self.phase, "completed": self.completed}
