"""Temporal workflows for scientific experiment methodology.

Defines ProbeWorkflow (orchestrates multiple trial child workflows) and
TrialWorkflow (executes a single trial activity with retry policy).
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

import research.activities


@workflow.defn
class TrialWorkflow:
    """Runs a single experiment trial via the run_trial_activity.

    Attributes:
        phase: Current workflow phase (created, running, complete).
        experiment_id: The experiment being executed.
        latest_result: Most recent activity result dict.
    """

    def __init__(self) -> None:
        self.phase = "created"
        self.experiment_id: int | None = None
        self.latest_result: dict | None = None

    @workflow.run
    async def run(self, db_path: str, experiment_id: int, attempt: int = 1) -> dict:
        """Execute the trial activity and return its result.

        Args:
            db_path: Path to the SQLite research database.
            experiment_id: ID of the experiment to run.
            attempt: Trial attempt number, defaults to 1.

        Returns:
            Activity result dict with status, metrics, and failure info.
        """
        self.phase = "running"
        self.experiment_id = experiment_id
        self.latest_result = await workflow.execute_activity(
            research.activities.run_trial_activity,
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
class ProbeWorkflow:
    """Orchestrates multiple TrialWorkflow children for a probe campaign.

    Attributes:
        phase: Current workflow phase (created, probe, complete).
        completed: Count of finished child trial workflows.
        selected_intents: Intent IDs selected during the probe.
    """

    def __init__(self) -> None:
        self.phase = "created"
        self.completed = 0
        self.selected_intents: list[int] = []

    @workflow.run
    async def run(self, db_path: str, experiment_ids: list[int]) -> dict:
        """Execute child TrialWorkflow for each experiment ID.

        Args:
            db_path: Path to the SQLite research database.
            experiment_ids: List of experiment IDs to probe.

        Returns:
            Dict with results list and completed count.
        """
        self.phase = "probe"
        results = []
        for experiment_id in experiment_ids:
            result = await workflow.execute_child_workflow(
                TrialWorkflow.run,
                args=[db_path, experiment_id, 1],
                id=f"trial-{experiment_id}",
            )
            results.append(result)
            self.completed += 1
        self.phase = "complete"
        return {"results": results, "completed": self.completed}

    @workflow.query
    def status(self) -> dict:
        """Return the current probe workflow state."""
        return {
            "phase": self.phase,
            "completed": self.completed,
            "selected_intents": list(self.selected_intents),
        }
