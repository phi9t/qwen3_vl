"""Generic Temporal workflows for scientific experiment execution."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy


DEFAULT_TRIAL_TIMEOUT = timedelta(hours=12)
DEFAULT_DEMO_ACTIVITY_TIMEOUT = timedelta(minutes=5)


def trial_workflow_id(experiment_id: int, attempt: int) -> str:
    """Return the deterministic workflow id for one trial attempt."""
    return f"research-trial-{experiment_id}-attempt-{attempt}"


@workflow.defn
class ResearchTrialWorkflow:
    """Runs one experiment trial through a concrete registered activity."""

    def __init__(self) -> None:
        self.phase = "created"
        self.experiment_id: int | None = None
        self.latest_result: dict | None = None

    @workflow.run
    async def run(
        self,
        db_path: str,
        experiment_id: int,
        trial_activity: str,
        attempt: int = 1,
    ) -> dict:
        """Execute one trial activity by registered Temporal activity name."""
        self.phase = "running"
        self.experiment_id = experiment_id
        try:
            self.latest_result = await workflow.execute_activity(
                trial_activity,
                args=[db_path, experiment_id, attempt],
                task_queue=workflow.info().task_queue,
                start_to_close_timeout=DEFAULT_TRIAL_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
        except Exception:
            self.phase = "failed"
            raise
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
class ResearchProbeWorkflow:
    """Runs a probe campaign as a sequence of trial child workflows."""

    def __init__(self) -> None:
        self.phase = "created"
        self.completed = 0
        self.failed = 0
        self.current_experiment_id: int | None = None
        self.results: list[dict] = []
        self.selection: dict | None = None

    @workflow.run
    async def run(
        self,
        db_path: str,
        experiment_ids: list[int],
        trial_activity: str,
        attempt: int = 1,
        objective: dict | None = None,
        selection_activity: str = "select_probe_results_activity",
    ) -> dict:
        """Execute child trial workflows for each queued experiment id."""
        self.phase = "probe"
        for experiment_id in experiment_ids:
            self.current_experiment_id = experiment_id
            try:
                result = await workflow.execute_child_workflow(
                    ResearchTrialWorkflow.run,
                    args=[db_path, experiment_id, trial_activity, attempt],
                    id=trial_workflow_id(experiment_id, attempt),
                    task_queue=workflow.info().task_queue,
                )
                self.results.append(
                    {
                        "experiment_id": experiment_id,
                        "status": "succeeded",
                        "result": result,
                    }
                )
            except Exception as exc:
                self.failed += 1
                self.results.append(
                    {
                        "experiment_id": experiment_id,
                        "status": "failed",
                        "failure": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )
            self.completed += 1
        self.current_experiment_id = None
        self.phase = "selecting"
        try:
            self.selection = await workflow.execute_activity(
                selection_activity,
                args=[db_path, experiment_ids, objective or {}],
                task_queue=workflow.info().task_queue,
                start_to_close_timeout=timedelta(minutes=5),
            )
        except Exception:
            self.phase = "selection_failed"
            raise
        self.phase = "complete_with_failures" if self.failed else "complete"
        return {
            "results": self.results,
            "completed": self.completed,
            "failed": self.failed,
            "selection": self.selection,
        }

    @workflow.query
    def status(self) -> dict:
        """Return the current probe workflow state."""
        return {
            "phase": self.phase,
            "completed": self.completed,
            "failed": self.failed,
            "current_experiment_id": self.current_experiment_id,
            "selection": self.selection,
        }


@workflow.defn
class ResearchCampaignDemoWorkflow:
    """Demonstrates Temporal loops, concurrent activities, signals, and queries."""

    def __init__(self) -> None:
        self.phase = "created"
        self.campaign_id: str | None = None
        self.round = 0
        self.batch_size = 2
        self.pause_after_each_round = False
        self.paused = False
        self.cancel_requested = False
        self.cancel_reason = ""
        self.pending: list[int] = []
        self.active: list[int] = []
        self.completed: list[int] = []
        self.round_summaries: list[dict] = []
        self.signals: list[dict] = []
        self.prepare_activity = "prepare_campaign_demo_activity"
        self.score_activity = "score_campaign_demo_activity"
        self.summarize_activity = "summarize_campaign_demo_round_activity"
        self.finalize_activity = "finalize_campaign_demo_activity"

    @workflow.run
    async def run(
        self,
        campaign_id: str,
        experiment_ids: list[int],
        config: dict | None = None,
    ) -> dict:
        """Run a signal-driven demonstration campaign in parallel batches."""
        self.campaign_id = campaign_id
        self.pending = list(experiment_ids)
        self._apply_config(config or {})

        if self.paused:
            self.phase = "paused"
        await self._wait_until_runnable()

        self.phase = "preparing"
        await workflow.execute_activity(
            self.prepare_activity,
            args=[campaign_id],
            task_queue=workflow.info().task_queue,
            start_to_close_timeout=DEFAULT_DEMO_ACTIVITY_TIMEOUT,
        )

        while self.pending and not self.cancel_requested:
            await self._wait_until_runnable()
            if self.cancel_requested:
                break

            self.round += 1
            self.phase = "scoring"
            batch = self._next_batch()
            self.active = list(batch)
            score_tasks = [
                workflow.execute_activity(
                    self.score_activity,
                    args=[campaign_id, self.round, experiment_id],
                    task_queue=workflow.info().task_queue,
                    start_to_close_timeout=DEFAULT_DEMO_ACTIVITY_TIMEOUT,
                )
                for experiment_id in batch
            ]
            score_results = await asyncio.gather(*score_tasks)
            self.completed.extend(batch)
            self.active = []

            self.phase = "summarizing"
            summary = await workflow.execute_activity(
                self.summarize_activity,
                args=[campaign_id, self.round, list(score_results)],
                task_queue=workflow.info().task_queue,
                start_to_close_timeout=DEFAULT_DEMO_ACTIVITY_TIMEOUT,
            )
            self.round_summaries.append(summary)
            if self.pause_after_each_round:
                self.paused = True
                self.phase = "paused"

        self.phase = "cancelling" if self.cancel_requested else "finalizing"
        finalize = await workflow.execute_activity(
            self.finalize_activity,
            args=[campaign_id, self.round_summaries, self.cancel_requested],
            task_queue=workflow.info().task_queue,
            start_to_close_timeout=DEFAULT_DEMO_ACTIVITY_TIMEOUT,
        )
        self.phase = "cancelled" if self.cancel_requested else "complete"
        return {
            "phase": self.phase,
            "campaign_id": campaign_id,
            "round": self.round,
            "completed": self.completed,
            "remaining": self.pending,
            "round_summaries": self.round_summaries,
            "cancel_reason": self.cancel_reason,
            "signals": self.signals,
            "finalize": finalize,
        }

    @workflow.signal
    async def pause(self) -> None:
        """Pause the campaign before the next activity batch starts."""
        self.paused = True
        if self.phase not in {"scoring", "summarizing", "finalizing"}:
            self.phase = "paused"
        self.signals.append({"name": "pause"})

    @workflow.signal
    async def resume(self) -> None:
        """Resume a paused campaign."""
        self.paused = False
        if self.phase == "paused":
            self.phase = "running"
        self.signals.append({"name": "resume"})

    @workflow.signal
    async def add_experiment(self, experiment_id: int) -> None:
        """Add an experiment id to the remaining campaign queue."""
        if experiment_id not in self.pending and experiment_id not in self.completed:
            self.pending.append(experiment_id)
        self.signals.append(
            {"name": "add_experiment", "experiment_id": experiment_id}
        )

    @workflow.signal
    async def set_batch_size(self, batch_size: int) -> None:
        """Change the maximum number of concurrent scoring activities."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size
        self.signals.append({"name": "set_batch_size", "batch_size": batch_size})

    @workflow.signal
    async def cancel_campaign(self, reason: str) -> None:
        """Request graceful cancellation after current awaited work finishes."""
        self.cancel_requested = True
        self.cancel_reason = reason
        self.paused = False
        self.signals.append({"name": "cancel_campaign", "reason": reason})

    @workflow.query
    def status(self) -> dict:
        """Return the signal-aware campaign state."""
        return {
            "phase": self.phase,
            "campaign_id": self.campaign_id,
            "round": self.round,
            "batch_size": self.batch_size,
            "paused": self.paused,
            "cancel_requested": self.cancel_requested,
            "cancel_reason": self.cancel_reason,
            "pending": self.pending,
            "active": self.active,
            "completed": self.completed,
            "round_summaries": self.round_summaries,
            "signals": self.signals,
        }

    def _apply_config(self, config: dict) -> None:
        self.batch_size = int(config.get("batch_size", self.batch_size))
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.paused = bool(config.get("initially_paused", self.paused))
        self.pause_after_each_round = bool(
            config.get("pause_after_each_round", self.pause_after_each_round)
        )
        self.prepare_activity = str(
            config.get("prepare_activity", self.prepare_activity)
        )
        self.score_activity = str(config.get("score_activity", self.score_activity))
        self.summarize_activity = str(
            config.get("summarize_activity", self.summarize_activity)
        )
        self.finalize_activity = str(
            config.get("finalize_activity", self.finalize_activity)
        )

    async def _wait_until_runnable(self) -> None:
        if self.paused:
            self.phase = "paused"
        await workflow.wait_condition(
            lambda: not self.paused or self.cancel_requested
        )

    def _next_batch(self) -> list[int]:
        batch = self.pending[: self.batch_size]
        self.pending = self.pending[self.batch_size :]
        return batch
