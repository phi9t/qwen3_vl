from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy


with workflow.unsafe.imports_passed_through():
    from research.activities import (
        load_profile_activity,
        plan_probe_trials_activity,
        plan_sweep_trials_activity,
        run_trial_activity,
        select_capacity_activity,
        select_capacity_from_results_activity,
        summarize_campaign_activity,
    )


@workflow.defn
class SingleTrialWorkflow:
    def __init__(self) -> None:
        self.phase = "created"
        self.active_trial = ""
        self.completed_trials = 0
        self.latest_metrics: dict | None = None
        self.latest_analysis: dict | None = None

    @workflow.run
    async def run(self, spec: dict, dry_run: bool = False) -> dict:
        self.phase = "trial"
        self.active_trial = spec["trial"]
        analysis = await workflow.execute_activity(
            run_trial_activity,
            args=[spec, dry_run, True],
            start_to_close_timeout=timedelta(hours=6),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        self.completed_trials = 1
        self.latest_analysis = analysis
        self.latest_metrics = analysis.get("metrics")
        self.phase = "complete"
        return {
            "spec": spec,
            "analysis": analysis,
            "metrics": self.latest_metrics,
            "phase": self.phase,
        }

    @workflow.query
    def status(self) -> dict:
        return {
            "phase": self.phase,
            "active_trial": self.active_trial,
            "completed_trials": self.completed_trials,
            "latest_metrics": self.latest_metrics,
            "latest_analysis": self.latest_analysis,
        }


@workflow.defn
class ProfiledExperimentWorkflow:
    def __init__(self) -> None:
        self.phase = "created"
        self.active_trial = ""
        self.completed_trials = 0
        self.latest_metrics: dict | None = None
        self.latest_analysis: dict | None = None
        self.selected_capacity: dict | None = None
        self.pause_after_current = False
        self.stop_after_phase = False
        self.skip_trials: set[str] = set()
        self.cancel_requested = False
        self.rerun_trials: list[str] = []

    async def _run_trial_specs(self, specs: list[dict], dry_run: bool) -> list[dict]:
        rows: list[dict] = []
        for spec in specs:
            if self.stop_after_phase:
                break
            if spec["trial"] in self.skip_trials:
                continue
            self.active_trial = spec["trial"]
            analysis = await workflow.execute_activity(
                run_trial_activity,
                args=[spec, dry_run],
                start_to_close_timeout=timedelta(hours=6),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            self.completed_trials += 1
            self.latest_analysis = analysis
            self.latest_metrics = analysis.get("metrics")
            rows.append({"spec": spec, "analysis": analysis, "metrics": self.latest_metrics})
            if self.pause_after_current:
                break
        return rows

    @workflow.run
    async def run(self, profile: str = "b200", dry_run: bool = False, command: str = "full") -> dict:
        if command == "summarize":
            self.phase = "summarize"
            summary = await workflow.execute_activity(
                summarize_campaign_activity,
                profile,
                start_to_close_timeout=timedelta(minutes=2),
            )
            self.phase = "complete"
            return {"summary": summary}

        self.phase = "load_profile"
        await workflow.execute_activity(
            load_profile_activity,
            profile,
            start_to_close_timeout=timedelta(seconds=30),
        )
        if command == "select":
            self.phase = "select"
            self.selected_capacity = await workflow.execute_activity(
                select_capacity_from_results_activity,
                profile,
                start_to_close_timeout=timedelta(minutes=2),
            )
            return {"selected_capacity": self.selected_capacity}

        if command == "sweep":
            self.phase = "sweep"
            sweep_specs = await workflow.execute_activity(
                plan_sweep_trials_activity,
                args=[profile, None],
                start_to_close_timeout=timedelta(seconds=30),
            )
            sweep_rows = await self._run_trial_specs(sweep_specs, dry_run)
            self.phase = "complete"
            return {"sweep_rows": sweep_rows}

        self.phase = "probe"
        specs = await workflow.execute_activity(
            plan_probe_trials_activity,
            profile,
            start_to_close_timeout=timedelta(seconds=30),
        )

        rows = await self._run_trial_specs(specs, dry_run)
        if command == "probe" or self.pause_after_current or self.stop_after_phase:
            self.phase = "paused" if self.pause_after_current else "stopped_after_probe"
            return {"probe_rows": rows, "phase": self.phase}

        self.phase = "select"
        self.selected_capacity = await workflow.execute_activity(
            select_capacity_activity,
            args=[profile, rows],
            start_to_close_timeout=timedelta(minutes=2),
        )
        if command == "select":
            return {"selected_capacity": self.selected_capacity}

        self.phase = "sweep"
        sweep_specs = await workflow.execute_activity(
            plan_sweep_trials_activity,
            args=[profile, self.selected_capacity["selected"]["env"]],
            start_to_close_timeout=timedelta(seconds=30),
        )
        sweep_rows = await self._run_trial_specs(sweep_specs, dry_run)
        if command == "sweep" or self.pause_after_current or self.stop_after_phase:
            self.phase = "paused" if self.pause_after_current else "stopped_after_sweep"
            return {"selected_capacity": self.selected_capacity, "sweep_rows": sweep_rows, "phase": self.phase}

        self.phase = "summarize"
        summary = await workflow.execute_activity(
            summarize_campaign_activity,
            profile,
            start_to_close_timeout=timedelta(minutes=2),
        )
        self.phase = "complete"
        return {"selected_capacity": self.selected_capacity, "sweep_rows": sweep_rows, "summary": summary}

    @workflow.query
    def status(self) -> dict:
        return {
            "phase": self.phase,
            "active_trial": self.active_trial,
            "completed_trials": self.completed_trials,
            "latest_metrics": self.latest_metrics,
            "latest_analysis": self.latest_analysis,
            "selected_capacity": self.selected_capacity,
            "cancel_requested": self.cancel_requested,
            "rerun_trials": list(self.rerun_trials),
        }

    @workflow.signal
    def pause_after_current_trial(self) -> None:
        self.pause_after_current = True

    @workflow.signal
    def stop_after_current_phase(self) -> None:
        self.stop_after_phase = True

    @workflow.signal
    def skip_trial(self, trial: str) -> None:
        self.skip_trials.add(trial)

    @workflow.signal
    def cancel_campaign(self) -> None:
        self.cancel_requested = True
        self.stop_after_phase = True

    @workflow.signal
    def rerun_trial(self, trial: str) -> None:
        self.rerun_trials.append(trial)
