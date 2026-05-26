"""Temporal activities for executing research experiment trials."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import traceback
from collections.abc import Callable

import temporalio.activity

import research.artifacts
import research.db
import research.models
import research.preflight
import research.probes
import research.reports
import research.runners


ProgressCallback = Callable[[research.models.ProgressUpdate], None]


def _intent_from_row(row: dict[str, object]) -> research.models.Intent:
    return research.models.Intent(
        adapter=str(row["adapter"]),
        model=str(row["model"]),
        profile=str(row["profile"]),
        phase=str(row["phase"]),
        name=str(row["name"]),
        config=dict(row["config_json"]),
        objective=dict(row["objective_json"]),
        source=str(row["source"]),
    )


def _write_preflight(
    artifact_dir: pathlib.Path,
    preflight: research.models.PreflightResult,
) -> pathlib.Path:
    preflight_path = artifact_dir / "preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "ok": preflight.ok,
                "checks": preflight.checks,
                "message": preflight.message,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return preflight_path


def _failed_report(reason: str, message: str) -> research.models.TrialReport:
    return research.models.TrialReport(
        status="failed",
        metrics={},
        failure={"reason": reason, "message": message},
        summary=message,
    )


def _latest_heartbeat(
    progress: list[research.models.ProgressUpdate],
) -> research.models.JsonDict:
    if not progress:
        return {}
    return dataclasses.asdict(progress[-1])


def heartbeat_progress(progress: research.models.ProgressUpdate) -> None:
    """Send a parsed progress update as a Temporal activity heartbeat."""
    try:
        temporalio.activity.heartbeat(dataclasses.asdict(progress))
    except RuntimeError:
        return


@temporalio.activity.defn
def select_probe_results_activity(
    db_path_s: str,
    experiment_ids: list[int],
    objective: research.models.JsonDict,
) -> dict[str, object]:
    """Select the best completed probe intents and persist intent statuses."""
    result = research.probes.select_probe_results(
        pathlib.Path(db_path_s),
        experiment_ids,
        objective=objective,
    )
    return dataclasses.asdict(result)


@temporalio.activity.defn
def prepare_campaign_demo_activity(campaign_id: str) -> dict:
    """Prepare a demo campaign before parallel scoring starts."""
    return {
        "campaign_id": campaign_id,
        "prepared": True,
        "message": "campaign resources prepared",
    }


@temporalio.activity.defn
def score_campaign_demo_activity(
    campaign_id: str,
    round_index: int,
    experiment_id: int,
) -> dict:
    """Score one demo experiment in a parallel activity batch."""
    return {
        "campaign_id": campaign_id,
        "round": round_index,
        "experiment_id": experiment_id,
        "score": max(0, 100 - experiment_id),
    }


@temporalio.activity.defn
def summarize_campaign_demo_round_activity(
    campaign_id: str,
    round_index: int,
    results: list[dict],
) -> dict:
    """Summarize one completed demo campaign round."""
    best = max(results, key=lambda result: int(result["score"])) if results else {}
    return {
        "campaign_id": campaign_id,
        "round": round_index,
        "count": len(results),
        "best_experiment_id": best.get("experiment_id"),
        "best_score": best.get("score"),
    }


@temporalio.activity.defn
def finalize_campaign_demo_activity(
    campaign_id: str,
    round_summaries: list[dict],
    cancelled: bool,
) -> dict:
    """Finalize a completed or gracefully cancelled demo campaign."""
    return {
        "campaign_id": campaign_id,
        "rounds": len(round_summaries),
        "cancelled": cancelled,
        "summary": "campaign cancelled" if cancelled else "campaign complete",
    }


def run_trial_with_adapter(
    db_path_s: str,
    experiment_id: int,
    adapter: research.models.ExperimentAdapter,
    attempt: int = 1,
    worktree: pathlib.Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run one trial attempt with a concrete adapter and persist terminal state."""
    db_path = pathlib.Path(db_path_s)
    experiment = research.db.get_experiment(db_path, experiment_id)
    intent = _intent_from_row(
        research.db.get_intent(db_path, int(experiment["intent_id"]))
    )
    artifact_root = pathlib.Path(str(experiment["artifact_root"]))
    artifact_dir = research.artifacts.attempt_dir_from_subdir(
        artifact_root,
        artifact_subdir=str(experiment["artifact_subdir"]),
        attempt=attempt,
    )
    heartbeat: research.models.JsonDict = {}
    trial_run_id = research.db.create_trial_run(
        db_path,
        experiment_id=experiment_id,
        attempt=attempt,
    )

    research.db.transition_experiment(db_path, experiment_id, "running")
    try:
        context = research.models.TrialContext(
            experiment_id=experiment_id,
            trial_run_id=trial_run_id,
            attempt=attempt,
            worktree=worktree or pathlib.Path.cwd(),
            artifact_dir=artifact_dir,
            db_path=db_path,
        )
        preflight = research.preflight.run_preflight(adapter, intent, context)
        _write_preflight(artifact_dir, preflight)

        if not preflight.ok:
            report = research.models.TrialReport(
                status="failed",
                metrics={},
                failure={
                    "reason": "preflight_failed",
                    "checks": preflight.checks,
                },
                summary=preflight.message,
            )
        else:
            research.db.transition_trial_run(db_path, trial_run_id, "running")
            command = adapter.build_trial(intent, context)
            run_result = research.runners.run_trial_command(
                adapter,
                command,
                context,
                progress_callback,
            )
            heartbeat = _latest_heartbeat(run_result.progress)
            report = adapter.analyze_result(context)
            if run_result.returncode != 0 and report.status == "succeeded":
                report = research.models.TrialReport(
                    status="failed",
                    metrics=report.metrics,
                    failure={
                        "reason": "launcher_failed",
                        "returncode": run_result.returncode,
                    },
                    summary=f"trial command exited {run_result.returncode}",
                )
    except Exception as exc:
        report = _failed_report(
            "activity_exception",
            f"{type(exc).__name__}: {exc}",
        )

    report_path = ""
    try:
        paths = research.reports.write_report(artifact_dir, report)
        report_path = str(paths["markdown"])
    except Exception as exc:
        report = research.models.TrialReport(
            status="failed",
            metrics=report.metrics,
            failure={
                **report.failure,
                "report_write_failed": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
            summary=report.summary,
        )
    terminal = "succeeded" if report.status == "succeeded" else "failed"
    research.db.transition_trial_run(
        db_path,
        trial_run_id,
        terminal,
        heartbeat=heartbeat,
        metrics=report.metrics,
        failure=report.failure,
        report_path=report_path,
    )
    research.db.transition_experiment(db_path, experiment_id, terminal)
    return {
        "status": report.status,
        "metrics": report.metrics,
        "failure": report.failure,
    }
