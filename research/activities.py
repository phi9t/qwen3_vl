"""Temporal activities for executing research experiment trials."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import traceback

import research.artifacts
import research.db
import research.models
import research.preflight
import research.reports
import research.runners


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


def run_trial_with_adapter(
    db_path_s: str,
    experiment_id: int,
    adapter: research.models.ExperimentAdapter,
    attempt: int = 1,
    worktree: pathlib.Path | None = None,
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
            run_result = research.runners.run_trial_command(adapter, command, context)
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
