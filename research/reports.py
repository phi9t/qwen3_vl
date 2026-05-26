"""Report rendering helpers for research trials."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sqlite3

import research.db
import research.models


def write_report(
    artifact_dir: pathlib.Path,
    report: research.models.TrialReport,
) -> dict[str, pathlib.Path]:
    """Write JSON and Markdown report artifacts."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifact_dir / "report.json"
    markdown_path = artifact_dir / "report.md"

    json_path.write_text(
        json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        "# Trial Report\n\n"
        f"Status: `{report.status}`\n\n"
        "## Summary\n\n"
        f"{report.summary or 'No summary.'}\n\n"
        "## Metrics\n\n"
        f"```json\n{json.dumps(report.metrics, indent=2, sort_keys=True)}\n```\n\n"
        "## Failure\n\n"
        f"```json\n{json.dumps(report.failure, indent=2, sort_keys=True)}\n```\n",
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}


def render_selected_intents_report(db_path: pathlib.Path) -> str:
    """Render a Markdown report of selected experiment intents."""
    rows = _selected_intent_rows(db_path)
    if not rows:
        return "# Selected Intents\n\nNo selected intents.\n"

    sections = ["# Selected Intents", ""]
    for row in rows:
        score = row["score_json"]
        config = row["config_json"]
        sections.extend(
            [
                f"## Intent {row['id']}: {row['name']}",
                "",
                f"- adapter: `{row['adapter']}`",
                f"- model: `{row['model']}`",
                f"- profile: `{row['profile']}`",
                f"- score: `{json.dumps(score, sort_keys=True)}`",
                "",
                "```json",
                json.dumps(config, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(sections)


def render_experiment_report(db_path: pathlib.Path, experiment_id: int) -> str:
    """Render a Markdown report for an experiment and its latest trial."""
    experiment = research.db.get_experiment(db_path, experiment_id)
    intent = research.db.get_intent(db_path, int(experiment["intent_id"]))
    trial_run = research.db.get_latest_trial_run_for_experiment(db_path, experiment_id)
    sections = [
        f"# Experiment {experiment_id}",
        "",
        f"- Intent: {intent['name']}",
        f"- Adapter: `{experiment['adapter']}`",
        f"- Experiment status: `{experiment['status']}`",
        f"- Workflow: `{experiment['temporal_workflow_id']}`",
        f"- Artifact root: `{experiment['artifact_root']}`",
        f"- Artifact subdir: `{experiment['artifact_subdir']}`",
        "",
        "## Intent Config",
        "",
        "```json",
        json.dumps(intent["config_json"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    if trial_run is None:
        sections.extend(["## Latest Trial", "", "No trial runs.", ""])
    else:
        sections.extend(
            [
                "## Latest Trial",
                "",
                f"- Attempt: `{trial_run['attempt']}`",
                f"- Status: {trial_run['status']}",
                f"- Report: `{trial_run['report_path']}`",
                "",
                "### Metrics",
                "",
                "```json",
                json.dumps(trial_run["metrics_json"], indent=2, sort_keys=True),
                "```",
                "",
                "### Failure",
                "",
                "```json",
                json.dumps(trial_run["failure_json"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(sections)


def _selected_intent_rows(db_path: pathlib.Path) -> list[research.db.JsonDict]:
    with research.db.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM intents
            WHERE status = 'selected'
            ORDER BY created_at, id
            """
        ).fetchall()
    return [_decode_intent_row(row) for row in rows]


def _decode_intent_row(row: sqlite3.Row) -> research.db.JsonDict:
    result = dict(row)
    for key in ("config_json", "objective_json", "score_json"):
        result[key] = json.loads(result[key] or "{}")
    return result
