"""Report rendering helpers for research trials."""

from __future__ import annotations

import dataclasses
import json
import pathlib

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
