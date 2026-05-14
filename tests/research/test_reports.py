"""Tests for research report rendering."""

from __future__ import annotations

import json
import pathlib

import research.models
import research.reports


def test_write_report_json_and_markdown(tmp_path: pathlib.Path) -> None:
    report = research.models.TrialReport(
        status="succeeded",
        metrics={"val_loss": 0.5},
        failure={},
        summary="trial succeeded",
    )

    paths = research.reports.write_report(tmp_path, report)
    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert json.loads(paths["json"].read_text(encoding="utf-8"))["status"] == (
        "succeeded"
    )
    assert "# Trial Report" in markdown
    assert "trial succeeded" in markdown
