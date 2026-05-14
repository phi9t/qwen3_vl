"""Tests for repository-local mise research tasks."""

from __future__ import annotations

import pathlib

import tomli


def test_mise_exposes_research_tasks() -> None:
    """Verify mise exposes thin wrappers around the research CLI."""
    data = tomli.loads(pathlib.Path("mise.toml").read_text(encoding="utf-8"))

    assert data["tasks"]["research-db-init"]["run"] == "uv run research db init"
    assert data["tasks"]["research-manager"]["run"] == "uv run research manager"
    assert "uv run research probe" in data["tasks"]["research-probe"]["run"]
    assert data["tasks"]["qwen-probe-b200"]["run"].startswith(
        "uv run research probe --adapter qwen_vl"
    )
