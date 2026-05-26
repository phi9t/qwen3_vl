"""Tests for repository-local mise research tasks."""

from __future__ import annotations

import pathlib

import tomli


def test_mise_exposes_research_tasks() -> None:
    """Verify mise exposes thin wrappers around the research CLI."""
    data = tomli.loads(pathlib.Path("mise.toml").read_text(encoding="utf-8"))

    assert data["tasks"]["research-db-init"]["run"] == "uv run research db init"
    assert data["tasks"]["research-temporal-start"]["run"] == (
        "uv run research temporal start-dev"
    )
    assert "research-manager" not in data["tasks"]
    assert "research-probe" not in data["tasks"]
    assert data["tasks"]["qwen-probe-b200"]["run"] == (
        "PYTHONPATH=.:qwen-vl-finetune uv run python -m "
        "experiments.qwen_experiments probe --model Qwen/Qwen3-VL-8B-Instruct "
        "--profile b200"
    )
    assert data["tasks"]["qwen-probe-b200-dry-run"]["run"] == (
        "PYTHONPATH=.:qwen-vl-finetune uv run python -m "
        "experiments.qwen_experiments probe --model Qwen/Qwen3-VL-8B-Instruct "
        "--profile b200 --dry-run --max-experiments 2"
    )
    assert data["tasks"]["qwen-manager"]["run"] == (
        "PYTHONPATH=.:qwen-vl-finetune uv run python -m "
        "experiments.qwen_experiments manager"
    )
