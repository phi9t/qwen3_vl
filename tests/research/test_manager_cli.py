"""Tests for the research Temporal manager CLI helpers."""

from __future__ import annotations

import research.cli


def test_build_worker_config_uses_research_task_queue() -> None:
    """Worker config should list the research workflows and activities."""
    config = research.cli.build_worker_config(
        address="localhost:7233",
        task_queue="research-local",
    )

    assert config["address"] == "localhost:7233"
    assert config["task_queue"] == "research-local"
    assert _names(config["workflows"]) == ["ProbeWorkflow", "TrialWorkflow"]
    assert _names(config["activities"]) == ["run_trial_activity"]


def _names(objects: object) -> list[str]:
    return [item.__name__ for item in objects]
