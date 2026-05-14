"""Tests for the Qwen-VL experiment CLI."""

from __future__ import annotations

import contextlib
import pathlib

import experiments.qwen_experiments

import research.db


def test_qwen_probe_cli_creates_candidate_experiments(
    tmp_path: pathlib.Path,
) -> None:
    """Qwen probe CLI should call generic research helpers with Qwen adapter."""
    db_path = tmp_path / "research.sqlite"
    artifact_root = tmp_path / "artifacts"

    rc = experiments.qwen_experiments.main(
        [
            "--db",
            str(db_path),
            "probe",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--profile",
            "b200",
            "--artifact-root",
            str(artifact_root),
        ]
    )

    assert rc == 0
    with contextlib.closing(research.db.connect(db_path)) as conn:
        intent_count = conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]
        experiment = conn.execute("SELECT * FROM experiments ORDER BY id").fetchone()

    assert intent_count == 24
    assert experiment["adapter"] == "qwen_vl"
    assert experiment["status"] == "queued"
    assert experiment["artifact_root"] == str(artifact_root)
    assert experiment["artifact_subdir"] == "qwen_vl/1"


def test_qwen_worker_config_uses_qwen_activity_and_workflows() -> None:
    """Qwen manager should register Qwen-specific Temporal objects."""
    config = experiments.qwen_experiments.build_worker_config(
        address="localhost:7233",
        task_queue="research-local",
    )

    assert config["address"] == "localhost:7233"
    assert config["task_queue"] == "research-local"
    assert _names(config["workflows"]) == ["QwenProbeWorkflow", "QwenTrialWorkflow"]
    assert _names(config["activities"]) == ["qwen_run_trial_activity"]


def _names(objects: object) -> list[str]:
    return [item.__name__ for item in objects]
