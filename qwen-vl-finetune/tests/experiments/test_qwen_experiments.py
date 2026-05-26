"""Tests for the Qwen-VL experiment CLI."""

from __future__ import annotations

import contextlib
import pathlib

import experiments.qwen_experiments

import research.db
import research.workflows


def test_qwen_probe_cli_creates_candidate_experiments(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    """Qwen probe CLI should create experiments and start a Temporal workflow."""
    db_path = tmp_path / "research.sqlite"
    artifact_root = tmp_path / "artifacts"
    started = {}

    async def fake_start_probe_workflow(**kwargs: object) -> str:
        started.update(kwargs)
        return "wf-probe"

    monkeypatch.setattr(
        experiments.qwen_experiments.research.temporal,
        "start_probe_workflow",
        fake_start_probe_workflow,
    )

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
            "--workflow-id",
            "wf-probe",
        ]
    )

    assert rc == 0
    with contextlib.closing(research.db.connect(db_path)) as conn:
        intent_count = conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]
        experiment = conn.execute("SELECT * FROM experiments ORDER BY id").fetchone()

    assert intent_count == 24
    assert started["address"] == "localhost:7233"
    assert started["task_queue"] == "research-local"
    assert started["db_path"] == db_path
    assert started["experiment_ids"] == list(range(1, 25))
    assert started["trial_activity"] == "qwen_run_trial_activity"
    assert started["workflow_id"] == "wf-probe"
    assert started["objective"] == {
        "metric": "val_loss",
        "direction": "minimize",
        "top_k": 1,
    }
    assert experiment["adapter"] == "qwen_vl"
    assert experiment["status"] == "submitted"
    assert experiment["temporal_workflow_id"] == (
        research.workflows.trial_workflow_id(1, 1)
    )
    assert experiment["artifact_root"] == str(artifact_root)
    assert experiment["artifact_subdir"] == "qwen_vl/1"


def test_qwen_probe_plan_only_does_not_start_temporal(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    """Plan-only probe mode should only populate the SQLite database."""
    db_path = tmp_path / "research.sqlite"
    artifact_root = tmp_path / "artifacts"
    started = False

    async def fake_start_probe_workflow(**kwargs: object) -> str:
        nonlocal started
        started = True
        return "wf-probe"

    monkeypatch.setattr(
        experiments.qwen_experiments.research.temporal,
        "start_probe_workflow",
        fake_start_probe_workflow,
    )

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
            "--plan-only",
        ]
    )

    assert rc == 0
    assert not started


def test_qwen_probe_dry_run_marks_intents_for_no_gpu_launcher(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    """Dry-run probe mode should create Temporal-submitted dry-run intents."""
    db_path = tmp_path / "research.sqlite"

    async def fake_start_probe_workflow(**kwargs: object) -> str:
        return "wf-probe"

    monkeypatch.setattr(
        experiments.qwen_experiments.research.temporal,
        "start_probe_workflow",
        fake_start_probe_workflow,
    )

    rc = experiments.qwen_experiments.main(
        [
            "--db",
            str(db_path),
            "probe",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--profile",
            "b200",
            "--dry-run",
        ]
    )

    assert rc == 0
    with contextlib.closing(research.db.connect(db_path)) as conn:
        config = conn.execute(
            "SELECT config_json FROM intents ORDER BY id LIMIT 1"
        ).fetchone()["config_json"]

    assert '"QWEN_RESEARCH_DRY_RUN": "1"' in config


def test_qwen_probe_max_experiments_limits_submitted_campaign(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    """Probe budget should cap both persisted and submitted experiments."""
    db_path = tmp_path / "research.sqlite"
    started = {}

    async def fake_start_probe_workflow(**kwargs: object) -> str:
        started.update(kwargs)
        return "wf-probe"

    monkeypatch.setattr(
        experiments.qwen_experiments.research.temporal,
        "start_probe_workflow",
        fake_start_probe_workflow,
    )

    rc = experiments.qwen_experiments.main(
        [
            "--db",
            str(db_path),
            "probe",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--profile",
            "b200",
            "--max-experiments",
            "2",
        ]
    )

    assert rc == 0
    with contextlib.closing(research.db.connect(db_path)) as conn:
        intent_count = conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]
        experiment_count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[
            0
        ]

    assert intent_count == 2
    assert experiment_count == 2
    assert started["experiment_ids"] == [1, 2]


def test_qwen_probe_waits_for_temporal_completion(
    monkeypatch,
    tmp_path: pathlib.Path,
    capsys,
) -> None:
    """Wait mode should block for workflow completion and print selection."""
    db_path = tmp_path / "research.sqlite"
    waited = {}

    async def fake_start_probe_workflow(**kwargs: object) -> str:
        return "wf-probe"

    async def fake_wait_for_workflow_result(**kwargs: object) -> dict[str, object]:
        waited.update(kwargs)
        return {"selection": {"selected_intent_ids": [1]}}

    monkeypatch.setattr(
        experiments.qwen_experiments.research.temporal,
        "start_probe_workflow",
        fake_start_probe_workflow,
    )
    monkeypatch.setattr(
        experiments.qwen_experiments.research.temporal,
        "wait_for_workflow_result",
        fake_wait_for_workflow_result,
    )

    rc = experiments.qwen_experiments.main(
        [
            "--db",
            str(db_path),
            "probe",
            "--model",
            "Qwen/Qwen3-VL-8B-Instruct",
            "--profile",
            "b200",
            "--workflow-id",
            "wf-probe",
            "--max-experiments",
            "1",
            "--wait",
            "--timeout-seconds",
            "30",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert waited == {
        "address": "localhost:7233",
        "workflow_id": "wf-probe",
        "timeout_seconds": 30.0,
    }
    assert "Workflow wf-probe completed" in out
    assert "selected_intent_ids" in out


def test_qwen_probe_marks_submission_failed_when_temporal_start_fails(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    """Failed Temporal submission should leave durable failed-submission state."""
    db_path = tmp_path / "research.sqlite"

    async def fail_start_probe_workflow(**kwargs: object) -> str:
        raise RuntimeError("temporal unavailable")

    monkeypatch.setattr(
        experiments.qwen_experiments.research.temporal,
        "start_probe_workflow",
        fail_start_probe_workflow,
    )

    try:
        experiments.qwen_experiments.main(
            [
                "--db",
                str(db_path),
                "probe",
                "--model",
                "Qwen/Qwen3-VL-8B-Instruct",
                "--profile",
                "b200",
            ]
        )
    except RuntimeError as exc:
        assert "temporal unavailable" in str(exc)
    else:
        raise AssertionError("Temporal submission failure should propagate")

    with contextlib.closing(research.db.connect(db_path)) as conn:
        statuses = [
            row["status"]
            for row in conn.execute("SELECT status FROM experiments ORDER BY id")
        ]

    assert statuses == ["submission_failed"] * 24


def test_qwen_worker_config_uses_research_workflows_and_qwen_activity() -> None:
    """Qwen manager should register generic workflows with the Qwen activity."""
    config = experiments.qwen_experiments.build_worker_config(
        address="localhost:7233",
        task_queue="research-local",
        activity_workers=2,
    )

    assert config.address == "localhost:7233"
    assert config.task_queue == "research-local"
    assert config.activity_workers == 2
    assert _names(config.workflows) == [
        "ResearchProbeWorkflow",
        "ResearchTrialWorkflow",
        "ResearchCampaignDemoWorkflow",
    ]
    assert _names(config.activities) == [
        "qwen_run_trial_activity",
        "select_probe_results_activity",
        "prepare_campaign_demo_activity",
        "score_campaign_demo_activity",
        "summarize_campaign_demo_round_activity",
        "finalize_campaign_demo_activity",
    ]


def _names(objects: object) -> list[str]:
    return [item.__name__ for item in objects]
