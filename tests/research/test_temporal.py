"""Tests for local Temporal development helpers."""

from __future__ import annotations

import asyncio
import pathlib

from temporalio.common import WorkflowIDReusePolicy

from research import temporal, workflows


def fake_activity() -> None:
    """Fake activity callable for worker config tests."""


def test_default_temporal_db_path_uses_home(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("RESEARCH_TEMPORAL_DB", raising=False)

    assert temporal.default_temporal_db_path() == (
        home / ".automata" / "research" / "temporal" / "temporal.db"
    )


def test_build_start_dev_command_creates_parent(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    db = tmp_path / "temporal" / "temporal.db"
    monkeypatch.setenv("RESEARCH_TEMPORAL_DB", str(db))

    command = temporal.build_start_dev_command()

    assert command == [
        "temporal",
        "server",
        "start-dev",
        "--db-filename",
        str(db),
    ]
    assert db.parent.exists()


def test_build_worker_config_registers_generic_research_workflows() -> None:
    config = temporal.build_worker_config(
        address="temporal:7233",
        task_queue="experiments",
        activities=[fake_activity],
        activity_workers=3,
    )

    assert config.address == "temporal:7233"
    assert config.task_queue == "experiments"
    assert config.activity_workers == 3
    assert _names(config.activities) == [
        "fake_activity",
        "select_probe_results_activity",
        "prepare_campaign_demo_activity",
        "score_campaign_demo_activity",
        "summarize_campaign_demo_round_activity",
        "finalize_campaign_demo_activity",
    ]
    assert list(config.workflows) == [
        workflows.ResearchProbeWorkflow,
        workflows.ResearchTrialWorkflow,
        workflows.ResearchCampaignDemoWorkflow,
    ]


def test_build_worker_config_rejects_empty_activity_list() -> None:
    try:
        temporal.build_worker_config(activities=[])
    except ValueError as exc:
        assert "at least one activity" in str(exc)
    else:
        raise AssertionError("empty activity registration should fail")


def test_start_trial_workflow_rejects_duplicate_workflow_ids(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    calls = []

    class FakeClient:
        async def start_workflow(self, workflow: object, **kwargs: object) -> None:
            calls.append({"workflow": workflow, **kwargs})

    async def fake_connect(address: str) -> FakeClient:
        assert address == "temporal:7233"
        return FakeClient()

    monkeypatch.setattr(temporal.temporalio.client.Client, "connect", fake_connect)

    workflow_id = asyncio.run(
        temporal.start_trial_workflow(
            address="temporal:7233",
            task_queue="experiments",
            db_path=tmp_path / "research.sqlite",
            experiment_id=7,
            trial_activity="fake_activity",
            attempt=2,
        )
    )

    assert workflow_id == "research-trial-7-attempt-2"
    assert calls[0]["id_reuse_policy"] == WorkflowIDReusePolicy.REJECT_DUPLICATE
    assert calls[0]["args"] == [
        str(tmp_path / "research.sqlite"),
        7,
        "fake_activity",
        2,
    ]


def test_start_probe_workflow_submits_selection_inputs(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    calls = []

    class FakeClient:
        async def start_workflow(self, workflow: object, **kwargs: object) -> None:
            calls.append({"workflow": workflow, **kwargs})

    async def fake_connect(address: str) -> FakeClient:
        assert address == "temporal:7233"
        return FakeClient()

    monkeypatch.setattr(temporal.temporalio.client.Client, "connect", fake_connect)

    workflow_id = asyncio.run(
        temporal.start_probe_workflow(
            address="temporal:7233",
            task_queue="experiments",
            db_path=tmp_path / "research.sqlite",
            experiment_ids=[3, 4],
            trial_activity="fake_activity",
            workflow_id="probe-1",
            objective={"metric": "val_loss", "direction": "minimize"},
            selection_activity="select_probe_results_activity",
        )
    )

    assert workflow_id == "probe-1"
    assert calls[0]["id_reuse_policy"] == WorkflowIDReusePolicy.REJECT_DUPLICATE
    assert calls[0]["args"] == [
        str(tmp_path / "research.sqlite"),
        [3, 4],
        "fake_activity",
        1,
        {"metric": "val_loss", "direction": "minimize"},
        "select_probe_results_activity",
    ]


def test_start_campaign_demo_workflow_submits_signal_demo_inputs(
    monkeypatch,
) -> None:
    calls = []

    class FakeClient:
        async def start_workflow(self, workflow: object, **kwargs: object) -> None:
            calls.append({"workflow": workflow, **kwargs})

    async def fake_connect(address: str) -> FakeClient:
        assert address == "temporal:7233"
        return FakeClient()

    monkeypatch.setattr(temporal.temporalio.client.Client, "connect", fake_connect)

    workflow_id = asyncio.run(
        temporal.start_campaign_demo_workflow(
            address="temporal:7233",
            task_queue="experiments",
            campaign_id="demo-1",
            experiment_ids=[1, 2, 3],
            workflow_id="campaign-demo-1",
            config={"batch_size": 3, "initially_paused": True},
        )
    )

    assert workflow_id == "campaign-demo-1"
    assert calls[0]["id_reuse_policy"] == WorkflowIDReusePolicy.REJECT_DUPLICATE
    assert calls[0]["args"] == [
        "demo-1",
        [1, 2, 3],
        {"batch_size": 3, "initially_paused": True},
    ]


def test_wait_for_workflow_result_uses_existing_handle(monkeypatch) -> None:
    calls = []

    class FakeHandle:
        async def result(self) -> dict[str, object]:
            return {"selection": {"selected_intent_ids": [2]}}

    class FakeClient:
        def get_workflow_handle(self, workflow_id: str) -> FakeHandle:
            calls.append(workflow_id)
            return FakeHandle()

    async def fake_connect(address: str) -> FakeClient:
        assert address == "temporal:7233"
        return FakeClient()

    monkeypatch.setattr(temporal.temporalio.client.Client, "connect", fake_connect)

    result = asyncio.run(
        temporal.wait_for_workflow_result(
            address="temporal:7233",
            workflow_id="probe-1",
            timeout_seconds=5,
        )
    )

    assert calls == ["probe-1"]
    assert result == {"selection": {"selected_intent_ids": [2]}}


def test_trial_workflow_id_includes_attempt() -> None:
    assert workflows.trial_workflow_id(7, 2) == "research-trial-7-attempt-2"


def _names(objects: object) -> list[str]:
    return [item.__name__ for item in objects]
