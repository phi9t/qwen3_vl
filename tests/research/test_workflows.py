"""Integration tests for generic Temporal research workflows."""

from __future__ import annotations

import asyncio
import concurrent.futures
import pathlib
import time
import uuid

from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

import research.activities
import research.db
import research.models
import research.workflows


_TRIAL_ACTIVITY_ORDER: list[int] = []
_DEMO_ACTIVITY_EVENTS: list[dict[str, object]] = []


@activity.defn(name="fake_sequential_trial_activity")
def fake_sequential_trial_activity(
    db_path_s: str,
    experiment_id: int,
    attempt: int = 1,
) -> dict[str, object]:
    """Persist deterministic fake metrics for a Temporal trial child."""
    db_path = pathlib.Path(db_path_s)
    _TRIAL_ACTIVITY_ORDER.append(experiment_id)
    trial_run_id = research.db.create_trial_run(
        db_path,
        experiment_id=experiment_id,
        attempt=attempt,
    )
    metric = {1: 0.7, 2: 0.2, 3: 0.5}[experiment_id]
    research.db.transition_experiment(db_path, experiment_id, "running")
    research.db.transition_trial_run(
        db_path,
        trial_run_id,
        "succeeded",
        metrics={"val_loss": metric},
    )
    research.db.transition_experiment(db_path, experiment_id, "succeeded")
    return {"status": "succeeded", "metrics": {"val_loss": metric}, "failure": {}}


def test_probe_workflow_runs_trials_sequentially_and_selects_best(
    tmp_path: pathlib.Path,
) -> None:
    """A real Temporal worker should run probe children in order and select."""
    asyncio.run(_run_probe_workflow_smoke(tmp_path))


async def _run_probe_workflow_smoke(tmp_path: pathlib.Path) -> None:
    _TRIAL_ACTIVITY_ORDER.clear()
    db_path = tmp_path / "research.sqlite"
    experiment_ids = _create_probe_experiments(db_path, tmp_path / "artifacts")
    objective = {"metric": "val_loss", "direction": "minimize", "top_k": 1}
    task_queue = f"research-test-{uuid.uuid4()}"

    async with await WorkflowEnvironment.start_local() as env:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[
                    research.workflows.ResearchProbeWorkflow,
                    research.workflows.ResearchTrialWorkflow,
                ],
                activities=[
                    fake_sequential_trial_activity,
                    research.activities.select_probe_results_activity,
                ],
                activity_executor=executor,
            ):
                handle = await env.client.start_workflow(
                    research.workflows.ResearchProbeWorkflow.run,
                    args=[
                        str(db_path),
                        experiment_ids,
                        "fake_sequential_trial_activity",
                        1,
                        objective,
                        "select_probe_results_activity",
                    ],
                    id=f"probe-{uuid.uuid4()}",
                    task_queue=task_queue,
                )
                try:
                    result = await asyncio.wait_for(handle.result(), timeout=10)
                except TimeoutError as exc:
                    status = await handle.query(
                        research.workflows.ResearchProbeWorkflow.status
                    )
                    raise AssertionError(f"workflow timed out at {status}") from exc

    assert experiment_ids == _TRIAL_ACTIVITY_ORDER
    assert result["completed"] == 3
    assert result["failed"] == 0
    assert result["selection"]["selected_intent_ids"] == [2]
    assert result["selection"]["rejected_intent_ids"] == [1, 3]
    assert research.db.get_intent(db_path, 2)["status"] == "selected"
    assert research.db.get_intent(db_path, 1)["status"] == "rejected"
    assert research.db.get_intent(db_path, 3)["status"] == "rejected"


@activity.defn(name="demo_prepare_campaign_activity")
def demo_prepare_campaign_activity(campaign_id: str) -> dict:
    """Fake setup activity for the complex Temporal demo workflow."""
    _DEMO_ACTIVITY_EVENTS.append(
        {"activity": "prepare", "campaign_id": campaign_id}
    )
    return {"campaign_id": campaign_id, "prepared": True}


@activity.defn(name="demo_score_experiment_activity")
def demo_score_experiment_activity(
    campaign_id: str,
    round_index: int,
    experiment_id: int,
) -> dict:
    """Fake per-experiment activity with a stable, inspectable event log."""
    time.sleep(0.05)
    _DEMO_ACTIVITY_EVENTS.append(
        {
            "activity": "score",
            "campaign_id": campaign_id,
            "round": round_index,
            "experiment_id": experiment_id,
        }
    )
    return {
        "experiment_id": experiment_id,
        "round": round_index,
        "score": 100 - experiment_id,
    }


@activity.defn(name="demo_summarize_round_activity")
def demo_summarize_round_activity(
    campaign_id: str,
    round_index: int,
    results: list[dict],
) -> dict:
    """Fake round aggregation activity."""
    _DEMO_ACTIVITY_EVENTS.append(
        {
            "activity": "summarize",
            "campaign_id": campaign_id,
            "round": round_index,
            "count": len(results),
        }
    )
    best = min(results, key=lambda result: int(result["experiment_id"]))
    return {
        "round": round_index,
        "count": len(results),
        "best_experiment_id": best["experiment_id"],
    }


@activity.defn(name="demo_finalize_campaign_activity")
def demo_finalize_campaign_activity(
    campaign_id: str,
    round_summaries: list[dict],
    cancelled: bool,
) -> dict:
    """Fake finalization activity for completed or cancelled campaigns."""
    _DEMO_ACTIVITY_EVENTS.append(
        {
            "activity": "finalize",
            "campaign_id": campaign_id,
            "cancelled": cancelled,
        }
    )
    return {
        "campaign_id": campaign_id,
        "rounds": len(round_summaries),
        "cancelled": cancelled,
    }


def test_campaign_demo_workflow_handles_parallel_batches_and_signals() -> None:
    """The demo workflow should exercise loops, async awaits, and signals."""
    asyncio.run(_run_campaign_demo_signal_smoke())


async def _run_campaign_demo_signal_smoke() -> None:
    _DEMO_ACTIVITY_EVENTS.clear()
    task_queue = f"research-demo-{uuid.uuid4()}"

    async with await WorkflowEnvironment.start_local() as env:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            async with Worker(
                env.client,
                task_queue=task_queue,
                workflows=[research.workflows.ResearchCampaignDemoWorkflow],
                activities=[
                    demo_prepare_campaign_activity,
                    demo_score_experiment_activity,
                    demo_summarize_round_activity,
                    demo_finalize_campaign_activity,
                ],
                activity_executor=executor,
            ):
                handle = await env.client.start_workflow(
                    research.workflows.ResearchCampaignDemoWorkflow.run,
                    args=[
                        "campaign-1",
                        [3, 1, 2],
                        {
                            "batch_size": 2,
                            "initially_paused": True,
                            "pause_after_each_round": True,
                            "prepare_activity": "demo_prepare_campaign_activity",
                            "score_activity": "demo_score_experiment_activity",
                            "summarize_activity": "demo_summarize_round_activity",
                            "finalize_activity": "demo_finalize_campaign_activity",
                        },
                    ],
                    id=f"campaign-demo-{uuid.uuid4()}",
                    task_queue=task_queue,
                )

                paused = await handle.query(
                    research.workflows.ResearchCampaignDemoWorkflow.status
                )
                assert paused["phase"] == "paused"

                await handle.signal(
                    research.workflows.ResearchCampaignDemoWorkflow.add_experiment,
                    4,
                )
                await handle.signal(
                    research.workflows.ResearchCampaignDemoWorkflow.set_batch_size,
                    3,
                )
                await handle.signal(
                    research.workflows.ResearchCampaignDemoWorkflow.resume
                )

                await _wait_for_demo_round(handle, expected_round=1)
                await handle.signal(
                    research.workflows.ResearchCampaignDemoWorkflow.pause
                )
                await handle.signal(
                    research.workflows.ResearchCampaignDemoWorkflow.cancel_campaign,
                    "operator stopped after first round",
                )
                result = await asyncio.wait_for(handle.result(), timeout=10)

    score_events = [
        event for event in _DEMO_ACTIVITY_EVENTS if event["activity"] == "score"
    ]
    first_round_scores = [
        event for event in score_events if event["round"] == 1
    ]
    assert len(first_round_scores) == 3
    assert result["phase"] == "cancelled"
    assert result["cancel_reason"] == "operator stopped after first round"
    assert result["completed"] == [3, 1, 2]
    assert result["remaining"] == [4]
    assert result["signals"] == [
        {"name": "add_experiment", "experiment_id": 4},
        {"name": "set_batch_size", "batch_size": 3},
        {"name": "resume"},
        {"name": "pause"},
        {
            "name": "cancel_campaign",
            "reason": "operator stopped after first round",
        },
    ]
    assert result["finalize"]["cancelled"] is True


async def _wait_for_demo_round(handle: object, expected_round: int) -> None:
    for _ in range(100):
        status = await handle.query(
            research.workflows.ResearchCampaignDemoWorkflow.status
        )
        if status["round"] >= expected_round:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"workflow did not reach round {expected_round}: {status}")


def _create_probe_experiments(
    db_path: pathlib.Path,
    artifact_root: pathlib.Path,
) -> list[int]:
    research.db.init_db(db_path)
    experiment_ids = []
    for name in ("slow", "best", "middle"):
        intent_id = research.db.insert_intent(
            db_path,
            research.models.Intent(
                "fake",
                "unit-model",
                "cpu",
                "probe",
                name,
                {},
            ),
        )
        experiment_ids.append(
            research.db.create_experiment(
                db_path,
                intent_id=intent_id,
                adapter="fake",
                artifact_root=str(artifact_root),
                artifact_subdir=f"fake/{intent_id}",
            )
        )
    return experiment_ids
