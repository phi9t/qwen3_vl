"""Local Temporal development server helpers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import os
import pathlib
import shutil
import uuid
from collections.abc import Callable, Sequence
from typing import Any

import temporalio.client
import temporalio.worker
from temporalio.common import WorkflowIDReusePolicy

import research.activities
import research.workflows


TEMPORAL_DB_ENV = "RESEARCH_TEMPORAL_DB"
DEFAULT_ADDRESS = "localhost:7233"
DEFAULT_TASK_QUEUE = "research-local"
DEFAULT_SELECTION_ACTIVITY = "select_probe_results_activity"


def default_temporal_db_path() -> pathlib.Path:
    """Return the local Temporal SQLite database path."""
    configured = os.environ.get(TEMPORAL_DB_ENV)
    if configured:
        return pathlib.Path(configured).expanduser()
    return pathlib.Path.home() / ".automata" / "research" / "temporal" / "temporal.db"


def ensure_temporal_cli() -> str:
    """Return the Temporal CLI path, or raise if it is unavailable."""
    temporal = shutil.which("temporal")
    if temporal is None:
        raise RuntimeError("temporal CLI not found on PATH")
    return temporal


def build_start_dev_command() -> list[str]:
    """Build the Temporal dev-server command and ensure DB parent exists."""
    db_path = default_temporal_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return ["temporal", "server", "start-dev", "--db-filename", str(db_path)]


@dataclasses.dataclass(frozen=True)
class WorkerConfig:
    """Concrete Temporal worker registration config."""

    address: str
    task_queue: str
    workflows: Sequence[type]
    activities: Sequence[Callable[..., Any]]
    activity_workers: int = 1


def build_worker_config(
    *,
    address: str = DEFAULT_ADDRESS,
    task_queue: str = DEFAULT_TASK_QUEUE,
    activities: Sequence[Callable[..., Any]],
    activity_workers: int = 1,
) -> WorkerConfig:
    """Build a generic research worker config with concrete activities."""
    if activity_workers < 1:
        raise ValueError("activity_workers must be at least 1.")
    if not activities:
        raise ValueError("at least one activity must be registered.")
    registered_activities = _with_probe_selection_activity(activities)
    return WorkerConfig(
        address=address,
        task_queue=task_queue,
        workflows=[
            research.workflows.ResearchProbeWorkflow,
            research.workflows.ResearchTrialWorkflow,
            research.workflows.ResearchCampaignDemoWorkflow,
        ],
        activities=_with_campaign_demo_activities(registered_activities),
        activity_workers=activity_workers,
    )


async def run_worker(config: WorkerConfig) -> None:
    """Run a Temporal worker for the configured research workflows."""
    client = await temporalio.client.Client.connect(config.address)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.activity_workers
    ) as executor:
        worker = temporalio.worker.Worker(
            client,
            task_queue=config.task_queue,
            workflows=list(config.workflows),
            activities=list(config.activities),
            activity_executor=executor,
        )
        await worker.run()


def make_probe_workflow_id(prefix: str = "research-probe") -> str:
    """Return a unique workflow id for a probe campaign."""
    return f"{prefix}-{uuid.uuid4().hex}"


async def start_probe_workflow(
    *,
    address: str,
    task_queue: str,
    db_path: pathlib.Path,
    experiment_ids: Sequence[int],
    trial_activity: str,
    workflow_id: str | None = None,
    attempt: int = 1,
    objective: dict[str, object] | None = None,
    selection_activity: str = DEFAULT_SELECTION_ACTIVITY,
) -> str:
    """Start a probe workflow and return its Temporal workflow id."""
    resolved_workflow_id = workflow_id or make_probe_workflow_id()
    client = await temporalio.client.Client.connect(address)
    await client.start_workflow(
        research.workflows.ResearchProbeWorkflow.run,
        args=[
            str(db_path),
            list(experiment_ids),
            trial_activity,
            attempt,
            objective or {},
            selection_activity,
        ],
        id=resolved_workflow_id,
        task_queue=task_queue,
        id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
    )
    return resolved_workflow_id


def make_campaign_demo_workflow_id(prefix: str = "research-campaign-demo") -> str:
    """Return a unique workflow id for a Temporal capability demo."""
    return f"{prefix}-{uuid.uuid4().hex}"


async def start_campaign_demo_workflow(
    *,
    address: str,
    task_queue: str,
    campaign_id: str,
    experiment_ids: Sequence[int],
    workflow_id: str | None = None,
    config: dict[str, object] | None = None,
) -> str:
    """Start the complex campaign demo workflow and return its workflow id."""
    resolved_workflow_id = workflow_id or make_campaign_demo_workflow_id()
    client = await temporalio.client.Client.connect(address)
    await client.start_workflow(
        research.workflows.ResearchCampaignDemoWorkflow.run,
        args=[campaign_id, list(experiment_ids), config or {}],
        id=resolved_workflow_id,
        task_queue=task_queue,
        id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
    )
    return resolved_workflow_id


async def wait_for_workflow_result(
    *,
    address: str,
    workflow_id: str,
    timeout_seconds: float = 0,
) -> Any:
    """Wait for an existing workflow execution and return its result."""
    client = await temporalio.client.Client.connect(address)
    handle = client.get_workflow_handle(workflow_id)
    result = handle.result()
    if timeout_seconds > 0:
        return await asyncio.wait_for(result, timeout=timeout_seconds)
    return await result


def _with_probe_selection_activity(
    activities: Sequence[Callable[..., Any]],
) -> Sequence[Callable[..., Any]]:
    names = {activity.__name__ for activity in activities}
    if DEFAULT_SELECTION_ACTIVITY in names:
        return activities
    return [*activities, research.activities.select_probe_results_activity]


def _with_campaign_demo_activities(
    activities: Sequence[Callable[..., Any]],
) -> Sequence[Callable[..., Any]]:
    registered = list(activities)
    names = {activity.__name__ for activity in registered}
    for demo_activity in [
        research.activities.prepare_campaign_demo_activity,
        research.activities.score_campaign_demo_activity,
        research.activities.summarize_campaign_demo_round_activity,
        research.activities.finalize_campaign_demo_activity,
    ]:
        if demo_activity.__name__ not in names:
            registered.append(demo_activity)
    return registered


async def start_trial_workflow(
    *,
    address: str,
    task_queue: str,
    db_path: pathlib.Path,
    experiment_id: int,
    trial_activity: str,
    workflow_id: str | None = None,
    attempt: int = 1,
) -> str:
    """Start a single trial workflow and return its Temporal workflow id."""
    resolved_workflow_id = workflow_id or research.workflows.trial_workflow_id(
        experiment_id,
        attempt,
    )
    client = await temporalio.client.Client.connect(address)
    await client.start_workflow(
        research.workflows.ResearchTrialWorkflow.run,
        args=[str(db_path), experiment_id, trial_activity, attempt],
        id=resolved_workflow_id,
        task_queue=task_queue,
        id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
    )
    return resolved_workflow_id
