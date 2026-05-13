from __future__ import annotations

import argparse
import asyncio
import concurrent.futures

from temporalio.client import Client
from temporalio.worker import Worker

from autonomy.activities import (
    launch_run,
    org_append_blocker,
    org_pick_next_ready,
    org_transition,
    run_done_gate,
    worktree_create,
    worktree_destroy_if_clean,
)
from autonomy.workflows import SymphonyTrackerWorkflow


async def run_worker(address: str, task_queue: str) -> None:
    client = await Client.connect(address)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as activity_executor:
        worker = Worker(
            client,
            task_queue=task_queue,
            workflows=[SymphonyTrackerWorkflow],
            activities=[
                org_pick_next_ready,
                org_transition,
                org_append_blocker,
                worktree_create,
                worktree_destroy_if_clean,
                launch_run,
                run_done_gate,
            ],
            activity_executor=activity_executor,
        )
        await worker.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the autonomy Temporal worker")
    parser.add_argument("--address", default="localhost:7233")
    parser.add_argument("--task-queue", default="autonomy")
    args = parser.parse_args()
    asyncio.run(run_worker(args.address, args.task_queue))
