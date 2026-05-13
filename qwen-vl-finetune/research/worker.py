from __future__ import annotations

import argparse
import asyncio
import concurrent.futures

from temporalio.client import Client
from temporalio.worker import Worker

from research.activities import (
    load_profile_activity,
    plan_probe_trials_activity,
    plan_sweep_trials_activity,
    run_trial_activity,
    select_capacity_activity,
    select_capacity_from_results_activity,
    summarize_campaign_activity,
)
from research.workflows import ProfiledExperimentWorkflow, SingleTrialWorkflow


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Qwen3-VL research Temporal worker")
    parser.add_argument("--address", default="localhost:7233")
    parser.add_argument("--task-queue", default="qwen3vl-b200")
    args = parser.parse_args()

    client = await Client.connect(args.address)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as activity_executor:
        worker = Worker(
            client,
            task_queue=args.task_queue,
            workflows=[ProfiledExperimentWorkflow, SingleTrialWorkflow],
            activities=[
                load_profile_activity,
                plan_probe_trials_activity,
                plan_sweep_trials_activity,
                run_trial_activity,
                select_capacity_from_results_activity,
                select_capacity_activity,
                summarize_campaign_activity,
            ],
            activity_executor=activity_executor,
        )
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
