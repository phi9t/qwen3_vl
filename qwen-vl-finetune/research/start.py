from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from temporalio.client import Client

from research.workflows import ProfiledExperimentWorkflow, SingleTrialWorkflow


def parse_env_overrides(overrides: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected KEY=VALUE for --env, got {override!r}")
        key, value = override.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Expected KEY=VALUE for --env, got {override!r}")
        env[key] = value
    return env


def build_single_trial_spec(
    *,
    profile: str,
    phase: str,
    trial_name: str,
    env_overrides: list[str],
) -> dict:
    if not trial_name:
        raise ValueError("trial name is required for --command trial")
    return {
        "profile": profile,
        "phase": phase,
        "trial": trial_name,
        "env": parse_env_overrides(env_overrides),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Start a Qwen3-VL research workflow")
    parser.add_argument("--address", default="localhost:7233")
    parser.add_argument("--task-queue", default="qwen3vl-b200")
    parser.add_argument("--profile", default="b200")
    parser.add_argument("--phase", default="probe")
    parser.add_argument("--trial-name", default="")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument(
        "--command",
        choices=["probe", "select", "sweep", "summarize", "full", "trial"],
        default="full",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workflow-id", default="")
    args = parser.parse_args()

    client = await Client.connect(args.address)
    workflow_id = args.workflow_id or (
        f"qwen3vl-{args.profile}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if args.command == "trial":
        try:
            spec = build_single_trial_spec(
                profile=args.profile,
                phase=args.phase,
                trial_name=args.trial_name,
                env_overrides=args.env,
            )
        except ValueError as exc:
            parser.error(str(exc))
        handle = await client.start_workflow(
            SingleTrialWorkflow.run,
            args=[spec, args.dry_run],
            id=workflow_id,
            task_queue=args.task_queue,
        )
        print(handle.id)
        return

    handle = await client.start_workflow(
        ProfiledExperimentWorkflow.run,
        args=[args.profile, args.dry_run, args.command],
        id=workflow_id,
        task_queue=args.task_queue,
    )
    print(handle.id)


if __name__ == "__main__":
    asyncio.run(main())
