from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
from pathlib import Path

from temporalio.client import Client

from autonomy.org.parser import read_tracker
from autonomy.workflows import SymphonyTrackerWorkflow


def _repo_root(start: Path | None = None) -> Path:
    configured = os.environ.get("AUTONOMY_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root")


async def _submit(client: Client, tracker_path: str, task_queue: str) -> None:
    tracker = Path(tracker_path).resolve()
    doc = read_tracker(tracker)
    slug = doc.header["AUTONOMY_RUN_SLUG"]
    workflow_id = f"autonomy-tracker-{slug}"
    handle = await client.start_workflow(
        SymphonyTrackerWorkflow.run,
        args=[str(tracker)],
        id=workflow_id,
        task_queue=task_queue,
    )
    print(f"started workflow {handle.id}")


async def _status(client: Client, slug: str) -> None:
    handle = client.get_workflow_handle(f"autonomy-tracker-{slug}")
    status = await handle.query(SymphonyTrackerWorkflow.status)
    print(
        f"active: {status['active_task']}   completed: {status['completed']}   "
        f"blocked: {status['blocked_ids']}   last_gate: {status['last_gate_summary']}"
    )


async def _signal(client: Client, slug: str, signal_name: str, task_id: str | None) -> None:
    handle = client.get_workflow_handle(f"autonomy-tracker-{slug}")
    if signal_name == "pause":
        await handle.signal(SymphonyTrackerWorkflow.pause_after_current)
    elif signal_name == "cancel":
        await handle.signal(SymphonyTrackerWorkflow.cancel)
    elif signal_name == "skip":
        if task_id is None:
            raise SystemExit("signal skip requires task_id")
        await handle.signal(SymphonyTrackerWorkflow.skip_task, args=[task_id])
    elif signal_name == "rerun":
        if task_id is None:
            raise SystemExit("signal rerun requires task_id")
        await handle.signal(SymphonyTrackerWorkflow.rerun_task, args=[task_id])
    else:
        raise SystemExit(f"unknown signal: {signal_name}")


async def _tail(client: Client, slug: str) -> None:
    handle = client.get_workflow_handle(f"autonomy-tracker-{slug}")
    status = await handle.query(SymphonyTrackerWorkflow.status)
    active = status.get("active_task")
    if not active:
        raise SystemExit("no active task")
    log_path = _repo_root() / ".autonomy" / "runs" / slug / "artifacts" / str(active) / "run.log"
    try:
        subprocess.run(["tail", "-f", str(log_path)], check=False)
    except KeyboardInterrupt:
        pass


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Autonomy CLI")
    parser.add_argument("--address", default="localhost:7233")
    parser.add_argument("--task-queue", default="autonomy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("tracker_path")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("slug")

    signal_parser = subparsers.add_parser("signal")
    signal_parser.add_argument("slug")
    signal_parser.add_argument("signal_name", choices=["pause", "cancel", "skip", "rerun"])
    signal_parser.add_argument("task_id", nargs="?")

    tail_parser = subparsers.add_parser("tail")
    tail_parser.add_argument("slug")

    args = parser.parse_args()
    client = await Client.connect(args.address)
    if args.command == "submit":
        await _submit(client, args.tracker_path, args.task_queue)
    elif args.command == "status":
        await _status(client, args.slug)
    elif args.command == "signal":
        await _signal(client, args.slug, args.signal_name, args.task_id)
    elif args.command == "tail":
        await _tail(client, args.slug)


def main() -> None:
    asyncio.run(_main())
