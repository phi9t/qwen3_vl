"""Qwen-VL experiment command-line entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import pathlib

import temporalio.client
import temporalio.worker

import experiments.qwen_adapter
import experiments.qwen_temporal
import experiments.qwen_workflows
import research.artifacts
import research.db
import research.models
import research.probes
import research.temporal


DEFAULT_DB = pathlib.Path(".research") / "research.sqlite"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    """Build the Qwen experiment CLI parser."""
    parser = argparse.ArgumentParser(description="Qwen-VL experiment manager")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--model", required=True)
    probe_parser.add_argument("--profile", required=True)
    probe_parser.add_argument("--artifact-root", default="")

    manager_parser = subparsers.add_parser("manager")
    manager_parser.add_argument("--address", default=research.temporal.DEFAULT_ADDRESS)
    manager_parser.add_argument(
        "--task-queue",
        default=research.temporal.DEFAULT_TASK_QUEUE,
    )

    subparsers.add_parser("status")
    return parser


def command_probe(args: argparse.Namespace, db_path: pathlib.Path) -> int:
    """Create Qwen-VL probe experiments using the generic research helper."""
    adapter = experiments.qwen_adapter.QwenVlAdapter()
    artifact_root = (
        pathlib.Path(args.artifact_root)
        if args.artifact_root
        else research.artifacts.default_artifact_root()
    )
    result = research.probes.create_probe_experiments(
        db_path,
        adapter=adapter,
        request=research.models.ProbeRequest(
            model=args.model,
            profile=args.profile,
        ),
        artifact_root=artifact_root,
        adapter_ref=adapter.name,
    )
    print(f"Created {len(result.experiment_ids)} experiments")
    return 0


def command_status(db_path: pathlib.Path) -> int:
    """Print a compact Qwen experiment status summary."""
    research.db.init_db(db_path)
    with contextlib.closing(research.db.connect(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    print("No experiments" if count == 0 else f"Experiments: {count}")
    return 0


def build_worker_config(address: str, task_queue: str) -> dict[str, object]:
    """Build the Qwen Temporal worker registration config."""
    return {
        "address": address,
        "task_queue": task_queue,
        "workflows": [
            experiments.qwen_workflows.QwenProbeWorkflow,
            experiments.qwen_workflows.QwenTrialWorkflow,
        ],
        "activities": [experiments.qwen_temporal.qwen_run_trial_activity],
    }


async def run_manager(address: str, task_queue: str) -> None:
    """Run the Qwen-VL Temporal worker."""
    config = build_worker_config(address, task_queue)
    client = await temporalio.client.Client.connect(str(config["address"]))
    worker = temporalio.worker.Worker(
        client,
        task_queue=str(config["task_queue"]),
        workflows=list(config["workflows"]),
        activities=list(config["activities"]),
    )
    await worker.run()


def main(argv: list[str] | None = None) -> int:
    """Run the Qwen experiment CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = pathlib.Path(args.db)

    if args.command == "probe":
        return command_probe(args, db_path)
    if args.command == "status":
        return command_status(db_path)
    if args.command == "manager":
        asyncio.run(run_manager(args.address, args.task_queue))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
