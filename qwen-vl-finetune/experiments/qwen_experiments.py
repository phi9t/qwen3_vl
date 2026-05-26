"""Qwen-VL experiment command-line entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import pathlib

import experiments.qwen_adapter
import experiments.qwen_temporal
import research.artifacts
import research.db
import research.models
import research.probes
import research.temporal
import research.workflows


DEFAULT_DB = pathlib.Path(".research") / "research.sqlite"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_TRIAL_ACTIVITY = "qwen_run_trial_activity"
DEFAULT_PROBE_OBJECTIVE = {
    "metric": "val_loss",
    "direction": "minimize",
    "top_k": 1,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the Qwen experiment CLI parser."""
    parser = argparse.ArgumentParser(description="Qwen-VL experiment manager")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--model", required=True)
    probe_parser.add_argument("--profile", required=True)
    probe_parser.add_argument("--artifact-root", default="")
    probe_parser.add_argument("--address", default=research.temporal.DEFAULT_ADDRESS)
    probe_parser.add_argument(
        "--task-queue",
        default=research.temporal.DEFAULT_TASK_QUEUE,
    )
    probe_parser.add_argument("--workflow-id", default="")
    probe_parser.add_argument("--plan-only", action="store_true")
    probe_parser.add_argument("--dry-run", action="store_true")
    probe_parser.add_argument("--max-experiments", default=0, type=int)
    probe_parser.add_argument("--wait", action="store_true")
    probe_parser.add_argument("--timeout-seconds", default=0.0, type=float)

    manager_parser = subparsers.add_parser("manager")
    manager_parser.add_argument("--address", default=research.temporal.DEFAULT_ADDRESS)
    manager_parser.add_argument(
        "--task-queue",
        default=research.temporal.DEFAULT_TASK_QUEUE,
    )
    manager_parser.add_argument("--activity-workers", default=1, type=int)

    run_trial_parser = subparsers.add_parser("run-trial")
    run_trial_parser.add_argument("experiment_id", type=int)
    run_trial_parser.add_argument("--attempt", default=1, type=int)
    run_trial_parser.add_argument(
        "--address", default=research.temporal.DEFAULT_ADDRESS
    )
    run_trial_parser.add_argument(
        "--task-queue",
        default=research.temporal.DEFAULT_TASK_QUEUE,
    )
    run_trial_parser.add_argument("--workflow-id", default="")

    subparsers.add_parser("status")
    return parser


def create_probe_plan(
    args: argparse.Namespace,
    db_path: pathlib.Path,
) -> research.probes.ProbeCreationResult:
    """Create Qwen-VL probe experiments using the generic research helper."""
    adapter = experiments.qwen_adapter.QwenVlAdapter()
    artifact_root = (
        pathlib.Path(args.artifact_root)
        if args.artifact_root
        else research.artifacts.default_artifact_root()
    )
    if args.max_experiments < 0:
        raise ValueError("--max-experiments must be non-negative")
    result = research.probes.create_probe_experiments(
        db_path,
        adapter=adapter,
        request=research.models.ProbeRequest(
            model=args.model,
            profile=args.profile,
            objective=DEFAULT_PROBE_OBJECTIVE,
            budget={"dry_run": args.dry_run},
        ),
        artifact_root=artifact_root,
        adapter_ref=adapter.name,
        max_experiments=args.max_experiments,
    )
    return result


async def command_probe(args: argparse.Namespace, db_path: pathlib.Path) -> int:
    """Create Qwen-VL probe experiments and start the Temporal probe workflow."""
    result = create_probe_plan(args, db_path)
    print(f"Created {len(result.experiment_ids)} experiments")
    if args.plan_only or not result.experiment_ids:
        if not result.experiment_ids:
            print("No experiments to submit")
        return 0
    workflow_id = args.workflow_id or research.temporal.make_probe_workflow_id(
        "qwen-probe"
    )
    try:
        workflow_id = await research.temporal.start_probe_workflow(
            address=args.address,
            task_queue=args.task_queue,
            db_path=db_path,
            experiment_ids=result.experiment_ids,
            trial_activity=QWEN_TRIAL_ACTIVITY,
            workflow_id=workflow_id,
            objective=DEFAULT_PROBE_OBJECTIVE,
        )
    except Exception:
        for experiment_id in result.experiment_ids:
            research.db.transition_experiment(
                db_path,
                experiment_id,
                "submission_failed",
            )
        raise
    for experiment_id in result.experiment_ids:
        research.db.transition_experiment(
            db_path,
            experiment_id,
            "submitted",
            workflow_id=research.workflows.trial_workflow_id(experiment_id, 1),
        )
    print(f"Started Temporal workflow {workflow_id}")
    if args.wait:
        workflow_result = await research.temporal.wait_for_workflow_result(
            address=args.address,
            workflow_id=workflow_id,
            timeout_seconds=args.timeout_seconds,
        )
        print(f"Workflow {workflow_id} completed")
        print(json.dumps(workflow_result, indent=2, sort_keys=True))
    return 0


def command_status(db_path: pathlib.Path) -> int:
    """Print a compact Qwen experiment status summary."""
    research.db.init_db(db_path)
    with contextlib.closing(research.db.connect(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    print("No experiments" if count == 0 else f"Experiments: {count}")
    return 0


def build_worker_config(
    address: str,
    task_queue: str,
    activity_workers: int = 1,
) -> research.temporal.WorkerConfig:
    """Build the Qwen Temporal worker registration config."""
    return research.temporal.build_worker_config(
        address=address,
        task_queue=task_queue,
        activities=[experiments.qwen_temporal.qwen_run_trial_activity],
        activity_workers=activity_workers,
    )


async def run_manager(
    address: str,
    task_queue: str,
    activity_workers: int = 1,
) -> None:
    """Run the Qwen-VL Temporal worker."""
    config = build_worker_config(address, task_queue, activity_workers)
    await research.temporal.run_worker(config)


async def command_run_trial(args: argparse.Namespace, db_path: pathlib.Path) -> int:
    """Start one Qwen-VL experiment as a Temporal trial workflow."""
    workflow_id = args.workflow_id or research.workflows.trial_workflow_id(
        args.experiment_id,
        args.attempt,
    )
    try:
        await research.temporal.start_trial_workflow(
            address=args.address,
            task_queue=args.task_queue,
            db_path=db_path,
            experiment_id=args.experiment_id,
            trial_activity=QWEN_TRIAL_ACTIVITY,
            workflow_id=workflow_id,
            attempt=args.attempt,
        )
    except Exception:
        research.db.transition_experiment(
            db_path,
            args.experiment_id,
            "submission_failed",
        )
        raise
    research.db.transition_experiment(
        db_path,
        args.experiment_id,
        "submitted",
        workflow_id=workflow_id,
    )
    print(f"Started Temporal workflow {workflow_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the Qwen experiment CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = pathlib.Path(args.db)

    if args.command == "probe":
        return asyncio.run(command_probe(args, db_path))
    if args.command == "status":
        return command_status(db_path)
    if args.command == "manager":
        asyncio.run(
            run_manager(
                args.address,
                args.task_queue,
                args.activity_workers,
            )
        )
        return 0
    if args.command == "run-trial":
        return asyncio.run(command_run_trial(args, db_path))
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
