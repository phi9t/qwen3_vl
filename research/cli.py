"""Command-line interface for scientific experiment management."""

from __future__ import annotations

import argparse
import contextlib
import pathlib

import research.adapters
import research.artifacts
import research.db
import research.models


DEFAULT_DB = pathlib.Path(".research") / "research.sqlite"


def build_parser() -> argparse.ArgumentParser:
    """Build the research command-line parser."""
    parser = argparse.ArgumentParser(description="Scientific experiment manager")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    subparsers = parser.add_subparsers(dest="command", required=True)

    db_parser = subparsers.add_parser("db")
    db_subparsers = db_parser.add_subparsers(dest="db_command", required=True)
    db_subparsers.add_parser("init")

    subparsers.add_parser("status")
    subparsers.add_parser("manager")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("id")

    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--adapter", required=True)
    probe_parser.add_argument("--model", required=True)
    probe_parser.add_argument("--profile", required=True)
    probe_parser.add_argument("--artifact-root", default="")
    return parser


def command_status(db_path: pathlib.Path) -> int:
    """Print a compact experiment status summary."""
    research.db.init_db(db_path)
    with contextlib.closing(research.db.connect(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    if count == 0:
        print("No experiments")
    else:
        print(f"Experiments: {count}")
    return 0


def command_probe(args: argparse.Namespace, db_path: pathlib.Path) -> int:
    """Generate candidate intents and queue experiments."""
    research.db.init_db(db_path)
    adapter = research.adapters.load_adapter(args.adapter)
    request = research.models.ProbeRequest(model=args.model, profile=args.profile)
    artifact_root = (
        pathlib.Path(args.artifact_root)
        if args.artifact_root
        else research.artifacts.default_artifact_root()
    )
    intents = adapter.generate_probe_intents(request)

    experiment_ids = []
    for intent in intents:
        intent_id = research.db.insert_intent(db_path, intent)
        experiment_id = research.db.create_experiment(
            db_path,
            intent_id=intent_id,
            adapter=args.adapter,
            artifact_root=str(artifact_root),
            artifact_subdir=f"{adapter.name}/{intent_id}",
        )
        experiment_ids.append(experiment_id)

    print(f"Created {len(experiment_ids)} experiments")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the research command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = pathlib.Path(args.db)

    if args.command == "db" and args.db_command == "init":
        research.db.init_db(db_path)
        print(db_path)
        return 0
    if args.command == "status":
        return command_status(db_path)
    if args.command == "probe":
        return command_probe(args, db_path)
    if args.command in {"manager", "report"}:
        parser.error(f"{args.command} is not implemented in this migration step")
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
