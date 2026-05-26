"""Command-line interface for scientific experiment management."""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import subprocess

import research.db
import research.reports
import research.temporal


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

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("target")

    temporal_parser = subparsers.add_parser("temporal")
    temporal_subparsers = temporal_parser.add_subparsers(
        dest="temporal_command",
        required=True,
    )
    temporal_subparsers.add_parser("start-dev")
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


def command_report(db_path: pathlib.Path, target: str) -> int:
    """Print a Markdown report for selected intents or one experiment."""
    research.db.init_db(db_path)
    if target == "selected":
        print(research.reports.render_selected_intents_report(db_path))
        return 0
    try:
        experiment_id = int(target)
    except ValueError as exc:
        raise SystemExit(f"Unknown report target: {target}") from exc
    print(research.reports.render_experiment_report(db_path, experiment_id))
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
    if args.command == "report":
        return command_report(db_path, args.target)
    if args.command == "temporal" and args.temporal_command == "start-dev":
        research.temporal.ensure_temporal_cli()
        return subprocess.call(research.temporal.build_start_dev_command())
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
