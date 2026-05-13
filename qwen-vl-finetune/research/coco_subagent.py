from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from research.subagents import (
    AgentProvider,
    SubagentTask,
    build_command,
    discover_providers,
    implementation_prompt,
)


DEFAULT_FILES = (
    "qwen-vl-finetune/research/",
    "qwen-vl-finetune/tests/research/",
)
DEFAULT_TEST_COMMAND = "cd qwen-vl-finetune && PYTHONPATH=. pytest tests/research -v"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return repo_root().parent


def build_coco_task(
    *,
    prompt: str | None,
    task_name: str,
    files: list[str],
    test_command: str,
    worktree: str,
    workspace: Path,
) -> SubagentTask:
    if prompt is None:
        prompt = implementation_prompt(task_name, files, test_command)
    return SubagentTask(
        provider=AgentProvider.COCO,
        worktree=worktree,
        workspace=workspace,
        prompt=prompt,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch or print a repo-local Coco CLI subagent command.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Direct prompt for Coco. If omitted, a standard implementation prompt is built.",
    )
    parser.add_argument("--task-name", default="Qwen3-VL research implementation")
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Owned file or directory. Repeatable. Defaults to research source and tests.",
    )
    parser.add_argument("--test-command", default=DEFAULT_TEST_COMMAND)
    parser.add_argument("--worktree", default="qwen3vl-coco-subagent")
    parser.add_argument("--workspace", type=Path, default=workspace_root())
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run Coco instead of printing the command.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    providers = discover_providers()
    if AgentProvider.COCO not in providers:
        raise SystemExit("coco CLI not found on PATH or ~/.local/bin/coco")

    task = build_coco_task(
        prompt=args.prompt,
        task_name=args.task_name,
        files=args.files or list(DEFAULT_FILES),
        test_command=args.test_command,
        worktree=args.worktree,
        workspace=args.workspace,
    )
    command = build_command(task)
    if not args.execute:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, cwd=args.workspace, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
