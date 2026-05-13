from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from shutil import which as shutil_which


class AgentProvider(StrEnum):
    COCO = "coco"
    CURSOR_AGENT = "cursor_agent"


@dataclass(frozen=True)
class SubagentTask:
    provider: AgentProvider
    worktree: str
    workspace: Path
    prompt: str


def discover_providers(
    which: Callable[[str], str | None] = shutil_which,
) -> dict[AgentProvider, str]:
    providers: dict[AgentProvider, str] = {}
    coco = which("coco")
    if coco:
        providers[AgentProvider.COCO] = coco
    cursor_agent = which("agent")
    if cursor_agent:
        providers[AgentProvider.CURSOR_AGENT] = cursor_agent
    return providers


def build_command(task: SubagentTask) -> list[str]:
    if task.provider == AgentProvider.COCO:
        return [
            "coco",
            "-p",
            "-y",
            "--worktree",
            task.worktree,
            "--query-timeout",
            "30m",
            task.prompt,
        ]
    if task.provider == AgentProvider.CURSOR_AGENT:
        return [
            "agent",
            "-p",
            "--trust",
            "--workspace",
            str(task.workspace),
            "-w",
            task.worktree,
            "--output-format",
            "text",
            task.prompt,
        ]
    raise ValueError(f"Unsupported provider: {task.provider}")


def implementation_prompt(task_name: str, files: list[str], test_command: str) -> str:
    file_list = "\n".join(f"- {path}" for path in files)
    return (
        f"Implement {task_name} in this repository.\n"
        "You are not alone in the codebase. Do not revert unrelated changes.\n"
        "Own only these files:\n"
        f"{file_list}\n"
        f"Run this verification command before finishing:\n{test_command}\n"
        "Return a concise summary and list changed files."
    )
