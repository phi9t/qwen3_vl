from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from autonomy.org.schema import OrgTask


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    log_path: str
    summary: str
    structured_result: dict[str, object] | None = None


class Executor(Protocol):
    name: str

    def build_command(self, task: OrgTask, worktree: Path) -> list[str]:
        ...

    def parse_result(self, exit_code: int, log_path: Path) -> RunResult:
        ...


REGISTRY: dict[str, Executor] = {}


def register(executor: Executor) -> Executor:
    REGISTRY[executor.name] = executor
    return executor


def get(name: str) -> Executor:
    if name in REGISTRY:
        return REGISTRY[name]
    known = ", ".join(sorted(REGISTRY)) or "<none>"
    raise KeyError(f"Unknown executor {name!r}; known executors: {known}")
