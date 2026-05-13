from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


KEYWORDS = (
    "TODO",
    "READY",
    "IN-PROGRESS",
    "BLOCKED",
    "AWAITING-GATE",
    "DONE",
    "WONTFIX",
    "FAILED",
)

WORKFLOW_OWNED_PROPS = (
    "OWNER",
    "STARTED",
    "FINISHED",
    "EXIT_CODE",
    "GATE_RESULT",
    "ARTIFACTS",
    "WORKTREE",
    "PR",
)


@dataclass(frozen=True)
class OrgTask:
    id: str
    executor: str
    gate: str
    depends: frozenset[str]
    timeout: timedelta
    gpus: int
    branch: str | None
    goal: str
    constraints: list[str]
    acceptance_cmds: list[str]
    slug: str
    state: str
    position: int


@dataclass
class TrackerDoc:
    header: dict[str, str]
    tasks: list[OrgTask]
    path: Path


def parse_timeout(s: str) -> timedelta:
    """Parse a Go-style duration string into a timedelta.

    Accepts forms such as ``6h``, ``2h``, ``30m``, ``45s``, ``1h30m``.
    """
    if not s:
        raise ValueError("timeout string must not be empty")
    total_seconds = 0.0
    for match in re.finditer(r"(\d+)([hms])", s):
        value = int(match.group(1))
        unit = match.group(2)
        if unit == "h":
            total_seconds += value * 3600
        elif unit == "m":
            total_seconds += value * 60
        elif unit == "s":
            total_seconds += value
        else:
            raise ValueError(f"unknown timeout unit: {unit}")
    if total_seconds == 0.0:
        raise ValueError(f"unable to parse timeout: {s!r}")
    return timedelta(seconds=int(total_seconds))
