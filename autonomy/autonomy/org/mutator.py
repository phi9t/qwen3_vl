from __future__ import annotations

import contextlib
import fcntl
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from autonomy.org.schema import WORKFLOW_OWNED_PROPS


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def _write_atomic(path: Path, lines: list[str]) -> None:
    """Write *lines* atomically to *path* using a tempfile in the same dir."""
    data = "\n".join(lines)
    if data and not data.endswith("\n"):
        data += "\n"
    dir_path = path.parent
    fd, tmp = tempfile.mkstemp(dir=dir_path, prefix=".tracker-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # fsync parent directory to ensure rename is durable
        dir_fd = os.open(dir_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _find_task_block(lines: list[str], task_id: str) -> tuple[int, int]:
    """Return (start_line_index, end_line_index) for the task heading whose
    first whitespace-delimited token after the TODO keyword is *task_id*.
    The block includes the heading line and runs until the next sibling
    heading at the same or higher level, or EOF.
    """
    # Locate the heading line
    heading_idx = -1
    heading_level = -1
    for i, line in enumerate(lines):
        m = re.match(r"^(\*+)\s+(\S+)\s+(\S.*)$", line)
        if not m:
            continue
        level = len(m.group(1))
        keyword = m.group(2).upper()
        rest = m.group(3).lstrip()
        first_token = rest.split()[0]
        if first_token == task_id and keyword in (
            "TODO",
            "READY",
            "IN-PROGRESS",
            "BLOCKED",
            "AWAITING-GATE",
            "DONE",
            "WONTFIX",
            "FAILED",
        ):
            heading_idx = i
            heading_level = level
            break

    if heading_idx == -1:
        raise ValueError(f"Task {task_id!r} not found")

    # Find end of block
    end_idx = len(lines)
    for j in range(heading_idx + 1, len(lines)):
        m = re.match(r"^(\*+)\s", lines[j])
        if m and len(m.group(1)) <= heading_level:
            end_idx = j
            break

    return heading_idx, end_idx


def _update_property(lines: list[str], prop_name: str, value: str) -> list[str]:
    """Return a new list of lines with *prop_name* set to *value* inside the
    first :PROPERTIES: ... :END: drawer encountered in *lines*.
    """
    out: list[str] = []
    in_drawer = False
    updated = False
    prop_key = f":{prop_name}:"
    for line in lines:
        if line.strip() == ":PROPERTIES:":
            in_drawer = True
            out.append(line)
            continue
        if line.strip() == ":END:":
            if in_drawer and not updated:
                out.append(f"   {prop_key:<12} {value}")
                updated = True
            in_drawer = False
            out.append(line)
            continue
        if in_drawer and line.strip().startswith(prop_key):
            # Preserve leading whitespace (typically 3 spaces)
            leading = line[: len(line) - len(line.lstrip())]
            out.append(f"{leading}{prop_key:<12} {value}")
            updated = True
            continue
        out.append(line)
    return out


def _replace_task_block(
    lines: list[str], start: int, end: int, new_block: list[str]
) -> list[str]:
    return lines[:start] + new_block + lines[end:]


def transition(
    path: Path,
    task_id: str,
    new_state: str,
    props: dict[str, str],
) -> None:
    """Atomically transition *task_id* to *new_state*, update workflow-owned
    properties, and append a logbook line.
    """
    note = props.pop("_note", "")
    with open(path, "r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            lines = f.read().splitlines()
            start, end = _find_task_block(lines, task_id)
            block = lines[start:end]

            # Replace TODO keyword on heading line
            heading = block[0]
            m = re.match(r"^(\*+\s+)\S+(\s+.*)$", heading)
            if not m:
                raise ValueError(f"Malformed heading: {heading!r}")
            old_state_match = re.search(r"^\*+\s+(\S+)", heading)
            old_state = old_state_match.group(1) if old_state_match else ""
            block[0] = m.group(1) + new_state + m.group(2)

            # Update workflow-owned properties
            for key, val in props.items():
                if key.upper() in WORKFLOW_OWNED_PROPS:
                    block = _update_property(block, key.upper(), val)

            # Append logbook entry
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %a %H:%M")
            log_line = (
                f'- State "{new_state}" from "{old_state}" '
                f"[{timestamp}] :: {note}"
            )
            block = _append_to_logbook(block, log_line)

            lines = _replace_task_block(lines, start, end, block)
            _write_atomic(path, lines)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _append_to_logbook(block: list[str], line: str) -> list[str]:
    """Append *line* into the ``*** Logbook :LOGBOOK:`` drawer inside *block*.
    Creates the subheading and drawer if missing.
    """
    out: list[str] = []
    logbook_start = -1
    logbook_end = -1

    for i, l in enumerate(block):
        m = re.match(r"^(\*+)\s+Logbook\s*:LOGBOOK:", l, re.IGNORECASE)
        if m:
            logbook_start = i
            len(m.group(1))
            continue
        if logbook_start != -1 and l.strip() == ":END:":
            logbook_end = i
            break

    if logbook_start == -1:
        # Create Logbook subheading at one deeper level than the task heading
        task_level_match = re.match(r"^(\*+)", block[0])
        task_level = len(task_level_match.group(1)) if task_level_match else 2
        sub = "*" * (task_level + 1)
        out = [*block, f"{sub} Logbook   :LOGBOOK:", f"    {line}", "    :END:"]
        return out

    # Insert before :END:
    out = (
        [*block[:logbook_end], f"    {line}", *block[logbook_end:]]
    )
    return out


def append_logbook(path: Path, task_id: str, line: str) -> None:
    """Atomically append *line* to the Logbook drawer of *task_id*."""
    with open(path, "r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            lines = f.read().splitlines()
            start, end = _find_task_block(lines, task_id)
            block = _append_to_logbook(lines[start:end], line)
            lines = _replace_task_block(lines, start, end, block)
            _write_atomic(path, lines)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_blocker(path: Path, task_id: str, reason: str, excerpt: str) -> None:
    """Atomically create or replace the ``*** Blocker`` subheading under
    *task_id* with *reason* and a fenced excerpt.
    """
    with open(path, "r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            lines = f.read().splitlines()
            start, end = _find_task_block(lines, task_id)
            block = lines[start:end]

            # Determine task heading level
            task_level_match = re.match(r"^(\*+)", block[0])
            task_level = len(task_level_match.group(1)) if task_level_match else 2
            sub = "*" * (task_level + 1)

            # Find existing Blocker subheading
            blocker_start = -1
            blocker_end = -1
            for i, l in enumerate(block):
                if re.match(rf"^{re.escape(sub)}\s+Blocker\b", l, re.IGNORECASE):
                    blocker_start = i
                    for j in range(i + 1, len(block)):
                        if re.match(rf"^{re.escape(sub)}\s", block[j]):
                            blocker_end = j
                            break
                    if blocker_end == -1:
                        blocker_end = len(block)
                    break

            new_blocker = [
                f"{sub} Blocker",
                f"    Reason: {reason}",
                "    Excerpt:",
                "    #+begin_src text",
            ]
            for ex_line in excerpt.splitlines():
                new_blocker.append(f"    {ex_line}")
            new_blocker.append("    #+end_src")

            if blocker_start != -1:
                block = block[:blocker_start] + new_blocker + block[blocker_end:]
            else:
                block = block + new_blocker

            lines = _replace_task_block(lines, start, end, block)
            _write_atomic(path, lines)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
