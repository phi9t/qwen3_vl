from __future__ import annotations

import re
from pathlib import Path

import orgparse

from autonomy.org.schema import KEYWORDS, OrgTask, TrackerDoc, parse_timeout


def read_tracker(path: Path) -> TrackerDoc:
    """Parse an org-mode tracker file into a :class:`TrackerDoc`."""
    root = orgparse.load(str(path))

    # Parse top-of-file #+KEY: value headers
    header: dict[str, str] = {}
    raw_lines = root.env._nodes[0]._lines if root.env._nodes else []
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("#+"):
            m = re.match(r"#\+([A-Z_][A-Z_0-9]*):\s*(.*)", stripped)
            if m:
                header[m.group(1)] = m.group(2).strip()
        elif stripped.startswith("* "):
            break

    slug = header.get("AUTONOMY_RUN_SLUG")
    if not slug:
        raise ValueError(f"Missing required header #+AUTONOMY_RUN_SLUG in {path}")

    default_executor = header.get("AUTONOMY_DEFAULT_EXECUTOR", "claude-code")
    default_gate = header.get("AUTONOMY_DEFAULT_GATE", "tests+typecheck+gpt2-smoke")

    # Find exactly one first-level "Tasks" heading (case-insensitive)
    tasks_node = None
    for node in root.children:
        if node.heading and node.heading.strip().lower() == "tasks":
            if tasks_node is not None:
                raise ValueError(f"Multiple 'Tasks' headings found in {path}")
            tasks_node = node
    if tasks_node is None:
        raise ValueError(f"Missing 'Tasks' heading in {path}")

    tasks: list[OrgTask] = []
    seen_ids: set[str] = set()

    for position, child in enumerate(tasks_node.children):
        heading = child.heading
        if not heading:
            continue

        # orgparse strips the TODO keyword into child.todo; heading is the rest
        state = (child.todo or "").upper()
        if state not in KEYWORDS:
            raise ValueError(
                f"Unknown state {state!r} in heading '{heading}' in {path}"
            )

        # First whitespace-delimited token of the heading text is the ID
        title_words = heading.split()
        if not title_words:
            raise ValueError(f"Task heading missing ID: '{heading}' in {path}")
        task_id = title_words[0]

        if task_id in seen_ids:
            raise ValueError(f"Duplicate task ID {task_id!r} in {path}")
        seen_ids.add(task_id)

        # Properties drawer
        props = child.properties or {}
        executor = (props.get("EXECUTOR", "") or default_executor).strip()
        gate = (props.get("GATE", "") or default_gate).strip()
        depends_raw = props.get("DEPENDS", "")
        depends = frozenset(depends_raw.split()) if depends_raw else frozenset()
        timeout_str = props.get("TIMEOUT", "")
        timeout = parse_timeout(timeout_str) if timeout_str else parse_timeout("6h")
        gpus = int(props.get("GPUS", "1") or "1")
        branch = props.get("BRANCH") or None

        # Body sections
        body_lines = child.body.splitlines() if child.body else []
        goal = ""
        constraints: list[str] = []
        acceptance_cmds: list[str] = []

        section = None
        for line in body_lines:
            stripped = line.strip()
            if stripped.lower().startswith("*goal.*"):
                section = "goal"
                goal = stripped[len("*goal.*") :].strip()
                continue
            elif stripped.lower().startswith("*constraints.*"):
                section = "constraints"
                continue
            elif stripped.lower().startswith("*acceptance.*"):
                section = "acceptance"
                continue
            elif stripped.startswith("*"):
                section = None
                continue

            if section == "goal":
                if goal:
                    goal += "\n" + stripped
                else:
                    goal = stripped
            elif section == "constraints":
                if stripped.startswith("-") or stripped.startswith("+"):
                    constraints.append(stripped.lstrip("-").lstrip("+").strip())
            elif section == "acceptance" and stripped.startswith("cmd:"):
                acceptance_cmds.append(stripped[len("cmd:") :].strip())

        tasks.append(
            OrgTask(
                id=task_id,
                executor=executor,
                gate=gate,
                depends=depends,
                timeout=timeout,
                gpus=gpus,
                branch=branch,
                goal=goal,
                constraints=constraints,
                acceptance_cmds=acceptance_cmds,
                slug=slug,
                state=state,
                position=position,
            )
        )

    return TrackerDoc(header=header, tasks=tasks, path=path)
