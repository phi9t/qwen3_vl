from __future__ import annotations

import dataclasses
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

from temporalio import activity

import autonomy.executors.shell
from autonomy import sandbox, worktree
from autonomy.executors.base import get as get_executor
from autonomy.org import mutator, parser
from autonomy.org.schema import OrgTask, parse_timeout


def _repo_root(start: Path | None = None) -> Path:
    configured = os.environ.get("AUTONOMY_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root")


def _format_timeout(value: timedelta) -> str:
    seconds = int(value.total_seconds())
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return "".join(parts)


def _task_to_payload(task: OrgTask) -> dict[str, Any]:
    payload = dataclasses.asdict(task)
    payload["depends"] = sorted(task.depends)
    payload["timeout"] = _format_timeout(task.timeout)
    return payload


def _task_from_payload(task_d: dict[str, Any]) -> OrgTask:
    timeout_value = task_d["timeout"]
    if isinstance(timeout_value, timedelta):
        timeout = timeout_value
    elif isinstance(timeout_value, str):
        timeout = parse_timeout(timeout_value)
    elif isinstance(timeout_value, (int, float)):
        timeout = timedelta(seconds=float(timeout_value))
    else:
        raise TypeError(f"Unsupported timeout payload: {timeout_value!r}")
    depends_value = task_d.get("depends", [])
    if isinstance(depends_value, str):
        depends = frozenset(depends_value.split()) if depends_value else frozenset()
    else:
        depends = frozenset(str(item) for item in depends_value)
    return OrgTask(
        id=str(task_d["id"]),
        executor=str(task_d["executor"]),
        gate=str(task_d["gate"]),
        depends=depends,
        timeout=timeout,
        gpus=int(task_d["gpus"]),
        branch=str(task_d["branch"]) if task_d.get("branch") is not None else None,
        goal=str(task_d.get("goal", "")),
        constraints=[str(item) for item in task_d.get("constraints", [])],
        acceptance_cmds=[str(item) for item in task_d.get("acceptance_cmds", [])],
        slug=str(task_d["slug"]),
        state=str(task_d["state"]),
        position=int(task_d["position"]),
    )


@activity.defn(name="org_pick_next_ready")
def org_pick_next_ready(tracker_path: str) -> dict[str, Any] | None:
    doc = parser.read_tracker(Path(tracker_path))
    done_ids = {task.id for task in doc.tasks if task.state == "DONE"}
    for task in doc.tasks:
        if task.state == "TODO" and task.depends.issubset(done_ids):
            return _task_to_payload(task)
    return None


@activity.defn(name="org_transition")
def org_transition(
    tracker_path: str,
    task_id: str,
    new_state: str,
    props_to_set: dict[str, str],
) -> None:
    mutator.transition(Path(tracker_path), task_id, new_state, dict(props_to_set))


@activity.defn(name="org_append_blocker")
def org_append_blocker(tracker_path: str, task_id: str, reason: str, excerpt: str) -> None:
    mutator.append_blocker(Path(tracker_path), task_id, reason, excerpt)


@activity.defn(name="worktree_create")
def worktree_create(slug: str, task_id: str, branch: str | None) -> str:
    return str(worktree.create(_repo_root(), slug, task_id, branch))


@activity.defn(name="worktree_destroy_if_clean")
def worktree_destroy_if_clean(worktree_path: str, ended_done: bool) -> bool:
    return worktree.destroy_if_clean(
        _repo_root(Path(worktree_path)),
        Path(worktree_path),
        ended_done=ended_done,
    )


@activity.defn(name="launch_run")
def launch_run(task_d: dict[str, Any], worktree_path: str) -> dict[str, Any]:
    task = _task_from_payload(task_d)
    executor = get_executor(task.executor)
    repo_root = _repo_root(Path(worktree_path))
    log_path = repo_root / ".autonomy" / "runs" / task.slug / "artifacts" / task.id / "run.log"
    result = sandbox.draccus_exec(
        executor.build_command(task, Path(worktree_path)),
        workspace=Path(worktree_path),
        gpus=task.gpus,
        log_path=log_path,
        timeout=task.timeout.total_seconds(),
    )
    parsed = executor.parse_result(result.exit_code, result.log_path)
    return {
        "exit_code": parsed.exit_code,
        "log_path": str(parsed.log_path),
        "summary": parsed.summary,
        "structured_result": parsed.structured_result,
        "stdout_tail": result.stdout_tail,
    }


@activity.defn(name="run_done_gate")
def run_done_gate(task_d: dict[str, Any], worktree_path: str) -> dict[str, Any]:
    from autonomy.done_gate import run_gate as _run_gate

    task = _task_from_payload(task_d)
    worktree = Path(worktree_path)
    repo_root = _repo_root(worktree)
    artifact_dir = repo_root / ".autonomy" / "runs" / task.slug / "artifacts" / task.id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    r = _run_gate(task, worktree, task.gate, repo_root, artifact_dir)
    return {
        "ok": r.ok,
        "artifacts": r.artifacts,
        "summary": r.summary,
        "checks": [dataclasses.asdict(c) for c in r.checks],
    }
