from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from autonomy.org.schema import parse_timeout


@workflow.defn
class SymphonyTrackerWorkflow:
    def __init__(self) -> None:
        self.active_task: str | None = None
        self.completed = 0
        self.blocked_ids: list[str] = []
        self.last_gate_summary: str | None = None
        self.pause_after_current_requested = False
        self.cancel_requested = False
        self.skipped_tasks: set[str] = set()
        self.rerun_requests: set[str] = set()

    def _now_iso(self) -> str:
        return workflow.now().isoformat()

    def _task_timeout(self, task: dict[str, Any]) -> timedelta:
        timeout = task.get("timeout", "6h")
        if isinstance(timeout, timedelta):
            return timeout
        if isinstance(timeout, str):
            return parse_timeout(timeout)
        if isinstance(timeout, (int, float)):
            return timedelta(seconds=float(timeout))
        raise TypeError(f"Unsupported timeout payload: {timeout!r}")

    def _artifacts(self, task: dict[str, Any], *values: object) -> list[str]:
        items: list[str] = []
        prefix = f".autonomy/runs/{task['slug']}/artifacts/{task['id']}/"
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                for raw in value.split(","):
                    item = raw.strip()
                    if not item:
                        continue
                    if prefix in item:
                        item = item.split(prefix, 1)[1]
                    items.append(item)
            elif isinstance(value, list):
                items.extend(self._artifacts(task, *value))
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    async def _apply_control_signals(self, tracker_path: str) -> None:
        for task_id in sorted(self.rerun_requests):
            await workflow.execute_activity(
                "org_transition",
                args=[tracker_path, task_id, "TODO", {"_note": "rerun requested"}],
                start_to_close_timeout=timedelta(seconds=30),
            )
            if task_id in self.blocked_ids:
                self.blocked_ids.remove(task_id)
        self.rerun_requests.clear()
        for task_id in sorted(self.skipped_tasks):
            await workflow.execute_activity(
                "org_transition",
                args=[tracker_path, task_id, "WONTFIX", {"_note": "skipped by signal"}],
                start_to_close_timeout=timedelta(seconds=30),
            )
            if task_id in self.blocked_ids:
                self.blocked_ids.remove(task_id)
        self.skipped_tasks.clear()

    async def _block(
        self,
        tracker_path: str,
        task: dict[str, Any],
        result: dict[str, Any],
        reason: str,
    ) -> None:
        summary = str(result.get("summary", "")).strip()
        self.last_gate_summary = summary or reason
        if task["id"] not in self.blocked_ids:
            self.blocked_ids.append(str(task["id"]))
        artifacts = self._artifacts(task, result.get("log_path"), result.get("artifacts", []))
        props = {
            "FINISHED": self._now_iso(),
            "EXIT_CODE": str(result.get("exit_code", 1)),
            "GATE_RESULT": f"fail:{reason}",
            "ARTIFACTS": ",".join(artifacts),
            "_note": summary or reason,
        }
        await workflow.execute_activity(
            "org_transition",
            args=[tracker_path, str(task["id"]), "BLOCKED", props],
            start_to_close_timeout=timedelta(seconds=30),
        )
        await workflow.execute_activity(
            "org_append_blocker",
            args=[
                tracker_path,
                str(task["id"]),
                reason,
                str(result.get("stdout_tail") or result.get("summary") or ""),
            ],
            start_to_close_timeout=timedelta(seconds=30),
        )

    @workflow.run
    async def run(self, tracker_path: str) -> dict[str, Any]:
        while not (self.cancel_requested or self.pause_after_current_requested):
            await self._apply_control_signals(tracker_path)
            picked = await workflow.execute_activity(
                "org_pick_next_ready",
                args=[tracker_path],
                start_to_close_timeout=timedelta(seconds=30),
            )
            if picked is None:
                break
            task = picked
            self.active_task = str(task["id"])
            await workflow.execute_activity(
                "org_transition",
                args=[
                    tracker_path,
                    str(task["id"]),
                    "IN-PROGRESS",
                    {
                        "OWNER": workflow.info().workflow_id,
                        "STARTED": self._now_iso(),
                        "_note": f"claimed by {workflow.info().workflow_id}",
                    },
                ],
                start_to_close_timeout=timedelta(seconds=30),
            )
            worktree = await workflow.execute_activity(
                "worktree_create",
                args=[str(task["slug"]), str(task["id"]), task.get("branch")],
                start_to_close_timeout=timedelta(minutes=2),
            )
            ended_done = False
            try:
                exec_result = await workflow.execute_activity(
                    "launch_run",
                    args=[task, worktree],
                    start_to_close_timeout=self._task_timeout(task),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
                if int(exec_result["exit_code"]) != 0:
                    await self._block(tracker_path, task, exec_result, reason="executor")
                    continue
                if task.get("gate") == "none":
                    self.last_gate_summary = "gate=none"
                    await workflow.execute_activity(
                        "org_transition",
                        args=[
                            tracker_path,
                            str(task["id"]),
                            "DONE",
                            {
                                "FINISHED": self._now_iso(),
                                "EXIT_CODE": "0",
                                "GATE_RESULT": "pass",
                                "ARTIFACTS": ",".join(self._artifacts(task, exec_result.get("log_path"))),
                                "_note": "gate=none",
                            },
                        ],
                        start_to_close_timeout=timedelta(seconds=30),
                    )
                    if str(task["id"]) in self.blocked_ids:
                        self.blocked_ids.remove(str(task["id"]))
                    ended_done = True
                else:
                    await workflow.execute_activity(
                        "org_transition",
                        args=[tracker_path, str(task["id"]), "AWAITING-GATE", {"_note": "executor exit 0"}],
                        start_to_close_timeout=timedelta(seconds=30),
                    )
                    gate = await workflow.execute_activity(
                        "run_done_gate",
                        args=[task, worktree],
                        start_to_close_timeout=timedelta(hours=1),
                        retry_policy=RetryPolicy(maximum_attempts=1),
                    )
                    self.last_gate_summary = str(gate.get("summary", ""))
                    if bool(gate.get("ok")):
                        await workflow.execute_activity(
                            "org_transition",
                            args=[
                                tracker_path,
                                str(task["id"]),
                                "DONE",
                                {
                                    "FINISHED": self._now_iso(),
                                    "EXIT_CODE": "0",
                                    "GATE_RESULT": "pass",
                                    "ARTIFACTS": ",".join(
                                        self._artifacts(task, exec_result.get("log_path"), gate.get("artifacts", []))
                                    ),
                                    "_note": self.last_gate_summary or "gate pass",
                                },
                            ],
                            start_to_close_timeout=timedelta(seconds=30),
                        )
                        if str(task["id"]) in self.blocked_ids:
                            self.blocked_ids.remove(str(task["id"]))
                        ended_done = True
                    else:
                        await self._block(tracker_path, task, gate, reason="gate")
            finally:
                await workflow.execute_activity(
                    "worktree_destroy_if_clean",
                    args=[worktree, ended_done],
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
                self.active_task = None
            self.completed += 1
        self.active_task = None
        return {
            "completed": self.completed,
            "phase": "paused" if self.pause_after_current_requested else "drained",
        }

    @workflow.signal
    def pause_after_current(self) -> None:
        self.pause_after_current_requested = True

    @workflow.signal
    def cancel(self) -> None:
        self.cancel_requested = True

    @workflow.signal
    def skip_task(self, task_id: str) -> None:
        self.skipped_tasks.add(task_id)

    @workflow.signal
    def rerun_task(self, task_id: str) -> None:
        self.rerun_requests.add(task_id)

    @workflow.query
    def status(self) -> dict[str, Any]:
        return {
            "active_task": self.active_task,
            "completed": self.completed,
            "blocked_ids": list(self.blocked_ids),
            "last_gate_summary": self.last_gate_summary,
        }
