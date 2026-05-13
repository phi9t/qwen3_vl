from __future__ import annotations

import json
from pathlib import Path

from autonomy.executors.base import Executor, RunResult, register
from autonomy.org.schema import OrgTask


class ClaudeCodeExecutor(Executor):
    name = "claude-code"

    def build_command(self, task: OrgTask, worktree: Path) -> list[str]:
        return [
            "claude",
            "-p",
            self._render_prompt(task),
            "--output-format=stream-json",
            "--permission-mode=acceptEdits",
        ]

    def parse_result(self, exit_code: int, log_path: Path) -> RunResult:
        if not log_path.exists():
            return RunResult(
                exit_code=exit_code,
                log_path=str(log_path),
                summary="no result event",
            )

        result_event = None
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                result_event = event
                break

        if result_event is None:
            return RunResult(
                exit_code=exit_code,
                log_path=str(log_path),
                summary="no result event",
            )

        is_error = result_event.get("is_error", False)
        if is_error:
            error_msg = result_event.get("error", "unknown error")
            summary = f"error: {error_msg}"
        else:
            result_text = result_event.get("result", "")
            summary = f"ok: {result_text[:120]}"

        return RunResult(
            exit_code=exit_code,
            log_path=str(log_path),
            summary=summary,
            structured_result=result_event,
        )

    @staticmethod
    def _render_prompt(task: OrgTask) -> str:
        parts = [f"Goal: {task.goal}"]
        if task.constraints:
            parts.append("Constraints:")
            for c in task.constraints:
                parts.append(f"- {c}")
        if task.acceptance_cmds:
            parts.append("Acceptance:")
            for a in task.acceptance_cmds:
                parts.append(f"- {a}")
        return "\n".join(parts)


register(ClaudeCodeExecutor())