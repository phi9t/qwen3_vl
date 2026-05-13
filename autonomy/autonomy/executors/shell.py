from __future__ import annotations

from pathlib import Path

from autonomy.executors.base import Executor, RunResult, register
from autonomy.org.schema import OrgTask


class ShellExecutor(Executor):
    name = "shell"

    def build_command(self, task: OrgTask, worktree: Path) -> list[str]:
        body = " && ".join(task.acceptance_cmds or ["true"])
        return ["/bin/bash", "-lc", body]

    def parse_result(self, exit_code: int, log_path: Path) -> RunResult:
        lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
        summary = ""
        for line in lines[-5:]:
            if line.strip():
                summary = line.strip()
                break
        return RunResult(exit_code=exit_code, log_path=str(log_path), summary=summary)


register(ShellExecutor())
