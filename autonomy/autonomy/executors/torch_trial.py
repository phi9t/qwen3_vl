from __future__ import annotations

from pathlib import Path

from autonomy.executors.base import Executor, RunResult, register
from autonomy.org.schema import OrgTask


class TorchTrialExecutor(Executor):
    name = "torch-trial"

    def build_command(self, task: OrgTask, worktree: Path) -> list[str]:
        return [
            "/bin/bash",
            "-lc",
            f"AUTONOMY_TASK_SLUG={task.slug} AUTONOMY_TASK_ID={task.id} python -m autonomy.executors._torch_trial_entry",
        ]

    def parse_result(self, exit_code: int, log_path: Path) -> RunResult:
        summary = ""
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                if line.strip():
                    summary = line.strip()
                    break
        return RunResult(
            exit_code=exit_code,
            log_path=str(log_path),
            summary=summary,
        )


register(TorchTrialExecutor())