from __future__ import annotations

import json
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from autonomy.executors.base import REGISTRY, RunResult
from autonomy.org.schema import OrgTask


def _make_task(**overrides) -> OrgTask:
    defaults = {
        "id": "T01",
        "executor": "claude-code",
        "gate": "tests+typecheck+gpt2-smoke",
        "depends": frozenset(),
        "timeout": timedelta(hours=6),
        "gpus": 1,
        "branch": None,
        "goal": "rename _DATASET_CACHE to _dataset_cache",
        "constraints": ["do not modify tests/"],
        "acceptance_cmds": ["pytest -x"],
        "slug": "test-run",
        "state": "TODO",
        "position": 0,
    }
    defaults.update(overrides)
    return OrgTask(**defaults)


class TestClaudeCodeExecutor:
    def test_build_command_includes_claude_and_prompt(self):
        from autonomy.executors.claude_code import ClaudeCodeExecutor

        executor = ClaudeCodeExecutor()
        task = _make_task()
        cmd = executor.build_command(task, Path("/tmp/worktree"))

        assert cmd[0] == "claude"
        assert "-p" in cmd
        prompt_idx = cmd.index("-p") + 1
        prompt = cmd[prompt_idx]
        assert "Goal:" in prompt
        assert "rename _DATASET_CACHE" in prompt
        assert "Constraints:" in prompt
        assert "do not modify tests/" in prompt
        assert "Acceptance:" in prompt
        assert "pytest -x" in prompt
        assert "--output-format=stream-json" in cmd
        assert "--permission-mode=acceptEdits" in cmd

    def test_parse_result_with_result_event(self):
        from autonomy.executors.claude_code import ClaudeCodeExecutor

        executor = ClaudeCodeExecutor()
        log_lines = [
            json.dumps(
                {
                    "type": "system",
                    "subtype": "init",
                    "message": "starting",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"text": "ok"}]},
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "is_error": False,
                    "result": "renamed _DATASET_CACHE to _dataset_cache",
                }
            ),
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write("\n".join(log_lines))
            log_path = Path(f.name)

        try:
            result = executor.parse_result(0, log_path)
            assert result.exit_code == 0
            assert result.summary.startswith("ok:")
            assert "_dataset_cache" in result.summary
            assert result.structured_result is not None
            assert result.structured_result["type"] == "result"
            assert result.structured_result["is_error"] is False
        finally:
            log_path.unlink(missing_ok=True)

    def test_parse_result_no_result_event(self):
        from autonomy.executors.claude_code import ClaudeCodeExecutor

        executor = ClaudeCodeExecutor()
        log_lines = [
            json.dumps(
                {
                    "type": "system",
                    "subtype": "init",
                    "message": "starting",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"text": "working"}]},
                }
            ),
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write("\n".join(log_lines))
            log_path = Path(f.name)

        try:
            result = executor.parse_result(1, log_path)
            assert result.summary == "no result event"
            assert result.structured_result is None
        finally:
            log_path.unlink(missing_ok=True)


class TestTorchTrialExecutor:
    def test_build_command_sets_env_vars(self):
        from autonomy.executors.torch_trial import TorchTrialExecutor

        executor = TorchTrialExecutor()
        task = _make_task(
            id="T03",
            slug="my-run",
            executor="torch-trial",
        )
        cmd = executor.build_command(task, Path("/tmp/worktree"))

        assert cmd[0] == "/bin/bash"
        assert cmd[1] == "-lc"
        assert "AUTONOMY_TASK_SLUG=my-run" in cmd[2]
        assert "AUTONOMY_TASK_ID=T03" in cmd[2]
        assert "python -m autonomy.executors._torch_trial_entry" in cmd[2]

    def test_registers_on_import(self):
        # Force re-import to test registration
        import importlib
        import autonomy.executors.torch_trial as tt

        importlib.reload(tt)
        assert "torch-trial" in REGISTRY
        assert REGISTRY["torch-trial"].name == "torch-trial"