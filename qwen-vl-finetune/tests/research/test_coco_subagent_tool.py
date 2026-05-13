from pathlib import Path

import pytest

from research import coco_subagent
from research.subagents import AgentProvider


def test_build_coco_task_uses_standard_prompt_when_prompt_omitted() -> None:
    task = coco_subagent.build_coco_task(
        prompt=None,
        task_name="Task 1",
        files=["qwen-vl-finetune/research/"],
        test_command="pytest tests/research -v",
        worktree="coco-task-1",
        workspace=Path("/repo/qwen3_vl"),
    )

    assert task.provider == AgentProvider.COCO
    assert task.worktree == "coco-task-1"
    assert "Implement Task 1" in task.prompt
    assert "qwen-vl-finetune/research/" in task.prompt
    assert "pytest tests/research -v" in task.prompt


def test_main_prints_coco_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        coco_subagent,
        "discover_providers",
        lambda: {AgentProvider.COCO: "/bin/coco"},
    )

    result = coco_subagent.main(
        [
            "--worktree",
            "coco-task",
            "--workspace",
            "/repo/qwen3_vl",
            "Implement the bounded task.",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert output.startswith("coco -p -y --worktree coco-task")
    assert "Implement the bounded task." in output


def test_main_fails_fast_when_coco_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(coco_subagent, "discover_providers", lambda: {})

    with pytest.raises(SystemExit, match="coco CLI not found"):
        coco_subagent.main([])
