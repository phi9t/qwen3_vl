from pathlib import Path

from research.subagents import AgentProvider, SubagentTask, build_command, discover_providers


def test_build_coco_command() -> None:
    task = SubagentTask(
        provider=AgentProvider.COCO,
        worktree="b200-models",
        workspace=Path("/repo/qwen3_vl"),
        prompt="Implement Task 1. Do not revert unrelated changes.",
    )

    command = build_command(task)

    assert command[:4] == ["coco", "-p", "-y", "--worktree"]
    assert command[4] == "b200-models"
    assert command[-1].startswith("Implement Task 1")


def test_build_cursor_agent_command() -> None:
    task = SubagentTask(
        provider=AgentProvider.CURSOR_AGENT,
        worktree="b200-metrics",
        workspace=Path("/repo/qwen3_vl"),
        prompt="Implement Task 3. Do not revert unrelated changes.",
    )

    command = build_command(task)

    assert command[:4] == ["agent", "-p", "--trust", "--workspace"]
    assert command[4] == "/repo/qwen3_vl"
    assert "-w" in command
    assert command[-1].startswith("Implement Task 3")


def test_discover_providers_uses_supplied_which() -> None:
    providers = discover_providers(
        which=lambda name: f"/bin/{name}"
        if name in {"coco", "agent"}
        else None
    )

    assert providers[AgentProvider.COCO] == "/bin/coco"
    assert providers[AgentProvider.CURSOR_AGENT] == "/bin/agent"
