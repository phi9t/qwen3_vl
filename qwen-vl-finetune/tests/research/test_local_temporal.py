import shutil
import sys
from pathlib import Path

import pytest

from research import runner
from research.local_temporal import (
    LOCAL_TEMPORAL_DB,
    build_start_dev_command,
    default_temporal_db_path,
    ensure_temporal_cli,
)


def test_build_start_dev_command_uses_file_backed_sqlite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    command = build_start_dev_command(tmp_path)

    assert command[:3] == ["temporal", "server", "start-dev"]
    assert "--db-filename" in command
    db_path = command[command.index("--db-filename") + 1]
    assert db_path == str(home / LOCAL_TEMPORAL_DB)


def test_build_start_dev_command_uses_env_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    configured = tmp_path / ".artifacts" / "temporal" / "temporal.db"
    monkeypatch.setenv("RESEARCH_TEMPORAL_DB", str(configured))

    assert default_temporal_db_path() == configured


def test_build_start_dev_command_creates_parent_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    command = build_start_dev_command(tmp_path)
    db_path = Path(command[command.index("--db-filename") + 1])

    assert db_path.parent.exists()


def test_ensure_temporal_cli_fails_fast_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="temporal CLI not found"):
        ensure_temporal_cli()


@pytest.mark.parametrize("command", ["probe", "select", "sweep", "summarize"])
def test_runner_routes_campaign_commands_to_local_implementations(
    monkeypatch,
    command: str,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(runner, f"command_{command}", lambda args: calls.append(command) or 0)

    monkeypatch.setattr(sys, "argv", ["runner", command, "--profile", "b200", "--dry-run"])

    assert runner.main() == 0
    assert calls == [command]


def test_runner_keeps_subagents_command_direct(monkeypatch) -> None:
    async def fail_if_called(profile: str, command: str, dry_run: bool) -> dict:
        raise AssertionError("subagents must not route through Temporal")

    monkeypatch.setattr(runner, "run_temporal_command", fail_if_called)
    monkeypatch.setattr(runner, "command_subagents", lambda args: 0)
    monkeypatch.setattr(sys, "argv", ["runner", "subagents", "--provider", "all"])

    assert runner.main() == 0
