from __future__ import annotations

import pathlib
import subprocess

import pytest

import research.coco_cli as coco_cli


def test_build_coco_command_with_workspace(tmp_path: pathlib.Path) -> None:
    command = coco_cli.build_coco_command(
        "/usr/bin/coco",
        "Return JSON",
        "deep-research-search",
        tmp_path,
    )

    assert command == [
        "/usr/bin/coco",
        "-p",
        "-y",
        "--worktree",
        "deep-research-search",
        "--workspace",
        str(tmp_path),
        "Return JSON",
    ]


def test_build_coco_command_without_workspace() -> None:
    command = coco_cli.build_coco_command(
        "/usr/bin/coco",
        "Return JSON",
        "deep-research-plan",
        None,
    )

    assert command == [
        "/usr/bin/coco",
        "-p",
        "-y",
        "--worktree",
        "deep-research-plan",
        "Return JSON",
    ]


def test_resolve_coco_path_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_COCO_BIN", "/opt/bin/coco")
    monkeypatch.delenv("PATH", raising=False)

    assert coco_cli.resolve_coco_path() == "/opt/bin/coco"


def test_run_coco_missing_workspace_fails(tmp_path: pathlib.Path) -> None:
    missing_workspace = tmp_path / "missing"

    with pytest.raises(coco_cli.CocoExecutionError, match="workspace"):
        coco_cli.run_coco(
            prompt="Return JSON",
            phase="search",
            worktree="deep-research-search",
            workspace=missing_workspace,
            runner=subprocess.run,
            coco_path="/usr/bin/coco",
        )


def test_run_coco_nonzero_exit_includes_error_tail() -> None:
    def failing_runner(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv,
            returncode=2,
            stdout="some stdout\npartial details",
            stderr="first line\nlast error line",
        )

    with pytest.raises(coco_cli.CocoExecutionError) as exc_info:
        coco_cli.run_coco(
            prompt="Return JSON",
            phase="search",
            worktree="deep-research-search",
            runner=failing_runner,
            coco_path="/usr/bin/coco",
        )

    assert exc_info.value.phase == "search"
    assert exc_info.value.returncode == 2
    assert "last error line" in str(exc_info.value)


def test_run_coco_returns_stdout_and_returncode(tmp_path: pathlib.Path) -> None:
    def passing_runner(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout='{"ok": true}',
            stderr="",
        )

    result = coco_cli.run_coco(
        prompt="Return JSON",
        phase="subtopics",
        worktree="deep-research-plan",
        workspace=tmp_path,
        runner=passing_runner,
        coco_path="/usr/bin/coco",
    )

    assert result.stdout == '{"ok": true}'
    assert result.returncode == 0
