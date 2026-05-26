"""Subprocess boundary for running the COCO CLI."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import subprocess
from collections.abc import Callable


COCO_BIN_ENV = "RESEARCH_COCO_BIN"
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclasses.dataclass(frozen=True)
class CocoResult:
    """Result payload from a COCO subprocess invocation."""

    argv: list[str]
    stdout: str
    stderr: str
    returncode: int


class CocoExecutionError(RuntimeError):
    """Raised when COCO cannot be found or exits with a non-zero status."""

    def __init__(
        self,
        *,
        phase: str,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        message: str = "",
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def resolve_coco_path() -> str:
    """Resolve the COCO executable path from config or PATH."""
    override = os.environ.get(COCO_BIN_ENV)
    if override:
        override_path = pathlib.Path(override)
        if not override_path.is_file() or not os.access(override, os.X_OK):
            raise CocoExecutionError(
                phase="discovery",
                message=(
                    f"Invalid {COCO_BIN_ENV} override: {override} must exist and be executable"
                ),
            )
        return override

    path = shutil.which("coco")
    if path is None:
        raise CocoExecutionError(
            phase="discovery",
            message="coco executable not found; set RESEARCH_COCO_BIN",
        )
    return path


def build_coco_command(
    coco_path: str,
    prompt: str,
    worktree: str,
    workspace: pathlib.Path | None,
) -> list[str]:
    """Build COCO CLI command arguments using the fixed required shape."""
    command = [coco_path, "-p", "-y", "--worktree", worktree]
    if workspace is not None:
        command.extend(["--workspace", str(workspace)])
    command.append(prompt)
    return command


def run_coco(
    prompt: str,
    phase: str,
    worktree: str,
    workspace: pathlib.Path | None = None,
    runner: Runner = subprocess.run,
    coco_path: str | None = None,
) -> CocoResult:
    """Execute a COCO prompt and return its captured output."""
    if workspace is not None and not workspace.exists():
        raise CocoExecutionError(
            phase=phase,
            message=f"workspace does not exist: {workspace}",
        )

    resolved_coco_path = coco_path or resolve_coco_path()
    argv = build_coco_command(
        resolved_coco_path,
        prompt,
        worktree,
        workspace,
    )
    try:
        completed = runner(
            argv,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise CocoExecutionError(
            phase=phase,
            returncode=None,
            stdout="",
            stderr="",
            message=(
                f"COCO phase {phase} failed to launch ({type(exc).__name__}): {exc}"
            ),
        ) from exc

    if completed.returncode != 0:
        tail = _tail(completed.stderr) or _tail(completed.stdout)
        raise CocoExecutionError(
            phase=phase,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            message=(
                f"COCO command failed during phase {phase}"
                f" (return code {completed.returncode}): {tail}"
            ),
        )

    return CocoResult(
        argv=argv,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def _tail(text: str, limit: int = 500) -> str:
    """Keep only the trailing line-limited output for error messages."""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[-limit:]
