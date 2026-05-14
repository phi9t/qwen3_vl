"""Subprocess runners for scientific experiment trials."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import signal
import subprocess

import research.models


_TERMINATE_TIMEOUT_SECONDS = 10


@dataclasses.dataclass(frozen=True)
class TrialRunResult:
    """Result produced by executing a trial command."""

    returncode: int
    log_path: pathlib.Path
    progress: list[research.models.ProgressUpdate] = dataclasses.field(
        default_factory=list
    )


def terminate_process_group(proc: subprocess.Popen[str]) -> None:
    """Terminates a trial process group."""
    if proc.poll() is not None:
        return

    process_group_id = os.getpgid(proc.pid)
    os.killpg(process_group_id, signal.SIGTERM)
    try:
        proc.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process_group_id, signal.SIGKILL)
        proc.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)


def run_trial_command(
    adapter: research.models.ExperimentAdapter,
    command: research.models.TrialCommand,
    context: research.models.TrialContext,
) -> TrialRunResult:
    """Runs a trial command and captures logs and progress updates."""
    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = context.artifact_dir / "run.log"
    environment = dict(os.environ)
    environment.update(command.env)
    cwd = command.cwd or context.worktree
    progress: list[research.models.ProgressUpdate] = []

    proc = subprocess.Popen(
        command.argv,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            if proc.stdout is None:
                raise RuntimeError("Trial subprocess stdout was not captured.")
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                update = adapter.parse_progress(line)
                if update is not None:
                    progress.append(update)
        returncode = proc.wait()
    except BaseException:
        terminate_process_group(proc)
        raise

    return TrialRunResult(
        returncode=returncode,
        log_path=log_path,
        progress=progress,
    )
