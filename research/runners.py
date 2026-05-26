"""Subprocess runners for scientific experiment trials."""

from __future__ import annotations

import codecs
import dataclasses
import os
import pathlib
import selectors
import signal
import subprocess
import typing
from collections.abc import Callable

import research.models


_TERMINATE_TIMEOUT_SECONDS = 10
_STDOUT_POLL_SECONDS = 0.1
ProgressCallback = Callable[[research.models.ProgressUpdate], None]


@dataclasses.dataclass(frozen=True)
class TrialRunResult:
    """Result produced by executing a trial command."""

    returncode: int
    log_path: pathlib.Path
    progress: list[research.models.ProgressUpdate] = dataclasses.field(
        default_factory=list
    )


def terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Terminates a trial process group."""
    process_group_id = os.getpgid(proc.pid)
    _terminate_process_group_id(process_group_id)
    if proc.poll() is not None:
        return
    try:
        proc.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _terminate_process_group_id(process_group_id, signal.SIGKILL)
        proc.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)


def _terminate_process_group_id(
    process_group_id: int,
    sig: signal.Signals = signal.SIGTERM,
) -> None:
    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        return


def _capture_output(
    proc: subprocess.Popen[bytes],
    adapter: research.models.ExperimentAdapter,
    log_file: typing.TextIO,
    progress_callback: ProgressCallback | None = None,
) -> list[research.models.ProgressUpdate]:
    if proc.stdout is None:
        raise RuntimeError("Trial subprocess stdout was not captured.")

    progress: list[research.models.ProgressUpdate] = []
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    file_descriptor = proc.stdout.fileno()

    def process_text(text: str) -> None:
        nonlocal pending
        pending += text
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            _record_line(
                adapter,
                log_file,
                progress,
                f"{line}\n",
                progress_callback,
            )

    stdout_open = True
    try:
        while stdout_open:
            for _key, _mask in selector.select(_STDOUT_POLL_SECONDS):
                chunk = os.read(file_descriptor, 8192)
                if not chunk:
                    stdout_open = False
                    break
                process_text(decoder.decode(chunk))
            if proc.poll() is not None:
                while selector.select(0):
                    chunk = os.read(file_descriptor, 8192)
                    if not chunk:
                        stdout_open = False
                        break
                    process_text(decoder.decode(chunk))
                stdout_open = False
    finally:
        remaining = pending + decoder.decode(b"", final=True)
        if remaining:
            _record_line(adapter, log_file, progress, remaining, progress_callback)
        selector.unregister(proc.stdout)
        selector.close()
        proc.stdout.close()
    return progress


def _record_line(
    adapter: research.models.ExperimentAdapter,
    log_file: typing.TextIO,
    progress: list[research.models.ProgressUpdate],
    line: str,
    progress_callback: ProgressCallback | None = None,
) -> None:
    log_file.write(line)
    log_file.flush()
    update = adapter.parse_progress(line)
    if update is not None:
        progress.append(update)
        if progress_callback is not None:
            progress_callback(update)


def run_trial_command(
    adapter: research.models.ExperimentAdapter,
    command: research.models.TrialCommand,
    context: research.models.TrialContext,
    progress_callback: ProgressCallback | None = None,
) -> TrialRunResult:
    """Runs a trial command and captures logs and progress updates."""
    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = context.artifact_dir / "run.log"
    environment = dict(os.environ)
    environment.update(command.env)
    cwd = command.cwd or context.worktree

    proc = subprocess.Popen(
        command.argv,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process_group_id = os.getpgid(proc.pid)
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = _capture_output(proc, adapter, log_file, progress_callback)
        returncode = proc.wait()
        _terminate_process_group_id(process_group_id)
    except BaseException:
        terminate_process_group(proc)
        raise

    return TrialRunResult(
        returncode=returncode,
        log_path=log_path,
        progress=progress,
    )
