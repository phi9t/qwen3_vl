from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from temporalio.client import Client


TEMPORAL_DB_ENV = "RESEARCH_TEMPORAL_DB"
LOCAL_TEMPORAL_DB = Path(".automata") / "research" / "temporal" / "temporal.db"
DEFAULT_ADDRESS = "localhost:7233"
DEFAULT_TASK_QUEUE = "qwen3vl-local"


def default_temporal_db_path() -> Path:
    configured = os.environ.get(TEMPORAL_DB_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / LOCAL_TEMPORAL_DB


def ensure_temporal_cli() -> str:
    temporal = shutil.which("temporal")
    if temporal is None:
        raise RuntimeError(
            "temporal CLI not found on PATH; local runs require "
            "`temporal server start-dev --db-filename ~/.automata/research/temporal/temporal.db`"
        )
    return temporal


def build_start_dev_command(root: Path) -> list[str]:
    db_path = default_temporal_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return ["temporal", "server", "start-dev", "--db-filename", str(db_path)]


async def temporal_is_reachable(address: str = DEFAULT_ADDRESS) -> bool:
    try:
        await Client.connect(address)
    except Exception:
        return False
    return True


async def ensure_local_temporal_server(
    root: Path,
    address: str = DEFAULT_ADDRESS,
) -> subprocess.Popen | None:
    ensure_temporal_cli()
    if await temporal_is_reachable(address):
        return None

    proc = subprocess.Popen(
        build_start_dev_command(root),
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    for _ in range(120):
        if await temporal_is_reachable(address):
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"local Temporal server exited before becoming reachable at {address}"
            )
        await asyncio.sleep(1)

    terminate_local_temporal_server(proc)
    raise RuntimeError(f"local Temporal server did not become reachable at {address}")


def terminate_local_temporal_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return

    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    if proc.poll() is None:
        os.killpg(pgid, signal.SIGKILL)
