"""Local Temporal development server helpers."""

from __future__ import annotations

import os
import pathlib
import shutil


TEMPORAL_DB_ENV = "RESEARCH_TEMPORAL_DB"
DEFAULT_ADDRESS = "localhost:7233"
DEFAULT_TASK_QUEUE = "research-local"


def default_temporal_db_path() -> pathlib.Path:
    """Return the local Temporal SQLite database path."""
    configured = os.environ.get(TEMPORAL_DB_ENV)
    if configured:
        return pathlib.Path(configured).expanduser()
    return pathlib.Path.home() / ".automata" / "research" / "temporal" / "temporal.db"


def ensure_temporal_cli() -> str:
    """Return the Temporal CLI path, or raise if it is unavailable."""
    temporal = shutil.which("temporal")
    if temporal is None:
        raise RuntimeError("temporal CLI not found on PATH")
    return temporal


def build_start_dev_command() -> list[str]:
    """Build the Temporal dev-server command and ensure DB parent exists."""
    db_path = default_temporal_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return ["temporal", "server", "start-dev", "--db-filename", str(db_path)]
