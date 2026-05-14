"""SQLite experiment database schema, connection helpers, and state transitions."""

from __future__ import annotations

import contextlib
import datetime
import json
import pathlib
import sqlite3
import typing

from research import models


JsonDict = dict[str, typing.Any]


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def connect(db_path: pathlib.Path) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled and Row factory."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _dump(value: JsonDict | None) -> str:
    return json.dumps(value or {}, sort_keys=True)


def _row(row: sqlite3.Row) -> JsonDict:
    result = dict(row)
    _json_keys = (
        "config_json",
        "objective_json",
        "score_json",
        "heartbeat_json",
        "metrics_json",
        "failure_json",
        "metadata_json",
    )
    for key in _json_keys:
        if key in result and isinstance(result[key], str):
            result[key] = json.loads(result[key] or "{}")
    return result


def init_db(db_path: pathlib.Path) -> None:
    """Create the experiment database schema if it does not exist."""
    with contextlib.closing(connect(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS intents (
              id INTEGER PRIMARY KEY,
              adapter TEXT NOT NULL,
              model TEXT NOT NULL,
              profile TEXT NOT NULL,
              phase TEXT NOT NULL,
              name TEXT NOT NULL,
              config_json TEXT NOT NULL,
              objective_json TEXT NOT NULL,
              source TEXT NOT NULL,
              status TEXT NOT NULL,
              score_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
              id INTEGER PRIMARY KEY,
              intent_id INTEGER NOT NULL REFERENCES intents(id),
              adapter TEXT NOT NULL,
              status TEXT NOT NULL,
              priority INTEGER NOT NULL,
              temporal_workflow_id TEXT NOT NULL,
              artifact_root TEXT NOT NULL,
              artifact_subdir TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trial_runs (
              id INTEGER PRIMARY KEY,
              experiment_id INTEGER NOT NULL REFERENCES experiments(id),
              attempt INTEGER NOT NULL,
              status TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              heartbeat_json TEXT NOT NULL,
              metrics_json TEXT NOT NULL,
              failure_json TEXT NOT NULL,
              report_path TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
              id INTEGER PRIMARY KEY,
              trial_run_id INTEGER NOT NULL REFERENCES trial_runs(id),
              kind TEXT NOT NULL,
              uri TEXT NOT NULL,
              size_bytes INTEGER,
              metadata_json TEXT NOT NULL
            );
            """
        )
        conn.commit()


def insert_intent(db_path: pathlib.Path, intent: models.Intent) -> int:
    """Insert a candidate intent and return its row id."""
    with contextlib.closing(connect(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO intents (
              adapter, model, profile, phase, name, config_json, objective_json,
              source, status, score_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent.adapter,
                intent.model,
                intent.profile,
                intent.phase,
                intent.name,
                _dump(intent.config),
                _dump(intent.objective),
                intent.source,
                "candidate",
                _dump({}),
                utc_now(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_intent(db_path: pathlib.Path, intent_id: int) -> JsonDict:
    """Return a JSON-decoded intent row by id."""
    with contextlib.closing(connect(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM intents WHERE id = ?", (intent_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"intent not found: {intent_id}")
    return _row(row)


def create_experiment(
    db_path: pathlib.Path,
    *,
    intent_id: int,
    adapter: str,
    artifact_root: str,
    artifact_subdir: str,
    priority: int = 0,
) -> int:
    """Create a queued experiment row and return its id."""
    now = utc_now()
    with contextlib.closing(connect(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO experiments (
              intent_id, adapter, status, priority, temporal_workflow_id,
              artifact_root, artifact_subdir, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent_id,
                adapter,
                "queued",
                priority,
                "",
                artifact_root,
                artifact_subdir,
                now,
                now,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_experiment(db_path: pathlib.Path, experiment_id: int) -> JsonDict:
    """Return a JSON-decoded experiment row by id."""
    with contextlib.closing(connect(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"experiment not found: {experiment_id}")
    return _row(row)


def get_trial_run(db_path: pathlib.Path, trial_run_id: int) -> JsonDict:
    """Return a JSON-decoded trial run row by id."""
    with contextlib.closing(connect(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM trial_runs WHERE id = ?", (trial_run_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"trial run not found: {trial_run_id}")
    return _row(row)


def create_trial_run(db_path: pathlib.Path, *, experiment_id: int, attempt: int) -> int:
    """Create a preflight trial_run row and return its id."""
    with contextlib.closing(connect(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO trial_runs (
              experiment_id, attempt, status, started_at, finished_at,
              heartbeat_json, metrics_json, failure_json, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                attempt,
                "preflight",
                utc_now(),
                None,
                _dump({}),
                _dump({}),
                _dump({}),
                "",
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def transition_intent(
    db_path: pathlib.Path,
    intent_id: int,
    status: str,
    *,
    score: JsonDict | None = None,
) -> None:
    """Update an intent's status and optional score."""
    with contextlib.closing(connect(db_path)) as conn:
        cur = conn.execute(
            "UPDATE intents SET status = ?, score_json = ? WHERE id = ?",
            (status, _dump(score), intent_id),
        )
        if cur.rowcount != 1:
            raise KeyError(f"intent not found: {intent_id}")
        conn.commit()


def transition_experiment(
    db_path: pathlib.Path,
    experiment_id: int,
    status: str,
    *,
    workflow_id: str | None = None,
) -> None:
    """Update an experiment's status and optional Temporal workflow id."""
    with contextlib.closing(connect(db_path)) as conn:
        if workflow_id is None:
            cur = conn.execute(
                "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), experiment_id),
            )
        else:
            cur = conn.execute(
                "UPDATE experiments SET status = ?, temporal_workflow_id = ?,"
                " updated_at = ? WHERE id = ?",
                (status, workflow_id, utc_now(), experiment_id),
            )
        if cur.rowcount != 1:
            raise KeyError(f"experiment not found: {experiment_id}")
        conn.commit()


def transition_trial_run(
    db_path: pathlib.Path,
    trial_run_id: int,
    status: str,
    *,
    heartbeat: JsonDict | None = None,
    metrics: JsonDict | None = None,
    failure: JsonDict | None = None,
    report_path: str = "",
) -> None:
    """Update a trial run's status, metrics, and optional report path."""
    finished_at = utc_now() if status in {"succeeded", "failed", "cancelled"} else None
    with contextlib.closing(connect(db_path)) as conn:
        cur = conn.execute(
            """
            UPDATE trial_runs
            SET status = ?, finished_at = ?, heartbeat_json = ?, metrics_json = ?,
                failure_json = ?, report_path = ?
            WHERE id = ?
            """,
            (
                status,
                finished_at,
                _dump(heartbeat),
                _dump(metrics),
                _dump(failure),
                report_path,
                trial_run_id,
            ),
        )
        if cur.rowcount != 1:
            raise KeyError(f"trial run not found: {trial_run_id}")
        conn.commit()
