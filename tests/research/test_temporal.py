"""Tests for local Temporal development helpers."""

from __future__ import annotations

import pathlib

from research import temporal


def test_default_temporal_db_path_uses_home(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("RESEARCH_TEMPORAL_DB", raising=False)

    assert temporal.default_temporal_db_path() == (
        home / ".automata" / "research" / "temporal" / "temporal.db"
    )


def test_build_start_dev_command_creates_parent(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    db = tmp_path / "temporal" / "temporal.db"
    monkeypatch.setenv("RESEARCH_TEMPORAL_DB", str(db))

    command = temporal.build_start_dev_command()

    assert command == [
        "temporal",
        "server",
        "start-dev",
        "--db-filename",
        str(db),
    ]
    assert db.parent.exists()
