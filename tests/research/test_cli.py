"""Tests for the research command-line interface."""

from __future__ import annotations

import pathlib
import subprocess
import sys

from research import cli


def test_db_init_command_creates_sqlite(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "research.sqlite"

    rc = cli.main(["--db", str(db_path), "db", "init"])

    assert rc == 0
    assert db_path.exists()


def test_status_command_handles_empty_db(tmp_path: pathlib.Path, capsys) -> None:
    db_path = tmp_path / "research.sqlite"
    cli.main(["--db", str(db_path), "db", "init"])

    rc = cli.main(["--db", str(db_path), "status"])

    assert rc == 0
    assert "No experiments" in capsys.readouterr().out


def test_cli_module_runs_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "research.cli", "--help"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert "Scientific experiment manager" in result.stdout
