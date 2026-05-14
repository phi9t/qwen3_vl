"""Tests for the research CLI command surface."""

from __future__ import annotations

import pathlib

import research.cli


def test_research_cli_rejects_probe_command(
    tmp_path: pathlib.Path,
) -> None:
    """Model-specific probe commands should live outside the generic CLI."""
    db_path = tmp_path / "research.sqlite"

    try:
        research.cli.main(["--db", str(db_path), "probe"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("generic research CLI should not expose probe")
