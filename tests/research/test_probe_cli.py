"""Tests for the research probe command."""

from __future__ import annotations

import contextlib
import pathlib

import research.cli
import research.db


def test_probe_cli_creates_candidate_experiments(
    tmp_path: pathlib.Path,
) -> None:
    """Probe should persist adapter-generated intents as queued experiments."""
    db_path = tmp_path / "research.sqlite"
    artifact_root = tmp_path / "artifacts"

    rc = research.cli.main(
        [
            "--db",
            str(db_path),
            "probe",
            "--adapter",
            "tests.research.fake_adapter:FakeAdapter",
            "--model",
            "unit-model",
            "--profile",
            "cpu",
            "--artifact-root",
            str(artifact_root),
        ]
    )

    assert rc == 0
    with contextlib.closing(research.db.connect(db_path)) as conn:
        intent_count = conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]
        experiment = conn.execute("SELECT * FROM experiments").fetchone()

    assert intent_count == 1
    assert experiment["adapter"] == "tests.research.fake_adapter:FakeAdapter"
    assert experiment["status"] == "queued"
    assert experiment["artifact_root"] == str(artifact_root)
    assert experiment["artifact_subdir"] == "fake/1"
