"""Tests for direct-object probe creation helpers."""

from __future__ import annotations

import contextlib
import pathlib

import research.db
import research.models
import research.probes
import tests.research.fake_adapter


def test_create_probe_experiments_uses_concrete_adapter(
    tmp_path: pathlib.Path,
) -> None:
    """Probe helper should persist intents without dynamic adapter loading."""
    db_path = tmp_path / "research.sqlite"
    artifact_root = tmp_path / "artifacts"
    adapter = tests.research.fake_adapter.FakeAdapter()

    result = research.probes.create_probe_experiments(
        db_path,
        adapter=adapter,
        request=research.models.ProbeRequest(
            model="unit-model",
            profile="cpu",
            objective={"metric": "min"},
        ),
        artifact_root=artifact_root,
        adapter_ref=adapter.name,
    )

    assert result.intent_ids == [1]
    assert result.experiment_ids == [1]
    with contextlib.closing(research.db.connect(db_path)) as conn:
        intent = conn.execute("SELECT * FROM intents").fetchone()
        experiment = conn.execute("SELECT * FROM experiments").fetchone()

    assert intent["adapter"] == "fake"
    assert experiment["adapter"] == "fake"
    assert experiment["artifact_root"] == str(artifact_root)
    assert experiment["artifact_subdir"] == "fake/1"
