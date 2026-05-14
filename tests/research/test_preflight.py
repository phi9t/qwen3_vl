from __future__ import annotations

import pathlib

import research.adapters
import research.models
import research.preflight


def test_run_preflight_combines_generic_and_adapter_checks(
    tmp_path: pathlib.Path,
) -> None:
    adapter = research.adapters.load_adapter(
        "tests.research.fake_adapter:FakeAdapter"
    )
    db_path = tmp_path / "research.sqlite"
    db_path.write_text("", encoding="utf-8")
    context = research.models.TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        db_path=db_path,
    )
    intent = research.models.Intent(
        "fake", "unit-model", "cpu", "probe", "small", {"batch_size": 1}
    )

    result = research.preflight.run_preflight(adapter, intent, context)

    assert result.ok is True
    assert result.checks["adapter"] == "fake"
    assert result.checks["worktree"] == "ok"
    assert result.checks["artifact_dir"] == "ok"
    assert result.checks["fake"] == "ok"
