from __future__ import annotations

import pathlib

import research.adapters
import research.models
import research.preflight


class MaskingAdapter:
    name = "masking"

    def preflight(
        self,
        intent: research.models.Intent,
        context: research.models.TrialContext,
    ) -> research.models.PreflightResult:
        return research.models.PreflightResult(
            ok=True,
            checks={"db": "ok", "uv": "ok", "adapter": "masked"},
            message="ok",
        )


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


def test_adapter_checks_cannot_mask_generic_failures(
    tmp_path: pathlib.Path,
) -> None:
    context = research.models.TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=tmp_path / "worktree",
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "missing.sqlite",
    )
    intent = research.models.Intent(
        "masking", "unit-model", "cpu", "probe", "small", {"batch_size": 1}
    )

    result = research.preflight.run_preflight(MaskingAdapter(), intent, context)

    assert result.ok is False
    assert result.checks["adapter"] == "masking"
    assert result.checks["db"] == "missing"


def test_path_creation_failure_returns_failed_preflight(
    tmp_path: pathlib.Path,
) -> None:
    worktree_file = tmp_path / "worktree"
    worktree_file.write_text("not a directory", encoding="utf-8")
    db_path = tmp_path / "research.sqlite"
    db_path.write_text("", encoding="utf-8")
    context = research.models.TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=worktree_file,
        artifact_dir=tmp_path / "artifacts",
        db_path=db_path,
    )
    intent = research.models.Intent(
        "fake", "unit-model", "cpu", "probe", "small", {"batch_size": 1}
    )

    result = research.preflight.run_preflight(MaskingAdapter(), intent, context)

    assert result.ok is False
    assert result.message == "preflight failed"
    assert str(result.checks["worktree"]).startswith("error:")
