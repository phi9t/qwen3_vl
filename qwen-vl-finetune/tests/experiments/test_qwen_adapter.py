"""Tests for the Qwen-VL research adapter."""

from __future__ import annotations

import pathlib

from research import adapters, models


def _load_qwen_adapter() -> models.ExperimentAdapter:
    return adapters.load_adapter("qwen_vl")


def test_qwen_probe_intents_use_request_model_and_profile() -> None:
    """Qwen probe intents should keep model/profile request-specific."""
    adapter = _load_qwen_adapter()

    intents = adapter.generate_probe_intents(
        models.ProbeRequest(
            model="Qwen/Qwen3-VL-8B-Instruct",
            profile="b200",
            objective={"metric": "throughput"},
        )
    )

    assert intents
    assert {intent.model for intent in intents} == {"Qwen/Qwen3-VL-8B-Instruct"}
    assert {intent.profile for intent in intents} == {"b200"}
    assert all("DATASETS" in intent.config for intent in intents)


def test_qwen_registry_adapter_loads() -> None:
    """The generic adapter registry should load the Qwen adapter."""
    adapter = _load_qwen_adapter()

    assert adapter.name == "qwen_vl"


def test_qwen_rejects_path_escape_profile() -> None:
    """Profiles must resolve inside the Qwen experiment profile directory."""
    adapter = _load_qwen_adapter()

    try:
        adapter.generate_probe_intents(models.ProbeRequest("model", "../outside"))
    except ValueError as exc:
        assert "Invalid profile name" in str(exc)
    else:
        raise AssertionError("expected invalid profile to raise ValueError")


def test_qwen_build_trial_uses_existing_launcher(
    tmp_path: pathlib.Path,
) -> None:
    """Qwen trials should call the existing shell launcher."""
    adapter = _load_qwen_adapter()
    intent = adapter.generate_probe_intents(
        models.ProbeRequest("Qwen/Qwen3-VL-8B-Instruct", "b200")
    )[0]
    context = models.TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=pathlib.Path(__file__).resolve().parents[2],
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "research.sqlite",
    )

    command = adapter.build_trial(intent, context)

    assert command.argv[-1].endswith("scripts/sft_qwen3_8b.sh")
    assert command.env["MODEL_NAME_OR_PATH"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert command.env["OUTPUT_DIR"].endswith("outputs/1/attempt-1")


def test_qwen_analysis_requires_validation_metric(
    tmp_path: pathlib.Path,
) -> None:
    """VRAM telemetry alone should not mark a training trial successful."""
    adapter = _load_qwen_adapter()
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "run.log").write_text("peak_vram_mb: 123\n", encoding="utf-8")
    context = models.TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=pathlib.Path(__file__).resolve().parents[2],
        artifact_dir=artifact_dir,
        db_path=tmp_path / "research.sqlite",
    )

    report = adapter.analyze_result(context)

    assert report.status == "failed"
    assert report.failure["reason"] == "missing_metrics"
