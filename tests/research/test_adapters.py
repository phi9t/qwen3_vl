from __future__ import annotations

import pathlib

import research.adapters
import research.models


def test_load_adapter_from_module_class_path() -> None:
    adapter = research.adapters.load_adapter("tests.research.fake_adapter:FakeAdapter")

    assert adapter.name == "fake"


def test_load_adapter_from_file_class_path() -> None:
    adapter = research.adapters.load_adapter("tests/research/fake_adapter.py:FakeAdapter")

    assert adapter.name == "fake"


def test_fake_adapter_generates_intents_and_trial_command(
    tmp_path: pathlib.Path,
) -> None:
    adapter = research.adapters.load_adapter("tests.research.fake_adapter:FakeAdapter")
    request = research.models.ProbeRequest(
        model="unit-model",
        profile="cpu",
        objective={"metric": "min"},
    )
    intents = adapter.generate_probe_intents(request)
    context = research.models.TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "research.sqlite",
    )

    command = adapter.build_trial(intents[0], context)

    assert intents[0].config == {"batch_size": 1}
    assert command.argv[:2] == ["python", "-c"]
