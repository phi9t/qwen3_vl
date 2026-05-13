import json
from pathlib import Path

from research.observability.artifact_store import (
    ArtifactRoot,
    default_artifact_root,
    reject_local_absolute_paths,
)
from research.observability.artifacts import TrialArtifactStore
from research.observability.schema import SCHEMA_VERSION


def test_attempt_directory_is_incremented(tmp_path: Path) -> None:
    base = tmp_path / "experiments" / "runs" / "b200" / "probe" / "trial"
    first = TrialArtifactStore.create_next(base)
    second = TrialArtifactStore.create_next(base)

    assert first.attempt == 1
    assert second.attempt == 2
    assert first.attempt_dir.name == "attempt_001"
    assert second.attempt_dir.name == "attempt_002"
    assert (base / "latest_attempt.txt").read_text(encoding="utf-8") == "attempt_002\n"


def test_artifact_root_uses_env_and_produces_relative_refs(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "external-artifacts"
    monkeypatch.setenv("RESEARCH_ARTIFACT_ROOT", str(root))
    artifact_root = ArtifactRoot.from_env(repo_root=tmp_path / "repo")

    ref = artifact_root.ref_for(
        "logs", "b200", "probe", "trial", "attempt_001", "run.log"
    )
    resolved = artifact_root.resolve(ref)

    assert artifact_root.root == root
    assert ref == "logs/b200/probe/trial/attempt_001/run.log"
    assert resolved == root / ref
    assert not ref.startswith("/")


def test_default_artifact_root_is_user_research_dir(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    assert default_artifact_root(tmp_path) == home / ".automata" / "research" / "experiments"


def test_reject_local_absolute_paths_rejects_metadata_paths() -> None:
    payload = {"artifact_refs": {"full_log": "/data02/home/user/run.log"}}

    try:
        reject_local_absolute_paths(payload)
    except ValueError as exc:
        assert "absolute local path" in str(exc)
    else:
        raise AssertionError("expected absolute path rejection")


def test_write_json_adds_schema_version(tmp_path: Path) -> None:
    store = TrialArtifactStore.create_next(tmp_path / "trial")

    store.write_json("intent.json", {"trial_id": "x"})

    payload = json.loads((store.attempt_dir / "intent.json").read_text(encoding="utf-8"))
    assert payload == {"schema_version": SCHEMA_VERSION, "trial_id": "x"}


def test_append_jsonl_adds_schema_version(tmp_path: Path) -> None:
    store = TrialArtifactStore.create_next(tmp_path / "trial")

    store.append_jsonl("lifecycle.jsonl", {"stage": "trial_started"})
    store.append_jsonl("lifecycle.jsonl", {"stage": "process_exited"})

    lines = (store.attempt_dir / "lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["schema_version"] == SCHEMA_VERSION
    assert json.loads(lines[0])["stage"] == "trial_started"
    assert json.loads(lines[1])["stage"] == "process_exited"
