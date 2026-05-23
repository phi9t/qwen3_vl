"""Tests for research artifact path helpers."""

from __future__ import annotations

import pathlib

import pytest

import research.artifacts


def test_default_artifact_root_uses_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("RESEARCH_ARTIFACT_ROOT", raising=False)

    assert (
        research.artifacts.default_artifact_root()
        == home / ".automata" / "research" / "artifacts"
    )


def test_attempt_dir_creates_isolated_directory(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "artifacts"
    path = research.artifacts.attempt_dir(
        root,
        adapter="fake",
        experiment_id=7,
        attempt=2,
    )

    assert path == root / "fake" / "7" / "attempt-2"
    assert path.exists()


def test_attempt_dir_from_subdir_creates_stored_artifact_path(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "artifacts"
    path = research.artifacts.attempt_dir_from_subdir(
        root,
        artifact_subdir="fake/7",
        attempt=2,
    )

    assert path == root / "fake" / "7" / "attempt-2"
    assert path.exists()


@pytest.mark.parametrize("adapter", ["../escape", "/tmp/escape", ".", "..", ""])
def test_attempt_dir_rejects_unsafe_adapter_segment(
    tmp_path: pathlib.Path,
    adapter: str,
) -> None:
    with pytest.raises(ValueError, match="single safe path segment"):
        research.artifacts.attempt_dir(
            tmp_path / "artifacts",
            adapter=adapter,
            experiment_id=7,
            attempt=2,
        )


@pytest.mark.parametrize("artifact_subdir", ["../escape", "/tmp/escape", ".", "..", ""])
def test_attempt_dir_from_subdir_rejects_unsafe_relative_path(
    tmp_path: pathlib.Path,
    artifact_subdir: str,
) -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        research.artifacts.attempt_dir_from_subdir(
            tmp_path / "artifacts",
            artifact_subdir=artifact_subdir,
            attempt=2,
        )


def test_relative_artifact_rejects_external_path(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "artifacts"
    inside = root / "fake" / "run.log"
    inside.parent.mkdir(parents=True)
    inside.write_text("log", encoding="utf-8")

    assert research.artifacts.relative_artifact(root, inside) == "fake/run.log"


def test_relative_artifact_raises_for_external_path(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "artifacts"
    external = tmp_path / "external" / "run.log"
    external.parent.mkdir(parents=True)
    external.write_text("log", encoding="utf-8")

    with pytest.raises(ValueError, match="outside artifact root"):
        research.artifacts.relative_artifact(root, external)
