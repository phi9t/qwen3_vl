"""Helpers for research artifact paths."""

from __future__ import annotations

import os
import pathlib


ARTIFACT_ROOT_ENV = "RESEARCH_ARTIFACT_ROOT"


def default_artifact_root() -> pathlib.Path:
    """Return the configured or default research artifact root."""
    configured = os.environ.get(ARTIFACT_ROOT_ENV)
    if configured:
        return pathlib.Path(configured).expanduser()
    return pathlib.Path.home() / ".automata" / "research" / "artifacts"


def attempt_dir(
    root: pathlib.Path,
    *,
    adapter: str,
    experiment_id: int,
    attempt: int,
) -> pathlib.Path:
    """Create and return an isolated artifact directory for one attempt."""
    path = root / adapter / str(experiment_id) / f"attempt-{attempt}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_artifact(root: pathlib.Path, path: pathlib.Path) -> str:
    """Return the artifact path relative to root, rejecting external paths."""
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    try:
        relative_path = path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Artifact path {path} is outside artifact root {root}."
        ) from exc
    return relative_path.as_posix()
