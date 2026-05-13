from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


ARTIFACT_ROOT_ENV = "RESEARCH_ARTIFACT_ROOT"
DEFAULT_ARTIFACT_ROOT_REF = "~/.automata/research/experiments"


def default_artifact_root(repo_root: Path) -> Path:
    return Path.home() / ".automata" / "research" / "experiments"


@dataclass(frozen=True)
class ArtifactRoot:
    root: Path
    root_ref: str

    @classmethod
    def from_env(cls, repo_root: Path) -> "ArtifactRoot":
        configured = os.environ.get(ARTIFACT_ROOT_ENV)
        if configured:
            return cls(root=Path(configured).expanduser(), root_ref=f"${{{ARTIFACT_ROOT_ENV}}}")
        return cls(
            root=default_artifact_root(repo_root),
            root_ref=DEFAULT_ARTIFACT_ROOT_REF,
        )

    def ref_for(self, *parts: str) -> str:
        ref = PurePosixPath(*[str(part).strip("/") for part in parts])
        if ref.is_absolute():
            raise ValueError(f"artifact ref must be relative: {ref}")
        return str(ref)

    def resolve(self, ref: str) -> Path:
        path = PurePosixPath(ref)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"artifact ref must stay under artifact root: {ref}")
        return self.root / Path(*path.parts)


def reject_local_absolute_paths(payload: Any, *, _artifact_ref_context: bool = False) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            reject_local_absolute_paths(
                value,
                _artifact_ref_context=_artifact_ref_context or key == "artifact_refs",
            )
        return
    if isinstance(payload, list):
        for value in payload:
            reject_local_absolute_paths(
                value,
                _artifact_ref_context=_artifact_ref_context,
            )
        return
    if _artifact_ref_context and isinstance(payload, str) and Path(payload).is_absolute():
        raise ValueError(f"metadata contains absolute local path: {payload}")
