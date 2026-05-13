from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.observability.artifact_store import reject_local_absolute_paths
from research.observability.schema import SCHEMA_VERSION


@dataclass(frozen=True)
class TrialArtifactStore:
    attempt_dir: Path
    attempt: int

    @classmethod
    def create_next(cls, trial_dir: Path) -> "TrialArtifactStore":
        trial_dir.mkdir(parents=True, exist_ok=True)
        attempts = [
            int(path.name.removeprefix("attempt_"))
            for path in trial_dir.glob("attempt_*")
            if path.is_dir() and path.name.removeprefix("attempt_").isdigit()
        ]
        attempt = max(attempts, default=0) + 1
        attempt_dir = trial_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir()
        (trial_dir / "latest_attempt.txt").write_text(
            f"{attempt_dir.name}\n",
            encoding="utf-8",
        )
        return cls(attempt_dir=attempt_dir, attempt=attempt)

    def _versioned(self, payload: dict[str, Any]) -> dict[str, Any]:
        versioned = {"schema_version": SCHEMA_VERSION}
        versioned.update(payload)
        reject_local_absolute_paths(versioned)
        return versioned

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.attempt_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._versioned(payload), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def append_jsonl(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.attempt_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._versioned(payload), sort_keys=True) + "\n")
        return path

    def path(self, name: str) -> Path:
        return self.attempt_dir / name

    def lifecycle(self, stage: str, **extra: Any) -> None:
        payload = {"stage": stage}
        payload.update(extra)
        self.append_jsonl("lifecycle.jsonl", payload)

    def action(self, action: str, *, reason: str, outcome: str, **extra: Any) -> None:
        payload = {"action": action, "reason": reason, "outcome": outcome}
        payload.update(extra)
        self.append_jsonl("actions.jsonl", payload)
