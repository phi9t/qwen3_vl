"""Probe helpers for creating experiments from concrete adapters."""

from __future__ import annotations

import dataclasses
import pathlib

import research.db
import research.models


@dataclasses.dataclass(frozen=True)
class ProbeCreationResult:
    """Rows created for a probe request."""

    intent_ids: list[int]
    experiment_ids: list[int]


def create_probe_experiments(
    db_path: pathlib.Path,
    *,
    adapter: research.models.ExperimentAdapter,
    request: research.models.ProbeRequest,
    artifact_root: pathlib.Path,
    adapter_ref: str,
) -> ProbeCreationResult:
    """Persist adapter-generated probe intents as queued experiments."""
    research.db.init_db(db_path)
    intent_ids = []
    experiment_ids = []
    for intent in adapter.generate_probe_intents(request):
        intent_id = research.db.insert_intent(db_path, intent)
        experiment_id = research.db.create_experiment(
            db_path,
            intent_id=intent_id,
            adapter=adapter_ref,
            artifact_root=str(artifact_root),
            artifact_subdir=f"{adapter.name}/{intent_id}",
        )
        intent_ids.append(intent_id)
        experiment_ids.append(experiment_id)
    return ProbeCreationResult(intent_ids=intent_ids, experiment_ids=experiment_ids)
