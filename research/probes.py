"""Probe helpers for creating experiments from concrete adapters."""

from __future__ import annotations

import dataclasses
import pathlib
import typing

import research.db
import research.models


@dataclasses.dataclass(frozen=True)
class ProbeCreationResult:
    """Rows created for a probe request."""

    intent_ids: list[int]
    experiment_ids: list[int]


@dataclasses.dataclass(frozen=True)
class ProbeSelectionResult:
    """Intent ids selected or rejected after probe trials complete."""

    selected_intent_ids: list[int]
    rejected_intent_ids: list[int]
    skipped_experiment_ids: list[int]


def create_probe_experiments(
    db_path: pathlib.Path,
    *,
    adapter: research.models.ExperimentAdapter,
    request: research.models.ProbeRequest,
    artifact_root: pathlib.Path,
    adapter_ref: str,
    max_experiments: int = 0,
) -> ProbeCreationResult:
    """Persist adapter-generated probe intents as queued experiments."""
    research.db.init_db(db_path)
    if max_experiments < 0:
        raise ValueError("max_experiments must be non-negative")
    intent_ids = []
    experiment_ids = []
    for intent in adapter.generate_probe_intents(request):
        if max_experiments and len(experiment_ids) >= max_experiments:
            break
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


def select_probe_results(
    db_path: pathlib.Path,
    experiment_ids: list[int],
    *,
    objective: research.models.JsonDict,
) -> ProbeSelectionResult:
    """Select top probe intents by a numeric objective metric."""
    metric = str(objective.get("metric", "")).strip()
    if not metric:
        return ProbeSelectionResult([], [], list(experiment_ids))
    direction = str(objective.get("direction", "minimize")).strip().lower()
    if direction not in {"minimize", "maximize"}:
        raise ValueError(f"Unsupported objective direction: {direction!r}.")
    top_k = int(typing.cast(int | str, objective.get("top_k", 1)))
    if top_k < 1:
        raise ValueError("objective top_k must be at least 1.")

    candidates: list[tuple[float, int]] = []
    intent_ids_by_experiment: dict[int, int] = {}
    skipped_experiment_ids = []
    for experiment_id in experiment_ids:
        experiment = research.db.get_experiment(db_path, experiment_id)
        intent_id = int(experiment["intent_id"])
        intent_ids_by_experiment[experiment_id] = intent_id
        trial_run = research.db.get_latest_trial_run_for_experiment(
            db_path,
            experiment_id,
        )
        if trial_run is None:
            skipped_experiment_ids.append(experiment_id)
            continue
        value = _numeric_metric(trial_run["metrics_json"], metric)
        if trial_run["status"] == "succeeded" and value is not None:
            candidates.append((value, intent_id))
        else:
            research.db.transition_intent(
                db_path,
                intent_id,
                "rejected",
                score={
                    "metric": metric,
                    "status": trial_run["status"],
                    "failure": trial_run["failure_json"],
                },
            )

    reverse = direction == "maximize"
    ranked = sorted(candidates, key=lambda candidate: candidate[0], reverse=reverse)
    selected_intent_ids = [intent_id for _value, intent_id in ranked[:top_k]]
    selected_set = set(selected_intent_ids)
    for value, intent_id in ranked:
        status = "selected" if intent_id in selected_set else "rejected"
        research.db.transition_intent(
            db_path,
            intent_id,
            status,
            score={"metric": metric, "value": value},
        )
    rejected_intent_ids = [
        intent_id
        for experiment_id, intent_id in intent_ids_by_experiment.items()
        if experiment_id not in skipped_experiment_ids and intent_id not in selected_set
    ]

    return ProbeSelectionResult(
        selected_intent_ids=selected_intent_ids,
        rejected_intent_ids=rejected_intent_ids,
        skipped_experiment_ids=skipped_experiment_ids,
    )


def _numeric_metric(
    metrics: research.models.JsonDict,
    metric: str,
) -> float | None:
    value = metrics.get(metric)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
