from __future__ import annotations

from pathlib import Path
from typing import Any

from research.models import FailureReason
from research.observability.log_parser import parse_metrics_and_symptoms
from research.observability.schema import TrialAnalysis


DEFAULT_ROOT_CAUSE = {
    FailureReason.NONE: "none",
    FailureReason.OOM: "capacity_exceeded",
    FailureReason.NCCL: "distributed_transport",
    FailureReason.LAUNCHER_ERROR: "launcher_failed",
    FailureReason.MISSING_FOOTER: "training_incomplete",
    FailureReason.CANCELLED: "operator_cancelled",
    FailureReason.UNKNOWN: "unknown",
    FailureReason.MODEL_LOAD_ERROR: "model_unavailable",
    FailureReason.DATASET_ERROR: "dataset_unavailable",
    FailureReason.HUB_OR_CACHE_ERROR: "model_snapshot_unavailable",
    FailureReason.TIMEOUT: "stalled_or_hung",
}


def _failure_value(failure_reason: FailureReason) -> str:
    return failure_reason.value


def _refine_root_cause(log_text: str, failure_reason: FailureReason) -> str:
    lower = log_text.lower()
    failure_value = _failure_value(failure_reason)

    if failure_value == "launcher_error":
        if "keyerror" in lower and "dataset" in lower:
            return "dataset_unavailable"
        if "filenotfounderror" in lower and "dataset" in lower:
            return "dataset_unavailable"
        if "modulenotfounderror" in lower or "importerror" in lower or "no module named" in lower:
            return "dependency_missing"
        if "snapshot" in lower or "huggingface" in lower or "hf_hub" in lower:
            return "model_snapshot_unavailable"
        if "failed to load model" in lower or "could not load model" in lower:
            return "model_unavailable"
    return DEFAULT_ROOT_CAUSE.get(failure_reason, "unknown")


def analyze_trial(
    *,
    trial_id: str,
    attempt: int,
    artifact_dir: Path,
    artifact_dir_ref: str | None = None,
    log_path: Path,
    return_code: int,
    expected_failure_reason: str | None,
    artifact_root_ref: str = "~/.automata/research/experiments",
    artifact_refs: dict[str, str] | None = None,
) -> TrialAnalysis:
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    metrics, symptoms = parse_metrics_and_symptoms(log_text, return_code)
    root_cause = _refine_root_cause(log_text, metrics.failure_reason)
    failure_value = _failure_value(metrics.failure_reason)
    expected_failure = expected_failure_reason == failure_value

    actions: list[dict[str, Any]] = []
    recommendations: list[str] = []
    if failure_value == "oom" and expected_failure:
        actions.append(
            {
                "action": "mark_capacity_boundary",
                "reason": "oom during expected capacity probe",
                "outcome": "continued",
            }
        )
        recommendations.append("Exclude this capacity point from sweep candidates.")
    elif failure_value != "none":
        recommendations.append("Inspect analysis evidence before continuing larger campaigns.")

    refs = artifact_refs or {}
    return TrialAnalysis(
        trial_id=trial_id,
        attempt=attempt,
        status=metrics.status,
        failure_reason=metrics.failure_reason,
        root_cause=root_cause,
        expected_failure=expected_failure,
        metrics=metrics,
        symptoms=symptoms,
        evidence_refs=[refs.get("full_log", str(log_path))],
        artifact_root_ref=artifact_root_ref,
        artifact_refs=refs,
        actions=actions,
        recommendations=recommendations,
        artifact_dir=artifact_dir_ref or artifact_dir.name,
    )
