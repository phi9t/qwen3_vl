from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research.models import FailureReason, TrialMetrics
from research.observability.redaction import redact_mapping


SCHEMA_VERSION = 1


def _failure_reason_value(value: FailureReason | str) -> str:
    if isinstance(value, FailureReason):
        return value.value
    return str(value)


def metrics_to_payload(metrics: TrialMetrics) -> dict[str, Any]:
    return {
        "status": metrics.status,
        "val_loss": metrics.val_loss,
        "peak_vram_mb": metrics.peak_vram_mb,
        "throughput_steps_per_sec": metrics.throughput_steps_per_sec,
        "failure_reason": metrics.failure_reason.value,
    }


@dataclass(frozen=True)
class TrialIntent:
    trial_id: str
    attempt: int
    profile: str
    phase: str
    trial: str
    research_intent: str
    planned_env: dict[str, str]
    success_criteria: dict[str, Any] = field(default_factory=dict)
    expected_failure_reason: str | None = None
    campaign_id: str | None = None
    workflow_id: str | None = None

    @classmethod
    def from_spec_payload(
        cls,
        *,
        profile: str,
        phase: str,
        trial: str,
        env: dict[str, str],
        attempt: int,
        research_intent: str,
        expected_failure_reason: str | None = None,
        success_criteria: dict[str, Any] | None = None,
        campaign_id: str | None = None,
        workflow_id: str | None = None,
    ) -> "TrialIntent":
        return cls(
            trial_id=f"{profile}/{phase}/{trial}",
            attempt=attempt,
            profile=profile,
            phase=phase,
            trial=trial,
            research_intent=research_intent,
            planned_env=dict(env),
            success_criteria=success_criteria or {},
            expected_failure_reason=expected_failure_reason,
            campaign_id=campaign_id,
            workflow_id=workflow_id,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "trial_id": self.trial_id,
            "attempt": self.attempt,
            "profile": self.profile,
            "phase": self.phase,
            "trial": self.trial,
            "research_intent": self.research_intent,
            "planned_env": self.planned_env,
            "success_criteria": self.success_criteria,
            "expected_failure_reason": self.expected_failure_reason,
            "campaign_id": self.campaign_id,
            "workflow_id": self.workflow_id,
        }


@dataclass(frozen=True)
class ResolvedTrialConfig:
    trial_id: str
    attempt: int
    git_commit: str
    command: list[str]
    env: dict[str, str]
    model_path: str
    datasets: list[str]
    eval_datasets: list[str]
    output_dir: str
    run_dir: str
    hardware_profile: str
    distributed: dict[str, Any]
    artifact_root_ref: str
    artifact_refs: dict[str, str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "trial_id": self.trial_id,
            "attempt": self.attempt,
            "git_commit": self.git_commit,
            "command": self.command,
            "env": redact_mapping(self.env),
            "model_path": self.model_path,
            "datasets": self.datasets,
            "eval_datasets": self.eval_datasets,
            "output_dir": self.output_dir,
            "run_dir": self.run_dir,
            "hardware_profile": self.hardware_profile,
            "distributed": self.distributed,
            "artifact_root_ref": self.artifact_root_ref,
            "artifact_refs": self.artifact_refs,
        }


@dataclass(frozen=True)
class TrialAnalysis:
    trial_id: str
    attempt: int
    status: str
    failure_reason: FailureReason | str
    root_cause: str
    expected_failure: bool
    metrics: TrialMetrics
    symptoms: list[dict[str, Any]]
    evidence_refs: list[str]
    actions: list[dict[str, Any]]
    recommendations: list[str]
    artifact_root_ref: str
    artifact_refs: dict[str, str]
    artifact_dir: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "trial_id": self.trial_id,
            "attempt": self.attempt,
            "status": self.status,
            "failure_reason": _failure_reason_value(self.failure_reason),
            "root_cause": self.root_cause,
            "expected_failure": self.expected_failure,
            "metrics": metrics_to_payload(self.metrics),
            "symptoms": self.symptoms,
            "evidence_refs": self.evidence_refs,
            "actions": self.actions,
            "recommendations": self.recommendations,
            "artifact_root_ref": self.artifact_root_ref,
            "artifact_refs": self.artifact_refs,
            "artifact_dir": self.artifact_dir,
        }
