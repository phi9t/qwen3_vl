from __future__ import annotations

from enum import StrEnum

from temporalio import activity

from research.experiment_helpers import (
    append_result_row_from_analysis,
    git_short,
    plan_sweep_trials,
    repo_root,
    results_path,
)
from research.models import FailureReason, TrialMetrics, TrialSpec
from research.observability.summarizer import summarize_campaign
from research.observability.trial_runner import SupervisorTrialRunner
from research.profile_config import load_profile, plan_probe_trials
from research.selection import choose_capacity, write_selected_env


def _coerce_failure_reason(value: object) -> FailureReason:
    if isinstance(value, FailureReason):
        return value
    if isinstance(value, StrEnum):
        return FailureReason(str(value))
    if isinstance(value, str):
        return FailureReason(value)
    return FailureReason.UNKNOWN


def _metrics_from_payload(payload: dict) -> TrialMetrics:
    return TrialMetrics(
        status=payload["status"],
        val_loss=payload.get("val_loss"),
        peak_vram_mb=payload.get("peak_vram_mb"),
        throughput_steps_per_sec=payload.get("throughput_steps_per_sec"),
        failure_reason=_coerce_failure_reason(payload.get("failure_reason", FailureReason.NONE)),
    )


def _metrics_to_payload(metrics: TrialMetrics) -> dict:
    return {
        "status": metrics.status,
        "val_loss": metrics.val_loss,
        "peak_vram_mb": metrics.peak_vram_mb,
        "throughput_steps_per_sec": metrics.throughput_steps_per_sec,
        "failure_reason": metrics.failure_reason.value,
    }


@activity.defn
def load_profile_activity(profile: str) -> dict:
    root = repo_root()
    loaded = load_profile(root / "research" / "profiles" / f"{profile}.env")
    return {
        **loaded.__dict__,
        "batch_sizes": list(loaded.batch_sizes),
        "grad_accum_steps": list(loaded.grad_accum_steps),
        "gradient_checkpointing": list(loaded.gradient_checkpointing),
        "max_pixels": list(loaded.max_pixels),
        "model_max_lengths": list(loaded.model_max_lengths),
    }


@activity.defn
def plan_probe_trials_activity(profile: str) -> list[dict]:
    root = repo_root()
    loaded = load_profile(root / "research" / "profiles" / f"{profile}.env")
    return [spec.__dict__ for spec in plan_probe_trials(loaded)]


@activity.defn
def run_trial_activity(spec_payload: dict, dry_run: bool = False) -> dict:
    root = repo_root()
    spec = TrialSpec(**spec_payload)
    analysis = SupervisorTrialRunner(root).run_from_spec(spec, dry_run=dry_run)
    append_result_row_from_analysis(
        results_path(root),
        spec,
        analysis,
        git_commit=git_short(root),
    )
    return analysis.to_payload()


@activity.defn
def plan_sweep_trials_activity(profile: str, selected_payload: dict | None = None) -> list[dict]:
    root = repo_root()
    if selected_payload is None:
        selected_path = root / "research" / "selected" / f"{profile}.env"
        selected_payload = {}
        for raw in selected_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                selected_payload[key] = value
    return [spec.__dict__ for spec in plan_sweep_trials(profile, selected_payload)]


@activity.defn
def select_capacity_activity(profile: str, rows_payload: list[dict]) -> dict:
    root = repo_root()
    loaded = load_profile(root / "research" / "profiles" / f"{profile}.env")
    rows: list[tuple[TrialSpec, TrialMetrics]] = []
    for row in rows_payload:
        rows.append((TrialSpec(**row["spec"]), _metrics_from_payload(row["metrics"])))
    selected = choose_capacity(rows, vram_ceiling_mb=loaded.vram_ceiling_mb)
    path = write_selected_env(root / "research" / "selected", selected)
    return {"selected": selected.__dict__, "path": str(path)}


@activity.defn
def select_capacity_from_results_activity(profile: str) -> dict:
    root = repo_root()
    loaded = load_profile(root / "research" / "profiles" / f"{profile}.env")
    specs = {spec.trial: spec for spec in plan_probe_trials(loaded)}
    rows: list[tuple[TrialSpec, TrialMetrics]] = []
    result_file = results_path(root)
    for line in result_file.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 22 or parts[2] != profile or parts[3] != "probe":
            continue
        spec = specs.get(parts[4])
        if spec is None:
            continue
        rows.append(
            (
                spec,
                TrialMetrics(
                    status=parts[18],
                    val_loss=float(parts[15]) if parts[15] else None,
                    peak_vram_mb=float(parts[16]) if parts[16] else None,
                    throughput_steps_per_sec=float(parts[17]) if parts[17] else None,
                    failure_reason=_coerce_failure_reason(parts[19] or FailureReason.NONE),
                ),
            )
        )
    selected = choose_capacity(rows, vram_ceiling_mb=loaded.vram_ceiling_mb)
    path = write_selected_env(root / "research" / "selected", selected)
    return {"selected": selected.__dict__, "path": str(path)}


@activity.defn
def summarize_campaign_activity(profile: str) -> str:
    root = repo_root()
    summary = root / "experiments" / "summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(summarize_campaign(root, profile=profile), encoding="utf-8")
    return str(summary)
