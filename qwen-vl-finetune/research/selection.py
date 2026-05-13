from __future__ import annotations

from pathlib import Path

from research.models import TrialMetrics, TrialSpec


REQUIRED_SELECTED_KEYS = (
    "BATCH_SIZE",
    "GRAD_ACCUM_STEPS",
    "GRADIENT_CHECKPOINTING",
    "MAX_PIXELS",
    "MODEL_MAX_LENGTH",
)


def choose_capacity(
    rows: list[tuple[TrialSpec, TrialMetrics]],
    *,
    vram_ceiling_mb: float,
) -> TrialSpec:
    candidates: list[tuple[TrialSpec, TrialMetrics]] = []
    for spec, metrics in rows:
        if not metrics.is_successful:
            continue
        if metrics.peak_vram_mb is None or metrics.peak_vram_mb > vram_ceiling_mb:
            continue
        candidates.append((spec, metrics))

    if not candidates:
        raise ValueError("No successful probe rows under the VRAM ceiling")

    def sort_key(row: tuple[TrialSpec, TrialMetrics]) -> tuple[float, float, float]:
        _, metrics = row
        throughput = metrics.throughput_steps_per_sec or 0.0
        val_loss = metrics.val_loss if metrics.val_loss is not None else float("inf")
        vram = metrics.peak_vram_mb if metrics.peak_vram_mb is not None else float("inf")
        return (-throughput, val_loss, vram)

    return sorted(candidates, key=sort_key)[0][0]


def write_selected_env(selected_dir: str | Path, spec: TrialSpec) -> Path:
    out_dir = Path(selected_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.profile}.env"

    lines = [f"# Selected from {spec.trial}"]
    for key in REQUIRED_SELECTED_KEYS:
        if key not in spec.env:
            raise ValueError(f"Selected trial missing env key: {key}")
        lines.append(f"{key}={spec.env[key]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
