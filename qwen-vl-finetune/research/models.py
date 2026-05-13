from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class FailureReason(StrEnum):
    NONE = "none"
    OOM = "oom"
    NCCL = "nccl"
    MISSING_FOOTER = "missing_footer"
    LAUNCHER_ERROR = "launcher_error"
    MODEL_LOAD_ERROR = "model_load_error"
    DATASET_ERROR = "dataset_error"
    HUB_OR_CACHE_ERROR = "hub_or_cache_error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    enabled: bool
    vram_ceiling_mb: float
    max_steps: int
    eval_steps: int
    save_steps: int
    datasets: str
    eval_datasets: str
    batch_sizes: tuple[int, ...]
    grad_accum_steps: tuple[int, ...]
    gradient_checkpointing: tuple[bool, ...]
    max_pixels: tuple[int, ...]
    model_max_lengths: tuple[int, ...]


@dataclass(frozen=True)
class TrialSpec:
    profile: str
    phase: str
    trial: str
    env: dict[str, str] = field(default_factory=dict)

    def run_dir(self, root_dir: Path) -> Path:
        return root_dir / "experiments" / "runs" / self.profile / self.phase / self.trial

    def log_path(self, root_dir: Path) -> Path:
        return self.run_dir(root_dir) / "run.log"

    def output_dir(self, root_dir: Path) -> Path:
        return self.run_dir(root_dir) / "output"


@dataclass(frozen=True)
class TrialMetrics:
    status: str
    val_loss: float | None = None
    peak_vram_mb: float | None = None
    throughput_steps_per_sec: float | None = None
    failure_reason: FailureReason = FailureReason.NONE

    @property
    def is_successful(self) -> bool:
        return (
            self.status == "ok"
            and self.val_loss is not None
            and self.peak_vram_mb is not None
            and self.failure_reason == FailureReason.NONE
        )
