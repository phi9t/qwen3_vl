from pathlib import Path

from research.models import TrialMetrics, TrialSpec
from research.selection import choose_capacity, write_selected_env


def test_choose_capacity_prefers_throughput_under_vram_ceiling() -> None:
    slow = TrialSpec("b200", "probe", "slow", {"BATCH_SIZE": "2"})
    fast = TrialSpec("b200", "probe", "fast", {"BATCH_SIZE": "4"})
    too_big = TrialSpec("b200", "probe", "too_big", {"BATCH_SIZE": "8"})
    rows = [
        (slow, TrialMetrics("ok", val_loss=0.50, peak_vram_mb=90000, throughput_steps_per_sec=0.2)),
        (fast, TrialMetrics("ok", val_loss=0.60, peak_vram_mb=120000, throughput_steps_per_sec=0.4)),
        (too_big, TrialMetrics("ok", val_loss=0.40, peak_vram_mb=170000, throughput_steps_per_sec=0.5)),
    ]

    selected = choose_capacity(rows, vram_ceiling_mb=160000)

    assert selected.trial == "fast"


def test_write_selected_env_writes_shell_assignments(tmp_path: Path) -> None:
    spec = TrialSpec(
        "b200",
        "probe",
        "chosen",
        {
            "BATCH_SIZE": "4",
            "GRAD_ACCUM_STEPS": "2",
            "GRADIENT_CHECKPOINTING": "False",
            "MAX_PIXELS": "50176",
            "MODEL_MAX_LENGTH": "8192",
        },
    )

    path = write_selected_env(tmp_path, spec)

    assert path == tmp_path / "b200.env"
    assert "BATCH_SIZE=4" in path.read_text(encoding="utf-8")
