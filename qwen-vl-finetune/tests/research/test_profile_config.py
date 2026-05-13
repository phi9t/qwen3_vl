from pathlib import Path

from research.models import FailureReason, TrialMetrics, TrialSpec
from research.profile_config import load_profile, plan_probe_trials


def test_trial_spec_derives_run_paths() -> None:
    spec = TrialSpec(
        profile="b200",
        phase="probe",
        trial="probe_bs2_ga4_gcTrue_px50176",
        env={
            "BATCH_SIZE": "2",
            "GRAD_ACCUM_STEPS": "4",
            "GRADIENT_CHECKPOINTING": "True",
            "MAX_PIXELS": "50176",
            "MODEL_MAX_LENGTH": "8192",
        },
    )

    run_dir = spec.run_dir(Path("/repo/qwen-vl-finetune"))

    assert run_dir == Path(
        "/repo/qwen-vl-finetune/experiments/runs/b200/probe/probe_bs2_ga4_gcTrue_px50176"
    )
    assert spec.log_path(Path("/repo/qwen-vl-finetune")).name == "run.log"


def test_trial_metrics_success_requires_footer_values() -> None:
    metrics = TrialMetrics(
        status="ok",
        val_loss=0.56,
        peak_vram_mb=95280.6,
        throughput_steps_per_sec=0.31,
        failure_reason=FailureReason.NONE,
    )

    assert metrics.is_successful


def test_load_profile_parses_b200_env(tmp_path: Path) -> None:
    profile_path = tmp_path / "b200.env"
    profile_path.write_text(
        "\n".join(
            [
                "PROFILE_NAME=b200",
                "PROFILE_ENABLED=true",
                "B200_VRAM_CEILING_MB=160000",
                "MAX_STEPS=50",
                "EVAL_STEPS=25",
                "SAVE_STEPS=50",
                "DATASETS=visualwebinstruct_train",
                "EVAL_DATASETS=visualwebinstruct_val",
                "PROBE_BATCH_SIZES=2,4",
                "PROBE_GRAD_ACCUM_STEPS=2,4",
                "PROBE_GRADIENT_CHECKPOINTING=True,False",
                "PROBE_MAX_PIXELS=50176",
                "PROBE_MODEL_MAX_LENGTHS=8192",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    profile = load_profile(profile_path)

    assert profile.name == "b200"
    assert profile.enabled is True
    assert profile.vram_ceiling_mb == 160000
    assert profile.batch_sizes == (2, 4)
    assert profile.gradient_checkpointing == (True, False)


def test_plan_probe_trials_expands_grid(tmp_path: Path) -> None:
    profile_path = tmp_path / "b200.env"
    profile_path.write_text(
        "\n".join(
            [
                "PROFILE_NAME=b200",
                "PROFILE_ENABLED=true",
                "B200_VRAM_CEILING_MB=160000",
                "MAX_STEPS=50",
                "EVAL_STEPS=25",
                "SAVE_STEPS=50",
                "DATASETS=visualwebinstruct_train",
                "EVAL_DATASETS=visualwebinstruct_val",
                "PROBE_BATCH_SIZES=2,4",
                "PROBE_GRAD_ACCUM_STEPS=2",
                "PROBE_GRADIENT_CHECKPOINTING=True,False",
                "PROBE_MAX_PIXELS=50176",
                "PROBE_MODEL_MAX_LENGTHS=8192",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    profile = load_profile(profile_path)

    trials = plan_probe_trials(profile)

    assert len(trials) == 4
    assert trials[0].env["MODEL_NAME_OR_PATH"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert trials[0].env["DATASETS"] == "visualwebinstruct_train"
    assert trials[0].phase == "probe"
