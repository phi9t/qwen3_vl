from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from research.models import TrialMetrics, TrialSpec
from research.observability.schema import TrialAnalysis


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return repo_root().parent


def results_header() -> str:
    return "\t".join(
        [
            "ts",
            "git",
            "profile",
            "phase",
            "trial",
            "regime",
            "learning_rate",
            "batch_size",
            "grad_accum",
            "gradient_checkpointing",
            "max_pixels",
            "model_max_length",
            "warmup_ratio",
            "weight_decay",
            "selected_capacity",
            "val_loss",
            "peak_vram_mb",
            "throughput_steps_per_sec",
            "status",
            "failure_reason",
            "output_dir",
            "log",
        ]
    )


def results_path(root: Path) -> Path:
    path = root / "experiments" / "results.tsv"
    if not path.exists():
        return path
    first = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    if first and first[0] == results_header():
        return path
    return root / "experiments" / "results.profiled.tsv"


def git_short(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def append_result_row(
    path: Path,
    spec: TrialSpec,
    metrics: TrialMetrics,
    *,
    git_commit: str,
    log_path: Path,
    output_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(results_header() + "\n", encoding="utf-8")
    else:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines and lines[0] != results_header():
            raise ValueError(f"Results schema mismatch in {path}")
        key = (spec.profile, spec.phase, spec.trial, str(output_dir))
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 22 and (parts[2], parts[3], parts[4], parts[20]) == key:
                return

    row = [
        datetime.now(timezone.utc).isoformat(),
        git_commit,
        spec.profile,
        spec.phase,
        spec.trial,
        spec.env.get("REGIME", ""),
        spec.env.get("LR", ""),
        spec.env.get("BATCH_SIZE", ""),
        spec.env.get("GRAD_ACCUM_STEPS", ""),
        spec.env.get("GRADIENT_CHECKPOINTING", ""),
        spec.env.get("MAX_PIXELS", ""),
        spec.env.get("MODEL_MAX_LENGTH", ""),
        spec.env.get("WARMUP_RATIO", ""),
        spec.env.get("WEIGHT_DECAY", ""),
        spec.env.get("SELECTED_CAPACITY", ""),
        "" if metrics.val_loss is None else str(metrics.val_loss),
        "" if metrics.peak_vram_mb is None else str(metrics.peak_vram_mb),
        ""
        if metrics.throughput_steps_per_sec is None
        else str(metrics.throughput_steps_per_sec),
        metrics.status,
        metrics.failure_reason.value,
        str(output_dir),
        str(log_path),
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def append_result_row_from_analysis(
    path: Path,
    spec: TrialSpec,
    analysis: TrialAnalysis,
    *,
    git_commit: str,
) -> None:
    append_result_row(
        path,
        spec,
        analysis.metrics,
        git_commit=git_commit,
        log_path=Path(analysis.artifact_refs["full_log"]),
        output_dir=Path(analysis.artifact_refs["output_dir"]),
    )


def plan_sweep_trials(profile: str, selected_env: dict[str, str]) -> list[TrialSpec]:
    base = {
        **selected_env,
        "DATASETS": "visualwebinstruct_train",
        "EVAL_DATASETS": "visualwebinstruct_val",
        "MODEL_NAME_OR_PATH": "Qwen/Qwen3-VL-8B-Instruct",
        "MAX_STEPS": "5000",
        "EVAL_STEPS": "500",
        "SAVE_STEPS": "1000",
        "SELECTED_CAPACITY": "true",
    }
    specs: list[TrialSpec] = []
    phase_a = [
        (
            "A1_proj_only",
            "proj_only",
            {
                "TUNE_MM_MLP": "True",
                "TUNE_MM_LLM": "False",
                "TUNE_MM_VISION": "False",
                "LORA_ENABLE": "False",
                "LR": "1e-5",
            },
        ),
        (
            "A2_proj_llm",
            "proj_llm",
            {
                "TUNE_MM_MLP": "True",
                "TUNE_MM_LLM": "True",
                "TUNE_MM_VISION": "False",
                "LORA_ENABLE": "False",
                "LR": "1e-5",
            },
        ),
        (
            "A3_lora",
            "lora",
            {
                "LORA_ENABLE": "True",
                "LORA_R": "16",
                "LORA_ALPHA": "32",
                "LORA_DROPOUT": "0.0",
                "LR": "2e-4",
            },
        ),
    ]
    for trial, regime, env in phase_a:
        specs.append(TrialSpec(profile, "A", trial, {**base, **env, "REGIME": regime}))
    for lr in ("5e-6", "1e-5", "2e-5"):
        specs.append(
            TrialSpec(
                profile, "B", f"B_lr{lr}", {**base, "LR": lr, "REGIME": "selected"}
            )
        )
    for name, pixels, length in (
        ("C1_224_8192", "50176", "8192"),
        ("C2_336_8192", "112896", "8192"),
        ("C3_224_4096", "50176", "4096"),
        ("C4_336_4096", "112896", "4096"),
    ):
        specs.append(
            TrialSpec(
                profile,
                "C",
                name,
                {
                    **base,
                    "MAX_PIXELS": pixels,
                    "MODEL_MAX_LENGTH": length,
                    "REGIME": "selected",
                },
            )
        )
    for warmup in ("0.0", "0.03", "0.05"):
        for wd in ("0.0", "0.01"):
            specs.append(
                TrialSpec(
                    profile,
                    "D",
                    f"D_warm{warmup}_wd{wd}",
                    {
                        **base,
                        "WARMUP_RATIO": warmup,
                        "WEIGHT_DECAY": wd,
                        "REGIME": "selected",
                    },
                )
            )
    return specs
