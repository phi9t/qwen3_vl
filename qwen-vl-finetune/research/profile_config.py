from __future__ import annotations

from itertools import product
from pathlib import Path

from research.models import HardwareProfile, TrialSpec


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid profile line in {path}: {raw!r}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _get(values: dict[str, str], key: str) -> str:
    try:
        return values[key]
    except KeyError as exc:
        raise ValueError(f"Missing required profile key: {key}") from exc


def _bool(value: str) -> bool:
    if value.lower() in {"1", "true", "yes"}:
        return True
    if value.lower() in {"0", "false", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _bool_tuple(value: str) -> tuple[bool, ...]:
    return tuple(_bool(part.strip()) for part in value.split(",") if part.strip())


def load_profile(path: str | Path) -> HardwareProfile:
    profile_path = Path(path)
    values = _parse_env(profile_path)
    return HardwareProfile(
        name=_get(values, "PROFILE_NAME"),
        enabled=_bool(_get(values, "PROFILE_ENABLED")),
        vram_ceiling_mb=float(_get(values, "B200_VRAM_CEILING_MB")),
        max_steps=int(_get(values, "MAX_STEPS")),
        eval_steps=int(_get(values, "EVAL_STEPS")),
        save_steps=int(_get(values, "SAVE_STEPS")),
        datasets=_get(values, "DATASETS"),
        eval_datasets=_get(values, "EVAL_DATASETS"),
        batch_sizes=_int_tuple(_get(values, "PROBE_BATCH_SIZES")),
        grad_accum_steps=_int_tuple(_get(values, "PROBE_GRAD_ACCUM_STEPS")),
        gradient_checkpointing=_bool_tuple(_get(values, "PROBE_GRADIENT_CHECKPOINTING")),
        max_pixels=_int_tuple(_get(values, "PROBE_MAX_PIXELS")),
        model_max_lengths=_int_tuple(_get(values, "PROBE_MODEL_MAX_LENGTHS")),
    )


def plan_probe_trials(profile: HardwareProfile) -> list[TrialSpec]:
    if not profile.enabled:
        raise ValueError(f"Profile {profile.name!r} is disabled")

    trials: list[TrialSpec] = []
    for batch, accum, checkpointing, max_pixels, model_max_length in product(
        profile.batch_sizes,
        profile.grad_accum_steps,
        profile.gradient_checkpointing,
        profile.max_pixels,
        profile.model_max_lengths,
    ):
        trial = (
            f"probe_bs{batch}_ga{accum}_gc{checkpointing}"
            f"_px{max_pixels}_ctx{model_max_length}"
        )
        env = {
            "MODEL_NAME_OR_PATH": "Qwen/Qwen3-VL-8B-Instruct",
            "RUN_NAME": trial,
            "DATASETS": profile.datasets,
            "EVAL_DATASETS": profile.eval_datasets,
            "MAX_STEPS": str(profile.max_steps),
            "EVAL_STEPS": str(profile.eval_steps),
            "SAVE_STEPS": str(profile.save_steps),
            "BATCH_SIZE": str(batch),
            "GRAD_ACCUM_STEPS": str(accum),
            "GRADIENT_CHECKPOINTING": str(checkpointing),
            "MAX_PIXELS": str(max_pixels),
            "MODEL_MAX_LENGTH": str(model_max_length),
        }
        trials.append(TrialSpec(profile=profile.name, phase="probe", trial=trial, env=env))
    return trials
