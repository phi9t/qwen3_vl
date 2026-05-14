"""Qwen-VL adapter for the generic research methodology."""

from __future__ import annotations

import itertools
import pathlib

import research.models


_DEFAULT_PROFILE_VALUES = {
    "DATASETS": "visualwebinstruct_train",
    "EVAL_DATASETS": "visualwebinstruct_val",
    "MAX_STEPS": "50",
    "EVAL_STEPS": "25",
    "SAVE_STEPS": "50",
    "BATCH_SIZES": "2",
    "GRAD_ACCUM_STEPS": "4",
    "GRADIENT_CHECKPOINTING": "true",
    "MAX_PIXELS": "50176",
    "MODEL_MAX_LENGTHS": "8192",
}


class QwenVlAdapter:
    """Adapter that turns Qwen-VL profile files into trial intents."""

    name = "qwen_vl"

    def generate_probe_intents(
        self, request: research.models.ProbeRequest
    ) -> list[research.models.Intent]:
        """Generate probe intents from the requested hardware profile."""
        profile = _load_profile(_profile_path(request.profile))
        if not _as_bool(profile.get("ENABLED", "false")):
            return []

        values = {**_DEFAULT_PROFILE_VALUES, **profile}
        intents = []
        for (
            batch_size,
            grad_accum,
            checkpointing,
            max_pixels,
            max_length,
        ) in itertools.product(
            _csv(values["BATCH_SIZES"]),
            _csv(values["GRAD_ACCUM_STEPS"]),
            _csv(values["GRADIENT_CHECKPOINTING"]),
            _csv(values["MAX_PIXELS"]),
            _csv(values["MODEL_MAX_LENGTHS"]),
        ):
            name = (
                f"probe_bs{batch_size}_ga{grad_accum}_gc{checkpointing}"
                f"_px{max_pixels}_ctx{max_length}"
            )
            config = {
                "MODEL_NAME_OR_PATH": request.model,
                "DATASETS": values["DATASETS"],
                "EVAL_DATASETS": values["EVAL_DATASETS"],
                "MAX_STEPS": values["MAX_STEPS"],
                "EVAL_STEPS": values["EVAL_STEPS"],
                "SAVE_STEPS": values["SAVE_STEPS"],
                "BATCH_SIZE": batch_size,
                "GRAD_ACCUM_STEPS": grad_accum,
                "GRADIENT_CHECKPOINTING": checkpointing,
                "MAX_PIXELS": max_pixels,
                "MODEL_MAX_LENGTH": max_length,
            }
            intents.append(
                research.models.Intent(
                    adapter=self.name,
                    model=request.model,
                    profile=request.profile,
                    phase="probe",
                    name=name,
                    config=config,
                    objective=request.objective,
                    source="probe",
                )
            )
        return intents

    def preflight(
        self, intent: research.models.Intent, context: research.models.TrialContext
    ) -> research.models.PreflightResult:
        """Check that the Qwen-VL launcher needed for the trial exists."""
        launcher = _launcher_path(context)
        ok = launcher.exists()
        checks = {
            "launcher": "ok" if ok else "missing",
            "launcher_path": str(launcher),
        }
        return research.models.PreflightResult(
            ok=ok,
            checks=checks,
            message="ok" if ok else f"missing launcher: {launcher}",
        )

    def build_trial(
        self, intent: research.models.Intent, context: research.models.TrialContext
    ) -> research.models.TrialCommand:
        """Build the shell command and environment for one Qwen-VL trial."""
        output_dir = (
            context.artifact_dir
            / "outputs"
            / str(context.experiment_id)
            / f"attempt-{context.attempt}"
        )
        env = {key: str(value) for key, value in intent.config.items()}
        env["MODEL_NAME_OR_PATH"] = intent.model
        env["OUTPUT_DIR"] = str(output_dir)
        env["RUN_NAME"] = intent.name
        return research.models.TrialCommand(
            argv=["bash", str(_launcher_path(context))],
            env=env,
            cwd=context.worktree,
        )

    def parse_progress(
        self, event_or_log_line: str
    ) -> research.models.ProgressUpdate | None:
        """Parse simple metric lines from Qwen-VL trial logs."""
        line = event_or_log_line.strip()
        if ":" not in line:
            return None
        key, raw_value = line.split(":", 1)
        if key not in {"val_loss", "peak_vram_mb"}:
            return None
        try:
            value = float(raw_value.strip())
        except ValueError:
            return None
        return research.models.ProgressUpdate(metrics={key: value}, message=line)

    def analyze_result(
        self, context: research.models.TrialContext
    ) -> research.models.TrialReport:
        """Summarize a Qwen-VL trial from its captured run log."""
        log_path = context.artifact_dir / "run.log"
        if not log_path.exists():
            return research.models.TrialReport(
                status="failed",
                failure={"reason": "missing_run_log"},
                summary="Qwen-VL trial failed: missing run.log",
            )

        metrics = {}
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            update = self.parse_progress(line)
            if update is not None:
                metrics.update(update.metrics)

        status = "succeeded" if "val_loss" in metrics or "DRY_RUN" in text else "failed"
        failure = {} if status == "succeeded" else {"reason": "missing_metrics"}
        return research.models.TrialReport(
            status=status,
            metrics=metrics,
            failure=failure,
            summary=f"Qwen-VL trial {status}",
        )


def _profile_path(profile: str) -> pathlib.Path:
    if pathlib.PurePath(profile).name != profile:
        raise ValueError(f"Invalid profile name: {profile!r}.")
    return pathlib.Path(__file__).with_name("profiles") / f"{profile}.env"


def _launcher_path(context: research.models.TrialContext) -> pathlib.Path:
    return context.worktree / "scripts" / "sft_qwen3_8b.sh"


def _load_profile(path: pathlib.Path) -> dict[str, str]:
    values = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid profile line {line_number} in {path}.")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _as_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}
